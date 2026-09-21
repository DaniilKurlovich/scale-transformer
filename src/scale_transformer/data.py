"""SFT dataset construction: chat-template rendering, completion-only masking, collation."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


# --------------------------------------------------------------------------- #
# Normalisation: every supported source shape becomes a list of chat messages. #
# --------------------------------------------------------------------------- #

def _to_messages(example: dict[str, Any], cfg: DataConfig) -> list[dict[str, str]]:
    if cfg.format == "messages":
        messages = list(example[cfg.messages_column])
    elif cfg.format == "alpaca":
        instruction = example["instruction"]
        context = (example.get("input") or "").strip()
        user = f"{instruction}\n\n{context}" if context else instruction
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["output"]},
        ]
    elif cfg.format == "prompt_completion":
        messages = [
            {"role": "user", "content": example[cfg.prompt_column]},
            {"role": "assistant", "content": example[cfg.completion_column]},
        ]
    else:
        raise ValueError(f"Unknown data.format: {cfg.format!r}")

    if cfg.system_prompt and (not messages or messages[0]["role"] != "system"):
        messages = [{"role": "system", "content": cfg.system_prompt}, *messages]
    return messages


# --------------------------------------------------------------------------- #
# Tokenisation + label masking                                                 #
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _render(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None,
) -> str:
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        # Qwen3's hybrid-thinking template reads this; templates that don't know it ignore it.
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **kwargs,
    )


def _assistant_char_spans(
    text: str, messages: list[dict[str, str]]
) -> list[tuple[int, int]] | None:
    """Locate each assistant message's content inside the rendered conversation.

    Spans are found by searching the *final* render, so it does not matter that chat
    templates rewrite history as the conversation grows -- Qwen3, for instance, emits an
    empty ``<think>`` block only for the last assistant turn. Walking every message in
    order keeps the cursor monotonic, so duplicated content (two "Yes." replies) still
    maps to the right occurrence. Returns None if an assistant turn cannot be located.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for message in messages:
        content = message.get("content") or ""
        if not content:
            continue
        index = text.find(content, cursor)
        if index < 0 and message["role"] == "assistant":
            # Templates strip <think> blocks out of historical assistant turns.
            content = _THINK_RE.sub("", content)
            index = text.find(content, cursor) if content else -1
        if index < 0:
            if message["role"] == "assistant":
                return None
            continue
        cursor = index + len(content)
        if message["role"] == "assistant":
            spans.append((index, cursor))
    return spans or None


def _encode(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
) -> dict[str, Any]:
    messages = _to_messages(example, cfg)
    text = _render(tokenizer, messages, enable_thinking=cfg.enable_thinking)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=cfg.train_on_completions_only,
    )
    input_ids = encoded["input_ids"]

    if not cfg.train_on_completions_only:
        labels = list(input_ids)
        located = True
    else:
        spans = _assistant_char_spans(text, messages)
        located = spans is not None
        labels = [IGNORE_INDEX] * len(input_ids)
        special_ids = set(tokenizer.all_special_ids)
        offsets = encoded["offset_mapping"]
        for start, end in spans or []:
            last = -1
            for i, (char_start, char_end) in enumerate(offsets):
                # Any token overlapping the span is supervised, including one that
                # straddles a boundary -- dropping it would break the text mid-token.
                if char_end > start and char_start < end:
                    labels[i] = input_ids[i]
                    last = i
            # Supervise the end-of-turn token as well, so the model learns to stop.
            if 0 <= last < len(input_ids) - 1 and input_ids[last + 1] in special_ids:
                labels[last + 1] = input_ids[last + 1]
        # When spans could not be located, every label stays masked and `_prepare`
        # drops the example -- far better than training on the user's prompt.

    input_ids = input_ids[: cfg.max_seq_len]
    labels = labels[: cfg.max_seq_len]
    n_supervised = sum(1 for label in labels if label != IGNORE_INDEX)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "n_supervised": n_supervised,
        "located": located,
    }


# --------------------------------------------------------------------------- #
# Public entry points                                                          #
# --------------------------------------------------------------------------- #

def build_datasets(
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
) -> tuple[Dataset, Dataset | None]:
    """Load, render and tokenise the train (and eval) splits."""
    if cfg.train_on_completions_only and not tokenizer.is_fast:
        raise ValueError(
            "data.train_on_completions_only needs a fast tokenizer for offset mapping; "
            "set it to false or use a checkpoint that ships tokenizer.json."
        )
    load_kwargs: dict[str, Any] = {}
    if cfg.data_files:
        load_kwargs["data_files"] = dict(cfg.data_files)
    raw = load_dataset(cfg.dataset_name, cfg.dataset_config, **load_kwargs)
    if isinstance(raw, Dataset):  # single-split loaders return a bare Dataset
        raw = DatasetDict({cfg.train_split: raw})

    train_raw = raw[cfg.train_split]
    eval_raw = raw[cfg.eval_split] if cfg.eval_split else None

    if eval_raw is None and cfg.eval_fraction > 0:
        split = train_raw.train_test_split(test_size=cfg.eval_fraction, seed=cfg.seed)
        train_raw, eval_raw = split["train"], split["test"]

    if cfg.max_train_samples:
        train_raw = train_raw.select(range(min(cfg.max_train_samples, len(train_raw))))
    if eval_raw is not None and cfg.max_eval_samples:
        eval_raw = eval_raw.select(range(min(cfg.max_eval_samples, len(eval_raw))))

    train = _prepare(train_raw, tokenizer, cfg, "train")
    evaluation = _prepare(eval_raw, tokenizer, cfg, "eval") if eval_raw is not None else None
    return train, evaluation


def _preprocessing_fingerprint(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
    name: str,
) -> str:
    """A cache key that actually covers how examples get encoded.

    `datasets` caches `map` output under a fingerprint that does not reliably change
    when the mapped function's source does, so editing the masking logic can leave a
    run silently training on the previous tokenization. Hashing the source of the
    encoding helpers -- along with the config, the chat template and the input
    fingerprint -- makes the cache correct instead of merely fast.
    """
    try:
        chat_template = tokenizer.get_chat_template()
    except Exception:
        chat_template = ""
    sources = "".join(
        inspect.getsource(fn)
        for fn in (_to_messages, _render, _assistant_char_spans, _encode)
    )
    payload = json.dumps(
        {
            "config": dataclasses.asdict(cfg),
            "tokenizer": str(tokenizer.name_or_path),
            "chat_template": chat_template or "",
            "code": sources,
            "dataset": dataset._fingerprint,
            "split": name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _prepare(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    cfg: DataConfig,
    name: str,
) -> Dataset:
    encoded = dataset.map(
        _encode,
        fn_kwargs={"tokenizer": tokenizer, "cfg": cfg},
        remove_columns=dataset.column_names,
        num_proc=cfg.num_proc if len(dataset) > 1000 else None,
        desc=f"tokenizing {name}",
        new_fingerprint=_preprocessing_fingerprint(dataset, tokenizer, cfg, name),
    )

    n_unlocated = sum(1 for ok in encoded["located"] if not ok)
    if n_unlocated:
        logger.warning(
            "%s: could not locate assistant turns in %d/%d rendered examples; they are "
            "dropped rather than trained with an unreliable loss mask.",
            name, n_unlocated, len(encoded),
        )

    before = len(encoded)
    encoded = encoded.filter(lambda ex: ex["n_supervised"] > 0)
    if len(encoded) < before:
        logger.warning(
            "%s: dropped %d/%d examples with no supervised tokens after truncation to %d.",
            name, before - len(encoded), before, cfg.max_seq_len,
        )

    return encoded.remove_columns(["n_supervised", "located"])


@dataclass
class CausalCollator:
    """Pads a batch to its longest sequence; label padding is masked out."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        longest = max(len(f["input_ids"]) for f in features)
        multiple = self.pad_to_multiple_of
        width = ((longest + multiple - 1) // multiple) * multiple
        pad_id = self.tokenizer.pad_token_id

        input_ids, labels, attention_mask = [], [], []
        for feature in features:
            ids = list(feature["input_ids"])
            lbl = list(feature["labels"])
            padding = width - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            labels.append(lbl + [IGNORE_INDEX] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
