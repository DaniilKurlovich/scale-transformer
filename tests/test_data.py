from scale_transformer.config import DataConfig
from scale_transformer.data import IGNORE_INDEX, CausalCollator, _encode

MULTI_TURN = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "It is 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "It is 6."},
    ]
}


def supervised_text(tokenizer, encoded):
    pairs = zip(encoded["input_ids"], encoded["labels"], strict=True)
    return tokenizer.decode([t for t, label in pairs if label != IGNORE_INDEX])


def test_multi_turn_supervises_assistant_turns_only(tokenizer):
    """Qwen3 rewrites history between renders; masking must survive that."""
    cfg = DataConfig(format="messages", enable_thinking=False, max_seq_len=512)
    encoded = _encode(MULTI_TURN, tokenizer, cfg)
    text = supervised_text(tokenizer, encoded)

    assert encoded["located"]
    assert "It is 4." in text and "It is 6." in text
    assert "2+2" not in text and "3+3" not in text
    # The end-of-turn token is supervised so the model learns to stop.
    assert tokenizer.eos_token in text or "<|im_end|>" in text


def test_every_turn_is_supervised_when_masking_disabled(tokenizer):
    cfg = DataConfig(format="messages", train_on_completions_only=False, max_seq_len=512)
    encoded = _encode(MULTI_TURN, tokenizer, cfg)
    assert all(label != IGNORE_INDEX for label in encoded["labels"])


def test_repeated_assistant_content_maps_to_distinct_spans(tokenizer):
    """Two identical replies must not both resolve to the first occurrence."""
    example = {
        "messages": [
            {"role": "user", "content": "First?"},
            {"role": "assistant", "content": "Yes."},
            {"role": "user", "content": "Second?"},
            {"role": "assistant", "content": "Yes."},
        ]
    }
    encoded = _encode(example, tokenizer, DataConfig(format="messages", max_seq_len=512))
    assert supervised_text(tokenizer, encoded).count("Yes.") == 2


def test_alpaca_format_masks_the_instruction(tokenizer):
    cfg = DataConfig(format="alpaca", system_prompt="You are terse.", max_seq_len=512)
    encoded = _encode(
        {"instruction": "Sort these.", "input": "3 1 2", "output": "1 2 3"}, tokenizer, cfg
    )
    text = supervised_text(tokenizer, encoded)
    assert "1 2 3" in text
    assert "Sort these" not in text and "You are terse" not in text


def test_prompt_completion_format(tokenizer):
    cfg = DataConfig(format="prompt_completion", max_seq_len=512)
    encoded = _encode({"prompt": "Capital of France?", "completion": "Paris."}, tokenizer, cfg)
    text = supervised_text(tokenizer, encoded)
    assert "Paris." in text and "France" not in text


def test_truncation_respects_max_seq_len(tokenizer):
    example = {
        "messages": [
            {"role": "user", "content": "word " * 500},
            {"role": "assistant", "content": "ok"},
        ]
    }
    encoded = _encode(example, tokenizer, DataConfig(format="messages", max_seq_len=32))
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == 32
    # The answer falls outside the window, so the example carries no signal and is dropped.
    assert encoded["n_supervised"] == 0


def test_collator_pads_without_supervising_padding(tokenizer):
    cfg = DataConfig(format="messages", max_seq_len=512)
    short = _encode(
        {"messages": [{"role": "user", "content": "Hi"},
                      {"role": "assistant", "content": "Hello."}]}, tokenizer, cfg
    )
    batch = CausalCollator(tokenizer)([short, _encode(MULTI_TURN, tokenizer, cfg)])

    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[1] % 8 == 0, "padded to a multiple of 8"
    padding = batch["attention_mask"] == 0
    assert (batch["labels"][padding] == IGNORE_INDEX).all()


def test_fingerprint_tracks_preprocessing_config(tokenizer):
    """A changed encoding config must not reuse a cached tokenization."""
    from datasets import Dataset

    from scale_transformer.data import _preprocessing_fingerprint

    dataset = Dataset.from_dict({"instruction": ["a"], "input": [""], "output": ["b"]})
    base = DataConfig(format="alpaca", max_seq_len=512)

    fingerprint = _preprocessing_fingerprint(dataset, tokenizer, base, "train")
    assert fingerprint == _preprocessing_fingerprint(dataset, tokenizer, base, "train")
    assert fingerprint != _preprocessing_fingerprint(dataset, tokenizer, base, "eval")

    for changed in (
        DataConfig(format="alpaca", max_seq_len=256),
        DataConfig(format="alpaca", max_seq_len=512, train_on_completions_only=False),
        DataConfig(format="alpaca", max_seq_len=512, enable_thinking=True),
    ):
        assert _preprocessing_fingerprint(dataset, tokenizer, changed, "train") != fingerprint
