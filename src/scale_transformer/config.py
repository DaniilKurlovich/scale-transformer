"""Experiment configuration: dataclasses, YAML loading, dotted-key overrides."""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-30B-A3B"
    # "bfloat16" everywhere except CPU-only debugging, where "float32" is the sane choice.
    dtype: str = "bfloat16"
    # "sdpa" works everywhere; "flash_attention_2" needs the flash-attn wheel; "eager" for debug.
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    # QLoRA. Requires bitsandbytes and a CUDA device.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    # MoE only: add the router load-balancing loss so experts don't collapse during SFT.
    # Ignored (with a warning) on dense models.
    output_router_logits: bool = True
    router_aux_loss_coef: float | None = 0.001


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    # Attention projections only, by default. See `target_expert_mlp` before widening this.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # Qwen3-30B-A3B has 128 experts x 48 layers. Adding the expert MLPs to the LoRA
    # targets creates ~18k extra adapter pairs, which is slow and memory-hungry for
    # little gain on most SFT runs. Off by default; flip it only if you know you want it.
    target_expert_mlp: bool = False
    # Never adapt the router (`mlp.gate`) -- perturbing routing early in training
    # destabilises the expert balance.
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    # A hub dataset id, or "json" / "csv" combined with `data_files` for local data.
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_config: str | None = None
    data_files: dict[str, str] | None = None
    train_split: str = "train"
    eval_split: str | None = None
    # Carve an eval set out of the train split when the dataset has no eval split.
    eval_fraction: float = 0.01
    # "messages" | "alpaca" | "prompt_completion"
    format: str = "alpaca"
    messages_column: str = "messages"
    prompt_column: str = "prompt"
    completion_column: str = "completion"
    system_prompt: str | None = None
    max_seq_len: int = 2048
    # Mask the prompt so loss is only taken on assistant turns.
    train_on_completions_only: bool = True
    # Qwen3 hybrid-thinking template switch. None = leave the template's own default alone.
    enable_thinking: bool | None = False
    num_proc: int = 4
    max_train_samples: int | None = None
    max_eval_samples: int | None = 256
    seed: int = 42


@dataclass
class TrainConfig:
    output_dir: str = "outputs/qwen3-30b-a3b-lora"
    seed: int = 42
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    # int = exact steps, float in [0, 1) = fraction of total steps.
    warmup_steps: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    # "adamw_torch" for LoRA; "paged_adamw_8bit" pairs with load_in_4bit on CUDA.
    optim: str = "adamw_torch"
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    eval_strategy: str = "steps"
    eval_steps: int = 200
    report_to: list[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str | None = None
    resume_from_checkpoint: str | None = None


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        unknown_sections = set(raw) - {f.name for f in dataclasses.fields(cls)}
        if unknown_sections:
            raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")
        return cls(
            model=_build(ModelConfig, raw.get("model", {})),
            lora=_build(LoraConfig, raw.get("lora", {})),
            data=_build(DataConfig, raw.get("data", {})),
            train=_build(TrainConfig, raw.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for override in overrides or []:
            _apply_override(raw, override)
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, values: dict[str, Any]):
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    coerced = {k: _coerce(v, hints[k], f"{cls.__name__}.{k}") for k, v in values.items()}
    return cls(**coerced)


def _coerce(value: Any, annotation: Any, where: str) -> Any:
    """Coerce a YAML/CLI value to the field's declared type.

    YAML 1.1 does not recognise `5e-5` as a float -- it comes back as the string
    "5e-5" -- so without this a plausible-looking `--set train.learning_rate=5e-5`
    would reach the optimizer as a string. Coercing against the dataclass types
    fixes that for config files and overrides alike, and turns a typo into an error
    at startup instead of a crash minutes into a run.
    """
    origin = typing.get_origin(annotation)

    if origin in (Union, types.UnionType):
        options = [a for a in typing.get_args(annotation) if a is not type(None)]
        if value is None:
            return None
        for option in options:
            try:
                return _coerce(value, option, where)
            except (TypeError, ValueError):
                continue
        raise ValueError(f"{where}: cannot interpret {value!r} as {annotation}")

    if value is None:
        raise ValueError(f"{where}: null is not allowed for {annotation}")

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError(f"{where}: expected a boolean, got {value!r}")

    if annotation is int:
        if isinstance(value, bool):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise ValueError(f"{where}: expected an integer, got {value!r}")

    if annotation is float:
        if isinstance(value, bool):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise ValueError(f"{where}: expected a number, got {value!r}")

    if annotation is str:
        if isinstance(value, str):
            return value
        raise ValueError(f"{where}: expected a string, got {value!r}")

    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a list, got {value!r}")
        (item_type,) = typing.get_args(annotation) or (Any,)
        return [_coerce(v, item_type, f"{where}[]") for v in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {value!r}")
        return dict(value)

    return value


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply a `section.key=value` override, parsing the value as YAML."""
    if "=" not in override:
        raise ValueError(f"Override must look like 'train.learning_rate=2e-5', got {override!r}")
    dotted, _, value = override.partition("=")
    node = raw
    keys = dotted.strip().split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot descend into {dotted!r}: {key!r} is not a mapping")
    node[keys[-1]] = yaml.safe_load(value)
