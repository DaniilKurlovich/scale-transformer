"""Model and tokenizer construction: dtype/attention selection, MoE knobs, LoRA wrapping."""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .config import ExperimentConfig, LoraConfig, ModelConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "auto": "auto",
}

# Qwen3-MoE expert / dense-layer MLP projections. Targeting these multiplies the adapter
# count by the number of experts (128 for 30B-A3B), so it is opt-in.
_MLP_PROJECTIONS = ["gate_proj", "up_proj", "down_proj"]


def resolve_dtype(name: str):
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype {name!r}; expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def build_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        # Qwen3 ships a distinct <|endoftext|> pad token; this is just a safety net.
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM training: pad on the right so positions line up with labels.
    tokenizer.padding_side = "right"
    return tokenizer


def _quantization_config(cfg: ModelConfig):
    if not cfg.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("model.load_in_4bit requires a CUDA device (bitsandbytes).")
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=resolve_dtype(cfg.dtype),
    )


def _apply_moe_settings(model_config, cfg: ModelConfig) -> bool:
    """Enable the router auxiliary loss on MoE checkpoints. Returns True if MoE."""
    is_moe = hasattr(model_config, "num_experts") or hasattr(model_config, "router_aux_loss_coef")
    if not is_moe:
        if cfg.output_router_logits:
            logger.info("model.output_router_logits ignored: %s is not an MoE checkpoint.",
                        cfg.name_or_path)
        return False

    model_config.output_router_logits = cfg.output_router_logits
    if cfg.router_aux_loss_coef is not None:
        model_config.router_aux_loss_coef = cfg.router_aux_loss_coef
    logger.info(
        "MoE checkpoint: %s experts, top-%s routed; aux loss %s (coef=%s).",
        getattr(model_config, "num_experts", "?"),
        getattr(model_config, "num_experts_per_tok", "?"),
        "on" if cfg.output_router_logits else "off",
        getattr(model_config, "router_aux_loss_coef", None),
    )
    return True


def build_model(cfg: ExperimentConfig):
    model_cfg = cfg.model
    model_config = AutoConfig.from_pretrained(
        model_cfg.name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    _apply_moe_settings(model_config, model_cfg)
    # The cache is dead weight during training and conflicts with gradient checkpointing.
    model_config.use_cache = False

    # Stream weights straight onto the GPU instead of materialising them in host RAM first;
    # this is single-device placement, not model parallelism.
    device_map = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name_or_path,
        config=model_config,
        dtype=resolve_dtype(model_cfg.dtype),
        attn_implementation=model_cfg.attn_implementation,
        trust_remote_code=model_cfg.trust_remote_code,
        quantization_config=_quantization_config(model_cfg),
        device_map=device_map,
    )

    if model_cfg.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    if cfg.lora.enabled:
        model = _wrap_lora(model, cfg.lora)

    if model_cfg.gradient_checkpointing:
        # LoRA freezes the embeddings, so without this the checkpointed blocks receive no
        # input that requires grad and the backward pass silently produces nothing.
        model.enable_input_require_grads()

    return model


def _wrap_lora(model, cfg: LoraConfig):
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    targets = list(cfg.target_modules)
    if cfg.target_expert_mlp:
        targets += [p for p in _MLP_PROJECTIONS if p not in targets]
        logger.warning(
            "lora.target_expert_mlp is on: every expert MLP gets an adapter pair. "
            "Expect a large trainable-parameter count and slower steps."
        )

    peft_config = PeftLoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=targets,
        modules_to_save=cfg.modules_to_save or None,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
