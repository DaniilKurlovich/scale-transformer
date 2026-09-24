"""YaRN RoPE scaling for the Qwen3 stack, injected without patching the package.

Same trick as `pg19.py`: importing this module mutates torchtitan's in-memory
registries, so the installed package stays untouched. `scale_transformer.train`
imports it, which is why that entry point (not `-m torchtitan.train`) is required.

Two things happen at import time:

1. `register_yarn_flavor("30B-A3B")` clones the stock `30B-A3B` model args into
   `30B-A3B-yarn`, a `Qwen3YarnModelArgs` carrying the YaRN knobs.
2. `Qwen3Model._precompute_rope_cache` is wrapped so that a model built from
   those args gets a YaRN-scaled rope cache instead of the plain one. Every
   other flavor keeps the original code path.

The rope cache is a non-persistent buffer, so nothing about this changes the
checkpoint format -- `Qwen3StateDictAdapter` maps the same tensors as before.
The scaling lives in the *config*, which is why `stamp_hf_config()` (see
`--stamp` below) has to write `rope_scaling` into the exported HF config.json;
otherwise transformers/vLLM will serve the model with unscaled RoPE and undo
the training you just paid for.

Math is a line-by-line port of `transformers.modeling_rope_utils
._compute_yarn_parameters`, so training-time and inference-time frequencies
agree. `python -m scale_transformer.yarn --check` asserts that against
transformers directly.
"""

import copy
import math
from dataclasses import dataclass, fields

import torch
from torchtitan.config import JobConfig
from torchtitan.models.qwen3 import qwen3_args
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from torchtitan.models.qwen3.model.model import Qwen3Model, precompute_rope_cache
from torchtitan.tools.logging import logger

# ---------------------------------------------------------------- the math ---


def _find_correction_dim(
    num_rotations: float, dim: int, base: float, original_max_seq_len: int
) -> float:
    """Which rope dimension completes `num_rotations` over the original window."""
    return (dim * math.log(original_max_seq_len / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _find_correction_range(
    beta_fast: float,
    beta_slow: float,
    dim: int,
    base: float,
    original_max_seq_len: int,
    truncate: bool,
) -> tuple[float, float]:
    low = _find_correction_dim(beta_fast, dim, base, original_max_seq_len)
    high = _find_correction_dim(beta_slow, dim, base, original_max_seq_len)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if low == high:
        high += 0.001  # prevent a divide by zero
    return torch.clamp((torch.arange(dim, dtype=torch.float32) - low) / (high - low), 0, 1)


def yarn_attention_factor(factor: float) -> float:
    """YaRN's 1/sqrt(t) temperature, folded into cos/sin (paper eq. 22)."""
    return 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_max_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    truncate: bool = True,
) -> torch.Tensor:
    """NTK-by-parts: interpolate low-frequency dims, extrapolate high-frequency ones."""
    pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)

    low, high = _find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_seq_len, truncate
    )
    # 1 at the fast dims (keep them untouched -> extrapolate), 0 at the slow
    # dims (stretch them by `factor`), a linear ramp in between.
    extrapolation_mask = 1 - _linear_ramp(low, high, dim // 2)
    return (
        inv_freq_interpolation * (1 - extrapolation_mask)
        + inv_freq_extrapolation * extrapolation_mask
    )


def precompute_rope_cache_yarn(
    dim: int,
    max_seq_len: int,
    base: float,
    factor: float,
    original_max_seq_len: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float | None = None,
    truncate: bool = True,
) -> torch.Tensor:
    """Drop-in replacement for `precompute_rope_cache`; same [max_seq_len, dim*2] layout."""
    inv_freq = yarn_inv_freq(
        dim, base, factor, original_max_seq_len, beta_fast, beta_slow, truncate
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    idx_theta = torch.outer(t, inv_freq).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)

    if attention_factor is None:
        attention_factor = yarn_attention_factor(factor)
    # Scaling cos and sin scales both q and k, i.e. the logits get
    # attention_factor**2 -- that squared term is YaRN's 1/t.
    return torch.cat([freqs.cos() * attention_factor, freqs.sin() * attention_factor], dim=-1)


# ------------------------------------------------------------- model args ---


@dataclass
class Qwen3YarnModelArgs(Qwen3ModelArgs):
    """Qwen3 args + YaRN knobs.

    `yarn_original_max_seq_len` is the window the weights were pretrained with
    (32768 for Qwen3-30B-A3B-Base, per its config.json) and must NOT follow
    `training.seq_len` -- it is the denominator the scaling is defined against.
    `max_seq_len` still tracks `training.seq_len`, because it only sizes the
    precomputed cos/sin table.
    """

    yarn_factor: float = 1.0
    yarn_original_max_seq_len: int = 32768
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_attention_factor: float | None = None  # None -> 0.1*ln(factor)+1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        target = int(self.yarn_original_max_seq_len * self.yarn_factor)
        if seq_len > target:
            raise ValueError(
                f"training.seq_len {seq_len} exceeds the YaRN target window {target} "
                f"({self.yarn_original_max_seq_len} x {self.yarn_factor}); "
                "raise yarn_factor instead of training past the scaled window."
            )
        self.max_seq_len = seq_len
        self.moe_args._debug_force_load_balance = job_config.debug.moe_force_load_balance
        attention_factor = self.yarn_attention_factor or yarn_attention_factor(self.yarn_factor)
        logger.info(
            f"YaRN enabled: factor={self.yarn_factor} "
            f"({self.yarn_original_max_seq_len} -> {target} tokens), "
            f"beta_fast={self.yarn_beta_fast}, beta_slow={self.yarn_beta_slow}, "
            f"attention_factor={attention_factor:.4f}; training at seq_len={seq_len}"
        )


def register_yarn_flavor(
    src_flavor: str,
    name: str | None = None,
    *,
    factor: float = 8.0,
    original_max_seq_len: int = 32768,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attention_factor: float | None = None,
) -> str:
    """Clone `qwen3_args[src_flavor]` into a YaRN flavor and register it."""
    src = qwen3_args[src_flavor]
    name = name or f"{src_flavor}-yarn"
    kwargs = {f.name: copy.deepcopy(getattr(src, f.name)) for f in fields(Qwen3ModelArgs)}
    qwen3_args[name] = Qwen3YarnModelArgs(
        **kwargs,
        yarn_factor=factor,
        yarn_original_max_seq_len=original_max_seq_len,
        yarn_beta_fast=beta_fast,
        yarn_beta_slow=beta_slow,
        yarn_attention_factor=attention_factor,
    )
    return name


# ------------------------------------------------------------- the patch ---

_stock_precompute_rope_cache = Qwen3Model._precompute_rope_cache


def _precompute_rope_cache_dispatch(self) -> torch.Tensor:
    args = self.model_args
    if not isinstance(args, Qwen3YarnModelArgs) or args.yarn_factor <= 1.0:
        return _stock_precompute_rope_cache(self)
    return precompute_rope_cache_yarn(
        args.head_dim,
        args.max_seq_len,
        args.rope_theta,
        factor=args.yarn_factor,
        original_max_seq_len=args.yarn_original_max_seq_len,
        beta_fast=args.yarn_beta_fast,
        beta_slow=args.yarn_beta_slow,
        attention_factor=args.yarn_attention_factor,
    )


Qwen3Model._precompute_rope_cache = _precompute_rope_cache_dispatch

# 32768 x 8 = 262144, the window the 30B-A3B flavor already declares.
register_yarn_flavor("30B-A3B", factor=8.0, original_max_seq_len=32768)


# ------------------------------------------------- HF config.json stamping ---


def stamp_hf_config(
    checkpoint_dir: str,
    factor: float,
    original_max_seq_len: int = 32768,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> dict:
    """Write `rope_scaling` into a HF config.json so inference reproduces training.

    torchtitan's HF export writes safetensors + index only, so copy config.json
    and the tokenizer from ./assets/qwen3-30b-a3b-base into the export dir first.
    """
    import json
    import os

    path = os.path.join(checkpoint_dir, "config.json")
    with open(path) as f:
        config = json.load(f)

    config["rope_scaling"] = {
        "rope_type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max_seq_len,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }
    config["max_position_embeddings"] = int(original_max_seq_len * factor)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return config["rope_scaling"]


def _check_against_transformers() -> None:
    """Assert our rope cache matches transformers' YaRN for the same config."""
    import shutil
    import tempfile

    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    dim, base, factor, original = 128, 1_000_000.0, 8.0, 32768
    seq_len = 4096

    # Round-trip through the stamped file: this checks `stamp_hf_config` output
    # is what transformers actually reads, not just that the formulas agree.
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy("./assets/qwen3-30b-a3b-base/config.json", tmp)
        stamp_hf_config(tmp, factor, original)
        config = AutoConfig.from_pretrained(tmp)
    hf_inv_freq, hf_attention_factor = ROPE_INIT_FUNCTIONS["yarn"](config, device="cpu")

    ours = yarn_inv_freq(dim, base, factor, original)
    torch.testing.assert_close(ours, hf_inv_freq)
    assert math.isclose(hf_attention_factor, yarn_attention_factor(factor)), hf_attention_factor

    # ... and the full cos/sin table, the thing the model actually consumes.
    t = torch.arange(seq_len, dtype=torch.float32)
    idx = torch.outer(t, hf_inv_freq)
    hf_freqs = torch.cat([idx, idx], dim=-1)
    hf_cache = torch.cat(
        [hf_freqs.cos() * hf_attention_factor, hf_freqs.sin() * hf_attention_factor], dim=-1
    )
    torch.testing.assert_close(
        precompute_rope_cache_yarn(dim, seq_len, base, factor, original), hf_cache
    )

    # And sanity-check the shape contract against the stock (unscaled) cache.
    assert precompute_rope_cache_yarn(dim, seq_len, base, factor, original).shape == (
        precompute_rope_cache(dim, seq_len, base).shape
    )
    print(
        f"OK: YaRN matches transformers (factor={factor}, "
        f"attention_factor={hf_attention_factor:.4f})"
    )
    print(f"registered flavors: {[k for k in qwen3_args if k.endswith('-yarn')]}")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify the math against transformers")
    p.add_argument("--stamp", metavar="DIR", help="write rope_scaling into DIR/config.json")
    p.add_argument("--factor", type=float, default=8.0)
    p.add_argument("--original-max-seq-len", type=int, default=32768)
    a = p.parse_args()

    if a.check:
        _check_against_transformers()
    if a.stamp:
        print(stamp_hf_config(a.stamp, a.factor, a.original_max_seq_len))


if __name__ == "__main__":
    main()
