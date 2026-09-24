"""A Qwen3-MoE flavor small enough for one GPU.

Same model class, router, grouped-expert kernels and tokenizer as `30B-A3B`, so
`configs/qwen3_smoke_1gpu.toml` exercises the real code path before the
multi-GPU run. Importing this module registers the flavor as `smoke-moe`.
"""

from torchtitan.models.moe import MoEArgs
from torchtitan.models.qwen3 import qwen3_args
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs

qwen3_args["smoke-moe"] = Qwen3ModelArgs(
    vocab_size=151936,  # must match the Qwen3 tokenizer
    max_seq_len=4096,
    head_dim=64,
    dim=512,
    n_layers=4,
    n_heads=8,
    n_kv_heads=4,
    qk_norm=True,
    hidden_dim=1024,
    rope_theta=1000000,
    moe_enabled=True,
    moe_inter_dim=256,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=0,
        top_k=2,
        score_func="softmax",
        route_norm=True,
        route_scale=1.0,
        score_before_experts=False,
    ),
)
