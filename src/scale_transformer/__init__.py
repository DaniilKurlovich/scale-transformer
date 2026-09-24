"""Qwen3-30B-A3B training on torchtitan: extra datasets, model flavors, profiling.

Every module here plugs into torchtitan's in-memory registries at import time
(``DATASETS``, ``qwen3_args``, the Qwen3 rope cache), so the installed package is
never patched on disk. ``scale_transformer.train`` is the entry point that
imports all of them; run it with ``torchrun -m scale_transformer.train``.
"""
