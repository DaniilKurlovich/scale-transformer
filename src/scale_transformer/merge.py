"""Merge a trained LoRA adapter back into the base weights for inference/serving."""

from __future__ import annotations

import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scale-merge", description="Merge a LoRA adapter into its base model."
    )
    parser.add_argument("adapter", help="Directory written by training (e.g. outputs/.../final).")
    parser.add_argument("output", help="Directory to write the merged model to.")
    parser.add_argument(
        "--base", default=None, help="Override the base model recorded in the adapter."
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_name = args.base or peft_config.base_model_name_or_path
    logger.info("loading base model %s", base_name)

    # A 4-bit adapter still merges into full-precision base weights; that is intentional.
    base = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=getattr(torch, args.dtype), device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    logger.info("merged model written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
