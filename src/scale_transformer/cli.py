"""Command line interface: `scale-train <config.yaml> [--set key.path=value ...]`."""

from __future__ import annotations

import argparse
import sys

from .config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scale-train",
        description="Single-device supervised finetuning for Qwen3 on Transformers.",
    )
    parser.add_argument("config", help="Path to a YAML experiment config (see configs/).")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.learning_rate=2e-5. Repeatable.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve the config, print it, and exit without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ExperimentConfig.load(args.config, args.overrides)

    if args.print_config:
        import json

        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    from .train import run

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
