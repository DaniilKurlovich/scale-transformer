"""Training entry point: torchtitan's trainer plus this repo's registrations.

    torchrun --nproc_per_node=8 -m scale_transformer.train \
        --job.config_file configs/qwen3_30b_a3b.toml

Use this instead of `-m torchtitan.train`. Importing the modules below is what
makes `training.dataset = "pg19"`, `model.flavor = "30B-A3B-yarn"` and
`model.flavor = "smoke-moe"` resolve. `ProfiledTrainer` is the stock trainer
unless `[nsys] enable = true`; see `nsys_profile.py` and `run_nsys.sh`.

`scale-train` (the console script) is the same `main`, for
`torchrun ... $(which scale-train)`.
"""

from torchtitan.train import main as _torchtitan_main

from scale_transformer import pg19, smoke, yarn  # noqa: F401  registry side effects
from scale_transformer.nsys_profile import ProfiledTrainer


def main() -> None:
    _torchtitan_main(ProfiledTrainer)


if __name__ == "__main__":
    main()
