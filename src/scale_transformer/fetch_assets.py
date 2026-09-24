"""Fetch what a training config needs from the HF Hub, idempotently.

    python -m scale_transformer.fetch_assets --config configs/qwen3_30b_a3b.toml

Reads the TOML and downloads, relative to the current directory, whatever is
missing:

  * `model.hf_assets_path`         tokenizer + config.json, plus the safetensors
                                   when `checkpoint.initial_load_path` points at
                                   the same directory with `initial_load_in_hf`
  * `training.dataset_path`        the PG-19 parquet shards, when
                                   `training.dataset` is pg19 (no path -> the run
                                   streams from the Hub and nothing is fetched)
  * `validation.dataset_path`      same, when `validation.enable` is on

Every directory gets a `.fetched` marker listing the patterns it holds; a later
call whose patterns the marker already covers makes no network requests at all,
so `train.sh` can run this on every start. `snapshot_download` resumes partial
downloads, so an interrupted fetch just picks up where it stopped.
"""

import argparse
import json
import os
from dataclasses import dataclass, field

from huggingface_hub import snapshot_download

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, same fallback torchtitan uses
    import tomli as tomllib

# Local directory basename -> Hub repo. Extend with --repo DIR=REPO.
ASSET_REPOS = {
    "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
    "qwen3-30b-a3b-base": "Qwen/Qwen3-30B-A3B-Base",
}
DATASET_REPOS = {
    "pg19": "emozilla/pg19",
    "pg19_validation": "emozilla/pg19",
}

# Same set the README's snapshot_download one-liner uses.
TOKENIZER_PATTERNS = ("tokenizer*", "*config*.json", "*.txt")
WEIGHT_PATTERNS = ("*.safetensors", "*.safetensors.index.json")
PG19_PATTERNS = ("data/train-*.parquet", "data/validation-*.parquet")

MARKER = ".fetched"


@dataclass
class Fetch:
    repo_id: str
    local_dir: str
    allow_patterns: tuple[str, ...]
    repo_type: str = "model"
    reasons: list[str] = field(default_factory=list)


def _norm(path: str) -> str:
    return os.path.normpath(path)


def plan(config: dict, repo_overrides: dict[str, str] | None = None) -> list[Fetch]:
    """Which repos to pull into which directories for this job config."""
    overrides = {_norm(k): v for k, v in (repo_overrides or {}).items()}
    fetches: dict[str, Fetch] = {}

    def add(repo_id, local_dir, patterns, repo_type, reason):
        key = _norm(local_dir)
        if key in fetches:
            f = fetches[key]
            if f.repo_id != repo_id:
                raise ValueError(f"{local_dir} is wanted from both {f.repo_id} and {repo_id}")
            f.allow_patterns = tuple(dict.fromkeys(f.allow_patterns + tuple(patterns)))
            f.reasons.append(reason)
        else:
            fetches[key] = Fetch(repo_id, key, tuple(patterns), repo_type, [reason])

    def asset_repo(local_dir: str) -> str:
        key = _norm(local_dir)
        if key in overrides:
            return overrides[key]
        name = os.path.basename(key)
        if name in ASSET_REPOS:
            return ASSET_REPOS[name]
        raise ValueError(
            f"no Hub repo known for assets dir {local_dir!r}; pass --repo {local_dir}=<org/name>"
        )

    model = config.get("model", {})
    assets = model.get("hf_assets_path")
    if assets:
        add(asset_repo(assets), assets, TOKENIZER_PATTERNS, "model", "model.hf_assets_path")

    ckpt = config.get("checkpoint", {})
    load_path = ckpt.get("initial_load_path")
    if load_path and ckpt.get("initial_load_in_hf", False):
        add(
            asset_repo(load_path),
            load_path,
            TOKENIZER_PATTERNS + WEIGHT_PATTERNS,
            "model",
            "checkpoint.initial_load_path (initial_load_in_hf)",
        )

    sections = [("training", config.get("training", {}))]
    validation = config.get("validation", {})
    if validation.get("enable", False):
        sections.append(("validation", validation))
    for name, section in sections:
        dataset = section.get("dataset")
        path = section.get("dataset_path")
        if dataset in DATASET_REPOS and path:
            add(DATASET_REPOS[dataset], path, PG19_PATTERNS, "dataset", f"{name}.dataset_path")

    return list(fetches.values())


def _marker_covers(local_dir: str, patterns: tuple[str, ...]) -> bool:
    try:
        with open(os.path.join(local_dir, MARKER)) as f:
            done = set(json.load(f).get("allow_patterns", []))
    except (OSError, ValueError):
        return False
    return set(patterns) <= done


def run(fetches: list[Fetch], *, force: bool = False) -> list[Fetch]:
    """Download every fetch whose marker does not already cover it; return the ones fetched."""
    fetched = []
    for f in fetches:
        if not force and _marker_covers(f.local_dir, f.allow_patterns):
            print(f"fetch_assets: {f.local_dir} already has {f.repo_id} ({', '.join(f.reasons)})")
            continue
        print(
            f"fetch_assets: {f.repo_id} [{f.repo_type}] -> {f.local_dir} "
            f"patterns={list(f.allow_patterns)} ({', '.join(f.reasons)})"
        )
        snapshot_download(
            f.repo_id,
            repo_type=f.repo_type,
            local_dir=f.local_dir,
            allow_patterns=list(f.allow_patterns),
        )
        previous = set()
        try:
            with open(os.path.join(f.local_dir, MARKER)) as fh:
                previous = set(json.load(fh).get("allow_patterns", []))
        except (OSError, ValueError):
            pass
        with open(os.path.join(f.local_dir, MARKER), "w") as fh:
            json.dump(
                {"repo_id": f.repo_id, "allow_patterns": sorted(previous | set(f.allow_patterns))},
                fh,
                indent=2,
            )
        fetched.append(f)
    return fetched


def load_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", required=True, help="torchtitan job TOML")
    p.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="DIR=ORG/NAME",
        help="Hub repo for an assets dir the built-in table does not know",
    )
    p.add_argument(
        "--force", action="store_true", help="re-run snapshot_download even if marked done"
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = p.parse_args(argv)

    overrides = dict(item.split("=", 1) for item in a.repo)
    fetches = plan(load_config(a.config), overrides)
    if not fetches:
        print("fetch_assets: nothing to fetch for this config")
        return
    if a.dry_run:
        for f in fetches:
            print(f"{f.repo_id} [{f.repo_type}] -> {f.local_dir} {list(f.allow_patterns)}")
        return
    run(fetches, force=a.force)


if __name__ == "__main__":
    main()
