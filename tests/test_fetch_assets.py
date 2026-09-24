"""fetch_assets: the plan derived from a job config, the marker that makes it
idempotent, and the PG-19 loader reading a local parquet mirror."""

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scale_transformer import fetch_assets as fa

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _by_dir(fetches):
    return {f.local_dir: f for f in fetches}


def test_tokenizer_only_when_streaming_from_hub():
    cfg = {"model": {"hf_assets_path": "./assets/qwen3-30b-a3b"}, "training": {"dataset": "pg19"}}
    (f,) = fa.plan(cfg)
    assert f.repo_id == "Qwen/Qwen3-30B-A3B" and f.repo_type == "model"
    assert f.local_dir == "assets/qwen3-30b-a3b"
    assert set(f.allow_patterns) == set(fa.TOKENIZER_PATTERNS)


def test_weights_and_dataset_for_continued_pretraining():
    cfg = {
        "model": {"hf_assets_path": "./assets/qwen3-30b-a3b-base"},
        "checkpoint": {
            "initial_load_path": "./assets/qwen3-30b-a3b-base",
            "initial_load_in_hf": True,
        },
        "training": {"dataset": "pg19", "dataset_path": "./data/pg19"},
        "validation": {"enable": True, "dataset": "pg19_validation", "dataset_path": "./data/pg19"},
    }
    plan = _by_dir(fa.plan(cfg))
    assert set(plan) == {"assets/qwen3-30b-a3b-base", "data/pg19"}
    weights = plan["assets/qwen3-30b-a3b-base"]
    assert weights.repo_id == "Qwen/Qwen3-30B-A3B-Base"
    assert set(fa.WEIGHT_PATTERNS) <= set(weights.allow_patterns)
    assert set(fa.TOKENIZER_PATTERNS) <= set(weights.allow_patterns)
    data = plan["data/pg19"]
    assert data.repo_id == "emozilla/pg19" and data.repo_type == "dataset"
    assert data.reasons == ["training.dataset_path", "validation.dataset_path"]


def test_no_weights_without_initial_load_in_hf():
    cfg = {
        "model": {"hf_assets_path": "./assets/qwen3-30b-a3b"},
        "checkpoint": {"initial_load_path": "./assets/qwen3-30b-a3b"},
    }
    (f,) = fa.plan(cfg)
    assert not set(fa.WEIGHT_PATTERNS) & set(f.allow_patterns)


def test_unknown_assets_dir_needs_repo_override():
    cfg = {"model": {"hf_assets_path": "./assets/other-model"}}
    with pytest.raises(ValueError, match="--repo"):
        fa.plan(cfg)
    (f,) = fa.plan(cfg, {"./assets/other-model": "org/other"})
    assert f.repo_id == "org/other"


@pytest.mark.parametrize(
    "name", ["qwen3_30b_a3b.toml", "qwen3_30b_a3b_yarn.toml", "qwen3_smoke_1gpu.toml"]
)
def test_shipped_configs_plan(name):
    plan = _by_dir(fa.plan(fa.load_config(os.path.join(ROOT, "configs", name))))
    assert any(f.repo_type == "model" for f in plan.values())
    if name == "qwen3_smoke_1gpu.toml":
        assert "data/pg19" not in plan  # smoke test streams from the Hub
    else:
        assert plan["data/pg19"].repo_type == "dataset"
    if name == "qwen3_30b_a3b_yarn.toml":
        assert set(fa.WEIGHT_PATTERNS) <= set(plan["assets/qwen3-30b-a3b-base"].allow_patterns)


def test_marker_makes_run_idempotent(monkeypatch, tmp_path):
    calls = []

    def fake_snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, tuple(kwargs["allow_patterns"])))
        os.makedirs(kwargs["local_dir"], exist_ok=True)

    monkeypatch.setattr(fa, "snapshot_download", fake_snapshot_download)
    d = str(tmp_path / "assets" / "m")
    tok = fa.Fetch("org/m", d, fa.TOKENIZER_PATTERNS)

    assert fa.run([tok]) == [tok] and len(calls) == 1
    assert fa.run([tok]) == [] and len(calls) == 1  # marker covers it: no download
    with open(os.path.join(d, fa.MARKER)) as f:
        assert set(json.load(f)["allow_patterns"]) == set(fa.TOKENIZER_PATTERNS)

    # Asking for more (weights) than the marker records fetches again ...
    full = fa.Fetch("org/m", d, fa.TOKENIZER_PATTERNS + fa.WEIGHT_PATTERNS)
    assert fa.run([full]) == [full] and len(calls) == 2
    # ... and the marker now covers both the old and the new request.
    assert fa.run([tok]) == [] and fa.run([full]) == [] and len(calls) == 2
    assert fa.run([tok], force=True) == [tok] and len(calls) == 3


def test_pg19_loader_reads_a_local_parquet_mirror(tmp_path):
    """`training.dataset_path` pointing at the snapshot_download layout works
    without a config name: datasets infers parquet + splits from data/<split>-*."""
    from torchtitan.hf_datasets.text_datasets import DATASETS

    from scale_transformer import pg19  # noqa: F401

    data = tmp_path / "data"
    data.mkdir()
    for split, texts in {"train": ["book one", "book two"], "validation": ["held out"]}.items():
        table = pa.table({"short_book_title": texts, "text": texts})
        pq.write_table(table, data / f"{split}-00000-of-00001.parquet")

    cfg = DATASETS["pg19"]
    rows = list(cfg.loader(str(tmp_path)))
    assert [cfg.sample_processor(r) for r in rows] == ["book one", "book two"]
    val = list(DATASETS["pg19_validation"].loader(str(tmp_path)))
    assert [r["text"] for r in val] == ["held out"]
