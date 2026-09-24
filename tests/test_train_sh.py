"""train.sh: what one node hands to torchrun, where it logs, and that the fetch
runs first. torchrun / python / flock / nvidia-smi are shims that log their argv."""

import os
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "train.sh")

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash script")


@pytest.fixture
def box(tmp_path):
    """A fake container: shims on PATH, an empty $WORKSPACE, the checkout as PROJECT_DIR."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("torchrun", "python", "nsys"):
        shim = bin_dir / name
        shim.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/{name}.log"\n')
        shim.chmod(0o755)
    flock = bin_dir / "flock"  # flock FILE CMD... -> just run CMD (macOS has no flock)
    flock.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n')
    flock.chmod(0o755)
    smi = bin_dir / "nvidia-smi"
    smi.write_text('#!/usr/bin/env bash\nprintf "GPU 0: H100\\nGPU 1: H100\\n"\n')
    smi.chmod(0o755)
    (tmp_path / "ws").mkdir()
    return tmp_path


def run(box, *args, **env):
    full_env = {
        **os.environ,
        "PATH": f"{box / 'bin'}:{os.environ['PATH']}",
        "WORKSPACE": str(box / "ws"),
        "PROJECT_DIR": ROOT,
    }
    for k in (
        "NNODES",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "NPROC_PER_NODE",
        "NSYS",
        "PYTHONPATH",
    ):
        full_env.pop(k, None)
    full_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        ["bash", SCRIPT, *args], cwd=str(box), env=full_env, capture_output=True, text=True
    )


def calls(box, name):
    log = box / f"{name}.log"
    return [shlex.split(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_single_node_fetches_then_trains(box):
    r = run(box, "configs/qwen3_30b_a3b.toml", "--training.steps", "5")
    assert r.returncode == 0, r.stderr
    config = os.path.join(ROOT, "configs/qwen3_30b_a3b.toml")

    (fetch,) = calls(box, "python")
    assert fetch[:3] == ["-m", "scale_transformer.fetch_assets", "--config"] and fetch[3] == config

    (tr,) = calls(box, "torchrun")
    assert tr[0] == "--nproc_per_node=2"  # from the nvidia-smi shim
    assert not any(a.startswith("--nnodes") for a in tr)
    i = tr.index("-m")
    assert tr[i + 1 : i + 4] == ["scale_transformer.train", "--job.config_file", config]
    assert tr[-2:] == ["--training.steps", "5"]

    logs = os.listdir(box / "ws" / "outputs" / "logs")
    assert len(logs) == 1 and logs[0].startswith("node0_")
    assert "train.sh: config=" in (box / "ws" / "outputs" / "logs" / logs[0]).read_text()


def test_two_nodes_rendezvous_on_master(box):
    for rank in range(2):
        r = run(box, NNODES=2, NODE_RANK=rank, MASTER_ADDR="10.0.0.1", NPROC_PER_NODE=8)
        assert r.returncode == 0, r.stderr
    trs = calls(box, "torchrun")
    assert len(trs) == 2
    for rank, tr in enumerate(trs):
        assert "--nproc_per_node=8" in tr and "--nnodes=2" in tr and f"--node_rank={rank}" in tr
        assert "--rdzv_backend=c10d" in tr and "--rdzv_endpoint=10.0.0.1:29500" in tr
        assert tr[tr.index("--job.config_file") + 1].endswith("configs/qwen3_30b_a3b.toml")
    names = sorted(os.listdir(box / "ws" / "outputs" / "logs"))
    assert [n[:5] for n in names] == ["node0", "node1"]


def test_multi_node_requires_master_addr(box):
    r = run(box, NNODES=2, NODE_RANK=1)
    assert r.returncode != 0 and "MASTER_ADDR" in r.stderr
    assert calls(box, "torchrun") == [] and calls(box, "python") == []


def test_missing_config_is_an_error(box):
    r = run(box, "configs/nope.toml")
    assert r.returncode == 2 and "config not found" in r.stderr


def test_nsys_mode_goes_through_run_nsys(box):
    r = run(box, "configs/qwen3_30b_a3b.toml", NSYS=1, NPROC_PER_NODE=8)
    assert r.returncode == 0, r.stderr
    (nsys,) = calls(box, "nsys")
    assert nsys[0] == "profile" and "torchrun" in nsys
    assert "--nsys.enable" in nsys and "scale_transformer.train" in nsys
    out = next(a for a in nsys if a.startswith("--output="))
    assert out.startswith(f"--output={box / 'ws'}/outputs/nsys/rank0_")


def test_synced_checkout_in_workspace_wins(box):
    """A checkout under $WORKSPACE puts its src/ on PYTHONPATH and supplies the config."""
    ws = box / "ws"
    (ws / "src" / "scale_transformer").mkdir(parents=True)
    (ws / "configs").mkdir()
    (ws / "configs" / "local.toml").write_text("[job]\n")
    env_dump = box / "bin" / "torchrun"
    env_dump.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$PYTHONPATH" >> "{box}/pythonpath.log"\n'
    )
    r = run(box, "configs/local.toml", PROJECT_DIR="")  # empty -> auto-detect
    assert r.returncode == 0, r.stderr
    assert (box / "pythonpath.log").read_text().splitlines() == [str(ws / "src")]
    (fetch,) = calls(box, "python")
    assert fetch[3] == str(ws / "configs" / "local.toml")
