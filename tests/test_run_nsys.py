"""run_nsys.sh launcher logic: which node gets wrapped in nsys, and what torchrun
sees. `nsys` and `torchrun` are replaced by shims that log their argv."""

import os
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "run_nsys.sh")

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="bash script")


@pytest.fixture
def shims(tmp_path):
    """Fake nsys/torchrun on PATH; each appends its argv (one line) to <name>.log."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("nsys", "torchrun"):
        shim = bin_dir / name
        shim.write_text(f'#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "{tmp_path}/{name}.log"\n')
        shim.chmod(0o755)
    return tmp_path


def run(shims, *args, **env):
    full_env = {
        **os.environ,
        "PATH": f"{shims / 'bin'}:{os.environ['PATH']}",
        "OUT_DIR": str(shims / "out"),
    }
    for k in ("NNODES", "NODE_RANK", "NSYS_RANK", "MASTER_ADDR", "MASTER_PORT", "ENTRY"):
        full_env.pop(k, None)
    full_env.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        ["bash", SCRIPT, *args], cwd=ROOT, env=full_env, capture_output=True, text=True
    )


def calls(shims, name):
    log = shims / f"{name}.log"
    return [shlex.split(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_single_node_wraps_torchrun_in_nsys(shims):
    r = run(shims, "8", "configs/qwen3_30b_a3b.toml", "--nsys.start_step", "3")
    assert r.returncode == 0, r.stderr
    (nsys,) = calls(shims, "nsys")
    assert calls(shims, "torchrun") == []  # the shim nsys does not exec its child
    assert nsys[0] == "profile" and "--capture-range=cudaProfilerApi" in nsys
    i = nsys.index("torchrun")
    tr = nsys[i:]
    assert "--nproc_per_node=8" in tr and not any(a.startswith("--nnodes") for a in tr)
    assert tr[tr.index("--nsys.rank") + 1] == "0"
    assert "--nsys.enable" in tr and "--nsys.stop_training_after_capture" in tr
    assert tr[-2:] == ["--nsys.start_step", "3"]  # overrides reach the entry point
    out = next(a for a in nsys if a.startswith("--output="))
    assert "/rank0_" in out


def test_multi_node_only_hosting_node_gets_nsys(shims):
    for node in range(4):
        r = run(
            shims, "8", "configs/qwen3_30b_a3b.toml", NNODES=4, NODE_RANK=node, MASTER_ADDR="n0"
        )
        assert r.returncode == 0, r.stderr
    nsys = calls(shims, "nsys")
    plain = calls(shims, "torchrun")
    assert len(nsys) == 1 and len(plain) == 3
    wrapped = nsys[0][nsys[0].index("torchrun") :]
    assert "--node_rank=0" in wrapped
    assert sorted(a for c in plain for a in c if a.startswith("--node_rank=")) == [
        "--node_rank=1",
        "--node_rank=2",
        "--node_rank=3",
    ]
    for c in plain + [wrapped]:
        assert "--nnodes=4" in c and "--rdzv_endpoint=n0:29500" in c and "--rdzv_backend=c10d" in c
        assert (
            "--nsys.enable" in c and "--nsys.stop_training_after_capture" in c
        )  # all ranks exit together


def test_profiled_rank_picks_its_node(shims):
    for node in range(2):
        r = run(
            shims,
            "8",
            "configs/qwen3_30b_a3b.toml",
            NNODES=2,
            NODE_RANK=node,
            NSYS_RANK=9,
            MASTER_ADDR="n0",
        )
        assert r.returncode == 0, r.stderr
    (nsys,) = calls(shims, "nsys")
    assert "--node_rank=1" in nsys
    assert nsys[nsys.index("--nsys.rank") + 1] == "9"
    assert "/rank9_" in next(a for a in nsys if a.startswith("--output="))
    (plain,) = calls(shims, "torchrun")
    assert "--node_rank=0" in plain and plain[plain.index("--nsys.rank") + 1] == "9"


def test_rank_outside_world_is_rejected(shims):
    r = run(shims, "8", "configs/qwen3_30b_a3b.toml", NNODES=2, NODE_RANK=0, NSYS_RANK=16)
    assert r.returncode == 2 and "outside the world" in r.stderr
    assert calls(shims, "nsys") == [] and calls(shims, "torchrun") == []


def test_nsys_rank_override_must_go_through_env(shims):
    r = run(shims, "8", "configs/qwen3_30b_a3b.toml", "--nsys.rank", "3")
    assert r.returncode == 2 and "NSYS_RANK" in r.stderr
    assert calls(shims, "nsys") == []
