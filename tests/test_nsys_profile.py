"""The rank-0 nsys capture window: start/stop on the right steps, only on the
profiled rank, NVTX ranges balanced, early exit after the window."""

import types

import pytest
import torch

from scale_transformer.nsys_profile import Nsys, NsysCapture, ProfiledTrainer


class FakeCudart:
    def __init__(self):
        self.calls = []

    def cudaProfilerStart(self):  # noqa: N802
        self.calls.append("start")

    def cudaProfilerStop(self):  # noqa: N802
        self.calls.append("stop")


@pytest.fixture
def nvtx_log(monkeypatch):
    log = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: log.append(("push", name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: log.append(("pop", None)))
    return log


def _run_steps(cap, n, body=None):
    for step in range(1, n + 1):
        cap.step_begin(step)
        with cap.range("train_step"):
            if body:
                body(step)
        cap.step_end(step)


def test_capture_window_on_profiled_rank(nvtx_log):
    cudart, syncs = FakeCudart(), []
    cap = NsysCapture(
        Nsys(enable=True, start_step=3, stop_step=5),
        rank=0,
        cudart=cudart,
        synchronize=lambda: syncs.append(1),
    )
    seen = {}
    _run_steps(cap, 8, body=lambda s: seen.__setitem__(s, cap.active))

    assert cudart.calls == ["start", "stop"]
    assert [s for s, a in seen.items() if a] == [3, 4, 5]
    assert len(syncs) == 1  # synchronize once, right before cudaProfilerStop
    assert cap.done


def test_other_ranks_never_touch_the_profiler(nvtx_log):
    cudart = FakeCudart()
    cap = NsysCapture(Nsys(enable=True, start_step=1, stop_step=2), rank=3, cudart=cudart)
    _run_steps(cap, 4)
    assert cudart.calls == []
    assert nvtx_log == []
    assert cap.done  # the *window* still passes on every rank (for the early exit)


def test_disabled_is_a_noop(nvtx_log):
    cudart = FakeCudart()
    cap = NsysCapture(Nsys(enable=False), rank=0, cudart=cudart)
    _run_steps(cap, 10)
    assert cudart.calls == [] and nvtx_log == [] and not cap.done


def test_nvtx_ranges_only_inside_window_and_balanced(nvtx_log):
    cap = NsysCapture(Nsys(enable=True, start_step=2, stop_step=3), rank=0, cudart=FakeCudart())

    def body(step):
        with cap.range("fwd_bwd"):
            pass

    _run_steps(cap, 4, body)
    pushes = [n for op, n in nvtx_log if op == "push"]
    assert pushes == ["step 2", "train_step", "fwd_bwd", "step 3", "train_step", "fwd_bwd"]
    assert sum(op == "pop" for op, _ in nvtx_log) == len(pushes)
    # depth returns to zero -> every push got its pop
    assert cap._range_depth == 0


def test_range_pops_on_exception(nvtx_log):
    cap = NsysCapture(Nsys(enable=True, start_step=1, stop_step=1), rank=0, cudart=FakeCudart())
    cap.step_begin(1)
    with pytest.raises(RuntimeError):
        with cap.range("boom"):
            raise RuntimeError("x")
    cap.step_end(1)
    assert nvtx_log.count(("pop", None)) == 2 and cap._range_depth == 0


def test_nvtx_can_be_disabled_independently(nvtx_log):
    cudart = FakeCudart()
    cap = NsysCapture(
        Nsys(enable=True, start_step=1, stop_step=2, nvtx=False), rank=0, cudart=cudart
    )
    _run_steps(cap, 2)
    assert cudart.calls == ["start", "stop"] and nvtx_log == []


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        NsysCapture(Nsys(enable=True, start_step=5, stop_step=4), rank=0)
    with pytest.raises(ValueError):
        NsysCapture(Nsys(enable=True, start_step=0, stop_step=4), rank=0)


def test_stop_step_beyond_training_end_still_stops_gracefully(nvtx_log):
    """Window longer than the run: no stop call is expected, and no crash."""
    cudart = FakeCudart()
    cap = NsysCapture(Nsys(enable=True, start_step=2, stop_step=100), rank=0, cudart=cudart)
    _run_steps(cap, 3)
    assert cudart.calls == ["start"] and cap._range_depth == 0


# ------------------------------------------------------ ProfiledTrainer glue ---


def _bare_profiled_trainer(nsys: Nsys, rank=0, total_steps=10):
    t = ProfiledTrainer.__new__(ProfiledTrainer)
    t.nsys = NsysCapture(nsys, rank, cudart=FakeCudart())
    t.step = 0
    t.job_config = types.SimpleNamespace(training=types.SimpleNamespace(steps=total_steps))
    return t


def test_trainer_stops_after_capture_when_asked(monkeypatch, nvtx_log):
    calls = []
    monkeypatch.setattr(
        "torchtitan.train.Trainer.train_step", lambda self, it: calls.append(self.step)
    )
    t = _bare_profiled_trainer(
        Nsys(enable=True, start_step=2, stop_step=3, stop_training_after_capture=True)
    )
    while t.should_continue_training():
        t.step += 1
        t.train_step(iter(()))
    assert calls == [1, 2, 3]
    assert t.nsys._cudart.calls == ["start", "stop"]


def test_trainer_runs_to_the_end_by_default(monkeypatch, nvtx_log):
    calls = []
    monkeypatch.setattr(
        "torchtitan.train.Trainer.train_step", lambda self, it: calls.append(self.step)
    )
    t = _bare_profiled_trainer(Nsys(enable=True, start_step=2, stop_step=3), total_steps=6)
    while t.should_continue_training():
        t.step += 1
        t.train_step(iter(()))
    assert calls == [1, 2, 3, 4, 5, 6]
    assert t.nsys._cudart.calls == ["start", "stop"]


def test_trainer_forward_backward_wrapped_in_range(monkeypatch, nvtx_log):
    monkeypatch.setattr(
        "torchtitan.train.Trainer.forward_backward_step",
        lambda self, *, input_dict, labels, global_valid_tokens: torch.tensor(1.0),
    )
    t = _bare_profiled_trainer(Nsys(enable=True, start_step=1, stop_step=1))
    t.step = 1
    t.nsys.step_begin(1)
    t.forward_backward_step(input_dict={}, labels=None, global_valid_tokens=None)
    t.nsys.step_end(1)
    assert ("push", "fwd_bwd") in nvtx_log
