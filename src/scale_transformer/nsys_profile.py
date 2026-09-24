"""Nsight Systems capture for one rank, wrapped around the stock torchtitan Trainer.

How it works
------------
`nsys` is started with `--capture-range=cudaProfilerApi`, so it records nothing
until a process calls `cudaProfilerStart()` and stops at `cudaProfilerStop()`.
Only the configured rank (default 0) ever calls those two, so a multi-GPU
`torchrun` job under `nsys profile` yields a trace of one rank's capture window
and the other ranks stay untouched (no `cudaProfilerStart` -> no capture).

`ProfiledTrainer` is the stock `Trainer` plus three things:
  * `cudaProfilerStart()` before `train_step` of `nsys.start_step`,
    `cudaProfilerStop()` after `train_step` of `nsys.stop_step` (both inclusive),
  * NVTX ranges around each step, forward/backward, grad clipping and the
    optimizer step, so the timeline is readable,
  * optional early exit right after the capture window, so `nsys` does not sit
    through the rest of the run (`nsys.stop_training_after_capture`).

With `nsys.enable = false` (the default) it is byte-for-byte the stock loop,
which is why `scale_transformer.train` always uses it.

Config (extends JobConfig via
`job.custom_config_module = "scale_transformer.nsys_profile"`):

    [nsys]
    enable = true
    start_step = 5
    stop_step = 8
    rank = 0
    stop_training_after_capture = true

Launch: see `run_nsys.sh`.
"""

import contextlib
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torchtitan.tools.logging import logger
from torchtitan.train import Trainer

# ------------------------------------------------------------- job config ---


@dataclass
class Nsys:
    enable: bool = False
    """Call cudaProfilerStart/Stop on `rank` around [start_step, stop_step]."""

    start_step: int = 5
    """First captured step (1-based, inclusive). Leave warmup/compile out."""

    stop_step: int = 8
    """Last captured step (inclusive)."""

    rank: int = 0
    """Global rank that triggers the capture. Every other rank is silent."""

    nvtx: bool = True
    """Push NVTX ranges (step / fwd_bwd / clip_grad / optimizer) on the profiled rank."""

    sync_before_stop: bool = True
    """torch.cuda.synchronize() before cudaProfilerStop so in-flight kernels land in the trace."""

    stop_training_after_capture: bool = False
    """End the run right after stop_step (all ranks) so nsys finalizes quickly."""


@dataclass
class JobConfig:
    """Merged into torchtitan's JobConfig by ConfigManager (job.custom_config_module)."""

    nsys: Nsys = field(default_factory=Nsys)


# ------------------------------------------------------------- the wrapper ---


class _NullCudart:
    """Stands in for torch.cuda.cudart() when CUDA is unavailable (tests, CPU)."""

    def cudaProfilerStart(self):  # noqa: N802 - mirrors the CUDA runtime name
        return 0

    def cudaProfilerStop(self):  # noqa: N802
        return 0


class NsysCapture:
    """cudaProfilerStart/Stop + NVTX ranges, gated on rank and step window.

    `cudart` and `synchronize` are injectable so the state machine is testable
    without a GPU.
    """

    def __init__(
        self,
        config: Nsys,
        rank: int,
        *,
        cudart: Any | None = None,
        synchronize=None,
    ):
        if config.enable and config.start_step > config.stop_step:
            raise ValueError(
                f"nsys.start_step ({config.start_step}) must be <= "
                f"nsys.stop_step ({config.stop_step})"
            )
        if config.enable and config.start_step < 1:
            raise ValueError("nsys.start_step must be >= 1 (steps are 1-based)")
        self.config = config
        self.rank = rank
        self.is_profiled_rank = config.enable and rank == config.rank
        self._cudart = cudart
        self._synchronize = synchronize
        self.started = False
        self.stopped = False
        self._range_depth = 0
        self._last_step_seen = 0

    # -- lazy so importing this module never touches the CUDA runtime
    @property
    def cudart(self):
        if self._cudart is None:
            self._cudart = torch.cuda.cudart() if torch.cuda.is_available() else _NullCudart()
        return self._cudart

    def synchronize(self) -> None:
        if self._synchronize is not None:
            self._synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    @property
    def active(self) -> bool:
        return self.started and not self.stopped

    @property
    def done(self) -> bool:
        """The whole capture window has passed (true on every rank, not only the profiled one)."""
        return self.config.enable and self._last_step_seen >= self.config.stop_step

    def step_begin(self, step: int) -> None:
        if not self.is_profiled_rank:
            return
        if step == self.config.start_step and not self.started:
            logger.info(f"[nsys] rank {self.rank}: cudaProfilerStart at step {step}")
            self.cudart.cudaProfilerStart()
            self.started = True
        if self.active:
            self.push(f"step {step}")

    def step_end(self, step: int) -> None:
        self._last_step_seen = max(self._last_step_seen, step)
        if not self.is_profiled_rank:
            return
        if self.active:
            self.pop()
        if step >= self.config.stop_step and self.started and not self.stopped:
            if self.config.sync_before_stop:
                self.synchronize()
            self.cudart.cudaProfilerStop()
            self.stopped = True
            logger.info(
                f"[nsys] rank {self.rank}: cudaProfilerStop at step {step} "
                f"(captured steps {self.config.start_step}..{step})"
            )

    # -- NVTX
    def _nvtx_enabled(self) -> bool:
        return self.is_profiled_rank and self.config.nvtx and self.active

    def push(self, name: str) -> None:
        if self._nvtx_enabled():
            torch.cuda.nvtx.range_push(name)
            self._range_depth += 1

    def pop(self) -> None:
        if self._nvtx_enabled() and self._range_depth > 0:
            torch.cuda.nvtx.range_pop()
            self._range_depth -= 1

    @contextlib.contextmanager
    def range(self, name: str):
        if not self._nvtx_enabled():
            yield
            return
        self.push(name)
        try:
            yield
        finally:
            self.pop()


# ------------------------------------------------------------- the trainer ---


class ProfiledTrainer(Trainer):
    """Stock torchtitan Trainer + one-rank nsys capture window + NVTX ranges."""

    def __init__(self, job_config):
        super().__init__(job_config)
        nsys_config = getattr(job_config, "nsys", None) or Nsys()
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else int(os.environ.get("RANK", 0))
        )
        self.nsys = NsysCapture(nsys_config, rank)
        if nsys_config.enable:
            logger.info(
                f"[nsys] enabled: capture rank {nsys_config.rank}, steps "
                f"{nsys_config.start_step}..{nsys_config.stop_step}, "
                f"stop_training_after_capture={nsys_config.stop_training_after_capture}"
            )
            self._wrap_in_ranges()

    def _wrap_in_ranges(self) -> None:
        """Wrap the optimizer step and grad clipping so they show up as NVTX ranges."""
        optimizers = self.optimizers
        original_step = optimizers.step

        def step_with_range(*args, **kwargs):
            with self.nsys.range("optimizer.step"):
                return original_step(*args, **kwargs)

        optimizers.step = step_with_range  # type: ignore[method-assign]

        import torchtitan.train as train_module

        dist_utils = train_module.dist_utils
        original_clip = dist_utils.clip_grad_norm_

        def clip_with_range(*args, **kwargs):
            with self.nsys.range("clip_grad_norm"):
                return original_clip(*args, **kwargs)

        # Patch the reference the Trainer looks up at call time, per process.
        dist_utils.clip_grad_norm_ = clip_with_range

    def train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
        self.nsys.step_begin(self.step)
        try:
            with self.nsys.range("train_step"):
                super().train_step(data_iterator)
        finally:
            self.nsys.step_end(self.step)

    def forward_backward_step(self, *, input_dict, labels, global_valid_tokens):
        with self.nsys.range("fwd_bwd"):
            return super().forward_backward_step(
                input_dict=input_dict, labels=labels, global_valid_tokens=global_valid_tokens
            )

    def should_continue_training(self) -> bool:
        if self.nsys.config.stop_training_after_capture and self.nsys.done:
            logger.info(f"[nsys] capture window closed at step {self.step}; ending the run")
            return False
        return super().should_continue_training()
