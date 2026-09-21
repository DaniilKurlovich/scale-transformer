# syntax=docker/dockerfile:1.9
#
# Training image for scale-transformer: CUDA toolkit, Nsight Systems (nsys) and NCCL.
#
#   Build:      DOCKER_BUILDKIT=1 docker build -t scale-transformer .
#   With tests: DOCKER_BUILDKIT=1 docker build --target nccl-tests -t scale-transformer:nccl .
#
#   Run:        docker run --rm -it --gpus all \
#                 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
#                 -v "$PWD":/workspace -v hf-cache:/cache/huggingface \
#                 scale-transformer
#
#   Profile:    add --cap-add=SYS_ADMIN  (nsys needs it for CPU/GPU sampling counters)
#
# Note: nccl-tests links against the system NCCL that the CUDA base image ships
# pinned (the build log prints its version). torch loads its own bundled
# nvidia-nccl-cu13 2.30.7 instead, so the two are not the same build -- keep that
# in mind when comparing their numbers.
#
# The torch wheels pinned in uv.lock are CUDA 13 builds (nvidia-nccl-cu13, cudnn-cu13),
# so the base image is CUDA 13 to keep nvcc and the driver ABI in step. If you re-pin
# torch to a cu12 build, move CUDA_VERSION back to a 12.x tag as well.

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04
ARG UV_VERSION=0.12.15

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv


# --------------------------------------------------------------------------- #
# runtime: CUDA toolkit + nsys + NCCL + the project's Python dependencies      #
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Base packages. `devel` (not `runtime`) is required -- nvcc is what lets you
# build flash-attn and nccl-tests in here.
#
# NCCL is deliberately absent from this list. The CUDA images ship libnccl2 and
# libnccl-dev already installed and `apt-mark hold`ed, pinned to the version
# NVIDIA matched to this CUDA release. Asking for them again makes apt try to
# upgrade a held package, which aborts the build with:
#   E: Held packages were changed and -y was used without --allow-change-held-packages
# Respecting the hold is also the right call -- NVIDIA pinned it on purpose --
# so only install NCCL if a future base image turns out not to carry it.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
        build-essential \
        ninja-build \
        numactl \
        libnuma-dev \
        libibverbs1 \
        rdma-core; \
    dpkg -s libnccl-dev >/dev/null 2>&1 \
        || apt-get install -y --no-install-recommends libnccl2 libnccl-dev; \
    echo "### system NCCL: $(dpkg-query -W -f='${Version}' libnccl2)"; \
    rm -rf /var/lib/apt/lists/*

# Nsight Systems, installed in its own layer so a failure here is unambiguous.
# The CUDA repo has no `nsight-systems-cli`; it publishes a CUDA-versioned meta
# package (cuda-nsight-systems-13-0) alongside release-versioned real packages
# (nsight-systems-2025.3.2). Try the meta package, fall back to the newest
# release the repo actually offers, and dump the candidate list if both fail so
# the build log says what went wrong instead of just exiting 100.
ARG CUDA_VERSION
RUN set -eu; \
    apt-get update; \
    meta="cuda-nsight-systems-$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr '.' '-')"; \
    echo "### CUDA_VERSION='${CUDA_VERSION}' -> trying '${meta}'"; \
    if apt-get install -y --no-install-recommends "${meta}"; then \
        echo "### installed ${meta}"; \
    else \
        echo "### '${meta}' failed. nsight packages this repo offers:"; \
        apt-cache search --names-only '^nsight-systems' || true; \
        apt-cache policy "${meta}" || true; \
        rel="$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -n1)"; \
        echo "### falling back to '${rel:-<none found>}'"; \
        [ -n "${rel}" ]; \
        apt-get install -y --no-install-recommends "${rel}"; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# The nsight-systems deb unpacks under /opt/nvidia/nsight-systems/<release>/ and
# does not reliably leave nsys on PATH, so link it and prove it runs.
RUN set -eux; \
    if ! command -v nsys >/dev/null 2>&1; then \
        nsys_bin="$(find /opt/nvidia/nsight-systems -type f -name nsys | sort -r | head -n1)"; \
        test -n "$nsys_bin"; \
        ln -s "$nsys_bin" /usr/local/bin/nsys; \
    fi; \
    nsys --version

COPY --from=uv /uv /uvx /usr/local/bin/

# The venv lives outside /workspace on purpose: bind-mounting your checkout over
# /workspace would otherwise shadow it and the image's dependencies would vanish.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace

# Dependencies resolve from the lockfile alone, so this layer is cached until
# pyproject.toml or uv.lock actually change.
COPY pyproject.toml uv.lock .python-version ./

ARG UV_EXTRAS="--extra quant --extra track"
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked ${UV_EXTRAS}

# Model and dataset caches belong on a volume, not in the image layers --
# Qwen3-30B-A3B alone is ~61 GB of weights.
ENV HF_HOME=/cache/huggingface \
    TRITON_CACHE_DIR=/cache/triton
RUN mkdir -p /cache/huggingface /cache/triton

# Fail the build early if the toolchain is not what we think it is. Builders
# usually have no GPU, so none of this may initialise CUDA or touch a device.
RUN nvcc --version && ldconfig -p | grep -q libnccl.so
RUN python -c "import torch, transformers; print('torch', torch.__version__, '/ cuda', torch.version.cuda); print('transformers', transformers.__version__)"

CMD ["/bin/bash"]


# --------------------------------------------------------------------------- #
# nccl-tests: collective benchmarks (all_reduce_perf and friends)              #
# --------------------------------------------------------------------------- #
# Opt in with --target nccl-tests. Built against the *system* NCCL from
# libnccl-dev; see the note below about torch shipping its own copy.
FROM runtime AS nccl-tests

RUN git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && make -C /opt/nccl-tests -j"$(nproc)" CUDA_HOME=/usr/local/cuda \
    && rm -rf /opt/nccl-tests/.git

ENV PATH=/opt/nccl-tests/build:$PATH
