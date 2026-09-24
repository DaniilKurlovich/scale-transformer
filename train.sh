#!/usr/bin/env bash
# One node of the training job, for the container: fetch what the config needs
# into $WORKSPACE, then launch torchrun with checkpoints, tensorboard, profiler
# traces, logs, the HF cache and the dataset all under $WORKSPACE.
#
#   train.sh [CONFIG] [extra torchtitan CLI overrides...]
#
#   train.sh                                             # configs/qwen3_30b_a3b.toml
#   train.sh configs/qwen3_30b_a3b_yarn.toml --training.steps 200
#   NSYS=1 train.sh configs/qwen3_30b_a3b.toml           # nsys capture run (ends after the window)
#
# Two nodes x 8 H100: run the same command on both with NNODES=2 and
# NODE_RANK=0 / NODE_RANK=1, MASTER_ADDR set to node 0's reachable address on
# both. torchrun rendezvous on MASTER_ADDR:MASTER_PORT and the job starts once
# both nodes are up.
#
# Env:
#   WORKSPACE        assets/, data/, outputs/ and the HF cache live here (default /workspace)
#   NNODES           number of nodes (default 1)
#   NODE_RANK        this node's rank 0..NNODES-1 (required when NNODES > 1)
#   MASTER_ADDR      node-0 address (required when NNODES > 1); MASTER_PORT default 29500
#   NPROC_PER_NODE   GPUs per node (default: what nvidia-smi lists, else 8)
#   NSYS             1 -> wrap the job in run_nsys.sh instead of plain torchrun
#   NSYS_RANK        rank to capture when NSYS=1 (default 0)
#   PROJECT_DIR      where configs/ and run_nsys.sh come from. Default: $WORKSPACE if a
#                    checkout is synced there, else the copy baked into the image.
#
# A checkout synced over $WORKSPACE (mutagen, bind mount) takes precedence over
# the image's copy: its src/ goes on PYTHONPATH and its configs/ are used.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
IMAGE_PROJECT_DIR=/opt/scale-transformer
if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -d "$WORKSPACE/src/scale_transformer" ]]; then
    PROJECT_DIR="$WORKSPACE"
  else
    PROJECT_DIR="$IMAGE_PROJECT_DIR"
  fi
fi
if [[ "$PROJECT_DIR" != "$IMAGE_PROJECT_DIR" && -d "$PROJECT_DIR/src/scale_transformer" ]]; then
  export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
fi

CONFIG="${1:-configs/qwen3_30b_a3b.toml}"; shift || true
[[ "$CONFIG" = /* ]] || CONFIG="$PROJECT_DIR/$CONFIG"
if [[ ! -f "$CONFIG" ]]; then
  echo "train.sh: config not found: $CONFIG" >&2
  exit 2
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?train.sh: MASTER_ADDR (node 0 address) is required when NNODES > 1}"
  if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "train.sh: NODE_RANK=$NODE_RANK is outside 0..$(( NNODES - 1 ))" >&2
    exit 2
  fi
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')" && (( n > 0 )); then
    NPROC_PER_NODE="$n"
  else
    NPROC_PER_NODE=8
  fi
fi

# Everything the run writes or reads is relative to $WORKSPACE: the configs use
# ./assets, ./data and ./outputs, and the Dockerfile points HF_HOME here too.
mkdir -p "$WORKSPACE/outputs/logs"
cd "$WORKSPACE"
LOG="$WORKSPACE/outputs/logs/node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "train.sh: config=$CONFIG node=$NODE_RANK/$NNODES nproc=$NPROC_PER_NODE" \
     "master=$MASTER_ADDR:$MASTER_PORT project=$PROJECT_DIR workspace=$WORKSPACE log=$LOG"

# Serialised on a lock file: with a network volume shared by both nodes, only
# one of them should be writing the same download at a time. The fetch is a
# no-op (no network) once the directories carry their .fetched markers.
flock "$WORKSPACE/.fetch.lock" python -m scale_transformer.fetch_assets --config "$CONFIG"

if [[ "${NSYS:-0}" == 1 ]]; then
  export NNODES NODE_RANK MASTER_ADDR MASTER_PORT
  export OUT_DIR="${OUT_DIR:-$WORKSPACE/outputs/nsys}"
  exec "$PROJECT_DIR/run_nsys.sh" "$NPROC_PER_NODE" "$CONFIG" "$@"
fi

TORCHRUN=(torchrun --nproc_per_node="$NPROC_PER_NODE")
if (( NNODES > 1 )); then
  TORCHRUN+=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
             --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT")
fi
exec "${TORCHRUN[@]}" -m scale_transformer.train --job.config_file "$CONFIG" "$@"
