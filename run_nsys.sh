#!/usr/bin/env bash
# Nsight Systems capture of one rank for a torchrun job, single- or multi-node.
#
#   ./run_nsys.sh [NPROC] [CONFIG] [extra torchtitan CLI overrides...]
#
# Single node:
#   ./run_nsys.sh 8 configs/qwen3_30b_a3b.toml
#   ./run_nsys.sh 1 configs/qwen3_smoke_1gpu.toml --nsys.start_step 3 --nsys.stop_step 5
#
# Multi-node: run the same command on every node with the launcher env set,
# exactly as you would run torchrun on every node:
#   NNODES=4 NODE_RANK=<0..3> MASTER_ADDR=<node 0 host> ./run_nsys.sh 8 configs/qwen3_30b_a3b.toml
#   srun -N4 --ntasks-per-node=1 bash -c \
#     'NNODES=4 NODE_RANK=$SLURM_NODEID MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1) \
#      ./run_nsys.sh 8 configs/qwen3_30b_a3b.toml'
#
# Env:
#   NSYS_RANK    global rank to capture (default 0). Use this, not --nsys.rank.
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT   torchrun rendezvous (default: 1 node)
#   ENTRY        module run via `torchrun -m` (default scale_transformer.train)
#   OUT_DIR      outputs/nsys
#
# Only the node hosting NSYS_RANK (node NSYS_RANK / NPROC) is launched under
# `nsys profile`; every other node runs plain torchrun, so it pays no injection
# overhead and writes no report. On the profiled node nsys is armed with
# --capture-range=cudaProfilerApi and only NSYS_RANK calls
# cudaProfilerStart/Stop (see scale_transformer/nsys_profile.py), so the report holds that rank's
# [nsys.start_step, nsys.stop_step] window. All ranks still run the training
# loop with the [nsys] config so they exit together after the window.
set -euo pipefail

NPROC="${1:-1}"; shift || true
CONFIG="${1:-configs/qwen3_smoke_1gpu.toml}"; shift || true
ENTRY="${ENTRY:-scale_transformer.train}"
OUT_DIR="${OUT_DIR:-outputs/nsys}"
NSYS_RANK="${NSYS_RANK:-0}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

for arg in "$@"; do
  case "$arg" in
    --nsys.rank|--nsys.rank=*)
      echo "run_nsys.sh: set NSYS_RANK=<global rank> instead of passing --nsys.rank;" \
           "the script needs it to pick the node to wrap with nsys" >&2
      exit 2 ;;
  esac
done

WORLD_SIZE=$(( NNODES * NPROC ))
if (( NSYS_RANK < 0 || NSYS_RANK >= WORLD_SIZE )); then
  echo "run_nsys.sh: NSYS_RANK=$NSYS_RANK is outside the world of $WORLD_SIZE ranks" \
       "($NNODES node(s) x $NPROC proc)" >&2
  exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "run_nsys.sh: NODE_RANK=$NODE_RANK is outside 0..$(( NNODES - 1 ))" >&2
  exit 2
fi
PROFILED_NODE=$(( NSYS_RANK / NPROC ))

TORCHRUN=(torchrun --nproc_per_node="$NPROC")
if (( NNODES > 1 )); then
  TORCHRUN+=(--nnodes="$NNODES" --node_rank="$NODE_RANK"
             --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT")
fi
TORCHRUN+=(-m "$ENTRY"
  --job.config_file "$CONFIG"
  --job.custom_config_module scale_transformer.nsys_profile
  --nsys.enable
  --nsys.rank "$NSYS_RANK"
  --nsys.stop_training_after_capture
  "$@")

if (( NODE_RANK != PROFILED_NODE )); then
  echo "run_nsys.sh: node $NODE_RANK does not host rank $NSYS_RANK (node $PROFILED_NODE does);" \
       "launching without nsys" >&2
  exec "${TORCHRUN[@]}"
fi

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/rank${NSYS_RANK}_$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
echo "run_nsys.sh: node $NODE_RANK hosts rank $NSYS_RANK; nsys report -> $REPORT.nsys-rep" >&2

exec nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output="$REPORT" \
  "${TORCHRUN[@]}"
