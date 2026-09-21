#!/usr/bin/env bash
# End-to-end pipeline check on a 0.6B model. Downloads ~1.5 GB on first run.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run scale-train configs/qwen3_0.6b_smoke.yaml "$@"
