#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
exec "$PYTHON" tracking/test.py \
    --checkpoint "$DMATRACK_CHECKPOINT" --output "${1:-output/test_gate}" \
    --gpus "${GPU_IDS[@]}" --workers-per-gpu "${WORKERS_PER_GPU:-3}" \
    --strategy hard_switch --tau-rgb 0.99 --tau-tir 0.95 --score 0.995 --iou 0.4
