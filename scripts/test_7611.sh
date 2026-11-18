#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
exec "$PYTHON" tracking/test.py \
    --checkpoint "$DMATRACK_CHECKPOINT" --output "${1:-output/test_7611}" \
    --gpus "${GPU_IDS[@]}" --workers-per-gpu "${WORKERS_PER_GPU:-3}" \
    --strategy always_rgbt --score 0.3 --iou 0.2 --verify-weight
