#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
exec "$PYTHON" tracking/test.py \
    --checkpoint "$DMATRACK_CHECKPOINT" --output "${1:-output/smoke_test}" \
    --gpus "${GPU_IDS[0]}" --workers-per-gpu 1 \
    --strategy always_rgbt --score 0.3 --iou 0.2 --verify-weight \
    --sequences 2girlup "leftgirl'swhitebag" rightcar-chongT
