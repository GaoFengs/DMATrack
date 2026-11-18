#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
export CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"
exec "$PYTHON" tracking/train_gate.py --checkpoint "$DMATRACK_CHECKPOINT" \
    --dataset "$LASHER_ROOT" --output "${1:-output/smoke_train}" \
    --fusion sum --branch-reduction sum --positive-boundary ge --label-threshold 0.5 \
    --smoke --smoke-batch-size 2
