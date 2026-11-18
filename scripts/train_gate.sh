#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/common.sh"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
    echo 'This fixed training recipe requires four GPUs (global batch 128, 32 per GPU).' >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")"
RESUME_ARGS=()
if [[ "${RESUME:-0}" == 1 ]]; then RESUME_ARGS=(--resume); fi
exec "$PYTHON" -m torch.distributed.run \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --rdzv_id="gate-$$" --nproc_per_node=4 \
    tracking/train_gate.py --checkpoint "$DMATRACK_CHECKPOINT" --dataset "$LASHER_ROOT" \
    --output "${1:-output/train_gate}" --epochs 10 --fusion sum \
    --branch-reduction sum --positive-boundary ge --label-threshold 0.5 \
    --batch-size 128 --micro-batch 32 --lr-drop-after 7 --lr-schedule step \
    --workers "${TRAIN_WORKERS:-8}" "${RESUME_ARGS[@]}"
