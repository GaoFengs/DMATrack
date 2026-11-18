#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -f local.env ]]; then
    set -a
    source local.env
    set +a
fi
PYTHON="${PYTHON:-python}"
LASHER_ROOT="${LASHER_ROOT:-$PROJECT_ROOT/data/LasHeR}"
DMATRACK_CHECKPOINT="${DMATRACK_CHECKPOINT:-$PROJECT_ROOT/checkpoints/DMATrack_ep0010.pth.tar}"
export LASHER_ROOT DMATRACK_CHECKPOINT
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
read -r -a GPU_IDS <<< "${GPUS:-0}"
if [[ ! -f "$DMATRACK_CHECKPOINT" ]]; then
    echo "Checkpoint not found: $DMATRACK_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$LASHER_ROOT/testingset" ]]; then
    echo "Set LASHER_ROOT to the LasHeR dataset root" >&2
    exit 1
fi
if [[ ! -f .build/prroi/_prroi_pooling.so ]]; then
    echo "Build the native operator first: bash scripts/build_ops.sh" >&2
    exit 1
fi
