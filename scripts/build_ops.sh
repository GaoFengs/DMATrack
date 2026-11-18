#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/.."
if [[ -f local.env ]]; then
    set -a
    source local.env
    set +a
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.1}"
export CC="${CC:-gcc-9}"
export CXX="${CXX:-g++-9}"
export MAX_JOBS="${MAX_JOBS:-2}"
read -r -a BUILD_GPU_IDS <<< "${GPUS:-0}"
export CUDA_VISIBLE_DEVICES="${BUILD_GPU_IDS[0]}"
"${PYTHON:-python}" tracking/build_ops.py
