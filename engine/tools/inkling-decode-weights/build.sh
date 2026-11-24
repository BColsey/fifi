#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
"${NVCC:-nvcc}" -ccbin "${CXX:-g++}" -arch=sm_75 -std=c++17 -O3 --Werror all-warnings \
    -I src/kernels/cuda/native -I vendor/ggml-cuda-mmq/src tools/inkling-decode-weights/probe.cu -o "${1:?output path}"
