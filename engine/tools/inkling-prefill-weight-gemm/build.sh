#!/usr/bin/env bash
set -eu
fixture=$(cd "$(dirname "$0")" && pwd)
engine=$(cd "$fixture/../.." && pwd)
native="$engine/src/kernels/cuda/native"
"${NVCC:-nvcc}" -ccbin "${CXX:-g++}" -DFIFI_VISIT_PROBE -arch=sm_75 -std=c++17 -O3 -lineinfo --Werror all-warnings \
    -I "$native" -I "$engine/vendor/ggml-cuda-mmq/src" "$native/resident/dense_gemv.cu" "$fixture/probe.cu" -o "${1:?output probe path}"
