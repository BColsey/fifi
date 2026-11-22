#!/usr/bin/env bash
# Compile just the native GEMV and component probe translation units (no GPU).
set -euo pipefail
fixture=$(cd -- "$(dirname -- "$0")" && pwd)
engine=$(cd -- "$fixture/../.." && pwd)
out=${1:?usage: build.sh scratch-output-directory}
mkdir -p -- "$out/tmp"
out=$(cd -- "$out" && pwd)
export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
native="$engine/src/kernels/cuda/native"
"${NVCC:-nvcc}" -ccbin "${CXX:-g++}" -arch=sm_75 -std=c++17 -O3 -lineinfo \
    --Werror all-warnings -Xptxas=-v \
    -I "$native" -I "$engine/vendor/ggml-cuda-mmq/src" \
    "$native/resident/dense_gemv.cu" "$fixture/probe.cu" -o "$out/probe"
