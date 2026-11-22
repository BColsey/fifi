#!/usr/bin/env bash
# CPU topology test and one CUDA TU; neither step accesses a GPU.
set -euo pipefail
fixture=$(cd -- "$(dirname -- "$0")" && pwd)
engine=$(cd -- "$fixture/../.." && pwd)
out=${1:?usage: build.sh scratch-output-directory [SM, default 75]}
mkdir -p -- "$out/tmp"
out=$(cd -- "$out" && pwd)
export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror \
    -I "$engine/src/kernels/cuda/native" "$fixture/snapshot_test.cc" -o "$out/snapshot-test"
"$out/snapshot-test"
"${NVCC:-nvcc}" -ccbin "${CXX:-g++}" -arch="sm_${2:-75}" -std=c++17 -O3 -lineinfo \
    --Werror all-warnings -Xptxas=-v -I "$engine/src/kernels/cuda/native" \
    "$fixture/probe.cu" -o "$out/probe"
