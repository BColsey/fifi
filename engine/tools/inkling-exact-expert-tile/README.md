# Exact Inkling expert projection tile (component only)

The same-file reference trace attributed about **2.216 s** of FiFi's 512-token prefill to packed GEMV. The subsequent IQ2_XXS grouped MMQ trial quantized activations to Q8_1 and failed strict full-vocabulary logits (max absolute delta 3.5060 vs 0.00002384). Neither the MMQ kernel nor the existing half-operand WMMA path is the arithmetic reference here. At the base tip, `project_resident_tile` already groups consecutive slots by expert, and `project_gguf_batches` submits ordinary dense GEMV per group.

The component entry `inference_cuda_gguf_exact_expert_tile_launch` accepts one expert's packed weights and its already-grouped F32 inputs. Each CTA stages 2 or 4 F32 activation rows (at most **18 KiB**) and each warp decodes its packed weight row once across the tile. Each output keeps the baseline layout, lane-to-column mapping, in-lane F32 FMA dependency chain and five XOR-butterfly adds. Tail slots read only valid rows and do not write past `m`. There are no padded weight copies, global workspaces, input conversions, host-side scales, or expert/rank reordering. The tile supports IQ2_XXS and Q2_0 only (`m` 2–512, output/input widths up to 2304). Singletons, other valid shapes and other supported types go through ordinary GEMV. Invalid calls and launch failures return errors rather than attempting a retry. No Rust wrapper, model dispatch, environment opt-in, or speculative routing is wired; the standalone fixture calls the new entry explicitly.

Primary sources: [Unsloth llama.cpp PR #264](https://github.com/unslothai/llama.cpp/pull/264) uses compact per-expert tiling to address sparse token counts; this prototype borrows neither its MMQ arithmetic nor its permutation path. [NVIDIA CUTLASS, Efficient GEMM, Thread-level GEMM](https://github.com/NVIDIA/cutlass/blob/e94e888df3551224738bfa505787b515eae8352f/media/docs/cpp/efficient_gemm.md#thread-level-gemm) describes the independent register accumulators that permit SIMT operand reuse. [NVIDIA Floating Point and IEEE 754, FMA](https://docs.nvidia.com/cuda/archive/12.8.0/floating-point/index.html#the-fused-multiply-add-fma) explains why separate multiply/add or changing the reduction tree is not the same floating-point operation. See NVIDIA's [warp shuffle rules](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#warp-shuffle-functions) and [Turing shared-memory limits](https://docs.nvidia.com/cuda/archive/12.8.0/turing-tuning-guide/index.html#memory-throughput). Less weight decoding does not guarantee a speedup: FMA and activation loads remain, and occupancy or cache pressure can change.

## Build and verify (host only)

From the worktree root, in Bash. This builds two translation units, not the engine. Do not use `-use_fast_math`; compare the two paths with the same compiler flags.

```bash
set -euo pipefail
fixture="$PWD/engine/tools/inkling-exact-expert-tile"
out=$(realpath -m ../tmp/exact-tile)
mkdir -p "$out"
NVCC=/usr/bin/nvcc CXX=/usr/bin/g++-13 bash "$fixture/build.sh" "$out" 75 \
  2>&1 | tee "$out/build.log"
"$out/probe" --host-only  # expects 11 ABI refusals; does not initialize a CUDA device
model="$HOME/fifi-scratch/ee8307597ab8/inkling-2-mini-20261120/quants/UD-IQ1_M/Inkling-2-Mini-UD-IQ1_M.gguf"
read -r gate down < <(python3 -B "$fixture/tensor-offsets.py" "$model")
```

The parser reads only the GGUF header, checks the two tensor descriptors and their spans, and returns offsets (for the 8,024,441,024-byte checkpoint: `1905048768 1808317120`). The probe reads the original Gate/Up IQ2_XXS and Down Q2_0 stacks for `blk.5`, all 128 experts. The two raw tensor SHA-256 values at these offsets are `b918e7dbf37bc72307bd1b643a2b82370eed0d31e5d324595e2c561325af9553` and `59170a5705f8eb3e2a2c6dcedd7008d881c50c593ddf5db682ccb58eb04a20` respectively. No repacking or whole-model loading.

## Device qualification (GPU lease owner only)

Use the canonical runner after the lease owner has confirmed the SM75 service is inactive/masked. Peak fixture buffers are about 276 MiB on device before CUDA context/runtime overhead; admit with at least 1024 MiB free. Commands below do not stop services. Use a fresh output directory per wave.

```bash
export FIFI_SCRATCH_ROOT="$HOME/fifi-qwen36"
export CUDA_VISIBLE_DEVICES=GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c  # RTX 2080 Ti
export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
mkdir "$out/results"
"$FIFI_SCRATCH_ROOT/gpu-run-local" bash -euo pipefail -c '
  out=$1; model=$2; gate=$3; down=$4
  test "/proc/$$/fd/9" -ef "$FIFI_SCRATCH_ROOT/gpu.lock"
  test "$(systemctl --user show fifi-qwen36-2080ti.service -p UnitFileState --value)" = masked
  test "$(systemctl --user show fifi-qwen36-2080ti.service -p ActiveState --value)" = inactive
  test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 1024
  timeout 120 "$out/probe" "$model" "$gate" "$down" 512 123 0 0 > "$out/results/verify.jsonl"
  timeout 180 compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$out/probe" "$model" "$gate" "$down" 32 321 1 0 > "$out/results/memcheck.log" 2>&1
  for skew in 0 1; do
    for seed in 123 321 573; do
      timeout 120 "$out/probe" "$model" "$gate" "$down" 512 "$seed" "$skew" 10
    done
  done > "$out/results/timing.jsonl"
  test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)"
' _ "$out" "$model" "$gate" "$down"
```

Expect two `verify.jsonl` records (one per codec, each `bitwise: 9437184`) and no sanitizer errors. The probe checks exact bits and finite written outputs, validates tiled vs fallback status, and checks stride padding, a leading output guard, and unchanged inputs on padded tail cases. It includes singletons; activation counts 2, 3, 4, 5 and 7; partial output warps/CTAs; and valid fallbacks for oversize count/width and unsupported codec. The sanitizer run also exercises skew with many small/empty expert groups. `timing.jsonl` should contain 12 records, four samples per arm each (ABBA then BAAB, milliseconds per full expert group schedule). Both arms use the same FP32 inputs, order, and number of launches. The event interval excludes transfers, routing/gather, activation/GLU, scales, and output verification. Timing is component-only; it is not prefill throughput or a full-model parity result. Require the bitwise fixtures before interpreting timings; later production routing would still need strict logits, prefix/continuation and speculative fallback checks on its exact tip.
