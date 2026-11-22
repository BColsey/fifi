# Exact F32 Inkling Q8_0 projection reuse (SM75)

This is a component experiment, not an engine opt-in. Normal builds do not
compile the candidate, and the fixture also defaults to ordinary GEMV. Only
`probe WEIGHTS_DIR --compare` requests the candidate. Build and host tests do
not initialize a GPU.

## Why these shapes

The same-file [Unsloth reference release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)
used batched MMQ; FiFi spent 2.216 s in packed GEMV in that 512-token trace
comparison (`standalone/inkling-llama-reference/CONTINUATION.md`). The FiFi
source is `731f26d8e9f566e7a871f57df5b1098aa476cd5c`; see
`standalone/inkling-100-1000-baseline/evidence/packed-gemv.json` and
`fde96bf55318/inkling-iq2-gate-dot/evidence/profile/1/inkling.sqlite`, under
`$HOME/fifi-scratch`. Grouping the Q8_0 `SharedWeights` kernels by `gridX` and
`dynamicSharedMemory` gives the following. Each baseline CTA processes eight
weight rows for one activation row; dynamic shared bytes divide by four for K.

| Activation rows M | Weight rows N | Width K | Calls | Summed GPU ms | Original GGUF example |
|---:|---:|---:|---:|---:|---|
| 63–64 | 2304 | 3072 | 336 | 466.626 | `blk.0.attn_output.weight` |
| 7–12 | 3072 | 2304 | 1,333 | 339.426 | `blk.1.attn_gate.weight` |
| 7–12 | 1024 | 2304 | 3,354 | 244.340 | `blk.0.attn_k.weight` |

These are geometry buckets (1.050 s total), not a per-tensor attribution or
wall-time savings estimate. A smaller `(N=384,K=2304)` Q8_0 bin (40.410 ms) is
outside this prototype. The CUDA-only IQ2_XXS/Q2_0 exact expert tile (`da1bdcc6`,
`9567f77e`) excluded Q8_0 and these larger widths. The prior
`standalone/inkling-prefill-q8-wmma/CONTINUATION.md` records an extractor only,
with **no** candidate compilation or GPU result. `d2015cbc` used quantized
activations instead; `41c66e39`'s measured IQ2_XXS Q8_1-activation MMQ gain
failed strict model logits. Neither is this experiment or an arithmetic oracle.

[NVIDIA's CUDA Best Practices Guide, §9.2.3.2](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)
identifies redundant reads as a reason to tile matrix multiply even when the
reads already coalesce. Here each CTA packs two F32 activation rows into
adjacent `float2` shared-memory elements, and each lane decodes the original
Q8_0 block value once for two independent F32 FMAs. Each lane still processes
`32*k+lane` in ascending `k`, with the same `decode<8>`, zero initializer and
XOR butterfly as ordinary packed GEMV. No split K, half operands, rounded
activations, weight preprocessing or workspace. The last odd row stages the
same valid input twice and stores only the first result. Both arms enqueue
one kernel. The paired arm schedules `ceil(M/2)` rather than `M` CTAs per
output-row tile; it does not halve activation bytes or FMA work for even M.

The resource tradeoff matters on [Turing](https://docs.nvidia.com/cuda/archive/12.8.0/turing-tuning-guide/index.html#unified-shared-memory-l1-texture-cache):
18/24 KiB shared per paired CTA versus 9/12 KiB baseline, plus six extra
registers/thread in the local CUDA 12.4 build (42 vs 36, no spills). The
paired SASS uses `LDS.U.64`/`STS.64`; this says nothing about observed cache
traffic or latency. No speedup is claimed. The candidate is limited to the
three geometries above, Q8_0, SM75 and M=2–64. Other valid calls fall back to
ordinary GEMV, and invalid operands fail before enqueue. Selected launch
errors are returned, never retried via the baseline. No production Rust or
model-loading path changed.

## Build and validate without a GPU

From the scratch worktree root:

```bash
set -euo pipefail
fixture="$PWD/engine/tools/inkling-q8-row-reuse"
out=$(realpath -m ../tmp/q8-build)
model="$HOME/fifi-scratch/ee8307597ab8/inkling-2-mini-20261120/quants/UD-IQ1_M/Inkling-2-Mini-UD-IQ1_M.gguf"
mkdir -p "$out"
NVCC=/usr/bin/nvcc CXX=/usr/bin/g++-13 bash "$fixture/build.sh" "$out" 2>&1 | tee "$out/build.log"
"$out/probe" --host-only   # 12 dispatch cases and 10 ABI refusals
python3 -B "$fixture/extract.py" "$model" "$out/weights"  # weights must be a new directory
(cd "$out/weights" && jq -r '.weights[] | "\(.sha256)  \(.file)"' manifest.json | sha256sum -c)
```

The extractor checks the kind, dimensions, offsets, and file bounds, then
copies just the three payloads. The 8,024,441,024-byte checkpoint has offsets
510361536, 759408576 and 512893888 in the order `attn_k`, `attn_gate`,
`attn_output`. The corresponding SHA-256 values are:

```text
f9da9936b7befd5783f7d6f187a5d2decbcfe441b073e7b18067e55a140b090a  q8-2304-1024.bin
2d5b4b6487087a98a17b33a252b7c3ee671bf0b288dd13e2d63c013c84e47519  q8-2304-3072.bin
4c7d41855851c18f9f63eff005f0dd567646bb142492642b4cda0e18ed219b1b  q8-3072-2304.bin
```

## Device qualification (lease owner only)

Run only after coordination and admission on the RTX 2080 Ti. The fixture's
peak device buffers are under 17 MiB, excluding context/runtime overhead.
Use at least 1024 MiB free and a new results directory. This command checks
service state; it does not stop or mask the service.

```bash
export FIFI_SCRATCH_ROOT="$HOME/fifi-qwen36"
export CUDA_VISIBLE_DEVICES=GPU-699004f8-e2f0-747e-5c02-30141e9d4b7c
export TMPDIR="$out/tmp" TMP="$out/tmp" TEMP="$out/tmp"
mkdir "$out/results"
"$FIFI_SCRATCH_ROOT/gpu-run-local" bash -euo pipefail -c '
  out=$1
  test "/proc/$$/fd/9" -ef "$FIFI_SCRATCH_ROOT/gpu.lock"
  test "$(systemctl --user show fifi-qwen36-2080ti.service -p UnitFileState --value)" = masked
  test "$(systemctl --user show fifi-qwen36-2080ti.service -p ActiveState --value)" = inactive
  test "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 1024
  timeout 60 "$out/probe" "$out/weights" > "$out/results/default.jsonl"
  timeout 60 "$out/probe" "$out/weights" --compare 0 123 > "$out/results/verify.jsonl"
  timeout 180 compute-sanitizer --tool memcheck --error-exitcode 1 \
    "$out/probe" "$out/weights" --compare 0 321 > "$out/results/memcheck.log" 2>&1
  for seed in 123 321 573; do
    timeout 120 "$out/probe" "$out/weights" --compare 100 "$seed" > "$out/results/timing-$seed.jsonl"
  done
  test -z "$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)"
' _ "$out"
jq -es 'length == 33 and all(.[]; .kind == "bits" and .pair_launches == 0 and .baseline_launches == 1)' \
  "$out/results/default.jsonl"
jq -es 'length == 33 and (map(.bitwise)|add) == 1261803 and (map(.pair_launches)|add) == 24 and (map(.baseline_launches)|add) == 9' \
  "$out/results/verify.jsonl"
grep -F 'ERROR SUMMARY: 0 errors' "$out/results/memcheck.log"
for file in "$out"/results/timing-*.jsonl; do
  jq -es '[.[] | select(.kind == "timing")] | length == 6 and all(.[]; .baseline_launches == 400 and .pair_launches == 400 and (.a_us|length) == 4 and (.b_us|length) == 4)' "$file"
done
```

Both arms use the same packed bytes, F32 data, strides and stream. Bitwise
checks include finiteness, output padding and guards, unchanged inputs,
2-byte-aligned packed weights, odd tails, M=2 and valid fallbacks (singletons,
M>64, unsupported output extent). Inputs are deterministic synthetic F32,
including signed zero and subnormal operands, **not recorded prompt hidden
states**. Compare medians of the four warm ABBA/BAAB samples per arm, shape
and seed (in microseconds per call). Uploads, verification and setup are outside the
CUDA event intervals. The fixture reuses one tensor across repetitions; the
engine contends for cache with other tensors and does other work between
calls. Treat bitwise or sanitizer failure as a stop; otherwise require
consistent shape-level gains before testing an engine integration. Serving
speed, strict logits, prefix/continuation and speculative parity are not
qualified by this component.
