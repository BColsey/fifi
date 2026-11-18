# Device-aware MLX prefill

Qwen3.6, Flash-Next and Flint can size their prefill work to the Apple Silicon GPU from the same startup policy. Metal supplies `recommendedMaxWorkingSetSize` in bytes. Each executor accounts for its resident weights and session memory, then uses the remaining budget and its configured row ceiling to select a chunk. All three use the same largest-fitting-chunk rule.

```sh
FIFI_MLX_AUTO_BUDGET=1 enginesrv --config /path/to/config.json
```

Use an Apple Silicon build with the `native-mlx` feature. The option is off by default. When enabled, it applies to one MLX model per server because the device reports one shared working set. Metal's recommendation is a sizing input for the total working set. Other apps, other server processes and allocation peaks during weight import can put additional pressure on memory. The policy checks resident memory and serving reserves before accepting jobs. Flint also uses a header-based estimate to choose where to keep its weights before import.

The 16 GB M4 mini reports 12,713,115,648 bytes (11.84 GiB). The implementation queries Metal on every opted-in model startup and uses the resulting byte count. Context length and KV format come from the family's configuration.

| Family | Sizing inputs in addition to resident weights and allocator headroom | Selection |
| --- | --- | --- |
| Qwen3.6 | Full-context KV capacity at the selected dtype, DeltaNet state, a layer's cache growth overlap and the aggregate session cap when prefix reuse or concurrency applies | Uses the existing chunk cost model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. Concurrent serving requires an explicit byte budget and a fitting chunk. |
| Flash-Next | Decode pool, full-context QSA including index keys, DeltaNet and PLE state, overlap while replacing session arrays, and configured prefix slots | Uses the existing expert-wave workspace model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. |
| Flint | Full KV planes, 513-position sliding rings, one layer's KV growth overlap, bounded prefix retention and a conservative upper estimate of the full-attention score, mask and softmax workspace | Chooses among 1,024, 512 and one row within the ceiling set by `FIFI_FLINT_MLX_BATCH_PREFILL`. The prefix cache records the selected width. |

Qwen3.6 and Flash-Next can fall back to one-token prefill if the workspace for a chunk does not fit. Startup logs carry the Metal recommendation, the charged resident and reserve bytes, and the selected path. [MLX budget code](../../engine/src/family/mlx_budget.rs) · [Environment reference](../reference/environment.md#apple-silicon-device-sizing)

## Disk-backed Flint experts

With automatic sizing enabled, Flint checks the checkpoint headers and serving reserves before importing weights. If the full resident model fits, Flint loads it as before. Otherwise it keeps attention, routers, shared experts, and the output head in memory and reads the routed experts by ID from the existing local safetensors files. Each layer imports only the experts selected for the current rows, restores the original routing order, and releases its staged weights after Metal completes the grouped projection. The model files stay on the local drive and the KV cache stays in unified memory.

The planner charges both the host staging buffers and their Metal copies. Flint reduces prefill rows when that staging space and attention workspace do not fit together. It refuses startup before weight import when the resident core and one token of expert staging exceed the budget. The startup log reports `weight placement=disk-experts` or `weight placement=resident` and the selected prefill width. With Flint prefix reuse explicitly enabled, an oversized checkpoint also needs an explicit `FIFI_FLINT_PREFIX_BUDGET_BYTES` because the legacy cap derivation charges the entire checkpoint.

Flash-Next already reads routed experts from disk and stages them in bounded page sets. Qwen3.6 still loads its expert banks into unified memory. Qwen dense models would need layer streaming to serve checkpoints whose resident core exceeds the budget.

On the 16 GB M4 mini, a sparse zero-weight fixture with Flint's full tensor layout selected disk-backed experts under a 2 GiB weight budget. Its 22.24 GB resident-weight estimate includes all expert banks. The disk-backed path estimated 952 MB of resident weights and peaked at 1.04 GiB of MLX allocations during two prefill rows and one decode step. A separate nonzero fixture compared the disk-backed and resident expert outputs bit for bit for one and two rows, with repeated and nonconsecutive expert IDs. The published checkpoint's latency, memory peak, and output agreement are still open.

Run these two tests from `engine/` on Apple Silicon:

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib -- \
  disk_experts_match_resident_grouped_predictions \
  sparse_oversized_flint_checkpoint_streams_and_decodes_on_mlx --nocapture --test-threads=1
```

## CUDA and MLX interfaces

| Responsibility | CUDA | MLX |
| --- | --- | --- |
| Device budget | Device memory and arena admission use the shared byte-reserve and fitting-candidate arithmetic. [CUDA admission](../../engine/src/family/prefill/admission.rs) | Metal supplies one unified-memory working set. The same byte-reserve arithmetic checks family charges and the MLX chunk chooser sizes prefill. [Shared reserve](../../engine/src/family/device_budget.rs) · [MLX planner](../../engine/src/family/prefill/mlx.rs) |
| KV format | KV row math accounts for plain and packed cache planes and their auxiliary values. | Qwen3.6 prices F32, BF16 and Q8 with the same KV row math. Flint uses BF16 KV and separate capacities for full and sliding layers. Flash-Next adds a separate F32 index row to its F32 KV cost. [KV formats](../../engine/src/family/kv_format.rs) |
| Expert staging | The expert bank tracks reusable slots and staged copies. [Expert bank](../../engine/src/family/expert_bank/mod.rs) | Flash-Next lends expert page sets through their synchronization point, Qwen3.6 holds resident packed experts, and Flint can stage selected experts from disk. The adapters share MLX's packed matmul. [Flash-Next slots](../../engine/src/family/flash_next/resident/slots.rs) |
| Kernel selection | The serving profile registers CUDA kernels for the engine. [CUDA serving kernels](../../engine/src/kernels/cuda/resident/exl3.rs) | Shared attention checks tensor dtype, shape and Metal threadgroup capacity. The families supply Qwen3.6's decode cutoffs, Flash-Next's sparse QSA and Flint's sink and ring rules. [MLX kernels](../../engine/src/kernels/mlx/mod.rs) |

## Verification on the mini

The Flash-Next factory loaded a synthetic checkpoint and selected four rows using the actual Metal recommendation. With a one-byte ceiling it took one-token steps. Both paths generated identical continuation tokens. Qwen3.6's resident adapter selected one, four and eight rows at injected budget boundaries. Flint's planner exercised the thresholds for one, 512 and 1,024 rows and checked the operator ceiling. The release server also refused a two-model config before tokenizer load with auto sizing enabled. The selected checks passed 166 tests with MLX and 60 on the host-only build. An explicit Flint attention check passed with both decode routes. Representative checkpoints, full-context memory peaks and performance for automatic sizing are still open.

Five fresh processes collected the same counters for Flint's grouped expert, attention-sink and rolling-cache fixtures. The cache fixture covered seven steps across the 513-position ring boundary and compared the two routes byte for byte. The expert and attention-sink fixtures checked against their row and scalar references.

| Fixture | Metal waits | Peak allocation above the fixture's starting point |
| --- | ---: | ---: |
| Grouped FFN, 2 to 17 rows | 1 | 58 to 285 KiB |
| Sink attention, 512 to 514 positions | 1 | 8.44 to 8.48 MiB |
| Fenced ring wrap | 113 | 15.87 MiB |
| Chained ring wrap | 64 | 225.78 MiB |

Chaining removed 49 waits and added about 210 MiB of peak allocation in this fixture. That makes graph lifetime and synchronization the first Flint optimization to investigate at serving shapes. Both simpler fixtures completed in one readback wait, so use larger shapes to establish the cost of rank sorting and sink attention before extracting family helpers. Compare the wall samples within the same case because the first invocation can include kernel setup. [All 40 fixture samples](../../scripts/benchmarks/mac-m4-flint-fixtures.csv)

Run from `engine/` on Apple Silicon. The first command runs one Flash-Next startup test. The second runs three Flint tests and emits eight counter rows.

```sh
FIFI_MLX_AUTO_BUDGET=1 cargo test --locked --release \
  --features native-mlx,native-fault-qualification --lib \
  family::flash_next::resident::executor::tests::flash_next_mlx_serving_prefill_runs_the_admitted_path \
  -- --exact --nocapture --test-threads=1

cargo test --locked --release --features native-mlx,native-fault-qualification --lib -- \
  batch_grouped_experts_match_row_reference_without_half_overflow \
  native_sinks_gqa_and_window_edges_match_scalar \
  chained_commit_matches_fenced_bytes_across_the_swa_wrap --nocapture --test-threads=1
```
