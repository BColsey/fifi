# Device-aware MLX prefill

Qwen3.6, Flash-Next and Flint can size their prefill work to the Apple Silicon GPU from the same startup policy. Metal supplies `recommendedMaxWorkingSetSize` in bytes. Each executor accounts for its resident weights and session memory, then uses the remaining budget and its configured row ceiling to select a chunk. All three use the same largest-fitting-chunk rule.

```sh
FIFI_MLX_AUTO_BUDGET=1 enginesrv --config /path/to/config.json
```

Use an Apple Silicon build with the `native-mlx` feature. The option is off by default. When enabled, it applies to one MLX model per server because the device reports one shared working set. Metal's recommendation is a sizing input for the total working set. Other apps, other server processes and allocation peaks during weight import can put additional pressure on memory. The policy validates the resident model and serving reserve after import, before the worker accepts jobs. An over-budget resident model and reserve fail startup with the charged terms.

The 16 GB M4 mini reports 12,713,115,648 bytes (11.84 GiB). The implementation queries Metal on every opted-in model startup and uses the resulting byte count. `FIFI_MLX_WORKING_SET_MIB` also activates automatic sizing and caps the plan at the lower of the configured ceiling and Metal's recommendation. Context length and KV format come from the family's configuration.

| Family | Sizing inputs in addition to resident weights and allocator headroom | Selection |
| --- | --- | --- |
| Qwen3.6 | Full-context KV capacity at the selected dtype, DeltaNet state, a layer's cache growth overlap and the aggregate session cap when prefix reuse or concurrency applies | Uses the existing chunk cost model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. Concurrent serving requires an explicit byte budget and a fitting chunk. |
| Flash-Next | Decode pool, full-context QSA including index keys, DeltaNet and PLE state, overlap while replacing session arrays, and configured prefix slots | Uses the existing expert-wave workspace model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. |
| Flint | Full KV planes, 513-position sliding rings, one layer's KV growth overlap, bounded prefix retention and a conservative upper estimate of the full-attention score, mask and softmax workspace | Chooses among 1,024, 512 and one row within the ceiling set by `FIFI_FLINT_MLX_BATCH_PREFILL`. The prefix cache records the selected width. |

Qwen3.6 and Flash-Next can fall back to one-token prefill if the workspace for a chunk does not fit. Startup logs carry the Metal recommendation, the charged resident and reserve bytes, and the selected path; the logs distinguish the Metal recommendation from the effective budget and, when set, the configured ceiling. [MLX budget code](../../engine/src/family/mlx_budget.rs) · [Environment reference](../reference/environment.md#apple-silicon-device-sizing)

## CUDA and MLX interfaces

| Responsibility | CUDA | MLX |
| --- | --- | --- |
| Device budget | Device memory and arena admission use the shared byte-reserve and fitting-candidate arithmetic. [CUDA admission](../../engine/src/family/prefill/admission.rs) | Metal supplies one unified-memory working set. The same byte-reserve arithmetic checks family charges and the MLX chunk chooser sizes prefill. [Shared reserve](../../engine/src/family/device_budget.rs) · [MLX planner](../../engine/src/family/prefill/mlx.rs) |
| KV format | KV row math accounts for plain and packed cache planes and their auxiliary values. | Qwen3.6 prices F32, BF16 and Q8 with the same KV row math. Flint uses BF16 KV and separate capacities for full and sliding layers. Flash-Next adds a separate F32 index row to its F32 KV cost. [KV formats](../../engine/src/family/kv_format.rs) |
| Expert staging | The expert bank tracks reusable slots and staged copies. [Expert bank](../../engine/src/family/expert_bank/mod.rs) | Flash-Next lends expert page sets through their synchronization point, Qwen3.6 holds resident packed experts, and Flint holds resident packed ranks. The adapters share MLX's packed matmul. [Flash-Next slots](../../engine/src/family/flash_next/resident/slots.rs) |
| Kernel selection | The serving profile registers CUDA kernels for the engine. [CUDA serving kernels](../../engine/src/kernels/cuda/resident/exl3.rs) | Shared attention checks tensor dtype, shape and Metal threadgroup capacity. The families supply Qwen3.6's decode cutoffs, Flash-Next's sparse QSA and Flint's sink and ring rules. [MLX kernels](../../engine/src/kernels/mlx/mod.rs) |

## Verification on the mini

The checks in this section are the PR author's (zbrdc/fifi#62, commit 88374d1c5), run on his 16 GB M4 mini; we carried the code and fixtures, not the runs.

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
