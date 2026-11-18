# Device-aware MLX prefill

Qwen3.6, Qwen3.8 dense, Flash-Next and Flint can size their prefill work to the Apple Silicon GPU from the same startup policy. Metal supplies `recommendedMaxWorkingSetSize` in bytes. Each executor accounts for its resident weights and session memory, then uses the remaining budget and its configured row ceiling to select a chunk. All four use the same largest-fitting-chunk rule.

```sh
FIFI_MLX_AUTO_BUDGET=1 enginesrv --config /path/to/config.json
```

Use an Apple Silicon build with the `native-mlx` feature. The option is off by default. When enabled, it applies to one MLX model per server because the device reports one shared working set. Metal's recommendation is a sizing input for the total working set. Other apps, other server processes and allocation peaks during weight import can put additional pressure on memory. The policy checks resident memory and serving reserves before accepting jobs. Flint also uses a header-based estimate to choose where to keep its weights before import.

The 16 GB M4 mini reports 12,713,115,648 bytes (11.84 GiB). The implementation queries Metal on every opted-in model startup and uses the resulting byte count. `FIFI_MLX_WORKING_SET_MIB` also activates automatic sizing and caps the plan at the lower of the configured ceiling and Metal's recommendation. A family-specific `prefill_budget_bytes` can bound its prefill workspace further. The logs distinguish the Metal recommendation from the effective budget and, when set, the configured ceiling. Context length and KV format come from the family's configuration.

| Family | Sizing inputs in addition to resident weights and allocator headroom | Selection |
| --- | --- | --- |
| Qwen3.6 and Qwen3.8 dense | Full-context KV capacity at the selected dtype, DeltaNet state, a layer's cache growth overlap and the aggregate session cap when prefix reuse or concurrency applies | Uses the existing chunk cost model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. Qwen3.6 concurrent serving requires an explicit byte budget and a fitting chunk. |
| Flash-Next | Decode pool, full-context QSA including index keys, DeltaNet and PLE state, overlap while replacing session arrays, and configured prefix slots | Uses the existing expert-wave workspace model and worker row ceiling. An explicit `prefill_budget_bytes` supplies a further upper bound. |
| Flint | Full KV planes, 513-position sliding rings, one layer's KV growth overlap, bounded prefix retention and a conservative upper estimate of the full-attention score, mask and softmax workspace | Chooses among 1,024, 512 and one row within the ceiling set by `FIFI_FLINT_MLX_BATCH_PREFILL`. With `FIFI_FLINT_MLX_ADAPTIVE_PREFILL=1`, the shared ladder also considers 256, 128, 64, 32, 16, eight and four rows. The prefix cache records the selected width. |

Qwen3.6 and Flash-Next can fall back to one-token prefill if the workspace for a chunk does not fit. Startup logs carry the Metal recommendation, the charged resident and reserve bytes, and the selected path. [MLX budget code](../../engine/src/family/mlx_budget.rs) · [Environment reference](../reference/environment.md#apple-silicon-device-sizing)

## Inspect the selected execution path

Run `enginesrv config effective <path> --json` with the serving environment to inspect the resolved configuration for Flint, Flash-Next, Qwen3.6 MoE or Qwen3.8 dense. The rows show the configured prefill ceiling, backend, KV format and applicable MLX switches. The capability bundle includes these rows.

Once an engine becomes ready, authenticated `GET /v1/system` includes its worker's startup decisions under `engines[].execution`. The report includes the chosen weight placement, prefill mode, reason and row ceiling. The fields `prefill_rows` and `worker_piece_rows` describe the maximum batch within the executor and the maximum input piece from the scheduler. A worker can submit a piece of 16 tokens that the executor processes one token at a time. Chunked execution can use smaller batches at request boundaries or deeper contexts.

When a prefill budget applies, `memory` lists the base charges, bytes left before chunk workspace, and any separate staging charge or sparse allowance. Device sizing also records the Metal recommendation and effective device ceiling. The API reads the record from successful startup without probing Metal during the request. [System API contract](../../api/openapi.yaml)

On the M4 mini, the published Flint checkpoint with a 4 GiB device ceiling and a configured 512-row maximum started with disk-backed experts and admitted 32 prefill rows. The reported row width, placement and memory charges matched the startup log. The live settings rows also matched `config effective --json` for the same environment.

## Shared CPU, SSD and Metal staging

The opt-in `FIFI_MLX_EXPERT_READ_RUNS=prefill` groups adjacent selected expert slices during multi-row prefill in disk-backed Flint, Flash-Next and Qwen3.6. It reads the same logical weight bytes and writes them directly into each family's admitted Metal page layout. Qwen's fused gate/up layout uses vectored reads into disjoint slots, with at most 64 destinations per call. F32 scales and biases still widen in the existing destination slots.

The opt-in `FIFI_MLX_OVERLAP_EXPERT_READS=1` submits the independent shared expert to Metal before the CPU stages the same layer's exact routed experts. The layer output depends on both computations. Each family keeps its existing page release point and staging bound. Flint uses its shared-page pipeline when either common option is enabled. Explicit Flint legacy settings can override the common choices. Resident Qwen and Qwen3.8 dense use their resident projections and share the scheduler, KV formats and prefix lifecycle. [Apple Silicon controls](../reference/environment.md#apple-silicon-device-sizing)

On the M4 mini, the published Flint checkpoint passed the shared-switch smoke test at an 8 GiB device ceiling with 256 admitted rows. The 16-token prefill used 26,244 reads with coalescing off and 20,979 with it on. Both arms read 5,159,780,352 logical prefill bytes, issued 7,614 decode reads, returned the same four-token completion and imported the staging pages without a copy fallback. The Qwen3.6 disk-backed fixture matched F32 and BF16 activations and issued 540 reads in the shared arm versus 594 in the control with 506,880 logical bytes each. Flash-Next's sparse-prefill and prefix fixtures passed with both common switches off and on. Full-checkpoint measurements on this mini cover Flint. [Flint serving comparison](mac-flint-ane-and-reads.md#balanced-serving-comparison-at-8-gib)

## KV appends and retained prefixes

The M4 mini measured a single synchronized append to representative cache planes at 8,191 and 32,767 prefix tokens. Each case used six timed samples, varied arm order, and checked the old prefix row, the appended row and the retained snapshot. Input allocation and readback sit outside the timing interval. The snapshot arm shares the buffer before appending. It measures the cost of that copy-on-write boundary while a saved prefix still owns the array.

| Plane at 32,767 prefix tokens | Unique append | Snapshot-backed append | Concatenate | Extra peak with snapshot |
| --- | ---: | ---: | ---: | ---: |
| Flash-Next F32 KV | 1.26 ms | 4.68 ms | 4.67 ms | 64 MiB |
| Qwen3.6 Q8 packed codes | 0.57 ms | 1.55 ms | — | 16 MiB |
| Qwen3.8 dense BF16 KV | 1.28 ms | 4.68 ms | 4.51 ms | 64 MiB |
| Flint BF16 KV | 0.80 ms | 2.83 ms | 2.83 ms | 32 MiB |

Unique appends took zero extra active allocation in every measured case. Each retained snapshot added one plane at peak. Qwen's Q8 row measures the packed U32 codes, with scales stored separately. Flash-Next's sample covers the F32 K plane, with its V and index-key planes accounted for separately in the serving budget. The table reports operator medians. Flint's two-turn serving test measures the net effect of retaining a real prefix, including save overhead and read savings. [132 KV measurements](../../scripts/benchmarks/mac-m4-kv-append-matrix.csv) · [Flint two-turn results](mac-flint-ane-and-reads.md#prefix-reuse-on-two-turn-requests)

Run from `engine/` on Apple Silicon:

```sh
FIFI_MLX_KV_APPEND_SAMPLES=6 cargo test --locked --release \
  --features native-mlx,native-fault-qualification --test native_kv_ops \
  mlx_kv_append_storage_matrix -- --ignored --exact --nocapture --test-threads=1
```

The test prints 132 `kv_append_sample` records with elapsed nanoseconds and extra active peak bytes.

## Constrained-memory comparisons

The adaptive Flint path prices each batch using the same context, resident weights, expert staging, and temporary attention cost as coarse admission. It uses the common resident prefill ladder and preserves one-row fallback at the bottom.

For the real-model comparison, the driver supplies a 4,096 MiB budget ceiling on the same 16 GB M4 mini. Both arms use the pinned internal-SSD Flint INT4 checkpoint, a 32,768-token limit, shared-page expert staging, prefix reuse off and 12 greedy completion tokens. The control uses the coarse ladder. The candidate uses the shared ladder. The worker reports the admitted width and the width it actually ran. This measures a constrained serving plan on the M4, with the full 16 GB still physically available.

Four fresh servers ran in control, candidate, candidate, control order. The coarse path admitted one row. The shared ladder admitted 32. Every completion matched, and the 45-token prompt ran as two worker pieces with the ladder enabled. The table shows medians from two servers per mode.

| Prompt tokens | Time to first token, one-row → adaptive | Logical prefill reads | Full request | Peak process footprint |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 7.90 → 6.69 s | 7.98 → 5.16 GB | 13.40 → 12.63 s | 1.73 → 1.82 GB |
| 45 | 23.28 → 16.27 s | 22.45 → 12.86 GB | 29.53 → 22.43 s | 1.73 → 1.92 GB |

Flint reached its first token 15% and 30% sooner on these prompts. The worker reused each staged expert across the rows that selected it. Decode still issued the same number of slice reads. In these short runs, measured decode throughput varied from 7% slower to 1% faster, and peak process footprint increased by up to 0.19 GB at the median. [All eight Flint requests](../../scripts/benchmarks/mac-m4-flint-adaptive-prefill.csv)

The Flash-Next, Qwen3.6 MoE and Qwen3.8 dense worker fixtures compared their one-row execution routes with device-budgeted eight-row prefill. The candidate arms use the same 4,096 MiB Metal ceiling along with the fixtures' explicit 2 GiB prefill budget. The worker uses the same eight-row submission piece in both arms, and the correct internal route is checked at startup. These are synthetic models. The Qwen fixtures use BF16 activations and F32 KV. All 192 continuation sequences matched.

| Fixture | Prompt tokens | Time to first token, one-row → chunked | Paired change | Metal waits, one-row → chunked |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6 MoE | 8 | 8.14 → 3.45 ms | −53% | 47 → 39 |
| Qwen3.6 MoE | 33 | 31.73 → 11.18 ms | −65% | 97 → 59 |
| Qwen3.8 dense | 8 | 3.89 → 1.45 ms | −63% | 47 → 36 |
| Qwen3.8 dense | 33 | 14.64 → 3.68 ms | −75% | 97 → 47 |
| Flash-Next | 8 | 11.07 → 3.18 ms | −72% | 69 → 45 |
| Flash-Next | 33 | 44.29 → 24.69 ms | −43% | 169 → 105 |

The timings use four fresh process pairs in each family and four recorded samples per row in each process. Latency columns report medians of process medians. The percentage column reports the median of the four paired changes. Decode timing varied by case. The 33-token dense fixture's median active MLX peak rose from 143 to 225 KiB. [Worker measurements](../../scripts/benchmarks/mac-m4-mlx-capped-worker-requests.csv)

## Disk-backed Flint experts

With automatic sizing enabled, Flint checks the checkpoint headers and serving reserves before importing weights. If the full resident model fits, Flint loads it as before. Otherwise it keeps attention, routers, shared experts, and the output head in memory and reads the routed experts by ID from the existing local safetensors files. Each layer imports only the experts selected for the current rows, restores the original routing order, and releases its staged weights after Metal completes the grouped projection. The model files stay on the local drive and the KV cache stays in unified memory.

The planner charges both the host staging buffers and their Metal copies. Flint reduces prefill rows when that staging space and attention workspace do not fit together. It refuses startup before weight import when the resident core and one token of expert staging exceed the budget. The startup log reports `weight placement=disk-experts` or `weight placement=resident` and the selected prefill width. With Flint prefix reuse explicitly enabled, an oversized checkpoint also needs an explicit `FIFI_FLINT_PREFIX_BUDGET_BYTES` because the legacy cap derivation charges the entire checkpoint.

Flash-Next reads routed experts from disk and stages them in bounded page sets. Qwen dense models would need layer streaming to serve checkpoints whose resident core exceeds the budget.

On the 16 GB M4 mini, a sparse zero-weight fixture with Flint's full tensor layout selected disk-backed experts under a 2 GiB weight budget. Its 22.24 GB resident-weight estimate includes all expert banks. The disk-backed path estimated 952 MB of resident weights and peaked at 1.04 GiB of MLX allocations during two prefill rows and one decode step. A separate nonzero fixture compared the disk-backed and resident expert outputs bit for bit for one and two rows, with repeated and nonconsecutive expert IDs. The published checkpoint was separately measured with the internal SSD and shared-page staging. [Flint measurements](mac-flint-resident-cache.md#internal-ssd-timing)

Run these two tests from `engine/` on Apple Silicon:

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib -- \
  disk_experts_match_resident_grouped_predictions \
  sparse_oversized_flint_checkpoint_streams_and_decodes_on_mlx --nocapture --test-threads=1
```

## Disk-backed Qwen3.6 experts

Enable `FIFI_Q36_MLX_DISK_EXPERTS=1` with `FIFI_MLX_AUTO_BUDGET=1` or a positive `FIFI_MLX_WORKING_SET_MIB` ceiling. The loader imports the non-routed weights first and prices the full set of expert pages against the working set after the serving reserve. If the full set fits, the engine uses its resident routing, prefill and compilation. If only a layer's selected experts fit, the engine uses host routing for one session and reads the selected weights from the original local shards. Startup prints the selected placement and the charged bytes.

The compact IDs keep the original route and rank order. Qwen uses the same packed gate/up and down layout for both placements. F32 activation widens the BF16 scales and biases in place while reading the selected pages. The engine evaluates each routed layer before releasing those pages. The chunk planner prices the largest layer's stage for up to `rows × top_k` distinct experts, capped at the layer's expert count. If no chunk above one row fits, the worker uses one-token prefill. Startup returns a memory error when one row of staging cannot fit or the engine would need streamed experts for concurrent serving. [Qwen weight placement](../../engine/src/family/qwen36/resident/weights.rs)

On the 16 GB M4 mini, a four-layer file-backed fixture priced 384 KiB for the resident expert banks and 96 KiB for a one-layer stage in both activation modes. The staged path kept zero resident expert pages and read 495 KiB of expert data across 10 prompt tokens and the test's forced decode steps. It matched the resident token and compared logits within the existing parity tolerance. The test checks admission one byte below and exactly at the one-row stage and four-row chunk boundaries. Full-checkpoint throughput and peak memory still need measurement.

Run the file-backed checkpoint test from `engine/` on Apple Silicon.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  family::qwen36::resident::executor::tests::capped_qwen_streams_selected_experts_from_a_file_backed_checkpoint \
  -- --exact --nocapture --test-threads=1
```

## CUDA and MLX interfaces

| Responsibility | CUDA | MLX |
| --- | --- | --- |
| Device budget | Device memory and arena admission use the shared byte-reserve and fitting-candidate arithmetic. [CUDA admission](../../engine/src/family/prefill/admission.rs) | Metal supplies one unified-memory working set. The same byte-reserve arithmetic checks family charges and the MLX chunk chooser sizes prefill. [Shared reserve](../../engine/src/family/device_budget.rs) · [MLX planner](../../engine/src/family/prefill/mlx.rs) |
| KV format | KV row math accounts for plain and packed cache planes and their auxiliary values. | Qwen3.6 prices F32, BF16 and Q8 with the same KV row math. Flint uses BF16 KV and separate capacities for full and sliding layers. Flash-Next adds a separate F32 index row to its F32 KV cost. [KV formats](../../engine/src/family/kv_format.rs) |
| Expert staging | The expert bank tracks reusable slots and staged copies. [Expert bank](../../engine/src/family/expert_bank/mod.rs) | Flash-Next lends expert page sets through their synchronization point. Qwen3.6 and Flint can stage selected experts from disk and share the compact expert-ID mapping. The adapters share MLX's packed matmul. [Flash-Next slots](../../engine/src/family/flash_next/resident/slots.rs) |
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

To rerun the constrained comparisons, build with `cargo build --locked --release --features native-mlx,native-fault-qualification --bin enginesrv` from `engine/`. Build the release library harness and save the library test executable path as `LIB_TEST`. Set `CHECKPOINT` to the verified internal Flint directory. The [staging study](mac-flint-expert-pipeline.md#reproduce) shows how to build the process counter. Use fresh scratch output paths for both commands.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib --no-run
cd ..
python3 scripts/bench_mlx_requests.py "$LIB_TEST" "$RUN_DIR/mlx-workers.csv" \
  --prefill-admission --pairs 4 --samples 4
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR/flint" --expert-timing --adaptive-prefill
```
