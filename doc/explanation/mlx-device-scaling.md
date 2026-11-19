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

## Reject oversized loads from headers

With the MLX device budget enabled, Qwen3.6 MoE, Qwen3.8 dense, and Flash-Next check the full-context session reserve before opening shard headers. They then sum the raw packed payloads that must stay resident. If the reserve and that lower bound exceed the device ceiling, startup returns a memory-plan error before reading tensor payloads or creating MLX allocations. Flint already plans from shard metadata and can select its disk-backed experts at this point.

Qwen counts routed expert payloads when serving requires resident banks. When disk-backed experts are enabled, the header check leaves them out and the placement planner prices resident banks and a one-token stage separately. Flash-Next includes recurrent state, full-context K/V and raw index keys, prefix retention when enabled, and capacity overlap in its session reserve. The later full planner accounts for packed scales, biases, staging and other resident allocations using the resident tensor sizes.

On the mini, file-backed Flash-Next, Qwen3.6 resident and streamed-eligible MoE, and Qwen3.8 dense fixtures refused a ceiling one byte below the reserve plus packed-payload floor. The payload counter stayed at zero for each refusal. All four cases loaded with a 4 GiB ceiling. The Flash-Next serving factory also preserved its chunked and one-token fallback behavior. Both seek/read and positioned shard reads passed these checks.

From `engine/` on Apple Silicon, the focused boundary check runs exactly two tests:

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification \
  --lib mlx_preimport_rejects -- --test-threads=1
```

## Shared CPU, SSD and Metal staging

The opt-in `FIFI_MLX_EXPERT_READ_RUNS=prefill` groups adjacent selected expert slices during multi-row prefill in disk-backed Flint, Flash-Next and Qwen3.6. It reads the same logical weight bytes and writes them directly into each family's admitted Metal page layout. Qwen's fused gate/up layout uses vectored reads into disjoint slots, with at most 64 destinations per call. F32 scales and biases still widen in the existing destination slots.

The opt-in `FIFI_MLX_OVERLAP_EXPERT_READS=1` submits the independent shared expert to Metal before the CPU stages the same layer's exact routed experts. The layer output depends on both computations. Each family keeps its existing page release point and staging bound. Flint uses its shared-page pipeline when either common option is enabled. Explicit Flint legacy settings can override the common choices. The Qwen executors share the scheduler, KV formats and prefix lifecycle. Dense Qwen can also stage its feed-forward weights under the device budget. [Apple Silicon controls](../reference/environment.md#apple-silicon-device-sizing)

On the M4 mini, the published Flint checkpoint passed the shared-switch smoke test at an 8 GiB device ceiling with 256 admitted rows. The 16-token prefill used 26,244 reads with coalescing off and 20,979 with it on. Both arms read 5,159,780,352 logical prefill bytes, issued 7,614 decode reads, returned the same four-token completion and imported the staging pages without a copy fallback. The Qwen3.6 disk-backed fixture matched F32 and BF16 activations and issued 540 reads in the shared arm versus 594 in the control with 506,880 logical bytes each. Flash-Next's sparse-prefill and prefix fixtures passed with both common switches off and on. [Flint serving comparison](mac-flint-ane-and-reads.md#balanced-serving-comparison-at-8-gib) · [Full Qwen3.6 checkpoint](#full-qwen36-checkpoint-on-the-m4-mini)

### Positioned local shard reads

`FIFI_MLX_POSITIONED_READS=1` uses `pread` for contiguous payloads and `preadv` for disjoint staging ranges in Flint, Flash-Next, Qwen3.6 and Qwen3.8 dense. The checked shard range supplies the offset. The reader advances it past partial reads and writes directly into the caller's buffers. Each operation issues the same requested byte ranges with a private offset, avoiding a shared cursor lock and the separate seek. Flash-Next and both Qwen families now share the same local indexed opener, which validates the index before opening the named shards.

On the internal SSD, two balanced pairs of fresh Flint servers compared the reader against seek/read. Both arms admitted 256 rows at 8 GiB and used the same prefill coalescing and placement. All eight 12-token completions matched, as did logical read bytes and expert read counts. The arithmetic prompt reached its first token about 3% sooner and decoded 7% faster. The summarization prompt had flat first-token time and decoded 3% slower. Total process-attributed disk reads were 56.98 GB for seek/read and 56.22 GB for positioned reads. The reader remains opt-in. [Eight serving requests](../../scripts/benchmarks/mac-m4-flint-positioned-reads.csv)

Two isolated warm replays each used 64 captured routes from Flint layer 24. Each arm filled the same 54 destinations four times per route with alternating arm order. Every output buffer matched. Positioned reads lowered paired median read time by 6.3% in the first replay and 0.7% in the repeat. The replays time read calls into preallocated buffers after warming each route. [Warm read measurements](../../scripts/benchmarks/mac-m4-flint-positioned-warm.csv)

Run the Flint serving driver with `--expert-timing --positioned-reads --working-set-mib 8192 --pairs 2` to repeat the comparison.

```sh
python3 scripts/mac_cache_study/flint_positioned_reads.py "$CHECKPOINT" \
  scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz "$SCRATCH/positioned-warm.csv"
```

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

### Capacity-backed Flash-Next QSA

With `FIFI_FLASH_NEXT_MLX_CAPACITY_KV=1`, Flash-Next decode and batched prefill use the same 256-row minimum and power-of-two growth policy as compact Flint KV. QSA keeps separate buffers for F32 keys, values and raw index keys. The worker budgets the full allocated rows before advancing. Attention and the indexer see only the valid prefix. Writes to a uniquely owned buffer can reuse its storage. A retained snapshot keeps the previous values through copy-on-write, and the worker charges the snapshot independently.

The file-layout worker fixture compared both paths at a 4 GiB device ceiling with 128-row prefill and sparse batching enabled. Four pairs of fresh processes ran four requests per length and arm. All 64 requests produced identical eight-token continuations. A separate test restored a snapshot, appended a decode token without changing the saved arrays, and verified that insufficient room blocked the next growth before mutating the state. The same growth succeeded after releasing the saved snapshot and produced matching logits. [Worker samples](../../scripts/benchmarks/mac-m4-flash-next-capacity-kv.csv)

| Prompt | Median first token | Paired TTFT change | Median decode interval | Paired decode change | Median MLX peak active |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65 tokens | 20.56 → 19.94 ms | −8.7% | 10.97 → 11.81 ms | +9.6% | 1.04 → 1.31 MB |
| 257 tokens | 86.43 → 88.93 ms | +2.5% | 13.17 → 10.84 ms | −12.3% | 1.15 → 1.29 MB |

The 257-token request crosses the initial capacity boundary as generation starts. Both arms performed the same number of Metal synchronizations at each length. The new buffers are opt-in while full-checkpoint and longer-context timings are open. Use the worker driver on the release MLX library test executable to reproduce the comparison:

```sh
python3 scripts/bench_mlx_requests.py "$MLX_LIB_TEST" "$SCRATCH/flash-capacity.csv" \
  --flash-capacity-kv --pairs 4 --samples 4
```

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

Flash-Next reads routed experts from disk and stages them in bounded page sets. The dense Qwen prototype stages packed feed-forward matrices under the same Metal budget.

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

### Full Qwen3.6 checkpoint on the M4 mini

The mini ran the pinned 20.42 GB Qwen3.6 MLX 4-bit checkpoint from the external SSD. All four shards were verified against the publisher manifest, and the tokenizer bundle matched the engine pins. The smoke used an 8 GiB Metal ceiling, 4,096-token context, BF16 activations, Q8 KV, 128-row prefill, a 256 MiB allocator cache limit, and a 1 GiB session cap. [Pinned Qwen3.6 checkpoint](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit/tree/38740b847e4cb78f352aba30aa41c76e08e6eb46)

The resident pre-import check required 18,543,034,368 bytes and refused the load. With `FIFI_Q36_MLX_DISK_EXPERTS=1`, the worker reported disk-backed experts, 1,167,156,748 bytes of resident weights, a 1,457,061,888-byte session reserve, and 452,984,832 bytes of expert staging. It admitted 128-row prefill.

The A/B compared shared-expert overlap and prefill read coalescing together against streamed staging with both switches off. Device admission and prefill widths were identical. The two pairs alternated order with a fresh server per arm and retained the OS file cache. All eight timed requests matched all 12 sampled tokens and their output hashes.

| Prompt | Median first token, control → shared | Attributed SSD reads | Decode throughput |
| --- | ---: | ---: | ---: |
| 27 tokens | 14.49 → 12.55 s | 8.65 → 8.26 GB | 1.39 → 1.36 tok/s |
| 157 tokens | 34.67 → 27.02 s | 19.13 → 16.36 GB | 1.02 → 1.43 tok/s |

The longer prompt reached its first token 22% sooner and attributed 15% fewer bytes to the external SSD. Decode throughput ranged from 0.87 to 1.70 tok/s across the runs. Median lifetime process peak was 3.020 GB for the control and 3.045 GB for shared staging. Those peaks include startup and the earlier request in each server process. [Serving and smoke measurements](../../scripts/benchmarks/mac-m4-qwen36-streamed-ab.csv)

Build the Apple Silicon server with `native-mlx,native-fault-qualification`. Use a verified checkpoint directory and the process I/O counter from the [staging study](mac-flint-expert-pipeline.md#reproduce). Run the resident-refusal and streamed smoke first. Use a new output directory and `--pairs 2` for the serving A/B.

```sh
python3 scripts/mac_cache_study/qwen36_streamed_ab.py \
  "$ENGINESRV" "$QWEN36_CHECKPOINT" "$SCRATCH/proc-io" "$SCRATCH/qwen36-stream-smoke" --smoke
```

### Qwen3.6 streaming attribution

Set `FIFI_Q36_MLX_STREAM_TIMING=1` in a `native-mlx,native-fault-qualification` Apple Silicon build to capture one `qwen36_stream_timing` event per successful disk-backed MoE forward. The event records its position and row count. Logging is opt-in and uses the worker's existing synchronization points.

The profiler measures CPU top-k selection and multi-row scheduling after the router readback. It also records the logical expert reads, their call time, the highest mapped stage, page-import copies, and how long each stage spans from selection and allocation through its final evaluate. The existing MLX counters cover wall time inside imports, readbacks and explicit evaluations. Readback can execute already queued Metal work. Stage fill includes shard reads and page import, and the stage lifetime includes the evaluation that makes its output safe to keep. These nested intervals describe the work and waits along the forward.

On the pinned Qwen3.6 checkpoint, the 27-token smoke produced the same four sampled tokens with the combined switches on or off. Prefill reads went from 30,447 calls to 22,041 at 5.99 GB of logical expert data in each arm. The profiler accounted for two prefill forwards and three decode forwards per request, with zero Metal page copies.

The four-arm comparison ran twice in opposite order with a fresh server per arm. Both prompts produced the same four sampled tokens and response hash in all 16 requests. Every arm admitted 128 rows at 8 GiB. All 88 forwards imported their expert pages with zero fallback copies.

| Prompt | Shared expert overlap | Read coalescing | Median first token | Prefill read calls | Median time in shard reads | Request SSD reads |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 27 tokens | Off | Off | 18.08 s | 30,447 | 16.82 s | 7.05 GB |
| 27 tokens | Off | On | 13.86 s | 22,041 | 13.06 s | 5.44 GB |
| 27 tokens | On | Off | 14.62 s | 30,447 | 13.86 s | 4.98 GB |
| 27 tokens | On | On | 13.79 s | 22,041 | 13.01 s | 5.39 GB |
| 157 tokens | Off | Off | 41.08 s | 74,403 | 38.88 s | 14.35 GB |
| 157 tokens | Off | On | 31.41 s | 43,407 | 29.51 s | 12.63 GB |
| 157 tokens | On | Off | 38.52 s | 74,403 | 36.64 s | 12.59 GB |
| 157 tokens | On | On | 31.65 s | 43,407 | 29.81 s | 12.71 GB |

Coalescing alone cut the longer prompt's first-token time by 24% and read calls by 42%. Its logical prefill reads stayed at 14.63 GB. Adding shared-expert overlap to coalescing changed median first-token time by less than 1% on either prompt. The runs kept the OS file cache between arms, so the observed overlap-only differences also reflect cache state.

Across the longer prompt's four arms, median host routing took 45–48 ms and page imports took 11–12 ms. Median wall inside native readback was 0.61–0.73 s and explicit evaluation was 0.99–1.20 s. The host read calls occupied approximately 94–95% of the time to first token. Decode stayed at 8,640 slice reads and 1.70 GB of logical expert data over its three forwards for the longer prompt in every arm. Its median throughput ranged from 1.24 to 1.42 tokens/s. The study leaves qualification and defaults open. [Request measurements](../../scripts/benchmarks/mac-m4-qwen36-attribution-requests.csv) · [Per-forward measurements](../../scripts/benchmarks/mac-m4-qwen36-attribution-forwards.csv)

Run the qualified server comparison with both switches independently under one device plan. The driver checks response and token identity, verifies every event from prefill through decode, and preserves raw per-forward timings in the output directory.

```sh
python3 scripts/mac_cache_study/qwen36_streamed_ab.py \
  "$ENGINESRV" "$QWEN36_CHECKPOINT" "$SCRATCH/proc-io" "$SCRATCH/qwen36-attribution-smoke" \
  --attribute --smoke
python3 scripts/mac_cache_study/qwen36_streamed_ab.py \
  "$ENGINESRV" "$QWEN36_CHECKPOINT" "$SCRATCH/proc-io" "$SCRATCH/qwen36-attribution" \
  --attribute --pairs 2
```

## Disk-backed dense Qwen feed-forward weights

Set `FIFI_Q36_MLX_DISK_DENSE_FFN=1` with `FIFI_MLX_AUTO_BUDGET=1` or a positive `FIFI_MLX_WORKING_SET_MIB` to try the dense Qwen3.8 path. It retains the attention and recurrent mixer weights, norms and output head in Metal. The loader reads the feed-forward matrix metadata at startup and releases each layer's host buffers as its resident weights import. Decode and chunked prefill read the gate, up and down matrices for one layer, evaluate the existing SwiGLU on Metal and retire the stage before the next layer.

The device plan charges the largest stage, including space for a possible Metal import copy, alongside resident weights and the full serving reserve. Chunk admission adds the same stage to the row workspace. The authenticated execution report shows `weight_placement=disk_dense_ffn` and `memory.workspace.dense_ffn_stage_bytes`. Every forward rereads these logical weight bytes. Wider prefill chunks amortize them over more tokens.

A file-backed fixture on the M4 mini matched resident prefill and decode logits bit for bit across F32/F32, BF16/F32, BF16/BF16 and BF16/Q8 activation/KV profiles. It checked the startup and four-row prefill admission boundaries to one byte. Each profile reserved a 288 KiB stage and read 172,800 logical bytes in 135 contiguous reads, with zero Metal import copies.

The pinned 27B 6-bit pack's headers put 13.90 GB in the deferred feed-forward matrices, 217.25 MB per layer. The inventory counts 6.92 GB of remaining resident weight payloads before conversion and serving reserves. It excludes the file-backed token embedding and unused vision and MTP tensors. [Pinned Qwen3.8 weights](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-6bit/tree/e05430e1ab7050b0b809a56ed2fdfe09570d4449)

Run the file-backed checkpoint test from `engine/` on Apple Silicon.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  family::qwen36::resident::executor::tests::capped_dense_qwen_stages_one_file_backed_ffn_at_a_time \
  -- --exact --nocapture --test-threads=1
```

### Full Qwen3.8 dense checkpoint on the M4 mini

The external SSD held all five verified shards from the pinned 27B 6-bit pack. The server used a separate tokenizer directory containing the exact engine-pinned files. The weights publisher's chat template and generation config matched the pins. The tokenizer JSON and tokenizer config came from two official Qwen revisions. [Weight pack and templates](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-6bit/tree/e05430e1ab7050b0b809a56ed2fdfe09570d4449) · [Tokenizer JSON](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/tokenizer.json) · [Tokenizer config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/8c0b3d30f3ae96f5c7f774de704bf0ada83e422a/tokenizer_config.json)

The resident header check required 20,657,078,272 bytes under the 10 GiB Metal ceiling and refused startup. Staged startup admitted 128 prefill rows with BF16 activations, Q8 KV, a 2,048-token context and 128 MiB of allocator cache. The server reported 6,922,976,268 bytes of resident weights, a 1,440,612,352-byte session reserve and 434,503,680 bytes for the largest FFN stage. Prefix reuse and compilation were off.

Both runs answered the 28-token arithmetic prompt with `42` and matched the two sampled token IDs and result hash.

| Run | Ready | First token | Decode | Request SSD reads | Lifetime process peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 17.27 s | 34.37 s | 0.06 tok/s | 41.54 GB | 10.82 GB |
| Reproduction | 19.17 s | 33.85 s | 0.06 tok/s | 41.53 GB | 10.82 GB |

The runs used the same checkpoint and serving settings without flushing the OS file cache between them. The process I/O counter covers each request after the server was ready. [Raw measurements](../../scripts/benchmarks/mac-m4-qwen38-dense-smoke.csv)

The token scheduler prefills 27 prompt rows, runs the pending prompt token to sample the first output token, then runs one decode step. Those three forwards imply 41.50 GB of logical FFN reads from this pack. The prompt-only forward skips the final layer's FFN. Wider prompts can amortize their chunk reads, while reducing first-token latency here requires reducing the number or size of the scans. The device ceiling covers the Metal plan. Process peak also includes host startup allocations.

Build the Apple Silicon server with `native-mlx,native-fault-qualification`. Use the pinned checkpoint, tokenizer directory and process I/O counter from the [staging study](mac-flint-expert-pipeline.md#reproduce).

```sh
python3 scripts/mac_cache_study/qwen38_dense_smoke.py \
  "$ENGINESRV" "$QWEN38_CHECKPOINT" "$QWEN38_TOKENIZER" "$SCRATCH/proc-io" \
  "$SCRATCH/qwen38-dense-smoke"
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
