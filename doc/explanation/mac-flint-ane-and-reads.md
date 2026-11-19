# The M4 Neural Engine and Flint expert reads

## Neural Engine on this mini

Core ML's device enumeration reported a CPU, GPU and 16-core Neural Engine on the 16 GB M4 Mac mini on 18 November 2026. The existing experimental fifi decision head passed all six ANE tests, including output checks against the CPU reference. The probe exercises a fixed two-layer decision head with 4,096 inputs, a 512-wide hidden layer and 260 outputs. Its bridge uses private Apple APIs. [Mac mini specifications](https://support.apple.com/en-ca/121555) · [Core ML compute devices](https://developer.apple.com/documentation/coreml) · [fifi ANE tests](../../engine/tests/decision_ane.rs)

Three fresh processes each timed 500 predictions per backend after 16 warm-ups. ANE median call latency was **131 µs**, with a median p99 of **443 µs** across processes. The scalar Rust CPU reference had a median call latency of **1.71 ms**. The timing includes each prediction's host handoff and excludes compilation, model load and Flint inference. [All six timing rows](../../scripts/benchmarks/mac-m4-ane-idle.csv)

Flint's serving path uses MLX packed expert projections on Metal and reads the routed weights from local storage. A small ANE expert predictor is worth evaluating independently. Its job would be to start reads early, with the actual GPU router still selecting the experts used in the forward pass. In the six-request trace, reusing the same layer's previous token choices predicted 5,965 of 34,404 requested experts, or 17.3% recall. A learned predictor should beat that baseline and add useful early reads after cache hits are excluded. Measure recall, wasted SSD bytes, added memory, and end-to-end request time. The public deployment route would use Core ML and check the graph's placement with `MLComputePlan`, then measure wall time separately. [Apple's placement and timing guidance](https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode)

## Contiguous expert reads

Flint stages sorted expert IDs from file-backed safetensors. When multiple selected experts are consecutive, both their source bytes and their destination slots are contiguous. The new option combines each such run into a checked read of the exact requested bytes. It uses the same page layout, output remapping and Metal dispatch as the shared-page pipeline. Missing tensors and invalid extents or destination lengths still fail before a read.

From the prior six-request trace, the number of read calls per tensor role would fall from 28,164 to 14,657 for multi-row stages, or 48.0%. For single-row stages it would fall from 36,096 to 35,396, or 1.9%. The logical bytes are identical. [Expert trace](mac-flint-expert-reuse.md)

The published checkpoint A/B used four fresh benchmark processes in control, coalesced, coalesced, control order. Both arms used shared pages, 512-row prefill admission, prefix reuse off and the same two prompts with 12 completion tokens. The benchmark discarded its first attempt after detecting an active live request. An idle guard checked the live service during the measured retry. All eight responses matched across both modes.

| Prompt tokens | Prefill, control → coalesced | Decode, control → coalesced | Full request time, control → coalesced |
| ---: | ---: | ---: | ---: |
| 16 | 7.72 → 6.91 s | 1.52 → 1.37 tokens/s | 14.99 → 14.98 s |
| 143 | 18.72 → 14.11 s | 1.46 → 1.70 tokens/s | 26.66 → 20.70 s |

The 143-token request cut prefill by 24.6% and total latency by 22.3%. At 16 tokens, total latency was level and decode varied across the runs. The attributed disk-read counter fell from 23.37 to 19.83 GB on the larger prompt. That count includes the file system's caching and read-ahead, while the reader requests the same logical bytes. Peak process footprint was within 1% on both prompts. The release server built and 66 selected Flint and shared-read regressions passed. A prefill-only setting and a larger prompt set would help establish where to enable coalescing by default. [All eight requests](../../scripts/benchmarks/mac-m4-flint-read-runs.csv)

The option is `FIFI_FLINT_MLX_EXPERT_READ_RUNS=1` with `FIFI_FLINT_MLX_EXPERT_PIPELINE=1`. The production defaults are off. The already measured shared-page pipeline was enabled on the live server as a local runtime opt-in. The GUI returned ready and listed Flint after that restart. The separate contiguous-read switch remains off on the live instance. [Shared-page pipeline study](mac-flint-expert-pipeline.md)

## Coalescing multi-row prefill on internal storage

`FIFI_FLINT_MLX_EXPERT_READ_RUNS=prefill` applies contiguous reads to transient expert staging whenever a Flint forward has more than one row and shared-page staging is active. Single-row forwards use the per-slice reader, including the final prompt token and every decode step. The existing `1` setting also coalesces single-row transient staging.

The comparison uses the pinned Flint INT4 checkpoint on the M4 mini's verified internal APFS copy. Both modes use shared pages, 512-row admission, and 12 greedy completion tokens per request. Prefix reuse and the expert cache are off. Each fresh server handles the same prompts in order, with servers running in control, candidate, candidate, control order. The 557-token chunk-boundary fixture repeats the same maintenance bulletin for four branches to exercise a full 512-row piece and its tail.

| Prompt tokens | Prefill read calls, control → coalesced | Time to first token, control → coalesced | Full request, control → coalesced |
| ---: | ---: | ---: | ---: |
| 16 | 26,244 → 20,979 | 5.96 → 5.50 s | 11.68 → 11.22 s |
| 45 | 52,821 → 29,727 | 13.47 → 9.58 s | 19.53 → 15.69 s |
| 143 | 82,575 → 23,472 | 17.01 → 12.07 s | 22.16 → 17.27 s |
| 557 | 142,722 → 40,392 | 28.65 → 21.23 s | 34.61 → 27.19 s |

The table shows medians from two fresh servers per mode. The 45-, 143-, and 557-token prompts reached their first token 26–29% sooner. The 16-token prompt improved 7.6% at the median, with more variation between its two control runs. The server requested the same logical bytes on every match, and all 16 completion digests matched their controls. Every run issued 27,918 single-row decode reads. Corresponding median decode rates were within 1%, and the median peak process footprint rose by at most 120 MB. The startup logs and full timing records confirm the multi-row path on the two-piece fixture. [All 16 requests](../../scripts/benchmarks/mac-m4-flint-prefill-read-runs.csv)

The mode remains opt-in. Broader performance qualification requires production-serving workloads beyond these two runs per prompt. The live server uses the shared-page pipeline with coalescing off.

### Balanced serving comparison at 8 GiB

The same published checkpoint ran on the internal SSD under an 8 GiB device ceiling. Both arms selected 256-row prefill, so the 557-token prompt used three pieces. Three pairs of fresh servers alternated their starting arm and each answered the same five prompts with 12 greedy completion tokens. The earlier 512-row comparison used four prompts. The new set adds an independent short explanation prompt.

| Prompt | Median first token | Paired change | Prefill reads | Median attributed disk reads |
| --- | ---: | ---: | ---: | ---: |
| 16 tokens, arithmetic | 6.13 → 5.09 s | −15.7% | 26,244 → 20,979 | 7.50 → 6.61 GB |
| 16 tokens, explanation | 6.16 → 5.00 s | −18.9% | 27,999 → 22,257 | 7.32 → 6.22 GB |
| 45 tokens, code review | 12.74 → 9.16 s | −26.4% | 52,821 → 29,727 | 14.81 → 12.08 GB |
| 143 tokens, summarization | 16.69 → 11.93 s | −28.3% | 82,575 → 23,472 | 21.56 → 18.08 GB |
| 557 tokens, multi-piece | 44.86 → 32.50 s | −27.3% | 216,099 → 66,033 | 53.83 → 45.35 GB |

Every completion digest, logical byte count and one-row decode read count matched the corresponding control. Neither arm had a Metal page-copy fallback. Total process-attributed disk reads fell from 312.57 to 265.79 GB across the 15 requests per arm. Paired median decode-rate changes ranged from −1.5% to +1.5%. Median peak footprints differed by up to about 54 MB. These results cover the M4 mini and its internal checkpoint. [All 30 requests](../../scripts/benchmarks/mac-m4-flint-coalescing-8gib.csv)

Run the serving driver with `--expert-timing --prefill-read-runs --working-set-mib 8192 --pairs 3` to repeat this comparison.

### Prefix reuse on two-turn requests

Two pairs of fresh servers compared prefix reuse off and on at the same 8 GiB ceiling and 256-row prefill width, with coalescing on in both arms. Each pair replayed two fixed transcripts with 12 greedy completion tokens per turn. The reuse arm reserved 448 MiB for snapshots and their replacement. Both arms processed the same tokenized inputs and produced identical completions on all 16 requests.

| Follow-up | Reused tokens | Median first token | Paired change | Median attributed disk reads |
| --- | ---: | ---: | ---: | ---: |
| 61 tokens, code review | 44 | 10.24 → 5.27 s | −48.6% | 14.03 → 5.78 GB |
| 159 tokens, summarization | 142 | 11.85 → 4.85 s | −59.1% | 18.88 → 5.59 GB |

Saving the prefix increased median first-token time by 3–7% on the priming turns. Across both turns of both transcripts, total process-attributed disk reads fell from 126.12 to 84.16 GB and total request time fell from 128.38 to 104.62 seconds. The retained snapshot charge was 147 MiB for these short requests. The summarization pair's median process peak rose by about 213 MB. An initial 128 MiB reserve refused the snapshot. Flint allocates its first full-attention KV planes for 4,096 rows, even for short prompts. A restored snapshot carries one pending token, so the first suffix forward re-evaluates one of the tokens counted as reused. [All 16 requests](../../scripts/benchmarks/mac-m4-flint-prefix-448.csv)

Run the serving driver with `--expert-timing --prefix-reuse --working-set-mib 8192 --pairs 2` to repeat this comparison.

### Compact Flint KV at 128 MiB

With `FIFI_FLINT_MLX_COMPACT_KV=1`, each full-attention KV plane starts at 256 rows or the next power of two covering the first piece. The planes grow geometrically up to the context limit. The 513-row sliding-window planes keep their allocation. The budget still reserves the declared full context and the prefix store charges the bytes it actually retains.

Both arms enabled prefix reuse and reserved 128 MiB under the same 8 GiB device ceiling. The control refused each save. Compact KV retained about 42 MiB per snapshot, down from the 147 MiB charge with the original capacity. All eight completions matched. The follow-ups reused 44 and 142 tokens, reaching the first token in 5.36 and 4.99 seconds from 10.59 and 12.43 seconds. Across priming and follow-up, attributed SSD reads fell from 64.88 to 42.81 GB. The result compares one request per turn and arm, with a separate four-request smoke test. [Eight serving requests](../../scripts/benchmarks/mac-m4-flint-compact-kv-128.csv)

Run the serving driver with `--expert-timing --compact-flint-kv --working-set-mib 8192 --pairs 1` to repeat the comparison.

### Compact Flint KV across two growth boundaries

Two balanced pairs of fresh servers ran a 557-token prompt with 24 greedy completion tokens. Both arms used an 8 GiB ceiling, 256 admitted rows, prefill coalescing, and prefix reuse off. The three prompt pieces evaluated 255, 256, and 45 rows. The first logits call evaluated the pending prompt token. The compact full-attention planes consequently grew from 256 to 512, then 1,024 rows. The baseline stayed at 4,096 rows.

| Serving measurement | Baseline | Compact |
| --- | ---: | ---: |
| Median time to first token | 33.39 s | 33.12 s |
| Median decode throughput | 1.735 tok/s | 1.750 tok/s |
| Median per-request decode forward p95 | 627 ms | 633 ms |
| Median process peak footprint | 2.220 GB | 2.115 GB |
| Median attributed SSD reads | 50.93 GB | 50.81 GB |

All four completions, logical expert bytes, and read counts matched. Each request contributed 23 decode forwards. The p95 measures model forwards after the first token. This workload shows about 104 MB less peak footprint and similar latency and SSD traffic. [Four serving measurements](../../scripts/benchmarks/mac-m4-flint-kv-growth-557.csv)

Run the Flint serving driver with `--compact-kv-growth --expert-timing --working-set-mib 8192 --pairs 2` to repeat the comparison.

### Flint KV growth during decode

The prompt renders to 1,017 tokens with the pinned tokenizer and leaves seven decode forwards before position 1,024. Both arms use an 8 GiB Metal ceiling, 256 admitted prefill rows, and the original shards with prefill coalescing. Expert caching and prefix reuse are off. Each fresh process runs one request with 24 greedy completion tokens. Two pairs alternate the order of the arms.

The qualification trace records the full-attention KV capacity after every successful forward. The compact arm is required to reach 1,024 rows during prefill and grow to 2,048 at position 1,024 in decode. The control keeps 4,096 rows. The driver compares that exact forward across both arms and records p95 separately for the seven preceding and 15 following decode forwards.

All four requests produced the same completion and read 65.45 GB of logical expert weights in prefill and 11.48 GB in decode. Both arms made 80,226 prefill and 58,374 decode read calls per request with zero Metal staging-page copy fallbacks. The worker scheduled four prefill pieces and produced five model forwards of 255, 256, 256, 249, and one row, including the pending prompt token.

| Serving measurement | Baseline | Compact |
| --- | ---: | ---: |
| Median time to first token | 53.74 s | 57.36 s |
| Median decode throughput | 1.460 tok/s | 1.375 tok/s |
| Median per-request decode forward p95, all 23 steps | 784 ms | 794 ms |
| Median p95 across the seven forwards before growth | 748 ms | 811 ms |
| Median forward at position 1,024 | 741 ms | 786 ms |
| Median p95 across the 15 forwards after growth | 870 ms | 783 ms |
| Median process peak footprint | 2.218 GB | 2.123 GB |
| Total attributed SSD reads | 149.93 GB | 150.22 GB |
| Total process page-ins | 109 | 561 |

At position 1,024 the compact forward took 81 ms and 8 ms longer in the two pairs. The timed expert read portion of that forward was within 2 ms in the first pair and 58 ms lower in the second. Before-growth tails also rose in both pairs, and after-growth tails fell in both. These p95 figures use nearest rank, which selects the slowest sample in each seven- or 15-forward window. Compact KV consistently reduced peak footprint by 91–98 MB, and throughput went in opposite directions across the two pairs. The trial uses one prompt and leaves the OS file cache in place. [Four serving measurements](../../scripts/benchmarks/mac-m4-flint-kv-decode-growth-1017.csv)

Build the server with `native-mlx,native-fault-qualification`. The driver validates the rendered token count through the local tokenizer before sending a completion.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/decode-growth" \
  --compact-kv-decode-growth --expert-timing --working-set-mib 8192 --pairs 2
```

### Flint KV growth at 2,048 rows

The pinned tokenizer renders the deeper prompt to 2,041 tokens. Both arms use the original checkpoint on the mini's internal SSD, an 8 GiB Metal ceiling, 256 admitted prefill rows, prefill read coalescing, and 24 greedy completion tokens. Expert caching and prefix reuse are off. One pair of fresh servers limits SSD traffic. The control runs first and starts at 4,096 full-attention rows. Geometric compact KV reaches 2,048 during prefill and grows to 4,096 on the eighth decode forward, at absolute position 2,048.

The trace measures the timed expert reads and the complete forward independently. At the growth step, the remaining elapsed time includes model work, KV growth, and synchronization outside the measured read intervals. The driver also records expert-read and whole-forward p95 for the 23 decode forwards and for the seven- and 15-forward windows around the growth step.

Both continuations, logical read bytes, and slice counts matched. Each request scheduled eight prefill pieces and produced nine model forwards, including the one-row pending prompt token. Expert staging read 126.44 GB logically in prefill and 11.48 GB in decode. Both arms reported zero staging-page copy fallbacks.

| Serving measurement | Baseline | Compact |
| --- | ---: | ---: |
| Time to first token | 107.90 s | 104.49 s |
| Decode throughput | 1.47 tok/s | 1.76 tok/s |
| Decode forward p95, all 23 steps | 752 ms | 700 ms |
| Expert-read p95, all 23 steps | 545 ms | 517 ms |
| Full forward at position 2,048 | 719 ms | 700 ms |
| Expert reads at position 2,048 | 496 ms | 517 ms |
| Elapsed time outside expert reads at position 2,048 | 223 ms | 183 ms |
| Forward p95 in the 15 steps after growth | 762 ms | 613 ms |
| Expert-read p95 in the 15 steps after growth | 570 ms | 475 ms |
| Process peak footprint | 2.216 GB | 2.125 GB |
| Attributed SSD reads | 136.09 GB | 135.59 GB |
| Process page-ins | 44 | 65 |

Compact KV used 91 MB less peak process memory. Timing at position 2,048 moved in opposite directions for the whole forward and the expert reads. The lower post-growth forward tail coincided with lower expert-read tails. Both runs spent about 86.6 seconds in timed prefill expert reads. The single pair used one prompt and the OS file cache stayed in place, so the latency and I/O comparisons are specific to these runs. [Two serving measurements](../../scripts/benchmarks/mac-m4-flint-kv-decode-growth-2041.csv) · [All 46 decode forwards](../../scripts/benchmarks/mac-m4-flint-kv-decode-forwards-2041.csv)

Build the server with `native-mlx,native-fault-qualification`. The trial used the qualification build from commit `4ed2b37d1c114f8ee92f1e39ed859b09326273df`. The driver verifies the tokenizer count before starting inference.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/decode-growth-2048" \
  --compact-kv-decode-growth --growth-position 2048 --expert-timing \
  --working-set-mib 8192 --pairs 1
```

### Request-sized Flint KV

With compact KV and a Metal working-set budget enabled, `FIFI_FLINT_MLX_REQUEST_KV=1` sets the initial full-attention capacity for a fresh request with an admitted generation limit. The expected fed rows are the prompt length plus the generation allowance minus the final pending token. The shared compact capacity policy rounds those rows up to a power of two with a 256-row floor, capped by the admitted context. The startup plan still reserves the full context. The first forward allocates each plane on demand, and the original growth path remains available. Requests with prefix boundaries or an unbounded generation allowance keep geometric growth and its existing snapshot charges.

Both arms used compact KV on the same 1,017-token prompt from the decode-growth trial, with 24 greedy completion tokens and an 8 GiB ceiling. The trace requires geometric growth from 256 to 512, 1,024, and 2,048 rows in the control. The request-sized arm must start at 2,048 rows and hold that capacity across prefill and decode. Both arms leave the sliding-window capacity at 513 rows.

The two balanced pairs matched all four greedy outputs, logical expert bytes, and slice read counts. Both arms reported zero Metal staging-page copy fallbacks. The pre-sized arm held 2,048 rows throughout and eliminated all three measured growth events.

| Serving measurement | Geometric growth | Request-sized |
| --- | ---: | ---: |
| Median time to first token | 58.39 s | 54.67 s |
| Median decode throughput | 1.635 tok/s | 1.460 tok/s |
| Median per-request decode forward p95 | 683 ms | 1,031 ms |
| Median forward at position 1,024 | 695 ms | 796 ms |
| Median timed expert reads at position 1,024 | 527 ms | 637 ms |
| Median process peak footprint | 2.110 GB | 2.162 GB |
| Total attributed SSD reads | 147.07 GB | 147.40 GB |
| Total process page-ins | 327 | 48 |

First-token latency moved in opposite directions across the two pairs. Decode p95 went from 637 to 1,320 ms in the first pair and 730 to 743 ms in the reverse-order pair. The slowest request-sized forward took 2.22 seconds, including 2.07 seconds in timed expert reads. Peak footprint rose by 39–66 MB. The sample covers one prompt and leaves the OS file cache in place. The flag remains experimental. [Four serving measurements](../../scripts/benchmarks/mac-m4-flint-request-kv-1017.csv)

A two-turn check ran both configurations with prefix reuse and a 128 MiB reserve. Each 16-token priming request asked for 256 output tokens and stopped after three. Both kept 256 full-attention rows through the prime and follow-up. The 33-token follow-ups reused 15 tokens, reported the same 44,111,872-byte snapshot charge, and returned the same completions.

Rebuild the qualification server and run the same driver with the request-sized mode.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/request-kv" \
  --compact-kv-request --expert-timing --working-set-mib 8192 --pairs 2
```

### Request-aware Flint prefill

With Metal budgeting and adaptive prefill enabled, `FIFI_FLINT_MLX_REQUEST_PREFILL=1` prices full-attention scratch from the admitted prompt span. The full-context KV, prefix, cache, pressure and layer-growth charges stay in the startup reserve. Every candidate still pays its full expert staging charge. The worker can admit a wider piece up to the configured ceiling. Prefix-marked jobs keep their existing tiling, and chained prefill uses the startup plan.

The serving comparison uses the 2,041-token prompt and 24 greedy completion tokens, geometric compact KV, an 8 GiB Metal ceiling and prefill expert read coalescing in both arms. Prefix reuse and the expert cache are off. The driver validates the effective flags and equality of the startup memory plan, then records actual prefill pieces, expert traffic and sampled token IDs. Changing the chunk width can change floating-point results, so the driver records the number of leading tokens that match as well as the output hash.

The control used eight 256-row pieces. The request-sized workspace admitted 1,024 rows and used two pieces. The actual prefill forwards were 255 rows, six 256-row forwards, 249 rows and one pending token in the control, and 1,023, 1,017 and one in the request-aware arm. The startup memory charges, output hash and all 24 sampled token IDs matched. Both arms reported zero staging-page copy fallbacks.

| Serving measurement | Startup width | Request-aware width |
| --- | ---: | ---: |
| Time to first token | 108.21 s | 31.40 s |
| Timed prefill expert reads | 93.83 s | 23.22 s |
| Logical prefill expert reads | 126.44 GB | 38.87 GB |
| Prefill slice read calls | 166,680 | 22,239 |
| Attributed SSD reads for the request | 135.63 GB | 45.98 GB |
| Process peak footprint | 2.121 GB | 2.297 GB |
| Decode forward p95 | 616 ms | 773 ms |
| Request wall time | 120.48 s | 46.49 s |

The wider pieces reduced logical prefill bytes by 69%, total attributed SSD bytes by 66%, and first-token latency by 71%. Peak footprint rose by 176 MB. The fixed-order pair used one prompt, with sampler tracing on in both arms. It also recorded a slower decode tail with the wider prefill, so the option remains experimental. [Two serving measurements](../../scripts/benchmarks/mac-m4-flint-request-prefill-2041.csv)

The routed banks hold at most 12,032 distinct layer-expert records. The measured prefill read 71,458 and 21,967 record-equivalents respectively, including the pending token. That gives approximate lower bounds of **5.94 and 1.83 logical reads per distinct record**. Exact expert identities across pieces would let us measure how much additional reuse a bounded per-layer schedule could capture. [TokenWing's layer-major prefill and waves](https://github.com/runmyai/tokenwing/blob/main/docs/how-it-works.md#11-prefill)

Build the server with `native-mlx,native-fault-qualification` and run one pair on the pinned local checkpoint.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/request-prefill" \
  --request-prefill --expert-timing --working-set-mib 8192 --pairs 1
```

### Exact expert reuse across Flint prefill pieces

A second 2,041-token pair enabled the existing qualification route trace on the internal SSD. The server records the sorted expert IDs for each routed layer and forward. The analyzer verifies the row sequence and checks each forward's logical bytes against its expert count before archiving the prefill routes. Output and decode timing come from the two serving requests with tracing enabled.

The replay uses unions of expert IDs separately for each layer over contiguous groups of recorded pieces. It charges one read per layer-expert per group, with the pending one-row prompt forward kept separate. The distinct-record count includes that pending row. Each group reports the storage for two BF16 hidden-state buffers, one group-wide BF16 routed-output buffer in token and rank order, the largest layer's staging charge, and how many 64-slot waves that layer would require. These are storage and logical-read estimates. A serving prototype still needs to account for attention and other activation scratch, preserve state and accumulation order, and measure output and decode tails under the new schedule.

Both arms reproduced the preceding benchmark's 24 sampled token IDs, prefill widths and logical read counts. The traced requests attributed 136.02 and 45.61 GB to SSD reads and reached their first token in 126.16 and 36.02 seconds. Measured decode forward p95 rose from 822 to 1,447 ms in this fixed-order traced pair. The slowest wider-arm decode forward spent 1.48 of 1.67 seconds in expert reads. [Captured prefill routes and serving counters](../../scripts/benchmarks/mac-m4-flint-prefill-routes-2041.jsonl.gz)

The eight-piece control read **71,458 records for 11,407 distinct layer-experts**. The two-piece arm read **21,967 records for 11,426 distinct layer-experts**. Every one of the 47 routed layers repeated reads in both captures. Each layer in the wider arm used between 222 and 253 distinct experts across the prompt. The replay holds each arm's own recorded routes fixed.

| Observed piece width | Pieces per replay group | Largest group | Logical prefill reads | Reads per distinct record |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 256 rows | 126.44 GB | 6.264 |
| 256 | 2 | 512 rows | 69.92 GB | 3.464 |
| 256 | 4 | 1,023 rows | 38.79 GB | 1.922 |
| 256 | 8 | 2,040 rows | 20.67 GB | 1.024 |
| 1,024 | 1 | 1,023 rows | 38.87 GB | 1.923 |
| 1,024 | 2 | 2,040 rows | 20.70 GB | 1.024 |

Grouping the two wide pieces saves **18.17 GB of logical reads in replay**. The largest layer has 253 records in that group. Staging them all at once charges 854 MiB using the existing two-copy estimate. Four waves of up to 64 records would bound that charge at 216 MiB. Two BF16 hidden buffers and one group-wide routed-output buffer together account for 63.75 MiB at 2,040 rows. The routed-output buffer keeps token and expert rank order for the existing weighted reduction. [All six replay cases](../../scripts/benchmarks/mac-m4-flint-prefill-reuse-2041.csv)

Reproduce with the request-aware prefill command above and `FIFI_FLINT_EXPERT_TRACE=1` in the benchmark process environment. Capture or replay the routes with the same analyzer.

```sh
python3 scripts/mac_cache_study/flint_prefill_reuse.py "$SCRATCH/request-prefill" \
  "$SCRATCH/prefill-reuse.csv" --archive "$SCRATCH/prefill-routes.jsonl.gz"
python3 scripts/mac_cache_study/flint_prefill_reuse.py \
  "$SCRATCH/prefill-routes.jsonl.gz" "$SCRATCH/prefill-reuse-replay.csv"
```

### Bounded two-piece Flint layer fixture

The fixture loads one routed layer at a time from the pinned Apple affine checkpoint. It uses the real weights for sliding-window layer 35 and full-attention layer 37. Deterministic BF16 inputs have the same 1,023- and 1,017-row shapes as the wider serving run. Attention advances one KV plane in piece order. The real router runs separately for each piece, then all staging schedules consume those same activations and route choices.

The reference stages the selected experts independently per piece and uses the existing grouped projections. The shared stage reads the union once and runs the same grouped projections at the original widths. The wave arm stages up to 64 experts at a time and evaluates only pairs belonging to that wave. It restores token and expert rank order before the existing BF16 weighted reduction. Every arm checks that its pages retire before the next read. The fixture checks stage counts, exact record bytes, zero page-copy fallbacks and finite BF16 outputs with the original piece shapes. It compares each layer's outputs bit for bit.

The test measures the MLX allocator peak during attention and routing, the allocator peak during expert work and output reconstruction, and the maximum staging mapping. It admits those components under an explicit device budget. The MLX cache limit is zero for the fixture. File-cache state stays in place across the three schedules.

| Layer | Expert schedule | Expert records | Logical reads (MB) | Slice reads | Peak mapping (MiB) | Expert-phase MLX peak (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 35, sliding window | Separate pieces | 511 | 904.20 | 27 | 432.03 | 601.57 |
| 35, sliding window | Shared stage | 256 | 452.98 | 9 | 432.05 | 601.61 |
| 35, sliding window | 64-expert waves | 256 | 452.98 | 36 | 108.02 | 198.40 |
| 37, full attention | Separate pieces | 512 | 905.97 | 18 | 432.03 | 604.53 |
| 37, full attention | Shared stage | 256 | 452.98 | 9 | 432.05 | 604.58 |
| 37, full attention | 64-expert waves | 256 | 452.98 | 36 | 108.02 | 202.25 |

Both layers produced bitwise-identical output with each schedule at full width and in the 32- and 28-row smoke run. The shared stage halved logical read bytes. Four waves delivered the same byte savings with one quarter of the peak staging mapping and more slice reads than the shared stage. The mapping and allocator columns measure separate interfaces to unified memory and can overlap. The fixture measures layer output, logical reads, and staging memory. The opt-in serving trial will measure physical reads and latency. [All 12 measurements](../../scripts/benchmarks/mac-m4-flint-prefill-reuse-layer-fixture.csv)

From `engine/`, build the selection before reserving the GPU, then set the pinned checkpoint and device ceiling. Run with `FIFI_FLINT_PREFILL_REUSE_SMOKE=1` first to verify the 32- and 28-row case. The full fixture uses the same command with that switch set to zero.

```sh
cargo test --locked --offline --release --features native-mlx,native-fault-qualification --lib --no-run
unset FIFI_FLINT_EXPERT_RECORDS FLINT_QUAL_STAGE_SYNC FLINT_QUAL_LAYER_CAPTURE
FLINT_MODEL_DIR="$FLINT_ARTIFACT" FLINT_QUAL_GO=parent-owned-gpu-window \
FIFI_FLINT_PREFILL_REUSE_SMOKE=1 FIFI_FLINT_EXPERT_TIMING=1 FIFI_FLINT_MLX_PREFILL_CHAIN=0 \
FIFI_MLX_AUTO_BUDGET=0 FIFI_MLX_WORKING_SET_MIB=8192 FIFI_MLX_POSITIONED_READS=0 \
cargo test --locked --offline --release --features native-mlx,native-fault-qualification --lib \
  family::flint::mlx_forward::prefill_reuse_fixture::two_piece_prefill_reuse_preserves_layer_output \
  -- --ignored --exact --nocapture --test-threads=1
```

### Opt-in two-piece Flint serving

`FIFI_FLINT_MLX_PREFILL_PAIR=1` arms the shared-stage path for disk-backed Flint with the shared-page pipeline, adaptive device budgeting, and request-aware prefill. It groups a fresh prompt when the normal worker plan has exactly two multi-row forwards and no prefix boundary. The memory admission covers both attention workspaces, both pieces' activation buffers, and one full expert stage. A declined group leaves the session untouched and the worker feeds its pieces separately. Cancellation is checked before the group is dispatched.

Each layer appends the first piece's KV rows before attending the second piece. The full-attention planes allocate the group's final capacity before the first append. Each attention call uses its original piece width and visible KV span. The router selects experts independently for each piece. Their union is staged once, and each piece uses the existing sorted expert projections and weighted reduction at its original width. The stage retires before the next read. The session keeps the final prompt token pending for the first decode forward.

The comparison uses the same binary, pinned checkpoint, two-piece geometry, greedy sampler, and 8 GiB Metal ceiling in both arms. It checks worker piece reports, all sampled token IDs, request output hashes, and logical decode traffic. The smoke also covers a 16-token request that takes the regular path with the switch armed. The serving trial uses a single stage per layer, capped at 256 experts. The bounded layer fixture above records the smaller mapping and additional reads from 64-expert waves.

On the 16 GB M4 mini, two pairs alternated the order of the arms, with a fresh server for every run. The OS file cache stayed in place. The 2,041-token prompt kept its 1,023- and 1,017-row prefill forwards and one pending token. All four requests matched the same output hash and all 24 sampled token IDs. The shared path needed 94 page imports including the pending token, down from 141, and both arms reported zero staging-copy fallbacks.

| Serving measurement | Separate stages | Shared stage |
| --- | ---: | ---: |
| Time to first token | 28.06 s | 16.18 s |
| Timed prefill expert reads | 21.04 s | 10.49 s |
| Logical prefill expert reads | 38.87 GB | 20.70 GB |
| Prefill slice read calls | 22,239 | 8,037 |
| Attributed SSD reads for the request | 45.95 GB | 27.56 GB |
| Process peak footprint | 2.297 GB | 2.489 GB |
| Decode forward p95 | 621 ms | 621 ms |
| Request wall time | 40.90 s | 28.98 s |

The table reports medians of two runs per arm. Logical prefill traffic matched the earlier route replay exactly. First-token latency fell 42% and attributed SSD traffic fell 40%. Peak footprint rose by 192 MB. Decode p95 ranged from 615 to 626 ms over the four requests, and all four read 11.48 GB of logical expert bytes during decode. The 557-token smoke also matched sampled tokens and output while reducing logical prefill reads from 28.06 GB to 19.35 GB. The path remains opt-in. [Serving and smoke measurements](../../scripts/benchmarks/mac-m4-flint-prefill-pair-serving.csv)

The 557-token request also completed at a 4,000 MiB ceiling with the option armed. The remaining workspace was 1,058,326,976 bytes against a group charge of 1,088,723,840 bytes. The worker kept the separate 511- and 45-row forwards and one pending token, with the expected 141 page imports, logical read bytes, and all four sampled tokens.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/prefill-pair" \
  --prefill-pair --expert-timing --working-set-mib 8192 --pairs 2
```

### Bounded expert waves in Flint serving

`FIFI_FLINT_MLX_PREFILL_PAIR=auto` first checks admission for a single shared stage. If that charge exceeds the remaining workspace, it prices a 64-expert stage plus the buffers that restore the routed outputs to their original order. The worker falls back to separate pieces if neither charge fits. The `1` setting requests the single stage. The `waves64` setting requests waves directly for comparisons at a ceiling that fits both.

Each wave loads the next group from the sorted union of both pieces' expert IDs, then projects both pieces' pairs that selected those experts. The stage retires before the next read. The worker restores every projected pair to its original token and rank before the existing weighted reduction. The layer fixture exercises this same wave implementation with the real sliding-window and full-attention weights.

The tight comparison uses the 557-token prompt at 4,000 MiB with `1` and `auto`. It checks that the single-stage request runs separately and that `auto` selects waves. The wide comparison uses the 557- and 2,041-token prompts at 8 GiB with `1` and `waves64`. Both comparisons use the original piece widths, check output hashes and sampled tokens, and report the largest actual expert staging mapping separately from the process footprint.

At 4,000 MiB the mini reported 1.058 GB of remaining workspace. The two-piece plan charged 1.089 GB for the single stage and 0.437 GB for waves. All four requests matched the output hash and 24 sampled token IDs. The wave path used a peak staging mapping of 108.02 MiB. Logical expert reads fell from 28.06 to 19.35 GB, and slice reads fell from 40,392 to 14,904.

| Budget and prompt | Control | First token, control → waves | Attributed SSD reads | Peak footprint | Decode forward p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4,000 MiB, 557 tokens | Separate stages | 22.27 → 14.63 s | 35.83 → 26.65 GB | 2.151 → 1.877 GB | 624 → 634 ms |
| 8 GiB, 557 tokens | One shared stage | 13.82 → 14.51 s | 26.51 → 26.71 GB | 2.192 → 1.875 GB | 614 → 620 ms |
| 8 GiB, 2,041 tokens | One shared stage | 15.74 → 17.82 s | 27.00 → 26.96 GB | 2.327 → 1.875 GB | 606 → 606 ms |

Each row reports medians from two pairs with alternating control and wave order. Each arm starts a fresh server, and the operating system retains its file cache. In the 8 GiB arms the server handles the 557-token prompt before the 2,041-token prompt, so its lifetime peak footprint includes the earlier request. All 12 requests matched the control output hash and all 24 sampled token IDs for their prompt. Both arms at 8 GiB read the same logical bytes for each prompt. Waves cut the peak staging mapping from about 407–427 MiB to 108 MiB and increased prefill slice reads by 8% and 15% on the two prompts. Time to first token rose by 5% and 13% at 8 GiB. The `auto` policy keeps the faster single stage when that plan fits. [Serving and smoke measurements](../../scripts/benchmarks/mac-m4-flint-prefill-waves-serving.csv)

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/prefill-waves-tight" \
  --prefill-pair-waves tight --expert-timing --working-set-mib 4000 --pairs 2
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/prefill-waves-wide" \
  --prefill-pair-waves wide --expert-timing --working-set-mib 8192 --pairs 2
```

## Expert-major storage probe

A one-row Flint stage selects six experts. The published INT4 checkpoint stores gate, up and down payloads, scales and biases in separate tensors. That produces 54 slice reads for each routed layer. A contiguous record per expert can fill the nine destination ranges in one `preadv` call. Replaying six captured requests gives 324,864 slice reads for 6,016 one-row layer stages. Grouping existing consecutive slices reduces that count to 318,564. Reading one record per selected expert reduces it to 36,096. All three carry the same logical weight bytes. [Recorded routes](../../scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz)

On 18 November 2026, the M4 mini copied only Flint layer 24 into a temporary 432 MiB expert-major file on the internal SSD. Both arms filled identical destination buffers directly. They alternated order across 32 route pairs per cache mode, each reading 10,616,832 logical bytes from six experts. Every output buffer matched byte for byte. The timed interval covers file reads into the buffers. Model execution, Metal imports and conversion to the record layout are outside it.

| Read mode | Median slice reads | Median expert records | Paired time change | Attributed disk bytes across the 32 pairs |
| --- | ---: | ---: | ---: | ---: |
| Both file handles with `F_NOCACHE` | 12.72 ms | 4.17 ms | −67.1% | 310.2 → 339.7 MB |
| Both layouts warmed immediately before timing | 0.98 ms | 0.79 ms | −16.2% | 0 → 0 B |

The file counter reflects process-attributed storage reads. The uncached arm still had file-cache effects, including two pairs with zero source bytes attributed to the sliced layout. Among the 24 pairs where both arms attributed more than 90% of the logical bytes to storage, expert records reduced total attributed bytes by 2.3% and median paired time by 67.4%. A serving trial needs to compare cumulative traffic for warm, mixed and cold requests at the same memory ceiling. [All 128 measurements](../../scripts/benchmarks/mac-m4-flint-expert-records.csv)

A two-pair cleanup check immediately afterward found the original slices cached and the newly built record file still reading from storage under `F_NOCACHE`. Median time rose from 1.09 to 4.30 ms in those pairs. The buffered, immediately warmed comparison still favored the record layout.

### Bounded expert-record serving trial

The qualification server can use a 432 MiB sidecar for layer 24. The builder captures the local source shard identities and writes the nine roles of each expert consecutively. The loader checks those identities and the declared size at startup. During a single-row stage, the reader scatters each selected expert into the same Metal staging pages using one positioned vector read. The remaining layers and multi-row stages read the original shards. The serving driver enables positioned shard reads and prefill coalescing in both arms.

At the same 8 GiB ceiling, two pairs of fresh servers alternated order and each answered the same 16-token request twice with 12 completion tokens. The repeat ran immediately after the first request. The OS file cache stayed in place. All eight completions and logical bytes matched. Every layer-24 decode stage took six reads, down from 54. Total decode read calls per request fell from 27,918 to 27,390 and Metal page-copy fallbacks stayed at zero.

| Request | Layer-24 read time per decode step | Paired layer change | Paired full decode rate | Median attributed SSD reads |
| --- | ---: | ---: | ---: | ---: |
| First after startup | 8.39 → 3.67 ms | −56.3% | +2.5% | 8.03 → 7.59 GB |
| Immediate repeat | 8.00 → 3.65 ms | −54.4% | −0.8% | 7.08 → 7.18 GB |

Total attributed SSD reads were 30.22 GB with shard slices and 29.55 GB with expert records. Full-request timings varied across the first and repeat visits. The single-layer result supports faster reads into staging pages and leaves the serving impact of wider use open. [Eight serving measurements](../../scripts/benchmarks/mac-m4-flint-layer24-records-serving.csv)

Build the server with `native-mlx,native-fault-qualification` and use the established Flint serving driver. The sidecar stays in scratch beside the benchmark inputs.

```sh
python3 scripts/mac_cache_study/build_flint_layer24_records.py \
  "$FLINT_ARTIFACT" "$SCRATCH/layer-24.records"
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/records-serving" \
  --expert-records "$SCRATCH/layer-24.records" --expert-timing \
  --working-set-mib 8192 --pairs 2
```

### Four-layer Flint serving trial

The sidecar reader accepts a local directory with up to four routed layers. Each file binds one layer to the same source shard identities and has a fixed record width and exact file size. A four-file set adds 1.69 GiB on disk. Both arms open and validate the same four files. The control reads expert weights from the original shards and the candidate reads the sidecars only for single-row stages in the selected layers. Both use the same shared Metal staging path and leave the expert cache disabled.

We ranked layers by hits in a 12-expert LRU per layer over 96 single-row stages from the published trace, with both arithmetic requests excluded. We added the least-reused layer from each depth range to the existing layer 24. Those were layer 2 for layers 2–16, layer 30 for layers 17–32, and layer 47 for layers 33–48. Their hit rates were 4.34%, 15.62%, and 15.62%. Layer 24 had a 19.97% hit rate. The serving trial used the excluded arithmetic prompt. [Published trace](../../scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz)

At the 8 GiB ceiling, the worker admitted 256 prefill rows and exercised all four sidecars on each single-row stage. The eight output hashes matched, with 5.16 GB of logical prefill reads and 5.49 GB of logical decode reads per request in both arms. Decode read calls fell from 27,918 to 25,806 per request and Metal page-copy fallbacks stayed at zero.

| Request | Selected layer read time per decode step | Decode tokens/s | Per-request decode forward p95 | Attributed SSD reads |
| --- | ---: | ---: | ---: | ---: |
| First after startup | 52.84 → 18.78 ms | 1.325 → 1.345 | 819.94 → 850.17 ms | 10.01 → 9.90 GB |
| Immediate repeat | 53.80 → 18.71 ms | 1.280 → 1.370 | 870.44 → 794.54 ms | 9.96 → 9.96 GB |

The table shows medians for each arm and visit. The p95 uses nearest rank over 11 decode forwards per request. Across all visits, attributed SSD reads totaled 39.95 → 39.72 GB and process page-ins rose from 155 to 206, with individual runs ranging from 0 to 86. Process peak stayed near 1.82 GB. The selected reads took about 65% less time, with a 7% increase in repeat decode throughput and a 3.7% increase in first-request p95. The mixed tails, extra page-ins and storage cost keep this path limited to qualification builds. [Eight serving measurements](../../scripts/benchmarks/mac-m4-flint-four-records-serving.csv)

Build the server and run the same two-pair comparison as above. Each process answers the 16-token prompt twice with 12 completion tokens. The OS file cache stays in place. Keep the sidecars on the same local volume as the verified checkpoint so the comparison uses the same SSD.

```sh
mkdir "$SCRATCH/records"
for layer in 2 24 30 47; do
  python3 scripts/mac_cache_study/build_flint_layer24_records.py \
    "$FLINT_ARTIFACT" "$SCRATCH/records/layer-$layer.records" --layer "$layer"
done
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$FLINT_ARTIFACT" "$SCRATCH/proc-io" "$SCRATCH/four-records-serving" \
  --expert-records "$SCRATCH/records" --expert-timing --working-set-mib 8192 --pairs 2
```

### Reproduce the expert record probe

Run on Apple Silicon with Python `os.preadv` support. Use the verified local Flint checkpoint and a scratch directory on the same volume as its shards. The probe releases the temporary record file after measurement and retains the CSV.

```sh
clang -O2 scripts/mac_cache_study/proc-io.c -o "$SCRATCH/proc-io"
python3 scripts/mac_cache_study/flint_expert_records.py "$FLINT_ARTIFACT" \
  scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz \
  "$SCRATCH/proc-io" "$SCRATCH/expert-records" 32
```

## Reproduce

From the repository root, build the public device probe in scratch.

```sh
xcrun swiftc -O scripts/mac_cache_study/coreml-devices.swift -o "$SCRATCH/coreml-devices"
"$SCRATCH/coreml-devices"
```

From `engine/`, run the six ANE checks and the one ignored latency probe.

```sh
cargo test --locked --release --features native-ane --test decision_ane -- --test-threads=1
cargo test --locked --release --features native-ane --test decision_ane \
  ane_idle_decision_latency -- --ignored --exact --nocapture --test-threads=1
```

The Flint driver keeps the shared-page pipeline on and varies only read coalescing. Run it during an idle window and use a fresh scratch output directory. The [pipeline study](mac-flint-expert-pipeline.md#reproduce) shows how to build `ENGINESRV` and `PROC_IO`.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR" --read-runs
```

To reproduce the prefill-only comparison, build the release server from `engine/` with `cargo build --locked --release --features native-mlx,native-fault-qualification --bin enginesrv`. Use the verified internal checkpoint and a fresh output directory. Add `--smoke` for the short parity check.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR" --expert-timing --prefill-read-runs
```
