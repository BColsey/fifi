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

## Expert-major storage probe

A one-row Flint stage selects six experts. The published INT4 checkpoint stores gate, up and down payloads, scales and biases in separate tensors. That produces 54 slice reads for each routed layer. A contiguous record per expert can fill the nine destination ranges in one `preadv` call. Replaying six captured requests gives 324,864 slice reads for 6,016 one-row layer stages. Grouping existing consecutive slices reduces that count to 318,564. Reading one record per selected expert reduces it to 36,096. All three carry the same logical weight bytes. [Recorded routes](../../scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz)

On 18 November 2026, the M4 mini copied only Flint layer 24 into a temporary 432 MiB expert-major file on the internal SSD. Both arms filled identical destination buffers directly. They alternated order across 32 route pairs per cache mode, each reading 10,616,832 logical bytes from six experts. Every output buffer matched byte for byte. The timed interval covers file reads into the buffers. Model execution, Metal imports and conversion to the record layout are outside it.

| Read mode | Median slice reads | Median expert records | Paired time change | Attributed disk bytes across the 32 pairs |
| --- | ---: | ---: | ---: | ---: |
| Both file handles with `F_NOCACHE` | 12.72 ms | 4.17 ms | −67.1% | 310.2 → 339.7 MB |
| Both layouts warmed immediately before timing | 0.98 ms | 0.79 ms | −16.2% | 0 → 0 B |

The file counter reflects process-attributed storage reads. The uncached arm still had file-cache effects, including two pairs with zero source bytes attributed to the sliced layout. Among the 24 pairs where both arms attributed more than 90% of the logical bytes to storage, expert records reduced total attributed bytes by 2.3% and median paired time by 67.4%. A serving trial needs to compare cumulative traffic for warm, mixed and cold requests at the same memory ceiling. [All 128 measurements](../../scripts/benchmarks/mac-m4-flint-expert-records.csv)

A two-pair cleanup check immediately afterward found the original slices cached and the newly built record file still reading from storage under `F_NOCACHE`. Median time rose from 1.09 to 4.30 ms in those pairs. The buffered, immediately warmed comparison still favored the record layout.

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
