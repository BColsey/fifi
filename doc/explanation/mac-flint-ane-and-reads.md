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
