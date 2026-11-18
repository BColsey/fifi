# Flint expert reuse on the M4 mini

On 17 November 2026, we traced completed expert stages on the 16 GB M4 Mac mini with the published Flint INT4 checkpoint at revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`. The trace came from a separate loopback server using automatic memory sizing, 32,768 positions and no prefix reuse. Its startup admitted 512-row prefill and placed routed experts on disk. The live server remained available on port 18791. [Prior cache study](mac-m4-cache-study.md)

## What the replay counts

An access is one distinct expert at one layer of an already batched stage. Each record is 1,769,472 logical bytes for the gate, up and down projections, including scales and biases. Expert IDs are local to their layer. The trace captures all 47 routed layers in each completed forward pass and validates the row total against prompt and completion token counts. No additional GPU readback is needed, since the serving path already brings the selected IDs to the host.

The replay starts cold and carries state across requests in order. It resolves the hits for the whole layer batch before admitting missed experts in ascending ID order. It compares one global LRU, evenly partitioned layer LRUs, and layer LRUs that admit misses only for single-row stages. The last policy lets broad prefill stages use and refresh existing hits. Cache capacity charges packed record bytes and excludes allocator alignment, bookkeeping and the staging buffers needed for a miss.

Single-row results include decode-shaped calls, including the pending prompt token. Multi-row results cover the chunked prefill for this workload. These hit rates and saved logical bytes model cache behavior. Device copy time, actual disk traffic, and request speed need a cache implementation and a separate A/B.

## Results

The six requests used 16–143 prompt tokens and generated 128 completion tokens in total. The workload covered arithmetic, explanation, code review and summarization, followed by two exact repeats. The output digests matched for both repeats. It captured 64,260 layer-expert accesses, including 36,096 single-row accesses. Across all requests, 11,136 of the checkpoint's 12,032 layer-expert records were touched. [Compressed trace](../../scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz) · [Full capacity and policy replay](../../scripts/benchmarks/mac-m4-flint-expert-cache-replay.csv)

| Per-layer LRU payload | Records | All hits | Single-row hits | Multi-row hits | Logical bytes avoided across six requests |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 MiB | 303 | 10.72% | 17.02% | 2.63% | 12.19 GB |
| 1 GiB | 606 | 15.97% | 24.01% | 5.67% | 18.16 GB |
| 1.5 GiB | 910 | 20.61% | 30.16% | 8.37% | 23.43 GB |
| 2 GiB | 1,213 | 23.57% | 33.66% | 10.64% | 26.80 GB |

Global LRU had zero hits through 256 MiB. Each one-row forward sweeps 47 layers and touches 282 experts, which already exceeds that global capacity. At 1 GiB, global LRU hit 12.26% overall and 21.62% on single-row stages. Restricting per-layer admission to single-row misses raised those figures from 15.97% to 16.13% and from 24.01% to 24.30%. The more selective policy's advantage was under 0.4 percentage points at 1–2 GiB. For this workload and range, ordinary per-layer LRU is a reasonable starting policy.

The prototype candidate is **1 GiB of packed weights in per-layer LRUs**, with 512 MiB and 2 GiB as comparison points. The serving plan left 2,228,718,016 bytes for a cache after reserving the admitted 512-row attention and staging workspace. A 2 GiB payload leaves about 77 MiB for additional cache memory under that plan. At 1 GiB, roughly 1.08 GiB of modeled headroom remains. Measure the real resident footprint, evictions, time to first token, and decode throughput in an A/B before choosing the default.

## Reproduce on the mini

From `engine/`, build the trace-enabled server with `cargo build --locked --release --features native-mlx,native-fault-qualification --bin enginesrv`. Run the capture commands from the repository root. The capture driver starts and stops its own server and uses port 18792 by default. Set `CHECKPOINT` to the published local Flint INT4 directory, `TRACE_BINARY` to the release server, and `TRACE_DIR` to a fresh scratch path. Run `--smoke` with a separate output directory for a short check. The trace contains expert IDs and output digests.

```sh
python3 scripts/mac_cache_study/flint_expert_trace.py self-test
python3 scripts/mac_cache_study/flint_expert_trace.py capture \
  "$CHECKPOINT" "$TRACE_BINARY" "$TRACE_DIR"
python3 scripts/mac_cache_study/flint_expert_trace.py replay \
  "$TRACE_DIR/requests.jsonl" "$TRACE_DIR/replay.csv"
```

The runner checks the required model placement, output agreement on both repeated prompts, full layer coverage and row counts. The round trips used for tracing are already present in the disk-backed serving path. The qualification build enables trace logging with `FIFI_FLINT_EXPERT_TRACE=1`. [Trace and replay driver](../../scripts/mac_cache_study/flint_expert_trace.py)
