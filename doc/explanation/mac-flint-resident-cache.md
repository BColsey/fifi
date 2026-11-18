# A resident expert cache for disk-backed Flint

Flint's automatic Metal working-set plan keeps the resident core in unified memory and reads routed expert weights from local safetensors when the full checkpoint exceeds the budget. With `FIFI_FLINT_MLX_EXPERT_CACHE_MIB=1024`, the disk-backed route also reserves 606 packed expert slots across its 47 routed layers. On the 16 GB M4 mini, the page-aligned banks reserve 1,073,070,080 bytes including the local route IDs. The startup log accounts for that reserve alongside resident weights, KV state and transient staging. Both measured modes admitted up to 512 prefill rows. An unset or zero cache budget keeps the cache off.

For each one-row forward, the GPU router supplies the exact expert IDs. The CPU reads the missing gate, up and down weights into each layer's least recently used slots, preserving all hits in the current route. Metal evaluates the grouped projection directly from the shared pages. The bank remains pinned until that evaluation completes. Multi-row prefill stages experts in transient buffers. The cache retains weights between requests and still imports page-backed tensors for each layer dispatch. [Prior cache replay](mac-flint-expert-reuse.md) · [Shared-page staging](mac-flint-expert-pipeline.md)

## Checkpoint comparison

The balanced comparison used four fresh benchmark servers on the 16 GB M4 mini in control, cached, cached, control order. Both arms used shared-page staging, no contiguous read coalescing, no prefix reuse, and 12 completion tokens for each of two prompts. The live Flint process stayed up and an idle guard checked its state during the runs. Both benchmark arms used the same verified external APFS SSD copy of Flint INT4 at revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`. The earlier overlapping-read and contiguous-read benchmarks used internal storage.

The table shows medians. Every response digest matched its control.

| Prompt tokens | Prefill, control → cached | Decode, control → cached | Request wall time, control → cached | Process disk reads, control → cached | Peak process footprint, control → cached |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 13.39 → 14.39 s | 0.93 → 0.76 tokens/s | 25.22 → 28.89 s | 9.87 → 10.46 GB | 1.83 → 2.81 GB |
| 45 | 24.34 → 24.57 s | 0.79 → 0.81 tokens/s | 38.28 → 38.28 s | 17.76 → 17.73 GB | 1.90 → 2.90 GB |

The short request took 14.6% longer, and the longer request finished in the same time. Both prompts used about 1 GB more peak memory with the cache. The process disk counter covers file-system read-ahead and caching effects. The trace replay estimates logical reads, while this run measured file-system reads and elapsed time. The 1 GiB setting stays experimental and default-off. [All eight requests](../../scripts/benchmarks/mac-m4-flint-resident-cache.csv)

The native regression suite passed 38 selected tests, including byte-for-byte grouped expert parity on reused slots. The oversized sparse checkpoint separately passed with the 1 GiB cache enabled.

## Internal SSD timing

The qualification build records completed forward passes for the disk-backed route. The reader counts exact checkpoint bytes requested and wall time inside the file reads. The importer times the host call that validates and wraps packed pages for MLX, and the native counter records any copy fallback. Forward time includes attention, routed and shared experts, and synchronization. File reads can overlap the shared expert on the GPU.

The A/B driver separates prompt forwards from decode using the reported token positions. The final prompt row belongs to prefill. It checks that every forward and routed layer appears in the counters and that output digests agree across modes.

On 18 November 2026, we hash-checked a durable internal APFS copy of the same Flint INT4 revision. The live API resumed on the internal copy after a verified external fallback and produced the same completion. The sweep ran separately while live Flint was idle. Eight fresh benchmark servers ran cache-off, 512 MiB, 1 GiB, and 2 GiB, then repeated the order in reverse. Each server answered two greedy prompts with 12 completion tokens each. Prefix reuse and contiguous-read coalescing were off throughout. All output digests matched, all modes admitted 512 prefill rows, and MLX reported zero page-copy fallbacks.

| Prompt tokens | Expert cache | Decode tokens/s | Logical decode reads | Time in decode file reads | Time in decode page imports | Full request | Peak footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | Off | 1.91 | 5.49 GB | 4.50 s | 38 ms | 12.22 s | 1.83 GB |
| 16 | 512 MiB | 1.58 | 4.44 GB | 5.02 s | 55 ms | 14.75 s | 2.28 GB |
| 16 | 1 GiB | 1.60 | 4.12 GB | 4.71 s | 59 ms | 14.08 s | 2.81 GB |
| 16 | 2 GiB | 1.41 | 3.78 GB | 4.78 s | 60 ms | 15.14 s | 3.89 GB |
| 45 | Off | 1.43 | 5.49 GB | 5.80 s | 56 ms | 21.02 s | 1.96 GB |
| 45 | 512 MiB | 1.72 | 4.52 GB | 4.68 s | 53 ms | 20.63 s | 2.50 GB |
| 45 | 1 GiB | 1.54 | 4.14 GB | 4.96 s | 56 ms | 21.68 s | 3.04 GB |
| 45 | 2 GiB | 1.44 | 3.53 GB | 4.47 s | 60 ms | 22.46 s | 4.11 GB |

The table gives medians from two runs per prompt and budget. With the cache off, checkpoint reads occupied 91–92% of prefill and 74–77% of decode forward wall time. Page imports occupied under 1% of decode in every run. The 2 GiB bank saved 31–36% of exact decode read bytes yet raised total request time on both prompts. The 512 MiB bank cut 45-token request time by 1.9% and raised 16-token request time by 20.7%. The process-wide disk-read counter, which includes prefill and file-system caching effects, increased in every cached median. The cache remains experimental and default-off. [All 16 requests and phase counters](../../scripts/benchmarks/mac-m4-flint-internal-expert-timing.csv)


## Held-out next-token read-ahead

We replayed the six previously captured requests, with all instances of one prompt family held out in each fold. Training included only the distinct prompts in the other three families. Predictions could use the preceding token's same-layer route and choose only experts absent from a demand-only 1 GiB LRU. Multi-row stages bypassed that cache. The first one-row stage of each request could warm the cache, and later stages supplied the 25,894 eligible demand misses.

| Cold read-ahead strategy | Picks per layer | Useful reads | Precision | Share of misses covered | Extra logical reads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform random expectation | 1 | 106 | 1.86% | 0.41% | 9.96 GB |
| Per-layer frequency | 1 | 598 | 10.43% | 2.31% | 9.09 GB |
| Previous-route transitions | 1 | 834 | 14.54% | 3.22% | 8.67 GB |
| Previous-route transitions | 2 | 1,551 | 13.52% | 5.99% | 17.55 GB |
| Previous-route transitions | 4 | 2,679 | 11.68% | 10.35% | 35.84 GB |

The most precise learned policy would start almost six unnecessary reads for every useful early read. These figures model exact packed record bytes and assume the predicted reads can finish before demand. The M4's Neural Engine measured a 131 µs median on the separate idle decision-head probe. Faster classifier execution would leave the observed prediction errors and their read traffic in place. [Per-family replay](../../scripts/benchmarks/mac-m4-flint-read-ahead.csv) · [Neural Engine probe](mac-flint-ane-and-reads.md)

## Reproduce

From `engine/`, build the release server with `cargo build --locked --release --features native-mlx --bin enginesrv`. Use the published Flint INT4 checkpoint and build the process counter as in the [pipeline study](mac-flint-expert-pipeline.md#reproduce). From the repository root, choose a fresh scratch output directory and an idle window. Add `--smoke` for the short comparison before the full run.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR" --cache-mib 1024
python3 scripts/mac_cache_study/flint_read_ahead.py --self-test
python3 scripts/mac_cache_study/flint_read_ahead.py \
  scripts/benchmarks/mac-m4-flint-expert-trace.jsonl.gz "$RUN_DIR/read-ahead.csv"
```

For the internal-storage attribution sweep, build with `cargo build --locked --release --features native-mlx,native-fault-qualification --bin enginesrv` from `engine/`. Point `CHECKPOINT` at the verified internal copy and use a fresh `RUN_DIR`. Run on an idle Mac.

```sh
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR" --expert-timing \
  --cache-mib 512 --cache-mib 1024 --cache-mib 2048
```
