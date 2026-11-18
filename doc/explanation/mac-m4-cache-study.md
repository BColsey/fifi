# Cache study on the 16 GB M4 Mac mini

Measured on 17 November 2026 on a Mac16,10. The live server ran commit `6c2a1e19d041eb55232ed0c73105656063dd99db` with the published Flint INT4 checkpoint at revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`. All 22,579,847,971 downloaded bytes were checked against the pinned release manifest. Metal reported an 11.84 GiB working set. Flint chose disk-backed experts and a 512-row prefill ceiling at a configured context of 32,768 tokens. It became ready in 2.04 seconds and returned `4` for a live arithmetic request. [Pinned Flint release](https://huggingface.co/RecursiveSI/flint-int4/tree/3b93f9053b530488405ae9fca51d4900d6ce21f0)

## M4 hardware caches

macOS reports a 128-byte cache line, four performance cores with 128 KiB of L1 data per core and 16 MiB of L2 shared by four cores, and six efficiency cores with 64 KiB of L1 data per core and 4 MiB of L2 shared by six cores. The L3 or system-level cache capacity remains unverified.

The benchmark follows a dependent pointer chain with one node per cache line. In the global shuffle, the next node can be anywhere in the working set. The page-local variant shuffles pages and visits a shuffled sequence of lines within each page. That reduces translation pressure and changes prefetch opportunities. The QoS classes are scheduler hints. Each figure is the median of three runs at that size after a full warm traversal.

| Working set | Interactive, global shuffle | Interactive, page-local | Background, page-local |
| ---: | ---: | ---: | ---: |
| 32 KiB | 0.68 ns | 0.68 ns | 1.63 ns |
| 64 KiB | 0.70 ns | 0.68 ns | 1.79 ns |
| 80 KiB | 0.68 ns | 0.68 ns | 7.11 ns |
| 128 KiB | 0.77 ns | 0.68 ns | 8.12 ns |
| 160 KiB | 4.59 ns | 4.03 ns | 8.37 ns |
| 4 MiB | 7.24 ns | 4.34 ns | 22.63 ns |
| 8 MiB | 10.30 ns | 4.40 ns | 111.88 ns |
| 16 MiB | 31.82 ns | 9.74 ns | 102.58 ns |
| 32 MiB | 89.34 ns | 10.74 ns | 132.41 ns |
| 64 MiB | 97.26 ns | 22.75 ns | 146.78 ns |
| 128 MiB | 100.69 ns | 27.67 ns | 140.30 ns |

The L1 boundaries are visible above 64 KiB in the background runs and above 128 KiB in the interactive runs. The page-local interactive run stays near 4.4 ns through 8 MiB and slows around 14–18 MiB, consistent with the reported performance-cluster L2. Beyond L2, traversal choice changes the result substantially. The interactive global shuffle is already near 90 ns at 32 MiB, while the page-local case rises from about 14 ns at 48 MiB to 23 ns at 64 MiB. These measurements give a useful latency range. Attributing the larger transition to a particular system-cache capacity requires counter or additional placement evidence. They describe dependent CPU reads and leave Metal's cache geometry unmeasured. [All 408 hardware samples](../../scripts/benchmarks/mac-m4-cache-latency.csv)

## Flint expert weights

Here L1 means expert tensors already resident in MLX, L2 means host-memory service through packed buffers or the macOS file cache, and L3 means service from the SSD. Flint currently releases its staged expert tensors after each layer invocation. It retains attention, routers, shared experts, and the output head. The operating system can retain checkpoint pages.

### Live request baseline

The production disk-backed path handled three identical streaming requests with prefix reuse disabled. Each used 16 prompt tokens and consumed a 12-token completion budget. The generated deltas matched exactly. The per-process measurements use macOS `proc_pid_rusage` before and after each request. They count disk reads attributed to the process, including any read-ahead, and cannot determine the fraction of expert requests that hit a cache by themselves. [Apple's process counter interface](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.h)

| Repetition | Attributed disk reads | Time to first token | Decode tokens/s |
| ---: | ---: | ---: | ---: |
| 1 | 7.77 GB | 7.64 s | 1.62 |
| 2 | 6.41 GB | 6.92 s | 1.77 |
| 3 | 5.75 GB | 6.87 s | 1.86 |

These repeats started after two smoke requests. The system was already warm, yet the third repeat still registered 5.75 GB of storage reads. The server's footprint after the three requests was about 1.75 GB. The separate completed arithmetic request used 48 completion tokens and reached its visible answer in 34.42 seconds. [Live request samples](../../scripts/benchmarks/mac-m4-flint-cache-live.csv)

### File cache and SSD

Each of 24 paired trials reads the same six unique experts from one real Flint layer. The three projections have payload, scale and bias tensors, for 54 reads and 10,616,832 logical bytes per trial. The benchmark stops after the file reads. The output bytes matched in every pair.

| Read path | Samples | Median time | Median attributed disk reads |
| --- | ---: | ---: | ---: |
| Buffered file descriptor, immediately warmed | 24 | 1.09 ms | 0 B |
| `F_NOCACHE` first, positive disk-read counter | 11 | 11.49 ms | 9.55 MB |

All twelve `F_NOCACHE` samples run after the paired warm read reported zero disk-read bytes. One of twelve samples with `F_NOCACHE` first also reported zero. The positive-counter subset therefore measures storage-backed service with mixed hits and read-ahead. The 10.5× ratio between medians characterizes these sampled read paths. [All 48 file-read samples](../../scripts/benchmarks/mac-m4-flint-file-reads.csv)

### Resident MLX weights, host-packed weights, and warm file reads

A separate Metal test used layer 24 of the published checkpoint, the same BF16 input rows, and the same distinct expert IDs in every path. The resident path reused the staged Metal tensors and local IDs. The host-packed path imported preassembled packed weights and IDs for each invocation. The warm-file path exercised the production disk-backed code after a warm-up, including reading back the selected IDs and assembling the requested slices. Each arm evaluated the grouped projection. Output readback happened outside the timing window. The benchmark rotated arm order and matched output bytes in all 81 trials.

| Rows | Distinct experts | Reused Metal tensors | Host-packed import | Warm file path |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 0.788 ms | 3.700 ms | 4.317 ms |
| 4 | 24 | 1.530 ms | 11.761 ms | 16.359 ms |
| 16 | 96 | 3.502 ms | 48.265 ms | 65.469 ms |

This isolates the cost of staging one layer. The live server still reads its routed weights from local shards and stages them for each invocation. Both host-backed and Metal-resident caches draw from the same unified memory, so their capacity must share the serving budget. [All 81 Metal samples](../../scripts/benchmarks/mac-m4-flint-cache-staging.csv)

The next experiment is to trace layer and expert IDs across representative prompts and replay that order against a byte-bounded resident cache. Use the traces to compare hit rates and evicted bytes at several capacities. The measured round trips and staging costs can then be compared with end-to-end first-token latency, decode throughput and peak memory.

## Reproduce

Use the same published checkpoint and release build. Run the hardware sweep between GPU and file-read runs. For the live and file-read drivers, copy the scripts into `cache-study/` beside the server's `api-key`, `server.pid`, `server.log`, and `checkpoint/`. Keep the API key in a local file with mode 0600. The C I/O probe must run as the same user as the server.

```sh
xcrun clang -O3 -Wall -Wextra -Werror dependent-read.c -o dependent-read
xcrun clang -O2 -Wall -Wextra -Werror proc-io.c -o proc-io
for qos in interactive background
do
  for order in global_shuffle page_local
  do
    ./dependent-read "$qos" "$order"
  done
done
python3 live-repeats.py
python3 expert-file-reads.py
```

From `engine/`, substitute the absolute path to the pinned local checkpoint. The test runs one layer, verifies each output byte, and reports 81 timing rows with MLX allocation and synchronization counters.

```sh
FIFI_FLINT_CACHE_MODEL=/path/to/pinned/flint-int4 cargo test --locked --release \
  --features native-mlx,native-fault-qualification --lib \
  family::flint::weights::cache_bench::flint_mlx_expert_cache_tiers \
  -- --ignored --exact --nocapture --test-threads=1
```

[Benchmark sources](../../scripts/mac_cache_study) · [Flint staging benchmark](../../engine/src/family/flint/weights_cache_bench.rs)
