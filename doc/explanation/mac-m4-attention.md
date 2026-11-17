# Q8 attention on a 16 GB M4 Mac mini

The BF16 follow-up on this mini found a repeatable decode latency benefit at 16,384 positions and a mixed result at 12,288. Padded flash consistently reduced dense-prefill temporary memory. Its latency benefit depended on prompt width and context depth.

The records in this document were measured by the PR author on his own M4 Mac mini; they are carried as contributor records and have not been reproduced here.

Measured 17 November 2026 on a 10-GPU-core M4, macOS 26.3, with the locked release build, Rust 1.98.1, Xcode 26.5 and `native-mlx`, on the author's Mac performance branch. The benchmark test and the dense-prefill reference correction are the ported commit `3b307c6d8` (`chore(mlx): measure Q8 decode dispatch and align the dense prefill oracle`). Both benchmarks used synthetic resident inputs and timed one attention layer through a synchronized host readback. Whole-model decode throughput, time to first token and generation quality still need a checkpoint run.

The [MLX architecture guide](mlx-architecture.md) describes the shared execution paths and the checks that fit on this mini.

## Initial Qwen3.6 decode baseline

One BF16 query, 16 query heads, two KV heads, head width 256, Q8 cache with group size 64. Cache capacity rounded up to 256 positions. Each depth ran in three fresh benchmark processes, with four warm-up pairs and 20 timed pairs per process. Each pair used the same query for both arms, varied the query between pairs and alternated the first arm. The times below are the median of the three per-process medians. The last column gives the range of within-process latency reductions.

| Cached positions | Composite (µs) | Fused (µs) | Fused latency compared with composite |
| ---: | ---: | ---: | ---: |
| 255 | 280 | 396 | 41–55% higher |
| 2,047 | 395 | 453 | 13–15% higher |
| 8,191 | 663 | 643 | 1–8% lower |
| 12,287 | 855 | 734 | 12–18% lower |
| 16,383 | 1,056 | 829 | 21–25% lower |
| 16,384 | 1,142 | 866 | 18–30% lower |
| 16,385 | 1,092 | 836 | 21–27% lower |
| 24,575 | 1,500 | 1,070 | 27–32% lower |
| 32,767 | 1,861 | 1,198 | 35–38% lower |
| 65,535 | 3,730 | 2,220 | 37–41% lower |

At 12,287 positions the fused path won 54 of 60 pairs. The two paths agreed within the existing BF16 tolerance on every timed pair, with a largest absolute difference of 0.015625. The separate 131,072-position test passed against the dequantized reference in both BF16 and F32. Host readback and dispatch are included in every sample. The repeated use of one layer's cache and the per-layer synchronization differ from a whole decoder step. [All 600 paired measurements](../../scripts/benchmarks/mac-m4-q8-decode.csv)

## Initial F32 dense-prefill baseline

One 512-row prompt chunk attending over 16,384 positions, with F32 queries, 24 query heads, four KV heads and width 256. The default composite path processes two 256-row blocks. The experimental path pads the six query heads in each KV group to eight, runs flash attention, then discards the extra heads. Each number is the fastest of four warm timings in a fresh process. Each timing includes host readback. The composite arm ran first in every process.

| Process | Composite (ms) | Padded flash (ms) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 157.6 | 144.5 | 8% |
| 2 | 177.6 | 137.1 | 23% |
| 3 | 174.4 | 144.8 | 17% |

The previous benchmark flattened each reference block before appending it, which mixed up the head and row order. The corrected check joins the blocks along the row dimension. Maximum absolute error against F32 was 0.000508 for composite and 0.000513 for padded flash in all three processes. The BF16 follow-up below alternates arm order and uses the model's configured activation dtype.

## BF16 serving-shape follow-up

The paired sweep on the same M4 mini found no repeatable decode attention saving at 12,288 positions. Fused decode's process medians were lower in all four repetitions from 14,336 positions through the last tested depth, 17,408. The production cutoff remains 16,384. For dense prefill, padded flash consistently reduced temporary allocation. Its strongest latency result came at 256 rows and 32,768 positions.

The release build used `native-mlx` on the port tree with the benchmark changes this series carries. Both sweeps use BF16 queries and group-64 Q8 keys and values with BF16 scales and biases. Each pair runs the same query through both arms and varies it before the next pair. The first arm alternates. Four fresh processes per shape also alternate their starting arm. The tables show medians of process medians. The latency change is the median of the four percentage changes from each process's composite median. A negative change means lower latency.

### Qwen3.6 decode around the cutoffs

Each process ran four warm-up pairs and 20 timed pairs per depth before progressing through the depth grid. The capacity rounds up to a 256-position multiple. All 960 pairs agreed within the existing BF16 row tolerance, with a largest absolute difference of 0.015625.

| Valid KV positions | Composite (µs) | Fused (µs) | Fused latency change | Fused wins / 80 |
| ---: | ---: | ---: | ---: | ---: |
| 10,240 | 1,034 | 1,046 | +4.6% | 46 |
| 11,264 | 1,120 | 1,071 | +0.3% | 53 |
| 12,287 | 1,082 | 1,061 | −2.0% | 51 |
| 12,288 | 1,032 | 1,061 | +1.0% | 46 |
| 12,289 | 1,062 | 1,069 | −1.3% | 50 |
| 13,312 | 1,062 | 1,035 | −2.5% | 57 |
| 14,336 | 1,119 | 1,072 | −4.6% | 62 |
| 15,360 | 1,228 | 1,185 | −5.0% | 60 |
| 16,383 | 1,286 | 1,213 | −6.5% | 67 |
| 16,384 | 1,293 | 1,168 | −7.6% | 70 |
| 16,385 | 1,329 | 1,133 | −10.1% | 68 |
| 17,408 | 1,386 | 1,282 | −7.4% | 67 |

At 12,288 the four process medians ranged from 4% slower to 15% faster with fused attention. At 16,384 all four showed reductions of 7–25%. Additional MLX allocation for a call was 2.02 MiB for fused attention at both cutoffs, versus 0.39 MiB at 12,288 and 0.52 MiB at 16,384 for the composite. The measurements time each kernel through synchronized host readback after the cache and query are resident. [All 960 decode pairs](../../scripts/benchmarks/mac-m4-q8-decode-cutoff.csv)

### Qwen3.8-27B dense prefill with BF16 activations

The model's 24 query heads, four KV heads and width 256 determine the test shape. The composite uses the resident query-block helper. The flash arm uses the same six-to-eight-head padding and output extraction as serving. Two warm-up pairs and six timed pairs ran per process. Every pair was checked against F32 attention over the same BF16-rounded inputs before Q8 quantization.

| Chunk rows | Valid KV positions | Composite blocks | Composite → flash (ms) | Flash latency change | Flash wins / 24 | Extra MLX peak, composite → flash (MiB) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 8,192 | Whole chunk | 12.47 → 14.06 | +14.3% | 4 | 50.5 → 8.25 |
| 256 | 8,192 | Whole chunk | 26.76 → 25.01 | −5.6% | 14 | 101 → 16.5 |
| 512 | 8,192 | Whole chunk | 51.36 → 51.29 | +1.1% | 12 | 202 → 33.0 |
| 128 | 16,384 | Whole chunk | 26.08 → 27.81 | +5.1% | 6 | 99.6 → 8.25 |
| 256 | 16,384 | Whole chunk | 52.26 → 52.36 | +1.7% | 8 | 199 → 16.5 |
| 512 | 16,384 | Whole chunk | 106.36 → 105.48 | −0.5% | 16 | 398 → 33.0 |
| 512 | 16,384 | 256 rows | 106.51 → 105.13 | −0.7% | 15 | 407 → 33.0 |
| 256 | 32,768 | Whole chunk | 126.75 → 106.85 | −14.6% | 24 | 395 → 16.5 |

At 512 rows and 16,384 positions, peak extra MLX memory fell about 92% with padded flash. Latency was close in both configurations. At 256 rows and 32,768 positions, the four process medians all improved by 14–20%. The largest absolute error across the grid against F32 was 0.00359 for composite and 0.00308 for padded flash. Extra MLX peak subtracts active bytes at the start of each call from the peak through host readback. The largest whole-process footprint of either sweep was 1.80 GiB. [All 192 prefill pairs](../../scripts/benchmarks/mac-m4-bf16-dense-prefill.csv)

## Reproduce the BF16 sweep

From `engine/`, build the ignored probes for Apple Silicon. Set `DECODE_TEST_BINARY` and `LIB_TEST_BINARY` to the two executable paths printed by Cargo. Set `RUN_DIR` to a writable scratch directory. The driver enumerates both tests and saves per-process logs and validated CSV files. It uses a 4 GiB process-footprint limit for the sweep.

```sh
cargo test --locked --release --features native-mlx --lib --test native_q8_decode --no-run
python3 ../scripts/bench_mlx_attention.py "$DECODE_TEST_BINARY" "$LIB_TEST_BINARY" "$RUN_DIR/smoke" --smoke
python3 ../scripts/bench_mlx_attention.py "$DECODE_TEST_BINARY" "$LIB_TEST_BINARY" "$RUN_DIR/attention-sweep"
```

The first two sections describe the original benchmark protocol. The BF16 follow-up uses the same probe names.

## Checkpoint qualification

The 16 GB mini can run the attention benchmarks above and the shared-executor fixtures. The documented 4-bit Qwen3.6 profile needs at least 36 GB, so whole-model qualification is outside the mini measurements. To reproduce the checkpoint comparison on a host that meets that profile, set `Q36_CKPT_DIR` to the checkpoint directory, `Q36_PROMPT_IDS` to the same file of at least 24,576 little-endian u32 token IDs for all arms, and `RUN_DIR` to a fresh writable scratch directory. From `engine/`, compare whole-decoder latency and generated token chains around the proposed cutoff. The threshold override below is available in the native library test.

```sh
(
  set -eu
  export Q36_CKPT_DIR Q36_PROMPT_IDS
  export Q36_ACTIVATION=bf16 Q36_KV_DTYPE=q8 QWEN36_G2A_COMPILE=1
  export DT_IDS_SOURCE=file DT_DEPTHS=8192,12288,14336,16384,24576 DT_TOKENS=64
  pass=0
  for cutoff in 16384 12288 12288 16384; do
    pass=$((pass + 1))
    Q36_OUT_DIR="${RUN_DIR:?}/decode-$pass-$cutoff" Q36_KV8_FUSED_MIN_LEN="$cutoff" \
      cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
      family::qwen36::resident::decode_timing::decode_phase_timing_by_mode \
      -- --ignored --exact --nocapture --test-threads=1
  done
)
```

Compare the served-sampling wall time at 12,288 and 14,336, where the two cutoffs select different implementations. The other depths are controls. Verify the per-depth outputs and token chains before interpreting a latency change.
