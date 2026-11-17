# Q8 attention on a 16 GB M4 Mac mini

On that Mac, the fused Q8 decode path breaks even between 8,191 and 12,287 cached positions and saves at double-digit rates above that; production switches at 16,384. The dense-prefill comparison below measures the opt-in padded flash path.

The records in this document were measured by the PR author on his own M4 Mac mini; they are carried as contributor records and have not been reproduced here.

Measured 17 November 2026 on a 10-GPU-core M4, macOS 26.3, with the locked release build, Rust 1.98.1, Xcode 26.5 and `native-mlx`, on the author's Mac performance branch. The benchmark test and the dense-prefill reference correction are the ported commit `3b307c6d8` (`chore(mlx): measure Q8 decode dispatch and align the dense prefill oracle`). Both benchmarks used synthetic resident inputs and timed one attention layer through a synchronized host readback. Whole-model decode throughput, time to first token and generation quality still need a checkpoint run.

## Qwen3.6 decode

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

## Dense prefill

One 512-row prompt chunk attending over 16,384 positions, with F32 queries, 24 query heads, four KV heads and width 256. The default composite path processes two 256-row blocks. The experimental path pads the six query heads in each KV group to eight, runs flash attention, then discards the extra heads. Each number is the fastest of four warm timings in a fresh process. Each timing includes host readback. The composite arm ran first in every process.

| Process | Composite (ms) | Padded flash (ms) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 157.6 | 144.5 | 8% |
| 2 | 177.6 | 137.1 | 23% |
| 3 | 174.4 | 144.8 | 17% |

The previous benchmark flattened each reference block before appending it, which mixed up the head and row order. The corrected check joins the blocks along the row dimension. Maximum absolute error against F32 was 0.000508 for composite and 0.000513 for padded flash in all three processes. The dense serving comparison should alternate arm order and use the model's configured activation dtype.

## Reproduce locally

From `engine/` in the benchmark checkout, run these commands on an Apple Silicon device. Each filter should run exactly one test. The largest observed benchmark process footprint was 2.01 GB.

```sh
cargo test --locked --release --features native-mlx --test native_q8_decode \
  q8_decode_dispatch_timing -- --ignored --exact --nocapture --test-threads=1
cargo test --locked --release --features native-mlx --lib \
  family::qwen36::resident::prefill::attn::tests::dense_rep6_flash_pad_bench \
  -- --ignored --exact --nocapture --test-threads=1
```

## Next checkpoint measurement

The documented 4-bit Qwen3.6 profile needs a Mac with at least 36 GB. On that host, set `Q36_CKPT_DIR` to the checkpoint directory, `Q36_PROMPT_IDS` to the same file of at least 24,576 little-endian u32 token IDs for all arms, and `RUN_DIR` to a fresh writable scratch directory. From `engine/`, compare whole-decoder latency and generated token chains around the proposed cutoff. The threshold override below is available in the native library test.

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
