# MLX worker request measurements on an M4 Mac mini

Flash-Next chunked prefill lowered time to first token in every paired run of the small worker fixture. Qwen3.6 grouped embedding import consistently reduced synchronizations. Its request latency varied enough that this run gives no consistent request-level speedup.

Measured on 17 November 2026 on a 16 GB M4 Mac mini running macOS 26.3. The base was `39975d5beb34b6007eb08e3ac5fa33fcd8e04cb9`, with the request benchmark added. The release library test build used `native-mlx,native-fault-qualification` to count Metal synchronizations. Each arm ran a fresh single-session worker through the normal prefill and decode calls with a 16 MiB MLX allocator cache limit and prefix reuse disabled. These workers used synthetic checkpoints and pretokenized input. The timing window starts at worker submission and excludes checkpoint load, HTTP, and tokenization.

## Results

The cells show control → candidate. Each value is the median of eight process medians. The paired latency column takes each pair's percentage change from the control and reports the median and the count of pairs in which the candidate was faster.

| Change | Prompt tokens | Time to first token (ms) | Paired latency change | Effective prompt tok/s | Steady decode tok/s | Metal synchronizations | MLX peak active KiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash-Next chunked prefill | 8 | 37.26 → 10.16 | −70.2%, 8/8 | 216 → 791 | 194 → 210 | 69 → 45 | 238 → 468 |
| Flash-Next chunked prefill | 33 | 160.77 → 88.78 | −45.3%, 8/8 | 206 → 375 | 188 → 185 | 169 → 105 | 284 → 535 |
| Qwen3.6 grouped embedding import | 8 | 9.82 → 12.13 | +41.8%, 3/8 | 822 → 682 | 278 → 246 | 41 → 39 | 569 → 569 |
| Qwen3.6 grouped embedding import | 33 | 39.06 → 33.34 | +3.3%, 3/8 | 870 → 998 | 260 → 257 | 67 → 59 | 658 → 658 |

Flash-Next's two-layer fixture holds 19 dense prompt positions. Its 33-token case feeds five worker pieces and takes one-token steps past the dense part. The chunked arm had a median whole-process peak footprint of 64.8 MiB, up from 63.1 MiB, including both prompt sizes and their warm-ups. At 33 prompt tokens, the paired first-token reductions ranged from 35% to 59%.

Both Qwen3.6 arms use eight-row prefill pieces and the four-layer BF16 activation fixture. Grouped import removed two waits in the short request and eight in the long request. Whole-process peak footprint was 65.2 MiB in both arms to one decimal place. At 33 prompt tokens, the paired first-token changes ranged from a 63% reduction to a 34% increase. Steady decode results varied across pairs for both families. Every measured request matched the eight continuation token IDs of its family's control. [All 512 timed requests](../../scripts/benchmarks/mac-m4-mlx-requests.csv)

## Measurement boundaries

- **Time to first token** runs from worker submission to receipt of the first token batch.
- **Effective prompt rate** divides the prompt length by the interval from the worker's `Started` event to the first token. This follows the server's request log and includes generating that token.
- **Steady decode rate** divides the seven remaining tokens by the elapsed time from the first token to the last.
- **Metal synchronizations** count the full worker request. The MLX peak is active allocation after resetting the process-wide counter before each request. The process footprint is the macOS high-water mark reported for the entire test process, including load and all requests.

## Larger prefill pieces and KV formats

The 16 GB M4 mini ran 448 measured synthetic worker requests using 65- and 257-token prompts, eight completion tokens, and 32- or 128-row prefill pieces. The Qwen3.6 MoE and Qwen3.8 dense fixtures use the same 64-lane attention shape for each of their F32, BF16, and Q8 KV comparisons and BF16 activations. Flash-Next uses its F32 QSA state and index rows. All continuations matched at both widths within each KV profile.

The Qwen planner already priced composite score workspace in query blocks when `Q36_PREFILL_QUERY_SLICE` was set. The BF16 execution path used a full-chunk call in that case. It now uses the same causal query-block routine as F32. Native checks for each of the three KV forms verify that a split dispatches more than one block and matches whole-chunk output to within FP32 tolerance. The benchmark uses 16-row attention slices in both Qwen arms so the outer prefill width is the only setting changed within each profile.

| Fixture | KV | First token at 257 prompt tokens, 32 → 128 rows | Paired change | Metal waits | Peak active MLX KiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6 MoE | F32 | 16.32 → 6.43 ms | −61% | 83 → 47 | 2,417 → 3,922 |
| Qwen3.6 MoE | BF16 | 16.80 → 6.02 ms | −65% | 83 → 47 | 3,167 → 3,222 |
| Qwen3.6 MoE | Q8 | 17.49 → 7.85 ms | −55% | 83 → 47 | 2,466 → 3,488 |
| Qwen3.8 dense | F32 | 7.78 → 2.73 ms | −65% | 59 → 41 | 682 → 1,791 |
| Qwen3.8 dense | BF16 | 7.19 → 2.70 ms | −63% | 59 → 41 | 796 → 1,791 |
| Qwen3.8 dense | Q8 | 7.03 → 3.08 ms | −56% | 59 → 41 | 653 → 1,791 |
| Flash-Next | F32 | 335.73 → 339.91 ms | +1% | 993 → 993 | 1,001 → 998 |

Each cell shows the median across four process medians, except the change column, which shows the median of the paired percentage changes. The raw measurements include time spent inside import, readback, evaluation, and forced retirement calls. Those wall times include synchronization and can cover queued work. Decode timings varied. [All 448 requests](../../scripts/benchmarks/mac-m4-mlx-kv-worker-matrix.csv)

The Flash-Next fixture has 19 dense QSA positions. Both arms in the table above feed one token per step beyond that limit, which accounts for the matching synchronization counts.

## Flash-Next sparse prefill

With `FIFI_FLASH_NEXT_MLX_SPARSE_PREFILL=1`, Flash-Next can use up to eight rows per batch after the dense QSA limit. Each layer batches its projections and cache append. Each row runs the decode selector against its own prefix, including the incomplete final block. Ranking still uses the decode path's stable block order and additive attention mask. Layer tests compare attention output and all three QSA cache arrays to one-token decode, and alter later rows to check that earlier rows cannot see them. [Flash-Next prefill split](../../engine/src/family/flash_next/resident/prefill/step.rs)

The shared MLX planner first admits the ordinary prefill chunk. The sparse path can use up to 512 MiB of the remaining workspace for one layer's per-row graphs, including prefix slices, index scoring, sorting and masked attention. Startup reports this allowance as `sparse_workspace_bytes`. As the prefix grows, fewer rows fit in the same allowance. When fewer than two fit, execution takes one-token steps.

On 18 November, the 16 GB M4 mini ran 64 timed requests through the same two-layer fixture. Both arms used 128-row worker pieces and eight completion tokens. The candidate lowered first-token latency in all four pairs at both lengths, and every continuation matched. Cells show medians across four process medians. The paired change uses each pair's relative first-token latency.

| Prompt tokens | First token | Paired change | Metal waits | Peak active MLX KiB | Time for the last seven tokens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | 67.50 → 20.29 ms | −69.4% | 225 → 69 | 1,009 → 1,009 | 10.42 → 11.21 ms |
| 257 | 336.97 → 84.18 ms | −75.0% | 993 → 169 | 1,006 → 1,119 | 10.23 → 10.97 ms |

The median process peak footprint rose from 67.2 to 69.9 MiB. Decode took longer in all four pairs at both prompt lengths. The option remains off by default while full-checkpoint parity, memory and decode behavior are open. [All 64 timed requests](../../scripts/benchmarks/mac-m4-flash-next-sparse-prefill.csv)

## Reproduce on Apple Silicon

From `engine/`, build the release library tests. Set `MLX_TEST_BINARY` to the executable path printed by Cargo and `RUN_DIR` to a writable scratch directory. The driver enumerates one exact test per family. It alternates which arm runs first in each of eight pairs per family. Each process runs two warm-ups and eight recorded samples at each prompt length, with eight generated tokens per request.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib --no-run
python3 ../scripts/bench_mlx_requests.py "$MLX_TEST_BINARY" "$RUN_DIR/requests.csv"
```

Pass `--pairs 1 --samples 1` for an eight-request smoke run and use a fresh output path for the full run. A row-level embedding benchmark gives the separate measurement of the upload operation in isolation. [MLX architecture and embedding measurements](mlx-architecture.md)

Use the same release harness for the larger worker comparison. A fresh output path and `--pairs 1 --samples 1` give a quick parity and dispatch check before the full run.

```sh
python3 ../scripts/bench_mlx_requests.py "$MLX_TEST_BINARY" "$RUN_DIR/kv-workers.csv" \
  --kv-matrix --pairs 4 --samples 4
```

Use the same harness for the sparse Flash-Next comparison, with 128-row worker pieces, a 4 GiB device ceiling and a 2 GiB family prefill budget in both arms.

```sh
python3 ../scripts/bench_mlx_requests.py "$MLX_TEST_BINARY" "$RUN_DIR/flash-sparse.csv" \
  --flash-sparse-prefill --pairs 4 --samples 4
```
