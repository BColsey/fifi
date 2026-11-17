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

## Reproduce on Apple Silicon

From `engine/`, build the release library tests. Set `MLX_TEST_BINARY` to the executable path printed by Cargo and `RUN_DIR` to a writable scratch directory. The driver enumerates one exact test per family. It alternates which arm runs first in each of eight pairs per family. Each process runs two warm-ups and eight recorded samples at each prompt length, with eight generated tokens per request.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib --no-run
python3 ../scripts/bench_mlx_requests.py "$MLX_TEST_BINARY" "$RUN_DIR/requests.csv"
```

Pass `--pairs 1 --samples 1` for an eight-request smoke run and use a fresh output path for the full run. A row-level embedding benchmark gives the separate measurement of the upload operation in isolation. [MLX architecture and embedding measurements](mlx-architecture.md)
