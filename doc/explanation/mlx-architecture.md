# Shared execution on Apple Silicon

Qwen3.6, Flash-Next and Flint use the same worker and generation session on MLX. Qwen3.6 and Flash-Next also use one host-side prefill planner and one native operation for a packed projection of a single row. A change at either of those seams can now be checked against both model families on a 16 GB Mac mini.

| Responsibility | Shared implementation | Family-owned inputs and behavior |
| --- | --- | --- |
| Serving and sampling | [ExecutorOps](../../engine/src/family/executor.rs) and [DecodeSession and TokenStepper](../../engine/src/family/session.rs) | Each adapter supplies model steps, state, cache growth and restore rules. |
| Prefill admission | [CUDA arena admission](../../engine/src/family/prefill/admission.rs) and [MLX resident admission](../../engine/src/family/prefill/mlx.rs) | The MLX planner picks the largest admissible chunk from 2,048 down to four rows. Qwen3.6 supplies a worker cap and a workspace cost that bounds attention scores. Flash-Next prices expert pages and waves and charges its decode pool. A one-token result uses ordinary steps. |
| DeltaNet memory | [MLX retire cap and sub-chunk sizing](../../engine/src/family/prefill/mlx.rs) | Both families use the 512 MiB chunk retire cap. Flash-Next splits its recurrence to fit the available room. The Qwen3.6 gated kernel consumes all chunk rows in one call. |
| Packed projection | [MlxKernels](../../engine/src/kernels/mlx/mod.rs) | Both resident decoders call `qmm_row` for a single activation with the packed weight's input width. Multi-row projections keep the row axis. |
| Flint prefill | [Flint MLX worker adapter](../../engine/src/family/flint/mlx_executor.rs) | Flint implements `feed_rows` with its own attention and cache layout. It shares the worker contract and MLX backend. |

## Serving coverage

Qwen3.6 startup calls the planner when its settings specify a prefill budget. Its single-session path falls back to one-token prefill when the budget or worker cap excludes every chunk. The Flash-Next executor supports planning and chunked worker jobs. Its server factory currently opens the executor with one-token prefill. Exposing chunk admission there requires charging the session, MLX allocator cache, pressure margin, S3 cache and decode pool, and bounding the plan by the worker's configured piece size. [Qwen3.6 startup](../../engine/src/family/qwen36/start.rs) · [Flash-Next startup](../../engine/src/family/flash_next/start.rs) · [Flash-Next executor](../../engine/src/family/flash_next/resident/executor.rs)

## Measured opportunity in Qwen3.6 embedding import

The fused MLX embedding path packs the prompt rows into one weight, then uploads its payload, scales and biases. Each upload synchronizes the device. The default-off `Q36_BATCH_EMBED_IMPORT=1` sends those same tensors through the existing MLX batch-import operation, reducing three import synchronizations to one. On the four-layer fixture, the total for an eight-row device-routed prompt chunk fell from eight synchronizations to six.

Measured on the 16 GB M4 mini on 17 November 2026 with a release build and qualification counters. Each timed sample reads affine embedding rows from a synthetic checkpoint, imports them, dequantizes to F32 and reads the result back. The fixture has eight vocabulary entries of width 64. The token sequence rotates between samples. Every returned value matched CPU dequantization bit for bit.

| Prompt rows | Import waits, default → grouped | Median paired latency reduction | Faster with grouped import |
| ---: | ---: | ---: | ---: |
| 8 | 3 → 1 | 10.8% | 8 of 12 process pairs |
| 128 | 3 → 1 | 11.2% | 12 of 12 process pairs |

Each of 24 fresh processes ran four warm-ups and 32 timed samples per size. The eight-process order was default, grouped, grouped, default, grouped, default, default, grouped, repeated three times. Each adjacent pair compares the two process medians at the same size. The 8-row result varied from 32% slower to 39% faster. For 128 rows, reductions ranged from 3% to 40%. These timings cover the embedding operation through readback. Whole-prompt latency and generation throughput remain unmeasured. [All 1,536 timings](../../scripts/benchmarks/mac-m4-embedding-import.csv)

To repeat one balanced block from `engine/`, use the default fused embedding setting. Each invocation runs one ignored test and prints 64 timing records.

```sh
for grouped in 0 1 1 0 1 0 0 1
do
  Q36_BATCH_EMBED_IMPORT="$grouped" cargo test --locked --release \
    --features native-mlx,native-fault-qualification --lib \
    family::qwen36::resident::prefill::step::tests::packed_embedding_import_timing \
    -- --ignored --exact --nocapture --test-threads=1 || exit 1
done
```

## Check both families on the mini

Run from `engine/` on Apple Silicon. The first command checks both cost models with host arithmetic. The second uses synthetic weights through the real MLX executors and compares chunked prefill with token-by-token execution. Each family checks session state, logits and continuation tokens. The filter in the second command should run exactly two tests.

```sh
cargo test --locked --release --lib -- family::qwen36::resident::chunk_plan::tests \
  family::flash_next::prefill_plan::tests --test-threads=1
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  a_chunked_prefill_leaves_the_session_of_one_token_steps -- --test-threads=1
```

The M4 attention measurements cover Q8 decode and dense prefill at production layer dimensions. Full-checkpoint throughput and quality remain separate measurements. [M4 attention results](mac-m4-attention.md)
