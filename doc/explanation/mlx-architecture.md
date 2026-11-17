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

## Check both families on the mini

Run from `engine/` on Apple Silicon. The first command checks both cost models with host arithmetic. The second uses synthetic weights through the real MLX executors and compares chunked prefill with token-by-token execution. Each family checks session state, logits and continuation tokens. The filter in the second command should run exactly two tests.

```sh
cargo test --locked --release --lib -- family::qwen36::resident::chunk_plan::tests \
  family::flash_next::prefill_plan::tests --test-threads=1
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  a_chunked_prefill_leaves_the_session_of_one_token_steps -- --test-threads=1
```

The M4 attention measurements cover Q8 decode and dense prefill at production layer dimensions. Full-checkpoint throughput and quality remain separate measurements. [M4 attention results](mac-m4-attention.md)
