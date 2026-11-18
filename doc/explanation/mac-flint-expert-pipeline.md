# Overlapping Flint expert reads and Metal work

The full Flint INT4 checkpoint exceeds the M4 mini's working set. In disk-backed mode, the CPU waits for the Metal router to produce the expert IDs, reads their packed weights, assembles the layer's stack, and submits the routed expert computation to Metal. The next layer needs the current layer's output to select its own experts. That dependency limits exact lookahead in a single sequence.

Flint's shared expert uses the same layer input and runs independently of the routed weights. With `FIFI_FLINT_MLX_EXPERT_PIPELINE=1`, fifi submits the shared expert to Metal after router readback. The CPU then reads the routed expert slices into page-aligned host buffers while the shared computation is in flight. The routed stack and local indices use the same page-import API as Flash-Next. The importer can wrap these unified-memory pages directly. The buffers stay pinned through the routed computation's evaluation. The existing memory plan still reserves space for the copied path.

This option applies to Flint's disk-backed route and starts off. Routing, tensor dtype and quantization, attention, and token budgets are identical in the A/B runs. Every selected expert still comes through the storage reader. A resident expert cache can cut both reads and imports. The separate trace replay sized the first candidate at 1 GiB. [Expert reuse study](mac-flint-expert-reuse.md)

## Mini measurements

Measured on 18 November 2026 with the verified published checkpoint on the 16 GB M4 Mac mini. Four fresh servers ran in control, pipeline, pipeline, control order. Each served the same two greedy prompts to 12 completion tokens, with prefix reuse disabled. Output digests matched across all eight requests. The table shows the median within each mode and prompt.

| Prompt tokens | Prefill, control → pipeline | Decode, control → pipeline | Peak process footprint, control → pipeline |
| ---: | ---: | ---: | ---: |
| 16 | 8.77 → 6.95 s | 1.43 → 1.76 tokens/s | 2.39 → 1.82 GB |
| 45 | 17.35 → 13.83 s | 1.43 → 1.72 tokens/s | 2.98 → 1.96 GB |

Prefill fell 20–21% and decode throughput rose 20–23%. The attributed storage reads fell 1–4% on these runs. Each arm still stages every selected expert, so the storage byte variation depends in part on file caching and read-ahead. The process footprint counter includes startup and any previous requests in that same process. [Eight live requests](../../scripts/benchmarks/mac-m4-flint-pipeline-live.csv)

A separate one-layer test compared only packed-weight staging, with the shared expert excluded. At one row, the warm-file median fell from 4.71 to 1.96 ms. At 16 rows it fell from 66.17 to 23.16 ms. The page-backed arm eliminated one Metal wait at every shape. All 162 trials matched output bytes. The experiment shows that changing how the CPU hands weights to Metal contributes to the whole-request improvement. The independent effect of running the shared expert early remains unmeasured. [Layer samples](../../scripts/benchmarks/mac-m4-flint-pipeline-layer.csv)

The release server built and 43 selected regressions passed with the option enabled. Longer prompts, sustained generation, and broader parity are open. The live GUI stayed available during the experiment.

## Reproduce

Use the published Flint INT4 checkpoint from the [Flint runbook](../how-to/flint.md#on-a-mac). Build `enginesrv` from `engine/` with `cargo build --locked --release --features native-mlx --bin enginesrv`. Compile the existing process counter helper in a scratch directory. Run the driver from the repository root, using a fresh output directory and a free loopback port. Use `--smoke` with a separate output directory for the short parity check.

```sh
xcrun clang -O2 -Wall -Wextra -Werror scripts/mac_cache_study/proc-io.c -o "$PROC_IO"
python3 scripts/mac_cache_study/flint_pipeline_ab.py \
  "$ENGINESRV" "$CHECKPOINT" "$PROC_IO" "$RUN_DIR"
```

The driver starts and shuts down one benchmark instance per run. It verifies disk placement and the prefill cap, compares complete output digests, and saves request telemetry and per-process storage and footprint counters. [A/B driver](../../scripts/mac_cache_study/flint_pipeline_ab.py)
