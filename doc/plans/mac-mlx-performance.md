# Apple Silicon architecture and performance TODO

Use the 16 GB M4 Mac mini for this work. The baseline is `58a997309a7a425c08ed2bb5d2600842e8bdb5a9` on the local Mac performance branch. Measure release builds with the same inputs and alternating control and candidate runs. Keep latency, peak memory, output agreement and the path that actually ran alongside each result.

## Completed on the mini

- [x] Map the common worker, generation session and native MLX entry points for Qwen3.6, Flash-Next and Flint. Share Qwen3.6 and Flash-Next chunk admission, DeltaNet memory arithmetic and single-row packed projection. [Architecture map](../explanation/mlx-architecture.md)
- [x] Establish Q8 decode and dense-prefill attention baselines at production layer dimensions. Archive paired Q8 timings and check the dense reference. [Attention results](../explanation/mac-m4-attention.md)
- [x] Measure grouped Qwen3.6 embedding uploads against CPU dequantization and check both executors with the option off and on. Grouping cuts three import waits to one. [Embedding results](../explanation/mlx-architecture.md#measured-opportunity-in-qwen36-embedding-import)

## Work in order

1. [x] **Connect Flash-Next MLX startup to shared prefill admission.** Startup uses the explicit byte budget, prices the full serving context and bounds the selected rows by the worker piece. The mini fixture verified an exact fit, a budget one byte short, the prefix reserve, the worker cap and CUDA option rejection. Through the real factory, an omitted budget and a low budget took the one-token path. A fitting budget selected four-row chunks and produced identical continuation tokens with fewer Metal synchronizations. [Configuration and fallback rules](../reference/configuration.md#flash-next-mlx-prefill)
2. [ ] **Measure complete synthetic requests on both resident executors.** Start with Flash-Next one-token versus admitted chunks, then Qwen3.6 grouped versus default embedding import. Use the same prompt and decode length per pair and check continuation tokens. Report time to first token, prompt processing rate, steady decode rate, synchronizations and peak memory. Include one short prompt and one that spans multiple worker pieces. Keep model-fixture results labeled.
3. [ ] **Evaluate the attention candidates at serving shapes.** Sweep Q8 context depths around 12,288 and 16,384 with BF16 queries. Repeat dense prefill with the configured activation dtype, alternating arm order and checking each arm against the reference. Record latency and memory at the chunk widths that the mini admits.
4. [ ] **Audit shared architecture across all three MLX families.** Map CUDA’s device-budget, KV-format, expert-staging and kernel-selection interfaces to their MLX counterparts. Exercise Flint’s row, attention-sink and rolling-cache fixtures with the same counters. Rank duplicated work by measured cost and promote helpers where layout, lifetime and error semantics match.
5. [ ] **Make the selected Mac execution path visible to operators.** Compare startup, `config effective` and capability reporting for each MLX family against CUDA's existing reporting. Reuse the existing decision rows for supported settings. Verify reported backend, row cap, memory terms and fallback reason against the executor that serves the job.
6. [ ] **Establish which real checkpoints the mini can serve.** Check local artifacts against supported formats and estimate weights, caches and workspace together. Where one fits, run fixed prompts through the server, record time to first token, decode rate and peak memory, and compare output with the reference. Record any missing artifact or capacity limit.
7. [ ] **Review and integrate verified slices locally.** Check the exact tip, existing qualification inventory, default and opt-in behavior, numerical parity and recovery. Record integration, measurement, qualification and default status independently for each candidate.

## Reproduce the Flash-Next startup check

Run from `engine/` on the mini. The command should run exactly one test. Its output includes the chosen four-row chunk and the insufficient-budget fallback. The test loads a synthetic checkpoint through the real engine factory and compares all three configurations through worker shutdown.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  family::flash_next::resident::executor::tests::flash_next_mlx_serving_prefill_runs_the_admitted_path \
  -- --exact --nocapture --test-threads=1
```

The initial slice passed 58 selected host tests and 115 selected MLX tests on the mini. The release server built. Chunked startup is an opt-in and is fixture-validated. Full-checkpoint performance and qualification are open.

## Qualification boundary

- [ ] **35B whole-checkpoint throughput, quality and cutoff decision.** The documented 4-bit profile requires at least 36 GB. The Mac performance lane can resume if a supported profile and pinned checkpoint fit this mini. Until then use the layer and small executor fixtures above and keep the production cutoff at 16,384.
- [ ] **Default decisions.** The integration coordinator can revisit each opt-in when representative checkpoint output, memory and performance evidence exists for that path or Dan explicitly makes a default decision. The mini's microbenchmarks characterize individual operations.
