# Apple Silicon architecture and performance TODO

Use the 16 GB M4 Mac mini for this work. The baseline is `58a997309a7a425c08ed2bb5d2600842e8bdb5a9` on the local Mac performance branch. Measure release builds with the same inputs and alternating control and candidate runs. Keep latency, peak memory, output agreement and the path that actually ran alongside each result.

## Completed on the mini

- [x] Map the common worker, generation session and native MLX entry points for Qwen3.6, Flash-Next and Flint. Share Qwen3.6 and Flash-Next chunk admission, DeltaNet memory arithmetic and single-row packed projection. [Architecture map](../explanation/mlx-architecture.md)
- [x] Establish Q8 decode and dense-prefill attention baselines at production layer dimensions. Archive paired Q8 timings and check the dense reference. [Attention results](../explanation/mac-m4-attention.md)
- [x] Measure grouped Qwen3.6 embedding uploads against CPU dequantization and check both executors with the option off and on. Grouping cuts three import waits to one. [Embedding results](../explanation/mlx-architecture.md#measured-opportunity-in-qwen36-embedding-import)

## Work in order

1. [x] **Connect Flash-Next MLX startup to shared prefill admission.** Startup uses the explicit byte budget, prices the full serving context and bounds the selected rows by the worker piece. The mini fixture verified an exact fit, a budget one byte short, the prefix reserve, the worker cap and CUDA option rejection. Through the real factory, an omitted budget and a low budget took the one-token path. A fitting budget selected four-row chunks and produced identical continuation tokens with fewer Metal synchronizations. [Configuration and fallback rules](../reference/configuration.md#flash-next-mlx-prefill)
2. [x] **Measure complete synthetic requests on both resident executors.** Ran 512 timed requests through the Flash-Next and Qwen3.6 workers. Both fixture comparisons preserve all generated tokens. Flash-Next chunking consistently reduced first-token latency. Grouped Qwen3.6 import consistently saved synchronizations and showed mixed request latency. [Worker measurements and reproduction](../explanation/mac-mlx-requests.md)
3. [x] **Evaluate the attention candidates at serving shapes.** Ran 960 BF16 decode pairs around both cutoffs and 192 dense-prefill pairs. The sweep found no consistent latency benefit at 12,288 and confirmed one at 16,384. Padded flash cut temporary allocations at every prefill shape, and its largest latency benefit was at 256 rows and 32K context. [BF16 attention results and reproduction](../explanation/mac-m4-attention.md#bf16-serving-shape-follow-up)
4. [x] **Share device-aware prefill across all three MLX families.** Mapped CUDA's device-budget, KV-format, expert-staging and kernel-selection interfaces to MLX. The default-off startup policy uses the Metal working set and each family's reserve to size prefill. Passed the actual Flash-Next startup check on the mini and tested Qwen3.6 and Flint sizing at multiple byte thresholds. Measured Flint's grouped experts, sink attention and rolling-cache routes with the same counters. [Policy, evidence and reproduction](../explanation/mlx-device-scaling.md)
5. [x] **Stream oversized Flint checkpoints from local storage.** Automatically select disk-backed routed experts before import when the resident model would exceed the Metal budget. Reserve one layer's staging and check numerical parity with the resident grouped projections. [Flint weight placement](../explanation/mlx-device-scaling.md#disk-backed-flint-experts)
6. [ ] **Extend disk-backed weights to Qwen.** Reuse expert staging for Qwen3.6 MoE and price the widened scales and biases. Preserve the existing resident compile and batching paths when weights fit. Design layer staging for oversized dense weights and test end-to-end output before enabling any fallback by default.
7. [ ] **Make the selected Mac execution path visible to operators.** Compare startup, `config effective` and capability reporting for each MLX family against CUDA's existing reporting. Reuse the existing decision rows for supported settings. Verify reported backend, row cap, memory terms and fallback reason against the executor that serves the job.
8. [ ] **Establish which real checkpoints the mini can serve.** Check local artifacts against supported formats and estimate weights, caches and workspace together. Where one fits, run fixed prompts through the server, record time to first token, decode rate and peak memory, and compare output with the reference. Record any missing artifact or capacity limit.
9. [ ] **Review and integrate verified slices locally.** Check the exact tip, existing qualification inventory, default and opt-in behavior, numerical parity and recovery. Record integration, measurement, qualification and default status independently for each candidate.

## Reproduce the Flash-Next startup check

Run from `engine/` on the mini. The command should run exactly one test. Its output includes the chosen four-row chunk and the insufficient-budget fallback. The test loads a synthetic checkpoint through the real engine factory and compares all three configurations through worker shutdown.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification --lib \
  family::flash_next::resident::executor::tests::flash_next_mlx_serving_prefill_runs_the_admitted_path \
  -- --exact --nocapture --test-threads=1
```

The initial slice passed 58 selected host tests and 115 selected MLX tests on the mini. The release server built. Chunked startup is an opt-in and is fixture-validated. Full-checkpoint performance and qualification are open.

## Cache study

- [x] **Measure hardware and expert weight cache effects on the M4 mini.** Collected 408 CPU cache-latency observations, 48 paired file reads, 81 Metal staging samples, and three repeated live requests. All five shards of the published Flint checkpoint were verified. The live disk-backed server generated a complete visible answer. [Cache study and reproduction](../explanation/mac-m4-cache-study.md)
- [x] **Measure resident expert cache hit rates.** Traced 64,260 layer-expert accesses across six live requests. A 1 GiB per-layer LRU replay hit 15.97% overall and 24.01% on single-row stages. [Trace, replay and reproduction](../explanation/mac-flint-expert-reuse.md)
- [x] **Overlap independent Metal work with disk staging.** Added an opt-in Flint path that runs its shared expert while the host fills pages for routed experts. On the published checkpoint, both prompts returned the same output across eight runs. Prefill fell about 20% and decode throughput rose 20–23%. [Pipeline A/B and reproduction](../explanation/mac-flint-expert-pipeline.md)
- [x] **Measure coalesced SSD reads and probe ANE on the mini.** Consecutive-read batching lowered 143-token Flint prefill by 24.6% in the isolated A/B. Short total latency was level. Core ML enumerated 16 Neural Engine cores, six decision-head tests passed, and the idle experimental head measured 131 µs median. The live GUI now uses the earlier shared-page pipeline. [M4 ANE and SSD results](../explanation/mac-flint-ane-and-reads.md)
- [x] **Screen next-token read-ahead before ANE integration.** Held-out routes showed 3.22% recall on cache misses at 14.54% precision for one predicted expert per layer. The observed read overhead rules out deploying this predictor. [Held-out routes](../explanation/mac-flint-resident-cache.md#held-out-next-token-read-ahead)
- [x] **Prototype the byte-bounded resident expert cache.** Charged its real page footprint alongside session state and staging. A/B on the external SSD with the pinned Flint checkpoint matched outputs and preserved 512-row admission. At 1 GiB, short request time rose 14.6%, longer request time was level, and peak footprint rose approximately 1 GB. [A/B results](../explanation/mac-flint-resident-cache.md#checkpoint-comparison)
- [ ] **Attribute cache costs and repeat on internal storage.** Time expert reads and page imports separately on the pinned checkpoint. Use that breakdown to choose 512 MiB and 2 GiB follow-ups. The current live process still holds the internal checkpoint files from the old scratch directory. Use a durable internal copy at a controlled service restart.

## Qualification boundary

The mini's Metal working set is 11.84 GiB. The documented Qwen3.6 MLX checkpoint is 20.4 GB and Flint is 22.6 GB before serving memory. The Mac lane can measure full requests when a supported, pinned checkpoint is available locally and the resident or disk-backed placement is admitted. The published Flint checkpoint ran on the mini in disk-backed mode. Representative quality and full-context memory measurements are open. [Qwen3.6 requirements](../how-to/qwen36.md#on-a-mac) · [Flint requirements](../how-to/flint.md#on-a-mac)

- [ ] **35B whole-checkpoint throughput, quality and cutoff decision.** The documented 4-bit profile requires at least 36 GB. The Mac performance lane can resume if a supported profile and pinned checkpoint fit this mini. Until then use the layer and small executor fixtures above and keep the production cutoff at 16,384.
- [ ] **Default decisions.** The integration coordinator can revisit each opt-in when representative checkpoint output, memory and performance evidence exists for that path or Dan explicitly makes a default decision. The mini's microbenchmarks characterize individual operations.
