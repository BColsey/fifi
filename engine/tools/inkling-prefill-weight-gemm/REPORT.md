# FiFi prefill weight GEMM — RTX 2080 Ti

**The 400 ms five-family budget and the 512 ms / 1000 tok/s whole-request target were not met.** On the latest release, measured code-prompt five-family time fell **2644.432 → 764.469 ms**; all GPU kernels fell **3024.094 → 1067.815 ms**. Matched unprofiled code medians were **7022.832 → 5059.904 ms**, or **72.76 → 100.99 prefill tok/s** (four runs per arm, 511 rows). The same optimized binary without MLP workspace reuse achieved 5033.313 ms / 101.52 tok/s: the extra MLP flag did not improve wall time despite reducing GPU time. On the latest release the prose five-family sum was **2754.787 → 793.031 ms** (MLP reuse off but other optimizations on: 789.411 ms, so no incremental prose win). These are real grouped model calls with fixed token IDs, no prefix reuse. Nsight durations, unprofiled wall times, and original-trace projections below are different measurements; none is an HTTP benchmark.

## What changed

`FIFI_INKLING_CUDA_PREFILL_Q8_VISIT=1` selects FiFi's own SM75 kernels. A CTA shares activation panels across output weights; aligned F32 inputs use a shape-selected vector prefetch, odd strides use the exact fallback. Q8/Q6 and the exact expert tiles retain FP32 lane FMA and reduction order. Deployed WMMA experts retain half operands and MMA order; IQ1_S loads its codebook once per CTA. Independent K/V and gain/forget projections share launches. Controller down must finish before the heads consume it; it instead reuses weights across rows.

Optional additions (all default off): `FIFI_INKLING_CUDA_PREFILL_WEIGHT_WIDE=1` requests a bounded 64-row controller workspace; `..._WEIGHT_WINDOW=128` or `256` requests a larger one with smaller fallback. `..._WEIGHT_PANELS=1` allows two-stage expert WMMA in a dead, disjoint arena span. `..._WEIGHT_MLP=1` allows the later MLP controller to reuse an already admitted wide workspace. The full prefix in each case is `FIFI_INKLING_CUDA_PREFILL`. No reduced memory floor, resident half-precision weight-bank assumption, new CLI, or default change. Keep these flags off by default: width and scratch choices improved particular kernels, not every wall-time sample.

**Latest code request, measured ledger.** Packed + Q8 pair/replacements **1437.315 → 418.486 ms** (Q6 and Q8 reuse now share typed tiles); exact expert including legacy Q6 **468.496 → 125.457**; expert WMMA including panel conversion **550.156 → 183.147**; dense including heads/down **188.465 → 37.379**. The MLP workspace alone reduced reported controller tiles **25 → 2** and dense launches **2384 → 728**, against the same binary with MLP reuse off (`ledger-mlp-code.log`, `trace-code-{0,8,9}-mlp*`).

## Supplied 511-row trace: baseline and projection

Only the baseline column was measured in the supplied request. For the estimate, multiply real-weight CUDA-event times by the eligible counts in `evidence/request.csv`; keep the observed durations for other calls. Short Q8 pair bins bracket M7/8, long pairs and Q6 bracket M63/64; Q6 Y4 is M8. WMMA is priced at the full M32 tile endpoint. For Q8 grouping, assume all 1024 adjacent candidate pairs also meet host-group and alias checks (1024 observed separately on a fresh narrow trace). For panels, assume eligible scratch fits. Do **not** transplant narrow launch counts to a wide workspace or infer ragged eligibility from adjacency alone.

| Family | Supplied trace, ms | Tuned narrow projection with panels, ms |
|---|---:|---:|
| Packed GEMV | 1156.323 | 458.133 |
| Exact expert | 461.445 | 253.598–254.788 |
| Expert WMMA | 441.430 | 136.576 |
| Q8 row pair | 350.653 | 108.899–110.746 |
| Dense | 198.329 | 98.417 |
| **Total** | **2608.179** | **1055.623–1058.659** |

Without temporary panels the projection is **1089.031–1092.068 ms**; without either panels or Q8 grouping **1092.542–1096.041 ms**. No ragged or MLP savings are credited. See `final-project-*.log` and `project_all.py`. Earlier two-row full visit, exact 32×8, and first multicodec projections were **2310.152–2339.829**, **1894.616–1899.449**, and **1286.171–1289.088 ms** respectively. They are historical comparisons, not extra savings.

## Component measurements and parity

`evidence/final-shapes.csv` gives **305 matched shape/arm rows** with microseconds, logical bytes, rates and exact-word counts. Examples below are from the same `expert-final.csv` run. Weights come from the actual UD-IQ1_M checkpoint; event-probe inputs are deterministic synthetic activations. Weight bytes count logical sweeps, not DRAM transactions. NCU denied counter access (`ERR_NVGPUCTRPERM`), so the one-sweep 350 GB/s target is **not established**.

| M,N,K / path | µs off → on | Logical weight bytes off → on | GB/s off → on | TFLOP/s off → on |
|---|---:|---:|---:|---:|
| 8,2304,3072 Q8 | 180.377 → 50.875 | 60,162,048 → 7,520,256 | 333.54 → 147.82 | 0.6278 → 2.2260 |
| 64,3072,2304 Q6 | 675.456 → 229.120 | 185,794,560 → 23,224,320 | 275.07 → 101.36 | 1.3413 → 3.9541 |
| 32,2304,2304 IQ1_S WMMA | 253.952 → 38.946 | 1,036,800 → 1,036,800 | 4.08 → 26.62 | 1.3378 → 8.7233 |
| 8,2304,2304 IQ2_XXS exact | 130.827 → 65.200 | 5,474,304 → 1,368,576 | 41.84 → 20.99 | 0.6492 → 1.3027 |
| 8,4608,64 paired heads | 21.042 → 6.639 | 9,437,184 → 1,179,648 | 448.49 → 177.68 | 0.2242 → 0.7107 |
| 8,64,4608 controller down | 11.629 → 7.945 | 9,437,184 → 4,718,592 | 811.52 → 593.91 | 0.4058 → 0.5939 |

The initial queued probe ran **49/49 timed arms**, 2,330,624 exact candidate words including strided cases (`events.*`). The held-out prefetch sweeps used four input/order seeds; select on the first two, evaluate on the other two (`tune-report.log`, `extra-tune-report.log`, `expert-tune-summary.log`). Tests cover output guards, misalignment, odd strides, temporary workspace too small, overlapping writes, cross-call dependencies, incompatible ragged bins and unsupported shapes. Relative to the deployed reference, selected outputs are bitwise exact. Non-bitwise split-K, FP16 conversion and residual INT8 MMA experiments stayed probe-only: relative L2 <0.001 and max error/reference peak <0.002 were fixture gates, **not** a model acceptance criterion. For example Q6 M64 residual MMA took 514.172–543.160 µs vs 239.232 µs exact and relative L2 ≈1.20e−5 (`q6-imma.csv/.stderr`).

Model validation: `weight_visit_request` compared all **200,058** final logits in each unprofiled arm; `device_grouped_prefill_last_logits` compared **24 × 200,058 logits** and 48 route receipts per prompt. Latest code and prose forced-prefix oracles each had **zero differing words or routes**, max absolute error zero (`mlp-{code,prose}-oracle.log`); the code window comparison also matched at 64/128/256 (`final-code-oracle.log`). Latest release device-required tests: **15 resident dense + 1 controlled-attention + 2 controller + 1 workspace = 19 passed**, no skips (`unit-mlp*.log`). `cargo fmt --check`, CUDA test check and release build passed (`{fmt,check,release}-mlp.status`).

## Three remaining costs and next step

1. **Packed projections** are still largest. An exact single-sweep tile took 349.400 µs vs 248.137 for the tiled Q8 M64 case; it used 217 vs 80 registers/thread. In the controlled M64 Q8 ablation, normal decode/reduction took 307.840 µs, a repeated packed panel 217.152, cached scalar per lane 117.952. Removing fresh weights does not eliminate the FP32 work. Compact lane-major Q8 also lost **even excluding repack** (M8/N1024: 37.155 vs 15.819 µs tuned). Next: focus on the isolated decode/FMA cost rather than assuming one matrix sweep fixes it.
2. **Exact experts:** small independent routed groups underfill the GPU. Q2_0 two-call, 19-row benchmark: legacy serial 81.364 µs, tuned serial 61.525, joint 49.436, all 43,776 outputs exact. The later code trace actually issued **654** joint launches; the exact-expert bucket fell from 136.434 to 130.138 ms versus the preceding release, but its whole five-family trace was **805.772 vs 785.651 ms**. Do not claim a total model speedup from this incremental feature. A controlled IQ2_XXS M8 ablation still took 85.165 µs with a repeated packed panel vs 74.997 normal and 23.360 with a cached scalar. Next: qualify larger independent groups only where the routed schedule admits them.
3. **WMMA experts:** table lookup and operand conversion matter. Shared IQ1_S codebook cut M64 from 157.661 to 63.552 µs; 2×2 register fragments in the panel alternative lost to 32×32. The latest code trace still spent 183.147 ms on expert WMMA including packing. Separately, its first-to-last-kernel window contained **931.872 ms** of copies and **3268.918 ms** with no kernel or copy from this process (`timeline-mlp-code.log`). These are not extra kernel costs. The 512 ms whole-request budget requires reducing this non-kernel interval as well.

**Scope and unfinished work.** Hardware counters, live HTTP serving, and a full-request replay of the exact historical prompt were not obtained. The latest code sample series is complete (`mlp-code-quality.log`); use the separately retained `mlp-prose-quality.log` only once its four-trial comparison completes. The preceding fully qualified release reached 99.30 prose tok/s (four-run median, `ragged-samples.log`). Separately from code quality, both latest forced-prefix oracles are complete.

**Evidence and provenance.** Everything is under `inkling-prefill-weight-gemm/`: command scripts, raw CSV/stderr, Nsight SQLite, per-run commands/flags, hashes and exit statuses in `evidence/`; fixture/code in `wt/engine/tools/inkling-prefill-weight-gemm/`. Every device command used the queueing runner; no other holder was disturbed. Branch `task/inkling-prefill-weight-gemm`, base `54389798b1dd594531de10ad9032bfca382c74b3`, inherited `3ab8443e613adc90b779cf3843c08bcd9a289082`. Implementation milestones: `9fa3c0c6df7c6ae5bc5cb17855b9998db09af432`, `65ac6449cacb267d3bd5c9d50afd51e59b10f2c9`, `05b6bbce80c14bed7dfd02a32d9b378f401d06da`, `7d4ccccd839ebfb0a4f802ef2fa95e72c24e9985`, `1a88316f6b747243b3e432ffdf0cdc7eb622c4ab`, `f4435bf4f646de6de28ec1ff74471f157f8ae78c`, `6d2d5c2070ec54088cc2a7e94a1ee51c7a631c31`, `4f5fd4d4676869e1e6e48933748aea0a8889e515`, `3fb0c25657802777a1365f9c7302a3069ca276f9`, `258d7552fb9a5dbca1ba9369dfbdb306897a80b5`, `93d1735b00eee18213164f36a934579ea4e535b1`. Not pushed.
