# fifi and llama.cpp on the same weights: how far apart their next-token distributions are

The record of the run is [kld-vs-llamacpp-20261125.json](../../scripts/benchmarks/kld-vs-llamacpp-20261125.json). The scripts that made it are in [scripts/kld_vs_llamacpp](../../scripts/kld_vs_llamacpp/README.md).

## The question

Three KL figures were being set side by side, and they did not measure the same thing:

- Flint's q8 KV profile against its f16 profile, teacher-forced on 8,448 repository-source positions:
  a mean of 0.254 nats on the legacy kernel and 0.266 on the tiled kernel, which the profile uses by
  default (`doc/reference/environment.md`, the `FIFI_FLINT_CUDA_Q8_TILED` row).
- llama.cpp's q8_0 KV cache against its f16 cache: published by others for this model on other corpora with other scorers.
- the engine against llama.cpp: not measured.

The first two compare an engine with itself under two cache forms. Neither says how far the two engines are from each other. This run measures both kinds in one harness: one GGUF file, one token stream, one metric, both engines, and each engine against itself under a second numerical path.

## What ran

| Arm | What scores the positions |
|---|---|
| `engine-q8` | The engine's `kv_quality_leg` with the int8 (`Int8G32`) KV cache: the cache form of the shipped 3090 profiles, `qwen36-cuda-3090-kv8-prefix.json` among them. |
| `engine-f16` | The same leg with the fp16 KV cache: the control cache. On the CUDA arm a start keeps the cache this way for `kv_dtype: f32`, the member's default (`settings.rs`, `kv_dtype_of`; `start.rs`, `kv_mode_of`), so it is the form of the `qwen36-rtx3090-24gb-fp16kv.json` profile and of a start whose config names no `kv_dtype`. |
| `llama-ub1` | A driver on `libllama`: f16 KV cache under flash attention (llama.cpp's `auto` setting, which resolved to on), the prefix in 512-token batches, then one-token decodes for the scored positions. |
| `llama-ub512` | The same driver with the scored positions inside 512-token batches: llama.cpp against itself under a second path. |
| `llama-q8` | The same driver as `llama-ub1`, with llama.cpp's q8_0 KV cache: llama.cpp against itself under a second cache form. |

All arms read the same file (`Qwen3.6-35B-A3B-Q4_K_M.gguf`, SHA-256 in the record) through the same token ids. The ids are the twelve documents of `qualification/flash-next-quality-corpus`, tokenized once; llama.cpp's tokenizer gives the same ids. Each document is one window of 2,402 to 32,768 tokens with 768 scored positions: 9,216 positions in total.

- Engine: `main` at `5ad3798aa`, release build with the `native-cuda` feature.
- llama.cpp: `acecd5603`, the CUDA backend.
- Card: one RTX 3090.

Each row is narrowed to the sampler vocabulary (248,077 ids) and normalized in f64. For a pair of arms, `KL(a || b)` is summed over the 256 most probable ids of a and `KL(b || a)` over those of b. The terms outside those rows, where about 1.4% of the probability mass lies, are left out and are signed, so neither figure is a bound on the KL over the whole vocabulary. For llama.cpp's two cache forms, the pair where both arms keep every logit, the KL over all 248,077 ids is 0.0240 and 0.0236 against the top-row sums 0.0233 and 0.0232: 2% to 3% higher. Bounds are 95% intervals from a bootstrap over the twelve windows.

Each arm was run at least twice on this day: `llama-ub512`, `llama-ub1` and `engine-f16`
directly and then through `scripts/kld_vs_llamacpp/run.sh`, and `llama-q8` twice through `run.sh`, with
every sidecar byte for byte the same. A first run under the `llama-q8` arm name had kept the f16 cache,
because the arm passed no cache type at the time: its twelve sidecars were byte for byte those of
`llama-ub1`, and were discarded. The int8 arm ran once against each reference, with every field but the
two reference terms the same. The last `llama-ub1` and `llama-q8` runs used the driver built from the
committed source (a rebuild from it gives the same binary), and the committed `score_all.sh` on that run
directory produced the score files this record gathers. Against the first scoring of the same arms every
number is the same.

## Results

| Pair | KL, first against second | KL, second against first | Same argmax | NLL of the true token, first minus second | Greedy hits on the true token |
|---|---:|---:|---:|---:|---:|
| engine int8 cache, llama.cpp f16 cache | 0.0363 | 0.0364 | 96.65% | +0.0090 | 66.96% against 67.24% |
| engine fp16 cache, llama.cpp f16 cache | 0.0366 | 0.0351 | 96.80% | +0.0108 | 67.12% against 67.24% |
| llama.cpp, 512-token batches, 1-token decodes | 0.0330 | | 96.88% | +0.0033 | 67.05% against 67.24% |
| engine int8 cache, engine fp16 cache | | 0.0377 | 96.69% | −0.0019 | 66.96% against 67.12% |
| llama.cpp q8_0 cache, llama.cpp f16 cache | 0.0232 | 0.0233 | 96.70% | −0.0003 | 67.08% against 67.24% |

KL in nats per position, means over the 9,216 positions. A blank cell is a direction the harness does not give for that pair: `KL(x || y)` over the top rows of x needs the full logit rows of y, which are kept for `llama-ub1` and `llama-q8`, or the engine's leg with y as its reference.

Means of the NLL over all positions: engine 1.5105 with the int8 cache and 1.5124 with the fp16 cache, llama.cpp 1.5016 with the f16 cache and 1.5012 with the q8_0 cache.

## Where the divergence sits

- **Few positions carry the mean.** The median KL is 0.0006 to 0.001 nats in every pair. Between 0.41%
  and 0.55% of the positions lie above one nat, and they hold 73% to 83% of the sum. Without them the
  means are 0.006 to 0.007.
- **They are the same positions.** Of the 69 positions where the engine with the int8 cache and llama.cpp disagree on the true token's log-probability by more than one nat, 35 are also such positions between llama.cpp's q8_0 and f16 caches, and 39 between llama.cpp's two batching paths. Chance would give about 0.5.
- **Two documents hold most of it.** The window at 6,774 tokens has 0.151 to 0.214 nats in every pair,
  and the second of the seven windows at 32,768 tokens 0.059 to 0.145. Together they hold 71% to 81%
  of each pair's sum. The other ten lie between 0.002 and 0.055.
- **No arm is flatter.** The entropy over the top 256 rows is 1.082 to 1.087 nats in all five arms, the mean probability of the true token 0.604 to 0.605.

## Reading

1. **The engines are as far from each other as each engine is from itself under a numerical change.** Engine against llama.cpp: 0.036 nats and 96.65% agreement. llama.cpp against itself between two batching paths: 0.033 and 96.9%. A KL of this size does not say which of the two distributions is nearer to the model.
2. **The true token's NLL is the one measure with a reference outside the two arms.** It is 0.009 nats higher for the engine with the int8 cache and 0.011 higher with the fp16 cache than for llama.cpp: 0.9% and 1.1% in perplexity. Without the 69 and 78 outlier positions the difference is 0.0015 and 0.0009. The bootstrap intervals are +0.0024 to +0.0162 and +0.0009 to +0.0240.
3. **The int8 cache costs about what llama.cpp's q8_0 cache costs, in this harness.** Against its own fp16 cache the engine's int8 cache gives 0.038, llama.cpp's q8_0 cache 0.023 against its f16 cache. The medians (0.0008 and 0.0009) and the agreement rates (96.69% and 96.70%) are the same; the mean is 1.6 times larger because 0.50% of the positions lie above one nat against 0.41%. Neither cache form moves the NLL: −0.0019 for the engine, −0.0003 for llama.cpp, both intervals across zero.
4. **Figures of this kind compare only inside one harness.** An independent measurement of llama.cpp's q8_0 cache on this model (top-40 log-probabilities, BF16 weights, about 250,000 tokens in six categories) gives 0.039, and 0.024 for the dense Qwen3.6 27B ([localbench, 2026-04-24](https://localbench.substack.com/p/kv-cache-quantization-benchmark)). The 0.023 here is of that size; a figure of one order lower would need a corpus without its outlier positions.

## Limits

- Neither engine was scored against full-precision weights: only the Q4_K_M file was on the host. Every figure is a difference between two runs of the same quantized weights.
- The engine arms run the leg's plain step: the prefix through the chunked prefill path, then one decode step per position. Graph capture, prefix reuse, the pooled start's on-demand KV, and contexts beyond 32,768 of the profile's 180,224 were not exercised.
- Twelve documents and 9,216 positions: the KL intervals are about twice their means wide, and three of the five NLL intervals cross zero.
- Throughput was not measured.
