# Mac engine and packed projection coverage

This lane uses the 16 GB M4 Mac mini. Its Metal working-set recommendation is 11.84 GiB. The verified Flint INT4 checkpoint is available on the internal SSD. The other native MLX families below have local test fixtures. The published Qwen3.6 MLX checkpoint alone is about 20 GB before serving workspace. [Model registry and checkpoint formats](../reference/models.md) · [Device sizing](mlx-device-scaling.md)

| Engine | Mac execution path | Evidence available on this mini |
| --- | --- | --- |
| Flint | Native MLX with publisher affine INT4 weights, INT8 embeddings and disk-streamed routed experts | Published checkpoint through the server, plus packed projection and grouped-expert fixtures |
| Qwen3.6-35B-A3B | Native MLX MoE with affine safetensors and shared packed projections | Worker-level synthetic requests and resident execution fixtures |
| Qwen3.8 Flash-Next | Native MLX with disk-staged MoE and manifest-driven packed weights | Worker-level synthetic requests and an affine 4-bit fixture through the real factory |
| Qwen3.8-27B dense | Native MLX dense executor in the Qwen family | Dense decode and prefill fixture, with host and device routing settings |
| Nemotron 3.5 Super | CPU reference for NVFP4. The published serving profile uses CUDA | CPU packed-expert fixtures |
| Gemma4 and fake | Gemma4 has a staged registry entry. Fake is a fixed-response test engine | Startup and contract tests |

## Same-shape reshape

For an unchanged shape, the pinned MLX implementation returns its input array. The Rust adapter still crossed the native boundary and validated a new wrapper around that array. With `FIFI_MLX_REUSE_IDENTITY_RESHAPE=1`, it validates readiness, ownership, dtype, shape and byte length, then shares the existing resident handle. Shape-changing calls use the normal native reshape. This switch is opt-in for the shared MLX adapter.

The projection benchmark uses 2,048-by-2,048 resident weights, one or 32 activation rows, and F32 or BF16 activations. Its 80 cases cover affine 2-, 3-, 4-, 5-, 6- and 8-bit weights at group sizes 32, 64 and 128, and MXFP4/MXFP8 at group size 32. Each case compares projection output bytes with and without a same-shape reshape. The driver checks the output digest across fresh processes and captures both submission time and time through synchronized evaluation. These are operator-level measurements. Model loaders and publishers supply distinct checkpoint profiles and require their own serving and quality measurements.

## Results on the mini

The release projection test ran 12 samples per case in each of four fresh processes, ordered control, candidate, candidate, control. All 320 case records matched their controls, including both activation dtypes and both batch sizes. The values below are median paired changes from each case's process medians. Submission measures CPU construction of the reshape and QMM. The synchronized interval includes waiting for the output.

| Packed weights | Case pairs | Host submission | Synchronized projection |
| --- | ---: | ---: | ---: |
| Affine, all tested bit and group sizes | 144 | −31.5% | −1.9% |
| MXFP4 | 8 | −35.4% | +2.3% |
| MXFP8 | 8 | −17.6% | +8.4% |
| All formats | 160 | −31.5% | −0.5% |

The median host submission time across cases fell from 0.71 to 0.48 µs. It improved in 150 of 160 pairs. The synchronized interval improved in 86 of 160 pairs. MXFP8's synchronized measurements were slower in seven of eight pairs. The change remains experimental and default-off. [Projection data](../../scripts/benchmarks/mac-m4-mlx-projections.csv)

There were 128 synthetic worker requests across four fresh process pairs per family. Every sequence of continuation tokens matched. The table shows median paired changes across the four pairs at each prompt length.

| Fixture | Prompt tokens | Time to first token | Steady decode time |
| --- | ---: | ---: | ---: |
| Qwen3.6 MoE | 8 | −4.3% | −3.5% |
| Qwen3.6 MoE | 33 | −10.1% | −1.7% |
| Flash-Next | 8 | −0.2% | +6.0% |
| Flash-Next | 33 | +5.9% | −6.6% |

All four Qwen3.6 pairs at 33 prompt tokens reached their first token sooner. Flash-Next varied by prompt length and phase. Median peak active allocation varied by under 1% in these worker fixtures. [Worker request data](../../scripts/benchmarks/mac-m4-mlx-reshape-requests.csv)

The targeted native library suite passed eight checks in each mode. It includes Flash-Next and Qwen3.6 session and prefill behavior, Qwen3.8 dense decode and prefill, Flint streamed expert parity, the Nemotron NVFP4 CPU reference, and GGUF tensor dequantization. A separate check passed in both modes for wrong-owner rejection, BF16 admission, shape validation, and U32 output.

On the published Flint INT4 checkpoint, four short requests through the normal release server returned identical completions across both reshape modes and both prefill reader policies. The four requests check output parity under the production build.

## Reproduce

Build in `engine/` and use the two executable paths printed by Cargo. Profile on an idle Mac with a fresh scratch output directory. Use `--smoke --samples 1` for the first projection check.

```sh
cargo test --locked --release --features native-mlx,native-fault-qualification \
  --test mlx_projection_matrix --lib --no-run
python3 ../scripts/bench_mlx_projections.py "$PROJECTION_TEST" "$RUN_DIR/projections.csv"
python3 ../scripts/bench_mlx_requests.py "$LIB_TEST" "$RUN_DIR/requests.csv" \
  --identity-reshape --pairs 4 --samples 4
```

The request driver fixes the Flash-Next chunked and Qwen3.6 grouped-embedding configurations across both arms. It checks complete continuation token sequences and reports first-token time, steady decode and memory alongside the projection results.
