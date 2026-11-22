/*
 * Launch entries of the per-layer-embedding block of a decode step: decode the rows of an
 * IQ4_NL embedding table, gate the projected rows against the residual stream, and run the
 * causal convolution with its ring. Three launch-only entries on an explicit stream; nothing
 * here allocates, copies or synchronizes. The arithmetic follows the host reference
 * (family/flash_next/ple.rs `step`) so the host oracle checks every kernel.
 */
#ifndef INFERENCE_CUDA_RESIDENT_PLE_BLOCK_H
#define INFERENCE_CUDA_RESIDENT_PLE_BLOCK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decodes `rows` rows of `head_dim` IQ4_NL values (18 bytes per 32 values) and multiplies
 * them by `scale`: out f32 [rows][head_dim]. head_dim is a multiple of 32, 32..8192. */
typedef struct inference_cuda_ple_rows_args {
    float *out;
    const void *raw;
    float scale;
    int rows;
    int head_dim;
} inference_cuda_ple_rows_args;

/* The gate of one token. Per branch b of `branches`, over `hidden` values: kn = rms0(keys_b,
 * norm_key_b), qn = rms0(hyper_b, norm_query_b), d = dot(kn, qn) / sqrt(hidden),
 * shaped = copysign(sqrt(max(|d|, 1e-6)), d), g = sigmoid(shaped); gated_b = g * value;
 * normed_b = rms0(gated_b, norm_conv_b). rms0(x, w) = x * (1 + w) / sqrt(mean(x^2) + eps).
 * keys, hyper, norm_*, gated and normed are f32 [branches * hidden]; value is f32 [hidden]. */
typedef struct inference_cuda_ple_gate_args {
    const float *keys;
    const float *hyper;
    const float *value;
    const float *norm_key;
    const float *norm_query;
    const float *norm_conv;
    float *gated;
    float *normed;
    float eps;
    int branches;
    int hidden;
} inference_cuda_ple_gate_args;

/* The convolution of one token over `width` channels. Channel c: a = conv[c][taps] *
 * normed[c] + sum over k < taps of conv[c][k] * ring[c][k * ngram], out[c] = gated[c] +
 * silu(a); then ring[c] shifts left by one and its last value becomes normed[c]. conv is f32
 * [width][conv_kernel] (taps = conv_kernel - 1), ring f32 [width][taps * ngram], read and
 * written in place. */
typedef struct inference_cuda_ple_conv_args {
    const float *gated;
    const float *normed;
    const float *conv;
    float *ring;
    float *out;
    int width;
    int conv_kernel;
    int ngram;
} inference_cuda_ple_conv_args;

int inference_cuda_ple_rows_launch(const inference_cuda_ple_rows_args *args, void *stream,
                                   int *out_stage);
int inference_cuda_ple_gate_launch(const inference_cuda_ple_gate_args *args, void *stream,
                                   int *out_stage);
int inference_cuda_ple_conv_launch(const inference_cuda_ple_conv_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_PLE_BLOCK_H */
