/* The hyper-connection (four-branch gated residual) read and apply of Qwen3.8 Flash-Next,
 * per-kernel header (native/resident/hyper.h). Included by hyper.cu and read by the Rust FFI
 * mirror; nothing else.
 *
 * The residual stream of a token is f32 [branches][hidden] (width = branches * hidden), one row
 * per token. The weights of one hyper-connection site are dense, all in one dtype (0: binary16,
 * 1: bfloat16): norm_w [width], down [rank][width], up [width][rank], and inject
 * [branches][width] (NULL for the final mixer, which has none). Every product accumulates in f32.
 *
 * read, per token t and with C = branches:
 *   normed[t][b][d] = x[t][b][d] * (1 + norm_w[b][d]) / sqrt(mean_d(x[t][b][d]^2) + eps)
 *   low[t][r]       = silu(dot(down[r], normed[t]) / C)
 *   mix[t][i]       = dot(up[i], low[t])
 *   mixed[t][d]     = sum_b sigmoid(mix[t][b][d]) * normed[t][b][d] / C
 *   gate[t][b]      = 2 * sigmoid(dot(inject[b], normed[t]) / C)      (only with inject)
 * normed, low and mix are caller scratch; mixed [m][hidden] and gate [m][branches] are outputs.
 *
 * apply, per token: x[t][b][d] += gate[t][b] * block[t][d], in place.
 *
 * Limits: m in [1, 32768], branches in [1, 16], hidden in [1, 65536], width <= 262144,
 * rank in [1, 4096]. f32 ranges are 4-byte aligned, weights 2-byte aligned. Written ranges
 * overlap no other range (weights and x may be read by several launches). Each entry enqueues
 * its kernels on stream (not NULL) and returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch
 * for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_HYPER_H
#define INFERENCE_CUDA_RESIDENT_HYPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_hyper_read_args {
    const void *x;
    const void *norm_w;
    const void *down;
    const void *up;
    const void *inject;
    void *normed;
    void *low;
    void *mix;
    void *mixed;
    void *gate;
    int m;
    int branches;
    int hidden;
    int rank;
    int weight_dtype;
    float eps;
} inference_cuda_hyper_read_args;

typedef struct inference_cuda_hyper_apply_args {
    void *x;
    const void *gate;
    const void *block;
    int m;
    int branches;
    int hidden;
} inference_cuda_hyper_apply_args;

int inference_cuda_hyper_read_launch(const inference_cuda_hyper_read_args *args, void *stream,
                                     int *out_stage);

int inference_cuda_hyper_apply_launch(const inference_cuda_hyper_apply_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
