/*
 * Launch entry of the in-house dequantizing prefill GEMM: y = W . x with W the file's own
 * native quant blocks (dequantized into shared memory inside the kernel, never repacked and
 * never copied to f16) and x an f16 activation. Generic over the per-type decoder of
 * gemm_dq_decoders.cuh; this header declares the Q5_K instantiation. The same GEMM for a new
 * type is one more entry here plus one explicit instantiation in gemm_dq.cu.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like the resident entries. Every includer is a C++ TU of the native adapter,
 * so the layout is pinned with a C++ static_assert here and by the Rust mirror.
 */
#ifndef INFERENCE_CUDA_PREFILL_GEMM_DQ_H
#define INFERENCE_CUDA_PREFILL_GEMM_DQ_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gemm_dq_args {
    const void *w; /* native blocks [rows][k / qk] of the type */
    const void *x; /* f16 activation, row-major [k][ncols] */
    float *y;      /* f32 out, row-major [rows][ncols] */
    int k;         /* a positive multiple of the type's block size */
    int rows;      /* positive */
    int ncols;     /* positive */
} inference_cuda_gemm_dq_args;

/* One launch on stream (must not be NULL) with the Q5_K decoder: k must be a multiple of 256
 * (refused with INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call otherwise, as is any
 * other bad argument), w holds rows * k/256 * 176 bytes, x holds k * ncols f16 values, y
 * holds rows * ncols floats and shares no byte with w or x. Returns the launch status. */
int inference_cuda_gemm_dq_q5k_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_gemm_dq_args) == 40 &&
                  offsetof(inference_cuda_gemm_dq_args, w) == 0 &&
                  offsetof(inference_cuda_gemm_dq_args, x) == 8 &&
                  offsetof(inference_cuda_gemm_dq_args, y) == 16 &&
                  offsetof(inference_cuda_gemm_dq_args, k) == 24 &&
                  offsetof(inference_cuda_gemm_dq_args, rows) == 28 &&
                  offsetof(inference_cuda_gemm_dq_args, ncols) == 32,
              "inference_cuda_gemm_dq_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_H */
