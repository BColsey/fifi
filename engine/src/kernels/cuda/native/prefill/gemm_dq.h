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
    const void *x; /* f16 activation: row-major [k][ncols], or token rows
                    * [ncols][x_col_stride] when x_col_stride is set */
    float *y;      /* f32 out: row-major [rows][ncols], or token rows
                    * [ncols][y_col_stride] when y_col_stride is set */
    int k;         /* a positive multiple of the type's block size */
    int rows;      /* positive */
    int ncols;     /* positive */
    int x_col_stride; /* 0 reads x row-major by k; else >= k: x[col][row_k] */
    int y_col_stride; /* 0 writes y row-major by rows; else >= rows: y[col][row] */
} inference_cuda_gemm_dq_args;

/* One launch on stream (must not be NULL) with the Q5_K decoder: k must be a multiple of 256
 * (refused with INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call otherwise, as is any
 * other bad argument), w holds rows * k/256 * 176 bytes and must be 16-byte aligned (the
 * staging moves blocks as 16-byte words), x holds k * ncols f16 values, y holds rows * ncols
 * floats and shares no byte with w or x. Returns the launch status. */
int inference_cuda_gemm_dq_q5k_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                      int *out_stage);

/* The same launch with the Q4_0 decoder: k must be a multiple of 64, w holds
 * rows * k/32 * 18 bytes (2-byte aligned is enough), x and y as above. */
int inference_cuda_gemm_dq_q40_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                      int *out_stage);

/* The same launch with the Q3_K decoder: k must be a multiple of 256, w holds
 * rows * k/256 * 110 bytes, x and y as above. */
int inference_cuda_gemm_dq_q3k_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                      int *out_stage);

/* The same launch with the IQ4_XS decoder: k must be a multiple of 256, w holds
 * rows * k/256 * 136 bytes, x and y as above. */
int inference_cuda_gemm_dq_iq4xs_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                         int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_gemm_dq_args) == 48 &&
                  offsetof(inference_cuda_gemm_dq_args, w) == 0 &&
                  offsetof(inference_cuda_gemm_dq_args, x) == 8 &&
                  offsetof(inference_cuda_gemm_dq_args, y) == 16 &&
                  offsetof(inference_cuda_gemm_dq_args, k) == 24 &&
                  offsetof(inference_cuda_gemm_dq_args, rows) == 28 &&
                  offsetof(inference_cuda_gemm_dq_args, ncols) == 32 &&
                  offsetof(inference_cuda_gemm_dq_args, x_col_stride) == 36 &&
                  offsetof(inference_cuda_gemm_dq_args, y_col_stride) == 40,
              "inference_cuda_gemm_dq_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_H */
