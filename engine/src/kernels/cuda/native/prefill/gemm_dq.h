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
    int x_f32;        /* 1: x holds f32 values (converted on load); 0: f16 */
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

/* The same launch for the codebook types (GGUF ids 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS,
 * 20 IQ4_NL, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS, 42 Q2_0, and 10 Q2_K, 19 IQ1_S, 29 IQ1_M): k must be a multiple of the type's block size and
 * of 64, w is 2-byte aligned. table points at the type's lookup table words
 * (weights/gguf_tables.rs) and table_bytes is their exact byte count (0 for Q2_0); they are
 * copied into the launch arguments before the call returns. */
int inference_cuda_gemm_dq_codebook_launch(const inference_cuda_gemm_dq_args *args, int type,
                                           const void *table, size_t table_bytes, void *stream,
                                           int *out_stage);

/* The grouped (per-expert) launch of the GEMM for the GGUF type `type` (2 Q4_0, 11 Q3_K,
 * 13 Q5_K, and every type of the codebook entry above plus 23 IQ4_XS and 10 Q2_K, 19 IQ1_S,
 * 29 IQ1_M). w holds n_experts weights expert_stride bytes apart (16-byte aligned, a multiple
 * of 16), each rows x k native blocks; x holds f32 token rows of x_row_stride floats; y holds
 * f32 slot rows of y_row_stride floats. expert_bounds holds n_experts + 1 ints and ids_dst
 * total_cols ints: compact column j of expert e (expert_bounds[e] <= j < expert_bounds[e + 1])
 * is token slot ids_dst[j], which reads x row slot / x_div and writes y row slot (columns
 * 0..rows). table and table_bytes are as for the codebook entry. */
typedef struct inference_cuda_gemm_dq_mmid_args {
    const void *w;
    const float *x;
    float *y;
    const int *ids_dst;
    const int *expert_bounds;
    unsigned long long expert_stride;
    int k;
    int rows;
    int n_experts;
    int total_cols; /* m * top_k: every slot of the launch */
    int x_div;      /* slot / x_div is the activation row: top_k for the gate and up input, 1 for the down input */
    int x_row_stride;
    int y_row_stride;
    int pad;
} inference_cuda_gemm_dq_mmid_args;

int inference_cuda_gemm_dq_mmid_launch(const inference_cuda_gemm_dq_mmid_args *args, int type,
                                       const void *table, size_t table_bytes, void *stream,
                                       int *out_stage);

/* The same launch with the Q3_K decoder: k must be a multiple of 256, w holds
 * rows * k/256 * 110 bytes, x and y as above. */
int inference_cuda_gemm_dq_q3k_launch(const inference_cuda_gemm_dq_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_gemm_dq_mmid_args) == 80 &&
                  offsetof(inference_cuda_gemm_dq_mmid_args, ids_dst) == 24 &&
                  offsetof(inference_cuda_gemm_dq_mmid_args, expert_stride) == 40 &&
                  offsetof(inference_cuda_gemm_dq_mmid_args, k) == 48 &&
                  offsetof(inference_cuda_gemm_dq_mmid_args, y_row_stride) == 72,
              "inference_cuda_gemm_dq_mmid_args layout");
static_assert(sizeof(inference_cuda_gemm_dq_args) == 48 &&
                  offsetof(inference_cuda_gemm_dq_args, w) == 0 &&
                  offsetof(inference_cuda_gemm_dq_args, x) == 8 &&
                  offsetof(inference_cuda_gemm_dq_args, y) == 16 &&
                  offsetof(inference_cuda_gemm_dq_args, k) == 24 &&
                  offsetof(inference_cuda_gemm_dq_args, rows) == 28 &&
                  offsetof(inference_cuda_gemm_dq_args, ncols) == 32 &&
                  offsetof(inference_cuda_gemm_dq_args, x_col_stride) == 36 &&
                  offsetof(inference_cuda_gemm_dq_args, y_col_stride) == 40 &&
                  offsetof(inference_cuda_gemm_dq_args, x_f32) == 44,
              "inference_cuda_gemm_dq_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_H */
