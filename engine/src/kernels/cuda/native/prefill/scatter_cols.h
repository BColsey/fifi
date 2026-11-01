/* Launch entry of scatter_cols: copy a contiguous f32 block of `rows` rows of `n` values into
 * the columns col_off..col_off + n of the rows of a wider f32 matrix. The EXL3 projections write
 * their output contiguous, while the chunk's fused projection rows (the DeltaNet and attention
 * input rows) and the gate/up rows of the SwiGLU hold several projections side by side.
 * Launch-only: nothing here allocates, copies or synchronizes.
 *
 * dst[r * ld + col_off + c] = src[r * n + c] for r < rows, c < n; no other dst byte is written.
 *
 * src f32 [rows][n], read only. dst f32 [rows][ld] (the extent is (rows - 1) * ld + col_off + n
 * floats).
 * Rules (else INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call): stream, src, dst not NULL and
 * 16-byte aligned (the kernel moves float4); rows 1..16384; n a multiple of 4 in 4..65536;
 * col_off and ld multiples of 4 with ld >= col_off + n; the two extents share no byte.
 * The arguments carry no byte lengths: the caller proves that both ranges hold their extents
 * (the Rust wrapper does), and the entry cannot check them.
 */
#ifndef INFERENCE_CUDA_PREFILL_SCATTER_COLS_H
#define INFERENCE_CUDA_PREFILL_SCATTER_COLS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_scatter_cols_args {
    const void *src; /* f32 [rows][n], read only */
    void *dst;       /* f32 [rows][ld] out */
    int rows;        /* 1..16384 */
    int n;           /* a multiple of 4 in 4..65536 */
    int ld;          /* a multiple of 4, >= col_off + n */
    int col_off;     /* a multiple of 4 */
} inference_cuda_scatter_cols_args;

int inference_cuda_scatter_cols_launch(const inference_cuda_scatter_cols_args *args, void *stream,
                                       int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_scatter_cols_args) == 32 &&
              offsetof(inference_cuda_scatter_cols_args, src) == 0 &&
              offsetof(inference_cuda_scatter_cols_args, dst) == 8 &&
              offsetof(inference_cuda_scatter_cols_args, rows) == 16 &&
              offsetof(inference_cuda_scatter_cols_args, n) == 20 &&
              offsetof(inference_cuda_scatter_cols_args, ld) == 24 &&
              offsetof(inference_cuda_scatter_cols_args, col_off) == 28,
              "inference_cuda_scatter_cols_args layout");
#else
_Static_assert(sizeof(inference_cuda_scatter_cols_args) == 32 &&
               offsetof(inference_cuda_scatter_cols_args, src) == 0 &&
               offsetof(inference_cuda_scatter_cols_args, dst) == 8 &&
               offsetof(inference_cuda_scatter_cols_args, rows) == 16 &&
               offsetof(inference_cuda_scatter_cols_args, n) == 20 &&
               offsetof(inference_cuda_scatter_cols_args, ld) == 24 &&
               offsetof(inference_cuda_scatter_cols_args, col_off) == 28,
               "inference_cuda_scatter_cols_args layout");
#endif

#endif /* INFERENCE_CUDA_PREFILL_SCATTER_COLS_H */
