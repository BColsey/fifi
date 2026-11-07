/* Dense half-precision matvec of the Flash-Next step, per-kernel header
 * (native/resident/dense_gemv.h). Included by dense_gemv.cu and read by the Rust FFI mirror;
 * nothing else.
 *
 * For each of m activation rows t and each weight row r:
 *   y[t * ld_y + r] = sum_c w[r][c] * x[t * ld_x + c]
 * with w a dense row-major [rows][cols] matrix in its stored dtype (0: binary16, 1: bfloat16),
 * widened exactly to f32 in registers, x and y f32. Products and sums are f32: a lane adds its
 * columns in ascending order with fused multiply-adds, and the 32 lanes of a warp combine by a
 * fixed xor butterfly, so a result depends on cols and the alignment of w, never on m, rows or
 * the other rows. `ld_y` lets the result land inside a wider row; a result element is written
 * exactly once. The one user is the PLE key and value projection of Flash-Next ([10240, 2560] and [2560, 2560], F16).
 *
 * Limits: m in [1, 32768], rows in [1, 1048576], cols in [1, 12288] (a row of x is staged in
 * shared memory), ld_x in [cols, 1048576], ld_y in [rows, 1048576]. x and y are 4-byte aligned,
 * w 2-byte aligned (a 16-byte aligned w with cols % 8 == 0 takes a vector path). y overlaps
 * neither x nor w (x and w are read only). Enqueues one kernel on stream (not NULL) and returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_DENSE_GEMV_H
#define INFERENCE_CUDA_RESIDENT_DENSE_GEMV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_dense_gemv_args {
    const void *w;
    const void *x;
    void *y;
    int m;
    int rows;
    int cols;
    int ld_x;
    int ld_y;
    int weight_dtype;
} inference_cuda_dense_gemv_args;

int inference_cuda_dense_gemv_launch(const inference_cuda_dense_gemv_args *args, void *stream,
                                     int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
