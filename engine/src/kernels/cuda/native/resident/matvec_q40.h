/*
 * Launch entry of the in-house Q4_0 decode matvec: y = x_q8 . W^T over the file's native
 * 18-byte Q4_0 blocks (no repack, no vendored code). One warp per (weight row, activation
 * column); f32 arithmetic throughout. Launch-only: nothing here allocates, copies or
 * synchronizes. Per-kernel layout notes in kernels/cuda/resident/matvec_q.rs.
 *
 * Every includer is a C++ TU of the native adapter, so the layout is pinned with a C++
 * static_assert here and with the same numbers by the Rust mirror
 * (native/resident/matvec_q40.rs).
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_Q40_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_Q40_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_q40_args {
    const void *w;    /* native Q4_0 blocks [rows][k / 32], 18 bytes each */
    const void *x_q8; /* block_q8_1 [ncols][pad(k, 512) / 32], 36 bytes each */
    float *y;         /* f32 out [ncols][rows], column j at y + j * rows */
    int k;            /* a positive multiple of 32 */
    int rows;         /* any positive count (the kernel owns whole rows) */
    int ncols;        /* 1..8 */
} inference_cuda_matvec_q40_args;

/* One launch on stream (must not be NULL). The weights are read in the file's native block
 * layout; x_q8 is the padded quantize_q8_1 output of ncols activation rows, whose per-row
 * stride (padded blocks x 36) the caller passes in x_row_bytes. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the launch
 * status. Column j is computed from x_q8 row j alone, like the vendored matvec's contract. */
int inference_cuda_matvec_q40_launch(const inference_cuda_matvec_q40_args *args,
                                     size_t x_row_bytes, void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_matvec_q40_args) == 40 &&
                  offsetof(inference_cuda_matvec_q40_args, w) == 0 &&
                  offsetof(inference_cuda_matvec_q40_args, x_q8) == 8 &&
                  offsetof(inference_cuda_matvec_q40_args, y) == 16 &&
                  offsetof(inference_cuda_matvec_q40_args, k) == 24 &&
                  offsetof(inference_cuda_matvec_q40_args, rows) == 28 &&
                  offsetof(inference_cuda_matvec_q40_args, ncols) == 32,
              "inference_cuda_matvec_q40_args layout");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_Q40_H */
