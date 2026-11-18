/*
 * Launch entry of the in-house IQ4_XS decode matvec: y = x_q8 . W^T over the file's native
 * 136-byte IQ4_XS super-blocks (no repack, no vendored code). One warp per (weight row,
 * activation column); f32 arithmetic throughout. Launch-only: nothing here allocates, copies
 * or synchronizes. Per-kernel layout notes in kernels/cuda/resident/matvec_q.rs.
 *
 * Every includer is a C++ TU of the native adapter, so the layout is pinned with a C++
 * static_assert here and with the same numbers by the Rust mirror
 * (native/resident/matvec_iq4xs.rs).
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_IQ4XS_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_IQ4XS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_iq4xs_args {
    const void *w;    /* native IQ4_XS super-blocks [rows][k / 256], 136 bytes each */
    const void *x_q8; /* block_q8_1 [ncols][pad(k, 512) / 32], 36 bytes each */
    float *y;         /* f32 out [ncols][rows], column j at y + j * rows */
    int k;            /* a positive multiple of 256 */
    int rows;         /* any positive count (the kernel owns whole rows) */
    int ncols;        /* 1..8 */
} inference_cuda_matvec_iq4xs_args;

/* One launch on stream (must not be NULL). The weights are read in the file's native block
 * layout; x_q8 is the padded quantize_q8_1 output of ncols activation rows, whose per-row
 * stride (padded blocks x 36) the caller passes in x_row_bytes. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else the launch
 * status. Column j is computed from x_q8 row j alone, like the vendored matvec's contract. */
int inference_cuda_matvec_iq4xs_launch(const inference_cuda_matvec_iq4xs_args *args,
                                       size_t x_row_bytes, void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_matvec_iq4xs_args) == 40 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, w) == 0 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, x_q8) == 8 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, y) == 16 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, k) == 24 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, rows) == 28 &&
                  offsetof(inference_cuda_matvec_iq4xs_args, ncols) == 32,
              "inference_cuda_matvec_iq4xs_args layout");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_IQ4XS_H */
