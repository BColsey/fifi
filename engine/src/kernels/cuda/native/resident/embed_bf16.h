/*
 * Launch entry of embed_bf16: gather m rows of a BF16 table (n_rows rows of k values, stored as
 * the 16-bit codes of the checkpoint) into m f32 rows. The widening is exact: the code is the
 * upper half of the f32 bit pattern. Launch-only: nothing here allocates, copies or synchronizes.
 *
 *   w        u16 [n_rows][k], read, 4-byte aligned.
 *   ids      i32 [m], read when the kernel runs, 4-byte aligned (the token id word of StepParams).
 *   x        f32 [m][k], written in full, 16-byte aligned.
 *   status   one device i32, 4-byte aligned. An id outside [0, n_rows) writes a row of NaN, reads
 *            no table byte and ORs status bit 4 (the invalid-step-state bit of the step, the rule
 *            of embed.h) with one integer atomicOr. Valid ids leave it alone.
 *   m in [1, 4096]; n_rows in [1, 2^28]; k a multiple of 8 in [8, 32768].
 * x and status overlap no other range. Rejected arguments return INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call.
 */
#ifndef INFERENCE_CUDA_RESIDENT_EMBED_BF16_H
#define INFERENCE_CUDA_RESIDENT_EMBED_BF16_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_embed_bf16_args {
    const void *w;
    const int *ids;
    float *x;
    int *status;
    int m;
    int n_rows;
    int k;
} inference_cuda_embed_bf16_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_embed_bf16_args) == 48 &&
                  offsetof(inference_cuda_embed_bf16_args, w) == 0 &&
                  offsetof(inference_cuda_embed_bf16_args, ids) == 8 &&
                  offsetof(inference_cuda_embed_bf16_args, x) == 16 &&
                  offsetof(inference_cuda_embed_bf16_args, status) == 24 &&
                  offsetof(inference_cuda_embed_bf16_args, m) == 32 &&
                  offsetof(inference_cuda_embed_bf16_args, n_rows) == 36 &&
                  offsetof(inference_cuda_embed_bf16_args, k) == 40,
              "inference_cuda_embed_bf16_args layout");
#else
_Static_assert(sizeof(inference_cuda_embed_bf16_args) == 48 &&
                   offsetof(inference_cuda_embed_bf16_args, w) == 0 &&
                   offsetof(inference_cuda_embed_bf16_args, ids) == 8 &&
                   offsetof(inference_cuda_embed_bf16_args, x) == 16 &&
                   offsetof(inference_cuda_embed_bf16_args, status) == 24 &&
                   offsetof(inference_cuda_embed_bf16_args, m) == 32 &&
                   offsetof(inference_cuda_embed_bf16_args, n_rows) == 36 &&
                   offsetof(inference_cuda_embed_bf16_args, k) == 40,
               "inference_cuda_embed_bf16_args layout");
#endif

int inference_cuda_embed_bf16_launch(const inference_cuda_embed_bf16_args *args, void *stream,
                                     int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_EMBED_BF16_H */
