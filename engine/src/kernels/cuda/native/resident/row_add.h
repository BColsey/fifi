/*
 * Launch entry of row_add for the Gemma 4 Humanizer residual sums: one kernel that
 * computes out = x + y element-wise in f32. Launch-only: nothing here allocates,
 * copies or synchronizes.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ROW_ADD_H
#define INFERENCE_CUDA_RESIDENT_ROW_ADD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_row_add_args {
    void *out;       /* f32 [n], written, disjoint from x and y */
    const void *x;   /* f32 [n], read */
    const void *y;   /* f32 [n], read */
    int n;           /* values, a multiple of 32, 32..4194304 */
    /* Producer-fusion arm: 1 stores the bf16 roundtrip each published value
     * gets (nonfinite becomes 0 with flag bit 0 set, matching the separate
     * publish kernel); 0 stores the plain sum as before. */
    unsigned *flag;
    size_t flag_bytes;
    uint32_t fused_publish;
} inference_cuda_row_add_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_row_add_args) == 56 && offsetof(inference_cuda_row_add_args, out) == 0 && offsetof(inference_cuda_row_add_args, x) == 8 && offsetof(inference_cuda_row_add_args, y) == 16 && offsetof(inference_cuda_row_add_args, n) == 24 && offsetof(inference_cuda_row_add_args, flag) == 32 && offsetof(inference_cuda_row_add_args, fused_publish) == 48, "layout");
#else
_Static_assert(sizeof(inference_cuda_row_add_args) == 56 && offsetof(inference_cuda_row_add_args, out) == 0 && offsetof(inference_cuda_row_add_args, x) == 8 && offsetof(inference_cuda_row_add_args, y) == 16 && offsetof(inference_cuda_row_add_args, n) == 24 && offsetof(inference_cuda_row_add_args, flag) == 32 && offsetof(inference_cuda_row_add_args, fused_publish) == 48, "layout");
#endif

/* One launch on stream (must not be NULL). out is written; x and y are read.
 * Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument,
 * else the launch status. */
int inference_cuda_row_add_launch(const inference_cuda_row_add_args *args, void *stream,
                                  int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ROW_ADD_H */
