/* Experimental single-row Q6_K/F32 matvec. Launch-only; not a serving entry point. */
#ifndef INFERENCE_CUDA_RESIDENT_Q6_F32_H
#define INFERENCE_CUDA_RESIDENT_Q6_F32_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_q6_f32_args {
    const void *w;  /* rows * (k / 256) contiguous native 210-byte blocks */
    const float *x; /* k f32 inputs */
    float *y;       /* rows f32 results */
    int rows;
    int k;
} inference_cuda_q6_f32_args;

/* rows > 0, k > 0 and divisible by 256; y disjoint from w and x.
 * The caller retains allocations and stream until queued work completes. */
int inference_cuda_q6_f32_launch(const inference_cuda_q6_f32_args *args, void *stream,
                                 int *out_stage);
#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_q6_f32_args) == 32 &&
                  offsetof(inference_cuda_q6_f32_args, w) == 0 &&
                  offsetof(inference_cuda_q6_f32_args, x) == 8 &&
                  offsetof(inference_cuda_q6_f32_args, y) == 16 &&
                  offsetof(inference_cuda_q6_f32_args, rows) == 24 &&
                  offsetof(inference_cuda_q6_f32_args, k) == 28,
              "inference_cuda_q6_f32_args layout");
#endif
#endif
