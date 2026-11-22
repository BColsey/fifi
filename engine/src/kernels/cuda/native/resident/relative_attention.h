/* One queued F32 relative-attention decode step on an explicit stream. */
#ifndef INFERENCE_CUDA_RESIDENT_RELATIVE_ATTENTION_H
#define INFERENCE_CUDA_RESIDENT_RELATIVE_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_relative_attention_args {
    /* The first four ranges are written and must be disjoint from every
     * other range. The step position is read on the device. */
    float *out;
    float *keys;
    float *values;
    int *status;
    const float *q;
    const float *k;
    const float *v;
    const float *relative;
    const float *table;
    const float *sinks;
    const float *gates;
    const int *pos;
    int query_heads;
    int kv_heads;
    int relative_dims;
    int relative_extent;
    int capacity;
} inference_cuda_relative_attention_args;

int inference_cuda_relative_attention_launch(
    const inference_cuda_relative_attention_args *args, void *stream, int *out_stage);

/* Reads one certified, immutable pre-tile ring, then commits on the same stream.
 * Output and zero-initialized status are packed. The two strides are in 32-bit
 * words: input for Q/K/V/relative/position, side for sinks/gates; zero means
 * tightly packed per plane. Positions must match first + row. No graph replay. */
int inference_cuda_relative_attention_tile_launch(
    const inference_cuda_relative_attention_args *args, int first, int count,
    int input_stride, int side_stride, void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_relative_attention_args) == 120 &&
                  offsetof(inference_cuda_relative_attention_args, query_heads) == 96 &&
                  offsetof(inference_cuda_relative_attention_args, capacity) == 112,
              "relative attention ABI");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_RELATIVE_ATTENTION_H */
