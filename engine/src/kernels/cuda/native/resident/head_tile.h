/* Exact grouped-to-tiled value-head permutation for one f32 row. No allocation or sync.
 * src and dst hold key_heads * reps * head_dim values, must be 4-byte aligned and disjoint.
 * For grouped head h, dst[((h % reps) * key_heads + h / reps) * head_dim + j]
 * receives src[h * head_dim + j]. 1 <= total values <= 2^24, all dimensions positive.
 */
#ifndef INFERENCE_CUDA_RESIDENT_HEAD_TILE_H
#define INFERENCE_CUDA_RESIDENT_HEAD_TILE_H
#include <stddef.h>
extern "C" {
typedef struct inference_cuda_head_tile_args {
    void *dst;
    const void *src;
    int key_heads;
    int value_heads;
    int head_dim;
} inference_cuda_head_tile_args;
int inference_cuda_head_tile_launch(const inference_cuda_head_tile_args *args, void *stream,
                                    int *out_stage);
static_assert(sizeof(inference_cuda_head_tile_args) == 32 &&
              offsetof(inference_cuda_head_tile_args, dst) == 0 &&
              offsetof(inference_cuda_head_tile_args, src) == 8 &&
              offsetof(inference_cuda_head_tile_args, key_heads) == 16 &&
              offsetof(inference_cuda_head_tile_args, value_heads) == 20 &&
              offsetof(inference_cuda_head_tile_args, head_dim) == 24, "layout");
}
#endif
