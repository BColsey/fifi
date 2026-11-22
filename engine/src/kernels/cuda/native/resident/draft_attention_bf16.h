/* Full-block BF16 GQA over an immutable committed ring and a separate draft block. */
#ifndef INFERENCE_CUDA_RESIDENT_DRAFT_ATTENTION_BF16_H
#define INFERENCE_CUDA_RESIDENT_DRAFT_ATTENTION_BF16_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_draft_attention_bf16_args {
    uint16_t *out;
    const uint16_t *query;
    const uint16_t *draft_keys;
    const uint16_t *draft_values;
    const uint16_t *committed_keys;
    const uint16_t *committed_values;
    int query_heads;
    int kv_heads;
    int rows;
    int capacity;
    int first;
    int len;
} inference_cuda_draft_attention_bf16_args;

/* All ranges use [row][head][128] layout. Only out is written. */
int inference_cuda_draft_attention_bf16_launch(
    const inference_cuda_draft_attention_bf16_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_draft_attention_bf16_args) == 72 &&
                  offsetof(inference_cuda_draft_attention_bf16_args, query_heads) == 48 &&
                  offsetof(inference_cuda_draft_attention_bf16_args, len) == 68,
              "draft attention BF16 ABI");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_DRAFT_ATTENTION_BF16_H */
