/* In-place BF16 base/correction merge and softcap lookup on an explicit stream. */
#ifndef INFERENCE_CUDA_RESIDENT_CORRECT_DRAFT_LOGITS_BF16_H
#define INFERENCE_CUDA_RESIDENT_CORRECT_DRAFT_LOGITS_BF16_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_correct_draft_logits_bf16_args {
    float *base;
    const float *correction;
    const uint16_t *softcap_lut;
    size_t len;
} inference_cuda_correct_draft_logits_bf16_args;

/* Both logits rows hold len floats. The LUT holds 65,536 little-endian BF16
 * values. The written base range is disjoint from correction and the LUT. */
int inference_cuda_correct_draft_logits_bf16_launch(
    const inference_cuda_correct_draft_logits_bf16_args *args, void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_correct_draft_logits_bf16_args) == 32 &&
                  offsetof(inference_cuda_correct_draft_logits_bf16_args, len) == 24,
              "corrected draft logits BF16 ABI");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_CORRECT_DRAFT_LOGITS_BF16_H */
