/*
 * Launch entry of sample: one device-side draw from an on-device logits row.
 * The row stays on the device; only the chosen token id and its log probability
 * (plus a status word) are written. Launch-only.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SAMPLE_H
#define INFERENCE_CUDA_RESIDENT_SAMPLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_sample_args {
    void *token;       /* u32 out, chosen id */
    void *logprob;     /* f32 out, log probability of the chosen id */
    void *status;      /* u32 out, OR-ed: 2 empty distribution, 4 bad shape */
    void *probs;       /* f32 scratch [n], written then read */
    const void *logits;/* f32 [n], read */
    int n;             /* vocabulary: the sampler width, 1..4194304 */
    float temperature; /* finite, > 0 samples; <= 0 takes the argmax */
    int top_k;         /* 0 disables, else 1..n */
    float top_p;       /* (0, 1], 1 disables */
    float min_p;       /* [0, 1), 0 disables */
    unsigned int seed_lo;
    unsigned int seed_hi;
    unsigned int pos_lo;
    unsigned int pos_hi;
} inference_cuda_sample_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_sample_args) == 80 && offsetof(inference_cuda_sample_args, token) == 0 && offsetof(inference_cuda_sample_args, logprob) == 8 && offsetof(inference_cuda_sample_args, status) == 16 && offsetof(inference_cuda_sample_args, probs) == 24 && offsetof(inference_cuda_sample_args, logits) == 32 && offsetof(inference_cuda_sample_args, n) == 40 && offsetof(inference_cuda_sample_args, temperature) == 44 && offsetof(inference_cuda_sample_args, top_k) == 48 && offsetof(inference_cuda_sample_args, top_p) == 52 && offsetof(inference_cuda_sample_args, min_p) == 56 && offsetof(inference_cuda_sample_args, seed_lo) == 60 && offsetof(inference_cuda_sample_args, seed_hi) == 64 && offsetof(inference_cuda_sample_args, pos_lo) == 68 && offsetof(inference_cuda_sample_args, pos_hi) == 72, "layout");
#else
_Static_assert(sizeof(inference_cuda_sample_args) == 80 && offsetof(inference_cuda_sample_args, token) == 0 && offsetof(inference_cuda_sample_args, logprob) == 8 && offsetof(inference_cuda_sample_args, status) == 16 && offsetof(inference_cuda_sample_args, probs) == 24 && offsetof(inference_cuda_sample_args, logits) == 32 && offsetof(inference_cuda_sample_args, n) == 40 && offsetof(inference_cuda_sample_args, temperature) == 44 && offsetof(inference_cuda_sample_args, top_k) == 48 && offsetof(inference_cuda_sample_args, top_p) == 52 && offsetof(inference_cuda_sample_args, min_p) == 56 && offsetof(inference_cuda_sample_args, seed_lo) == 60 && offsetof(inference_cuda_sample_args, seed_hi) == 64 && offsetof(inference_cuda_sample_args, pos_lo) == 68 && offsetof(inference_cuda_sample_args, pos_hi) == 72, "layout");
#endif

/* One launch on stream. Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call
 * for a bad argument, else the launch status. */
int inference_cuda_sample_launch(const inference_cuda_sample_args *args, void *stream,
                                       int *out_stage);


/* One draw from a ranked shortlist already on the device (the record of the shortlist launch):
 * `entries` holds `count` (logit f32, id u32) pairs, larger logit first, unused ones with id
 * 0xFFFFFFFF; `flag` is the record's non_finite word. Writes the chosen id, its log probability
 * (over the kept set) and ORs the status word (2 empty set, 4 non-finite input); the status word
 * is not cleared, so a caller that owns bits there keeps them. temperature <= 0 takes entry 0.
 * top_k 0 keeps all `count` entries, else the first top_k; top_p (0, 1] keeps the smallest
 * prefix whose mass reaches top_p of the kept mass. The uniform is Philox-4x32-10 of (seed, pos),
 * as in the full-row sampler. count 1..64. Launch-only. */
typedef struct inference_cuda_sample_shortlist_args {
    void *token;          /* u32 out */
    void *logprob;        /* f32 out */
    void *status;         /* u32 in/out, OR-ed */
    const void *entries;  /* count * 8 bytes */
    const void *flag;     /* u32 */
    int count;
    float temperature;
    int top_k;
    float top_p;
    unsigned int seed_lo;
    unsigned int seed_hi;
    unsigned int pos_lo;
    unsigned int pos_hi;
} inference_cuda_sample_shortlist_args;

int inference_cuda_sample_shortlist_launch(const inference_cuda_sample_shortlist_args *args,
                                           void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SAMPLE_H */
