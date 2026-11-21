#ifndef INFERENCE_CUDA_RESIDENT_SHORTLIST_H
#define INFERENCE_CUDA_RESIDENT_SHORTLIST_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum {
    INFERENCE_CUDA_SHORTLIST_K = 20,
    INFERENCE_CUDA_SHORTLIST_TILES = 16,
    INFERENCE_CUDA_SHORTLIST_PART_BYTES = 176,
    INFERENCE_CUDA_SHORTLIST_RESULT_OFFSET = 2816,
    INFERENCE_CUDA_SHORTLIST_SCRATCH_BYTES = 2992,
    INFERENCE_CUDA_SHORTLIST_MAX_ROWS = 16,
    INFERENCE_CUDA_SHORTLIST_MASS_OFFSET = 2992,
    INFERENCE_CUDA_SHORTLIST_MASS_PARTS = 8,
    INFERENCE_CUDA_SHORTLIST_MASS_SCRATCH_BYTES = 3056
};
typedef struct inference_cuda_shortlist_candidate {
    float logit;
    uint32_t id;
} inference_cuda_shortlist_candidate;
typedef struct inference_cuda_shortlist_result {
    inference_cuda_shortlist_candidate entries[20];
    uint32_t non_finite;
    uint32_t pad[3];
} inference_cuda_shortlist_result;
typedef struct inference_cuda_shortlist_args {
    const void *logits;
    void *scratch;
    int n;
} inference_cuda_shortlist_args;
/* Launch-only stable top-20: larger logit first, then lower id; original f32 bits retained.
 * logits: 4-byte-aligned f32[n], n in1..2^30. scratch:16-byte-aligned2992bytes,disjoint.
 * Result is176bytes atscratch+2816;firstmin(n,20) entries are valid whennon_finite==0.
 * Any non-finite input setsnon_finite=1;unusedentries haveidUINT32_MAX/logit-INFINITY.
 * No allocation,copy,synchronizationorstep-statuswrite. Explicitnon-nullstreamrequired. */
int inference_cuda_shortlist_launch(const inference_cuda_shortlist_args *args,
                                    void *stream, int *out_stage);
typedef struct inference_cuda_shortlist_mass_args {
    const void *logits;
    void *scratch;
    int n;
    float inv_temperature;
} inference_cuda_shortlist_mass_args;
/* The softmax mass of the row a single-row shortlist launch just ranked, on the same scratch: the
 * sum over all n values of expf((logit - top) * inv_temperature), where top is the first entry of
 * the shortlist record at scratch + 2816. Eight blocks each write one f64 partial of their strided
 * share at scratch + 2992 (8 * 8 bytes); the caller adds the eight in order. Each partial adds f32
 * terms per thread and f64 across threads, so the total carries a relative error far below 1e-4
 * (f32 expf and f32 per-thread sums of at most n / 2048 terms); a caller that needs the host's f64
 * total must treat it as an estimate with that bound. Read only on the logits and the record; no
 * allocation, copy, synchronization or step-status write; explicit non-null stream. scratch is the
 * 3056 bytes of the shortlist scratch plus the partials, 16-byte aligned, disjoint from logits.
 * inv_temperature is finite and positive. */
int inference_cuda_shortlist_mass_launch(const inference_cuda_shortlist_mass_args *args,
                                         void *stream, int *out_stage);
typedef struct inference_cuda_shortlist_rows_args {
    const void *logits;
    void *scratch;
    void *out;
    long long ld;
    long long scratch_stride;
    int n;
    int m;
} inference_cuda_shortlist_rows_args;
/* The K-row form of the shortlist above: row r (0 <= r < m) of the m logits rows, each of n f32 values
 * and row r starting r * ld floats after the first, gets the record the single-row launch gives it
 * (same bytes: the selection is comparisons only). Partials of row r are written at
 * scratch + r * scratch_stride (16 * 176 = 2816 bytes used, stride a multiple of 16 and at least
 * 2816); the m final records of 176 bytes go to out + r * 176, contiguous, 16-byte aligned. Every
 * launch rewrites every record whole (entries, non_finite, three zero pad words). m in 1..16, n in
 * 1..2^30, ld >= n. logits, scratch and out overlap nowhere. No allocation, copy,
 * synchronization or step-status write; explicit non-null stream. Two kernels (16 x m blocks, then m). */
int inference_cuda_shortlist_rows_launch(const inference_cuda_shortlist_rows_args *args,
                                         void *stream, int *out_stage);
#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_shortlist_candidate)==8,"candidate layout");
static_assert(sizeof(inference_cuda_shortlist_result)==176 &&
              offsetof(inference_cuda_shortlist_result,non_finite)==160,"result layout");
static_assert(sizeof(inference_cuda_shortlist_mass_args)==24 &&
              offsetof(inference_cuda_shortlist_mass_args,n)==16 &&
              offsetof(inference_cuda_shortlist_mass_args,inv_temperature)==20,"mass argument layout");
static_assert(sizeof(inference_cuda_shortlist_args)==24 &&
              offsetof(inference_cuda_shortlist_args,n)==16,"argument layout");
static_assert(sizeof(inference_cuda_shortlist_rows_args)==48 &&
              offsetof(inference_cuda_shortlist_rows_args,logits)==0 &&
              offsetof(inference_cuda_shortlist_rows_args,scratch)==8 &&
              offsetof(inference_cuda_shortlist_rows_args,out)==16 &&
              offsetof(inference_cuda_shortlist_rows_args,ld)==24 &&
              offsetof(inference_cuda_shortlist_rows_args,scratch_stride)==32 &&
              offsetof(inference_cuda_shortlist_rows_args,n)==40 &&
              offsetof(inference_cuda_shortlist_rows_args,m)==44,"rows argument layout");
#endif
#endif
