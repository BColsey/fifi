/* The block indexer of Qwen3.8 Flash-Next QSA attention, per-kernel header
 * (native/resident/qsa_index.h). Included by qsa_index.cu and read by the Rust FFI mirror;
 * nothing else.
 *
 * QSA attends, per decode token, to the `budget / ratio` best blocks of `ratio` consecutive
 * positions plus the open tail block. The indexer ranks the completed blocks. Its projection row
 * per token is f32 [heads][dim] queries followed by one [dim] raw key, at column col_off of a
 * row of ld floats. With rms0(x, w) = x * (1 + w) / sqrt(mean(x^2) + eps) and rope the
 * rotate-half rotation of the first rotary_dim values (pairs (i, i + rotary_dim / 2), angle
 * pos / theta^((2 i) / rotary_dim) in f32, as family/flash_next/math.rs):
 *
 * index, for the m tokens at positions p0 .. p0 + m - 1:
 *   queries[t][h] = rope(rms0(row_t.query[h], q_norm), p0 + t)
 *   for every block b completed by the chunk ((b + 1) * ratio - 1 in [p0, p0 + m - 1]):
 *     pooled[i] = sum over r in 0 .. ratio - 1, in that order, of raw(b * ratio + r)[i] / ratio
 *     keys[b]   = rope(rms0(pooled, k_norm), b * ratio)
 *   where raw(p) is the key of row p - p0 for p >= p0 and ring[p % ratio] otherwise; then
 *   ring[p % ratio] = raw(p) for the last min(m, ratio) positions of the chunk, so the ring holds
 *   the raw keys of the open block for the next call. The pooling order is the host's, so a
 *   pooled value equals the host's bits for the same raw keys.
 * score, for the token at position p (one token):
 *   scores[b] = sum over h, in head order, of max(queries[h] . keys[b], 0), a NaN dot taken as 0
 *   for b < (p + 1) / ratio. The host's 1 / sqrt(dim) is left out: a positive scale does not
 *   change the ranking, and the selection is the only consumer.
 *
 * The position of token 0 is read from the device word `pos` when it is given (a decode graph
 * replays it), else it is the launch argument pos0. Poison rule (attn_prep's): if that position
 * is negative or p0 + m > cap_blocks * ratio (64-bit sum), nothing is written but NaN queries
 * (index) and status bits 4 are OR-ed into *status by one thread when status is given; score
 * writes nothing and raises the same bits.
 *
 * Limits: m in [1, 32768]; heads in [1, 64]; dim a multiple of 32 in [32, 1024] with heads * dim
 * <= 8192; rotary_dim even in [2, dim]; ratio in [1, 64]; cap_blocks in [1, 2^24]; ld >= col_off +
 * (heads + 1) * dim; theta finite and > 0; eps finite and >= 0. Every range is f32 and 4-byte
 * aligned; the written ranges (queries, ring, keys, scores, status) overlap no other range. Each
 * entry enqueues its kernels on stream (not NULL) and returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_QSA_INDEX_H
#define INFERENCE_CUDA_RESIDENT_QSA_INDEX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_qsa_index_args {
    const void *proj;   /* f32 [m][ld] */
    const void *q_norm; /* f32 [dim] */
    const void *k_norm; /* f32 [dim] */
    void *queries;      /* f32 [m][heads][dim] */
    void *ring;         /* f32 [ratio][dim], in place */
    void *keys;         /* f32 [cap_blocks][dim], completed blocks written */
    const void *pos;    /* device i32 position of token 0, or NULL for pos0 */
    void *status;       /* device i32, or NULL */
    int m;
    int ld;
    int col_off;
    int heads;
    int dim;
    int rotary_dim;
    int ratio;
    int cap_blocks;
    int pos0;
    float theta;
    float eps;
} inference_cuda_qsa_index_args;

typedef struct inference_cuda_qsa_score_args {
    const void *queries; /* f32 [heads][dim], the token at the position */
    const void *keys;    /* f32 [cap_blocks][dim] */
    void *scores;        /* f32 [cap_blocks], [0, (pos + 1) / ratio) written */
    const void *pos;     /* device i32 position, or NULL for pos0 */
    void *status;        /* device i32, or NULL */
    int heads;
    int dim;
    int ratio;
    int cap_blocks;
    int pos0;
} inference_cuda_qsa_score_args;

int inference_cuda_qsa_index_launch(const inference_cuda_qsa_index_args *args, void *stream,
                                    int *out_stage);

int inference_cuda_qsa_score_launch(const inference_cuda_qsa_score_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cstddef>
static_assert(sizeof(inference_cuda_qsa_index_args) == 112 &&
                  offsetof(inference_cuda_qsa_index_args, status) == 56 &&
                  offsetof(inference_cuda_qsa_index_args, m) == 64 &&
                  offsetof(inference_cuda_qsa_index_args, pos0) == 96 &&
                  offsetof(inference_cuda_qsa_index_args, eps) == 104,
              "inference_cuda_qsa_index_args layout");
static_assert(sizeof(inference_cuda_qsa_score_args) == 64 &&
                  offsetof(inference_cuda_qsa_score_args, status) == 32 &&
                  offsetof(inference_cuda_qsa_score_args, heads) == 40 &&
                  offsetof(inference_cuda_qsa_score_args, pos0) == 56,
              "inference_cuda_qsa_score_args layout");
#endif

#endif
