/* The sparse-attention relevance-score entry. Launch-only, like the entries of abi_resident.h.
 * Include after abi.h in C++; Rust mirrors the structs in native/resident/sparse_score.rs
 * (child module of native/resident.rs).
 *
 * The entry scores every candidate position of every KV head for the query-aware sparse KV
 * path: `scores[h][t] = sum over the query lanes of h of softmax_lane(logit)[t]`, the softmax
 * over the candidate range `[sink, kv_len - recent)`, `logit = scale * (B q_lane) . z(t)` with
 * `z(t)` the stored F16 sketch cell of `sketch[h][t]` and `B q_lane` the basis projection of
 * the lane's query row. Non-candidate cells are never written. The score matches the host
 * reference `family/qwen36/sparse_reference.rs::relevance_scores` up to f32 rounding.
 *
 * The kv length is read from the device: `kv_len = *pos + 1` (the decode entry scores the step
 * that appends at `*pos`). Poison rule (as attn_prep and kv8_append): if `*pos < 0` or
 * `*pos + 1 > cells` (64-bit sum) nothing is written and mask 4
 * (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h) is OR-ed into *status by
 * one thread. A kv length that leaves no candidates (`kv_len <= sink + recent`) writes nothing
 * and raises no mask; the host takes the dense fallback.
 *
 * Inputs: `q` f32 [n_q_head][head_dim] (post-norm, post-rope rows), `basis` f32
 * [n_kv_head][rank][head_dim] (output-major, the same basis the sketch plane holds),
 * `sketch` fp16 [n_kv_head][cells][rank]. Output: `scores` f32 [n_kv_head][cells].
 *
 * Geometry is run-time: n_q_head a multiple of n_kv_head with group = n_q_head / n_kv_head in
 * 1..16, group * rank <= 4096 (the shared projection budget), head_dim a multiple of 32 in
 * 32..1024, rank 2..head_dim, cells 1..2^30 - 1, sink + recent < cells. One thread block of
 * 256 per KV head; two deterministic passes over the candidate sketch (an online
 * max/sum pass, then the scoring pass), each cell's inner products accumulated in a fixed
 * serial order, so the stored scores are bit-deterministic per launch geometry. The entry
 * enqueues one kernel on stream (NULL: the default stream; the Rust wrapper always passes one)
 * and returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SPARSE_SCORE_H
#define INFERENCE_CUDA_RESIDENT_SPARSE_SCORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decode scoring entry: kv_len = *pos + 1. */
typedef struct inference_cuda_sparse_score_args {
    const void *q;        /* f32 [n_q_head][head_dim] */
    const void *basis;    /* f32 [n_kv_head][rank][head_dim] */
    const void *sketch;   /* fp16 [n_kv_head][cells][rank] */
    void *scores;         /* f32 [n_kv_head][cells], candidates only */
    const void *pos;      /* one device i32: the position of the token being appended */
    void *status;         /* one device i32: receives mask 4 on a refused launch */
    int n_q_head;         /* a multiple of n_kv_head, group 1..16 */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* a multiple of 32, 32..1024 */
    int rank;             /* 2..head_dim */
    int cells;            /* plane cells per head (the KV capacity), 1..2^30 - 1 */
    int sink;             /* leading non-candidates, 0..cells-1 */
    int recent;           /* trailing non-candidates, 0..cells-1 */
    float scale;          /* the attention logit scale (1/sqrt(head_dim)) */
} inference_cuda_sparse_score_args;

#ifdef __cplusplus
static_assert(alignof(inference_cuda_sparse_score_args) == 8 && sizeof(inference_cuda_sparse_score_args) == 80 && offsetof(inference_cuda_sparse_score_args, q) == 0 && offsetof(inference_cuda_sparse_score_args, basis) == 8 && offsetof(inference_cuda_sparse_score_args, sketch) == 16 && offsetof(inference_cuda_sparse_score_args, scores) == 24 && offsetof(inference_cuda_sparse_score_args, pos) == 32 && offsetof(inference_cuda_sparse_score_args, status) == 40 && offsetof(inference_cuda_sparse_score_args, n_q_head) == 48 && offsetof(inference_cuda_sparse_score_args, sink) == 68,
              "inference_cuda_sparse_score_args layout");
#else
_Static_assert(_Alignof(inference_cuda_sparse_score_args) == 8 && sizeof(inference_cuda_sparse_score_args) == 80 && offsetof(inference_cuda_sparse_score_args, q) == 0 && offsetof(inference_cuda_sparse_score_args, basis) == 8 && offsetof(inference_cuda_sparse_score_args, sketch) == 16 && offsetof(inference_cuda_sparse_score_args, scores) == 24 && offsetof(inference_cuda_sparse_score_args, pos) == 32 && offsetof(inference_cuda_sparse_score_args, status) == 40 && offsetof(inference_cuda_sparse_score_args, n_q_head) == 48 && offsetof(inference_cuda_sparse_score_args, sink) == 68,
               "inference_cuda_sparse_score_args layout");
#endif

/* One launch (grid n_kv_head, 256 threads). */
int inference_cuda_sparse_score_launch(const inference_cuda_sparse_score_args *args,
                                       void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SPARSE_SCORE_H */
