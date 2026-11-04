/* The sparse-attention deterministic top-K entry. Launch-only, like the entries of
 * abi_resident.h. Include after abi.h in C++; Rust mirrors the structs in
 * native/resident/sparse_topk.rs (child module of native/resident.rs).
 *
 * The entry turns the relevance scores of sparse_score into the selected index list of the
 * sparse KV path, per KV head: the `top_k` largest candidate scores in `[sink, kv_len -
 * recent)` ranked by value descending with ties broken by ascending index (the radix-select
 * rule of family/qwen36/sparse_reference.rs::deterministic_top_k), followed by the sink cells
 * `[0, sink)` and the recent window `[kv_len - recent, kv_len)`, each ascending. The head's
 * written count lands in `counts[head]`: `min(top_k, candidates) + sink + recent` indices.
 *
 * Scores are the softmax masses of sparse_score: finite, non-negative f32; they are ranked by
 * their bit patterns as unsigned integers (exact for [0, +inf)). The threshold comes from a
 * four-level 8-bit radix select over a shared histogram; the fill keeps ascending index order
 * through tiled exclusive prefix scans, so the output is bit-deterministic per launch geometry.
 *
 * The kv length is read from the device: `kv_len = *pos + 1`. Poison rule (as attn_prep and
 * kv8_append): if `*pos < 0` or `*pos + 1 > cells` (64-bit sum) nothing is written and mask 4
 * (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h) is OR-ed into *status by
 * one thread. A kv length that leaves no candidate (`kv_len <= sink + recent`) is refused the
 * same way - the host must take the dense fallback instead of launching.
 *
 * Inputs: `scores` f32 [n_kv_head][cells] (the sparse_score output; only `[sink, kv_len -
 * recent)` is read). Outputs: `indices` u32 [n_kv_head][top_k + sink + recent],
 * `counts` u32 [n_kv_head].
 *
 * Geometry is run-time: n_kv_head 1..64, top_k 1..2^24, sink + recent < cells, cells 1..2^30 -
 * 1. One thread block of 512 per KV head. The entry enqueues one kernel on stream (NULL: the
 * default stream; the Rust wrapper always passes one) and returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SPARSE_TOPK_H
#define INFERENCE_CUDA_RESIDENT_SPARSE_TOPK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_sparse_topk_args {
    const void *scores;   /* f32 [n_kv_head][cells], candidates only are read */
    void *indices;        /* u32 [n_kv_head][top_k + sink + recent] */
    void *counts;         /* u32 [n_kv_head] */
    const void *pos;      /* one device i32: the position of the token being appended */
    void *status;         /* one device i32: receives mask 4 on a refused launch */
    int n_kv_head;        /* 1..64 */
    int cells;            /* plane cells per head (the KV capacity), 1..2^30 - 1 */
    int top_k;            /* 1..2^24 */
    int sink;             /* leading forced cells, 0..cells-1 */
    int recent;           /* trailing forced cells, 0..cells-1 */
} inference_cuda_sparse_topk_args;

#ifdef __cplusplus
static_assert(alignof(inference_cuda_sparse_topk_args) == 8 && sizeof(inference_cuda_sparse_topk_args) == 64 && offsetof(inference_cuda_sparse_topk_args, scores) == 0 && offsetof(inference_cuda_sparse_topk_args, indices) == 8 && offsetof(inference_cuda_sparse_topk_args, counts) == 16 && offsetof(inference_cuda_sparse_topk_args, pos) == 24 && offsetof(inference_cuda_sparse_topk_args, status) == 32 && offsetof(inference_cuda_sparse_topk_args, n_kv_head) == 40,
              "inference_cuda_sparse_topk_args layout");
#else
_Static_assert(_Alignof(inference_cuda_sparse_topk_args) == 8 && sizeof(inference_cuda_sparse_topk_args) == 64 && offsetof(inference_cuda_sparse_topk_args, scores) == 0 && offsetof(inference_cuda_sparse_topk_args, indices) == 8 && offsetof(inference_cuda_sparse_topk_args, counts) == 16 && offsetof(inference_cuda_sparse_topk_args, pos) == 24 && offsetof(inference_cuda_sparse_topk_args, status) == 32 && offsetof(inference_cuda_sparse_topk_args, n_kv_head) == 40,
               "inference_cuda_sparse_topk_args layout");
#endif

/* One launch (grid n_kv_head, 512 threads). */
int inference_cuda_sparse_topk_launch(const inference_cuda_sparse_topk_args *args,
                                      void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SPARSE_TOPK_H */
