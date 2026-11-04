/* The sparse-attention decode entry: attention over the SELECTED cells only, phase 4c of the
 * query-aware sparse KV path. Launch-only, like the entries of abi_resident.h; Rust mirrors
 * the struct in native/resident/sparse_decode.rs (child module of native/resident.rs).
 *
 * The kernel computes, for one decode token, exactly what attn_decode_q8.cu computes over a
 * key subset: out[h][d] = sum_j softmax_j(scale * q[h] . K[j][kvh]) * V[j][kvh][d], where j
 * runs over the head's selected indices - the sparse_topk output (ascending top-K picks, then
 * sinks, then the recent window) read through `indices`/`counts`. Sparsity changes WHICH
 * tokens are attended, never HOW Q8 attention computes over them: int8 codes with one fp16
 * scale per 32 values (kQuant of attn_decode_q8.cu), dequantized in registers as
 * f32(scale) * (float)(signed char)code, q pre-scaled once, f32 fmaf dots with the same fixed
 * warp/butterfly reduction order, 64-cell tiles under an online softmax (m/corr/p/l), no
 * float atomics. Single block per KV head (correctness baseline; splits only with measured
 * evidence, and then only by reproducing the combine-k merge). The head's G query heads
 * (G = n_head / n_kv_head, one warp each) land in out as (kvh * G + g), the production GQA
 * mapping. Selected cells are a SET by contract: the selection range is disjoint from sinks
 * and the recent window by construction (sparse_topk), so no token contributes twice;
 * duplicate indices in a malformed buffer are garbage-in-garbage-out and the wrapper pins
 * disjointness instead.
 *
 * Poison rule (as sparse_topk): the device words decide. Each block reads its head's count
 * and validates count <= row and every index < cells (u32 against cells) BEFORE writing
 * anything for THAT head and ORs mask 4 (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of
 * resident/step_params.h) into *status by one thread. Refusal is per head: the validation is
 * data-dependent, so a valid head's block may complete and write its slice before the mask is
 * observed - a set mask always means the whole result is unusable and the host takes the dense
 * fallback. This is range validity of the
 * physical cell; logical-position and prefix-restore guarantees live in the orchestration
 * (sparse_topk owns index identity) and are pinned by the wrapper tests.
 *
 * Inputs: `q` f32 [n_head][head_dim] (one decode token, unscaled; `scale` applies inside),
 * `k_codes`/`v_codes` int8 [n_kv_head][cells][head_dim], `k_scales`/`v_scales` fp16 bits
 * u16 [n_kv_head][cells][head_dim / 32], `indices` u32 [n_kv_head][row], `counts`
 * u32 [n_kv_head] (both the sparse_topk output). Output: `out` f32 [n_head * head_dim],
 * normalized per head (acc / l). A zero count (impossible from sparse_topk, which always
 * emits sink + recent) writes zeros, not NaN.
 *
 * Geometry: n_kv_head 1..64, n_head = n_kv_head * G with G in 1..8, head_dim 128 or 256,
 * cells 1..2^30 - 1, row 1..2^24, count 0..row. One kernel on stream (NULL: default; the
 * Rust wrapper always passes one), grid n_kv_head, 256 threads. Bad arguments return
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SPARSE_DECODE_H
#define INFERENCE_CUDA_RESIDENT_SPARSE_DECODE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_sparse_decode_args {
    const void *q;         /* f32 [n_head][head_dim], one token, unscaled */
    const void *k_codes;   /* int8 [n_kv_head][cells][head_dim] */
    const void *k_scales;  /* u16 fp16 bits [n_kv_head][cells][head_dim / 32] */
    const void *v_codes;   /* int8 [n_kv_head][cells][head_dim] */
    const void *v_scales;  /* u16 fp16 bits [n_kv_head][cells][head_dim / 32] */
    const void *indices;   /* u32 [n_kv_head][row], the sparse_topk output */
    const void *counts;    /* u32 [n_kv_head], the sparse_topk counts */
    void *out;             /* f32 [n_head * head_dim] */
    const void *proj;      /* optional f32 [n_head][2*head_dim]: sigmoid of the head_dim half gates out (NULL = no gate) */
    void *status;          /* one device i32: receives mask 4 on a refused launch */
    int n_kv_head;         /* 1..64 */
    int n_head;            /* n_kv_head * G, G in 1..8 */
    int head_dim;          /* 128 or 256 */
    int cells;             /* KV capacity cells per head, 1..2^30 - 1 */
    int row;               /* indices row capacity = top_k + sink + recent, 1..2^24 */
    float scale;           /* the production attention scale */
} inference_cuda_sparse_decode_args;

#ifdef __cplusplus
static_assert(alignof(inference_cuda_sparse_decode_args) == 8 && sizeof(inference_cuda_sparse_decode_args) == 104 && offsetof(inference_cuda_sparse_decode_args, q) == 0 && offsetof(inference_cuda_sparse_decode_args, k_codes) == 8 && offsetof(inference_cuda_sparse_decode_args, k_scales) == 16 && offsetof(inference_cuda_sparse_decode_args, v_codes) == 24 && offsetof(inference_cuda_sparse_decode_args, v_scales) == 32 && offsetof(inference_cuda_sparse_decode_args, indices) == 40 && offsetof(inference_cuda_sparse_decode_args, counts) == 48 && offsetof(inference_cuda_sparse_decode_args, out) == 56 && offsetof(inference_cuda_sparse_decode_args, proj) == 64 && offsetof(inference_cuda_sparse_decode_args, status) == 72 && offsetof(inference_cuda_sparse_decode_args, n_kv_head) == 80,
              "inference_cuda_sparse_decode_args layout");
#else
_Static_assert(_Alignof(inference_cuda_sparse_decode_args) == 8 && sizeof(inference_cuda_sparse_decode_args) == 104 && offsetof(inference_cuda_sparse_decode_args, q) == 0 && offsetof(inference_cuda_sparse_decode_args, k_codes) == 8 && offsetof(inference_cuda_sparse_decode_args, k_scales) == 16 && offsetof(inference_cuda_sparse_decode_args, v_codes) == 24 && offsetof(inference_cuda_sparse_decode_args, v_scales) == 32 && offsetof(inference_cuda_sparse_decode_args, indices) == 40 && offsetof(inference_cuda_sparse_decode_args, counts) == 48 && offsetof(inference_cuda_sparse_decode_args, out) == 56 && offsetof(inference_cuda_sparse_decode_args, proj) == 64 && offsetof(inference_cuda_sparse_decode_args, status) == 72 && offsetof(inference_cuda_sparse_decode_args, n_kv_head) == 80,
               "inference_cuda_sparse_decode_args layout");
#endif

/* One launch (grid n_kv_head, 256 threads). */
int inference_cuda_sparse_decode_launch(const inference_cuda_sparse_decode_args *args,
                                        void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SPARSE_DECODE_H */
