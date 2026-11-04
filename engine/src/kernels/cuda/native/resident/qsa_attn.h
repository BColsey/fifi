/* The sparse decode attention of Qwen3.8 Flash-Next QSA, per-kernel header
 * (native/resident/qsa_attn.h). Included by qsa_attn.cu and read by the Rust FFI mirror; nothing
 * else.
 *
 * One token at position p attends to the positions its selection names: with
 * blocks = (p + 1) / ratio and take = min(blocks, budget_blocks), the ratio positions of each of
 * chosen[0 .. take) (qsa_select's output), then the open tail blocks * ratio .. p. Head h
 * (n_head query heads, group = n_head / n_kv_head per kv head) reads kv head h / group:
 *
 *   s_j    = scale * (q[h] . k_cache[j][h / group])
 *   out[h] = sigmoid(gate_h) * sum_j softmax(s)_j * v_cache[j][h / group]
 *
 * q is f32 [n_head][256] (attn_prep's normed and rotated q), proj the f32 projection row whose
 * head h gate is at column h * 512 + 256 + d (attn_prep's layout), the caches fp16
 * [rows][n_kv_head][256] with position j at row j, out f32 [n_head][256]. The positions are split
 * over max_splits blocks per kv head; part is f32 scratch of n_head * max_splits * 258 floats
 * (per split a max, a sum and 256 partial values), no initialization needed.
 *
 * The position is read from the device word `pos` when it is given (a decode graph replays it),
 * else it is the launch argument pos0. Poison rule (attn_prep's): if it is negative or
 * p + 1 > rows, or a chosen block is not below `blocks`, out is NaN and status bits 4 are OR-ed
 * into *status by one thread when status is given. No cache row outside [0, p] is read.
 *
 * Limits: head size 256; n_kv_head in [1, 64] dividing n_head, group in [1, 32]; rows in
 * [1, 2^30 - 1]; ratio in [1, 64]; budget_blocks in [1, 2^24]; max_splits in [1, 128];
 * ld >= n_head * 512; scale finite. q and the caches 16-byte aligned, the rest 4-byte aligned;
 * the written ranges (part, out, status) overlap no other range. The
 * entry enqueues two kernels on stream (not NULL) and returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_QSA_ATTN_H
#define INFERENCE_CUDA_RESIDENT_QSA_ATTN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_qsa_attn_args {
    const void *q;       /* f32 [n_head][256] */
    const void *proj;    /* f32 [ld], the gates */
    const void *k_cache; /* fp16 [rows][n_kv_head][256] */
    const void *v_cache; /* fp16 [rows][n_kv_head][256] */
    const void *chosen;  /* i32 [budget_blocks] */
    const void *pos;     /* device i32 position, or NULL for pos0 */
    void *part;          /* f32 [n_head][max_splits][258] */
    void *out;           /* f32 [n_head][256] */
    void *status;        /* device i32, or NULL */
    int ld;
    int n_head;
    int n_kv_head;
    int rows;
    int ratio;
    int budget_blocks;
    int max_splits;
    int pos0;
    float scale;
} inference_cuda_qsa_attn_args;

int inference_cuda_qsa_attn_launch(const inference_cuda_qsa_attn_args *args, void *stream,
                                   int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cstddef>
static_assert(sizeof(inference_cuda_qsa_attn_args) == 112 &&
                  offsetof(inference_cuda_qsa_attn_args, status) == 64 &&
                  offsetof(inference_cuda_qsa_attn_args, ld) == 72 &&
                  offsetof(inference_cuda_qsa_attn_args, pos0) == 100 &&
                  offsetof(inference_cuda_qsa_attn_args, scale) == 104,
              "inference_cuda_qsa_attn_args layout");
#endif

#endif
