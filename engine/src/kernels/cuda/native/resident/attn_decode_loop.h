/*
 * Split-KV decode attention for the SenseNova Loop family (native/resident/attn_decode_loop.h).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * struct in native/resident/attn_decode_loop.rs (child module of native/resident.rs).
 *
 * The structure is attn_decode.h's with the Loop-9B geometry fixed: query heads grouped 4 per kv
 * head (n_head == 4 * n_kv_head), head_dim exactly 256, a linear cache with every position
 * visible (no window, no ring), no sinks. The output gate is optional (proj NULL: no gate), as in
 * attn_decode.h.
 */
#ifndef INFERENCE_CUDA_ATTN_DECODE_LOOP_H
#define INFERENCE_CUDA_ATTN_DECODE_LOOP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decode attention for a batch of m tokens (m <= 16) over an fp16 KV cache, query heads
 * grouped 4 per kv head (n_head == 4 * n_kv_head), head_dim 256. q is f32 [m][n_head][256]
 * (attn_prep's q); k_cache and v_cache are fp16 [cache_rows][n_kv_head][256], 16-byte aligned.
 * proj is NULL (no output gate) or the f32 projection row [m][ld] (ld >= n_head * 512) whose
 * head h gate is at column h * 512 + 256. kv_len is a device i32 word: token t has kv_len + t
 * keys (positions 0 .. kv_len + t - 1), so attn_prep must have appended up to position
 * kv_len + m - 2 first. A kv_len outside [1, cache_rows - (m - 1)] makes the output NaN.
 * part is f32 scratch of at least inference_cuda_attn_decode_loop_part_floats(m, n_kv_head,
 * max_splits) floats, 16-byte aligned, no initialization needed. out is f32
 * [m][n_head * 256] = attention output (times sigmoid(gate) when proj is given).
 * max_splits (1..128) is the grid size; min_keys (a multiple of 32) is the fewest keys worth one
 * split; scale multiplies the scores. Enqueues two kernels on stream (NULL: the default stream);
 * returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
typedef struct inference_cuda_attn_decode_loop_args {
    const void *q;
    const void *proj;
    const void *k_cache;
    const void *v_cache;
    const void *kv_len;
    void *part;
    void *out;
    int ld;
    int m;
    int n_head;
    int n_kv_head;
    int cache_rows;
    int max_splits;
    int min_keys;
    float scale;
} inference_cuda_attn_decode_loop_args;
size_t inference_cuda_attn_decode_loop_part_floats(int m, int n_kv_head, int max_splits);
int inference_cuda_attn_decode_loop_launch(const inference_cuda_attn_decode_loop_args *args,
                                           void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
