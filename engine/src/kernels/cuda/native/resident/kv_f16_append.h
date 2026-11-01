/* The standalone fp16 KV append of the multi-device card path. Launch-only, like the entries of
 * abi_resident.h. Include after abi.h in C++; Rust mirrors the structs in
 * native/resident/kv_f16_append.rs (child module of native/resident.rs).
 *
 * The store arm of the vendored attn_prep kernel on its own: m rows (1..16) of gathered,
 * post-rope k and post-norm v, f32 [m][n_kv_head][head_dim] each, converted with the same
 * __float2half and written to the same fp16 plane layout the fused kernel writes,
 * [cells][n_kv_head][head_dim], row *pos + t of both caches, reading the device i32 word pos
 * (StepParams.pos). The card path needs the append separate because its rows arrive through the
 * peer-ring gathers, after the prep kernel already ran.
 *
 * head_dim is 256, the fused kernel's only head size (attn_prep's supported_configuration); a
 * different head size extends both or neither.
 *
 * Poison rule (as attn_prep and kv8_append): the kernel tests the word against the cells before
 * any other memory access; if *pos < 0 or *pos + m > cells (64-bit sum) no cache byte is written
 * and mask 4 (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h) is OR-ed into
 * *status by one thread.
 *
 * All other geometry is run-time: n_kv_head 1..64, cells m..2^30 - 1. The entry enqueues one
 * kernel on stream (NULL: the default stream; the Rust wrapper always passes one) and returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_KV_F16_APPEND_H
#define INFERENCE_CUDA_RESIDENT_KV_F16_APPEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decode append: m rows at the device position word. */
typedef struct inference_cuda_kv_f16_append_args {
    const void *k;    /* f32 [m][n_kv_head][head_dim] */
    const void *v;    /* f32 [m][n_kv_head][head_dim] */
    void *k_cache;    /* fp16 [cells][n_kv_head][head_dim], 16-byte aligned */
    void *v_cache;    /* fp16 [cells][n_kv_head][head_dim], 16-byte aligned */
    const void *pos;  /* one device i32: the position of token 0 */
    void *status;     /* one device i32: receives mask 4 on a refused launch */
    int m;            /* tokens, 1..16 */
    int n_kv_head;    /* 1..64 */
    int head_dim;     /* required: 256 */
    int cells;        /* plane rows (the KV capacity), m..2^30 - 1 */
} inference_cuda_kv_f16_append_args;

int inference_cuda_kv_f16_append_launch(const inference_cuda_kv_f16_append_args *args,
                                        void *stream, int *out_stage);

#ifdef __cplusplus
static_assert(alignof(inference_cuda_kv_f16_append_args) == 8 &&
                  sizeof(inference_cuda_kv_f16_append_args) == 64 &&
                  offsetof(inference_cuda_kv_f16_append_args, k_cache) == 16 &&
                  offsetof(inference_cuda_kv_f16_append_args, pos) == 32 &&
                  offsetof(inference_cuda_kv_f16_append_args, status) == 40 &&
                  offsetof(inference_cuda_kv_f16_append_args, m) == 48 &&
                  offsetof(inference_cuda_kv_f16_append_args, cells) == 60,
              "inference_cuda_kv_f16_append_args layout");
#else
_Static_assert(_Alignof(inference_cuda_kv_f16_append_args) == 8 &&
                   sizeof(inference_cuda_kv_f16_append_args) == 64 &&
                   offsetof(inference_cuda_kv_f16_append_args, k_cache) == 16 &&
                   offsetof(inference_cuda_kv_f16_append_args, pos) == 32 &&
                   offsetof(inference_cuda_kv_f16_append_args, status) == 40 &&
                   offsetof(inference_cuda_kv_f16_append_args, m) == 48 &&
                   offsetof(inference_cuda_kv_f16_append_args, cells) == 60,
               "inference_cuda_kv_f16_append_args layout");
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_KV_F16_APPEND_H */
