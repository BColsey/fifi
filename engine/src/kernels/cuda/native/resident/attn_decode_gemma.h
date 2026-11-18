/*
 * Launch entry of attn_decode_gemma for the Gemma 4 Humanizer resident step: one
 * single-pass decode attention over the KV8 planes (int8 codes, fp16 scale bits per
 * 32-value group). Scale is a parameter (Gemma uses 1.0); GQA maps query head h to
 * kv head h*n_kv_head/n_q_head; head_dim is 256 (local, window 1024) or 512 (global,
 * window 0); the device kv_len word is authoritative: token t sees keys 0..kv_len+t-1
 * cut to the last window positions; out is NaN when kv_len is out of range. No output
 * gate. split = 0 runs the v0 single-pass kernel (no split-K); split != 0 runs the
 * part+combine pair over `part` (see attn_split_reduce.cuh), sized by
 * inference_cuda_attn_decode_gemma_part_floats. Launch-only; faults surface at synchronize.
 */
#ifndef INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H
#define INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_attn_decode_gemma_args {
    void *out;               /* f32 [m][n_q_head][head_dim], written */
    void *status;            /* u32 word, OR-ed with 4 on an invalid kv_len, or NULL */
    const void *q;           /* f32 [m][n_q_head][head_dim] */
    const void *k_codes;     /* i8 [n_kv_head][cells][head_dim] */
    const void *k_scales;    /* f16 bits [n_kv_head][cells][head_dim/32] */
    const void *v_codes;     /* i8 [n_kv_head][cells][head_dim] */
    const void *v_scales;    /* f16 bits [n_kv_head][cells][head_dim/32] */
    const void *kv_len;      /* one i32 device word */
    int m;                   /* tokens, 1..16 */
    int n_q_head;            /* 16 (Humanizer) or 8 (Bolt) */
    int n_kv_head;           /* 1 or 8 (Humanizer), 2 (Bolt) */
    int head_dim;            /* 256 or 512 */
    int cells;               /* KV capacity per head */
    int window;              /* 0 all keys, else the last W keys */
    float scale;             /* |scale| < 1e30, Gemma uses 1.0 */
    float *part;             /* f32 split partials, required when split != 0 */
    int split;               /* 0: single-pass kernel; else the split-K path */
    int max_splits;          /* split capacity of part, 1..128 when split */
    int min_keys;            /* keys per split wanted, >= 8 when split */
} inference_cuda_attn_decode_gemma_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_attn_decode_gemma_args) == 120 && offsetof(inference_cuda_attn_decode_gemma_args, out) == 0 && offsetof(inference_cuda_attn_decode_gemma_args, status) == 8 && offsetof(inference_cuda_attn_decode_gemma_args, q) == 16 && offsetof(inference_cuda_attn_decode_gemma_args, k_codes) == 24 && offsetof(inference_cuda_attn_decode_gemma_args, k_scales) == 32 && offsetof(inference_cuda_attn_decode_gemma_args, v_codes) == 40 && offsetof(inference_cuda_attn_decode_gemma_args, v_scales) == 48 && offsetof(inference_cuda_attn_decode_gemma_args, kv_len) == 56 && offsetof(inference_cuda_attn_decode_gemma_args, m) == 64 && offsetof(inference_cuda_attn_decode_gemma_args, n_q_head) == 68 && offsetof(inference_cuda_attn_decode_gemma_args, n_kv_head) == 72 && offsetof(inference_cuda_attn_decode_gemma_args, head_dim) == 76 && offsetof(inference_cuda_attn_decode_gemma_args, cells) == 80 && offsetof(inference_cuda_attn_decode_gemma_args, window) == 84 && offsetof(inference_cuda_attn_decode_gemma_args, scale) == 88 && offsetof(inference_cuda_attn_decode_gemma_args, part) == 96 && offsetof(inference_cuda_attn_decode_gemma_args, split) == 104 && offsetof(inference_cuda_attn_decode_gemma_args, max_splits) == 108 && offsetof(inference_cuda_attn_decode_gemma_args, min_keys) == 112, "layout");
#else
_Static_assert(sizeof(inference_cuda_attn_decode_gemma_args) == 120 && offsetof(inference_cuda_attn_decode_gemma_args, out) == 0 && offsetof(inference_cuda_attn_decode_gemma_args, status) == 8 && offsetof(inference_cuda_attn_decode_gemma_args, q) == 16 && offsetof(inference_cuda_attn_decode_gemma_args, k_codes) == 24 && offsetof(inference_cuda_attn_decode_gemma_args, k_scales) == 32 && offsetof(inference_cuda_attn_decode_gemma_args, v_codes) == 40 && offsetof(inference_cuda_attn_decode_gemma_args, v_scales) == 48 && offsetof(inference_cuda_attn_decode_gemma_args, kv_len) == 56 && offsetof(inference_cuda_attn_decode_gemma_args, m) == 64 && offsetof(inference_cuda_attn_decode_gemma_args, n_q_head) == 68 && offsetof(inference_cuda_attn_decode_gemma_args, n_kv_head) == 72 && offsetof(inference_cuda_attn_decode_gemma_args, head_dim) == 76 && offsetof(inference_cuda_attn_decode_gemma_args, cells) == 80 && offsetof(inference_cuda_attn_decode_gemma_args, window) == 84 && offsetof(inference_cuda_attn_decode_gemma_args, scale) == 88 && offsetof(inference_cuda_attn_decode_gemma_args, part) == 96 && offsetof(inference_cuda_attn_decode_gemma_args, split) == 104 && offsetof(inference_cuda_attn_decode_gemma_args, max_splits) == 108 && offsetof(inference_cuda_attn_decode_gemma_args, min_keys) == 112, "layout");
#endif

/* One launch on stream (must not be NULL). Returns INFERENCE_CUDA_INVALID_ARGUMENT
 * before any CUDA call for a bad argument, else the launch status. */
int inference_cuda_attn_decode_gemma_launch(
    const inference_cuda_attn_decode_gemma_args *args, void *stream, int *out_stage);

/* Floats of `part` for the split path: m * n_q_head * max_splits * (2 + head_dim),
 * stats first then accumulators, per (token, q head, split) in that order. */
size_t inference_cuda_attn_decode_gemma_part_floats(
    int m, int n_q_head, int head_dim, int max_splits);

/* Host mirror of the device split plan: chunk keys per split and the active split
 * count of a visible range of n_keys keys. Identical arithmetic to the kernels;
 * the Rust side pins both to the same values. */
int inference_cuda_attn_decode_gemma_split_plan_raw(
    int n_keys, int max_splits, int min_keys, int *chunk, int *n_active);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_ATTN_DECODE_GEMMA_H */
