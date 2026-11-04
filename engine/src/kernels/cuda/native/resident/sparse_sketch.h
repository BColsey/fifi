/* The sparse-attention sketch append entry. Launch-only, like the entries of abi_resident.h.
 * Include after abi.h in C++; Rust mirrors the structs in native/resident/sparse_sketch.rs
 * (child module of native/resident.rs).
 *
 * The entry maintains the key-sketch plane of the query-aware sparse KV path: for every KV head
 * it projects the post-norm, post-rope k rows onto a fixed orthonormal basis and stores the
 * projection as one fp16 cell per position. Frozen layout: `sketch` is fp16
 * [n_kv_head][cells][rank], `cell = position`, and `basis` is f32 [n_kv_head][rank][head_dim]
 * (output-major, so each rank's row is contiguous). The stored value of rank r is
 * RNE_f16(sum_d basis[h][r][d] * k[t][h][d]), the inner product accumulated in f32 by a fixed
 * 32-lane shuffle tree (bit-deterministic per launch geometry); a NaN in k reaches the stored
 * fp16 cell and poisons the plane.
 *
 * Two entries, exactly as kv8_append: the decode append takes m rows (1..16) of k, f32
 * [m][n_kv_head][head_dim], and writes cell *pos + t of the plane, reading the device i32 word
 * pos (StepParams.pos, offset 4). Poison rule (as attn_prep and kv8_append): the kernel tests
 * the word against the cells before any other memory access; if *pos < 0 or *pos + m > cells
 * (64-bit sum) nothing is written and mask 4 (INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of
 * resident/step_params.h) is OR-ed into *status by one thread. The chunk append takes t rows of
 * prefill at the HOST base position pos0 (0 <= pos0, pos0 + t <= cells), checked here before
 * any launch.
 *
 * Geometry is run-time: n_kv_head 1..64, head_dim a multiple of 32 in 32..1024 (one warp holds
 * the row, one thread per 32-strided value), rank 2..head_dim, cells 1..2^30 - 1. Both entries
 * enqueue one kernel on stream (NULL: the default stream; the Rust wrapper always passes one)
 * and return INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_SPARSE_SKETCH_H
#define INFERENCE_CUDA_RESIDENT_SPARSE_SKETCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The decode append: m rows at the device position word. */
typedef struct inference_cuda_sparse_sketch_args {
    const void *k;        /* f32 [m][n_kv_head][head_dim] */
    const void *basis;    /* f32 [n_kv_head][rank][head_dim] */
    void *sketch;         /* fp16 [n_kv_head][cells][rank] */
    const void *pos;      /* one device i32: the position of token 0 */
    void *status;         /* one device i32: receives mask 4 on a refused launch */
    int m;                /* tokens, 1..16 */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* a multiple of 32, 32..1024 */
    int rank;             /* 2..head_dim */
    int cells;            /* plane cells per head (the KV capacity), 1..2^30 - 1 */
} inference_cuda_sparse_sketch_args;

/* The prefill chunk append: t rows at the host base position pos0. */
typedef struct inference_cuda_sparse_sketch_chunk_args {
    const void *k;        /* f32 [t][n_kv_head][head_dim] */
    const void *basis;    /* f32 [n_kv_head][rank][head_dim] */
    void *sketch;         /* fp16 [n_kv_head][cells][rank] */
    int pos0;             /* the position of token 0, 0..cells - t */
    int t;                /* rows, 1..65535 (grid y) */
    int n_kv_head;        /* 1..64 */
    int head_dim;         /* a multiple of 32, 32..1024 */
    int rank;             /* 2..head_dim */
    int cells;            /* plane cells per head, 1..2^30 - 1 */
} inference_cuda_sparse_sketch_chunk_args;

#ifdef __cplusplus
static_assert(alignof(inference_cuda_sparse_sketch_args) == 8 && sizeof(inference_cuda_sparse_sketch_args) == 64 && offsetof(inference_cuda_sparse_sketch_args, k) == 0 && offsetof(inference_cuda_sparse_sketch_args, basis) == 8 && offsetof(inference_cuda_sparse_sketch_args, sketch) == 16 && offsetof(inference_cuda_sparse_sketch_args, pos) == 24 && offsetof(inference_cuda_sparse_sketch_args, status) == 32 && offsetof(inference_cuda_sparse_sketch_args, m) == 40,
              "inference_cuda_sparse_sketch_args layout");
static_assert(alignof(inference_cuda_sparse_sketch_chunk_args) == 8 && sizeof(inference_cuda_sparse_sketch_chunk_args) == 48 && offsetof(inference_cuda_sparse_sketch_chunk_args, k) == 0 && offsetof(inference_cuda_sparse_sketch_chunk_args, basis) == 8 && offsetof(inference_cuda_sparse_sketch_chunk_args, sketch) == 16 && offsetof(inference_cuda_sparse_sketch_chunk_args, pos0) == 24 && offsetof(inference_cuda_sparse_sketch_chunk_args, t) == 28,
              "inference_cuda_sparse_sketch_chunk_args layout");
#else
_Static_assert(_Alignof(inference_cuda_sparse_sketch_args) == 8 && sizeof(inference_cuda_sparse_sketch_args) == 64 && offsetof(inference_cuda_sparse_sketch_args, k) == 0 && offsetof(inference_cuda_sparse_sketch_args, basis) == 8 && offsetof(inference_cuda_sparse_sketch_args, sketch) == 16 && offsetof(inference_cuda_sparse_sketch_args, pos) == 24 && offsetof(inference_cuda_sparse_sketch_args, status) == 32 && offsetof(inference_cuda_sparse_sketch_args, m) == 40,
               "inference_cuda_sparse_sketch_args layout");
_Static_assert(_Alignof(inference_cuda_sparse_sketch_chunk_args) == 8 && sizeof(inference_cuda_sparse_sketch_chunk_args) == 48 && offsetof(inference_cuda_sparse_sketch_chunk_args, k) == 0 && offsetof(inference_cuda_sparse_sketch_chunk_args, basis) == 8 && offsetof(inference_cuda_sparse_sketch_chunk_args, sketch) == 16 && offsetof(inference_cuda_sparse_sketch_chunk_args, pos0) == 24 && offsetof(inference_cuda_sparse_sketch_chunk_args, t) == 28,
               "inference_cuda_sparse_sketch_chunk_args layout");
#endif

/* One launch (grid n_kv_head by m, 32 threads). */
int inference_cuda_sparse_sketch_launch(const inference_cuda_sparse_sketch_args *args,
                                        void *stream, int *out_stage);
/* One launch (grid n_kv_head by t, 32 threads). */
int inference_cuda_sparse_sketch_chunk_launch(const inference_cuda_sparse_sketch_chunk_args *args,
                                              void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_SPARSE_SKETCH_H */
