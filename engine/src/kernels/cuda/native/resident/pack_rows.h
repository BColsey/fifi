/* The pack kernel of a K-row verify window, per-kernel header (native/resident/pack_rows.h).
 * Included by pack_rows.cu and read by the Rust FFI mirror; nothing else.
 *
 * The rows kernel of the EXL3 projections writes each role's output tight, f32 [m][n_i]; the
 * kernels that read a layer's projections (the delta rule for z, beta and alpha; attn_prep and
 * attn_decode for q, k and v) read ONE row of ld floats with the roles at fixed column offsets.
 * pack_rows copies up to three tight sources into that row layout in one launch:
 *   dst[r * ld + dst_off_i + c] = src_i[r * n_i + c]  for r < m, c < n_i, i < segs.
 * It is a bit-exact copy: every element is moved as one 32-bit word, never converted or touched
 * by float arithmetic, so a NaN payload, a negative zero or a denormal arrives as it left. Columns
 * of dst outside the segments are not written. The segments' column ranges must not overlap, each
 * must end within ld, and no source may overlap dst. Limits: m in [1, 16], segs in [1, 3], n_i in
 * [1, 65536], ld in [1, 1 << 20]; every pointer is 4-byte aligned. Enqueues one kernel on stream
 * (not NULL); returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. */
#ifndef INFERENCE_CUDA_RESIDENT_PACK_ROWS_H
#define INFERENCE_CUDA_RESIDENT_PACK_ROWS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_pack_segment {
    const void *src;
    int n;
    int dst_off;
} inference_cuda_pack_segment;

typedef struct inference_cuda_pack_rows_args {
    void *dst;
    inference_cuda_pack_segment seg[3];
    int m;
    int ld;
    int segs;
} inference_cuda_pack_rows_args;

int inference_cuda_pack_rows_launch(const inference_cuda_pack_rows_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_pack_segment) == 16 && offsetof(inference_cuda_pack_segment, n) == 8 &&
                  offsetof(inference_cuda_pack_segment, dst_off) == 12,
              "inference_cuda_pack_segment layout");
static_assert(sizeof(inference_cuda_pack_rows_args) == 72 && offsetof(inference_cuda_pack_rows_args, seg) == 8 &&
                  offsetof(inference_cuda_pack_rows_args, m) == 56 &&
                  offsetof(inference_cuda_pack_rows_args, ld) == 60 &&
                  offsetof(inference_cuda_pack_rows_args, segs) == 64,
              "inference_cuda_pack_rows_args layout");
}
#endif

#endif
