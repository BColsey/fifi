/*
 * Acceptance guard and F16 publication of KV cache rows, device-side twin of the
 * host `f16_row` guard in family/flint/cuda_ffn/device/runtime.rs. A value is
 * accepted only when it is finite, has its low 16 bits clear and does not
 * overflow F16; the stored half-word is the round-to-nearest-even conversion
 * (`__float2half_rn`, the same bits as host `half::f16::from_f32`), so the
 * device rows stay byte-identical to the host publication. Any reject
 * (dirty low bits including would-be subnormals, F16 overflow with clean bits,
 * NaN, +-inf) sets flag |= 1 and writes a zero half-word; the flag is never
 * cleared. y may be NULL for the guard-only Q8G32 path: then no half-row is
 * written and only the flag is computed, removing the host D2H read there too.
 */
#ifndef INFERENCE_CUDA_RESIDENT_KV_F16_ROWS_H
#define INFERENCE_CUDA_RESIDENT_KV_F16_ROWS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_kv_f16_args {
    const float *x;   /* f32 [rows][columns], read */
    size_t x_bytes;
    float *y;         /* f16 bits [rows][columns], written; NULL = guard only */
    size_t y_bytes;
    unsigned *flag;   /* sticky fault word, |= 1 on any reject */
    size_t flag_bytes;
    size_t rows;
    size_t columns;   /* values per row; the KV guard uses 512 */
} inference_cuda_kv_f16_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_kv_f16_args) == 64 &&
              offsetof(inference_cuda_kv_f16_args, y) == 16 &&
              offsetof(inference_cuda_kv_f16_args, flag) == 32 &&
              offsetof(inference_cuda_kv_f16_args, rows) == 48 &&
              offsetof(inference_cuda_kv_f16_args, columns) == 56, "KV F16 layout");
#endif

/* One launch on an explicit stream (must not be NULL). x, flag and — when y is
 * non-NULL — y must be sized, four-byte aligned and pairwise disjoint. Returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad argument, else
 * the launch status. Never synchronizes; the flag is read after completion. */
int inference_cuda_kv_f16_rows(const inference_cuda_kv_f16_args *args, void *stream,
                               int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
