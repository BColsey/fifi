/*
 * The in-house IQ4_XS decode matvec. No vendored includes: the weights are the file's own
 * 136-byte IQ4_XS super-blocks (f16 d, u16 scales_h, scales_l[4], qs[128]; see the layout
 * note at iq4xs_terms), the activation is the shared quantize_q8_1 output (36 bytes per 32
 * values: fp16 d, fp16 s, 32 int8 quants; only d and the quants feed this kernel), and the
 * arithmetic is plain f32. The 16-entry value table rides the kernel arguments from the
 * shared iq4_nl_values.cuh (never __constant__ storage; see that header).
 *
 * One warp owns one weight row and walks every activation column in turn, like matvec_q3k.
 * Each lane carries the eight sub-block values at its own position within the 32-value
 * sub-block: lanes 0..15 read the low nibble of qs[16*ib + lane], lanes 16..31 the high
 * nibble, weighed by the sub-block's 6-bit signed scale. The numerics match the CPU dequant
 * oracle (weights/gguf_dequant.rs iq4_xs) exactly.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "abi.h"
#include "device.h"
#include "iq4_nl_values.cuh"
#include "resident/matvec_iq4xs.h"

namespace {

const int kQK = 256;    /* values per IQ4_XS super-block */
const int kBlockBytes = 136;
const int kXBlock = 36; /* bytes per q8_1 block of 32 values */
const int kWarps = 8;   /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* One lane's f32 terms of super-block b: sub-block ib (0..7) carries values [32*ib,
 * 32*ib+32), and lane holds position lane of each. The 6-bit scale index is 4 low bits of
 * scales_l[ib/2] at nibble ib%2 plus the 2 high bits of scales_h at shift 2*ib, biased by
 * 32; the value weighs d * (scale - 32) * kvalues[nibble of qs[16*ib + (lane & 15)]]
 * against the q8_1 block of the same 32-value window (d8 folded in, like the Q5_K/Q3_K
 * kernels). */
__device__ __forceinline__ void iq4xs_terms(const unsigned char *block,
                                            const unsigned char *xrow, int b, int lane,
                                            const float *kv, float *acc) {
    const float d = half_at(block);
    unsigned short scales_h_raw;
    memcpy(&scales_h_raw, block + 2, sizeof(scales_h_raw));
    const unsigned int scales_h = scales_h_raw;
    const unsigned char *scales_l = block + 4;
    const unsigned char *qs = block + 8;
    const int l = lane & 15;
    #pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const int lo = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0f;
        const int hi = (int)((scales_h >> (2 * ib)) & 3u);
        const float dl = d * (float)((lo | (hi << 4)) - 32);
        const unsigned char q = qs[16 * ib + l];
        const int nib = lane < 16 ? (q & 0x0f) : (q >> 4);
        const unsigned char *xb = xrow + (size_t)(8 * b + ib) * kXBlock;
        const float xv = half_at(xb) * (float)(int8_t)xb[4 + lane];
        *acc = fmaf(dl * kv[nib], xv, *acc);
    }
}

__global__ void __launch_bounds__(256, 2) matvec_iq4xs(const unsigned char *w,
                                                       const unsigned char *x_q8, float *y,
                                                       int k, int rows, int ncols, int w_blocks,
                                                       size_t x_row_bytes,
                                                       iq4nl_table iq4nl) {
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int row = blockIdx.x * kWarps + warp;
    if (row >= rows) {
        return;
    }
    const unsigned char *wrow = w + (size_t)row * w_blocks * kBlockBytes;
    for (int col = 0; col < ncols; ++col) {
        const unsigned char *xrow = x_q8 + (size_t)col * x_row_bytes;
        float acc = 0.0f;
        for (int b = 0; b < w_blocks; ++b) {
            iq4xs_terms(wrow + (size_t)b * kBlockBytes, xrow, b, lane, iq4nl.v, &acc);
        }
        /* Warp reduce the 32 lane partials; every lane holds the sum after the loop. */
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            y[(size_t)col * rows + row] = acc;
        }
    }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_matvec_iq4xs_launch(const inference_cuda_matvec_iq4xs_args *args,
                                                  size_t x_row_bytes, void *stream,
                                                  int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || k % kQK != 0 || rows <= 0 || ncols <= 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        /* The Rust gate's twins: the column bound, and an x_row_bytes that covers at
         * least the unpadded k/32 q8_1 blocks of one activation row. */
        if (ncols > 8 || x_row_bytes < (size_t)(k / 32) * kXBlock) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t w_bytes = (size_t)rows * (size_t)(k / kQK) * kBlockBytes;
        const size_t x_bytes = x_row_bytes * (size_t)ncols;
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        if (!aligned(args->w, 4) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
        matvec_iq4xs<<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>(
            (const unsigned char *)args->w, (const unsigned char *)args->x_q8, args->y, k, rows,
            ncols, k / kQK, x_row_bytes, make_iq4nl_table());
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
