/*
 * The in-house Q3_K decode matvec. No vendored includes: the weights are the file's own
 * 110-byte Q3_K blocks (see the layout note at q3k_terms), the activation is the shared
 * quantize_q8_1 output (36 bytes per 32 values: fp16 d, fp16 s, 32 int8 quants; only d and
 * the quants feed this kernel), and the arithmetic is plain f32.
 *
 * One warp owns one weight row and walks every activation column in turn, like matvec_q5k.
 * Each lane carries the eight block values whose low nibble is lane & 15: two per chunk c (0..3)
 * at group g2 (0..1), where the half selector L >> 4 picks the qs bit pair and hmask bit
 * the value reads. The numerics match the CPU dequant oracle (weights/gguf_dequant.rs q3_k)
 * and the staged GEMM decoder (prefill/gemm_dq_decoders.cuh Q3kDecoder) exactly.
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
#include "resident/matvec_q3k.h"

namespace {

const int kQK = 256;    /* values per Q3_K block */
const int kBlockBytes = 110;
const int kXBlock = 36; /* bytes per q8_1 block of 32 values */
const int kWarps = 8;   /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* One of the 16 packed 6-bit sub-scales of the ggml aux shuffle, identical to
 * Q3kDecoder::scale_of (prefill/gemm_dq_decoders.cuh) and the CPU oracle: scale `is` is
 * byte is&3 of row is>>2, low nibble of ks[(row&1)*4 + byte] at shift 4*(row>>1), plus the
 * 2-bit high field of ks[8+byte] at shift 2*row. */
__device__ __forceinline__ int q3k_scale_of(const unsigned char *ks, int is) {
    const int row = is >> 2;
    const int byte = is & 3;
    const int lo = (ks[(row & 1) * 4 + byte] >> (4 * (row >> 1))) & 0x0f;
    const int hi = ((ks[8 + byte] >> (2 * row)) & 0x03) << 4;
    return lo | hi;
}

/* One lane's f32 terms of the block: chunk c (0..3) carries values [64c, 64c+64); within
 * it, j = 2*(c&1) + (lane>>4) fixes the qs bit pair and hmask bit, and group g2 owns
 * values [64c + 32*(lane>>4) + 16*g2, +16) at low nibble l = lane&15. The value weighs
 *   d * (scale - 32) * (low2 - (hmask clear ? 4 : 0))
 * against the q8_1 block of its 32-value window. */
__device__ __forceinline__ void q3k_terms(const unsigned char *block, const unsigned char *xrow,
                                          int b, int c, int lane, float *acc) {
    const float d = half_at(block + 108);
    const unsigned char *hm = block;
    const unsigned char *qs = block + 32 + 32 * (c >> 1);
    const unsigned char *ks = block + 96;
    const int l = lane & 15;
    const int j = 2 * (c & 1) + (lane >> 4);
    const unsigned mask_bit = 1u << (4 * (c >> 1) + j);
    #pragma unroll
    for (int g2 = 0; g2 < 2; ++g2) {
        const int qv = (int)((qs[16 * g2 + l] >> (2 * j)) & 3) -
                       ((hm[16 * g2 + l] & mask_bit) == 0 ? 4 : 0);
        const float dl = d * (float)(q3k_scale_of(ks, (c >> 1) * 8 + 2 * j + g2) - 32);
        /* Value 64c + 32*jp + 16g2 + l (jp = lane>>4 picks the half) sits in q8_1 block
         * 8b + 2c + jp at offset 16g2 + l; j is only the qs/hmask shift of the half. */
        const unsigned char *xb = xrow + (size_t)(8 * b + 2 * c + (lane >> 4)) * kXBlock;
        const float xv = half_at(xb) * (float)(int8_t)xb[4 + 16 * g2 + l];
        *acc = fmaf(dl * (float)qv, xv, *acc);
    }
}

__global__ void __launch_bounds__(256, 2) matvec_q3k(const unsigned char *w,
                                                     const unsigned char *x_q8, float *y, int k,
                                                     int rows, int ncols, int w_blocks,
                                                     size_t x_row_bytes) {
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
            const unsigned char *block = wrow + (size_t)b * kBlockBytes;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                q3k_terms(block, xrow, b, c, lane, &acc);
            }
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

extern "C" int inference_cuda_matvec_q3k_launch(const inference_cuda_matvec_q3k_args *args,
                                                size_t x_row_bytes, void *stream, int *out_stage) {
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
        if (!apart(args->w, w_bytes, args->y, y_bytes) || !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
        matvec_q3k<<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>(
            (const unsigned char *)args->w, (const unsigned char *)args->x_q8, args->y, k, rows,
            ncols, k / kQK, x_row_bytes);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
