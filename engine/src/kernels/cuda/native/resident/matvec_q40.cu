/*
 * The in-house Q4_0 decode matvec, the little sibling of matvec_q5k.cu. No vendored
 * includes: the weights are the file's own 18-byte Q4_0 blocks (an f16 scale d, then 16
 * code bytes whose byte j carries the codes of values j (low nibble) and j + 16 (high
 * nibble); value i of the block is d * (code_i - 8)), the activation is the shared quantize_q8_1 output
 * (36 bytes per 32 values: fp16 d, fp16 s, 32 int8 quants), and the arithmetic is plain f32.
 *
 * The geometry is simpler than Q5_K's: one 32-value weight block faces exactly one q8_1
 * block, so lane l owns value l of every block — one code nibble, one int8 quant, one fma
 * per block. One warp owns one weight row and walks every activation column in turn, so a
 * row's blocks stay hot in cache across columns.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q5k.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/matvec_q40.h"

namespace {

const int kQK = 32;     /* values per Q4_0 block */
const int kBlockBytes = 18;
const int kXBlock = 36; /* bytes per q8_1 block of 32 values */
const int kWarps = 8;   /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

__global__ void __launch_bounds__(256, 2) matvec_q40(const unsigned char *w,
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
            const unsigned char *xb = xrow + (size_t)b * kXBlock;
            /* Value lane of block b, GGUF order: lanes 0..16 read the LOW nibble of
             * code byte lane, lanes 16..32 the HIGH nibble of code byte lane - 16; the
             * quant is the lane's int8 of the same-indexed q8_1 block. */
            unsigned char code = block[2 + (lane & 15)];
            code = (lane < 16) ? (unsigned char)(code & 0x0f) : (unsigned char)(code >> 4);
            const float v = half_at(block) * ((float)(int)code - 8.0f);
            const float x = half_at(xb) * (float)(int8_t)xb[4 + lane];
            acc = fmaf(v, x, acc);
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

extern "C" int inference_cuda_matvec_q40_launch(const inference_cuda_matvec_q40_args *args,
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
        const int w_blocks = k / kQK;
        const size_t w_bytes = (size_t)rows * (size_t)w_blocks * kBlockBytes;
        const size_t x_bytes = x_row_bytes * (size_t)ncols;
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        /* The f16 scale of an 18-byte block starts at an even offset from w, so 2 bytes of
         * weight alignment are enough; the q8_1 rows and the f32 out want 4. */
        if (!aligned(args->w, 2) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
        matvec_q40<<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>(
            (const unsigned char *)args->w, (const unsigned char *)args->x_q8, args->y, k, rows,
            ncols, w_blocks, x_row_bytes);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
