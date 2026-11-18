/*
 * The in-house Q8_0 decode matvec, sibling of matvec_q40.cu. No vendored includes:
 * the weights are the file's own 34-byte Q8_0 blocks (an f16 scale d, then 32 int8
 * quants; value i of the block is d * q_i), the activation is the shared quantize_q8_1
 * output (36 bytes per 32 values: fp16 d, fp16 s, 32 int8 quants), and the arithmetic is
 * plain f32.
 *
 * The geometry is the simplest of the family: one 32-value weight block faces exactly one
 * q8_1 block, so lane l owns value l of every block — one int8 weight quant, one int8
 * activation quant, one fma per block. One warp owns one weight row and walks every
 * activation column in turn, so a row's blocks stay hot in cache across columns.
 *
 * Unlike the vendored mul_mat_vec_q launcher, whose launch geometry changes with the column
 * count (columns then differ between an ncols launch and single-column launches — the row
 * that fails at 5e-4 relative), this kernel computes every (row, column) dot with the same
 * lane arithmetics regardless of ncols: column j of an 8-column launch is bit-identical to
 * a 1-column launch over the same bytes.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q40.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/matvec_q80.h"

namespace {

const int kQK = 32;     /* values per Q8_0 block */
const int kBlockBytes = 34;
const int kXBlock = 36; /* bytes per q8_1 block of 32 values */
const int kWarps = 8;   /* warps per CTA */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* Four weight quants as one dp4a operand. The 34-byte Q8_0 block puts its quants
 * at offset 2, so only 2-byte alignment holds; two u16 loads build the int. */
__device__ __forceinline__ int quants4_w(const unsigned char *q, int group) {
    const uint16_t *p = (const uint16_t *)(q + group * 4);
    return (int)p[0] | ((int)p[1] << 16);
}

__global__ void __launch_bounds__(256, 2) matvec_q80(const unsigned char *w,
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
    /* One warp per row, one LANE per 32-value block: the lane chains eight dp4a
     * over the whole block (an exact integer dot), folds the two f16 scales in
     * once, and accumulates in block order with the b&32 split; the reduce is a
     * full xor tree. Every column repeats the same sequence, so the results do
     * not depend on how many columns share the launch. */
    float acc0[8] = {}, acc1[8] = {};
    for (int b = lane; b < w_blocks; b += 32) {
        const unsigned char *wb = wrow + (size_t)b * kBlockBytes;
        int v[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = quants4_w(wb + 2, i);
        }
        const float d_w = half_at(wb);
        for (int col = 0; col < ncols; ++col) {
            const unsigned char *act = x_q8 + (size_t)col * x_row_bytes + (size_t)b * kXBlock;
            const int *a = (const int *)(act + 4);
            int d = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                d = __dp4a(v[i], a[i], d);
            }
            const float s = d_w * half_at(act);
            /* __fmaf_rn pins the fused form: with the runtime-column accumulator
             * arrays in local memory the compiler emits FMUL+FADD (two roundings)
             * while the register-resident split-plane kernel contracts to one FFMA -
             * a per-block 1-ulp drift at k not tile-sized (k = 2560 = 80 blocks,
             * caught as the serve text divergence). */
            if (b & 32) {
                acc1[col] = __fmaf_rn(s, (float)d, acc1[col]);
            } else {
                acc0[col] = __fmaf_rn(s, (float)d, acc0[col]);
            }
        }
    }
    for (int col = 0; col < ncols; ++col) {
        float acc = acc0[col] + acc1[col];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o);
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

extern "C" int inference_cuda_matvec_q80_launch(const inference_cuda_matvec_q80_args *args,
                                                size_t x_row_bytes, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || k % kQK != 0 || rows <= 0 || ncols <= 0 || ncols > 8) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int w_blocks = k / kQK;
        const size_t w_bytes = (size_t)rows * (size_t)w_blocks * kBlockBytes;
        const size_t x_bytes = x_row_bytes * (size_t)ncols;
        const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
        /* The f16 scale of a 34-byte block starts at an even offset from w, so 2 bytes of
         * weight alignment are enough; the q8_1 rows and the f32 out want 4. */
        if (!aligned(args->w, 2) || !aligned(args->x_q8, 4) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->w, w_bytes, args->y, y_bytes) ||
            !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int blocks = (rows + kWarps - 1) / kWarps;
        matvec_q80<<<blocks, kWarps * 32, 0, (cudaStream_t)stream>>>(
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
