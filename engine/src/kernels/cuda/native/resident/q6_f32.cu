/* Experimental Q6_K/F32 projection. The Rust entry is test-only. */
#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/gguf_decode.cuh"
#include "resident/q6_f32.h"

namespace {
constexpr int kWarps = 8;
constexpr int kWarpSize = 32;
constexpr int kThreads = kWarps * kWarpSize;
constexpr int kBlockValues = 256;
constexpr int kBlockBytes = 210;

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? a_bytes <= y - x : b_bytes <= x - y;
}

/* Every warp owns one complete output row; a whole warp exits for a row tail. The full
 * shuffle mask is valid only with the fixed 256-thread launch below. Lane i handles column i
 * of each 32-value window, so adjacent lanes read adjacent weights and activations. */
__global__ void q6_f32_kernel(const unsigned char *w, const float *x, float *y, int rows, int k) {
    const int row = (int)(blockIdx.x * kWarps + threadIdx.x / kWarpSize);
    const int lane = (int)(threadIdx.x % kWarpSize);
    if (row >= rows) return;
    const int blocks = k / kBlockValues;
    const unsigned char *wr = w + (size_t)row * blocks * kBlockBytes;
    float sum = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        const unsigned char *p = wr + (size_t)b * kBlockBytes;
        const float d = gguf::rd_half(p + 208);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned lo0 = p[64 * h + lane];
            const unsigned lo1 = p[64 * h + 32 + lane];
            const unsigned hi = p[128 + 32 * h + lane];
#pragma unroll
            for (int qd = 0; qd < 4; ++qd) {
                const unsigned lo = (qd & 1) ? lo1 : lo0;
                const int q = (int)(((lo >> (4 * (qd >> 1))) & 15u) |
                                    (((hi >> (2 * qd)) & 3u) << 4)) - 32;
                const int sc = (int)(signed char)p[192 + 8 * h + 2 * qd + lane / 16];
                const float weight = __fmul_rn(__fmul_rn(d, (float)sc), (float)q);
                sum = fmaf(weight, x[(size_t)b * kBlockValues + 128 * h + 32 * qd + lane], sum);
            }
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[row] = sum;
}
}  // namespace

extern "C" int inference_cuda_q6_f32_launch(const inference_cuda_q6_f32_args *args,
                                             void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || args->rows <= 0 || args->rows > INT_MAX - (kWarps - 1) ||
            args->k <= 0 || args->k % kBlockValues != 0 ||
            (uintptr_t)args->w % 2 != 0 || (uintptr_t)args->x % 4 != 0 ||
            (uintptr_t)args->y % 4 != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t row_bytes = (size_t)(args->k / kBlockValues) * kBlockBytes;
        if ((size_t)args->rows > SIZE_MAX / row_bytes) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t w_bytes = (size_t)args->rows * row_bytes;
        const size_t x_bytes = (size_t)args->k * sizeof(float);
        const size_t y_bytes = (size_t)args->rows * sizeof(float);
        if (!apart(args->y, y_bytes, args->w, w_bytes) ||
            !apart(args->y, y_bytes, args->x, x_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const cudaError_t pending = cudaPeekAtLastError();
        if (pending != cudaSuccess && sticky_prior(pending, out_stage)) return (int)pending;
        const unsigned grid = (unsigned)((args->rows - 1) / kWarps + 1);
        q6_f32_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
            (const unsigned char *)args->w, args->x, args->y, args->rows, args->k);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
