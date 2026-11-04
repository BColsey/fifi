// The hyper-connection read and apply of Qwen3.8 Flash-Next (see resident/hyper.h for the
// formulas and the argument contract). Correctness first: one warp per output row of each dense
// product and plain f32 accumulation; nothing is fused across the five steps yet.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/hyper.h"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxBranches = 16;
constexpr int kMaxHidden = 65536;
// The up projection launches width / kWarps blocks in grid y, whose limit is 65535.
constexpr int kMaxWidth = 1 << 18;
constexpr int kMaxRank = 4096;
constexpr int kWarps = 8;
constexpr int kNormThreads = 256;
constexpr int kDtypeF16 = 0;
constexpr int kDtypeBf16 = 1;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

__device__ __forceinline__ float load(const __half *w, size_t i) { return __half2float(w[i]); }
__device__ __forceinline__ float load(const __nv_bfloat16 *w, size_t i) {
    return __bfloat162float(w[i]);
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One block per (token, branch): the RMS of the branch, then the (1 + w)-weighted row.
template <typename W>
__global__ void hyper_norm_k(const float *__restrict__ x, const W *__restrict__ norm_w,
                             float *__restrict__ normed, int branches, int hidden, float eps) {
    const int t = blockIdx.x / branches;
    const int b = blockIdx.x % branches;
    const size_t width = (size_t)branches * hidden;
    const float *row = x + (size_t)t * width + (size_t)b * hidden;
    float *out = normed + (size_t)t * width + (size_t)b * hidden;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) sum += row[d] * row[d];
    __shared__ float partial[kNormThreads / 32];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kNormThreads / 32 ? partial[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) partial[0] = v;
    }
    __syncthreads();
    const float denom = sqrtf(partial[0] / (float)hidden + eps);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float w = load(norm_w, (size_t)b * hidden + d);
        out[d] = row[d] * (1.0f + w) / denom;
    }
}

// One warp per output row r of the down projection (r < rank) and of inject (r >= rank).
template <typename W>
__global__ void hyper_down_k(const float *__restrict__ normed, const W *__restrict__ down,
                             const W *__restrict__ inject, float *__restrict__ low,
                             float *__restrict__ gate, int branches, int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * kWarps + (threadIdx.x >> 5);
    const int rows = rank + (inject != NULL ? branches : 0);
    if (r >= rows) return;
    const int t = blockIdx.x;
    const float *in = normed + (size_t)t * width;
    const W *w = r < rank ? down + (size_t)r * width : inject + (size_t)(r - rank) * width;
    float acc = 0.0f;
    for (int i = lane; i < width; i += 32) acc += load(w, i) * in[i];
    acc = warp_sum(acc);
    if (lane != 0) return;
    const float scaled = acc / (float)branches;
    if (r < rank) {
        low[(size_t)t * rank + r] = scaled * sigmoid_f(scaled);
    } else {
        gate[(size_t)t * branches + (r - rank)] = 2.0f * sigmoid_f(scaled);
    }
}

// One warp per output row i of the up projection.
template <typename W>
__global__ void hyper_up_k(const float *__restrict__ low, const W *__restrict__ up,
                           float *__restrict__ mix, int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (i >= width) return;
    const int t = blockIdx.x;
    const float *in = low + (size_t)t * rank;
    const W *w = up + (size_t)i * rank;
    float acc = 0.0f;
    for (int j = lane; j < rank; j += 32) acc += load(w, j) * in[j];
    acc = warp_sum(acc);
    if (lane == 0) mix[(size_t)t * width + i] = acc;
}

__global__ void hyper_combine_k(const float *__restrict__ normed, const float *__restrict__ mix,
                                float *__restrict__ mixed, int branches, int hidden) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const int t = blockIdx.x;
    const size_t base = (size_t)t * branches * hidden + d;
    float sum = 0.0f;
    for (int b = 0; b < branches; ++b) {
        const size_t at = base + (size_t)b * hidden;
        sum += sigmoid_f(mix[at]) * normed[at];
    }
    mixed[(size_t)t * hidden + d] = sum / (float)branches;
}

__global__ void hyper_apply_k(float *__restrict__ x, const float *__restrict__ gate,
                              const float *__restrict__ block, int branches, int hidden) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    const int width = branches * hidden;
    if (i >= width) return;
    const int t = blockIdx.x;
    const int b = i / hidden;
    const int d = i - b * hidden;
    x[(size_t)t * width + i] += gate[(size_t)t * branches + b] * block[(size_t)t * hidden + d];
}

bool shape_ok(int m, int branches, int hidden) {
    return m >= 1 && m <= kMaxM && branches >= 1 && branches <= kMaxBranches && hidden >= 1 &&
           hidden <= kMaxHidden && (long long)branches * hidden <= kMaxWidth;
}

// Mirrors the Rust wrapper term by term; the Rust side refuses everything this refuses.
bool valid_read(const inference_cuda_hyper_read_args &a, void *stream) {
    if (stream == NULL || a.x == NULL || a.norm_w == NULL || a.down == NULL || a.up == NULL ||
        a.normed == NULL || a.low == NULL || a.mix == NULL || a.mixed == NULL) {
        return false;
    }
    if ((a.inject == NULL) != (a.gate == NULL)) return false;
    if (!shape_ok(a.m, a.branches, a.hidden) || a.rank < 1 || a.rank > kMaxRank) return false;
    if (a.weight_dtype != kDtypeF16 && a.weight_dtype != kDtypeBf16) return false;
    if (!(a.eps >= 0.0f) || !(a.eps < INFINITY)) return false;
    const void *f32s[] = {a.x, a.normed, a.low, a.mix, a.mixed};
    for (const void *p : f32s) {
        if (!aligned(p, 4)) return false;
    }
    if (a.gate != NULL && !aligned(a.gate, 4)) return false;
    const void *weights[] = {a.norm_w, a.down, a.up};
    for (const void *p : weights) {
        if (!aligned(p, 2)) return false;
    }
    if (a.inject != NULL && !aligned(a.inject, 2)) return false;
    const size_t m = (size_t)a.m, width = (size_t)a.branches * a.hidden, rank = (size_t)a.rank;
    const Span written[] = {{a.normed, m * width * 4},
                            {a.low, m * rank * 4},
                            {a.mix, m * width * 4},
                            {a.mixed, m * (size_t)a.hidden * 4},
                            {a.gate, a.gate != NULL ? m * (size_t)a.branches * 4 : 0}};
    const Span read[] = {{a.x, m * width * 4},
                         {a.norm_w, width * 2},
                         {a.down, rank * width * 2},
                         {a.up, width * rank * 2},
                         {a.inject, a.inject != NULL ? (size_t)a.branches * width * 2 : 0}};
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 5; ++j) {
            if (!apart(written[i], written[j])) return false;
        }
        for (const Span &r : read) {
            if (!apart(written[i], r)) return false;
        }
    }
    return true;
}

bool valid_apply(const inference_cuda_hyper_apply_args &a, void *stream) {
    if (stream == NULL || a.x == NULL || a.gate == NULL || a.block == NULL) return false;
    if (!shape_ok(a.m, a.branches, a.hidden)) return false;
    if (!aligned(a.x, 4) || !aligned(a.gate, 4) || !aligned(a.block, 4)) return false;
    const size_t m = (size_t)a.m, width = (size_t)a.branches * a.hidden;
    const Span x = {a.x, m * width * 4};
    return apart(x, {a.gate, m * (size_t)a.branches * 4}) &&
           apart(x, {a.block, m * (size_t)a.hidden * 4});
}

template <typename W>
void launch_read(const inference_cuda_hyper_read_args &a, cudaStream_t s, Failure &failure) {
    const int width = a.branches * a.hidden;
    const W *norm_w = (const W *)a.norm_w;
    hyper_norm_k<W><<<a.m * a.branches, kNormThreads, 0, s>>>(
        (const float *)a.x, norm_w, (float *)a.normed, a.branches, a.hidden, a.eps);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const int down_rows = a.rank + (a.inject != NULL ? a.branches : 0);
    const dim3 down_grid(a.m, (down_rows + kWarps - 1) / kWarps);
    hyper_down_k<W><<<down_grid, kWarps * 32, 0, s>>>(
        (const float *)a.normed, (const W *)a.down, (const W *)a.inject, (float *)a.low,
        (float *)a.gate, a.branches, width, a.rank);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const dim3 up_grid(a.m, (width + kWarps - 1) / kWarps);
    hyper_up_k<W><<<up_grid, kWarps * 32, 0, s>>>((const float *)a.low, (const W *)a.up,
                                                   (float *)a.mix, width, a.rank);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const dim3 combine_grid(a.m, (a.hidden + 255) / 256);
    hyper_combine_k<<<combine_grid, 256, 0, s>>>((const float *)a.normed, (const float *)a.mix,
                                                 (float *)a.mixed, a.branches, a.hidden);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

}  // namespace

extern "C" int inference_cuda_hyper_read_launch(const inference_cuda_hyper_read_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_read(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        if (args->weight_dtype == kDtypeF16) {
            launch_read<__half>(*args, (cudaStream_t)stream, failure);
        } else {
            launch_read<__nv_bfloat16>(*args, (cudaStream_t)stream, failure);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_hyper_apply_launch(const inference_cuda_hyper_apply_args *args,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_apply(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int width = args->branches * args->hidden;
        const dim3 grid(args->m, (width + 255) / 256);
        Failure failure;
        hyper_apply_k<<<grid, 256, 0, (cudaStream_t)stream>>>(
            (float *)args->x, (const float *)args->gate, (const float *)args->block,
            args->branches, args->hidden);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
