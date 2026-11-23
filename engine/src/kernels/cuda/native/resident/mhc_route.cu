/*
 * The mHC stream routing kernels of the SenseNova Loop decode step.
 *
 * Three launch-only kernels over the two mHC residual streams; the math is the CPU reference's
 * (family/qwen/sensenova/step.rs: mhc_route and merge), element for element, in the arithmetic
 * the native TU is compiled with (warnings into errors, WITHOUT -use_fast_math), so expf and
 * log1pf are the IEEE-closest forms, as the Rust f32 reference's .exp() and .ln_1p() are.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "resident/mhc_route.h"
#include "device.h"

namespace {

constexpr int kUnit = 256;   // threads per block; hidden is a multiple of it
constexpr int kMaxH = 1 << 20;

__device__ inline float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

// The Rust softplus: x.max(0) + ln_1p(exp(-|x|)).
__device__ inline float softplus_(float x) { return fmaxf(x, 0.f) + log1pf(expf(-fabsf(x))); }

__global__ void __launch_bounds__(kUnit) mhc_route_k(
    const float *__restrict__ logits, const float *__restrict__ bias, const float *__restrict__ a,
    const float *__restrict__ b, float *__restrict__ input, float *__restrict__ route, int hidden,
    float pre_scale, int iterations, int fixed_identity) {
    const int j = blockIdx.x * kUnit + threadIdx.x;
    float l[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        l[k] = logits[k] + (bias == NULL ? 0.f : bias[k]);
    }
    if (j < hidden) {
        input[j] = (a[j] * sigmoidf_(l[0]) + b[j] * sigmoidf_(l[1])) * pre_scale;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        route[0] = 2.f * sigmoidf_(l[2]);
        route[1] = 2.f * sigmoidf_(l[3]);
        if (fixed_identity) {
            route[2] = 1.f; route[3] = 0.f; route[4] = 0.f; route[5] = 1.f;
            return;
        }
        float m[4] = {l[4], l[5], l[6], l[7]};
        for (int it = 0; it < iterations; ++it) {
            const float d01 = m[0] - m[1];
            m[0] = -softplus_(-d01);
            m[1] = -softplus_(d01);
            const float d23 = m[2] - m[3];
            m[2] = -softplus_(-d23);
            m[3] = -softplus_(d23);
            const float d02 = m[0] - m[2];
            m[0] = -softplus_(-d02);
            m[2] = -softplus_(d02);
            const float d13 = m[1] - m[3];
            m[1] = -softplus_(-d13);
            m[3] = -softplus_(d13);
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            route[2 + k] = expf(m[k]);
        }
    }
}

__global__ void __launch_bounds__(kUnit) mhc_merge_k(
    float *__restrict__ a, float *__restrict__ b, const float *__restrict__ branch,
    const float *__restrict__ route, int hidden) {
    const int j = blockIdx.x * kUnit + threadIdx.x;
    if (j < hidden) {
        const float m00 = route[2], m01 = route[3], m10 = route[4], m11 = route[5];
        const float post0 = route[0], post1 = route[1];
        const float aj = a[j], bj = b[j], y = branch[j];
        a[j] = aj * m00 + bj * m01 + y * post0;
        b[j] = aj * m10 + bj * m11 + y * post1;
    }
}

__global__ void __launch_bounds__(kUnit) mhc_average_k(
    const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ out,
    int hidden) {
    const int j = blockIdx.x * kUnit + threadIdx.x;
    if (j < hidden) {
        out[j] = 0.5f * (a[j] + b[j]);
    }
}

bool hidden_ok(int h) { return h >= kUnit && h % kUnit == 0 && h <= kMaxH; }

}  // namespace

extern "C" int inference_cuda_mhc_route_launch(const inference_cuda_mhc_route_args *in,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == NULL || in->logits == NULL || in->a == NULL || in->b == NULL ||
            in->input == NULL || in->route == NULL || !hidden_ok(in->hidden) ||
            !isfinite(in->pre_scale) || in->iterations < 0 || in->iterations > 16) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const dim3 grid((in->hidden + kUnit - 1) / kUnit);
        mhc_route_k<<<grid, kUnit, 0, (cudaStream_t)stream>>>(
            in->logits, in->bias, in->a, in->b, in->input, in->route, in->hidden, in->pre_scale,
            in->iterations, in->fixed_identity);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mhc_merge_launch(const inference_cuda_mhc_merge_args *in,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == NULL || in->a == NULL || in->b == NULL || in->branch == NULL ||
            in->route == NULL || !hidden_ok(in->hidden)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const dim3 grid((in->hidden + kUnit - 1) / kUnit);
        mhc_merge_k<<<grid, kUnit, 0, (cudaStream_t)stream>>>(in->a, in->b, in->branch, in->route,
                                                              in->hidden);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_mhc_average_launch(const inference_cuda_mhc_average_args *in,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == NULL || in->a == NULL || in->b == NULL || in->out == NULL ||
            !hidden_ok(in->hidden)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const dim3 grid((in->hidden + kUnit - 1) / kUnit);
        mhc_average_k<<<grid, kUnit, 0, (cudaStream_t)stream>>>(in->a, in->b, in->out,
                                                                in->hidden);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
