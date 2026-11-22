// Independent cross-pass controller roles, one block per role.
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;

__global__ void inkling_controller_norm(float *workspace, float *health, int width, float eps) {
    const int role = blockIdx.x;
    const int roles = gridDim.x;
    const int lane = threadIdx.x;
    const float *input = workspace + role * 3 * width;
    const float *message = input + width;
    const float *gamma = message + width;
    const float *gain = workspace + (3 * roles + role * 2) * width;
    const float *forget = gain + width;
    float *out = workspace + (5 * roles + role) * width;

    __shared__ float sums[threads];
    float squares = 0.0f;
    bool invalid = false;
    for (int i = lane; i < width; i += threads) {
        const float kept = __fsub_rn(input[i], __fmul_rn(tanhf(forget[i]), input[i]));
        const float x = __fadd_rn(kept, __fmul_rn(tanhf(gain[i]), message[i]));
        invalid |= !isfinite(gain[i]) || !isfinite(forget[i]) || !isfinite(x);
        out[i] = x;
        squares += x * x;
    }
    sums[lane] = squares;
    __syncthreads();
    for (int stride = threads / 2; stride != 0; stride /= 2) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(sums[0] / width + eps);
    invalid |= !isfinite(sums[0]) || !isfinite(scale);
    for (int i = lane; i < width; i += threads) {
        out[i] = __fmul_rn(__fmul_rn(out[i], scale), gamma[i]);
        invalid |= !isfinite(out[i]);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[role] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}

// Same packed roles and health contract as the generic path. The exact-model
// first tile matched host f32 tanh with FP64 tanh narrowed to f32; this is not
// a guarantee of identical transcendental results on other inputs or hosts.
__global__ void inkling_controller_norm_cpu_parity(float *workspace, float *health, int width, float eps) {
    const int role = blockIdx.x;
    const int roles = gridDim.x;
    const int lane = threadIdx.x;
    const float *input = workspace + role * 3 * width;
    const float *message = input + width;
    const float *gamma = message + width;
    const float *gain = workspace + (3 * roles + role * 2) * width;
    const float *forget = gain + width;
    float *out = workspace + (5 * roles + role) * width;

    extern __shared__ float x[];
    __shared__ float scale;
    bool invalid = false;
    for (int i = lane; i < width; i += threads) {
        const float f = forget[i], g = gain[i], in = input[i];
        const float tf = __double2float_rn(tanh(static_cast<double>(f)));
        const float tg = __double2float_rn(tanh(static_cast<double>(g)));
        const float kept = __fsub_rn(in, __fmul_rn(tf, in));
        const float value = __fadd_rn(kept, __fmul_rn(tg, message[i]));
        // Infinities in the coefficients can be masked by tanh saturation.
        invalid |= !isfinite(f) || !isfinite(g) || !isfinite(value);
        x[i] = value;
    }
    __syncthreads();
    if (lane == 0) {
        // Rust dot(x, x): ordered f32 multiply, then f32 sum (whose seed is -0).
        float squares = -0.0f;
        for (int i = 0; i < width; ++i) {
            const float value = x[i];
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        const float mean = __fdiv_rn(squares, static_cast<float>(width));
        const float root = __fsqrt_rn(__fadd_rn(mean, eps));
        scale = __frcp_rn(root);
        // A square, sum, or epsilon addition can overflow even if the output is finite.
        invalid |= !isfinite(squares) || !isfinite(root) || !isfinite(scale);
    }
    __syncthreads();
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(__fmul_rn(x[i], scale), gamma[i]);
        out[i] = value;
        invalid |= !isfinite(value);
    }
    const int failed = __syncthreads_or(invalid);
    if (lane == 0) health[role] = failed ? __int_as_float(0x7fc00000) : 0.0f;
}
}  // namespace

extern "C" int inference_cuda_inkling_controller_norm(float *workspace, float *health,
                                                      int width, int roles, float eps, int cpu_parity,
                                                      void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (workspace == nullptr || health == nullptr || stream == nullptr || width < 1 ||
        width > 4096 || roles < 1 || roles > 1536 || !isfinite(eps) || eps <= 0.0f ||
        (cpu_parity != 0 && cpu_parity != 1))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const uintptr_t work = (uintptr_t)workspace, status = (uintptr_t)health;
    if (work % 4 != 0 || status % 4 != 0 ||
        (work <= status ? status - work < 6 * (size_t)roles * width * sizeof(float)
                        : work - status < (size_t)roles * sizeof(float)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (cpu_parity) {
        inkling_controller_norm_cpu_parity<<<roles, threads, width * sizeof(float), (cudaStream_t)stream>>>(
            workspace, health, width, eps);
    } else {
        inkling_controller_norm<<<roles, threads, 0, (cudaStream_t)stream>>>(workspace, health, width, eps);
    }
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
