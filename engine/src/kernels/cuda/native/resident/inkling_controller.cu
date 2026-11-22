// Three independent cross-pass controller roles, one block per role.
#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;

__global__ void inkling_controller_norm(float *workspace, float *health, int width, float eps) {
    const int role = blockIdx.x;
    const int lane = threadIdx.x;
    const float *input = workspace + role * 3 * width;
    const float *message = input + width;
    const float *gamma = message + width;
    const float *gain = workspace + (9 + role * 2) * width;
    const float *forget = gain + width;
    float *out = workspace + (15 + role) * width;

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
}  // namespace

extern "C" int inference_cuda_inkling_controller_norm(float *workspace, float *health,
                                                      int width, float eps, void *stream,
                                                      int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (workspace == nullptr || health == nullptr || stream == nullptr || width < 1 ||
        width > 4096 || !isfinite(eps) || eps <= 0.0f)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    const uintptr_t work = (uintptr_t)workspace, status = (uintptr_t)health;
    if (work % 4 != 0 || status % 4 != 0 ||
        (work <= status ? status - work < 18 * (size_t)width * sizeof(float)
                        : work - status < 3 * sizeof(float)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    inkling_controller_norm<<<3, threads, 0, (cudaStream_t)stream>>>(workspace, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
