#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "abi.h"
#include "device.h"

namespace {
constexpr int kWidth = 128, kTaps = 4;

struct Args {
    float *q, *relative, *raw_k, *raw_v, *k, *v, *sinks, *gates;
    float *aux;
    uint32_t *status;
    int queries, kv, dims, history;
    float qk_limit, rel_limit, eps;
};

// One block per head and plane. Use the host's reduction and tap order; these
// short reductions run once per head, after parallel channel preparation.
__global__ void inkling_local_prep(Args a) {
    __shared__ float squares[kWidth], scale;
    const int d = threadIdx.x;
    int head = blockIdx.x;
    const int q_words = a.queries * kWidth;
    const int rel_words = a.queries * a.dims;
    const int kv_words = a.kv * kWidth;
    const float *q_scale = a.aux + 1; // first word is the attention position
    const float *rel_scale = q_scale + q_words;
    const float *kernels = rel_scale + rel_words;
    float *output;
    float x = 0.f;
    bool valid = true;
    int width = kWidth;
    float limit = a.qk_limit;
    bool rms = false;
    if (head < a.queries) {
        const int at = head * kWidth + d;
        output = a.q + at;
        x = *output * q_scale[at];
        valid = isfinite(a.gates[at]) && (d != 0 || isfinite(a.sinks[head]));
    } else if ((head -= a.queries) < a.queries) {
        width = a.dims;
        limit = a.rel_limit;
        const int at = head * width + d;
        output = a.relative + head * width + min(d, width - 1);
        if (d < width) x = *output * rel_scale[at];
    } else {
        head -= a.queries;
        rms = head >= a.kv;
        const int plane = rms ? 1 : 0;
        const int at = (head - plane * a.kv) * kWidth + d;
        output = (rms ? a.v : a.k) + at;
        const float raw = (rms ? a.raw_v : a.raw_k)[at];
        const float *w = kernels + (plane * kv_words + at) * kTaps;
        float *past = a.aux + 1 + q_words + rel_words + (2 * kTaps + plane * (kTaps - 1)) * kv_words;
        float conv = __fmul_rn(w[kTaps - 1], raw);
        for (int lag = 1; lag <= a.history; ++lag) {
            conv = __fadd_rn(conv, __fmul_rn(w[kTaps - 1 - lag], past[(lag - 1) * kv_words + at]));
        }
        // Each thread owns one channel of the mirror. Read every old tap
        // before shifting the history for that channel.
        for (int lag = min(a.history, kTaps - 2); lag > 0; --lag) {
            past[lag * kv_words + at] = past[(lag - 1) * kv_words + at];
        }
        past[at] = raw;
        x = __fmul_rn(__fadd_rn(conv, raw), 1.f / sqrtf(float(kTaps + 1)));
    }
    squares[d] = x * x;
    __syncthreads();
    if (d == 0) {
        float sum = 0.f;
        for (int i = 0; i < width; ++i) sum += squares[i];
        scale = rms ? 1.f / sqrtf(sum / width + a.eps)
                    : fminf(1.f, limit / (sqrtf(sum) + 1.e-6f));
        valid &= isfinite(sum);
    }
    __syncthreads();
    const float result = x * scale;
    const int invalid = __syncthreads_or(!valid || !isfinite(x) || !isfinite(result));
    if (d < width) *output = result;
    // Only one block writes status; per-head prep flags are distinct words.
    if (d == 0) {
        if (blockIdx.x == 0) *a.status = 0;
        a.status[1 + blockIdx.x] = invalid;
    }
}
// Static output convolution. Raw projected rows are immutable and become the
// next tile's history. Initial history is stored newest first.
__global__ void inkling_output_conv(const float *raw, const float *past, const float *weight,
                                    float *result, uint32_t *status, int rows, int width,
                                    int history, float scale) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    bool bad = false;
    if (channel < width) {
        const int at = row * width + channel;
        const float *w = weight + channel * kTaps;
        float sum = 0.f;
        for (int lag = 0; lag <= min(kTaps - 1, row + history); ++lag) {
            const int source = row - lag;
            const float value = source >= 0 ? raw[source * width + channel]
                                            : past[(-source - 1) * width + channel];
            sum = __fadd_rn(sum, __fmul_rn(w[kTaps - 1 - lag], value));
        }
        const float value = __fmul_rn(__fadd_rn(sum, raw[at]), scale);
        result[at] = value;
        bad = !isfinite(value);
    }
    const int invalid = __syncthreads_or(bad);
    if (threadIdx.x == 0 && invalid) atomicOr(status + row, 1u);
}
} // namespace

extern "C" int inference_cuda_inkling_output_conv(void *const *p, int rows, int width,
    int history, float scale, void *stream, int *out_stage) {
    constexpr int block = 128;
    inkling_output_conv<<<dim3((width + block - 1) / block, rows), block, 0, (cudaStream_t)stream>>>(
        static_cast<const float *>(p[0]), static_cast<const float *>(p[1]),
        static_cast<const float *>(p[2]), static_cast<float *>(p[3]),
        static_cast<uint32_t *>(p[4]), rows, width, history, scale);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

extern "C" int inference_cuda_inkling_local_prep(void *const *p, int queries, int kv,
    int dims, int history, float qk_limit, float rel_limit, float eps, void *stream,
    int *out_stage) {
    Args args{static_cast<float *>(p[0]), static_cast<float *>(p[1]),
              static_cast<float *>(p[2]), static_cast<float *>(p[3]),
              static_cast<float *>(p[4]), static_cast<float *>(p[5]),
              static_cast<float *>(p[6]), static_cast<float *>(p[7]),
              static_cast<float *>(p[8]), static_cast<uint32_t *>(p[9]),
              queries, kv, dims, history, qk_limit, rel_limit, eps};
    inkling_local_prep<<<2 * (queries + kv), kWidth, 0, (cudaStream_t)stream>>>(args);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}
