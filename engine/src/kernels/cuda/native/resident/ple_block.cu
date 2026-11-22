/*
 * The per-layer-embedding block of a decode step on an explicit stream. Glue and kernels in one
 * unit; every exported function has C linkage, takes and returns C types only, and wraps its
 * body in try/catch, like the other resident entries. The arithmetic mirrors the host reference
 * so the host oracle checks each kernel.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/ple_block.h"

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kIq4Block = 32;
constexpr int kIq4Bytes = 18;

// The 16 dequantization values of IQ4_NL (the GGUF kvalues_iq4nl table). They reach the kernel
// by value, never as a __constant__ table.
struct Iq4Table {
    float v[16];
};

Iq4Table make_iq4_table() {
    Iq4Table t = {{-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f, 1.0f,
                   13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f}};
    return t;
}

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

__device__ float block_sum(float v, float *shared) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < kWarps; ++w) total += shared[w];
        shared[0] = total;
    }
    __syncthreads();
    total = shared[0];
    __syncthreads();
    return total;
}

__global__ void ple_rows_kernel(float *out, const unsigned char *raw, float scale, int head_dim,
                                Iq4Table table) {
    const int row = blockIdx.x;
    const int blocks = head_dim / kIq4Block;
    const unsigned char *src = raw + (size_t)row * blocks * kIq4Bytes;
    float *dst = out + (size_t)row * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const int b = i / kIq4Block, j = i % kIq4Block;
        const unsigned char *blk = src + (size_t)b * kIq4Bytes;
        const __half d = *reinterpret_cast<const __half *>(blk);
        const unsigned char q = blk[2 + (j & 15)];
        const int nibble = j < 16 ? (q & 0xf) : (q >> 4);
        dst[i] = __half2float(d) * table.v[nibble] * scale;
    }
}

__global__ void ple_gate_kernel(inference_cuda_ple_gate_args a) {
    __shared__ float shared[kWarps];
    const int b = blockIdx.x;
    const size_t base = (size_t)b * a.hidden;
    float sk = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < a.hidden; i += blockDim.x) {
        const float k = a.keys[base + i], q = a.hyper[base + i];
        sk += k * k;
        sq += q * q;
    }
    sk = block_sum(sk, shared);
    sq = block_sum(sq, shared);
    const float dk = sqrtf(sk / (float)a.hidden + a.eps);
    const float dq = sqrtf(sq / (float)a.hidden + a.eps);
    float dotp = 0.0f;
    for (int i = threadIdx.x; i < a.hidden; i += blockDim.x) {
        const float kn = a.keys[base + i] * (1.0f + a.norm_key[base + i]) / dk;
        const float qn = a.hyper[base + i] * (1.0f + a.norm_query[base + i]) / dq;
        dotp += kn * qn;
    }
    dotp = block_sum(dotp, shared);
    const double d = (double)(dotp / sqrtf((float)a.hidden));
    const double magnitude = sqrt(fmax(fabs(d), 1e-6));
    const double shaped = d < 0.0 ? -magnitude : magnitude;
    const float g = (float)(1.0 / (1.0 + exp(-shaped)));
    float sg = 0.0f;
    for (int i = threadIdx.x; i < a.hidden; i += blockDim.x) {
        const float v = g * a.value[i];
        a.gated[base + i] = v;
        sg += v * v;
    }
    sg = block_sum(sg, shared);
    const float dg = sqrtf(sg / (float)a.hidden + a.eps);
    for (int i = threadIdx.x; i < a.hidden; i += blockDim.x) {
        a.normed[base + i] = a.gated[base + i] * (1.0f + a.norm_conv[base + i]) / dg;
    }
}

__global__ void ple_conv_kernel(inference_cuda_ple_conv_args a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.width) return;
    const int taps = a.conv_kernel - 1;
    const int stride = taps * a.ngram;
    const float *kernel = a.conv + (size_t)c * a.conv_kernel;
    float *ring = a.ring + (size_t)c * stride;
    const float n = a.normed[c];
    float acc = kernel[taps] * n;
    for (int k = 0; k < taps; ++k) acc += kernel[k] * ring[k * a.ngram];
    a.out[c] = a.gated[c] + acc / (1.0f + expf(-acc));
    for (int i = 0; i + 1 < stride; ++i) ring[i] = ring[i + 1];
    ring[stride - 1] = n;
}

}  // namespace

extern "C" int inference_cuda_ple_rows_launch(const inference_cuda_ple_rows_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->out == NULL || args->raw == NULL ||
            !isfinite(args->scale) || args->rows < 1 || args->rows > 65536 ||
            args->head_dim < kIq4Block || args->head_dim > 8192 ||
            args->head_dim % kIq4Block != 0 || !aligned(args->out, 4) ||
            !aligned(args->raw, 2)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        ple_rows_kernel<<<args->rows, kThreads, 0, (cudaStream_t)stream>>>(
            args->out, (const unsigned char *)args->raw, args->scale, args->head_dim,
            make_iq4_table());
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ple_gate_launch(const inference_cuda_ple_gate_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->keys == NULL || args->hyper == NULL ||
            args->value == NULL || args->norm_key == NULL || args->norm_query == NULL ||
            args->norm_conv == NULL || args->gated == NULL || args->normed == NULL ||
            !isfinite(args->eps) || args->eps < 0.0f || args->branches < 1 ||
            args->branches > 64 || args->hidden < 1 || args->hidden > (1 << 20) ||
            !aligned(args->keys, 4) || !aligned(args->hyper, 4) || !aligned(args->value, 4) ||
            !aligned(args->norm_key, 4) || !aligned(args->norm_query, 4) ||
            !aligned(args->norm_conv, 4) || !aligned(args->gated, 4) ||
            !aligned(args->normed, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        ple_gate_kernel<<<args->branches, kThreads, 0, (cudaStream_t)stream>>>(*args);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_ple_conv_launch(const inference_cuda_ple_conv_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->gated == NULL || args->normed == NULL ||
            args->conv == NULL || args->ring == NULL || args->out == NULL || args->width < 1 ||
            args->width > (1 << 22) || args->conv_kernel < 2 || args->conv_kernel > 16 ||
            args->ngram < 1 || args->ngram > 16 || !aligned(args->gated, 4) ||
            !aligned(args->normed, 4) || !aligned(args->conv, 4) || !aligned(args->ring, 4) ||
            !aligned(args->out, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const unsigned grid = (unsigned)((args->width + kThreads - 1) / kThreads);
        Failure failure;
        ple_conv_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(*args);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
