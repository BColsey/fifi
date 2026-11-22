#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/attn_split_reduce.cuh"
#include "resident/relative_attention.h"
#include "resident/step_params.h"

namespace {

constexpr int kHeadDim = 128;
constexpr int kMaxHeads = 64;
constexpr int kWordMax = (1 << 30) - 1;
constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * kWarp;
constexpr int kColumns = kHeadDim / kWarp;
constexpr float kScale = 0.08838834764831844f;  // 1 / sqrt(128)

using Args = inference_cuda_relative_attention_args;

struct Rows {
    int head, kv_head, slot, count;
};

struct Partials {
    float stats[kWarps * 2];
    float values[kWarps * kHeadDim];
};

__device__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One warp computes the weighted sum over its interleaved subset of newest-first rows.
__device__ void partial(const Args &a, const Rows &r, Partials &p) {
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    float query[kColumns];
    float acc[kColumns] = {};
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        query[i] = a.q[r.head * kHeadDim + lane + i * kWarp];
    }
    const float rel = lane < a.relative_dims ? a.relative[r.head * a.relative_dims + lane] : 0.f;
    float peak = -INFINITY, sum = 0.f;
    for (int distance = warp; distance < r.count; distance += kWarps) {
        const int row = r.slot - distance;
        const size_t slot = row < 0 ? row + a.capacity : row;
        const size_t src = (slot * a.kv_heads + r.kv_head) * kHeadDim;
        const float *key = distance == 0 ? a.k + r.kv_head * kHeadDim : a.keys + src;
        const float *value = distance == 0 ? a.v + r.kv_head * kHeadDim : a.values + src;
        float dot = 0.f;
#pragma unroll
        for (int i = 0; i < kColumns; ++i) {
            dot = fmaf(query[i], key[lane + i * kWarp], dot);
        }
        float bias = 0.f;
        if (distance < a.relative_extent) {
            const float term = lane < a.relative_dims
                                   ? rel * a.table[size_t(lane) * a.relative_extent + distance]
                                   : 0.f;
            bias = warp_sum(term);
        }
        const float score = kScale * warp_sum(dot) + bias;
        if (score == -INFINITY) continue;
        const float next = fmaxf(peak, score);
        const float correction = expf(peak - next);
        const float mass = expf(score - next);
        sum = fmaf(sum, correction, mass);
#pragma unroll
        for (int i = 0; i < kColumns; ++i) {
            acc[i] = fmaf(mass, value[lane + i * kWarp], acc[i] * correction);
        }
        peak = next;
    }
    if (lane == 0) {
        p.stats[warp * 2] = peak;
        p.stats[warp * 2 + 1] = sum;
    }
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        p.values[warp * kHeadDim + lane + i * kWarp] = acc[i];
    }
}

__device__ void finish(const Args &a, const Rows &r, const Partials &p) {
    __shared__ float sums[kWarps], masses[kWarps], maxima[kWarps];
    const int d = threadIdx.x;
    const int active = min(kWarps, r.count);
    const float sink = a.sinks[r.head];
    const float peak = fmaxf(sink, inference::attn_split_max<kThreads>(
                                      d, active, p.stats, 2, sums, maxima));
    if (d < active) masses[d] = expf(p.stats[d * 2] - peak);
    __syncthreads();
    if (d < kHeadDim) {
        float sum, value;
        inference::attn_split_sum(active, expf(sink - peak), masses, sums, p.values + d,
                                  kHeadDim, sum, value);
        const int out = r.head * kHeadDim + d;
        const float gate_logit = a.gates[out];
        const float gate = 1.f / (1.f + expf(-gate_logit));
        const float result = (value / sum) * gate;
        // A resident gate can bypass host readback; sigmoid alone masks infinite logits.
        if (!isfinite(result) || !isfinite(gate_logit) || (d == 0 && !isfinite(sink))) {
            atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        }
        a.out[out] = result;
    }
}

__global__ void relative_attention(Args a) {
    __shared__ Partials p;
    const int pos = *a.pos, head = blockIdx.x, d = threadIdx.x;
    if (pos < 0 || pos >= kWordMax) {
        if (head == 0 && d == 0) atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        return;
    }
    const int group = a.query_heads / a.kv_heads;
    const Rows r{head, head / group, pos % a.capacity, min(pos + 1, a.capacity)};
    // Only the first query head in each group writes the current cache row. All
    // blocks read the current K/V directly and read the cache only for older rows.
    if (head % group == 0 && d < kHeadDim) {
        const int src = r.kv_head * kHeadDim + d;
        const size_t dst = size_t(r.slot) * a.kv_heads * kHeadDim + src;
        a.keys[dst] = a.k[src];
        a.values[dst] = a.v[src];
    }
    partial(a, r, p);
    __syncthreads();
    finish(a, r, p);
}

bool valid(const inference_cuda_relative_attention_args &a) {
    if (a.kv_heads < 1 || a.query_heads < a.kv_heads || a.query_heads > kMaxHeads ||
        a.query_heads % a.kv_heads != 0 || a.relative_dims < 1 || a.relative_dims > kWarp ||
        a.relative_extent < 1 || a.relative_extent > kWordMax ||
        a.capacity < 1 || a.capacity > kWordMax) {
        return false;
    }
    const void *ranges[] = {a.out, a.keys, a.values, a.status, a.q, a.k,
                            a.v, a.relative, a.table, a.sinks, a.gates, a.pos};
    for (const void *p : ranges) {
        if (p == nullptr || uintptr_t(p) % 4 != 0) return false;
    }
    return true;
}

}  // namespace

extern "C" int inference_cuda_relative_attention_launch(
    const inference_cuda_relative_attention_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == nullptr || stream == nullptr || !valid(*in)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        relative_attention<<<in->query_heads, kThreads, 0, (cudaStream_t)stream>>>(*in);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
