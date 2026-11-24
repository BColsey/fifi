#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "abi.h"
#include "device.h"
#include "resident/attn_split_reduce.cuh"
#include "resident/draft_attention_bf16.h"

namespace {

using Args = inference_cuda_draft_attention_bf16_args;
constexpr int kHeadDim = 128;
constexpr int kMaxHeads = 64;
constexpr int kMaxRows = 65535;
constexpr int kMaxCapacity = (1 << 30) - 1;
constexpr int kWarp = 32;
constexpr int kWarps = 4;
constexpr int kColumns = kHeadDim / kWarp;
constexpr float kScale = 0.08838834764831844f;  // 1 / sqrt(128)

__device__ float read_bf16(uint16_t bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

__device__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

struct Partial {
    float peak = -INFINITY;
    float sum = 0.f;
    float values[kColumns] = {};

    __device__ void add(const float (&query)[kColumns],
                        const uint16_t *key, const uint16_t *value) {
        float dot = 0.f;
#pragma unroll
        for (int d = 0; d < kColumns; ++d) {
            dot = fmaf(query[d], read_bf16(key[d * kWarp]), dot);
        }
        const float score = kScale * warp_sum(dot);
        const float next = fmaxf(peak, score);
        const float correction = expf(peak - next);
        const float mass = expf(score - next);
        sum = fmaf(sum, correction, mass);
#pragma unroll
        for (int d = 0; d < kColumns; ++d) {
            values[d] = fmaf(mass, read_bf16(value[d * kWarp]), values[d] * correction);
        }
        peak = next;
    }
};

// One block serves one query row/head. Each warp streams a stripe of the
// visible keys through online softmax, then the block merges the partials.
__global__ void draft_attention(Args a) {
    __shared__ float stats[kWarps * 2], values[kWarps * kHeadDim];
    __shared__ float sums[kWarps], masses[kWarps], maxima[kWarps];
    const int d = threadIdx.x, lane = d % kWarp, warp = d / kWarp;
    const int kv_head = int(blockIdx.x) / (a.query_heads / a.kv_heads);
    const size_t q_base = (size_t(blockIdx.y) * a.query_heads + blockIdx.x) * kHeadDim;
    float query[kColumns];
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        query[i] = read_bf16(a.query[q_base + lane + i * kWarp]);
    }
    Partial p;
    const int count = a.len + a.rows;
    for (int row = warp; row < count; row += kWarps) {
        const bool committed = row < a.len;
        int slot = committed ? a.first + row : row - a.len;
        if (committed && slot >= a.capacity) slot -= a.capacity;
        const size_t offset = (size_t(slot) * a.kv_heads + kv_head) * kHeadDim + lane;
        const uint16_t *keys = committed ? a.committed_keys : a.draft_keys;
        const uint16_t *vals = committed ? a.committed_values : a.draft_values;
        p.add(query, keys + offset, vals + offset);
    }
    if (lane == 0) {
        stats[warp * 2] = p.peak;
        stats[warp * 2 + 1] = p.sum;
    }
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        values[warp * kHeadDim + lane + i * kWarp] = p.values[i];
    }
    __syncthreads();
    const int active = min(kWarps, count);
    const float peak = inference::attn_split_max<kHeadDim>(d, active, stats, 2, sums, maxima);
    if (d < active) masses[d] = expf(stats[d * 2] - peak);
    __syncthreads();
    float sum, value;
    inference::attn_split_sum(active, 0.f, masses, sums, values + d, kHeadDim, sum, value);
    a.out[q_base + d] = __bfloat16_as_ushort(__float2bfloat16_rn(value / sum));
}

bool valid(const Args &a) {
    if (a.kv_heads < 1 || a.query_heads < a.kv_heads || a.query_heads > kMaxHeads ||
        a.query_heads % a.kv_heads != 0 || a.rows < 1 || a.rows > kMaxRows ||
        a.capacity < 1 || a.capacity > kMaxCapacity || a.first < 0 || a.first >= a.capacity ||
        a.len < 0 || a.len > a.capacity) {
        return false;
    }
    const void *ranges[]{a.out, a.query, a.draft_keys, a.draft_values,
                         a.committed_keys, a.committed_values};
    for (const void *ptr : ranges) {
        if (ptr == nullptr || uintptr_t(ptr) % alignof(uint16_t) != 0) return false;
    }
    return true;
}

}  // namespace

extern "C" int inference_cuda_draft_attention_bf16_launch(
    const Args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == nullptr || stream == nullptr || !valid(*in)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const dim3 grid(in->query_heads, in->rows);
        draft_attention<<<grid, kHeadDim, 0, (cudaStream_t)stream>>>(*in);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
