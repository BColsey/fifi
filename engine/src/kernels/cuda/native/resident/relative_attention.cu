#include <cuda_runtime.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/attn_split_reduce.cuh"
#include "resident/relative_attention.h"
#include "resident/relative_attention_tile.h"
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

struct KvPair {
    const float *key, *value;
};

struct SerialKv {
    __device__ KvPair operator()(const Args &a, const Rows &r, int distance) const {
        const int row = r.slot - distance;
        const size_t slot = row < 0 ? row + a.capacity : row;
        const size_t src = (slot * a.kv_heads + r.kv_head) * kHeadDim;
        return {distance == 0 ? a.k + r.kv_head * kHeadDim : a.keys + src,
                distance == 0 ? a.v + r.kv_head * kHeadDim : a.values + src};
    }
};

// The score of the row at `distance`: the scaled QK dot of the lane's key columns, summed across
// the warp, plus the learned bias.
__device__ __forceinline__ float row_score(const Args &a, const float (&query)[kColumns], float rel,
                                           int distance, const float (&key)[kColumns]) {
    const int lane = threadIdx.x % kWarp;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        dot = fmaf(query[i], key[i], dot);
    }
    float bias = 0.f;
    if (distance < a.relative_extent) {
        const float term = lane < a.relative_dims
                               ? rel * a.table[size_t(lane) * a.relative_extent + distance]
                               : 0.f;
        bias = warp_sum(term);
    }
    return kScale * warp_sum(dot) + bias;
}

// The lane's columns of one K or V row.
__device__ __forceinline__ void load_columns(const float *row, float (&out)[kColumns]) {
    const int lane = threadIdx.x % kWarp;
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        out[i] = row[lane + i * kWarp];
    }
}

// Rows a warp loads and scores before it folds them. The loads and the scores of a batch do not
// depend on the running partial, so they are issued together; the folds then run in distance
// order, as one row at a time would, so the results keep their bits.
constexpr int kBatch = 4;

// One warp computes the weighted sum over its interleaved subset of newest-first rows: warp w
// adds the distances w, w + kWarps, ... Only the source addresses vary by reader; the FP32
// recurrence and order do not.
template <class ReadKv>
__device__ void partial(const Args &a, const Rows &r, Partials &p, ReadKv read_kv) {
    const int lane = threadIdx.x % kWarp, warp = threadIdx.x / kWarp;
    float query[kColumns];
    float acc[kColumns] = {};
#pragma unroll
    for (int i = 0; i < kColumns; ++i) {
        query[i] = a.q[r.head * kHeadDim + lane + i * kWarp];
    }
    const float rel = lane < a.relative_dims ? a.relative[r.head * a.relative_dims + lane] : 0.f;
    float peak = -INFINITY, sum = 0.f;
    for (int first = warp; first < r.count; first += kWarps * kBatch) {
        float keys[kBatch][kColumns], values[kBatch][kColumns];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int distance = first + j * kWarps;
            if (distance < r.count) {
                const auto [key, value] = read_kv(a, r, distance);
                load_columns(key, keys[j]);
                load_columns(value, values[j]);
            }
        }
        float scores[kBatch];
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const int distance = first + j * kWarps;
            scores[j] =
                distance < r.count ? row_score(a, query, rel, distance, keys[j]) : -INFINITY;
        }
#pragma unroll
        for (int j = 0; j < kBatch; ++j) {
            const float score = scores[j];
            if (score == -INFINITY) continue;
            const float next = fmaxf(peak, score);
            const float correction = expf(peak - next);
            const float mass = expf(score - next);
            sum = fmaf(sum, correction, mass);
#pragma unroll
            for (int i = 0; i < kColumns; ++i) {
                acc[i] = fmaf(mass, values[j][i], acc[i] * correction);
            }
            peak = next;
        }
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
    partial(a, r, p, SerialKv{});
    __syncthreads();
    finish(a, r, p);
}

__device__ int row_stride(int stride, int width) { return stride == 0 ? width : stride; }

struct SnapshotKv {
    int query, stride;
    __device__ KvPair operator()(const Args &a, const Rows &r, int distance) const {
        const auto src = snapshot::source(query, r.slot, distance, a.capacity);
        const int channels = a.kv_heads * kHeadDim;
        const size_t offset = size_t(src.row) * (src.staged ? stride : channels)
                            + r.kv_head * kHeadDim;
        return {(src.staged ? a.k : a.keys) + offset,
                (src.staged ? a.v : a.values) + offset};
    }
};

__global__ void snapshot_queries(Args a, int first, int input_stride, int query_stride,
                                 int side_stride) {
    __shared__ Partials p;
    const int query = blockIdx.y, head = blockIdx.x, pos = first + query;
    a.status += query;
    const int channels = a.query_heads * kHeadDim;
    a.out += query * channels;
    if (a.pos[query * row_stride(input_stride, 1)] != pos) {
        if (head == 0 && threadIdx.x == 0) atomicOr(a.status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        // Output projection can already be queued. Give it initialized inputs even
        // though the host will discard the whole tile based on status.
        if (threadIdx.x < kHeadDim) a.out[head * kHeadDim + threadIdx.x] = 0.f;
        return;
    }
    const int group = a.query_heads / a.kv_heads;
    const Rows r{head, head / group, pos % a.capacity, min(pos + 1, a.capacity)};
    // K/V stay at the tile base; the reader chooses a staged row or the pre-tile ring.
    a.q += query * row_stride(query_stride, channels);
    a.relative += query * row_stride(query_stride, a.query_heads * a.relative_dims);
    a.sinks += query * row_stride(side_stride, a.query_heads);
    a.gates += query * row_stride(side_stride, channels);
    partial(a, r, p, SnapshotKv{query, row_stride(input_stride, a.kv_heads * kHeadDim)});
    __syncthreads();
    finish(a, r, p);
}

__global__ void snapshot_commit(Args a, int first, int count, int input_stride) {
    __shared__ int valid;
    if (threadIdx.x == 0) {
        valid = 1;
        for (int i = 0; i < count; ++i) valid &= (a.status[i] == 0);
    }
    __syncthreads();
    if (!valid) return;
    const int row = snapshot::first_commit(count, a.capacity) + blockIdx.x;
    const int slot = (first + row) % a.capacity;
    const int channels = a.kv_heads * kHeadDim;
    for (int d = threadIdx.x; d < channels; d += blockDim.x) {
        const size_t src = size_t(row) * row_stride(input_stride, channels) + d;
        const size_t dst = size_t(slot) * channels + d;
        a.keys[dst] = a.k[src];
        a.values[dst] = a.v[src];
    }
}

// The host uses an ordered FP32 sum, not a parallel reduction. Parallelize
// across heads and channels without changing the accumulation or rounding.
// Side rows hold Q, `gap` other words, relative coefficients, sinks and gates.
__global__ void clip_prefill_queries(float *side, const float *scales, uint32_t *status,
                                     int heads, int dims, int gap, float q_limit,
                                     float rel_limit) {
    const int row = blockIdx.y, d = threadIdx.x;
    const bool relative = int(blockIdx.x) >= heads;
    const int head = blockIdx.x % heads;
    const int width = relative ? dims : kHeadDim;
    const int query_words = heads * kHeadDim;
    const int stride = 2 * query_words + gap + heads * (dims + 1);
    const int at = (relative ? query_words + gap : 0) + head * width + d;
    const int scale_at = (relative ? query_words : 0) + head * width + d;
    __shared__ float squares[kHeadDim], factor;
    float raw = 0.f, x = 0.f;
    if (d < width) {
        raw = side[row * stride + at];
        x = __fmul_rn(raw, scales[scale_at]);
        squares[d] = __fmul_rn(x, x);
    }
    __syncthreads();
    if (d == 0) {
        float sum = 0.f;
        for (int i = 0; i < width; ++i) sum = __fadd_rn(sum, squares[i]);
        factor = fminf(1.f, __fdiv_rn(relative ? rel_limit : q_limit,
                                      __fadd_rn(__fsqrt_rn(sum), 1.e-6f)));
    }
    __syncthreads();
    const float result = __fmul_rn(x, factor);
    // Test the raw projection too: clipping and sigmoid must not mask errors
    // that used to be caught during the projection readback.
    const int invalid = __syncthreads_or(!isfinite(raw) || !isfinite(x) || !isfinite(result));
    if (d == 0 && invalid) atomicOr(status + row, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
    if (d < width) side[row * stride + at] = result;
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

extern "C" int inference_cuda_relative_attention_clip_queries(
    float *side, const float *scales, uint32_t *status, int rows, int heads,
    int dims, int gap, float q_limit, float rel_limit, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (side == nullptr || scales == nullptr || status == nullptr || stream == nullptr ||
        (uintptr_t(side) | uintptr_t(scales) | uintptr_t(status)) % 4 != 0 ||
        rows < 1 || rows > 512 || heads < 1 || heads > kMaxHeads || dims < 1 || dims > kWarp ||
        gap < 0 ||
        !isfinite(q_limit) || q_limit <= 0.f || !isfinite(rel_limit) || rel_limit <= 0.f) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    clip_prefill_queries<<<dim3(2 * heads, rows), kHeadDim, 0, (cudaStream_t)stream>>>(
        side, scales, status, heads, dims, gap, q_limit, rel_limit);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

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

extern "C" int inference_cuda_relative_attention_tile_launch(
    const Args *in, int first, int count, int input_stride, int query_stride,
    int side_stride, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == nullptr || stream == nullptr || !valid(*in) ||
            count < 2 || count > snapshot::kLimit || !snapshot::valid_interval(first, count)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int channels = in->query_heads * kHeadDim;
        auto valid_stride = [](int stride, int width) {
            return stride == 0 || (stride >= width && stride <= snapshot::kMaxStride);
        };
        if (!valid_stride(input_stride, in->kv_heads * kHeadDim) ||
            !valid_stride(query_stride, channels) || !valid_stride(side_stride, channels)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        const auto s = (cudaStream_t)stream;
        snapshot_queries<<<dim3(in->query_heads, count), kThreads, 0, s>>>(
            *in, first, input_stride, query_stride, side_stride);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            // The stream, not CTA ordering, separates all snapshot reads from writes.
            snapshot_commit<<<min(count, in->capacity), kThreads, 0, s>>>(
                *in, first, count, input_stride);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
