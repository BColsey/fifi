// The sparse decode attention of Qwen3.8 Flash-Next QSA (see resident/qsa_attn.h). The split
// kernel runs one block per (kv head, split) with one warp per query head of the group: the
// split's attended positions go through shared memory 32 at a time, fp16 keys and values in
// 16-byte loads, and each warp keeps an online softmax (max, sum, 8 partial values per lane).
// The combine kernel merges the splits per head and applies the sigmoid output gate.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/qsa_attn.h"

namespace {

constexpr int kHead = 256;
constexpr int kLaneDims = kHead / 32;
constexpr int kTile = 32;
constexpr int kMaxGroup = 32;
constexpr int kMaxKvHeads = 64;
constexpr int kMaxRatio = 64;
constexpr int kMaxBudget = 1 << 24;
constexpr int kMaxSplits = 128;
constexpr int kMaxRows = (1 << 30) - 1;
constexpr int kPartStride = kHead + 2;
constexpr int kStatusBits = 4;  // INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of step_params.h

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    if (a.p == NULL || b.p == NULL) return true;
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

bool shape_ok(const inference_cuda_qsa_attn_args &a) {
    if (a.n_kv_head < 1 || a.n_kv_head > kMaxKvHeads || a.n_head < a.n_kv_head ||
        a.n_head % a.n_kv_head != 0 || a.n_head / a.n_kv_head > kMaxGroup) {
        return false;
    }
    return a.rows >= 1 && a.rows <= kMaxRows && a.ratio >= 1 && a.ratio <= kMaxRatio &&
           a.budget_blocks >= 1 && a.budget_blocks <= kMaxBudget && a.max_splits >= 1 &&
           a.max_splits <= kMaxSplits && (long long)a.ld >= (long long)a.n_head * 2 * kHead &&
           std::isfinite(a.scale);
}

bool valid(const inference_cuda_qsa_attn_args &a, void *stream) {
    if (stream == NULL || a.q == NULL || a.proj == NULL || a.k_cache == NULL ||
        a.v_cache == NULL || a.chosen == NULL || a.part == NULL || a.out == NULL || !shape_ok(a)) {
        return false;
    }
    const void *wide[] = {a.q, a.k_cache, a.v_cache};
    for (const void *p : wide) {
        if (!aligned(p, 16)) return false;
    }
    const void *narrow[] = {a.proj, a.chosen, a.pos, a.part, a.out, a.status};
    for (const void *p : narrow) {
        if (p != NULL && !aligned(p, 4)) return false;
    }
    const size_t f = 4, heads = (size_t)a.n_head;
    const size_t cache = (size_t)a.rows * a.n_kv_head * kHead * 2;
    const Span written[] = {{a.part, heads * a.max_splits * kPartStride * f},
                            {a.out, heads * kHead * f},
                            {a.status, f}};
    const Span read[] = {{a.q, heads * kHead * f},       {a.proj, heads * 2 * kHead * f},
                         {a.k_cache, cache},             {a.v_cache, cache},
                         {a.chosen, (size_t)a.budget_blocks * f}, {a.pos, f}};
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            if (!apart(written[i], written[j])) return false;
        }
        for (const Span &r : read) {
            if (!apart(written[i], r)) return false;
        }
    }
    return true;
}

struct AttnArgs {
    const float *q, *proj;
    const __half *k, *v;
    const int *chosen, *pos;
    float *part, *out;
    int *status;
    int ld, n_head, n_kv, rows, ratio, budget, splits, pos0;
    float scale;
};

/* The attended positions of the token: `count` of them, the chosen blocks' first. */
struct Window {
    bool ok;
    int blocks, take, count;
};

__device__ Window window_of(const AttnArgs &a) {
    const long long p = a.pos != NULL ? (long long)*a.pos : (long long)a.pos0;
    Window w{p >= 0 && p + 1 <= a.rows, 0, 0, 0};
    if (!w.ok) return w;
    w.blocks = (int)((p + 1) / a.ratio);
    w.take = w.blocks < a.budget ? w.blocks : a.budget;
    w.count = w.take * a.ratio + (int)(p + 1 - (long long)w.blocks * a.ratio);
    return w;
}

/* The cache row of attended entry j, or -1 when its chosen block is not a completed block. */
__device__ int row_of(const AttnArgs &a, const Window &w, int j) {
    const int chosen_end = w.take * a.ratio;
    if (j >= chosen_end) return w.blocks * a.ratio + (j - chosen_end);
    const int b = a.chosen[j / a.ratio];
    return b >= 0 && b < w.blocks ? b * a.ratio + j % a.ratio : -1;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ void widen(const __half *src, float *dst) {
    const uint4 raw = *reinterpret_cast<const uint4 *>(src);
    const __half2 *h2 = reinterpret_cast<const __half2 *>(&raw);
#pragma unroll
    for (int i = 0; i < kLaneDims / 2; ++i) {
        const float2 f2 = __half22float2(h2[i]);
        dst[2 * i] = f2.x;
        dst[2 * i + 1] = f2.y;
    }
}

struct Tiles {
    __half k[kTile][kHead];
    __half v[kTile][kHead];
    int rows[kTile];
    int bad;
};

/* Stages the cache rows of entries t0 .. t0 + n - 1 for kv head g; a bad entry stages zeros
 * and marks the block. Every thread of the block calls it. */
__device__ void stage(const AttnArgs &a, const Window &w, int g, int t0, int n, Tiles &s) {
    if (threadIdx.x < kTile) {
        const int r = threadIdx.x < n ? row_of(a, w, t0 + threadIdx.x) : -1;
        if (threadIdx.x < n && r < 0) s.bad = 1;
        s.rows[threadIdx.x] = r;
    }
    __syncthreads();
    constexpr int kChunks = kHead / 8;
    for (int i = threadIdx.x; i < 2 * n * kChunks; i += blockDim.x) {
        const bool is_v = i >= n * kChunks;
        const int rest = is_v ? i - n * kChunks : i;
        const int entry = rest / kChunks, c = rest % kChunks;
        const int r = s.rows[entry];
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (r >= 0) {
            const __half *src = (is_v ? a.v : a.k) + ((size_t)r * a.n_kv + g) * kHead + c * 8;
            val = *reinterpret_cast<const uint4 *>(src);
        }
        __half *dst = (is_v ? s.v[entry] : s.k[entry]) + c * 8;
        *reinterpret_cast<uint4 *>(dst) = val;
    }
    __syncthreads();
}

// One block per (kv head, split), group * 32 threads, warp w = query head g * group + w.
__global__ void __launch_bounds__(kMaxGroup * 32) qsa_attn_split_k(AttnArgs a) {
    __shared__ __align__(16) Tiles s;
    const int g = blockIdx.x, split = blockIdx.y;
    const int group = a.n_head / a.n_kv;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int h = g * group + warp;
    const Window w = window_of(a);
    if (!w.ok) return;
    const int per = (w.count + a.splits - 1) / a.splits;
    const int lo = split * per;
    const int hi = lo + per < w.count ? lo + per : w.count;
    float q[kLaneDims], acc[kLaneDims];
    const float4 *q4 = reinterpret_cast<const float4 *>(a.q + (size_t)h * kHead + lane * kLaneDims);
    const float4 qa = q4[0], qb = q4[1];
    q[0] = qa.x, q[1] = qa.y, q[2] = qa.z, q[3] = qa.w;
    q[4] = qb.x, q[5] = qb.y, q[6] = qb.z, q[7] = qb.w;
#pragma unroll
    for (int i = 0; i < kLaneDims; ++i) acc[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    if (threadIdx.x == 0) s.bad = 0;
    __syncthreads();
    for (int t0 = lo; t0 < hi; t0 += kTile) {
        const int n = hi - t0 < kTile ? hi - t0 : kTile;
        stage(a, w, g, t0, n, s);
        for (int j = 0; j < n; ++j) {
            if (s.rows[j] < 0) continue;
            float k[kLaneDims], v[kLaneDims];
            widen(&s.k[j][lane * kLaneDims], k);
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < kLaneDims; ++i) dot += q[i] * k[i];
            const float score = warp_sum(dot) * a.scale;
            const float m_new = fmaxf(m, score);
            const float carry = expf(m - m_new);
            const float e = expf(score - m_new);
            widen(&s.v[j][lane * kLaneDims], v);
            l = l * carry + e;
#pragma unroll
            for (int i = 0; i < kLaneDims; ++i) acc[i] = acc[i] * carry + e * v[i];
            m = m_new;
        }
        __syncthreads();
    }
    float *part = a.part + ((size_t)h * a.splits + split) * kPartStride;
    if (lane == 0) {
        part[0] = s.bad ? __int_as_float(0x7fc00000) : m;
        part[1] = l;
    }
#pragma unroll
    for (int i = 0; i < kLaneDims; ++i) part[2 + lane * kLaneDims + i] = acc[i];
}

// One block per query head, one thread per value.
__global__ void __launch_bounds__(kHead) qsa_attn_combine_k(AttnArgs a) {
    const int h = blockIdx.x, d = threadIdx.x;
    const Window w = window_of(a);
    bool poisoned = !w.ok;
    float top = -INFINITY;
    for (int sp = 0; !poisoned && sp < a.splits; ++sp) {
        const float m = a.part[((size_t)h * a.splits + sp) * kPartStride];
        poisoned = m != m;
        top = fmaxf(top, m);
    }
    float *out = a.out + (size_t)h * kHead + d;
    if (poisoned) {
        *out = __int_as_float(0x7fc00000);
        if (a.status != NULL && h == 0 && d == 0) atomicOr(a.status, kStatusBits);
        return;
    }
    float sum = 0.0f, value = 0.0f;
    for (int sp = 0; sp < a.splits; ++sp) {
        const float *part = a.part + ((size_t)h * a.splits + sp) * kPartStride;
        if (part[1] > 0.0f) {
            const float f = expf(part[0] - top);
            sum += part[1] * f;
            value += part[2 + d] * f;
        }
    }
    const float gate = a.proj[(size_t)h * 2 * kHead + kHead + d];
    *out = value / sum * (1.0f / (1.0f + expf(-gate)));
}

}  // namespace

extern "C" int inference_cuda_qsa_attn_launch(const inference_cuda_qsa_attn_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const inference_cuda_qsa_attn_args &x = *args;
        const AttnArgs a{(const float *)x.q,      (const float *)x.proj,  (const __half *)x.k_cache,
                         (const __half *)x.v_cache, (const int *)x.chosen, (const int *)x.pos,
                         (float *)x.part,         (float *)x.out,         (int *)x.status,
                         x.ld,                    x.n_head,               x.n_kv_head,
                         x.rows,                  x.ratio,                x.budget_blocks,
                         x.max_splits,            x.pos0,                 x.scale};
        const cudaStream_t s = (cudaStream_t)stream;
        const int group = x.n_head / x.n_kv_head;
        Failure failure;
        qsa_attn_split_k<<<dim3(x.n_kv_head, x.max_splits), group * 32, 0, s>>>(a);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            qsa_attn_combine_k<<<x.n_head, kHead, 0, s>>>(a);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
