// The block indexer of Qwen3.8 Flash-Next QSA attention (see resident/qsa_index.h for the
// formulas and the argument contract). index enqueues three kernels: the normalized and rotated
// queries, the pooled keys of the blocks the chunk completes, then the raw-key ring of the open
// block (after the keys, which read the ring). score enqueues one, a grid row per token. Plain
// f32 arithmetic in the host's expression order where it is a fixed order (the pooling);
// reductions are warp trees.
#include <cuda_runtime.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/qsa_index.h"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxHeads = 64;
constexpr int kMaxDim = 1024;
constexpr int kMaxQueryFloats = 8192;
constexpr int kMaxRatio = 64;
constexpr int kMaxCapBlocks = 1 << 24;
constexpr int kStatusBits = 4;  // INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of step_params.h
constexpr int kScoreThreads = 256;
constexpr int kScoreWarps = kScoreThreads / 32;
constexpr int kMaxScoreBlocks = 1024;

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

/* Every written span apart from every other span (written or read). */
bool disjoint(const Span *written, int nw, const Span *read, int nr) {
    for (int i = 0; i < nw; ++i) {
        for (int j = i + 1; j < nw; ++j) {
            if (!apart(written[i], written[j])) return false;
        }
        for (int j = 0; j < nr; ++j) {
            if (!apart(written[i], read[j])) return false;
        }
    }
    return true;
}

bool shape_ok(int heads, int dim, int ratio, int cap_blocks) {
    return heads >= 1 && heads <= kMaxHeads && dim >= 32 && dim <= kMaxDim && dim % 32 == 0 &&
           heads * dim <= kMaxQueryFloats && ratio >= 1 && ratio <= kMaxRatio &&
           cap_blocks >= 1 && cap_blocks <= kMaxCapBlocks;
}

bool valid_index(const inference_cuda_qsa_index_args &a, void *stream) {
    if (stream == NULL || a.proj == NULL || a.q_norm == NULL || a.k_norm == NULL ||
        a.queries == NULL || a.ring == NULL || a.keys == NULL) {
        return false;
    }
    if (a.m < 1 || a.m > kMaxM || !shape_ok(a.heads, a.dim, a.ratio, a.cap_blocks) ||
        a.rotary_dim < 2 || a.rotary_dim > a.dim || a.rotary_dim % 2 != 0 || a.col_off < 0 ||
        (long long)a.ld < (long long)a.col_off + (long long)(a.heads + 1) * a.dim ||
        !std::isfinite(a.theta) || !(a.theta > 0.0f) || !std::isfinite(a.eps) || !(a.eps >= 0.0f)) {
        return false;
    }
    const void *ptrs[] = {a.proj, a.q_norm, a.k_norm, a.queries, a.ring, a.keys, a.pos, a.status};
    for (const void *p : ptrs) {
        if (p != NULL && !aligned(p, 4)) return false;
    }
    const size_t f = sizeof(float), m = (size_t)a.m, dim = (size_t)a.dim;
    const size_t row = (size_t)a.col_off + (size_t)(a.heads + 1) * dim;
    const Span written[] = {{a.queries, m * a.heads * dim * f},
                            {a.ring, (size_t)a.ratio * dim * f},
                            {a.keys, (size_t)a.cap_blocks * dim * f},
                            {a.status, f}};
    const Span read[] = {{a.proj, ((m - 1) * (size_t)a.ld + row) * f},
                         {a.q_norm, dim * f},
                         {a.k_norm, dim * f},
                         {a.pos, f}};
    return disjoint(written, 4, read, 4);
}

bool valid_score(const inference_cuda_qsa_score_args &a, void *stream) {
    if (stream == NULL || a.queries == NULL || a.keys == NULL || a.scores == NULL ||
        a.m < 1 || a.m > kMaxM || !shape_ok(a.heads, a.dim, a.ratio, a.cap_blocks)) {
        return false;
    }
    const void *ptrs[] = {a.queries, a.keys, a.scores, a.pos, a.status};
    for (const void *p : ptrs) {
        if (p != NULL && !aligned(p, 4)) return false;
    }
    const size_t f = sizeof(float), m = (size_t)a.m, dim = (size_t)a.dim;
    const Span written[] = {{a.scores, m * a.cap_blocks * f}, {a.status, f}};
    const Span read[] = {{a.queries, m * a.heads * dim * f},
                         {a.keys, (size_t)a.cap_blocks * dim * f},
                         {a.pos, f}};
    return disjoint(written, 2, read, 3);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* The block sum for blockDim.x a multiple of 32 (at most 1024), returned to every thread. */
__device__ float block_sum(float v) {
    __shared__ float part[32];
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = lane < (int)blockDim.x / 32 ? part[lane] : 0.0f;
    return warp_sum(v);
}

/* The position of token 0, and whether `first + m` fits the cache (the 64-bit guard). */
__device__ __forceinline__ bool position(const int *pos, int pos0, int m, int ratio, int cap,
                                         long long *first) {
    *first = pos != NULL ? (long long)*pos : (long long)pos0;
    return *first >= 0 && *first + m <= (long long)cap * ratio;
}

/* rms0 of the block's row value `v` (one value per thread), staged in `x`, then the rotation
 * at position `at`. Returns this thread's output. Every thread of the block calls it. */
__device__ float norm_rope(float v, const float *w, float eps, int rotary_dim, float theta,
                           long long at, float *x) {
    const int d = threadIdx.x, dim = blockDim.x;
    const float mean = block_sum(v * v) / (float)dim;
    x[d] = v * (1.0f + w[d]) / sqrtf(mean + eps);
    __syncthreads();
    float out = x[d];
    const int half = rotary_dim / 2;
    if (d < rotary_dim) {
        const int i = d % half;
        const float exponent = (float)(2 * i) / (float)rotary_dim;
        const float angle = (float)at / powf(theta, exponent);
        float sn, cs;
        sincosf(angle, &sn, &cs);
        out = d < half ? x[d] * cs - x[d + half] * sn : x[d] * cs + x[d - half] * sn;
    }
    return out;
}

struct IndexArgs {
    const float *proj, *q_norm, *k_norm;
    float *queries, *ring, *keys;
    const int *pos;
    int *status;
    int m, ld, col_off, heads, dim, rotary_dim, ratio, cap_blocks, pos0;
    float theta, eps;
};

// One block per (head, token), dim threads.
__global__ void __launch_bounds__(kMaxDim) qsa_query_k(IndexArgs a) {
    __shared__ float x[kMaxDim];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    long long first;
    float *out = a.queries + ((size_t)t * a.heads + h) * a.dim;
    if (!position(a.pos, a.pos0, a.m, a.ratio, a.cap_blocks, &first)) {
        if (a.status != NULL && h == 0 && t == 0 && d == 0) atomicOr(a.status, kStatusBits);
        out[d] = __int_as_float(0x7fc00000);
        return;
    }
    const float v = a.proj[(size_t)t * a.ld + a.col_off + (size_t)h * a.dim + d];
    out[d] = norm_rope(v, a.q_norm, a.eps, a.rotary_dim, a.theta, first + t, x);
}

// One block per candidate block b = first / ratio + blockIdx.x, dim threads; the blocks the
// chunk does not complete return at once.
__global__ void __launch_bounds__(kMaxDim) qsa_key_k(IndexArgs a) {
    __shared__ float x[kMaxDim];
    const int d = threadIdx.x;
    long long first;
    if (!position(a.pos, a.pos0, a.m, a.ratio, a.cap_blocks, &first)) return;
    const long long b = first / a.ratio + blockIdx.x;
    if ((b + 1) * a.ratio > first + a.m) return;
    const int key_col = a.col_off + a.heads * a.dim + d;
    float pooled = 0.0f;
    for (int r = 0; r < a.ratio; ++r) {
        const long long p = b * a.ratio + r;
        const float raw = p >= first ? a.proj[(size_t)(p - first) * a.ld + key_col]
                                     : a.ring[(size_t)(p % a.ratio) * a.dim + d];
        pooled += raw / (float)a.ratio;
    }
    a.keys[(size_t)b * a.dim + d] =
        norm_rope(pooled, a.k_norm, a.eps, a.rotary_dim, a.theta, b * a.ratio, x);
}

// One block of dim threads: the raw keys of the last min(m, ratio) positions into the ring.
__global__ void __launch_bounds__(kMaxDim) qsa_ring_k(IndexArgs a) {
    const int d = threadIdx.x;
    long long first;
    if (!position(a.pos, a.pos0, a.m, a.ratio, a.cap_blocks, &first)) return;
    const int key_col = a.col_off + a.heads * a.dim + d;
    const int lo = a.m > a.ratio ? a.m - a.ratio : 0;
    for (int t = lo; t < a.m; ++t) {
        a.ring[(size_t)((first + t) % a.ratio) * a.dim + d] = a.proj[(size_t)t * a.ld + key_col];
    }
}

struct ScoreArgs {
    const float *queries, *keys;
    float *scores;
    const int *pos;
    int *status;
    int heads, dim, ratio, cap_blocks, pos0, m;
};

// Grid row y is token y. Warps stride over its completed blocks; each warp scores one block per
// trip. A grid block with no completed block to score leaves before loading the queries.
__global__ void __launch_bounds__(kScoreThreads) qsa_score_k(ScoreArgs a) {
    __shared__ float q[kMaxQueryFloats];
    long long first;
    if (!position(a.pos, a.pos0, a.m, a.ratio, a.cap_blocks, &first)) {
        if (a.status != NULL && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
            atomicOr(a.status, kStatusBits);
        }
        return;
    }
    const int dim = a.dim;
    const long long blocks = (first + blockIdx.y + 1) / a.ratio;
    if ((long long)blockIdx.x * kScoreWarps >= blocks) return;
    const float *queries = a.queries + (size_t)blockIdx.y * a.heads * dim;
    float *scores = a.scores + (size_t)blockIdx.y * a.cap_blocks;
    for (int i = threadIdx.x; i < a.heads * dim; i += blockDim.x) q[i] = queries[i];
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const long long stride = (long long)gridDim.x * kScoreWarps;
    for (long long b = (long long)blockIdx.x * kScoreWarps + threadIdx.x / 32; b < blocks;
         b += stride) {
        const float *k = a.keys + (size_t)b * dim;
        float total = 0.0f;
        for (int h = 0; h < a.heads; ++h) {
            float dot = 0.0f;
            for (int j = lane; j < dim; j += 32) dot += q[h * dim + j] * k[j];
            dot = warp_sum(dot);
            // fmaxf takes the number over a NaN, as the host's f64::max does.
            total += fmaxf(dot, 0.0f);
        }
        if (lane == 0) scores[b] = total;
    }
}

IndexArgs to_args(const inference_cuda_qsa_index_args &a) {
    return IndexArgs{(const float *)a.proj, (const float *)a.q_norm, (const float *)a.k_norm,
                     (float *)a.queries,    (float *)a.ring,         (float *)a.keys,
                     (const int *)a.pos,    (int *)a.status,         a.m,
                     a.ld,                  a.col_off,               a.heads,
                     a.dim,                 a.rotary_dim,            a.ratio,
                     a.cap_blocks,          a.pos0,                  a.theta,
                     a.eps};
}

}  // namespace

extern "C" int inference_cuda_qsa_index_launch(const inference_cuda_qsa_index_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_index(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const IndexArgs k = to_args(*args);
        const cudaStream_t s = (cudaStream_t)stream;
        Failure failure;
        qsa_query_k<<<dim3(k.heads, k.m), k.dim, 0, s>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.err == cudaSuccess) {
            qsa_key_k<<<k.m / k.ratio + 1, k.dim, 0, s>>>(k);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        if (failure.err == cudaSuccess) {
            qsa_ring_k<<<1, k.dim, 0, s>>>(k);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_qsa_score_launch(const inference_cuda_qsa_score_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_score(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const inference_cuda_qsa_score_args &a = *args;
        int grid = (a.cap_blocks + kScoreWarps - 1) / kScoreWarps;
        grid = grid < kMaxScoreBlocks ? grid : kMaxScoreBlocks;
        Failure failure;
        const ScoreArgs k{(const float *)a.queries, (const float *)a.keys, (float *)a.scores,
                          (const int *)a.pos,       (int *)a.status,        a.heads,
                          a.dim,                    a.ratio,                a.cap_blocks,
                          a.pos0,                   a.m};
        qsa_score_k<<<dim3(grid, a.m), kScoreThreads, 0, (cudaStream_t)stream>>>(k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
