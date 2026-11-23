/*
 * Split-KV decode attention of the SenseNova Loop decode step.
 *
 * The attn_decode.cu structure (Flash-Decoding, Dao et al. 2023) specialized to the Loop-9B
 * geometry: 4 query heads per kv head, head_dim 256, a linear fp16 cache with every position
 * visible, no sinks. The cache and q/gate row layouts are attn_prep's. The gate is Loop's own
 * sigmoid(gate) output gate, taken from the second half of each q row of the projection, and is
 * optional (proj == NULL) so the kernel family keeps the attn_decode convention.
 *
 * Two launches on one stream:
 *   attn_loop_part_k    grid (max_splits, n_kv_head, m), 256 threads. A block owns one key range
 *                       of one kv head and the 4 query heads that share it. Tiles of 32 keys,
 *                       online softmax; one partial (max, sum, weighted V) per query head goes
 *                       to `part`.
 *   attn_loop_combine_k grid (n_head, m), 256 threads (one per head dimension). Merges the
 *                       partials of the active splits in index order, divides, applies the gate,
 *                       writes out.
 * kv_len is read from device memory when the kernels run, so one captured graph serves every
 * position. There is no float atomic and every reduction has a fixed order. A call whose kv_len
 * is outside [1, cache_rows - (m - 1)] writes NaN to out.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "resident/attn_decode_loop.h"
#include "device.h"

namespace {

constexpr int kGroup = 4;        // query heads per kv head; the lane layout of reduce4 assumes 4
constexpr int kTile = 32;        // keys per tile; also the granule of a key range
constexpr int kDim = 256;        // head_dim, fixed by the Loop family
constexpr int kMaxM = 16;
constexpr int kMaxSplits = 128;  // one thread per split in the combine statistics step

struct Args {
    const float *q;         // [m][n_head][kDim] f32
    const float *proj;      // NULL (no gate) or [m][ld] f32; head h's gate is at h * 2 * kDim + kDim
    const __half *k_cache;  // [cache_rows][n_kv_head][kDim] fp16
    const __half *v_cache;
    const int *kv_len;      // device word: keys visible to token 0; token t sees kv_len + t
    float *part;
    float *out;  // [m][n_head * kDim] f32
    int ld, m, n_head, n_kv_head, cache_rows, max_splits, min_keys;
    float scale;
};

// Key range plan from the number of keys alone, as in attn_decode.cu: n_split splits of `chunk`
// keys (a multiple of the tile) cover n_keys; split s owns [s * chunk, min(n_keys, (s+1) * chunk)).
__device__ inline void split_plan(int n_keys, int max_splits, int min_keys, int &chunk,
                                  int &n_active) {
    const int want = (n_keys + min_keys - 1) / min_keys;
    const int n_split = min(max(want, 1), max_splits);
    chunk = (((n_keys + n_split - 1) / n_split) + kTile - 1) / kTile * kTile;
    n_active = (n_keys + chunk - 1) / chunk;
}

__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ inline float warp_sum_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// Reduces four per-lane partial sums (one per query head) over the 32 lanes: each level halves
// the values a lane keeps and doubles the lanes that share a head, the reduce8 of attn_decode.cu
// with one level less. On return, lane l with (l & 1) == 0 holds the full sum of head
// ((l >> 3) & 1) * 2 + ((l >> 2) & 1); the order is fixed.
__device__ inline float reduce4(const float (&a)[4], int lane, int &head) {
    const unsigned full = 0xffffffffu;
    const bool h8 = lane & 8, h4 = lane & 4;
    float b[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = h8 ? a[i + 2] : a[i], give = h8 ? a[i] : a[i + 2];
        b[i] = keep + __shfl_xor_sync(full, give, 8);
    }
    const float keep = h4 ? b[1] : b[0], give = h4 ? b[0] : b[1];
    float d = keep + __shfl_xor_sync(full, give, 4);
    d += __shfl_xor_sync(full, d, 2);
    d += __shfl_xor_sync(full, d, 1);
    head = (h8 ? 2 : 0) + (h4 ? 1 : 0);
    return d;
}

__global__ void __launch_bounds__(256, 2) attn_loop_part_k(Args a) {
    constexpr int G = kGroup, NCH = kDim / 128;  // 2 chunks of 4 dims per lane over 256 dims
    __shared__ float s_sc[G][kTile + 1];         // scores of the tile, head-major, padded
    __shared__ __align__(16) float s_p[kTile][G];  // probabilities, key-major so a key's 4 are contiguous
    __shared__ float s_corr[G];
    const int s = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int kv_len = *a.kv_len;
    if (kv_len < 1 || kv_len > a.cache_rows - (a.m - 1)) {
        return;
    }
    const int n_keys = kv_len + t;
    int chunk, n_active;
    split_plan(n_keys, a.max_splits, a.min_keys, chunk, n_active);
    if (s >= n_active) {
        return;
    }
    const int k_begin = s * chunk, k_end = min(n_keys, k_begin + chunk);
    const size_t row_stride = (size_t)a.n_kv_head * kDim;

    // This lane's dims are 4*lane .. +3 and 128 + 4*lane .. +3. q is pre-scaled once.
    float4 qv[G][NCH];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float *qh = a.q + ((size_t)t * a.n_head + (size_t)kvh * G + g) * kDim;
#pragma unroll
        for (int c = 0; c < NCH; ++c) {
            qv[g][c] = *reinterpret_cast<const float4 *>(qh + 128 * c + 4 * lane);
            qv[g][c].x *= a.scale; qv[g][c].y *= a.scale; qv[g][c].z *= a.scale; qv[g][c].w *= a.scale;
        }
    }
    float acc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        acc[g] = 0.f;
    }
    float m_run = -INFINITY, l_run = 0.f;  // head `warp`'s running max and sum, same in every lane

    for (int base = k_begin; base < k_end; base += kTile) {
        const int nvalid = min(kTile, k_end - base);
        // V of this tile, one dimension per thread, issued first so its latency overlaps the scores.
        unsigned short vr[kTile];
        const unsigned short *vcol =
            reinterpret_cast<const unsigned short *>(a.v_cache) + (size_t)kvh * kDim + tid;
#pragma unroll
        for (int j = 0; j < kTile; ++j) {
            vr[j] = j < nvalid ? vcol[(size_t)(base + j) * row_stride] : (unsigned short)0;
        }
        // Phase 1: scores. Warp w scores keys w, w + 8, w + 16, w + 24 of the tile.
#pragma unroll
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const __half *kr = a.k_cache + (size_t)(base + j) * row_stride + (size_t)kvh * kDim;
                float kf[4 * NCH];
#pragma unroll
                for (int c = 0; c < NCH; ++c) {
                    const uint2 k4 = *reinterpret_cast<const uint2 *>(kr + 128 * c + 4 * lane);
                    kf[4 * c] = h2f(k4.x & 0xffffu); kf[4 * c + 1] = h2f(k4.x >> 16);
                    kf[4 * c + 2] = h2f(k4.y & 0xffffu); kf[4 * c + 3] = h2f(k4.y >> 16);
                }
                float dots[G];
#pragma unroll
                for (int g = 0; g < G; ++g) {
                    float p = qv[g][0].x * kf[0];
                    p = fmaf(qv[g][0].y, kf[1], p); p = fmaf(qv[g][0].z, kf[2], p);
                    p = fmaf(qv[g][0].w, kf[3], p);
#pragma unroll
                    for (int c = 1; c < NCH; ++c) {
                        p = fmaf(qv[g][c].x, kf[4 * c], p); p = fmaf(qv[g][c].y, kf[4 * c + 1], p);
                        p = fmaf(qv[g][c].z, kf[4 * c + 2], p); p = fmaf(qv[g][c].w, kf[4 * c + 3], p);
                    }
                    dots[g] = p;
                }
                int head;
                const float sum = reduce4(dots, lane, head);
                if ((lane & 1) == 0) {
                    s_sc[head][j] = sum;
                }
            }
        }
        __syncthreads();
        // Phase 2: online softmax statistics, warp w for head w (the 4 warps of 8; the rest idle).
        if (warp < G) {
            const float sc = lane < nvalid ? s_sc[warp][lane] : -INFINITY;
            const float m_new = fmaxf(m_run, warp_max_a(sc));
            const float p = lane < nvalid ? expf(sc - m_new) : 0.f;
            const float corr = expf(m_run - m_new);
            l_run = l_run * corr + warp_sum_a(p);
            m_run = m_new;
            s_p[lane][warp] = p;
            if (lane == 0) {
                s_corr[warp] = corr;
            }
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid].
#pragma unroll
        for (int g = 0; g < G; ++g) {
            acc[g] *= s_corr[g];
        }
#pragma unroll
        for (int j = 0; j < kTile; ++j) {
            const float v = h2f(vr[j]);
            const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
            acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
            acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
        }
    }
    // Partial of (token t, kv head, split s): all stats first, then all accumulators.
    const size_t cell = ((size_t)t * a.n_kv_head + kvh) * a.max_splits + s;
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    float *ml = a.part + cell * G * 2;
    float *ac = a.part + n_cells * G * 2 + cell * G * kDim;
    if (lane == 0 && warp < G) {
        ml[warp * 2] = m_run;
        ml[warp * 2 + 1] = l_run;
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        ac[(size_t)g * kDim + tid] = acc[g];
    }
}

// Element of the projection row that holds the gate of (token t, head h, dimension d).
__device__ inline size_t gate_index(const Args &a, int t, int h, int d) {
    const size_t row = (size_t)t * a.ld;
    return row + (size_t)h * 2 * kDim + kDim + d;
}

__global__ void __launch_bounds__(kDim) attn_loop_combine_k(Args a) {
    constexpr int G = kGroup;
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[kDim / 32];
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int kvh = h / G, g = h % G;
    float *o = a.out + ((size_t)t * a.n_head + h) * kDim + d;
    const int kv_len = *a.kv_len;
    if (kv_len < 1 || kv_len > a.cache_rows - (a.m - 1)) {
        *o = __int_as_float(0x7fc00000);  // NaN: the caller's non-finite check ends the job
        return;
    }
    const int n_keys = kv_len + t;
    int chunk, n_active;
    split_plan(n_keys, a.max_splits, a.min_keys, chunk, n_active);
    const size_t n_cells = (size_t)a.m * a.n_kv_head * a.max_splits;
    const size_t cell0 = ((size_t)t * a.n_kv_head + kvh) * a.max_splits;
    const float *ml = a.part + (cell0 * G + g) * 2;
    const float *ac = a.part + n_cells * G * 2 + (cell0 * G + g) * kDim + d;
    const size_t ml_step = (size_t)G * 2, ac_step = (size_t)G * kDim;
    // Statistics: thread s < n_active reads split s; the block max is exact in any order.
    float ms = -INFINITY;
    if (d < n_active) {
        ms = ml[d * ml_step];
        s_l[d] = ml[d * ml_step + 1];
    }
    const float wm = warp_max_a(ms);
    if ((d & 31) == 0) {
        s_red[d >> 5] = wm;
    }
    __syncthreads();
    float M = s_red[0];
#pragma unroll
    for (int i = 1; i < kDim / 32; ++i) {
        M = fmaxf(M, s_red[i]);
    }
    if (d < n_active) {
        s_w[d] = expf(ms - M);
    }
    __syncthreads();
    float L = 0.f;
    for (int s = 0; s < n_active; ++s) {
        L = fmaf(s_l[s], s_w[s], L);
    }
    float A = 0.f;  // index order; the loads do not depend on the weights, so they overlap
#pragma unroll 16
    for (int s = 0; s < n_active; ++s) {
        A = fmaf(ac[s * ac_step], s_w[s], A);
    }
    const float sig = a.proj == NULL
                          ? 1.f
                          : 1.f / (1.f + expf(-a.proj[gate_index(a, t, h, d)]));
    *o = (A / L) * sig;
}

bool aligned_to(const void *p, size_t n) { return ((uintptr_t)p % n) == 0; }

bool valid(const inference_cuda_attn_decode_loop_args *a) {
    if (a == NULL || a->q == NULL || a->k_cache == NULL || a->v_cache == NULL ||
        a->kv_len == NULL || a->part == NULL || a->out == NULL) {
        return false;
    }
    const bool gate_ok = a->proj == NULL ? true : a->ld >= a->n_head * 2 * kDim;
    const bool scale_ok = a->scale > 0.f && a->scale < 1e30f;
    return gate_ok && scale_ok && a->m >= 1 && a->m <= kMaxM && a->n_kv_head >= 1 &&
           a->n_kv_head <= 64 && a->n_head == a->n_kv_head * kGroup && a->cache_rows >= 1 &&
           a->max_splits >= 1 && a->max_splits <= kMaxSplits && a->min_keys >= kTile &&
           a->min_keys % kTile == 0 && a->min_keys <= (1 << 20) && aligned_to(a->q, 16) &&
           (a->proj == NULL || aligned_to(a->proj, 4)) && aligned_to(a->k_cache, 16) &&
           aligned_to(a->v_cache, 16) && aligned_to(a->kv_len, 4) && aligned_to(a->part, 16) &&
           aligned_to(a->out, 4);
}

}  // namespace

// Floats of `part` for the rule above; the Rust side pins the same numbers.
extern "C" size_t inference_cuda_attn_decode_loop_part_floats(int m, int n_kv_head,
                                                              int max_splits) {
    return (size_t)m * n_kv_head * max_splits * kGroup * (2 + kDim);
}

extern "C" int inference_cuda_attn_decode_loop_launch(
    const inference_cuda_attn_decode_loop_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(in)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = {};
        a.q = (const float *)in->q;
        a.proj = (const float *)in->proj;
        a.k_cache = (const __half *)in->k_cache;
        a.v_cache = (const __half *)in->v_cache;
        a.kv_len = (const int *)in->kv_len;
        a.part = (float *)in->part;
        a.out = (float *)in->out;
        a.ld = in->ld;
        a.m = in->m;
        a.n_head = in->n_head;
        a.n_kv_head = in->n_kv_head;
        a.cache_rows = in->cache_rows;
        a.max_splits = in->max_splits;
        a.min_keys = in->min_keys;
        a.scale = in->scale;
        const cudaStream_t s = (cudaStream_t)stream;
        attn_loop_part_k<<<dim3(a.max_splits, a.n_kv_head, a.m), 256, 0, s>>>(a);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            attn_loop_combine_k<<<dim3(a.n_head, a.m), kDim, 0, s>>>(a);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
