/* The sparse-attention relevance-score kernel of the resident decode step: one block of 256
 * threads per KV head. Phase 0 projects every query lane of the head onto the basis (shared
 * f32); phase 1 walks the candidate cells once with an online max/sum per lane (Chan's
 * streaming softmax) and block-reduces the (m, sum) pairs in a fixed tree order; phase 2 walks
 * the candidates again and stores `sum_lane softmax_lane(logit)[t]` per cell, in a fixed serial
 * lane order. The contract is in `resident/sparse_score.h`; the poison rule mirrors attn_prep
 * and kv8_append.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/sparse_score.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxKvHeads = 64;
constexpr int kMinHead = 32;
constexpr int kMaxHead = 1024;
constexpr int kMaxRank = 1024;
constexpr int kMaxGroup = 16;         // query lanes per KV head
constexpr int kMaxProjection = 4096;  // group * rank values in shared
constexpr int kMaxCells = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *q;       // [n_q_head][head_dim]
    const float *basis;   // [n_kv_head][rank][head_dim]
    const __half *sketch; // [n_kv_head][cells][rank]
    float *scores;        // [n_kv_head][cells], candidates only
    const int *pos_dev;
    int *status;
    int n_q_head, n_kv_head, head_dim, rank, cells, sink, recent;
    float scale;
};

__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// Tree max of the shared array in a fixed order (every thread ends with the max in s[tid]).
__device__ inline void tree_max(float *s) {
#pragma unroll
    for (int o = kThreads / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + o]);
        __syncthreads();
    }
}

// Tree sum of the shared array (the caller has already rescaled every element onto one
// common max); deterministic in the fixed tree order.
__device__ inline void tree_sum(float *s) {
    __syncthreads();
#pragma unroll
    for (int o = kThreads / 2; o > 0; o >>= 1) {
        if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
        __syncthreads();
    }
}

__global__ void sparse_score_k(Args a) {
    // Poison rule, as in attn_prep and kv8_append: the device word decides the kv length,
    // tested in 64 bits against the cells before any other memory access. A refused launch
    // writes nothing and ORs mask 4 into status (one thread, integer atomic).
    const int kv_len = *a.pos_dev + 1;
    if (*a.pos_dev < 0 || (long long)*a.pos_dev + 1 > a.cells) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int kvh = blockIdx.x, tid = threadIdx.x;
    const int group = a.n_q_head / a.n_kv_head;
    const int lo = a.sink;
    const int hi = kv_len - a.recent;
    extern __shared__ float smem[];
    // Layout: qz [group][rank], red [kThreads], m [group], inv_s [group].
    float *qz = smem;
    float *red = smem + group * a.rank;
    float *m = red + kThreads;
    float *inv_s = m + group;

    // Phase 0: qz[lane][r] = sum_d basis[h][r][d] * q[lane][d], one serial fma chain per pair
    // (fixed order, so the projection is bit-deterministic).
    const int pairs = group * a.rank;
    for (int p = tid; p < pairs; p += kThreads) {
        const int lane = p / a.rank, r = p % a.rank;
        const float *qrow = a.q + (size_t)(kvh * group + lane) * a.head_dim;
        const float *brow = a.basis + ((size_t)kvh * a.rank + r) * a.head_dim;
        float acc = 0.f;
#pragma unroll 4
        for (int d = 0; d < a.head_dim; ++d) acc = __fmaf_rn(brow[d], qrow[d], acc);
        qz[p] = acc;
    }
    __syncthreads();
    if (hi <= lo) return;  // no candidates: the host takes the dense fallback

    // Phase 1: online (m, sum) per lane over the candidate cells, then one block reduction per
    // lane. The logit of cell t for a lane is scale * sum_r qz[lane][r] * z(t)[r], serial in r.
    {
        float lm[kMaxGroup], ls[kMaxGroup];
        for (int lane = 0; lane < group; ++lane) {
            lm[lane] = -INFINITY;
            ls[lane] = 0.f;
        }
        const __half *plane = a.sketch + (size_t)kvh * a.cells * a.rank;
        for (int t = lo + tid; t < hi; t += kThreads) {
            const __half *z = plane + (size_t)t * a.rank;
            for (int lane = 0; lane < group; ++lane) {
                const float *lane_qz = qz + lane * a.rank;
                float dot = 0.f;
                for (int r = 0; r < a.rank; ++r)
                    dot = __fmaf_rn(lane_qz[r], h2f(__half_as_ushort(z[r])), dot);
                const float l = dot * a.scale;
                if (l > lm[lane]) {
                    ls[lane] = ls[lane] * expf(lm[lane] - l) + 1.f;
                    lm[lane] = l;
                } else {
                    ls[lane] += expf(l - lm[lane]);
                }
            }
        }
        // Block-reduce (m, sum) per lane: two shared trees per lane (max, then rescaled sum).
        for (int lane = 0; lane < group; ++lane) {
            red[tid] = lm[lane];
            __syncthreads();
            tree_max(red);
            const float m_lane = red[0];
            __syncthreads();
            red[tid] = (lm[lane] == -INFINITY) ? -INFINITY : ls[lane] * expf(lm[lane] - m_lane);
            tree_sum(red);
            if (tid == 0) {
                m[lane] = m_lane;
                inv_s[lane] = (red[0] > 0.f) ? 1.f / red[0] : 0.f;
            }
            __syncthreads();
        }
    }

    // Phase 2: score every candidate cell: sum over lanes of exp(logit - m_lane) * inv_s_lane.
    {
        float *out = a.scores + (size_t)kvh * a.cells;
        const __half *plane = a.sketch + (size_t)kvh * a.cells * a.rank;
        for (int t = lo + tid; t < hi; t += kThreads) {
            const __half *z = plane + (size_t)t * a.rank;
            float score = 0.f;
            for (int lane = 0; lane < group; ++lane) {
                const float *lane_qz = qz + lane * a.rank;
                float dot = 0.f;
                for (int r = 0; r < a.rank; ++r)
                    dot = __fmaf_rn(lane_qz[r], h2f(__half_as_ushort(z[r])), dot);
                score = __fmaf_rn(expf(dot * a.scale - m[lane]), inv_s[lane], score);
            }
            out[t] = score;
        }
    }
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

bool valid_args(const inference_cuda_sparse_score_args *in) {
    if (in == NULL || in->q == NULL || in->basis == NULL || in->sketch == NULL ||
        in->scores == NULL || in->pos == NULL || in->status == NULL) {
        return false;
    }
    if (!(aligned_to(in->q, 4) && aligned_to(in->basis, 4) && aligned_to(in->sketch, 4) &&
          aligned_to(in->scores, 4) && aligned_to(in->pos, 4) && aligned_to(in->status, 4))) {
        return false;
    }
    const int group = in->n_kv_head >= 1 ? in->n_q_head / in->n_kv_head : 0;
    return in->n_kv_head >= 1 && in->n_kv_head <= kMaxKvHeads && group >= 1 && group <= kMaxGroup &&
           group * in->n_kv_head == in->n_q_head && in->head_dim >= kMinHead &&
           in->head_dim <= kMaxHead && in->head_dim % 32 == 0 && in->rank >= 2 &&
           in->rank <= kMaxRank && in->rank <= in->head_dim && group * in->rank <= kMaxProjection &&
           in->cells >= 1 && in->cells <= kMaxCells && in->sink >= 0 && in->recent >= 0 &&
           (long long)in->sink + in->recent < in->cells;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    const int shared = (a.n_q_head / a.n_kv_head) * a.rank * 4 /* qz */ + kThreads * 4 /* red */ +
                       (a.n_q_head / a.n_kv_head) * 8 /* m, inv_s */;
    sparse_score_k<<<a.n_kv_head, kThreads, shared, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_sparse_score_launch(const inference_cuda_sparse_score_args *in,
                                                  void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_args(in) || !(in->scale > 0.f) || in->scale != in->scale) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = {};
        a.q = (const float *)in->q;
        a.basis = (const float *)in->basis;
        a.sketch = (const __half *)in->sketch;
        a.scores = (float *)in->scores;
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        a.n_q_head = in->n_q_head;
        a.n_kv_head = in->n_kv_head;
        a.head_dim = in->head_dim;
        a.rank = in->rank;
        a.cells = in->cells;
        a.sink = in->sink;
        a.recent = in->recent;
        a.scale = in->scale;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
