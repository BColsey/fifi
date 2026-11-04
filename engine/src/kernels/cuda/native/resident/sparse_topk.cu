/* The sparse-attention deterministic top-K kernel of the resident decode step: one block of 512
 * threads per KV head. A four-level 8-bit radix select over a shared histogram finds the bit
 * pattern of the take-th largest candidate score (scores are non-negative f32, so their bit
 * patterns rank as unsigned integers); two tiled exclusive-prefix fill passes then emit the
 * strictly-greater cells and the lowest-index ties, each ascending, followed by the sink cells
 * and the recent window. The contract is in `resident/sparse_topk.h`; the poison rule mirrors
 * attn_prep and kv8_append.
 */
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/sparse_topk.h"

namespace {

constexpr int kThreads = 512;
constexpr int kMaxKvHeads = 64;
constexpr int kMaxCells = (1 << 30) - 1;
constexpr int kMaxTopK = 1 << 24;
// Mask 4 of StepParams.status (invalid step state); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *scores;  // [n_kv_head][cells]
    unsigned *indices;    // [n_kv_head][top_k + sink + recent]
    unsigned *counts;     // [n_kv_head]
    const int *pos_dev;
    int *status;
    int n_kv_head, cells, top_k, sink, recent;
};

// Inclusive scan of `s` over `n` elements (n <= kThreads) in the fixed step order; every thread
// ends with its inclusive prefix in s[threadIdx.x].
__device__ inline void scan_inclusive(unsigned *s, int n) {
    const int i = threadIdx.x;
    for (int off = 1; off < n; off <<= 1) {
        unsigned add = (i >= off && i < n) ? s[i - off] : 0u;
        __syncthreads();
        if (i < n) s[i] += add;
        __syncthreads();
    }
}

__global__ void sparse_topk_k(Args a) {
    // Poison rule, as in attn_prep and kv8_append: the device word decides the kv length,
    // tested in 64 bits against the cells (and against a candidate-free selection) before any
    // other memory access. A refused launch writes nothing and ORs mask 4 into status (one
    // thread, integer atomic).
    const int kv_len = *a.pos_dev + 1;
    if (*a.pos_dev < 0 || (long long)*a.pos_dev + 1 > a.cells ||
        kv_len <= a.sink + a.recent) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int kvh = blockIdx.x, tid = threadIdx.x;
    const int lo = a.sink, hi = kv_len - a.recent;
    const int candidates = hi - lo;
    const int take = candidates < a.top_k ? candidates : a.top_k;
    __shared__ unsigned hist[256];
    __shared__ unsigned scan[kThreads];
    __shared__ unsigned prefix_s;
    __shared__ unsigned threshold_bits;
    __shared__ int greater_count;
    __shared__ unsigned tile_g, tile_e;  // tile totals: one writer, read by all for the carries

    const float *plane = a.scores + (size_t)kvh * a.cells;
    unsigned *out = a.indices + (size_t)kvh * (a.top_k + a.sink + a.recent);

    // Radix select, 8 bits per level from the top: after each level `prefix` holds the fixed
    // high bytes and `remaining` how many picks still fall inside the surviving range.
    // The radix prefix lives in SHARED memory: only thread 0 advances it per level, and every
    // thread must filter the next level against the same value. A private copy left the
    // non-zero threads filtering against zero: empty histograms, a 0xFF threshold, and no
    // output writes at all (the 2026-11-02 device failure).
    unsigned prefix = 0u;
    int remaining = take;
    if (tid == 0) prefix_s = 0u;
    for (int level = 3; level >= 0; --level) {
        for (int b = tid; b < 256; b += kThreads) hist[b] = 0u;
        __syncthreads();
        const int shift = level * 8;
        // Bytes above this level are fixed by the prefix; level 3 has none fixed (mask 0),
        // since a <<32 shift would be undefined.
        const unsigned mask = (level == 3) ? 0u : (0xffffffffu << ((level + 1) * 8));
        for (int t = lo + tid; t < hi; t += kThreads) {
            const unsigned bits = __float_as_uint(plane[t]);
            if ((bits & mask) == prefix_s) atomicAdd(&hist[(bits >> shift) & 255], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            // Walk the buckets from the top: the first bucket whose cumulative count reaches
            // `remaining` is the surviving byte.
            unsigned above = 0u;
            int bucket = 255;
            for (; bucket >= 0; --bucket) {
                const unsigned cum = above + hist[bucket];
                if (cum >= (unsigned)remaining) break;
                above = cum;
            }
            prefix |= (unsigned)bucket << shift;
            remaining -= (int)above;
            prefix_s = prefix;
        }
        __syncthreads();
    }
    if (tid == 0) {
        threshold_bits = prefix_s;
        greater_count = 0;
    }
    __syncthreads();

    // Fill, ascending by cell index (the wrapper contract: picks ascending, then sinks, then
    // the recent window). Pass 0 only counts the strictly-greater cells. Pass 1 writes each
    // kept pick at its final ascending rank: a greater cell's rank is the greater cells below
    // it plus the kept ties below it; a tie survives only while its tie-rank (equal cells at
    // lower indices) stays below `need` — ties beyond `need` are dropped, which is exactly the
    // value-desc-then-ascending-index order of the reference selection.
    for (int pass = 0; pass < 2; ++pass) {
        const unsigned threshold = threshold_bits;
        unsigned carry_g = 0u, carry_e = 0u;
        for (int base = lo; base < hi; base += kThreads) {
            const int n = (hi - base) < kThreads ? (hi - base) : kThreads;
            unsigned g = 0u, e = 0u;
            if (tid < n) {
                const unsigned bits = __float_as_uint(plane[base + tid]);
                g = (bits > threshold) ? 1u : 0u;
                e = (bits == threshold) ? 1u : 0u;
            }
            scan[tid] = g;
            scan_inclusive(scan, n);
            const unsigned g_inc = (tid < n) ? scan[tid] : 0u;
            if (tid == n - 1) tile_g = g_inc;
            __syncthreads();
            scan[tid] = e;
            scan_inclusive(scan, n);
            const unsigned e_inc = (tid < n) ? scan[tid] : 0u;
            if (tid == n - 1) tile_e = e_inc;
            __syncthreads();
            if (pass == 1 && tid < n) {
                const unsigned need = (unsigned)(take - greater_count);
                const unsigned gb = carry_g + g_inc - g;
                const unsigned eb = carry_e + e_inc - e;
                unsigned rank = 0xffffffffu;
                if (g != 0u) {
                    rank = gb + (eb < need ? eb : need);
                } else if (e != 0u && eb < need) {
                    rank = gb + eb;
                }
                if (rank != 0xffffffffu) out[rank] = (unsigned)(base + tid);
            }
            carry_g += tile_g;
            carry_e += tile_e;
            __syncthreads();
        }
        if (pass == 0 && tid == 0) greater_count = (int)carry_g;
        __syncthreads();
    }

    // Sinks and the recent window after the selected cells, then the head's count.
    for (int i = tid; i < a.sink; i += kThreads) out[take + i] = (unsigned)i;
    for (int j = tid; j < a.recent; j += kThreads)
        out[take + a.sink + j] = (unsigned)(kv_len - a.recent + j);
    if (tid == 0) a.counts[kvh] = (unsigned)(take + a.sink + a.recent);
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

bool valid_args(const inference_cuda_sparse_topk_args *in) {
    if (in == NULL || in->scores == NULL || in->indices == NULL || in->counts == NULL ||
        in->pos == NULL || in->status == NULL) {
        return false;
    }
    if (!(aligned_to(in->scores, 4) && aligned_to(in->indices, 4) &&
          aligned_to(in->counts, 4) && aligned_to(in->pos, 4) &&
          aligned_to(in->status, 4))) {
        return false;
    }
    return in->n_kv_head >= 1 && in->n_kv_head <= kMaxKvHeads && in->cells >= 1 &&
           in->cells <= kMaxCells && in->top_k >= 1 && in->top_k <= kMaxTopK &&
           in->sink >= 0 && in->recent >= 0 && (long long)in->sink + in->recent < in->cells;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    sparse_topk_k<<<a.n_kv_head, kThreads, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_sparse_topk_launch(const inference_cuda_sparse_topk_args *in,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_args(in)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = {};
        a.scores = (const float *)in->scores;
        a.indices = (unsigned *)in->indices;
        a.counts = (unsigned *)in->counts;
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        a.n_kv_head = in->n_kv_head;
        a.cells = in->cells;
        a.top_k = in->top_k;
        a.sink = in->sink;
        a.recent = in->recent;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
