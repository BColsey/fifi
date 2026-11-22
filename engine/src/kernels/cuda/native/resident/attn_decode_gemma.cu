/* Entry of attn_decode_gemma on an explicit stream. Glue only. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <cmath>

#include "abi.h"
#include "device.h"
#include "resident/attn_decode_gemma.h"
#include "resident/multi_row.h"
#include "resident/attn_split_reduce.cuh"

namespace {

constexpr int kMaxSplits = 128;  // one thread per split in the combine statistics step

// Split plan of the gemma split path: n_split = min(ceil(n_keys / min_keys), max_splits)
// chunks of exactly ceil(n_keys / n_split) keys (no tile rounding: the kernel strides
// cells by warp, not by tile); split s owns [s * chunk, min(n_keys, (s + 1) * chunk)).
// One __host__ __device__ definition serves the kernels and the extern-C
// split_plan_raw export, so the Rust pin test checks the very arithmetic the
// device runs.
__host__ __device__ inline void gemma_split_plan(
    int n_keys, int max_splits, int min_keys, int &chunk, int &n_active) {
    const int want = (n_keys + min_keys - 1) / min_keys;
    const int n_split = want < 1 ? 1 : (want > max_splits ? max_splits : want);
    const int ch = (n_keys + n_split - 1) / n_split;
    chunk = ch;
    n_active = (n_keys + ch - 1) / ch;
}


// Ring cache (ring > 0): the key at position p sits in cell p % ring and a length is bounded by the
// position range, not by the physical cells. A linear cache (ring 0) keeps cell = position.
__host__ __device__ inline bool gemma_len_ok(long long len, int cells, int ring) {
    return len >= 1 && (ring > 0 ? len < (1 << 30) : len <= cells);
}
__host__ __device__ inline int gemma_cell(int position, int ring) {
    return ring > 0 ? position % ring : position;
}

// The planes, the visible-key word and the status word of one grid row: the launch's own for a
// single sequence (token t sees kv_len + t keys), the row table's for the many-rows entry (row t
// is one token of its own sequence and sees exactly the kv_len word of its step block).
struct GemmaRow {
    const signed char *k_codes;
    const unsigned short *k_scales;
    const signed char *v_codes;
    const unsigned short *v_scales;
    int kv_len;
    unsigned int *status;
};

template <bool kMany>
__device__ inline GemmaRow gemma_row(const signed char *k_codes, const unsigned short *k_scales,
                                     const signed char *v_codes, const unsigned short *v_scales,
                                     const int *kv_len_word, unsigned int *status,
                                     const inference_cuda_kv_row *rows, int t) {
    if constexpr (kMany) {
        const inference_cuda_kv_row &r = rows[t];
        return {(const signed char *)r.k_codes, (const unsigned short *)r.k_scales,
                (const signed char *)r.v_codes, (const unsigned short *)r.v_scales,
                *(const int *)((const char *)r.step + INFERENCE_CUDA_STEP_KV_LEN_OFFSET),
                (unsigned int *)((char *)r.step + INFERENCE_CUDA_STEP_STATUS_OFFSET)};
    } else {
        return {k_codes, k_scales, v_codes, v_scales, *kv_len_word, status};
    }
}

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

template <bool kMany = false>
__global__ void attn_decode_gemma_kernel(
    float *out, unsigned int *status, const float *q,
    const signed char *k_codes, const unsigned short *k_scales,
    const signed char *v_codes, const unsigned short *v_scales, const int *kv_len_word,
    int n_q_head, int n_kv_head, int head_dim, int cells, int window, float scale, int ring,
    const inference_cuda_kv_row *rows = nullptr) {
    const int token = blockIdx.y;
    const int qh = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int nwarps = (int)blockDim.x / 32;
    const int kvh = (int)(((long long)qh * n_kv_head) / n_q_head);
    const GemmaRow rw = gemma_row<kMany>(k_codes, k_scales, v_codes, v_scales, kv_len_word, status,
                                         rows, token);
    const int64_t requested_len = (int64_t)rw.kv_len + (kMany ? 0 : token);
    const int out_of_range = !gemma_len_ok(requested_len, cells, ring);
    const size_t out_base = ((size_t)token * n_q_head + qh) * (size_t)head_dim;
    if (out_of_range) {
        for (int d = tid; d < head_dim; d += blockDim.x)
            out[out_base + d] = __int_as_float(0x7fc00000);
        if (tid == 0 && rw.status != NULL) atomicOr(rw.status, 4u);
        return;
    }
    const int kv_len = (int)requested_len;
    const int visible = (window > 0 && window < kv_len) ? window : kv_len;
    const int first = kv_len - visible;
    const int groups = head_dim / 32;
    const int per_lane = head_dim / 32;
    float qv[16];
    for (int k = 0; k < per_lane; ++k) qv[k] = q[out_base + lane + 32 * k];
    float acc[16];
    for (int k = 0; k < per_lane; ++k) acc[k] = 0.0f;
    float run_m = -INFINITY;
    float run_l = 0.0f;
    for (int cell = first + warp; cell < kv_len; cell += nwarps) {
        const size_t at = (size_t)kvh * cells + gemma_cell(cell, ring);
        const size_t plane = at * (size_t)head_dim;
        const size_t sbase = at * (size_t)groups;
        float partial = 0.0f;
        for (int k = 0; k < per_lane; ++k) {
            const float kscale = __half2float(__ushort_as_half(rw.k_scales[sbase + k]));
            partial += qv[k] * ((float)rw.k_codes[plane + lane + 32 * k] * kscale);
        }
        float score = partial;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) score += __shfl_xor_sync(0xffffffffu, score, o);
        score *= scale;
        const float new_m = fmaxf(run_m, score);
        const float p = expf(score - new_m);
        const float correction = (run_l == 0.0f) ? 0.0f : expf(run_m - new_m);
        run_l = run_l * correction + p;
        for (int k = 0; k < per_lane; ++k) {
            const float vscale = __half2float(__ushort_as_half(rw.v_scales[sbase + k]));
            acc[k] = acc[k] * correction + p * ((float)rw.v_codes[plane + lane + 32 * k] * vscale);
        }
        run_m = new_m;
    }
    // The combine keeps one staged value per lane per step: a [warps][32][16] array
    // would be 64 KB and exceed the static shared limit.
    __shared__ float sh_m[32];
    __shared__ float sh_l[32];
    __shared__ float sh_a[32][32];
    if (lane == 0) {
        sh_m[warp] = run_m;
        sh_l[warp] = run_l;
    }
    __syncthreads();
    float peak = 0.0f;
    float denom = 0.0f;
    if (warp == 0) {
        peak = -INFINITY;
        for (int w = 0; w < nwarps; ++w) peak = fmaxf(peak, sh_m[w]);
        for (int w = 0; w < nwarps; ++w) {
            const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
            denom += sh_l[w] * wl;
        }
    }
    for (int k = 0; k < per_lane; ++k) {
        sh_a[warp][lane] = acc[k];
        __syncthreads();
        if (warp == 0) {
            float combined = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
                combined += sh_a[w][lane] * wl;
            }
            const int d = lane + 32 * k;
            if (d < head_dim) out[out_base + d] = combined / denom;
        }
        __syncthreads();
    }
}

// The split path. One block per (q head, token, split slot): the block computes the
// same online-softmax partial as the single-pass kernel over its chunk of the visible
// range (identical arithmetic per key), then stages the cross-warp merge the
// single-pass kernel performs and writes ONE split partial: stats (max, sum) then the
// accumulator vector, in the attn_decode_q8 partial order (stats region first).
template <bool kMany = false>
__global__ void __launch_bounds__(256) attn_decode_gemma_part_k(
    float *part, unsigned int *status, const float *q,
    const signed char *k_codes, const unsigned short *k_scales,
    const signed char *v_codes, const unsigned short *v_scales, const int *kv_len_word,
    int n_q_head, int n_kv_head, int head_dim, int cells, int window, float scale,
    int max_splits, int min_keys, int ring, const inference_cuda_kv_row *rows = nullptr) {
    const int token = blockIdx.y;
    const int qh = blockIdx.x;
    const int slot = blockIdx.z;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int nwarps = (int)blockDim.x / 32;
    const int kvh = (int)(((long long)qh * n_kv_head) / n_q_head);
    const GemmaRow rw = gemma_row<kMany>(k_codes, k_scales, v_codes, v_scales, kv_len_word, status,
                                         rows, token);
    const int64_t requested_len = (int64_t)rw.kv_len + (kMany ? 0 : token);
    if (!gemma_len_ok(requested_len, cells, ring)) {
        // The combine kernel owns the NaN output and the status word; nothing to stage.
        if (tid == 0 && rw.status != NULL) atomicOr(rw.status, 4u);
        return;
    }
    const int kv_len = (int)requested_len;
    const int visible = (window > 0 && window < kv_len) ? window : kv_len;
    const int first = kv_len - visible;
    int chunk, n_active;
    gemma_split_plan(visible, max_splits, min_keys, chunk, n_active);
    if (slot >= n_active) return;
    const int begin = first + slot * chunk;
    const int end = min(kv_len, begin + chunk);
    const int groups = head_dim / 32;
    const int per_lane = head_dim / 32;
    const size_t out_base = ((size_t)token * n_q_head + qh) * (size_t)head_dim;
    float qv[16];
    for (int k = 0; k < per_lane; ++k) qv[k] = q[out_base + lane + 32 * k];
    float acc[16];
    for (int k = 0; k < per_lane; ++k) acc[k] = 0.0f;
    float run_m = -INFINITY;
    float run_l = 0.0f;
    for (int cell = begin + warp; cell < end; cell += nwarps) {
        const size_t at = (size_t)kvh * cells + gemma_cell(cell, ring);
        const size_t plane = at * (size_t)head_dim;
        const size_t sbase = at * (size_t)groups;
        float partial = 0.0f;
        for (int k = 0; k < per_lane; ++k) {
            const float kscale = __half2float(__ushort_as_half(rw.k_scales[sbase + k]));
            partial += qv[k] * ((float)rw.k_codes[plane + lane + 32 * k] * kscale);
        }
        float score = partial;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) score += __shfl_xor_sync(0xffffffffu, score, o);
        score *= scale;
        const float new_m = fmaxf(run_m, score);
        const float p = expf(score - new_m);
        const float correction = (run_l == 0.0f) ? 0.0f : expf(run_m - new_m);
        run_l = run_l * correction + p;
        for (int k = 0; k < per_lane; ++k) {
            const float vscale = __half2float(__ushort_as_half(rw.v_scales[sbase + k]));
            acc[k] = acc[k] * correction + p * ((float)rw.v_codes[plane + lane + 32 * k] * vscale);
        }
        run_m = new_m;
    }
    // The staged cross-warp merge of the single-pass kernel, verbatim: sh_m/sh_l per
    // warp, then warp 0 combines each per-lane accumulator column in warp order.
    __shared__ float sh_m[32];
    __shared__ float sh_l[32];
    __shared__ float sh_a[32][32];
    if (lane == 0) {
        sh_m[warp] = run_m;
        sh_l[warp] = run_l;
    }
    __syncthreads();
    float peak = 0.0f;
    float denom = 0.0f;
    if (warp == 0) {
        peak = -INFINITY;
        for (int w = 0; w < nwarps; ++w) peak = fmaxf(peak, sh_m[w]);
        for (int w = 0; w < nwarps; ++w) {
            const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
            denom += sh_l[w] * wl;
        }
    }
    // Partial of (token, q head, split slot): stats region first, then accumulators.
    const size_t n_cells = (size_t)gridDim.y * (size_t)n_q_head * (size_t)max_splits;
    const size_t cellx = ((size_t)token * n_q_head + qh) * (size_t)max_splits + slot;
    float *ml = part + cellx * 2;
    float *ac = part + n_cells * 2 + cellx * (size_t)head_dim;
    for (int k = 0; k < per_lane; ++k) {
        sh_a[warp][lane] = acc[k];
        __syncthreads();
        if (warp == 0) {
            float combined = 0.0f;
            for (int w = 0; w < nwarps; ++w) {
                const float wl = (sh_l[w] == 0.0f) ? 0.0f : expf(sh_m[w] - peak);
                combined += sh_a[w][lane] * wl;
            }
            ac[lane + 32 * k] = combined;
        }
        __syncthreads();
    }
    if (tid == 0) {
        ml[0] = peak;
        ml[1] = denom;
    }
}

template <int D, bool kMany = false>
__global__ void __launch_bounds__(D) attn_decode_gemma_combine_k(
    float *out, unsigned int *status, const int *kv_len_word, const float *part,
    int n_q_head, int m, int head_dim, int cells, int window, int max_splits,
    int min_keys, int ring, const inference_cuda_kv_row *rows = nullptr) {
    __shared__ float s_w[kMaxSplits], s_l[kMaxSplits], s_red[D / 32];
    const int qh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    float *o = out + ((size_t)t * n_q_head + qh) * (size_t)head_dim + d;
    // The combine reads no plane: the single-sequence instance keeps its original reads, the
    // many-rows instance takes the length and status words of its row's step block.
    int64_t requested_len = (int64_t)*kv_len_word + t;
    unsigned int *row_status = status;
    if constexpr (kMany) {
        const char *step = (const char *)rows[t].step;
        requested_len = *(const int *)(step + INFERENCE_CUDA_STEP_KV_LEN_OFFSET);
        row_status = (unsigned int *)((char *)step + INFERENCE_CUDA_STEP_STATUS_OFFSET);
    }
    if (!gemma_len_ok(requested_len, cells, ring)) {
        // Same contract as the single-pass kernel: NaN out, status bit 4.
        *o = __int_as_float(0x7fc00000);
        if (d == 0 && row_status != NULL) atomicOr(row_status, 4u);
        return;
    }
    const int kv_len = (int)requested_len;
    const int visible = (window > 0 && window < kv_len) ? window : kv_len;
    int chunk, n_active;
    gemma_split_plan(visible, max_splits, min_keys, chunk, n_active);
    const size_t n_cells = (size_t)m * (size_t)n_q_head * (size_t)max_splits;
    const size_t cellx = ((size_t)t * n_q_head + qh) * (size_t)max_splits;
    const float *ml = part + cellx * 2;
    const float *ac = part + n_cells * 2 + cellx * (size_t)head_dim + d;
    const size_t ml_step = 2, ac_step = (size_t)head_dim;
    // The shared two-phase merge of attn_split_reduce.cuh; no sink, no gate: the
    // n_active == 1 case reproduces the single-pass kernel's combined / denom.
    const float peak = inference::attn_split_max<D>(d, n_active, ml, ml_step, s_l, s_red);
    if (d < n_active) {
        s_w[d] = expf(ml[d * ml_step] - peak);
    }
    __syncthreads();
    float l, a;
    inference::attn_split_sum(n_active, 0.f, s_w, s_l, ac, ac_step, l, a);
    *o = a / l;
}

}  // namespace

// The argument rules of the single-sequence entry; the many-rows entry shares the shape and
// alignment rules (extents false) and proves the extents of its rows on the Rust side: INFERENCE_CUDA_OK or INFERENCE_CUDA_INVALID_ARGUMENT.
static int gemma_check(const inference_cuda_attn_decode_gemma_args *args, bool extents) {
    if (args->out == NULL || args->q == NULL || args->k_codes == NULL ||
        args->k_scales == NULL || args->v_codes == NULL || args->v_scales == NULL ||
        args->kv_len == NULL)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (args->m < 1 || args->m > 16) return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (!(args->n_q_head == 16 && (args->n_kv_head == 1 || args->n_kv_head == 8))
        && !(args->n_q_head == 8 && args->n_kv_head == 2))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (args->head_dim != 256 && args->head_dim != 512)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (args->cells < 1 || args->window < 0 || args->window > args->cells)
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (args->ring < 0 || (args->ring > 0 && (args->window < 1 ||
                                              args->ring < args->window + args->m - 1 ||
                                              args->ring > args->cells)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (!(args->scale > -1e30f && args->scale < 1e30f))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (!std::isfinite(args->scale)) return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (!aligned(args->out, 4) || !aligned(args->q, 4) || !aligned(args->kv_len, 4) ||
        !aligned(args->k_scales, 2) || !aligned(args->v_scales, 2) ||
        (args->status != NULL && !aligned(args->status, 4)))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (!extents) return INFERENCE_CUDA_OK;  // the many-rows entry proves extents per row on the Rust side
    const size_t plane = (size_t)args->n_kv_head * args->cells;
    const size_t codes_bytes = plane * args->head_dim;
    const size_t scales_bytes = plane * (args->head_dim / 32) * 2;
    const size_t q_bytes = (size_t)args->m * args->n_q_head * args->head_dim * 4;
    const size_t out_bytes = q_bytes;
    const void *reads[] = {args->q, args->k_codes, args->k_scales,
                           args->v_codes, args->v_scales, args->kv_len};
    const size_t sizes[] = {q_bytes, codes_bytes, scales_bytes,
                            codes_bytes, scales_bytes, sizeof(int)};
    for (int i = 0; i < 6; ++i) {
        if (!apart(args->out, out_bytes, reads[i], sizes[i]) ||
            (args->status != NULL && !apart(args->status, 4, reads[i], sizes[i])))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if (args->status != NULL && !apart(args->out, out_bytes, args->status, 4))
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    if (args->split != 0) {
        if (args->part == NULL || args->max_splits < 1 || args->max_splits > kMaxSplits ||
            args->min_keys < 8 || !aligned(args->part, 16))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        const size_t part_bytes = (size_t)args->m * args->n_q_head *
                                  args->max_splits * (2 + args->head_dim) * 4;
        for (int i = 0; i < 6; ++i) {
            if (!apart(args->part, part_bytes, reads[i], sizes[i]))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!apart(args->out, out_bytes, args->part, part_bytes) ||
            (args->status != NULL && !apart(args->status, 4, args->part, part_bytes)))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    return INFERENCE_CUDA_OK;
}

extern "C" int inference_cuda_attn_decode_gemma_launch(
    const inference_cuda_attn_decode_gemma_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const int checked = gemma_check(args, true);
        if (checked != INFERENCE_CUDA_OK) return checked;
        Failure failure;
        const cudaStream_t s = (cudaStream_t)stream;
        if (args->split != 0) {
            const dim3 part_grid((unsigned)args->n_q_head, (unsigned)args->m,
                                 (unsigned)args->max_splits);
            attn_decode_gemma_part_k<><<<part_grid, 256, 0, s>>>(
                (float *)args->part, (unsigned int *)args->status, (const float *)args->q,
                (const signed char *)args->k_codes, (const unsigned short *)args->k_scales,
                (const signed char *)args->v_codes, (const unsigned short *)args->v_scales,
                (const int *)args->kv_len, args->n_q_head, args->n_kv_head,
                args->head_dim, args->cells, args->window, args->scale,
                args->max_splits, args->min_keys, args->ring);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            if (failure.ok()) {
                const dim3 comb_grid((unsigned)args->n_q_head, (unsigned)args->m);
                if (args->head_dim == 512) {
                    attn_decode_gemma_combine_k<512><<<comb_grid, 512, 0, s>>>(
                        (float *)args->out, (unsigned int *)args->status,
                        (const int *)args->kv_len, (const float *)args->part,
                        args->n_q_head, args->m, args->head_dim, args->cells,
                        args->window, args->max_splits, args->min_keys, args->ring);
                } else {
                    attn_decode_gemma_combine_k<256><<<comb_grid, 256, 0, s>>>(
                        (float *)args->out, (unsigned int *)args->status,
                        (const int *)args->kv_len, (const float *)args->part,
                        args->n_q_head, args->m, args->head_dim, args->cells,
                        args->window, args->max_splits, args->min_keys, args->ring);
                }
                failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            }
        } else {
            const dim3 grid((unsigned)args->n_q_head, (unsigned)args->m);
            attn_decode_gemma_kernel<><<<grid, 256, 0, s>>>(
                (float *)args->out, (unsigned int *)args->status, (const float *)args->q,
                (const signed char *)args->k_codes, (const unsigned short *)args->k_scales,
                (const signed char *)args->v_codes, (const unsigned short *)args->v_scales,
                (const int *)args->kv_len, args->n_q_head, args->n_kv_head,
                args->head_dim, args->cells, args->window, args->scale, args->ring);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

// The many-rows entry: n rows, each the single token of a sequence of its own (multi_row.h). The
// part scratch is laid out as for m = n tokens.
extern "C" int inference_cuda_attn_decode_gemma_multi_launch(
    const inference_cuda_attn_decode_gemma_multi_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == NULL || stream == NULL || in->rows == NULL || in->q == NULL || in->out == NULL ||
            in->n < 1 || in->n > INFERENCE_CUDA_MULTI_ROW_MAX || !aligned(in->rows, 8))
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        // The shared rules of the single entry on a stand-in of one token whose plane and kv_len
        // pointers are placeholders of the right alignment; the extents of the real rows are proved
        // on the Rust side.
        inference_cuda_attn_decode_gemma_args shape = {};
        shape.out = in->out;
        shape.q = in->q;
        shape.k_codes = shape.k_scales = shape.v_codes = shape.v_scales = shape.kv_len = in->rows;
        shape.m = 1;
        shape.n_q_head = in->n_q_head;
        shape.n_kv_head = in->n_kv_head;
        shape.head_dim = in->head_dim;
        shape.cells = in->cells;
        shape.window = in->window;
        shape.scale = in->scale;
        shape.part = (float *)in->part;
        shape.split = in->split;
        shape.max_splits = in->max_splits;
        shape.min_keys = in->min_keys;
        shape.ring = in->ring;
        const int checked = gemma_check(&shape, false);
        if (checked != INFERENCE_CUDA_OK) return checked;
        Failure failure;
        const cudaStream_t s = (cudaStream_t)stream;
        const unsigned n = (unsigned)in->n;
        if (in->split != 0) {
            attn_decode_gemma_part_k<true><<<dim3((unsigned)in->n_q_head, n, (unsigned)in->max_splits), 256, 0, s>>>(
                (float *)in->part, NULL, (const float *)in->q, NULL, NULL, NULL, NULL, NULL,
                in->n_q_head, in->n_kv_head, in->head_dim, in->cells, in->window, in->scale,
                in->max_splits, in->min_keys, in->ring, in->rows);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            if (failure.ok()) {
                const dim3 comb_grid((unsigned)in->n_q_head, n);
                if (in->head_dim == 512) {
                    attn_decode_gemma_combine_k<512, true><<<comb_grid, 512, 0, s>>>(
                        (float *)in->out, NULL, NULL, (const float *)in->part, in->n_q_head, in->n,
                        in->head_dim, in->cells, in->window, in->max_splits, in->min_keys, in->ring,
                        in->rows);
                } else {
                    attn_decode_gemma_combine_k<256, true><<<comb_grid, 256, 0, s>>>(
                        (float *)in->out, NULL, NULL, (const float *)in->part, in->n_q_head, in->n,
                        in->head_dim, in->cells, in->window, in->max_splits, in->min_keys, in->ring,
                        in->rows);
                }
                failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
            }
        } else {
            attn_decode_gemma_kernel<true><<<dim3((unsigned)in->n_q_head, n), 256, 0, s>>>(
                (float *)in->out, NULL, (const float *)in->q, NULL, NULL, NULL, NULL, NULL,
                in->n_q_head, in->n_kv_head, in->head_dim, in->cells, in->window, in->scale,
                in->ring, in->rows);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" size_t inference_cuda_attn_decode_gemma_part_floats(
    int m, int n_q_head, int head_dim, int max_splits) {
    if (m < 1 || n_q_head < 1 || head_dim < 32 || max_splits < 1) return 0;
    return (size_t)m * n_q_head * max_splits * (2 + head_dim);
}

extern "C" int inference_cuda_attn_decode_gemma_split_plan_raw(
    int n_keys, int max_splits, int min_keys, int *chunk, int *n_active) {
    /* n_keys 0 would leave ch 0 and divide by zero below; the kernels never
     * pass it (visible >= 1), so the export refuses it outright. */
    if (chunk == NULL || n_active == NULL || n_keys < 1 || max_splits < 1 ||
        max_splits > kMaxSplits || min_keys < 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    gemma_split_plan(n_keys, max_splits, min_keys, *chunk, *n_active);
    return 0;
}
