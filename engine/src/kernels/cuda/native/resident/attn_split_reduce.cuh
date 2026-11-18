/*
 * The two-phase merge of split-KV attention partials, shared by the
 * split-combine kernels of the resident decode paths.
 *
 * Both kernels store, per (token, head-group, split), a (max, sum) pair and a
 * weighted-V accumulator vector; every split block writes exactly one such
 * partial and the combine merges the n_active splits of the visible range in
 * split-index order with no float atomic (see attn_decode_q8.cu's header).
 *
 * Phase order (identical float operations in both users):
 *   1. attn_split_max: block max over the active splits' maxima, `-inf` splits
 *      excluded naturally by fmaxf; thread d < n_active also stages the split
 *      sums into s_l. Returns the split max only; callers fold any extra term
 *      (attn_decode_q8's attention sink) into the returned max themselves.
 *   2. the caller computes the weights s_w[d] = expf(ml[d*ml_step] - M) for
 *      d < n_active and syncs.
 *   3. attn_split_sum: denominator L = l_init + sum_s l_s*w_s and accumulator
 *      A = sum_s acc_s*w_s, both in split-index order, l_init first.
 *
 * The extra term one caller folds around these phases stays in that caller:
 * attn_decode_q8 folds its attention sink into the max that phase 1 returned
 * and passes expf(sink - M) as `l_init` of phase 3, which reproduces that
 * kernel's original summation order exactly.
 *
 * Shared arrays are caller-owned because their sizes depend on max_splits and
 * the thread count: s_w[n_active_max], s_l[n_active_max], s_red[D/32].
 */
#ifndef INFERENCE_CUDA_RESIDENT_ATTN_SPLIT_REDUCE_CUH
#define INFERENCE_CUDA_RESIDENT_ATTN_SPLIT_REDUCE_CUH

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace inference {

// Phase 1. `d` is threadIdx.x of a D-thread block, one output element each.
// Returns the block max M of the active split maxima; the per-split sums
// land in `s_l`.
template <int D>
__device__ inline float attn_split_max(
    int d, int n_active, const float *ml, size_t ml_step,
    float *s_l, float *s_red) {
    float ms = -INFINITY;
    if (d < n_active) {
        ms = ml[d * ml_step];
        s_l[d] = ml[d * ml_step + 1];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ms = fmaxf(ms, __shfl_xor_sync(0xffffffffu, ms, o));
    if ((d & 31) == 0) s_red[d >> 5] = ms;
    __syncthreads();
    float m = s_red[0];
#pragma unroll
    for (int i = 1; i < D / 32; ++i) {
        m = fmaxf(m, s_red[i]);
    }
    return m;
}

// Phase 3. Denominator and one accumulator element, split-index order, l_init
// first. The accumulator walk keeps attn_decode_q8's `#pragma unroll 16`.
__device__ inline void attn_split_sum(
    int n_active, float l_init, const float *s_w, const float *s_l, const float *ac,
    size_t ac_step, float &l, float &a) {
    l = l_init;
    for (int s = 0; s < n_active; ++s) {
        l = fmaf(s_l[s], s_w[s], l);
    }
    a = 0.f;  // index order; the loads do not depend on the weights, so they overlap
#pragma unroll 16
    for (int s = 0; s < n_active; ++s) {
        a = fmaf(ac[s * ac_step], s_w[s], a);
    }
}

}  // namespace inference

#endif /* INFERENCE_CUDA_RESIDENT_ATTN_SPLIT_REDUCE_CUH */
