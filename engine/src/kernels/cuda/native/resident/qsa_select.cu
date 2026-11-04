// The block selection of Qwen3.8 Flash-Next QSA attention (see resident/qsa_select.h). One
// block of 1024 threads: a radix select over the f32 bits of the scores (a nonnegative f32
// orders as its bits) finds the take-th largest score T and how many scores equal to T are
// kept, then one ordered pass writes the blocks above T and the lowest-indexed blocks equal
// to T, so the result is the host's set and comes out in ascending order.
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/qsa_select.h"

namespace {

constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / 32;
constexpr int kMaxRatio = 64;
constexpr int kMaxCapBlocks = 1 << 24;
constexpr int kStatusBits = 4;  // INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of step_params.h
constexpr int kDigitBits = 8;
constexpr int kDigits = 1 << kDigitBits;

bool aligned(const void *p) { return (uintptr_t)p % 4 == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    if (a == NULL || b == NULL) return true;
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? a_bytes <= y - x : b_bytes <= x - y;
}

bool valid(const inference_cuda_qsa_select_args &a, void *stream) {
    if (stream == NULL || a.scores == NULL || a.blocks == NULL) return false;
    if (a.ratio < 1 || a.ratio > kMaxRatio || a.cap_blocks < 1 || a.cap_blocks > kMaxCapBlocks ||
        a.budget_blocks < 1 || a.budget_blocks > a.cap_blocks) {
        return false;
    }
    const void *ptrs[] = {a.scores, a.blocks, a.pos, a.status};
    for (const void *p : ptrs) {
        if (p != NULL && !aligned(p)) return false;
    }
    const size_t scores = (size_t)a.cap_blocks * 4, blocks = (size_t)a.budget_blocks * 4;
    return apart(a.blocks, blocks, a.scores, scores) && apart(a.blocks, blocks, a.pos, 4) &&
           apart(a.blocks, blocks, a.status, 4) && apart(a.status, 4, a.scores, scores) &&
           apart(a.status, 4, a.pos, 4);
}

__device__ __forceinline__ unsigned key_of(float score) {
    const unsigned bits = __float_as_uint(score);
    // Negative values and NaN (sign bit or exponent all ones with a payload) count as +0.
    return (bits >> 31) != 0 || bits > 0x7f800000u ? 0u : bits;
}

/* The count of `flag` over the threads before this one, and the block total in *total.
 * Every thread of the block calls it; it synchronizes twice. */
__device__ int exclusive_count(bool flag, int *total) {
    __shared__ int warp_count[kWarps];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    const int in_warp = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) warp_count[warp] = __popc(ballot);
    __syncthreads();
    int before = 0, all = 0;
    for (int w = 0; w < kWarps; ++w) {
        before += w < warp ? warp_count[w] : 0;
        all += warp_count[w];
    }
    __syncthreads();
    *total = all;
    return before + in_warp;
}

__global__ void __launch_bounds__(kThreads)
    qsa_select_k(const float *scores, int *out, const int *pos, int *status, int ratio,
                 int cap_blocks, int budget_blocks, int pos0) {
    __shared__ unsigned hist[kDigits];
    __shared__ unsigned s_prefix;
    __shared__ int s_need;
    const int tid = threadIdx.x;
    const long long at = pos != NULL ? (long long)*pos : (long long)pos0;
    if (at < 0 || at + 1 > (long long)cap_blocks * ratio) {
        if (status != NULL && tid == 0) atomicOr(status, kStatusBits);
        return;
    }
    const int blocks = (int)((at + 1) / ratio);
    const int take = blocks < budget_blocks ? blocks : budget_blocks;
    if (take == 0) return;
    // Radix select, most significant digit first: after the last pass `prefix` is T and `need`
    // is the number of scores equal to T inside the top `take`.
    unsigned prefix = 0, mask = 0;
    int need = take;
    for (int shift = 32 - kDigitBits; shift >= 0; shift -= kDigitBits) {
        for (int d = tid; d < kDigits; d += kThreads) hist[d] = 0;
        __syncthreads();
        for (int i = tid; i < blocks; i += kThreads) {
            const unsigned k = key_of(scores[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & (kDigits - 1)], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int above = 0;
            for (int d = kDigits - 1; d >= 0; --d) {
                const int here = (int)hist[d];
                if (above + here >= need) {
                    s_prefix = prefix | ((unsigned)d << shift);
                    s_need = need - above;
                    break;
                }
                above += here;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask |= (unsigned)(kDigits - 1) << shift;
    }
    // Ordered compaction: every block above T, and the first `need` blocks equal to T.
    int written = 0, equal_seen = 0;
    for (int chunk = 0; chunk < blocks; chunk += kThreads) {
        const int i = chunk + tid;
        const unsigned k = i < blocks ? key_of(scores[i]) : 0u;
        const bool above = i < blocks && k > prefix;
        const bool equal = i < blocks && k == prefix;
        int equal_total = 0, chosen_total = 0;
        const int equal_before = exclusive_count(equal, &equal_total);
        const bool chosen = above || (equal && equal_seen + equal_before < need);
        const int chosen_before = exclusive_count(chosen, &chosen_total);
        if (chosen) out[written + chosen_before] = i;
        written += chosen_total;
        equal_seen += equal_total;
    }
}

}  // namespace

extern "C" int inference_cuda_qsa_select_launch(const inference_cuda_qsa_select_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const inference_cuda_qsa_select_args &a = *args;
        Failure failure;
        qsa_select_k<<<1, kThreads, 0, (cudaStream_t)stream>>>(
            (const float *)a.scores, (int *)a.blocks, (const int *)a.pos, (int *)a.status, a.ratio,
            a.cap_blocks, a.budget_blocks, a.pos0);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
