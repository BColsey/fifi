#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/shortlist.h"

namespace {
using Candidate = inference_cuda_shortlist_candidate;
using Part = inference_cuda_shortlist_result;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kTiles = INFERENCE_CUDA_SHORTLIST_TILES;
constexpr int kK = INFERENCE_CUDA_SHORTLIST_K;
constexpr size_t kScratch = INFERENCE_CUDA_SHORTLIST_SCRATCH_BYTES;

__device__ Candidate empty_candidate() { return {-INFINITY, UINT32_MAX}; }

__device__ bool better(Candidate a, Candidate b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

__device__ Candidate warp_best(Candidate best) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const Candidate other{__shfl_xor_sync(0xffffffffu, best.logit, offset),
                              __shfl_xor_sync(0xffffffffu, best.id, offset)};
        if (better(other, best)) best = other;
    }
    return best;
}

__device__ Candidate block_best(Candidate best, Candidate *warps, Candidate *chosen) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    best = warp_best(best);
    if (lane == 0) warps[warp] = best;
    __syncthreads();
    if (warp == 0) {
        best = warp_best(lane < kWarps ? warps[lane] : empty_candidate());
        if (lane == 0) *chosen = best;
    }
    __syncthreads();
    return *chosen;
}

__global__ void shortlist_parts_k(const float *logits, int n, Part *parts) {
    __shared__ Candidate warps[kWarps], chosen;
    __shared__ unsigned warp_bad[kWarps];
    const size_t first = size_t(n) * blockIdx.x / kTiles;
    const size_t end = size_t(n) * (blockIdx.x + 1) / kTiles;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Candidate cutoff{INFINITY, 0};
    bool bad = false;
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (size_t index = first + threadIdx.x; index < end; index += kThreads) {
            const Candidate value{logits[index], unsigned(index)};
            if (!isfinite(value.logit)) {
                bad = true;
            } else if (better(cutoff, value) && better(value, best)) {
                best = value;
            }
        }
        if (rank == 0) {
            const unsigned any_bad = __any_sync(0xffffffffu, bad);
            if (lane == 0) warp_bad[warp] = any_bad;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) parts[blockIdx.x].entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned any_bad = 0;
#pragma unroll
        for (int i = 0; i < kWarps; ++i) any_bad |= warp_bad[i];
        parts[blockIdx.x].non_finite = any_bad;
        for (int i = 0; i < 3; ++i) parts[blockIdx.x].pad[i] = 0;
    }
}

__global__ void shortlist_final_k(Part *parts) {
    __shared__ Candidate warps[kWarps], chosen;
    Candidate cutoff{INFINITY, 0};
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (int index = threadIdx.x; index < kTiles * kK; index += kThreads) {
            const Candidate value = parts[index / kK].entries[index % kK];
            if (value.id != UINT32_MAX && better(cutoff, value) && better(value, best)) best = value;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) parts[kTiles].entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned bad = 0;
#pragma unroll
        for (int i = 0; i < kTiles; ++i) bad |= parts[i].non_finite;
        parts[kTiles].non_finite = bad;
        for (int i = 0; i < 3; ++i) parts[kTiles].pad[i] = 0;
    }
}

/* The K-row shortlist: row r of the m logits rows (row r starts r * ld floats after the first) gets
 * the record that shortlist_parts_k and shortlist_final_k give it as a single-row launch. The loops
 * are those of the two kernels above with the row taken from the block index; the selection is made
 * of comparisons only, so the record of a row is a function of that row alone. Differences, on
 * purpose: the partials of row r live at scratch + r * scratch_stride, and the final record goes to
 * out + r * sizeof(Part) instead of the last part of the scratch, so the m records are contiguous
 * for one device-to-host read. */
__global__ void shortlist_parts_rows_k(const float *logits, long long ld, int n, char *scratch,
                                       long long stride) {
    __shared__ Candidate warps[kWarps], chosen;
    __shared__ unsigned warp_bad[kWarps];
    const float *row_logits = logits + (size_t)blockIdx.y * (size_t)ld;
    Part *parts = reinterpret_cast<Part *>(scratch + (size_t)blockIdx.y * (size_t)stride);
    const size_t first = size_t(n) * blockIdx.x / kTiles;
    const size_t end = size_t(n) * (blockIdx.x + 1) / kTiles;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    Candidate cutoff{INFINITY, 0};
    bool bad = false;
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (size_t index = first + threadIdx.x; index < end; index += kThreads) {
            const Candidate value{row_logits[index], unsigned(index)};
            if (!isfinite(value.logit)) {
                bad = true;
            } else if (better(cutoff, value) && better(value, best)) {
                best = value;
            }
        }
        if (rank == 0) {
            const unsigned any_bad = __any_sync(0xffffffffu, bad);
            if (lane == 0) warp_bad[warp] = any_bad;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) parts[blockIdx.x].entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned any_bad = 0;
#pragma unroll
        for (int i = 0; i < kWarps; ++i) any_bad |= warp_bad[i];
        parts[blockIdx.x].non_finite = any_bad;
        for (int i = 0; i < 3; ++i) parts[blockIdx.x].pad[i] = 0;
    }
}

__global__ void shortlist_final_rows_k(const char *scratch, long long stride, Part *out) {
    __shared__ Candidate warps[kWarps], chosen;
    const Part *parts = reinterpret_cast<const Part *>(scratch + (size_t)blockIdx.x * (size_t)stride);
    Part *result = out + blockIdx.x;
    Candidate cutoff{INFINITY, 0};
#pragma unroll 1
    for (int rank = 0; rank < kK; ++rank) {
        Candidate best = empty_candidate();
        for (int index = threadIdx.x; index < kTiles * kK; index += kThreads) {
            const Candidate value = parts[index / kK].entries[index % kK];
            if (value.id != UINT32_MAX && better(cutoff, value) && better(value, best)) best = value;
        }
        cutoff = block_best(best, warps, &chosen);
        if (threadIdx.x == 0) result->entries[rank] = cutoff;
    }
    if (threadIdx.x == 0) {
        unsigned bad = 0;
#pragma unroll
        for (int i = 0; i < kTiles; ++i) bad |= parts[i].non_finite;
        result->non_finite = bad;
        for (int i = 0; i < 3; ++i) result->pad[i] = 0;
    }
}

bool valid(const inference_cuda_shortlist_args *a, void *stream) {
    if (!a || !stream || !a->logits || !a->scratch || a->n < 1 || a->n > (1 << 30)) return false;
    const uintptr_t x = uintptr_t(a->logits), s = uintptr_t(a->scratch);
    if (x % 4 || s % 16) return false;
    const size_t bytes = size_t(a->n) * 4;
    return s <= x ? kScratch <= x - s : bytes <= s - x;
}

bool valid_rows(const inference_cuda_shortlist_rows_args *a, void *stream) {
    if (!a || !stream || !a->logits || !a->scratch || !a->out || a->n < 1 || a->n > (1 << 30) ||
        a->m < 1 || a->m > INFERENCE_CUDA_SHORTLIST_MAX_ROWS || a->ld < a->n || a->ld > (1LL << 40) ||
        a->scratch_stride < (long long)(kTiles * sizeof(Part)) || a->scratch_stride % 16) {
        return false;
    }
    const uintptr_t x = uintptr_t(a->logits), s = uintptr_t(a->scratch), o = uintptr_t(a->out);
    if (x % 4 || s % 16 || o % 16) return false;
    const uintptr_t x_end = x + (uintptr_t(a->m - 1) * uintptr_t(a->ld) + uintptr_t(a->n)) * 4;
    const uintptr_t s_end = s + uintptr_t(a->m) * uintptr_t(a->scratch_stride);
    const uintptr_t o_end = o + uintptr_t(a->m) * sizeof(Part);
    const bool x_s = x_end <= s || s_end <= x;
    const bool x_o = x_end <= o || o_end <= x;
    const bool s_o = s_end <= o || o_end <= s;
    return x_s && x_o && s_o;
}
}

extern "C" int inference_cuda_shortlist_launch(const inference_cuda_shortlist_args *a,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(a, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        auto parts = static_cast<Part *>(a->scratch);
        shortlist_parts_k<<<kTiles, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<const float *>(a->logits), a->n, parts);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            shortlist_final_k<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(parts);
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_shortlist_rows_launch(const inference_cuda_shortlist_rows_args *a,
                                                    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_rows(a, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        shortlist_parts_rows_k<<<dim3(kTiles, a->m), kThreads, 0, s>>>(
            static_cast<const float *>(a->logits), a->ld, a->n, static_cast<char *>(a->scratch),
            a->scratch_stride);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok()) {
            shortlist_final_rows_k<<<a->m, kThreads, 0, s>>>(static_cast<const char *>(a->scratch),
                                                             a->scratch_stride,
                                                             static_cast<Part *>(a->out));
            failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        }
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
