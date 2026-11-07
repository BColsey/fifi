/*
 * pack_rows: the bit-exact copy of up to three tight [m][n_i] f32 sources into the [m][ld] row
 * layout that the GDN delta rule and the attention kernels read (see pack_rows.h). One launch per
 * layer for z (or for q, k and v) of a K-row verify window. The code is original CUDA C for sm_86.
 *
 * Every element moves as one 32-bit word through an unsigned load and store, so no float
 * instruction touches it. The result does not depend on the order of the threads (every
 * destination word has exactly one source word), so there is nothing to make deterministic.
 */

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/pack_rows.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxRows = 16;
constexpr int kMaxSegs = 3;
constexpr int kMaxN = 65536;
constexpr int kMaxLd = 1 << 20;

struct Seg {
    const unsigned *src;
    int n;
    int dst_off;
};

struct PackArgs {
    unsigned *dst;
    Seg seg[kMaxSegs];
    int ld;
};

/* Grid (ceil(max n / kThreads), m, segs): block (x, row, s) copies kThreads columns of segment s of
 * row `row`. */
__global__ void __launch_bounds__(kThreads) pack_rows_k(const __grid_constant__ PackArgs a) {
    const Seg s = a.seg[blockIdx.z];
    const int c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= s.n) return;
    const size_t row = blockIdx.y;
    a.dst[row * (size_t)a.ld + (size_t)s.dst_off + (size_t)c] = s.src[row * (size_t)s.n + (size_t)c];
}

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + a_bytes <= pb || pb + b_bytes <= pa;
}

/* Mirrors the Rust wrapper term by term; a term added here needs its twin there. */
bool valid(const inference_cuda_pack_rows_args *a, void *stream) {
    if (a == NULL || a->dst == NULL || stream == NULL || a->m < 1 || a->m > kMaxRows || a->segs < 1 ||
        a->segs > kMaxSegs || a->ld < 1 || a->ld > kMaxLd || !aligned(a->dst, 4)) {
        return false;
    }
    size_t reach = 0;  // the columns of one dst row that the segments span
    for (int i = 0; i < a->segs; ++i) {
        const inference_cuda_pack_segment &s = a->seg[i];
        if (s.src == NULL || s.n < 1 || s.n > kMaxN || s.dst_off < 0 || (long)s.dst_off + s.n > a->ld ||
            !aligned(s.src, 4)) {
            return false;
        }
        reach = (size_t)(s.dst_off + s.n) > reach ? (size_t)(s.dst_off + s.n) : reach;
    }
    const size_t dst_bytes = (((size_t)a->m - 1) * (size_t)a->ld + reach) * sizeof(float);
    for (int i = 0; i < a->segs; ++i) {
        const inference_cuda_pack_segment &s = a->seg[i];
        if (!apart(s.src, (size_t)a->m * (size_t)s.n * sizeof(float), a->dst, dst_bytes)) return false;
        for (int j = i + 1; j < a->segs; ++j) {
            const inference_cuda_pack_segment &t = a->seg[j];
            const bool columns_apart = s.dst_off + s.n <= t.dst_off || t.dst_off + t.n <= s.dst_off;
            if (!columns_apart) return false;
        }
    }
    return true;
}

}  // namespace

extern "C" int inference_cuda_pack_rows_launch(const inference_cuda_pack_rows_args *args, void *stream,
                                               int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid(args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        PackArgs p = {};
        p.dst = (unsigned *)args->dst;
        p.ld = args->ld;
        int max_n = 1;
        for (int i = 0; i < args->segs; ++i) {
            p.seg[i] = {(const unsigned *)args->seg[i].src, args->seg[i].n, args->seg[i].dst_off};
            max_n = args->seg[i].n > max_n ? args->seg[i].n : max_n;
        }
        const dim3 grid((max_n + kThreads - 1) / kThreads, args->m, args->segs);
        Failure failure;
        pack_rows_k<<<grid, kThreads, 0, (cudaStream_t)stream>>>(p);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
