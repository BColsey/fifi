// Launch shim for the vendored conv_silu_k (kern::conv_silu in ops.cu). No kernel is added;
// this file only proves the arguments the kernel does not check and makes one extern "C" call.
// The kernel is vendored xavier-qwen code.
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "ops.h"
#include "resident/gdn_conv.h"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxChannels = 65536;
constexpr int kMaxLd = 1 << 20;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

// True when [a, a + a_bytes) and [b, b + b_bytes) share no byte.
bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t pa = (uintptr_t)a, pb = (uintptr_t)b;
    return pa + a_bytes <= pb || pb + b_bytes <= pa;
}

// Mirrors the Rust wrapper term by term; a term added here needs its twin there. The Rust side
// refuses everything this refuses, so a C refusal means a wrapper defect.
bool valid(const inference_cuda_gdn_conv_args &a, void *stream) {
    if (a.proj == NULL || a.w == NULL || a.state == NULL || a.out == NULL || stream == NULL) {
        return false;
    }
    if (a.m < 1 || a.m > kMaxM || a.channels < 1 || a.channels > kMaxChannels ||
        a.ld < a.channels || a.ld > kMaxLd) {
        return false;
    }
    if (!aligned(a.proj, 4) || !aligned(a.state, 4) || !aligned(a.out, 4) || !aligned(a.w, 16)) {
        return false;
    }
    const size_t n = (size_t)a.channels;
    const size_t proj_bytes = (((size_t)a.m - 1) * (size_t)a.ld + n) * sizeof(float);
    const size_t w_bytes = n * 4 * sizeof(float);
    const size_t state_bytes = 3 * n * sizeof(float);
    const size_t out_bytes = (size_t)a.m * n * sizeof(float);
    // state and out are written; proj and w are read. Reads may share bytes with each other.
    return apart(a.state, state_bytes, a.out, out_bytes) &&
           apart(a.state, state_bytes, a.proj, proj_bytes) &&
           apart(a.state, state_bytes, a.w, w_bytes) && apart(a.out, out_bytes, a.proj, proj_bytes) &&
           apart(a.out, out_bytes, a.w, w_bytes);
}

// The snapshot range of the strided entry: [snap, snap + snap_bytes) shares no byte with any range
// the kernel reads or writes, and holds the m - 1 snapshots of 12 * channels bytes at snap_stride.
bool valid_snap(const inference_cuda_gdn_conv_snap_args &s) {
    const inference_cuda_gdn_conv_args &a = s.conv;
    const size_t n = (size_t)a.channels;
    const size_t row = 3 * n * sizeof(float);
    if (s.snap == NULL || a.m < 2 || !aligned(s.snap, 4) || s.snap_stride % 4 != 0 ||
        s.snap_stride < row) {
        return false;
    }
    const size_t reach = ((size_t)a.m - 2) * s.snap_stride + row;
    if (s.snap_bytes < reach) {
        return false;
    }
    const size_t proj_bytes = (((size_t)a.m - 1) * (size_t)a.ld + n) * sizeof(float);
    return apart(s.snap, reach, a.state, row) && apart(s.snap, reach, a.out, (size_t)a.m * n * sizeof(float)) &&
           apart(s.snap, reach, a.proj, proj_bytes) && apart(s.snap, reach, a.w, n * 4 * sizeof(float));
}

}  // namespace

extern "C" int inference_cuda_gdn_conv_launch(const inference_cuda_gdn_conv_args *args,
                                              void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        // snap stays null: this entry does not use the speculative-rollback snapshots.
        kern::ConvArgs call = {(const float *)args->proj, args->ld,   args->m,
                               args->channels,            (const float *)args->w,
                               (float *)args->state,      NULL,       (float *)args->out};
        Failure failure;
        failure.record(kern::conv_silu(call, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gdn_conv_snap_launch(const inference_cuda_gdn_conv_snap_args *args,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(args->conv, stream) || !valid_snap(*args)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const inference_cuda_gdn_conv_args &a = args->conv;
        kern::ConvArgs call = {(const float *)a.proj, a.ld,  a.m,
                               a.channels,            (const float *)a.w,
                               (float *)a.state,      (float *)args->snap, (float *)a.out,
                               args->snap_stride};
        Failure failure;
        failure.record(kern::conv_silu(call, (cudaStream_t)stream), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
