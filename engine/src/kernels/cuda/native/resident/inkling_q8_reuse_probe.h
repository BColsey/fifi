// Private component interface, compiled only by tools/inkling-q8-row-reuse.
#pragma once

#include "resident/dense_gemv.h"

struct InklingQ8ReusePolicy {
    bool enabled = false;
    int compute_capability = 0;  // obtained once for the current CUDA device

    constexpr bool use_pair(const inference_cuda_dense_gemv_args &a) const {
        // The shape envelope is deliberately limited to the three measured bins.
        if (!enabled || compute_capability != 75 || a.weight_dtype != 256 + 8 ||
            a.m < 2 || a.m > 64) return false;
        return (a.cols == 2304 && (a.rows == 1024 || a.rows == 3072)) ||
               (a.cols == 3072 && a.rows == 2304);
    }
};

struct InklingQ8ReuseLaunch {
    int stage = 0;
    bool used_pair = false;
};

// Same checked operand/stream contract as dense GEMV. Unselected valid calls
// dispatch ordinary GEMV; launch failures are returned rather than retried.
int inkling_q8_reuse_probe_launch(const inference_cuda_dense_gemv_args *args, void *stream,
                                 InklingQ8ReusePolicy policy, InklingQ8ReuseLaunch &out);
