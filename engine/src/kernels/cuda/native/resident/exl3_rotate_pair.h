/* The gate and up input rotation of the EXL3 job-table projection in one pass. Included by exl3.cu
 * inside its unnamed namespace, after exl3_rotate_wide.h; it declares no namespace of its own. */
#pragma once

#include "exl3_rotate_butterfly.h"

// Two projections that read the same rows (the gate and the up matrix of the routed experts)
// rotate the same x by their own scales. This kernel reads each x group once and writes both
// rotated rows: xh of call `a` with the scales of `a`, xh of call `b` with the scales of `b`.
// Each output is computed by wide_rotate_group, the arithmetic of the separate rotation of that
// call, so both are byte-identical to two separate calls. The two calls share x, the row list,
// the table, the shape and the flag (jobs_pair_consistent).
enum PairMutant {
    pair_exact = 0,
    pair_swapped_outputs,    // each rotation lands in the other call's xh
    pair_first_scales_twice,  // the second output is rotated by the first call's scales
    pair_drop_b_flag,         // the non-finite values of the second output do not flag the call
};

// The facts two calls must share to be rotated together: the same input rows read the same way
// through the same table, and a flag to report to, with the two outputs apart.
bool jobs_pair_consistent(const inference_cuda_exl3_jobs_args &a,
                          const inference_cuda_exl3_jobs_args &b) {
    if (!(a.x == b.x && a.x_rows == b.x_rows && a.in_rows == b.in_rows && a.jobs == b.jobs &&
          a.jobs_count == b.jobs_count && a.max_rows == b.max_rows &&
          a.total_rows == b.total_rows && a.k == b.k && a.members == b.members &&
          a.flag == b.flag && a.xh != b.xh && apart(a.xh, a.xh_bytes, b.xh, b.xh_bytes)))
        return false;
    // Both outputs are written before either call projects: apart from everything the rotation
    // reads.
    for (const uint16_t *xh : {a.xh, b.xh}) {
        const size_t bytes = xh == a.xh ? a.xh_bytes : b.xh_bytes;
        if (!apart(xh, bytes, a.x, a.x_bytes) || !apart(xh, bytes, a.suh, a.suh_bytes) ||
            !apart(xh, bytes, b.suh, b.suh_bytes) || !apart(xh, bytes, a.jobs, a.jobs_bytes) ||
            (a.in_rows && !apart(xh, bytes, a.in_rows, a.in_rows_bytes)))
            return false;
    }
    return true;
}

// Whether the pair rotation can run: consistent calls, and both meet the alignment rules of the
// wide kernel.
bool jobs_pair_usable(const inference_cuda_exl3_jobs_args &a,
                      const inference_cuda_exl3_jobs_args &b) {
    return jobs_pair_consistent(a, b) && jobs_wide_input_usable(a, true) &&
        jobs_wide_input_usable(b, true);
}

// Grid and groups as jobs_input_rotate_wide; one flag (a.flag == b.flag).
template <int Mutant>
__global__ void __launch_bounds__(wide_warps * 32)
jobs_input_rotate_pair(const __grid_constant__ inference_cuda_exl3_jobs_args a,
                       const __grid_constant__ inference_cuda_exl3_jobs_args b) {
    const unsigned lane = threadIdx.x & 31u;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < a.jobs_count;
         index += size_t(gridDim.x) * blockDim.x) {
        if (!job_in_bounds(a, a.jobs[index])) atomicOr(a.flag, 1u);
    }
    __syncwarp();
    const uint32_t per_row = uint32_t(a.k / 128);
    const uint32_t total = uint32_t(a.total_rows) * per_row;
    const uint32_t first = (blockIdx.x * wide_warps + threadIdx.x / 32) * wide_groups;
    if (first >= total) return;
    uint32_t row = first / per_row, in_row = first - row * per_row;

    const float *xs[wide_groups], *suh_a[wide_groups], *suh_b[wide_groups];
    uint2 *dst_a[wide_groups], *dst_b[wide_groups];
    bool live[wide_groups];
    WideOwner owner{-1, 1, 0};
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        const uint32_t group = first + u;
        live[u] = group < total;
        if (!live[u]) continue;
        if (u > 0 && ++in_row == per_row) { in_row = 0; ++row; }
        const size_t column = size_t(in_row) * 128 + 4 * lane;
        if (row < owner.lo || row >= owner.hi) owner = wide_find_owner(a, row);
        live[u] = owner.member >= 0;
        if (!live[u]) continue;
        const size_t source = a.in_rows ? size_t(a.in_rows[row]) : row;
        const bool inside = source < a.x_rows;
        if (!inside && lane == 0) atomicOr(a.flag, 1u);
        xs[u] = inside ? a.x + source * a.k + column : nullptr;
        suh_a[u] = inference_cuda_exl3_member_suh(a, owner.member) + column;
        suh_b[u] = inference_cuda_exl3_member_suh(Mutant == pair_first_scales_twice ? a : b,
                                                  owner.member) + column;
        dst_a[u] = reinterpret_cast<uint2 *>((Mutant == pair_swapped_outputs ? b.xh : a.xh) +
                                             size_t(row) * a.k + column);
        dst_b[u] = reinterpret_cast<uint2 *>((Mutant == pair_swapped_outputs ? a.xh : b.xh) +
                                             size_t(row) * a.k + column);
    }

    float4 x[wide_groups], s_a[wide_groups], s_b[wide_groups];
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        x[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        s_a[u] = x[u];
        s_b[u] = x[u];
        if (!live[u]) continue;
        s_a[u] = __ldg(reinterpret_cast<const float4 *>(suh_a[u]));
        s_b[u] = __ldg(reinterpret_cast<const float4 *>(suh_b[u]));
        if (xs[u]) x[u] = __ldg(reinterpret_cast<const float4 *>(xs[u]));
    }

    bool non_finite = false, non_finite_b = false;
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        if (!live[u]) continue;
        *dst_a[u] = wide_rotate_group<wide_exact>(x[u], s_a[u], lane, non_finite);
        *dst_b[u] = wide_rotate_group<wide_exact>(x[u], s_b[u], lane, non_finite_b);
    }
    if (Mutant != pair_drop_b_flag) non_finite = non_finite || non_finite_b;
    if (__any_sync(0xffffffffu, non_finite) && lane == 0) atomicOr(a.flag, 1u);
}

template <int Mutant>
void launch_pair_input(const inference_cuda_exl3_jobs_args &a,
                       const inference_cuda_exl3_jobs_args &b, cudaStream_t s) {
    const size_t groups = a.total_rows * (a.k / 128);
    const size_t warps = (groups + wide_groups - 1) / wide_groups;
    jobs_input_rotate_pair<Mutant><<<dim3(unsigned((warps + wide_warps - 1) / wide_warps)),
                                     wide_warps * 32, 0, s>>>(a, b);
}
