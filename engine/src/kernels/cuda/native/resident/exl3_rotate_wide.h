/* The warp-wide input rotation of the EXL3 job-table projection. Included by exl3.cu inside its
 * unnamed namespace, after job_in_bounds; it declares no namespace of its own. */
#pragma once

// ---- the warp-wide input rotation ------------------------------------------------------------
// jobs_input_rotate with one warp per 128-value group instead of one thread per value: a lane
// holds four consecutive values, the first two butterfly steps (distance 1 and 2) run in its
// registers, the other five (distance 4 to 64) are lane exchanges, so no step waits at a block
// barrier, and a warp keeps the loads of several groups in flight. The grid covers the rows of
// the table (total_rows), and the job that owns a row is found on the device, so a short job
// leaves no idle block behind.
//
// Every value takes the arithmetic of jobs_input_rotate in the same order: the F32 product of
// x and suh, the non-finite check, the seven steps (the upper partner computes other - own, the
// lower own + other), the F32 scale by rsqrtf(128), the non-finite check, one round to F16. The
// products and sums use the _rn intrinsics, which the compiler never fuses: a contracted
// x * suh + y rounds once instead of twice and changes bits.
constexpr unsigned wide_groups = 4;  // 128-value groups one warp rotates per pass
constexpr unsigned wide_warps = 4;   // warps of a block

// Deliberately wrong variants, built only for the mutation tests (the feature that defines
// INFERENCE_CUDA_EXL3_ROTATE_MUTANTS); the production instantiation is wide_exact.
enum WideMutant {
    wide_exact = 0,
    wide_reversed_butterfly,  // the seven steps in the opposite order
    wide_drop_suh,            // the multiply by suh is skipped
    wide_contract,            // the first butterfly step takes fused multiply-adds
    wide_gather_row,          // the row list is read at the neighbouring row
    wide_tail,                // the last group of the table is dropped
};

// The rows [lo, hi) of the table that one job owns; member < 0 when no job does.
struct WideOwner {
    int32_t member;
    uint32_t lo, hi;
};

__device__ __forceinline__ bool job_holds(const inference_cuda_exl3_jobs_args &p,
                                          const inference_cuda_exl3_job &job, uint32_t row) {
    return job_in_bounds(p, job) && uint32_t(job.row0) <= row &&
        row < uint32_t(job.row0) + uint32_t(job.rows);
}

// A table lists its jobs back to back in row order, so the owner of a row is the last entry that
// starts at or before it, found by bisection. A table in another order, or a row that no job
// holds, is settled by a scan of every entry: the result is the same job the one-block-per-job
// kernel would have run for the row, or none.
__device__ WideOwner wide_find_owner(const inference_cuda_exl3_jobs_args &p, uint32_t row) {
    uint32_t low = 0, high = p.jobs_count;
    while (low < high) {
        const uint32_t middle = (low + high) / 2;
        if (p.jobs[middle].row0 <= int32_t(row)) low = middle + 1; else high = middle;
    }
    if (low > 0) {
        const inference_cuda_exl3_job job = p.jobs[low - 1];
        if (job_holds(p, job, row))
            return {job.member, uint32_t(job.row0), uint32_t(job.row0) + uint32_t(job.rows)};
    }
    for (uint32_t index = 0; index < p.jobs_count; ++index) {
        const inference_cuda_exl3_job job = p.jobs[index];
        if (job_holds(p, job, row))
            return {job.member, uint32_t(job.row0), uint32_t(job.row0) + uint32_t(job.rows)};
    }
    return {-1, row, row + 1};
}

// One butterfly step across lanes: the lane whose bit is set computes other - own, which is the
// sum of other and the negated own value (a subtraction is that sum, bit for bit), so one add
// serves both partners after the sign of own is flipped on the upper one.
template <unsigned Bit>
__device__ __forceinline__ void wide_lane_step(float (&v)[4], unsigned lane) {
    const unsigned flip = (lane & (1u << Bit)) ? 0x80000000u : 0u;
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        const float own = v[i], other = __shfl_xor_sync(0xffffffffu, own, 1u << Bit);
        v[i] = __fadd_rn(__uint_as_float(__float_as_uint(own) ^ flip), other);
    }
}
// Steps at distance 1 and 2 over the four values of a lane.
__device__ __forceinline__ void wide_step_one(float (&v)[4]) {
    const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = __fadd_rn(a, b);
    v[1] = __fsub_rn(a, b);
    v[2] = __fadd_rn(c, d);
    v[3] = __fsub_rn(c, d);
}
__device__ __forceinline__ void wide_step_two(float (&v)[4]) {
    const float a = v[0], b = v[1], c = v[2], d = v[3];
    v[0] = __fadd_rn(a, c);
    v[2] = __fsub_rn(a, c);
    v[1] = __fadd_rn(b, d);
    v[3] = __fsub_rn(b, d);
}

template <int Mutant>
__device__ __forceinline__ void wide_butterfly(float (&v)[4], unsigned lane) {
    if constexpr (Mutant == wide_reversed_butterfly) {
        wide_lane_step<4>(v, lane);
        wide_lane_step<3>(v, lane);
        wide_lane_step<2>(v, lane);
        wide_lane_step<1>(v, lane);
        wide_lane_step<0>(v, lane);
        wide_step_two(v);
        wide_step_one(v);
    } else {
        if constexpr (Mutant != wide_contract) wide_step_one(v);
        wide_step_two(v);
        wide_lane_step<0>(v, lane);
        wide_lane_step<1>(v, lane);
        wide_lane_step<2>(v, lane);
        wide_lane_step<3>(v, lane);
        wide_lane_step<4>(v, lane);
    }
}

__device__ __forceinline__ unsigned wide_pack(float low, float high) {
    return unsigned(__half_as_ushort(__float2half_rn(low))) |
        (unsigned(__half_as_ushort(__float2half_rn(high))) << 16);
}

// The four values of a lane for one group: the product of x and suh, the seven steps, the scale
// and the round to F16, packed as four halves. `non_finite` is set when a value fails a check.
template <int Mutant>
__device__ __forceinline__ uint2 wide_rotate_group(float4 x, float4 s, unsigned lane,
                                                   bool &non_finite) {
    float v[4];
    if constexpr (Mutant == wide_drop_suh) {
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
    } else {
        v[0] = __fmul_rn(x.x, s.x); v[1] = __fmul_rn(x.y, s.y);
        v[2] = __fmul_rn(x.z, s.z); v[3] = __fmul_rn(x.w, s.w);
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        if (!isfinite(v[i])) { non_finite = true; v[i] = 0.0f; }
    }
    if constexpr (Mutant == wide_contract) {
        // What the compiler may do to x * suh + y when the product is not pinned.
        const float a = fmaf(x.x, s.x, v[1]), b = fmaf(x.x, s.x, -v[1]);
        const float c = fmaf(x.z, s.z, v[3]), d = fmaf(x.z, s.z, -v[3]);
        v[0] = a; v[1] = b; v[2] = c; v[3] = d;
    }
    wide_butterfly<Mutant>(v, lane);
    const float scale = rsqrtf(128.0f);
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        v[i] = __fmul_rn(v[i], scale);
        if (!isfinite(v[i])) { non_finite = true; v[i] = 0.0f; }
    }
    return make_uint2(wide_pack(v[0], v[1]), wide_pack(v[2], v[3]));
}

// Grid (ceil(ceil(total_rows * K / 128 / wide_groups) / wide_warps)): warp w of the grid rotates
// the groups wide_groups * w .. wide_groups * w + wide_groups - 1, counted across the rows of the
// table. K is a multiple of 128, so a group lies inside one row.
template <int Mutant>
__global__ void __launch_bounds__(wide_warps * 32)
jobs_input_rotate_wide(const __grid_constant__ inference_cuda_exl3_jobs_args p) {
    const unsigned lane = threadIdx.x & 31u;
    // A job outside its bounds flags the call, as in jobs_input_rotate; the check is spread
    // over the threads of the grid.
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < p.jobs_count;
         index += size_t(gridDim.x) * blockDim.x) {
        if (!job_in_bounds(p, p.jobs[index])) atomicOr(p.flag, 1u);
    }
    __syncwarp();
    const uint32_t per_row = uint32_t(p.k / 128);
    const uint32_t total = uint32_t(p.total_rows) * per_row;
    const uint32_t first = (blockIdx.x * wide_warps + threadIdx.x / 32) * wide_groups;
    if (first >= total) return;
    uint32_t row = first / per_row, in_row = first - row * per_row;

    // Where each group of the warp reads and writes. Every lane computes the same values.
    const float *xs[wide_groups], *suhs[wide_groups];
    uint2 *dst[wide_groups];
    bool live[wide_groups];
    WideOwner owner{-1, 1, 0};
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        const uint32_t group = first + u;
        live[u] = Mutant == wide_tail ? group + 1 < total : group < total;
        if (!live[u]) continue;
        // The groups of a warp are consecutive, so the row advances by subtraction.
        if (u > 0 && ++in_row == per_row) { in_row = 0; ++row; }
        const size_t column = size_t(in_row) * 128 + 4 * lane;
        if (row < owner.lo || row >= owner.hi) owner = wide_find_owner(p, row);
        live[u] = owner.member >= 0;
        if (!live[u]) continue;
        size_t source = row;
        if (p.in_rows) {
            const bool neighbour = Mutant == wide_gather_row && (row ^ 1u) < p.total_rows;
            const uint32_t listed = neighbour ? row ^ 1u : row;
            source = p.in_rows[listed];
        }
        // A list entry past the rows of x drops the row (zero input) and flags the call.
        const bool inside = source < p.x_rows;
        if (!inside && lane == 0) atomicOr(p.flag, 1u);
        xs[u] = inside ? p.x + source * p.k + column : nullptr;
        suhs[u] = inference_cuda_exl3_member_suh(p, owner.member) + column;
        dst[u] = reinterpret_cast<uint2 *>(p.xh + size_t(row) * p.k + column);
    }

    float4 x[wide_groups], s[wide_groups];
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        x[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        s[u] = x[u];
        if (!live[u]) continue;
        s[u] = __ldg(reinterpret_cast<const float4 *>(suhs[u]));
        if (xs[u]) x[u] = __ldg(reinterpret_cast<const float4 *>(xs[u]));
    }

    bool non_finite = false;
    #pragma unroll
    for (unsigned u = 0; u < wide_groups; ++u) {
        if (!live[u]) continue;
        *dst[u] = wide_rotate_group<Mutant>(x[u], s[u], lane, non_finite);
    }
    if (__any_sync(0xffffffffu, non_finite) && lane == 0) atomicOr(p.flag, 1u);
}

// Whether a call may take jobs_input_rotate_wide: the switch on and the vector accesses aligned
// (16 bytes for x and suh, which includes the member stride of the scales, and 8 for xh).
// Everything else runs jobs_input_rotate.
bool jobs_wide_input_usable(const inference_cuda_exl3_jobs_args &a, bool enabled) {
    return enabled && uintptr_t(a.x) % 16 == 0 && uintptr_t(a.suh) % 16 == 0 &&
        uintptr_t(a.xh) % 8 == 0 && (a.members == 1 || (a.scale_stride * 4) % 16 == 0);
}

template <int Mutant>
void launch_wide_input(const inference_cuda_exl3_jobs_args &a, cudaStream_t s) {
    const size_t groups = a.total_rows * (a.k / 128);
    const size_t warps = (groups + wide_groups - 1) / wide_groups;
    jobs_input_rotate_wide<Mutant><<<dim3(unsigned((warps + wide_warps - 1) / wide_warps)),
                                     wide_warps * 32, 0, s>>>(a);
}

// FIFI_EXL3_JOBS_WARP_ROTATE: 1 on, 0 or unset off, anything else is unrecognized (-1).
int jobs_wide_input_setting(const char *text) {
    if (!text) return 0;
    if (text[0] == '1' && text[1] == '\0') return 1;
    if (text[0] == '0' && text[1] == '\0') return 0;
    return -1;
}

// Read once. An unrecognized value keeps the one-thread-per-value kernel, as unset does.
bool jobs_wide_input_enabled() {
    static const bool enabled = jobs_wide_input_setting(std::getenv("FIFI_EXL3_JOBS_WARP_ROTATE")) == 1;
    return enabled;
}
