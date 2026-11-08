/* The 128-point butterfly of the EXL3 warp-wide rotations, on the registers of one warp. Included
 * inside the unnamed namespace of exl3.cu by exl3_rotate_wide.h and exl3_rotate_pair.h; it
 * declares no namespace of its own and knows nothing of jobs, tables or kernels. An includer
 * outside exl3.cu must wrap the include in an unnamed namespace as well: the two non-template
 * functions below have external linkage otherwise. */
#pragma once

// Contract of every function below.
//
// Layout. A warp holds one group of 128 values: lane `l` holds the four consecutive values
// 4l..4l+3 in v[0..3]. Steps at distance 1 and 2 pair values inside a lane (wide_step_one,
// wide_step_two); the steps at distance 4, 8, 16, 32 and 64 pair a value with the same register
// of the lane `l ^ 1`, `l ^ 2`, `l ^ 4`, `l ^ 8` and `l ^ 16` (wide_lane_step<0..4>). In every
// step the element whose step bit is set computes `other - own` and the other `own + other`.
//
// Arithmetic. Every sum and difference is an _rn intrinsic (__fadd_rn, __fsub_rn), which the
// compiler never fuses with a neighbouring multiply: a contracted x * suh + y rounds once instead
// of twice and changes bits.
//
// Input and output. `lane` must be the calling thread's own lane index in the warp
// (threadIdx.x & 31): a wrong value pairs the wrong partner and applies the wrong sign. The
// result is the unnormalised transform in natural order. The caller applies any scale factor
// (the rotations multiply by rsqrtf(128.0f)) and any sign multiplication (suh) outside, and the
// input must already carry what the transform is meant to rotate. `v` must stay in registers
// (indexed with constants after unrolling); a runtime index would move it to local memory.
//
// Instantiations. Production code calls wide_butterfly<false, false> (all seven steps, in order).
// The two booleans are hooks of the mutation builds, not options for other callers.
//
// Precondition. wide_lane_step exchanges with __shfl_xor_sync over the full mask, so all 32 lanes
// of the warp must reach it together, with the same instruction stream: no lane may have left the
// kernel or diverged. Neither a partial warp nor a divergent caller is supported.

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

// The seven steps in the order 1, 2, 4, ..., 64. `Reversed` runs them in the opposite order and
// `FirstStepDone` skips step 1 because the caller already took it; both exist only for the
// mutation builds, and the production kernels instantiate <false, false>.
template <bool Reversed, bool FirstStepDone>
__device__ __forceinline__ void wide_butterfly(float (&v)[4], unsigned lane) {
    if constexpr (Reversed) {
        wide_lane_step<4>(v, lane);
        wide_lane_step<3>(v, lane);
        wide_lane_step<2>(v, lane);
        wide_lane_step<1>(v, lane);
        wide_lane_step<0>(v, lane);
        wide_step_two(v);
        wide_step_one(v);
    } else {
        if constexpr (!FirstStepDone) wide_step_one(v);
        wide_step_two(v);
        wide_lane_step<0>(v, lane);
        wide_lane_step<1>(v, lane);
        wide_lane_step<2>(v, lane);
        wide_lane_step<3>(v, lane);
        wide_lane_step<4>(v, lane);
    }
}
