// Fused input rotation + one-tile wide projection (flash-next plan item 6, bench-gated).
//
// The served decode chain runs three launches per projection: grouped_input_rotate
// (x * suh, butterfly, the xh global round trip), grouped_project_small_wide_r56 and
// grouped_output_rotate. This header folds the input rotation into the projection
// kernel: every block of the split-K 2 launch reads only its own K half, so the block
// rotates that half through the SAME butterfly network and add order as
// grouped_input_rotate (rotate128_pair rounds, two 128-element chunks per 256-thread
// round, bit-identical to rotate128 per its comment) into shared memory, and the
// projection body reads the rotated half from shared instead of xh. The output
// rotation stays its own launch: 3a's prefill A/B measured a fused epilogue at twice
// the cost of the standalone rotate, so only the input side fuses here.
//
// Bit-identity: the staged values are finite_or_flag(x * suh) butterfly-rotated and
// finite_or_flag'ed again exactly as grouped_input_rotate writes them, and the
// projection body is the untouched one-tile body reading a GroupEntry whose xh points
// at the shared half: every sum and every reduction is the served kernel's.
//
// Constraints (validated by the entry, INFERENCE_CUDA_INVALID_ARGUMENT otherwise):
// every group is book 2 at rate 10 or 6 with one row, and k is a multiple of 512 (a
// half of 256-aligned chunks for the pair rounds). The Flash-Next dense projections
// are k 2560, n 32-multiples, one row: all pass.

template <bool Dp4a>
__global__ void __launch_bounds__(256, wide_min_blocks<1>) grouped_project_small_wide_r56_fin(
    const __grid_constant__ GroupArgs p) {
    extern __shared__ float s_rot[];  // this block's split-K half of the rotated row
    __shared__ float stage[256];      // the pair butterfly's exchange slots
    __shared__ unsigned wtab[8][128];
    __shared__ float partial[8][16];
    const GroupEntry &a = p.entries[p.row_to_group[blockIdx.y]];
    const size_t row = size_t(blockIdx.y) - a.first;
    const unsigned half = unsigned(p.k) / 2u, base = blockIdx.z * half;
    for (unsigned r = 0; r < half / 256u; ++r) {
        const unsigned local = 256u * r + threadIdx.x;
        const float value = finite_or_flag(
            a.x[row * p.k + base + local] * a.suh[base + local], p.flag);
        s_rot[local] = finite_or_flag(rotate128_pair(value, stage), p.flag);
    }
    __syncthreads();
    GroupEntry e = a;
    // The body's xh index row * k + tile * 16 + packed_row(lane) lands in
    // [base, base + half): point xh at the shared half so it reads the staged values.
    e.xh = s_rot - base;
    if (a.bits_twice == 10)
        one_tile_by_alignment<2, 10, GroupArgs, Dp4a>(p, e, wtab[threadIdx.x / 32], partial, 2);
    else
        one_tile_by_alignment<2, 6, GroupArgs, Dp4a>(p, e, wtab[threadIdx.x / 32], partial, 2);
}
