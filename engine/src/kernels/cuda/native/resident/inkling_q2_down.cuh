// Included by dense_gemv.cu inside its implementation namespace.
constexpr int kInklingDownRows = kInklingGateWidth;
constexpr int kInklingDownCols = kInklingDownRows / 2;
constexpr int kInklingDownPicks = 8;
constexpr int kQ2RowsPerWarp = 4;
constexpr int kQ2RowsPerCta = kQ2RowsPerWarp * kWarps;
static_assert(kInklingDownRows % kQ2RowsPerCta == 0 && kInklingDownCols % 64 == 0);

constexpr bool inkling_q2_down_shape(int hidden, int middle, int count, int dtype) {
    return hidden == kInklingDownRows && middle == kInklingDownCols &&
           count == kInklingDownPicks && dtype == fifi_gguf::kDtypeBase + 42;
}
static_assert(inkling_q2_down_shape(2304, 1152, 8, 256 + 42) &&
              !inkling_q2_down_shape(2305, 1152, 8, 256 + 42) &&
              !inkling_q2_down_shape(2304, 1088, 8, 256 + 42) &&
              !inkling_q2_down_shape(2304, 1152, 7, 256 + 42) &&
              !inkling_q2_down_shape(2304, 1152, 8, 256 + 8));

// Eight physical lanes per row, four adjacent baseline lanes per thread. A
// group decodes two packed bytes and one scale for its eight F32 FMAs, instead
// of each logical lane independently reading both bytes and the scale. Keep all
// four accumulators independent to reproduce the baseline lane FMA sequence.
__global__ void inkling_q2_down(const uint8_t *down, const float *middle, float *out,
                                DeviceSelectedWeights index) {
    constexpr int cols = kInklingDownCols, rows = kInklingDownRows;
    constexpr int blocks = cols / 64, bytes = 18;
    __shared__ float x[cols];
    const int slot = blockIdx.y;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) x[col] = middle[slot * cols + col];
    __syncthreads();

    const int group = threadIdx.x % 8;
    const int row = blockIdx.x * kQ2RowsPerCta + threadIdx.x / 8;
    const auto *w = down + index.offset(slot) + size_t(row) * blocks * bytes;
    float acc[4] = {};
    for (int block = 0; block < blocks; ++block) {
        const auto *b = w + block * bytes;
        const float d = fifi_gguf::half_value(b);
        const unsigned lo = b[2 + group], hi = b[10 + group];
        const int col = block * 64 + 4 * group;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float w0 = d * float(int((lo >> (2 * j)) & 3) - 1);
            const float w1 = d * float(int((hi >> (2 * j)) & 3) - 1);
            acc[j] = fmaf(w0, x[col + j], acc[j]);
            acc[j] = fmaf(w1, x[col + 32 + j], acc[j]);
        }
    }
    // Logical XOR distances 16, 8, 4 become 4, 2, 1 within each eight-lane
    // group. The last two stages (logical 2 and 1) are register adds.
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], offset, 8);
    }
    if (group == 0) out[slot * rows + row] = (acc[0] + acc[2]) + (acc[1] + acc[3]);
}
