// Cached dictionaries preserve the original integer dot and FP32 reduction
// order. Each warp owns several independent rows from one selected expert.
template<int Type, class Entry>
__device__ __forceinline__ GateGrid cached_gate_grid(const uint8_t *b, int lane, const Entry *table) {
    const int win = lane / 4, seg = lane % 4;
    const float d = __half2float(*reinterpret_cast<const __half *>(b));
    if constexpr (Type == 19) {
        const unsigned tag = reinterpret_cast<const uint16_t *>(b + 34)[win];
        const unsigned code = b[2 + lane] | (((tag >> (3 * seg)) & 7) << 8);
        const unsigned grid = table[code], delta = tag & 0x8000 ? 0x09090909u : 0x07070707u;
        return {int(__vsub4((grid & 0x0f0f0f0fu) << 3, delta)),
                int(__vsub4((grid & 0xf0f0f0f0u) >> 1, delta)),
                d * (0.125f * float(1 + 2 * ((tag >> 12) & 7)))};
    } else {
        unsigned code, signs, factor;
        if constexpr (Type == 16) {
            const auto *v = reinterpret_cast<const uint16_t *>(b + 6 + 8*win);
            const unsigned tag = unsigned(v[0]) | unsigned(v[1]) << 16;
            code = b[2 + 8*win + seg]; signs = (tag >> (7*seg)) & 127; factor = tag >> 28;
        } else if constexpr (Type == 17) {
            const unsigned tag = reinterpret_cast<const uint16_t *>(b+2)[lane];
            code = tag & 511; signs = tag >> 9;
            factor = fifi_gguf::nibble(b[66+win], seg/2);
        } else {
            code = b[2+lane] | ((unsigned(b[66+win]) << (8-2*seg)) & 768);
            signs = b[34+lane]; factor = fifi_gguf::nibble(b[74+win], seg/2);
        }
        if constexpr (Type != 22) signs |= (__popc(signs) & 1) << 7;
        const uint64_t grid = table[code];
        const unsigned m0 = (((signs & 15)*0x00204081u) & 0x01010101u)*255u;
        const unsigned m1 = (((signs >> 4)*0x00204081u) & 0x01010101u)*255u;
        return {int((uint32_t(grid)^m0) + (m0 & 0x01010101u)),
                int((uint32_t(grid>>32)^m1) + (m1 & 0x01010101u)),
                d * (0.5f + float(factor)) * 0.25f};
    }
}

template<int Type, int Bytes, int Rows = 4>
__global__ void decode_gate_cached(const GateDigits *packed, const uint8_t *gate,
                                   float *out, DeviceSelectedWeights idx) {
    constexpr int entries = Type == 16 ? 256 : Type == 17 ? 512 : Type == 19 ? 2048 : 1024;
    using Entry = std::conditional_t<Type == 19, uint32_t, uint64_t>;
    __shared__ Entry table[entries];
    for (int i = threadIdx.x; i < entries; i += blockDim.x) {
        if constexpr (Type == 16) table[i] = iq2xxs_grid[i];
        if constexpr (Type == 17) table[i] = iq2xs_grid[i];
        if constexpr (Type == 19) table[i] = iq1s_grid_gpu[i];
        if constexpr (Type == 22) table[i] = iq2s_grid[i];
    }
    __shared__ int words[9][6][32];
    __shared__ float scales[9][32];
    for (int i = threadIdx.x; i < 288; i += blockDim.x) {
        auto a = packed[i];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            words[i/32][2*j][i%32] = a.lo[j];
            words[i/32][2*j+1][i%32] = a.hi[j];
        }
        scales[i/32][i%32] = a.scale;
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    for (int row = blockIdx.x * (8*Rows) + threadIdx.x/32; row < 2304; row += gridDim.x*(8*Rows)) {
        const auto *w = gate + idx.offset(blockIdx.y) + size_t(row)*9*Bytes;
        float sums[Rows] = {};
#pragma unroll 3
        for (int k = 0; k < 9; ++k) {
            int aw[6];
#pragma unroll
            for (int j = 0; j < 6; j++) aw[j] = words[k][j][lane];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const auto g = cached_gate_grid<Type>(w + (k+r*72)*Bytes, lane, table);
                int dot = 0;
#pragma unroll
                for (int j = 0; j < 3; ++j) dot = 128*dot + __dp4a(g.hi,aw[2*j+1],__dp4a(g.lo,aw[2*j],0));
                sums[r] = fmaf(float(dot), scales[k][lane]*g.scale, sums[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            float v = warp_sum(sums[r]);
            if (!lane) out[size_t(blockIdx.y)*2304 + row + 8*r] = v;
        }
    }
}
