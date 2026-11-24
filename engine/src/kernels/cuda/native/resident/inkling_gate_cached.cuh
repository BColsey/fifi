// Stage one IQ2 dictionary per CTA. Individual warp lanes index unrelated
// entries; shared memory avoids serialized divergent constant-table loads.
template<int Type, int Bytes>
__global__ void decode_gate_cached(const GateDigits *packed, const uint8_t *gate,
                                   float *out, DeviceSelectedWeights idx) {
    constexpr int entries = Type == 16 ? 256 : Type == 17 ? 512 : 1024;
    __shared__ uint64_t table[entries];
    for (int i = threadIdx.x; i < entries; i += blockDim.x) {
        if constexpr (Type == 16) table[i] = iq2xxs_grid[i];
        if constexpr (Type == 17) table[i] = iq2xs_grid[i];
        if constexpr (Type == 22) table[i] = iq2s_grid[i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32, win = lane / 4, seg = lane % 4;
    int row = blockIdx.x * 16 + threadIdx.x / 32;
    auto *w = gate + idx.offset(blockIdx.y) + size_t(row)*9*Bytes;
    float sums[2] = {};
#pragma unroll 3
    for (int k = 0; k < 9; ++k) {
        auto a = packed[k*32+lane];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            auto *b = w + (k+r*72)*Bytes;
            unsigned code, signs, factor;
            if constexpr (Type == 16) {
                const auto *v = reinterpret_cast<const uint16_t *>(b+6+8*win);
                unsigned tags = unsigned(v[0]) | unsigned(v[1]) << 16;
                code = b[2+8*win+seg]; signs = (tags >> (7*seg)) & 127; factor = tags >> 28;
            } else if constexpr (Type == 17) {
                unsigned tag = reinterpret_cast<const uint16_t *>(b+2)[lane];
                code = tag & 511; signs = tag >> 9;
                factor = fifi_gguf::nibble(b[66+win],seg/2);
            } else {
                code = b[2+lane] | ((unsigned(b[66+win]) << (8-2*seg)) & 768);
                signs = b[34+lane]; factor = fifi_gguf::nibble(b[74+win],seg/2);
            }
            if constexpr (Type != 22) signs |= (__popc(signs) & 1) << 7;
            uint64_t grid = table[code];
            int lo = inkling_signed_grid(uint32_t(grid),signs);
            int hi = inkling_signed_grid(uint32_t(grid>>32),signs>>4);
            float scale = __half2float(*reinterpret_cast<const __half *>(b)) * (0.5f+float(factor)) * 0.25f;
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 3; ++j) dot = 128*dot + __dp4a(hi,a.hi[j],__dp4a(lo,a.lo[j],0));
            sums[r] = fmaf(float(dot),a.scale*scale,sums[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float v = warp_sum(sums[r]);
        if (!lane) out[size_t(blockIdx.y)*2304+row+8*r] = v;
    }
}
