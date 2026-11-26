// IQ4_XS vocabulary only. Gather four byte indices with three register
// permutations; the register bytes are FiFi's weights/gguf_tables.rs IQ4_NL
// levels, in little-endian order. No divergent read of a constant table.
__device__ __forceinline__ int vocab_four_levels(unsigned q) {
    unsigned v = q & 0x0f0f0f0f;
    v = (v | (v >> 4)) & 0xff00ff;
    const unsigned sel = (v | (v >> 8)) & 65535, low = sel & 0x7777;
    const unsigned a = __byte_perm(0xbfad9881, 0xf6eaddcf, low);
    const unsigned b = __byte_perm(0x26190d01, 0x71594535, low);
    return __byte_perm(a, b, 0x3210 | ((sel >> 1) & 0x4444));
}

__global__ void decode_vocab_iq4(inference_cuda_dense_gemv_args a) {
    // Reuse a high-precision activation representation across all rows of a
    // persistent CTA. Each packed word of four low/high nibbles needs two
    // four-byte activation words per radix-128 digit.
    __shared__ int digits[9][6][32];
    __shared__ float scales[9][32];
    const auto *x = static_cast<const float *>(a.x);
    for (int i = threadIdx.x; i < 288; i += blockDim.x) {
        const int k = i / 32, lane = i % 32;
        const int base = k * 256 + lane / 4 * 32 + lane % 4 * 4;
        float v[8], peak = 0;
        bool finite = true;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            v[j] = x[base + j % 4 + 16 * (j / 4)];
            peak = fmaxf(peak, fabsf(v[j]));
            finite &= isfinite(v[j]);
        }
        scales[k][lane] = finite ? peak * (1.f / (127 * 16384)) : __int_as_float(0x7fc00000);
#pragma unroll
        for (int j = 0; j < 8; j++) v[j] = peak > 0 ? (v[j] / peak) * 127.f : 0;
#pragma unroll
        for (int p = 0; p < 3; p++) {
            unsigned words[2] = {};
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int digit = __float2int_rn(v[j]);
                v[j] = (v[j] - digit) * 128;
                words[j / 4] |= unsigned(uint8_t(digit)) << (8 * (j % 4));
            }
            digits[k][2*p][lane] = words[0];
            digits[k][2*p+1][lane] = words[1];
        }
    }
    __syncthreads();
    const int lane = threadIdx.x % 32, win = lane / 4;
    for (int row = blockIdx.x * 8 + threadIdx.x / 32; row < a.rows; row += gridDim.x * 8) {
        const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * 9 * 136;
        float sum = 0;
#pragma unroll 3
        for (int k = 0; k < 9; k++) {
            const auto *b = w + k * 136;
            int l = (b[4 + win / 2] >> (4 * (win % 2))) & 15;
            int h = (*reinterpret_cast<const uint16_t *>(b + 2) >> (2 * win)) & 3;
            float scale = __half2float(*reinterpret_cast<const __half *>(b)) * float((l | h << 4) - 32);
            const unsigned q = reinterpret_cast<const unsigned *>(b + 8)[lane];
            const int lo = vocab_four_levels(q), hi = vocab_four_levels(q >> 4);
            int dot = 0;
            // Even the conservative bound 8*127*(127*16384+64*128+64)
            // fits signed int32. The final multiply-add uses F32.
#pragma unroll
            for (int p = 0; p < 3; p++) {
                dot = 128 * dot + __dp4a(hi, digits[k][2*p+1][lane], __dp4a(lo, digits[k][2*p][lane], 0));
            }
            sum = fmaf(float(dot), scales[k][lane] * scale, sum);
        }
        sum = warp_sum(sum);
        if (!lane) static_cast<float *>(a.y)[row] = sum;
    }
}
