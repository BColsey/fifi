// One warp owns a vocabulary row. Each lane loads one aligned word
// of eight samples and reuses a window scale across its partial dot.
__global__ void decode_vocab_iq4(inference_cuda_dense_gemv_args a) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = float(kvalues_iq4nl[threadIdx.x]);
    __syncthreads();
    int lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    int win = lane / 4, word = lane % 4;
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * 9 * 136;
    const auto *x = static_cast<const float4 *>(a.x);
    float acc = 0;
#pragma unroll 3
    for (int k = 0; k < 9; ++k) {
        const auto *b = w + k * 136;
        int lo = (b[4 + win / 2] >> (4 * (win % 2))) & 15;
        int hi = (*reinterpret_cast<const uint16_t *>(b + 2) >> (2 * win)) & 3;
        float scale = fifi_gguf::half_value(b) * float((lo | hi << 4) - 32);
        unsigned q = reinterpret_cast<const unsigned *>(b + 8)[lane];
        float4 u = x[64*k + 8*win + word], v = x[64*k + 8*win + word + 4];
        float p = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            p = fmaf(lut[(q >> (8*j)) & 15], reinterpret_cast<const float *>(&u)[j], p);
            p = fmaf(lut[(q >> (8*j+4)) & 15], reinterpret_cast<const float *>(&v)[j], p);
        }
        acc = fmaf(scale, p, acc);
    }
    acc = warp_sum(acc);
    if (!lane) static_cast<float *>(a.y)[row] = acc;
}
