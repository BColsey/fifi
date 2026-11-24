// Included after warp_sum and the packed-weight decoder in dense_gemv.cu.
// Each CTA owns two complete weight rows for the lifetime of the projection.
// Warps consume both against disjoint activation rows. No inter-CTA reduction.
template<int K, int Codec = 8, int Block = 32, int Bytes = 34>
__global__ void inkling_q8_visit(inference_cuda_dense_gemv_args a) {
    __shared__ float w[2][K];
    const int n = blockIdx.x * 2;
    const int lane = threadIdx.x & 31;
    const auto *base = static_cast<const uint8_t *>(a.w) + size_t(n) * (K / Block * Bytes);
    for (int i = threadIdx.x; i < 2 * K; i += blockDim.x) {
        w[i / K][i % K] = fifi_gguf::decode<Codec>(base + (i / Block) * Bytes, i % Block);
    }
    __syncthreads();
    for (int m = threadIdx.x / 32; m < a.m; m += blockDim.x / 32) {
        const auto *x = static_cast<const float *>(a.x) + size_t(m) * a.ld_x;
        float lo = 0.f, hi = 0.f;
#pragma unroll 8
        for (int c = lane; c < K; c += 32) {
            float v = x[c];
            lo = fmaf(w[0][c], v, lo);
            hi = fmaf(w[1][c], v, hi);
        }
        lo = warp_sum(lo);
        hi = warp_sum(hi);
        if (!lane) {
            auto *out = static_cast<float *>(a.y) + size_t(m) * a.ld_y + n;
            out[0] = lo;
            out[1] = hi;
        }
    }
}
