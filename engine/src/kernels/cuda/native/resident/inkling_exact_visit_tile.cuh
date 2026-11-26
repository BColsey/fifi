#include "resident/inkling_shared_codes.cuh"
// 32 output weights x 8 activations; one warp owns four weight rows.
// The register tile keeps the packed reference's lane assignment and FMA order.
template<int Codec, int Block, int Bytes, int Rows = 4, bool RepeatWeight = false, bool CacheCodes = false>
__device__ __forceinline__ void inkling_visit_dot(inference_cuda_dense_gemv_args a, int bx, float (&activ)[8][128]) {
    struct Empty{};
    __shared__ std::conditional_t<CacheCodes,InklingCodebook<Codec>,Empty> codes;
    if constexpr(CacheCodes)codes.init();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int m0 = blockIdx.y * 8, n0 = bx * (Rows*8) + warp * Rows;
    const auto *weights = static_cast<const uint8_t *>(a.w);
    float sums[Rows][8] = {};
    for (int k0 = 0; k0 < a.cols; k0 += 128) {
#pragma unroll
        for (int j = lane; j < 128; j += 32)
            activ[warp][j] = m0 + warp < a.m ? static_cast<const float *>(a.x)[(m0 + warp) * a.ld_x + k0 + j] : 0.f;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + i * 32 + lane;
#pragma unroll
            for (int n = 0; n < Rows; ++n) {
                int block = (n0+n)*(a.cols/Block) + k/Block;
                float w;
                if constexpr (RepeatWeight) {
                    // Probe-only ablation: cache one dynamic weight per row and lane,
                    // keeping panel loads and the same number of dependent FP32 FMAs.
                    w = static_cast<const float *>(a.x)[(warp * Rows + n) * 32 + lane];
                } else if constexpr(CacheCodes){
                    w=codes.scalar(weights+block*Bytes,k%Block);
                } else {
                    w = fifi_gguf::decode<Codec>(weights + block * Bytes, k%Block);
                }
#pragma unroll
                for (int m = 0; m < 8; ++m) sums[n][m] = fmaf(activ[m][i*32+lane], w, sums[n][m]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < Rows; ++n) {
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            float v = warp_sum(sums[n][m]);
            if (!lane && m0+m < a.m) static_cast<float *>(a.y)[(m0+m)*a.ld_y+n0+n] = v;
        }
    }
}

template<int Codec, int Block, int Bytes, int Rows = 4, bool RepeatWeight = false, bool CacheCodes = false>
__global__ void inkling_exact_visit_tile(inference_cuda_dense_gemv_args a) {
    __shared__ float activ[8][128];
    inkling_visit_dot<Codec,Block,Bytes,Rows,RepeatWeight,CacheCodes>(a,blockIdx.x,activ);
}
