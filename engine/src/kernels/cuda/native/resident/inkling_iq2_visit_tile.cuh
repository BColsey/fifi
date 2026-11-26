#include <type_traits>
#include "resident/inkling_shared_codes.cuh"
// Each thread decodes a whole weight octet. Paired IQ coefficients share one
// codebook lookup and scale. Input conversion and MMA K order match gguf_wmma_kernel.
template<int Codec, int Block, int Bytes>
__device__ __forceinline__ void inkling_decode_octet(const uint8_t *row, int c, __half *dst) {
    if constexpr (Codec == 17 || Codec == 22) {
        const auto *b = row + (c / Block) * Bytes;
        const int i = c % Block;
        const auto g = fifi_gguf::iq2_group<Codec>(b, i / 32, (i % 32) / 8, fifi_gguf::half_value(b));
#pragma unroll
        for (int j = 0; j < 8; ++j)
            dst[j] = __float2half_rn(fifi_gguf::grid_value(g.grid, j, g.scale, g.signs));
    } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[j] = wmma_weight<Codec,Block,Bytes>(row, c + j);
    }
}

template<int Codec, int K, int Block, int Bytes, bool Vector = false, int BM = 32, int BK = 128, bool CacheCodes = false>
__device__ __forceinline__ void inkling_iq2_visit_body(inference_cuda_dense_gemv_args a, int m0) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    static_assert(K % BK == 0 && BM % 16 == 0);
    constexpr int pitch = BK + 8, stride = BK / 4;
    __shared__ __align__(32) __half x[BM * pitch], w[32 * pitch];
    __shared__ __align__(32) float out[BM * 32];
    const int t = threadIdx.x, warp = t / 32, n0 = blockIdx.x * 32;
    struct Empty{};
    __shared__ std::conditional_t<CacheCodes,InklingCodebook<Codec>,Empty> codes;
    if constexpr(CacheCodes) codes.init();
    WmmaAccumulator acc;
    fill_fragment(acc, 0.f);
    const auto *weights = static_cast<const uint8_t *>(a.w);
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = t; i < 32 * BK / 8; i += 512) {
            const int row = i / (BK / 8), c = 8 * (i % (BK / 8));
            const auto *wr = weights + size_t(n0 + row) * (K / Block * Bytes);
            __half *dst = w + row * pitch + c;
            if constexpr (Vector) {
                __align__(16) __half tmp[8];
                if constexpr(CacheCodes)codes.octet(wr,k+c,tmp);else inkling_decode_octet<Codec,Block,Bytes>(wr, k + c, tmp);
                *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(tmp);
            } else {
                inkling_decode_octet<Codec,Block,Bytes>(wr, k + c, dst);
            }
        }
        if constexpr (Vector) {
#pragma unroll
            for (int j = t; j < BM * stride; j += 512) {
                const int m = m0 + j / stride, c = 4 * (j % stride);
                float4 v = {};
                if (m < a.m) v = *reinterpret_cast<const float4 *>(static_cast<const float *>(a.x) + size_t(m) * a.ld_x + k + c);
                auto *dst = reinterpret_cast<__half2 *>(x + j / stride * pitch + c);
                dst[0] = __floats2half2_rn(v.x, v.y);
                dst[1] = __floats2half2_rn(v.z, v.w);
            }
        } else {
#pragma unroll
            for (int j = t; j < BM * BK; j += 512) {
                const int m = m0 + j / BK, c = j % BK;
                x[j / BK * pitch + c] = __float2half_rn(m < a.m ? static_cast<const float *>(a.x)[size_t(m) * a.ld_x + k + c] : 0.f);
            }
        }
        __syncthreads();
        if (warp < BM / 8) {
#pragma unroll
            for (int step = 0; step < BK; step += 16) {
                fragment<matrix_a,16,16,16,__half,row_major> l;
                fragment<matrix_b,16,16,16,__half,col_major> r;
                load_matrix_sync(l, x + (warp / 2) * 16 * pitch + step, pitch);
                load_matrix_sync(r, w + (warp % 2) * 16 * pitch + step, pitch);
                mma_sync(acc, l, r, acc);
            }
        }
        __syncthreads();
    }
    if (warp < BM / 8) store_matrix_sync(out + (warp / 2) * 512 + (warp % 2) * 16, acc, 32, mem_row_major);
    __syncthreads();
    for (int i = t; i < BM * 32; i += 512)
        if (m0 + i / 32 < a.m) static_cast<float *>(a.y)[size_t(m0 + i / 32) * a.ld_y + n0 + i % 32] = out[i];
#endif
}

template<int Codec, int K, int Block, int Bytes, bool Vector = false, int BM = 32, int BK = 128, bool CacheCodes = false>
__global__ void inkling_iq2_visit_tile(inference_cuda_dense_gemv_args a) {
    inkling_iq2_visit_body<Codec,K,Block,Bytes,Vector,BM,BK,CacheCodes>(a, blockIdx.y * BM);
}
