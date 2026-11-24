// Included inside dense_gemv.cu's implementation namespace. One CTA per
// output avoids starving SMs on small projections; four warps partition K.
bool decode_split_enabled() {
    const char *v = std::getenv("FIFI_INKLING_CUDA_DECODE_SPLIT");
    return v && v[0] == '1' && v[1] == '\0';
}

__device__ void decode_split_store(float acc, float *out) {
    __shared__ float partial[4];
    acc = warp_sum(acc);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) *out = (partial[0] + partial[2]) + (partial[1] + partial[3]);
}

#include "resident/inkling_decode_exact_dense.cuh"

template<int Type, int Block, int Bytes, typename Index>
__global__ void decode_split_packed(inference_cuda_dense_gemv_args a, Index index) {
    const auto *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    int blocks = a.cols / Block, lane = threadIdx.x % 32;
    const auto *row = matrix_base(a.w,index,blockIdx.y) + size_t(blockIdx.x) * blocks * Bytes;
    float sum = 0;
    if constexpr (Type == 8) {
        // Eight threads share one 34-byte Q8 block, using two aligned pairs
        // of samples per thread. Four-byte weight alignment is not guaranteed.
        for (int k = threadIdx.x / 8; k < blocks; k += 16) {
            const auto *b = row + k * Bytes;
            int off = 4*(threadIdx.x % 8);
            const auto *words = reinterpret_cast<const uint16_t *>(b + 2 + off);
            unsigned v = unsigned(words[0]) | (unsigned(words[1]) << 16);
            float d = __half2float(*reinterpret_cast<const __half *>(b));
#pragma unroll
            for (int j=0;j<4;++j) sum=fmaf(d*float(int8_t(v>>(8*j))),x[k*32+off+j],sum);
        }
    } else for (int k = threadIdx.x / 32; k < blocks; k += 4) {
        const auto *b = row + k * Bytes;
#pragma unroll
        for (int j = 0; j < Block; j += 32)
            sum = fmaf(fifi_gguf::decode<Type>(b,j+lane), x[k*Block+j+lane], sum);
    }
    decode_split_store(sum,static_cast<float *>(a.y) + size_t(blockIdx.y)*a.ld_y + blockIdx.x);
}

// Select E2M1 magnitudes in half units from registers. Keep positive zero.
__device__ __forceinline__ float decode_fp4_reg(unsigned nibble) {
    int q = __byte_perm(0x03020100u, 0x0c080604u, nibble & 7u) & 255;
    return (nibble & 8 ? -q : q) * 0.5f;
}

__global__ void decode_nv_down(inference_cuda_dense_gemv_args a, DeviceSelectedWeights idx) {
    int row=blockIdx.x*8+threadIdx.x/32, lane=threadIdx.x%32;
    const auto *w=matrix_base(a.w,idx,blockIdx.y)+size_t(row)*18*36;
    const auto *x=static_cast<const float *>(a.x)+size_t(blockIdx.y)*a.ld_x;
    float sum=0;
#pragma unroll 3
    for(int k=0;k<18;++k) {
        auto *b=w+k*36;
        float scale=gguf_scale(b[lane/8]);
        unsigned pair=b[4+lane];
        int c=k*64+lane/8*16+lane%8;
        sum=fmaf(decode_fp4_reg(pair&15)*scale,x[c],sum);
        sum=fmaf(decode_fp4_reg(pair>>4)*scale,x[c+8],sum);
    }
    sum=warp_sum(sum);
    if (!lane) static_cast<float *>(a.y)[size_t(blockIdx.y)*a.ld_y+row]=sum;
}

#include "resident/inkling_vocab_iq4.cuh"
#include "resident/inkling_decode_q8_word.cuh"
#include "resident/inkling_decode_kwindow.cuh"
#include "resident/inkling_decode_iq3.cuh"

template<typename Index>
bool launch_decode_split(const inference_cuda_dense_gemv_args &a, Index idx, cudaStream_t s) {
    const int type = a.weight_dtype == kDtypeGgufNvfp4 ? 40 : a.weight_dtype - fifi_gguf::kDtypeBase;
    if constexpr (std::is_same_v<Index, DeviceSelectedWeights>) {
        if (type == 40 && a.m == 8 && a.cols == 1152 && a.rows == 2304 && a.ld_x == 1152) {
            decode_nv_down<<<dim3(a.rows/8,a.m),256,0,s>>>(a,idx);
            return true;
        }
    }
    if (!std::is_same_v<Index, SharedWeights> || a.m != 1) return false;
    // The old eight-rows/CTA grid rounds the unpadded head up to 200064.
    // Both the exact model width and the archived benchmark width are safe:
    // each persistent warp tests its actual row against a.rows.
    if (type == 23 && a.cols == 2304 && (a.rows == 200058 || a.rows == 200064) &&
        aligned(a.x, 16) && aligned(a.w, 4)) {
        decode_vocab_iq4<<<544, 256, 0, s>>>(a);
        return true;
    }
    if (!((a.cols == 2304 && (a.rows == 384 || a.rows == 1024 || a.rows == 3072 || a.rows == 200064)) ||
          (a.cols == 3072 && a.rows == 2304))) return false;
    if (type == 8 && a.rows != 200064 && aligned(a.x,16)) {
        if (a.cols == 2304) decode_q8_word<2304><<<a.rows/2,128,0,s>>>(a);
        else decode_q8_word<3072><<<a.rows/2,128,0,s>>>(a);
        return true;
    }
    if (type == 18 && a.cols == 2304 && a.rows == 3072 && aligned(a.x,16)) {
        decode_iq3_words<<<a.rows/2,192,0,s>>>(a);
        return true;
    }
    if (a.rows != 200064 && aligned(a.x,16) && aligned(a.w,4)) {
#define KW(t,b) case t: launch_kwindow<t,b>(a,s); return true;
        switch(type) { KW(12,144) KW(13,176) KW(14,210) }
#undef KW
    }
    switch (type) {
#define SPLIT(t,b,n) case t: decode_split_packed<t,b,n><<<dim3(a.rows,a.m),128,0,s>>>(a,idx); break;
        SPLIT(8,32,34) SPLIT(12,256,144) SPLIT(13,256,176) SPLIT(14,256,210)
        SPLIT(18,256,98) SPLIT(23,256,136)
#undef SPLIT
        default: return false;
    }
    return true;
}
