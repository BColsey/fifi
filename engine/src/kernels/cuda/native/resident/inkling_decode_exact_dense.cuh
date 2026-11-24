// Reproduce the original vector matvec's 32 independent FMA chains and XOR
// reduction. On a 64-row controller, the original eight-warps/CTA launch had
// only eight CTAs; here each row gets a whole CTA to load its two operands.
__global__ void decode_controller_down_exact(inference_cuda_dense_gemv_args a) {
    constexpr int cols = 4608, words = cols/4;
    __shared__ float4 weights[words], input[words];
    const auto *w = static_cast<const float4 *>(a.w) + blockIdx.x*words;
    const auto *x = static_cast<const float4 *>(a.x);
    for (int i=threadIdx.x;i<words;i+=256) {
        weights[i]=w[i]; input[i]=x[i];
    }
    __syncthreads();
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    float sum=0;
#pragma unroll 3
    for (int i=lane;i<words;i+=32) {
        const float4 v=weights[i], u=input[i];
        sum=fmaf(v.x,u.x,sum); sum=fmaf(v.y,u.y,sum);
        sum=fmaf(v.z,u.z,sum); sum=fmaf(v.w,u.w,sum);
    }
    sum=warp_sum(sum);
    if (!lane) static_cast<float *>(a.y)[blockIdx.x]=sum;
}

// The aligned BF16 reference reads eight adjacent weights per logical lane.
// The mixed grid has 128 threads; only its first warp is needed on these rows.
__device__ __forceinline__ void decode_bf16_exact_row(inference_cuda_dense_gemv_args a,int row) {
    if (threadIdx.x >= 32) return;
    const int lane=threadIdx.x;
    constexpr int words=2304/8;
    const auto *w=static_cast<const uint4 *>(a.w)+row*words;
    const auto *x=static_cast<const float *>(a.x);
    float sum=0;
#pragma unroll 3
    for (int i=lane;i<words;i+=32) {
        const uint4 word=w[i];
#pragma unroll
        for (int j=0;j<8;++j) sum=fmaf(widen(reinterpret_cast<const __nv_bfloat16 *>(&word)[j]),x[8*i+j],sum);
    }
    sum=warp_sum(sum);
    if (!lane) static_cast<float *>(a.y)[row]=sum;
}

__global__ void decode_bf16_exact(inference_cuda_dense_gemv_args a) {
    decode_bf16_exact_row(a,blockIdx.x);
}
