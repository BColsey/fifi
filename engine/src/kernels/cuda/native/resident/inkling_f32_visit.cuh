// Low-rank controller down: reuse each vector of four F32 weights across BM
// token rows. The lane's four FMA order and XOR reduction match dense_gemv_k.
template<int BM,int R>
__global__ void inkling_f32_visit(inference_cuda_dense_gemv_args a){
    constexpr int K=4608;__shared__ float4 x[BM][32];int t=threadIdx.x,warp=t/32,l=t%32,m0=blockIdx.y*BM,n0=blockIdx.x*(8*R)+warp*R;
    const auto *w=static_cast<const float4 *>(a.w);float sum[R][BM]={};
    for(int k=0;k<K;k+=128){
#pragma unroll
        for(int i=t;i<BM*32;i+=256){int m=m0+i/32;float4 v={};if(m<a.m)v=*reinterpret_cast<const float4 *>(static_cast<const float *>(a.x)+size_t(m)*a.ld_x+k+4*(i%32));x[i/32][i%32]=v;}
        __syncthreads();
#pragma unroll
        for(int r=0;r<R;r++){auto v=w[(n0+r)*(K/4)+k/4+l];
#pragma unroll
            for(int j=0;j<BM;j++){auto z=x[j][l];auto &s=sum[r][j];s=fmaf(v.x,z.x,s);s=fmaf(v.y,z.y,s);s=fmaf(v.z,z.z,s);s=fmaf(v.w,z.w,s);}}
        __syncthreads();}
#pragma unroll
    for(int r=0;r<R;r++){
#pragma unroll
        for(int j=0;j<BM;j++){float v=warp_sum(sum[r][j]);if(!l && m0+j<a.m)static_cast<float *>(a.y)[size_t(m0+j)*a.ld_y+n0+r]=v;}}
}
// Direct activation loads avoid staging barriers on this small output grid.
template<int BM>
__global__ void inkling_f32_rank_direct(inference_cuda_dense_gemv_args a){
    constexpr int K=4608;int lane=threadIdx.x%32,n=blockIdx.x*8+threadIdx.x/32,m0=blockIdx.y*BM;
    float sum[BM]={};for(int j=lane;j<K/4;j+=32){auto v=static_cast<const float4 *>(a.w)[n*K/4+j];
#pragma unroll
        for(int m=0;m<BM;m++){if(m0+m>=a.m)break;auto z=*reinterpret_cast<const float4 *>(static_cast<const float *>(a.x)+size_t(m0+m)*a.ld_x+j*4);auto &s=sum[m];s=fmaf(v.x,z.x,s);s=fmaf(v.y,z.y,s);s=fmaf(v.z,z.z,s);s=fmaf(v.w,z.w,s);}}
#pragma unroll
    for(int m=0;m<BM;m++){float v=warp_sum(sum[m]);if(!lane&&m0+m<a.m)static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n]=v;}
}
