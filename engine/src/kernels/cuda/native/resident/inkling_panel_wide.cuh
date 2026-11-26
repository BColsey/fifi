// Native panel layout: tile multiple N32 x K64 panels per CTA and reuse
// fragments for a 2x2 warp output grid. Retains previous WMMA K order.
template<int K,int BM,int BN>
__global__ void inkling_panel_wide(inference_cuda_dense_gemv_args a,const __half *tmp){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;constexpr int pitch=72,TM=BM/32,TN=BN/32;
    __shared__ __align__(32) __half x[BM*pitch],w[BN*pitch];__shared__ __align__(32) float out[BM*BN];
    int t=threadIdx.x,warp=t/32,n0=blockIdx.x*BN,m0=blockIdx.y*BM;WmmaAccumulator acc[TM][TN];
#pragma unroll
    for(int m=0;m<TM;m++){
#pragma unroll
        for(int n=0;n<TN;n++)fill_fragment(acc[m][n],0.f);}
    for(int k=0;k<K;k+=64){
#pragma unroll
        for(int i=t;i<BN*8;i+=128){int row=i/8;const uint4 *p=reinterpret_cast<const uint4 *>(tmp+((n0/32+row/32)*(K/64)+k/64)*2048);*reinterpret_cast<uint4 *>(w+row*pitch+i%8*8)=p[(row%32)*8+i%8];}
#pragma unroll
        for(int i=t;i<BM*16;i+=128){int m=m0+i/16,c=i%16*4;float4 v={};if(m<a.m)v=*reinterpret_cast<const float4 *>(static_cast<const float *>(a.x)+size_t(m)*a.ld_x+k+c);auto *p=reinterpret_cast<__half2 *>(x+i/16*pitch+c);p[0]=__floats2half2_rn(v.x,v.y);p[1]=__floats2half2_rn(v.z,v.w);}
        __syncthreads();
#pragma unroll
        for(int c=0;c<64;c+=16){fragment<matrix_a,16,16,16,__half,row_major> l[TM];fragment<matrix_b,16,16,16,__half,col_major> r[TN];
#pragma unroll
            for(int m=0;m<TM;m++)load_matrix_sync(l[m],x+(m*32+warp/2*16)*pitch+c,pitch);
#pragma unroll
            for(int n=0;n<TN;n++)load_matrix_sync(r[n],w+(n*32+warp%2*16)*pitch+c,pitch);
#pragma unroll
            for(int m=0;m<TM;m++){
#pragma unroll
                for(int n=0;n<TN;n++)mma_sync(acc[m][n],l[m],r[n],acc[m][n]);}}
        __syncthreads();}
#pragma unroll
    for(int m=0;m<TM;m++){
#pragma unroll
        for(int n=0;n<TN;n++)store_matrix_sync(out+(m*32+warp/2*16)*BN+n*32+warp%2*16,acc[m][n],BN,mem_row_major);}
    __syncthreads();for(int i=t;i<BM*BN;i+=128)if(m0+i/BN<a.m)static_cast<float *>(a.y)[size_t(m0+i/BN)*a.ld_y+n0+i%BN]=out[i];
#endif
}
