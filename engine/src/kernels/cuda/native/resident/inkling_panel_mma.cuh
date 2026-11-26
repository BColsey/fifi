// Probe-only two-stage alternative: decode to N32 x K64 native panels then
// issue aligned global loads into WMMA shared tiles. Scratch is timed on every
// projection; there is no assumed packed-weight cache between requests.
template<int Codec,int K,int Block,int Bytes>
__global__ void inkling_decode_panels(inference_cuda_dense_gemv_args a,__half *tmp){
    int octet=blockIdx.x*256+threadIdx.x;
    int panel=octet/256,row=(panel/(K/64))*32+(octet%256)/8;
    int k=(panel%(K/64))*64+(octet%8)*8;
    const auto *wr=static_cast<const uint8_t *>(a.w)+size_t(row)*(K/Block*Bytes);
    __align__(16) __half out[8];
    if constexpr(Codec==17 || Codec==22){const auto *b=wr+(k/Block)*Bytes; int i=k%Block;
        auto g=fifi_gguf::iq2_group<Codec>(b,i/32,(i%32)/8,fifi_gguf::half_value(b));
#pragma unroll
        for(int j=0;j<8;j++)out[j]=__float2half_rn(fifi_gguf::grid_value(g.grid,j,g.scale,g.signs));
    } else {
#pragma unroll
        for(int j=0;j<8;j++) out[j]=wmma_weight<Codec,Block,Bytes>(wr,k+j);
    }
    reinterpret_cast<uint4 *>(tmp)[octet]=*reinterpret_cast<const uint4 *>(out);
}
template<int K>
__global__ void inkling_panel_mma(inference_cuda_dense_gemv_args a,const __half *tmp){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;constexpr int pitch=72;
    __shared__ __align__(32) __half x[32*pitch],w[32*pitch];__shared__ __align__(32) float out[32*32];
    int t=threadIdx.x,warp=t/32,n0=blockIdx.x*32,m0=blockIdx.y*32;
    WmmaAccumulator acc;fill_fragment(acc,0.f);
    for(int k=0;k<K;k+=64){
        const auto *pack=reinterpret_cast<const uint4 *>(tmp+(blockIdx.x*(K/64)+k/64)*2048);
#pragma unroll
        for(int i=t;i<256;i+=128) *reinterpret_cast<uint4 *>(w+(i/8)*pitch+(i%8)*8)=pack[i];
#pragma unroll
        for(int i=t;i<512;i+=128){int m=m0+i/16,c=i%16*4;float4 v={};if(m<a.m)v=*reinterpret_cast<const float4 *>(static_cast<const float *>(a.x)+size_t(m)*a.ld_x+k+c);
            auto *p=reinterpret_cast<__half2 *>(x+(i/16)*pitch+c);p[0]=__floats2half2_rn(v.x,v.y);p[1]=__floats2half2_rn(v.z,v.w);}
        __syncthreads();
#pragma unroll
        for(int i=0;i<64;i+=16){fragment<matrix_a,16,16,16,__half,row_major> l;fragment<matrix_b,16,16,16,__half,col_major> r;
            load_matrix_sync(l,x+(warp/2)*16*pitch+i,pitch);load_matrix_sync(r,w+(warp%2)*16*pitch+i,pitch);mma_sync(acc,l,r,acc);}
        __syncthreads();
    }
    store_matrix_sync(out+(warp/2)*512+(warp%2)*16,acc,32,mem_row_major);__syncthreads();
    for(int i=t;i<1024;i+=128)if(m0+i/32<a.m)static_cast<float *>(a.y)[size_t(m0+i/32)*a.ld_y+n0+i%32]=out[i];
#endif
}
