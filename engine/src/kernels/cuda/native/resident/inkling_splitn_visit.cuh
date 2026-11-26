// A CTA covers one 16-output-column WMMA fragment but distributes K among
// 8 warps. Load activations once per CTA, across the complete token group.
// No inter-CTA reduction and no global scratch. Probe-only numerical trial.
template<int Codec,int Block,int Bytes,int M>
__global__ void inkling_splitn_visit(inference_cuda_dense_gemv_args a) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    constexpr int pitch=136;
    __shared__ __align__(32) union { struct{__half x[M*pitch],w[16*pitch];} tile;float parts[8][M*16];} sm;
    int t=threadIdx.x,warp=t/32,n0=blockIdx.x*16;
    WmmaAccumulator c[M/16];
#pragma unroll
    for(int i=0;i<M/16;i++) fill_fragment(c[i],0.f);
    for(int k0=0;k0<a.cols;k0+=128){
        for(int i=t;i<M*128;i+=256){int m=i/128,j=i%128;
            sm.tile.x[m*pitch+j]=__float2half_rn(m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+k0+j]:0.f);}
#pragma unroll
        for(int group=t;group<16*128/4;group+=256){int n=group/32,j=4*(group%32),k=k0+j;
            const auto *b=static_cast<const uint8_t *>(a.w)+((n0+n)*(a.cols/Block)+k/Block)*Bytes;
#pragma unroll
            for(int v=0;v<4;v++) sm.tile.w[n*pitch+j+v]=__float2half_rn(fifi_gguf::decode<Codec>(b,k%Block+v));
        }
        __syncthreads();
        fragment<matrix_b,16,16,16,__half,col_major> w;load_matrix_sync(w,sm.tile.w+warp*16,pitch);
#pragma unroll
        for(int i=0;i<M/16;i++){fragment<matrix_a,16,16,16,__half,row_major> x;
            load_matrix_sync(x,sm.tile.x+i*16*pitch+warp*16,pitch); mma_sync(c[i],x,w,c[i]);}
        __syncthreads();
    }
#pragma unroll
    for(int i=0;i<M/16;i++) store_matrix_sync(sm.parts[warp]+i*256,c[i],16,mem_row_major);
    __syncthreads();
    for(int i=t;i<M*16;i+=256){int m=i/16;if(m<a.m){float sum=sm.parts[0][i];
#pragma unroll
        for(int j=1;j<8;j++) sum+=sm.parts[j][i];static_cast<float *>(a.y)[size_t(m)*a.ld_y+n0+i%16]=sum;}}
#endif
}
