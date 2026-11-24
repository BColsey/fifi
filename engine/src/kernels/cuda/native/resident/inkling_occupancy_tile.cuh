// Small 2-D warp grid without materializing decoded weights in shared memory.
// Warps sharing an output interval visit different activations; each warp
// retains Rows x Tokens accumulators. K reduction order is unchanged.
template<int Codec,int Block,int Bytes,int WarpsN,int WarpsM,int Rows,int Tokens,int BK>
__global__ void inkling_occupancy_tile(inference_cuda_dense_gemv_args a){
    constexpr int BM=WarpsM*Tokens,BN=WarpsN*Rows,T=WarpsM*WarpsN*32;
    __shared__ float x[BM][BK]; int t=threadIdx.x,warp=t/32,lane=t%32;
    int n0=blockIdx.x*BN+warp%WarpsN*Rows,m0=blockIdx.y*BM+warp/WarpsN*Tokens;
    float sum[Rows][Tokens]={};auto *base=static_cast<const uint8_t *>(a.w);
    for(int kb=0;kb<a.cols;kb+=BK){
#pragma unroll
        for(int i=t;i<BM*BK;i+=T){int m=blockIdx.y*BM+i/BK;
            x[i/BK][i%BK]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+kb+i%BK]:0.f;}
        __syncthreads();
#pragma unroll
        for(int j=lane;j<BK;j+=32){int k=kb+j;
#pragma unroll
            for(int n=0;n<Rows;n++){auto b=base+size_t((n0+n)*(a.cols/Block)+k/Block)*Bytes;float v=fifi_gguf::decode<Codec>(b,k%Block);
#pragma unroll
                for(int m=0;m<Tokens;m++)sum[n][m]=fmaf(x[(warp/WarpsN)*Tokens+m][j],v,sum[n][m]);}}
        __syncthreads();
    }
#pragma unroll
    for(int n=0;n<Rows;n++){
#pragma unroll
        for(int m=0;m<Tokens;m++){float v=warp_sum(sum[n][m]);if(!lane && m0+m<a.m)static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n0+n]=v;}}
}
