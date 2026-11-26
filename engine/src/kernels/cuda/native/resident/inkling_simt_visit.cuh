// Probe only: conventional register-blocked SIMT SGEMM. Unlike the exact
// path, each output has one FMA chain, in increasing K order.
template<int Codec,int Block,int Bytes,int BM,int BN>
__global__ void inkling_simt_visit(inference_cuda_dense_gemv_args a){
    constexpr int BK=16, TN=BN/16, TM=BM/8;
    __shared__ float x[BM][BK+1],w[BK][BN+1];
    int t=threadIdx.x,n0=blockIdx.x*BN,m0=blockIdx.y*BM;
    float sum[TM][TN]={};
    const auto *base=static_cast<const uint8_t *>(a.w);
    for(int k0=0;k0<a.cols;k0+=BK){
#pragma unroll
        for(int i=t;i<BM*BK;i+=128){int r=i/BK,c=i%BK;
            x[r][c]=m0+r<a.m?static_cast<const float *>(a.x)[size_t(m0+r)*a.ld_x+k0+c]:0.f;}
#pragma unroll
        for(int i=t;i<BN*BK;i+=128){int r=i/BK,c=i%BK,k=k0+c;
            auto b=base+size_t((n0+r)*(a.cols/Block)+k/Block)*Bytes;
            w[c][r]=fifi_gguf::decode<Codec>(b,k%Block);}
        __syncthreads();
#pragma unroll
        for(int c=0;c<BK;c++){
#pragma unroll
            for(int i=0;i<TM;i++){float v=x[t/16+i*8][c];
#pragma unroll
                for(int j=0;j<TN;j++)sum[i][j]=fmaf(v,w[c][t%16+j*16],sum[i][j]);}}
        __syncthreads();
    }
#pragma unroll
    for(int i=0;i<TM;i++){
#pragma unroll
        for(int j=0;j<TN;j++){int m=m0+t/16+i*8;if(m<a.m) static_cast<float *>(a.y)[size_t(m)*a.ld_y+n0+t%16+j*16]=sum[i][j];}}
}
