// Probe: map lanes to activation rows instead of columns. Preserve all 32
// reference lane accumulators per output. The token-major shared transpose
// makes each warp's activation fetches bank-conflict free within a cohort.
template<int Codec,int Block,int Bytes,int BM>
__global__ void inkling_token_visit(inference_cuda_dense_gemv_args a) {
    constexpr int pitch=BM+1; __shared__ float x[32*pitch];
    const int t=threadIdx.x,n=blockIdx.x*(256/BM)+t/BM,rel=t%BM,m=blockIdx.y*BM+rel;
    const auto *w=static_cast<const uint8_t *>(a.w)+size_t(n)*(a.cols/Block*Bytes);
    float sums[32]={};
    for(int k=0;k<a.cols;k+=32){
#pragma unroll
        for(int i=t;i<BM*32;i+=256){int mt=blockIdx.y*BM+i/32;
            x[(i%32)*pitch+i/32]=mt<a.m?static_cast<const float *>(a.x)[size_t(mt)*a.ld_x+k+i%32]:0.f;}
        __syncthreads();
        float decoded[32/BM];
#pragma unroll
        for(int j=0;j<32/BM;j++) decoded[j]=fifi_gguf::decode<Codec>(w+(k/Block)*Bytes,k%Block+j*BM+rel);
#pragma unroll
        for(int c=0;c<32;c++){
            float v=__shfl_sync(0xffffffff,decoded[c/BM],c%BM,BM);
            sums[c]=fmaf(v,x[c*pitch+rel],sums[c]);}
        __syncthreads();
    }
#pragma unroll
    for(int d=16;d;d/=2){
#pragma unroll
        for(int c=0;c<d;c++)sums[c]+=sums[c+d];}
    if(m<a.m)static_cast<float *>(a.y)[size_t(m)*a.ld_y+n]=sums[0];
}
