// Stage interleaved loads for a complete M x 16 tile. Each warp owns one
// output weight row and computes its M dot products in reference order.
template<int Codec,int Block,int Bytes,int M>
__global__ void inkling_grid_split(inference_cuda_dense_gemv_args a){
    constexpr int BN=16, BK=128;
    __shared__ float x[M][BK], w[BN][BK];
    int t=threadIdx.x,warp=t/32,lane=t%32,n0=blockIdx.x*BN,m0=blockIdx.y*M;
    // Each of the 16 warps owns a complete weight row; its lanes compute
    // partial sums for one column residue, exactly as in the packed reference.
    float sum[M]={};
    for(int base=0;base<a.cols;base+=BK){
#pragma unroll
        for(int i=t;i<M*BK;i+=512){int m=m0+i/BK,j=i%BK;
            x[i/BK][j]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+base+j]:0.f;}
#pragma unroll
        for(int i=t;i<BN*BK;i+=512){int n=n0+i/BK,j=i%BK,k=base+j;
            auto b=static_cast<const uint8_t *>(a.w)+size_t(n*(a.cols/Block)+k/Block)*Bytes;
            w[i/BK][j]=fifi_gguf::decode<Codec>(b,k%Block);}
        __syncthreads();
#pragma unroll
        for(int j=lane;j<BK;j+=32){float weight=w[warp][j];
#pragma unroll
            for(int m=0;m<M;m++)sum[m]=fmaf(x[m][j],weight,sum[m]);}
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<M;m++){float v=warp_sum(sum[m]);if(!lane&&m0+m<a.m) static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n0+warp]=v;}
}
