// Full-visit exact panel: 32 output rows per CTA. Warps stage disjoint Q8/Q6
// weight columns into shared memory, then each warp visits BM/8 activations
// and four weights, preserving all lane chains. BW and FMA in isolation can
// favor different (BM,BK) choices.
template<int Codec,int Block,int Bytes,int BM,int BK>
__global__ __launch_bounds__(1024,1) void inkling_group64(inference_cuda_dense_gemv_args a){
    __shared__ float x[BM][BK],w[32][BK];
    const int t=threadIdx.x,warp=t/32,lane=t%32,m0=blockIdx.y*BM,n0=blockIdx.x*32;
    // 32 warps partition token rows in four subgroups, weight rows in eights.
    constexpr int MT=BM/4;float sum[4][MT]={};int wg=warp%8,mg=warp/8;
    auto *base=static_cast<const uint8_t *>(a.w);
    for(int k0=0;k0<a.cols;k0+=BK){
        for(int i=t;i<BM*BK;i+=1024){int m=m0+i/BK;
            x[i/BK][i%BK]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+k0+i%BK]:0.f;}
        for(int i=t;i<32*BK;i+=1024){int n=n0+i/BK,k=k0+i%BK;
            w[i/BK][i%BK]=fifi_gguf::decode<Codec>(base+size_t(n*(a.cols/Block)+k/Block)*Bytes,k%Block);}
        __syncthreads();
#pragma unroll
        for(int j=lane;j<BK;j+=32){
#pragma unroll
            for(int m=0;m<MT;m++){float v=x[mg*MT+m][j];
#pragma unroll
                for(int n=0;n<4;n++)sum[n][m]=fmaf(v,w[wg*4+n][j],sum[n][m]);}}
        __syncthreads();
    }
#pragma unroll
    for(int n=0;n<4;n++){
#pragma unroll
        for(int m=0;m<MT;m++){float val=warp_sum(sum[n][m]);int row=m0+mg*MT+m;
            if(!lane && row<a.m)static_cast<float *>(a.y)[size_t(row)*a.ld_y+n0+wg*4+n]=val;}}
}
