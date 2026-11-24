// Exact cache-layout experiment: eight adjacent output rows interleaved by
// K residue. Each thread owns one of the 32 reference lane chains. A CTA
// reads a contiguous 256-byte int8 panel for each K block of 32 columns.
template<int Codec>
__global__ void inkling_interleave_encode(inference_cuda_dense_gemv_args a, signed char *q,float *sc){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx>=a.rows*a.cols)return;
    int k=idx/8%a.cols,n=idx/(8*a.cols)*8+idx%8;const auto *row=static_cast<const uint8_t *>(a.w);
    int out=idx;
    if constexpr(Codec==8){const auto *b=row+(n*(a.cols/32)+k/32)*34;q[out]=static_cast<int8_t>(b[2+k%32]);
        if(k%32==0)sc[((n/8)*(a.cols/32)+k/32)*8+n%8]=fifi_gguf::half_value(b);}
    else {const auto *b=row+(n*(a.cols/256)+k/256)*210;int i=k%256,h=i/128,part=i%128/32,l=i%32;
        unsigned lo=fifi_gguf::nibble(b[64*h+32*(part%2)+l],part/2),hi=(b[128+32*h+l]>>(2*part))&3;
        q[out]=int(lo|(hi<<4))-32;
        if(k%16==0)sc[((n/8)*(a.cols/16)+k/16)*8+n%8]=fifi_gguf::half_value(b+208)*float(int8_t(b[192+i/16]));}
}
template<int Codec,int BM>
__global__ void inkling_interleave_visit(inference_cuda_dense_gemv_args a,const signed char *codes,const float *sc){
    __shared__ union {float x[BM][128];float partial[BM][8][33];} sm;
    const int t=threadIdx.x,c=t/8,n=t%8,m0=blockIdx.y*BM;
    const int ng=blockIdx.x;const signed char *q=codes+size_t(ng)*a.cols*8;
    sc+=size_t(ng)*(a.cols/(Codec==8?32:16))*8;
    float sums[BM]={};
    for(int kb=0;kb<a.cols;kb+=128){
#pragma unroll
        for(int i=t;i<BM*128;i+=256){int m=m0+i/128;
            sm.x[i/128][i%128]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+kb+i%128]:0.f;}
        __syncthreads();
#pragma unroll
        for(int j=0;j<4;j++){
            int k=kb+j*32+c;
            float v=float(q[k*8+n])*sc[(k/(Codec==8?32:16))*8+n];
#pragma unroll
            for(int m=0;m<BM;m++)sums[m]=fmaf(sm.x[m][j*32+c],v,sums[m]);}
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<BM;m++)sm.partial[m][n][c]=sums[m];
    __syncthreads();int row=t/32;
#pragma unroll
    for(int m=0;m<BM;m++){float v=warp_sum(sm.partial[m][row][t%32]);if(t%32==0 && m0+m<a.m) static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+ng*8+row]=v;}
}
