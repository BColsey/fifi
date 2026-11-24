// IQ3_XXS: gather each four-level dictionary word once, not once per
// activation. A lane dots two code words against two contiguous float4s.
__global__ void decode_iq3_words(inference_cuda_dense_gemv_args a) {
    __shared__ unsigned table[256];
    __shared__ float part[2][3];
    for (int i=threadIdx.x; i<256; i+=blockDim.x) table[i]=iq3xxs_grid[i];
    __syncthreads();
    int warp=threadIdx.x/32, lane=threadIdx.x%32, row=2*blockIdx.x+warp/3;
    auto *w=static_cast<const uint8_t *>(a.w)+size_t(row)*9*98;
    auto *x=static_cast<const float4 *>(a.x)+2*lane;
    float acc=0;
#pragma unroll
    for(int k=warp%3;k<9;k+=3) {
        auto *b=w+k*98;
        auto tag=kw_read32(b+66+4*(lane/4));
        unsigned signs=(tag>>(7*(lane%4)))&127;
        signs|=(__popc(signs)&1)<<7;
        unsigned q=reinterpret_cast<const uint16_t *>(b+2)[lane];
        unsigned g[2]={table[q&255],table[q>>8]};
        float dot=0;
#pragma unroll
        for(int j=0;j<2;j++) {
            unsigned m=((((signs>>(4*j))&15)*0x00204081u)&0x01010101u)*255u;
            unsigned v=(g[j]^m)+(m&0x01010101u);
            float4 u=x[k*64+j];
#pragma unroll
            for(int p=0;p<4;p++) dot=fmaf(float(int8_t(v>>(8*p))),reinterpret_cast<float *>(&u)[p],dot);
        }
        float scale=__half2float(*reinterpret_cast<const __half *>(b))*(0.5f+float(tag>>28))*0.5f;
        acc=fmaf(scale,dot,acc);
    }
    acc=warp_sum(acc);
    if(!lane) part[warp/3][warp%3]=acc;
    __syncthreads();
    if(warp%3==0 && !lane) {
        auto *p=part[warp/3]; static_cast<float *>(a.y)[row]=(p[0]+p[1])+p[2];
    }
}
