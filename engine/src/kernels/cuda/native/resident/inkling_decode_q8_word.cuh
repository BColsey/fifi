// 9 or 12 warps per row underutilize the 3072-column Q8 matrix.
// Instead, let a few warps enumerate tile indices, with two rows per CTA.
template<int Kw>
__global__ void decode_q8_word(inference_cuda_dense_gemv_args a) {
    __shared__ float part[2][Kw];
    int warp=threadIdx.x/32,lane=threadIdx.x%32,row=2*blockIdx.x+warp/Kw;
    auto *w=static_cast<const uint8_t *>(a.w)+size_t(row)*96*34;
    auto *x=static_cast<const float4 *>(a.x);
    float acc=0;
#pragma unroll
    for(int tile=warp%Kw; tile<12; tile+=Kw) {
        int blk=tile*8+lane/4, j=lane%4;
        auto *b=w+blk*34;
        auto *q=reinterpret_cast<const uint16_t *>(b+2+8*j);
        unsigned v[2]={unsigned(q[0])|(unsigned(q[1])<<16),unsigned(q[2])|(unsigned(q[3])<<16)};
        float dot=0;
#pragma unroll
        for(int p=0;p<2;p++) {
            float4 u=x[blk*8+2*j+p];
#pragma unroll
            for(int i=0;i<4;i++) dot=fmaf(float(int8_t(v[p]>>(8*i))),reinterpret_cast<float *>(&u)[i],dot);
        }
        acc=fmaf(__half2float(*reinterpret_cast<const __half *>(b)),dot,acc);
    }
    acc=warp_sum(acc);
    if(!lane) part[warp/Kw][warp%Kw]=acc;
    __syncthreads();
    if(warp%Kw==0 && !lane) {
        float s=part[warp/Kw][0];
#pragma unroll
        for(int i=1;i<Kw;i++) s+=part[warp/Kw][i];
        static_cast<float *>(a.y)[row]=s;
    }
}
