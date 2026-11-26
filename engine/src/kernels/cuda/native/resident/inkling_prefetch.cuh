#pragma once
// Aligned F32 SM75 prefill: vector loads stage BM activation rows, optionally
// preloading the following K tile before the decode/FMA loop. Each warp owns
// R output weights; its 32 lane chains and final sum match FiFi packed GEMV.
template<int C,int BM,int R,int BK,bool Pipeline>
__device__ __forceinline__ void inkling_prefetch_body(inference_cuda_dense_gemv_args a,int bx,int by=blockIdx.y){
    constexpr int Block=C==8?32:C==42?64:256, Bytes=C==8?34:C==12?144:C==13?176:C==14?210:C==16?66:C==42?18:98;
    static_assert(C==8 || C==12 || C==13 || C==14 || C==18 || C==16 || C==42);
    constexpr int V=BM*BK/4,Loads=(V+255)/256;
    __shared__ __align__(16) float x[BM][BK];
    const int t=threadIdx.x,warp=t/32,lane=t%32,n=bx*(8*R)+warp*R,m=by*BM;
    float sums[R][BM]={};float4 reg[Loads];
    auto load=[&](int k){
#pragma unroll
        for(int l=0;l<Loads;l++){int i=t+256*l,rel=i/(BK/4);reg[l]={};
            if(i<V && m+rel<a.m && k<a.cols)reg[l]=*reinterpret_cast<const float4 *>(static_cast<const float *>(a.x)+size_t(m+rel)*a.ld_x+k+i%(BK/4)*4);}};
    if constexpr(Pipeline)load(0);
    for(int kb=0;kb<a.cols;kb+=BK){
        if constexpr(!Pipeline)load(kb);
#pragma unroll
        for(int l=0;l<Loads;l++){int i=t+256*l;if(i<V)reinterpret_cast<float4 *>(&x[0][0])[i]=reg[l];}
        __syncthreads();if constexpr(Pipeline)load(kb+BK);
#pragma unroll
        for(int j=0;j<BK;j+=32){int k=kb+j+lane;
#pragma unroll
            for(int r=0;r<R;r++){auto b=static_cast<const uint8_t *>(a.w)+size_t((n+r)*(a.cols/Block)+k/Block)*Bytes;
                float w=fifi_gguf::decode<C>(b,k%Block);
#pragma unroll
                for(int u=0;u<BM;u++)sums[r][u]=fmaf(w,x[u][j+lane],sums[r][u]);}}
        __syncthreads();
    }
#pragma unroll
    for(int r=0;r<R;r++){
#pragma unroll
        for(int u=0;u<BM;u++){float s=sums[r][u];
#pragma unroll
            for(int d=16;d;d/=2)s+=__shfl_down_sync(0xffffffff,s,d);
            if(!lane && m+u<a.m)static_cast<float *>(a.y)[size_t(m+u)*a.ld_y+n+r]=s;}}
}
template<int C,int BM,int R,int BK,bool Pipeline>
__global__ void inkling_prefetch(inference_cuda_dense_gemv_args a){
    inkling_prefetch_body<C,BM,R,BK,Pipeline>(a,blockIdx.x);
}
