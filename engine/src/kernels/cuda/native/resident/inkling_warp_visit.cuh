// No CTA barriers: each warp computes two output rows and eight activations.
// Weight values are decoded once into registers for eight FP32 FMAs.
template<int Codec,int Block,int Bytes,int K,int BN,int BM>
__global__ void inkling_warp_visit(inference_cuda_dense_gemv_args a){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x%32;
    int n0=warp*BN,m0=blockIdx.y*BM; if(n0>=a.rows)return;
    const auto *w=static_cast<const uint8_t *>(a.w)+size_t(n0)*(K/Block*Bytes);
    const auto *x=static_cast<const float *>(a.x);
    float sum[BN][BM]={};
    int first[BM];
#pragma unroll
    for(int m=0;m<BM;m++) first[m]=(m0+m<a.m?m0+m:m0)*a.ld_x;
#pragma unroll 2
    for(int k=lane;k<K;k+=32){float v[BN];
#pragma unroll
        for(int n=0;n<BN;n++) v[n]=fifi_gguf::decode<Codec>(w+n*(K/Block*Bytes)+(k/Block)*Bytes,k%Block);
#pragma unroll
        for(int m=0;m<BM;m++) { float act=__ldg(x+first[m]+k);
#pragma unroll
            for(int n=0;n<BN;n++)sum[n][m]=fmaf(act,v[n],sum[n][m]);}
    }
#pragma unroll
    for(int n=0;n<BN;n++){
#pragma unroll
        for(int m=0;m<BM;m++){float v=warp_sum(sum[n][m]);if(!lane&&m0+m<a.m)static_cast<float *>(a.y)[(m0+m)*a.ld_y+n0+n]=v;}}
}
