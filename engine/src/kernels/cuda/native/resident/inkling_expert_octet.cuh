// One lane decodes eight coefficients sharing an IQ codebook entry. Preserve
// the reference's 32 accumulator chains in four lanes, each holding an octet.
template<int Codec,int Bytes,int BM>
__global__ void inkling_expert_octet(inference_cuda_dense_gemv_args a){
    constexpr int K=2304;int t=threadIdx.x,sub=t%4,n=blockIdx.x*(blockDim.x/4)+t/4,m0=blockIdx.y*BM;
    const auto *weights=static_cast<const uint8_t *>(a.w)+size_t(n)*(K/256*Bytes);
    float sum[BM][8]={}; __shared__ float x[BM][256];
    for(int base=0;base<K;base+=256){
        for(int i=t;i<BM*256;i+=blockDim.x){int m=m0+i/256;
            x[i/256][i%256]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+base+i%256]:0.f;}
        __syncthreads();const auto *b=weights+base/256*Bytes;float scale=fifi_gguf::half_value(b);
#pragma unroll
        for(int seg=0;seg<8;seg++){
            float w[8];
            if constexpr(Codec==16 || Codec==17 || Codec==22){auto g=fifi_gguf::iq2_group<Codec>(b,seg,sub,scale);
#pragma unroll
                for(int j=0;j<8;j++)w[j]=fifi_gguf::grid_value(g.grid,j,g.scale,g.signs);
            }else{
#pragma unroll
                for(int j=0;j<8;j++)w[j]=fifi_gguf::decode<Codec>(b,seg*32+sub*8+j);
            }
#pragma unroll
            for(int m=0;m<BM;m++){
#pragma unroll
                for(int j=0;j<8;j++)sum[m][j]=fmaf(w[j],x[m][seg*32+sub*8+j],sum[m][j]);}
        }
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<BM;m++){
#pragma unroll
        for(int j=0;j<8;j++){
            sum[m][j]+=__shfl_xor_sync(0xffffffff,sum[m][j],2,4);
            sum[m][j]+=__shfl_xor_sync(0xffffffff,sum[m][j],1,4);}
        if(!sub&&m0+m<a.m){float *s=sum[m];float v=((s[0]+s[4])+(s[2]+s[6]))+((s[1]+s[5])+(s[3]+s[7]));
            static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n]=v;}
    }
}
