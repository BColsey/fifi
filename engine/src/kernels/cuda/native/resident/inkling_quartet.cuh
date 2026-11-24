// Four adjacent logical lanes in each physical thread. XOR distances 16, 8,
// 4 operate in the eight-thread cohort; distances 2 and 1 operate locally.
template<int Codec,int Block,int Bytes,int K,int Warps>
__global__ void inkling_quartet(inference_cuda_dense_gemv_args a){
    constexpr int panel=256;
    __shared__ float4 x[8][panel/4];
    int t=threadIdx.x,g=t%8,n=blockIdx.x*(Warps*4)+t/8,m0=8*blockIdx.y;
    const auto *w=static_cast<const uint8_t *>(a.w)+size_t(n)*(K/Block*Bytes);
    float4 sum[8]={};
    for(int base=0;base<K;base+=panel){
#pragma unroll
        for(int i=t;i<8*panel/4;i+=Warps*32){int m=m0+i/(panel/4),c=base+4*(i%(panel/4));float4 v={};
            if(m<a.m){auto *p=static_cast<const float *>(a.x)+size_t(m)*a.ld_x+c;
                if((size_t(p)&15)==0)v=*reinterpret_cast<const float4 *>(p);else v=make_float4(p[0],p[1],p[2],p[3]);}
            x[i/(panel/4)][i%(panel/4)]=v;}
        __syncthreads();
#pragma unroll
        for(int j=0;j<panel/32;j++){int c=base+j*32+4*g;auto *b=w+(c/Block)*Bytes;float4 f;
            if constexpr(Codec==8){float d=fifi_gguf::half_value(b);auto *q=reinterpret_cast<const int8_t *>(b+2+4*g);
                f=make_float4(float(q[0])*d,float(q[1])*d,float(q[2])*d,float(q[3])*d);}
            else f=make_float4(fifi_gguf::decode<Codec>(b,c%Block),fifi_gguf::decode<Codec>(b,c%Block+1),fifi_gguf::decode<Codec>(b,c%Block+2),fifi_gguf::decode<Codec>(b,c%Block+3));
#pragma unroll
            for(int m=0;m<8;m++){float4 v=x[m][j*8+g];sum[m].x=fmaf(f.x,v.x,sum[m].x);sum[m].y=fmaf(f.y,v.y,sum[m].y);sum[m].z=fmaf(f.z,v.z,sum[m].z);sum[m].w=fmaf(f.w,v.w,sum[m].w);}
        }
        __syncthreads();
    }
#pragma unroll
    for(int m=0;m<8;m++){float4 v=sum[m];
#pragma unroll
        for(int d=4;d;d/=2){v.x+=__shfl_xor_sync(0xffffffff,v.x,d,8);v.y+=__shfl_xor_sync(0xffffffff,v.y,d,8);v.z+=__shfl_xor_sync(0xffffffff,v.z,d,8);v.w+=__shfl_xor_sync(0xffffffff,v.w,d,8);}
        float result=(v.x+v.z)+(v.y+v.w);if(!g&&m0+m<a.m)static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n]=result;}
}
