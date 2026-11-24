// Probe only. Each physical Q6 block has signed subscale per 16 values.
// MMA integer dots in consecutive K16 pieces retain both activation digits;
// scale intermediate dot products per subblock before accumulating.
template<int MW,int NW,int KW>
__global__ void inkling_q6_imma(inference_cuda_dense_gemv_args a,const ResidualBlock *x){
    static_assert(MW*NW*KW==8);__shared__ float2 parts[MW][NW][KW][32];
    int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,l=lane%4;
    int kw=warp%KW,nw=warp/KW%NW,mw=warp/KW/NW;
    int m0=(blockIdx.y*MW+mw)*8,n0=(blockIdx.x*NW+nw)*8;
    int m=min(m0+g,a.m-1),blocks=a.cols/32;const auto *row=static_cast<const uint8_t *>(a.w)+size_t(n0+g)*(a.cols/256)*210;float2 sum={};
    for(int j=kw;j<a.cols/16;j+=KW){int k=16*j+4*l,at=k%256,half=at/128,p=at%128/32,ix=at%32;
        auto b=row+(k/256)*210;auto *low=b+64*half+32*(p%2)+ix;auto *hi=b+128+32*half+ix;
        unsigned q=0;
#pragma unroll
        for(int u=0;u<4;u++){unsigned lo=fifi_gguf::nibble(low[u],p/2),h=(hi[u]>>(2*p))&3;q|=((unsigned(int(lo|(h<<4))-32)&255)<<(8*u));}
        float d=fifi_gguf::half_value(b+208)*float(int8_t(b[192+at/16]));float d0=__shfl_sync(0xffffffff,d,8*l),d1=__shfl_sync(0xffffffff,d,8*l+4);
        const auto &v=x[m*blocks+j/2];int i=(j%2)*4+l;int2 h=inkling_byte_mma({},v.hi[i],q),r=inkling_byte_mma({},v.lo[i],q);
        sum.x=fmaf(fmaf(float(r.x),1.f/254.f,float(h.x)),v.scale*d0,sum.x);
        sum.y=fmaf(fmaf(float(r.y),1.f/254.f,float(h.y)),v.scale*d1,sum.y);
    }
    parts[mw][nw][kw][lane]=sum;__syncthreads();
    if(!kw && m0+g<a.m){float2 z=parts[mw][nw][0][lane];
#pragma unroll
        for(int j=1;j<KW;j++){z.x+=parts[mw][nw][j][lane].x;z.y+=parts[mw][nw][j][lane].y;}
        auto y=static_cast<float *>(a.y)+size_t(m0+g)*a.ld_y+n0+2*l;y[0]=z.x;y[1]=z.y;}
}
