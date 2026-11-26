// Standalone numerical experiment. Native Q8 weights stay packed. Quantize
// activations to signed bytes plus a second signed residual digit per block.
// SM75 PTX ISA mma.m8n8k16: lane/4 is row(A) or column(B), lane%4 owns
// four contiguous signed bytes. Each lane receives two consecutive C columns.
struct ResidualBlock { unsigned hi[8],lo[8];float scale; };
__global__ void inkling_residual_pack(inference_cuda_dense_gemv_args a, ResidualBlock *out){
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32,blocks=a.cols/32;
    if(warp>=a.m*blocks)return;int row=warp/blocks,b=warp%blocks;
    float v=static_cast<const float *>(a.x)[size_t(row)*a.ld_x+32*b+lane],maximum=fabsf(v);
#pragma unroll
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    float scale=maximum/127.f; int q=scale>0?__float2int_rn(v/scale):0;
    int r=scale>0?__float2int_rn(fmaf(-float(q),scale,v)*254.f/scale):0;
    unsigned hi=(unsigned(q)&255)<<(8*(lane%4)),lo=(unsigned(max(-127,min(127,r)))&255)<<(8*(lane%4));
    hi|=__shfl_xor_sync(0xffffffff,hi,2);lo|=__shfl_xor_sync(0xffffffff,lo,2);
    hi|=__shfl_xor_sync(0xffffffff,hi,1);lo|=__shfl_xor_sync(0xffffffff,lo,1);
    if(lane%4==0){out[warp].hi[lane/4]=hi;out[warp].lo[lane/4]=lo;}
    if(!lane)out[warp].scale=scale;
}
__device__ __forceinline__ int2 inkling_byte_mma(int2 c,unsigned a,unsigned b){
#if __CUDA_ARCH__ >= 750
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1},{%2},{%3},{%0,%1};" : "+r"(c.x),"+r"(c.y) : "r"(a),"r"(b));
#endif
    return c;
}
__global__ void inkling_residual_imma(inference_cuda_dense_gemv_args a,const ResidualBlock *x){
    __shared__ float2 result[8][32];int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,l=lane%4;
    int n0=blockIdx.x*8,m0=blockIdx.y*8,blocks=a.cols/32;
    auto *w=static_cast<const uint8_t *>(a.w)+size_t(n0+g)*blocks*34;
    int m=min(m0+g,a.m-1);float2 sum={};
    for(int j=warp;j<blocks;j+=8){auto b=w+j*34;const auto &v=x[m*blocks+j];
        float d=fifi_gguf::half_value(b);float d0=__shfl_sync(0xffffffff,d,8*l),d1=__shfl_sync(0xffffffff,d,8*l+4);
        int2 h={},r={};
#pragma unroll
        for(int i=0;i<2;i++){unsigned q=fifi_gguf::read32(b+2+16*i+4*l);
            h=inkling_byte_mma(h,v.hi[4*i+l],q);r=inkling_byte_mma(r,v.lo[4*i+l],q);}
        sum.x=fmaf(fmaf(float(r.x),1.f/254.f,float(h.x)),v.scale*d0,sum.x);
        sum.y=fmaf(fmaf(float(r.y),1.f/254.f,float(h.y)),v.scale*d1,sum.y);
    }
    result[warp][lane]=sum;__syncthreads();
    if(!warp && m0+g<a.m){float2 v=result[0][lane];
#pragma unroll
        for(int i=1;i<8;i++){v.x+=result[i][lane].x;v.y+=result[i][lane].y;}
        float *out=static_cast<float *>(a.y)+size_t(m0+g)*a.ld_y+n0+2*l;out[0]=v.x;out[1]=v.y;}
}
