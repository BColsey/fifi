// Compact weight panel, lane-major for eight output weights. Keep exactly the
// signed Q8 byte and original FP16 scale; no approximation. One warp reads
// contiguous signed-byte vectors instead of a native block's 34-byte stride.
struct InklingWeightPanel{__half scale[8];signed char q[32][8];};
static_assert(sizeof(InklingWeightPanel)==272);
__global__ void inkling_weight_repack(inference_cuda_dense_gemv_args a,InklingWeightPanel *out){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=a.rows*a.cols)return;
    int i=j/256,r=j%8,l=j/8%32,n=i/(a.cols/32)*8+r,b=i%(a.cols/32);auto p=static_cast<const uint8_t *>(a.w)+(n*(a.cols/32)+b)*34;
    out[i].q[l][r]=int8_t(p[2+l]);if(!l)out[i].scale[r]=__ushort_as_half(fifi_gguf::read16(p));}
template<int BM,int BN>
__global__ void inkling_weight_panel(inference_cuda_dense_gemv_args a,const InklingWeightPanel *panels){
    constexpr int BK=128;__shared__ float x[BM][BK],w[BN][BK];int t=threadIdx.x,l=t%32,warp=t/32,n0=blockIdx.x*BN,m0=blockIdx.y*BM;float sum[BN/8][BM]={};
    for(int kb=0;kb<a.cols;kb+=BK){for(int i=t;i<BM*BK;i+=256){int m=m0+i/BK;x[i/BK][i%BK]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+kb+i%BK]:0.f;}
        for(int i=t;i<BN*BK;i+=256){int r=i%8,c=i/8%128,ng=i/(8*BK);auto &p=panels[((n0/8+ng)*(a.cols/32))+(kb+c)/32];w[ng*8+r][c]=__half2float(p.scale[r])*float(p.q[c%32][r]);}
        __syncthreads();
#pragma unroll
        for(int j=l;j<BK;j+=32){
#pragma unroll
            for(int r=0;r<BN/8;r++){float v=w[warp*(BN/8)+r][j];
#pragma unroll
                for(int m=0;m<BM;m++)sum[r][m]=fmaf(v,x[m][j],sum[r][m]);}}
        __syncthreads();}
#pragma unroll
    for(int r=0;r<BN/8;r++){
#pragma unroll
        for(int m=0;m<BM;m++){float v=warp_sum(sum[r][m]);if(!l && m0+m<a.m)static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n0+warp*(BN/8)+r]=v;}}
}
