// Controlled limiter probes. These produce artificial outputs, NOT parity
// candidates. Mode 0 retains all global transactions+codec, mode 1 uses a
// repeated K128 panel instead of traversing weights; mode 2 reads a cached
// float per logical lane. Mode 3 reads FP32 weights (already decoded), and
// mode 4 traverses fresh packed blocks but skips the FMAs (digest instead).
#define main full_probe_main
#include "probe.cu"
#undef main
#include <cuda_fp16.h>
#include "resident/gguf_gemv.cuh"
template<int C,int B,int S,int MODE>
__global__ void model(inference_cuda_dense_gemv_args a,const float *fp){
    __shared__ float x[8][128];int t=threadIdx.x,warp=t/32,l=t%32,n0=blockIdx.x*32+warp*4,m0=blockIdx.y*8;float sum[4][8]={};float digest=0;
    for(int kb=0;kb<a.cols;kb+=128){for(int c=l;c<128;c+=32)x[warp][c]=m0+warp<a.m?static_cast<const float *>(a.x)[size_t(m0+warp)*a.ld_x+kb+c]:0.f;__syncthreads();
#pragma unroll
        for(int j=0;j<4;j++){int k=kb+j*32+l;
#pragma unroll
            for(int r=0;r<4;r++){int n=n0+r;auto w=static_cast<const uint8_t *>(a.w)+size_t(n)*(a.cols/B*S);float v;
                if constexpr(MODE==2)v=fp[(warp*4+r)*32+l];else if constexpr(MODE==3)v=fp[size_t(n)*a.cols+k];else if constexpr(MODE==1)v=fifi_gguf::decode<C>(w+((j*32+l)/B)*S,(j*32+l)%B);else v=fifi_gguf::decode<C>(w+(k/B)*S,k%B);
                if constexpr(MODE==4)digest+=v*x[j][l];else{
#pragma unroll
                    for(int m=0;m<8;m++)sum[r][m]=fmaf(v,x[m][j*32+l],sum[r][m]);}}}__syncthreads();}
    if constexpr(MODE==4){
#pragma unroll
        for(int d=16;d;d/=2)digest+=__shfl_xor_sync(0xffffffff,digest,d);
        if(!l)static_cast<float *>(a.y)[size_t(m0)*a.ld_y+n0]=digest;}else{
#pragma unroll
        for(int r=0;r<4;r++){
#pragma unroll
            for(int m=0;m<8;m++){float val=sum[r][m];
#pragma unroll
                for(int d=16;d;d/=2)val+=__shfl_xor_sync(0xffffffff,val,d);
                if(!l && m0+m<a.m)static_cast<float *>(a.y)[size_t(m0+m)*a.ld_y+n0+r]=val;}}}
}
template<int C,int B,int S>void testmodel(const char *dir,int n,int k,int m){
    size_t bytes=size_t(n)*k/B*S;auto w=load(std::string(dir)+"/"+(C==8?"q8-":"extra-"+std::to_string(C)+"-")+std::to_string(k)+"-"+std::to_string(n)+".bin",bytes);Stream s;Buffer<uint8_t> dw(bytes);Buffer<float> x(m*k),y(m*n),fp(size_t(n)*k);std::vector<float> v(size_t(n)*k);for(size_t i=0;i<v.size();i++)v[i]=float(int(i%79)-26)/153.f;check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(x.p,v.data(),4*m*k,cudaMemcpyHostToDevice));check(cudaMemcpy(fp.p,v.data(),4*v.size(),cudaMemcpyHostToDevice));inference_cuda_dense_gemv_args a{dw.p,x.p,y.p,m,n,k,k,n,256+C};
    for(int arm=0;arm<5;arm++){auto launch=[&]{dim3 g(n/32,(m+7)/8);
#define MOD(M) model<C,B,S,M><<<g,256,0,s.id>>>(a,fp.p)
        switch(arm){case 0:MOD(0);break;case 1:MOD(1);break;case 2:MOD(2);break;case 3:MOD(3);break;case 4:MOD(4);break;}
#undef MOD
    };for(int i=0;i<5;i++)launch();Event st,ed;check(cudaEventRecord(st.id,s.id));for(int i=0;i<32;i++)launch();check(cudaEventRecord(ed.id,s.id));check(cudaEventSynchronize(ed.id));float ms;check(cudaEventElapsedTime(&ms,st.id,ed.id));printf("%d,%d,%d,%d,%d,%.3f\n",C,m,n,k,arm,ms*1000./32);}}
int main(int argc,char **argv){try{require(argc==2,"weights");puts("codec,m,n,k,mode,us");for(int m:{8,64}){testmodel<8,32,34>(argv[1],3072,2304,m);testmodel<14,256,210>(argv[1],3072,2304,m);testmodel<13,256,176>(argv[1],3072,2304,m);}for(int m:{4,8,16,32}){testmodel<16,256,66>(argv[1],2304,2304,m);testmodel<42,64,18>(argv[1],2304,1152,m);}return 0;}catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}}
