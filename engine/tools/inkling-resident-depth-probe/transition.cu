// Isolated attention-output -> normalized-current transition. On seal, the caller
// keeps the old normalized plane immutable and provides a new output plane.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}} while(0)
// Adjacent lanes own independent prompt rows to keep each f32 sum in CPU order.
// Row-major accesses here prioritize parity, not final kernel performance.
__global__ void transition(float* cur,const float* att,float* normalized,int rows,int width,bool seal){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;auto off=size_t(r)*width;float s=-0.f;for(int c=0;c<width;c++){float v=seal?att[off+c]:__fadd_rn(cur[off+c],att[off+c]);cur[off+c]=v;s=__fadd_rn(s,__fmul_rn(v,v));}float scale=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(__fdiv_rn(s,float(width)),1e-6f)));for(int c=0;c<width;c++)normalized[off+c]=__fmul_rn(cur[off+c],scale);}
int main(){for(int rows:{1,33,511}){int width=2304;size_t n=size_t(rows)*width;std::vector<float>cur(n),att(n),expected(n),out(n),sealed(n);for(size_t i=0;i<n;i++){cur[i]=float(int(i%191)-95)/13;att[i]=float(int(i%127)-63)/33;}cur[0]=-0.f;att[0]=-0.f;float *c,*a,*norm,*old;CK(cudaMalloc(&c,n*4));CK(cudaMalloc(&a,n*4));CK(cudaMalloc(&norm,n*4));CK(cudaMalloc(&old,n*4));CK(cudaMemcpy(c,cur.data(),n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(a,att.data(),n*4,cudaMemcpyHostToDevice));for(int step=0;step<3;step++){bool seal=step==1;if(seal){std::swap(norm,old);sealed=expected;}transition<<<(rows+31)/32,32>>>(c,a,norm,rows,width,seal);CK(cudaGetLastError());for(int r=0;r<rows;r++){auto off=size_t(r)*width;float s=-0.f;for(int j=0;j<width;j++){auto i=off+j;float v=seal?att[i]:cur[i]+att[i];cur[i]=v;s+=v*v;}float scale=1.f/std::sqrt(s/float(width)+1e-6f);for(int j=0;j<width;j++)expected[off+j]=cur[off+j]*scale;}CK(cudaMemcpy(out.data(),c,n*4,cudaMemcpyDeviceToHost));if(memcmp(out.data(),cur.data(),n*4))return 3;CK(cudaMemcpy(out.data(),norm,n*4,cudaMemcpyDeviceToHost));if(memcmp(out.data(),expected.data(),n*4))return 4;if(step>=1){CK(cudaMemcpy(out.data(),old,n*4,cudaMemcpyDeviceToHost));if(memcmp(out.data(),sealed.data(),n*4))return 5;}}printf("%d rows x %d: add/seal/add, current, normalized and sealed bits match\n",rows,width);CK(cudaFree(old));CK(cudaFree(norm));CK(cudaFree(a));CK(cudaFree(c));}}
