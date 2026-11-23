// Component probe of the next device consumer of transition.cu's normalized plane.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}} while(0)
constexpr int Width=2304, MaxDepth=9;

struct View {
    const float* sealed[MaxDepth-1]; // oldest first; immutable
    const float* current;
    int count, first;
    __host__ __device__ const float* plane(int d) const {
        return d < count-first ? sealed[count-1-d] : current;
    }
    __host__ __device__ int depth() const { return count-first+1; }
};

// One thread owns each dot, so K is reduced in scalar CPU order. Block threads
// then share that row's scores and mix independent output columns in parallel.
__global__ void depth_mix(View hist, const float* query, float scale, float* y, float* weights,
                          float* scores, const float* reference_weights = nullptr) {
    __shared__ float score[MaxDepth], weight[MaxDepth], mass;
    int r=blockIdx.x, lane=threadIdx.x;
    size_t off=size_t(r)*Width;
    if(lane<hist.depth()) {
        const float* x=hist.plane(lane)+off;
        float sum=-0.f;
        for(int k=0;k<Width;++k) sum=__fadd_rn(sum,__fmul_rn(x[k],query[k]));
        score[lane]=__fmul_rn(sum,scale);
        scores[r*hist.depth()+lane]=score[lane];
    }
    __syncthreads();
    if(lane==0) {
        float peak=-INFINITY;
        for(int d=0;d<hist.depth();++d) peak=fmaxf(peak,score[d]);
        float sum=0.f;
        for(int d=0;d<hist.depth();++d) {
            auto at=r*hist.depth()+d;
            const float delta=__fsub_rn(score[d],peak);
#ifdef PROBE_EXP_F64
            const float value=__double2float_rn(exp(static_cast<double>(delta)));
#else
            const float value=expf(delta);
#endif
            weight[d]=reference_weights ? reference_weights[at] : value;
            weights[at]=weight[d];
            sum=__fadd_rn(sum,weight[d]);
        }
        mass=sum;
    }
    __syncthreads();
    for(int k=lane;k<Width;k+=blockDim.x) {
        float sum=0.f;
        for(int d=0;d<hist.depth();++d)
            sum=__fadd_rn(sum,__fmul_rn(weight[d],hist.plane(d)[off+k]));
        y[off+k]=__fdiv_rn(sum,mass);
    }
}

struct Differences {
    size_t bits=0;
    float max_abs=0;
    void compare(float a,float b) {
        if(!std::isfinite(a)||!std::isfinite(b)) std::exit(3);
        bits+=std::memcmp(&a,&b,sizeof(float))!=0;
        max_abs=std::max(max_abs,std::abs(a-b));
    }
};

void check(int rows,int first,const std::vector<float>& history,const float* resident,
           const std::vector<float>& q,const float* query,float* y,float* weights) {
    size_t n=size_t(rows)*Width;
    View h{}, dev{};
    h.count=dev.count=MaxDepth-1;
    h.first=dev.first=first;
    for(int d=0;d<h.count;++d){h.sealed[d]=history.data()+d*n;dev.sealed[d]=resident+d*n;}
    h.current=history.data()+h.count*n;
    dev.current=resident+dev.count*n;
    size_t count=rows*h.depth(), bytes=count*sizeof(float);
    std::vector<float> output(n), expected(n), gpu_weights(count), gpu_scores(count), host_weights(count);
    float *scores,*override_weights;
    CK(cudaMalloc(&scores,bytes));
    CK(cudaMalloc(&override_weights,bytes));
    float scale=1.f/std::sqrt(float(Width));
    depth_mix<<<rows,128>>>(dev,query,scale,y,weights,scores);
    CK(cudaGetLastError());
    CK(cudaMemcpy(output.data(),y,n*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpu_weights.data(),weights,gpu_weights.size()*sizeof(float),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gpu_scores.data(),scores,bytes,cudaMemcpyDeviceToHost));
    Differences dot,w,out;
    for(int r=0;r<rows;++r) {
        int off=r*Width;
        float scores[MaxDepth], soft[MaxDepth], peak=-INFINITY, mass=0.f;
        for(int d=0;d<h.depth();++d) {
            float sum=-0.f;
            for(int k=0;k<Width;++k) sum+=h.plane(d)[off+k]*q[k];
            scores[d]=sum*scale;
            dot.compare(scores[d],gpu_scores[r*h.depth()+d]);
            peak=std::max(peak,scores[d]);
        }
        for(int d=0;d<h.depth();++d) {
            soft[d]=std::exp(scores[d]-peak);
            mass+=soft[d];
            host_weights[r*h.depth()+d]=soft[d];
            w.compare(soft[d],gpu_weights[r*h.depth()+d]);
        }
        for(int k=0;k<Width;++k) {
            float sum=0.f;
            for(int d=0;d<h.depth();++d) sum+=soft[d]*h.plane(d)[off+k];
            expected[off+k]=sum/mass;
            out.compare(expected[off+k], output[off+k]);
        }
    }
    printf("rows=%d depth=%d first=%d weights_unequal=%zu/%zu max_weight_delta=%.9g output_unequal=%zu/%zu max_output_delta=%.9g\n",
           rows,h.depth(),first,w.bits,gpu_weights.size(),w.max_abs,out.bits,n,out.max_abs);
    if(dot.bits || out.max_abs>1e-6f || (h.depth()==1 && out.bits)) std::exit(4);
    // Fault localization, not a proposed serving path: inject CPU weights and
    // leave device scores, mass and weighted accumulation otherwise unchanged.
    CK(cudaMemcpy(override_weights,host_weights.data(),bytes,cudaMemcpyHostToDevice));
    depth_mix<<<rows,128>>>(dev,query,scale,y,weights,scores,override_weights);
    CK(cudaGetLastError());
    CK(cudaMemcpy(output.data(),y,n*sizeof(float),cudaMemcpyDeviceToHost));
    if(std::memcmp(output.data(),expected.data(),n*sizeof(float))) std::exit(5);
    printf("  scores and CPU-weight replay bit-identical\n");
    CK(cudaFree(override_weights));CK(cudaFree(scores));
}
int main() {
    unsigned seed=15;
    auto val=[&](){seed=seed*1664525u+1013904223u;return float(seed>>8)*0x1p-23f-1.f;};
    std::vector<float> q(Width);
    for(float&f:q) f=val()*3.f;
    float *query;
    CK(cudaMalloc(&query,Width*sizeof(float)));
    CK(cudaMemcpy(query,q.data(),Width*sizeof(float),cudaMemcpyHostToDevice));
    for(int rows:{1,33,511}) {
        size_t n=size_t(rows)*Width;
        std::vector<float> hist(n*MaxDepth);
        for(size_t off=0;off<hist.size();off+=Width) {
            float s=0;
            for(int k=0;k<Width;++k){float f=val();hist[off+k]=f;s+=f*f;}
            float scale=1.f/std::sqrt(s/Width+1e-6f);
            for(int k=0;k<Width;++k) hist[off+k]*=scale;
        }
        float *resident,*y,*weights;
        CK(cudaMalloc(&resident,hist.size()*sizeof(float)));
        CK(cudaMalloc(&y,n*sizeof(float)));
        CK(cudaMalloc(&weights,MaxDepth*rows*sizeof(float)));
        CK(cudaMemcpy(resident,hist.data(),hist.size()*sizeof(float),cudaMemcpyHostToDevice));
        for(int first:{8,7,3,0}) check(rows,first,hist,resident,q,query,y,weights);
        CK(cudaFree(resident));CK(cudaFree(y));CK(cudaFree(weights));
    }
    CK(cudaFree(query));
}
