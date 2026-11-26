#pragma once
// Q8_0 prefill tiles for code built for sm_80 and newer. cp.async copies the BM activation rows and
// the 8*R weight rows of the next K tile into shared memory while the current tile runs, so the
// loads overlap the FMAs. Lane chains, the Q8_0 decode and the final shuffle sum are those of
// inkling_prefetch, so each output is bit-identical to it for every tile shape.

template<int BM,int R,int BK>
constexpr int inkling_staged_bytes(){return 2*BM*BK*int(sizeof(float))+2*8*R*(BK/32)*34;}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void inkling_staged_copy(void *shared,const void *global){
    const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(address),"l"(global):"memory");
}
template<int Pending>
__device__ __forceinline__ void inkling_staged_wait(){asm volatile("cp.async.wait_group %0;"::"n"(Pending):"memory");}
#endif

template<int BM,int R,int BK,int MinBlocks>
__global__ void __launch_bounds__(256,MinBlocks) inkling_prefetch_staged(inference_cuda_dense_gemv_args a){
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int NB=BK/32,WB=NB*34,NR=8*R;
    static_assert(WB%16==0,"a staged weight row is a whole number of 16-byte copies");
    extern __shared__ __align__(16) unsigned char staged_tiles[];
    float *xs=reinterpret_cast<float *>(staged_tiles);                 // [2][BM][BK]
    unsigned char *ws=staged_tiles+2*BM*BK*sizeof(float);              // [2][NR][WB]
    const int t=threadIdx.x,warp=t/32,lane=t%32,n0=blockIdx.x*NR,m0=blockIdx.y*BM;
    const size_t row_bytes=size_t(a.cols/32)*34;
    const auto *w=static_cast<const unsigned char *>(a.w);
    const auto *x=static_cast<const float *>(a.x);
    auto issue=[&](int stage,int kb){
        for(int i=t;i<BM*BK/4;i+=256){
            const int u=i/(BK/4),c=i%(BK/4);
            float *dst=xs+(stage*BM+u)*BK+c*4;
            if(m0+u<a.m)inkling_staged_copy(dst,x+size_t(m0+u)*a.ld_x+kb+c*4);
            else *reinterpret_cast<float4 *>(dst)=make_float4(0.f,0.f,0.f,0.f);
        }
        for(int i=t;i<NR*WB/16;i+=256){
            const int r=i/(WB/16),c=i%(WB/16);
            inkling_staged_copy(ws+(stage*NR+r)*WB+c*16,w+size_t(n0+r)*row_bytes+size_t(kb/32)*34+c*16);
        }
        asm volatile("cp.async.commit_group;":::"memory");
    };
    float sums[R][BM]={};
    const int tiles=a.cols/BK;
    issue(0,0);
    for(int tile=0;tile<tiles;tile++){
        const int s=tile&1;
        if(tile+1<tiles){issue(s^1,(tile+1)*BK);inkling_staged_wait<1>();}else inkling_staged_wait<0>();
        __syncthreads();
        const float *xt=xs+s*BM*BK;
        const unsigned char *wt=ws+(s*NR+warp*R)*WB;
#pragma unroll
        for(int jb=0;jb<NB;jb++){
            float xv[BM];
#pragma unroll
            for(int u=0;u<BM;u++)xv[u]=xt[u*BK+jb*32+lane];
#pragma unroll
            for(int r=0;r<R;r++){
                const float wv=fifi_gguf::decode<8>(wt+r*WB+jb*34,lane);
#pragma unroll
                for(int u=0;u<BM;u++)sums[r][u]=fmaf(wv,xv[u],sums[r][u]);
            }
        }
        __syncthreads();
    }
    const int n=n0+warp*R;
#pragma unroll
    for(int r=0;r<R;r++){
#pragma unroll
        for(int u=0;u<BM;u++){float sum=sums[r][u];
#pragma unroll
            for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
            if(!lane && m0+u<a.m)static_cast<float *>(a.y)[size_t(m0+u)*a.ld_y+n+r]=sum;}}
#else
    __trap();
#endif
}

// Whether this tile runs on the current device. Its loaded code must be sm_80 or newer (a
// compute_75 image JIT-compiled for a newer card holds the trap), and its dynamic shared memory
// is raised once per device. A failed query or raise leaves the device on the shipped tiles.
template<int BM,int R,int BK,int MinBlocks>
bool inkling_staged_ready(){
    static std::atomic<int> state[64];  // 0 unknown, 1 runs, 2 does not run
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess || device<0 || device>=64){(void)cudaGetLastError();return false;}
    int known=state[device].load(std::memory_order_acquire);
    if(!known){
        const auto kernel=inkling_prefetch_staged<BM,R,BK,MinBlocks>;
        cudaFuncAttributes attr={};
        bool runs=cudaFuncGetAttributes(&attr,kernel)==cudaSuccess && attr.ptxVersion>=80 &&
            cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,inkling_staged_bytes<BM,R,BK>())==cudaSuccess;
        if(!runs)(void)cudaGetLastError();
        known=runs?1:2;
        state[device].store(known,std::memory_order_release);
    }
    return known==1;
}

// The Q8_0 rows the 16-row shipped tiles serve (63/64 and the large chunks), when the weight rows
// and activations are whole 16-byte copies. True means a kernel was enqueued: the caller checks
// its launch status exactly once.
inline bool inkling_staged_select(const inference_cuda_dense_gemv_args &a,int codec,cudaStream_t s){
    if(codec!=8 || !(a.m==63 || a.m==64 || a.m==127 || a.m==128 || a.m==255 || a.m==256))return false;
    if(!aligned(a.w,16) || !aligned(a.x,16) || a.ld_x%4 || a.cols%256 || a.rows%32)return false;
#define STAGED(R,MB) {if(!inkling_staged_ready<16,R,256,MB>())return false; \
    inkling_prefetch_staged<16,R,256,MB><<<dim3(a.rows/(8*R),(a.m+15)/16),256,inkling_staged_bytes<16,R,256>(),s>>>(a);return true;}
    // Narrow projections take half-width CTAs, two per SM, so the grid still fills the card.
    if(a.rows>=1024 && a.rows%64==0)STAGED(8,1)
    STAGED(4,2)
#undef STAGED
}
