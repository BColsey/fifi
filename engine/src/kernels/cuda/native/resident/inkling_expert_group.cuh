// The group is embedded in kernel parameters (no upload, workspace or
// dynamic schedulers). grid.y flattens only live token tiles across experts.
struct InklingExpertGroup { inference_cuda_dense_gemv_args calls[16]; int ends[16]; };
template<int Codec,int K,int Block,int Bytes>
__global__ void inkling_expert_group(const __grid_constant__ InklingExpertGroup group) {
    int expert=0;while(blockIdx.y>=group.ends[expert]) ++expert;
    int tile=int(blockIdx.y)-(expert?group.ends[expert-1]:0);
    inkling_iq2_visit_body<Codec,K,Block,Bytes,true>(group.calls[expert],tile*32);
}
template<int C,int BM,int R>
__global__ void inkling_ragged_exact(const __grid_constant__ InklingExpertGroup group){
    int e=0;while(blockIdx.y>=group.ends[e])++e;
    int tile=int(blockIdx.y)-(e?group.ends[e-1]:0);
    inkling_prefetch_body<C,BM,R,128,true>(group.calls[e],blockIdx.x,tile);
}
