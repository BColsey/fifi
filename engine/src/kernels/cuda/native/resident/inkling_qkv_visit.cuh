#include "resident/inkling_prefetch.cuh"
// Consecutive independent Q8 projections, possibly at different offsets in
// wider strided input/output rows. Flatten the group's output tiles in grid.x.
struct InklingQkvGroup {
    inference_cuda_dense_gemv_args a[4];
    unsigned end[4];
};
__global__ void inkling_qkv_visit(InklingQkvGroup group){
    unsigned bx=blockIdx.x,i=bx<group.end[0]?0:bx<group.end[1]?1:bx<group.end[2]?2:3;
    // Avoid taking an address of the parameter array (local-memory copy).
    __shared__ float tile[8][128];
    if(i==0) inkling_visit_dot<8,32,34,4>(group.a[0],bx,tile);
    else if(i==1) inkling_visit_dot<8,32,34,4>(group.a[1],bx-group.end[0],tile);
    else if(i==2) inkling_visit_dot<8,32,34,4>(group.a[2],bx-group.end[1],tile);
    else inkling_visit_dot<8,32,34,4>(group.a[3],bx-group.end[2],tile);
}
__global__ void inkling_qkv_prefetch(InklingQkvGroup group){
    const unsigned bx=blockIdx.x;
    if(bx<group.end[0])inkling_prefetch_body<8,8,2,128,true>(group.a[0],bx);
    else if(bx<group.end[1])inkling_prefetch_body<8,8,2,128,true>(group.a[1],bx-group.end[0]);
    else if(bx<group.end[2])inkling_prefetch_body<8,8,2,128,true>(group.a[2],bx-group.end[1]);
    else inkling_prefetch_body<8,8,2,128,true>(group.a[3],bx-group.end[2]);
}
