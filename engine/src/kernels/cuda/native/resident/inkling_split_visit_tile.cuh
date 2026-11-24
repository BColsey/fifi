// SM75: 32 weights x 16 activations. Eight warp pairs own disjoint K slices.
// Same CTA reduces the FP32 partials; no workspace, atomics, or second launch.
template<int Codec, int Block, int Bytes>
__global__ void inkling_split_visit_tile(inference_cuda_dense_gemv_args a) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    constexpr int pitch = 136;
    __shared__ __align__(32) union { struct { __half x[16*pitch], w[32*pitch]; } stage; float part[8][16*32]; } sm;
    int warp = threadIdx.x/32, g = warp/2, n0 = blockIdx.x*32, m0 = blockIdx.y*16;
    WmmaAccumulator c; fill_fragment(c,0.f);
    auto *w = static_cast<const uint8_t *>(a.w);
    for (int base=0; base<a.cols; base+=128) {
#pragma unroll
        for(int i=threadIdx.x;i<32*128;i+=512){
            int r=i/128,k=i%128;
            sm.stage.w[r*pitch+k]=wmma_weight<Codec,Block,Bytes>(w+size_t(n0+r)*(a.cols/Block*Bytes),base+k);
        }
#pragma unroll
        for(int i=threadIdx.x;i<16*128;i+=512){
            int m=m0+i/128,k=i%128;
            sm.stage.x[i/128*pitch+k]=__float2half_rn(m<a.m ? static_cast<const float *>(a.x)[m*a.ld_x+base+k] : 0.f);
        }
        __syncthreads();
        fragment<matrix_a,16,16,16,__half,row_major> x;
        fragment<matrix_b,16,16,16,__half,col_major> b;
        load_matrix_sync(x,sm.stage.x+16*g,pitch);
        load_matrix_sync(b,sm.stage.w+(warp%2)*16*pitch+16*g,pitch);
        mma_sync(c,x,b,c); __syncthreads();
    }
    store_matrix_sync(sm.part[g]+(warp%2)*16,c,32,mem_row_major); __syncthreads();
    int m=m0+threadIdx.x/32;
    if(m<a.m){
        float val=sm.part[0][threadIdx.x];
#pragma unroll
        for(int i=1;i<8;++i) val+=sm.part[i][threadIdx.x];
        static_cast<float *>(a.y)[m*a.ld_y+n0+threadIdx.x%32]=val;
    }
#endif
}
