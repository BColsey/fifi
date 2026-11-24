// Two F32 controller heads share one activation panel. Compute eight token
// rows per CTA, retaining the vector dense GEMV four-FMA and butterfly order.
__global__ void inkling_controller_pair(inference_cuda_dense_gemv_args a,
                                        inference_cuda_dense_gemv_args b) {
    __shared__ float x[8][64];
    const int t=threadIdx.x, lane=t%16, m0=blockIdx.y*8;
    const auto *weight=static_cast<const float4 *>(blockIdx.z?b.w:a.w);
    auto *output=static_cast<float *>(blockIdx.z?b.y:a.y);
    const int ld=blockIdx.z?b.ld_y:a.ld_y;
    for(int i=t;i<512;i+=256){
        int m=m0+i/64;
        x[i/64][i%64]=m<a.m?static_cast<const float *>(a.x)[size_t(m)*a.ld_x+i%64]:0.f;
    }
    __syncthreads();
    int r=blockIdx.x*16+t/16;
    const float4 v=weight[r*16+lane];
#pragma unroll
    for(int j=0;j<8;++j){
        float s=fmaf(v.x,x[j][lane*4],0.f);
        s=fmaf(v.y,x[j][lane*4+1],s);s=fmaf(v.z,x[j][lane*4+2],s);s=fmaf(v.w,x[j][lane*4+3],s);
        s=__fadd_rn(s,0.f);
#pragma unroll
        for(int d=8;d;d/=2) s+=__shfl_xor_sync(0xffffffff,s,d,16);
        if(!lane && m0+j<a.m) output[size_t(m0+j)*ld+r]=s;
    }
}
