// Two independent 64 -> 2304 F32 projections (controller gain/forget).
// Match dense_gemv_k<float,true>: sixteen lanes each dot four samples,
// then the same XOR reduction, except for the redundant sum with zero.
__global__ void decode_control_pair(inference_cuda_dense_gemv_args first,
                                    inference_cuda_dense_gemv_args second) {
    const auto a = blockIdx.y == 0 ? first : second;
    const int lane = threadIdx.x % 16, row = blockIdx.x * 16 + threadIdx.x / 16;
    const auto *w = static_cast<const float4 *>(a.w) + row*16;
    const float4 v = w[lane];
    const auto *x = static_cast<const float *>(a.x) + lane*4;
    float s = fmaf(v.x,x[0],0.f);
    s = fmaf(v.y,x[1],s);
    s = fmaf(v.z,x[2],s);
    s = fmaf(v.w,x[3],s);
#pragma unroll
    for (int offset=8; offset; offset/=2) s += __shfl_xor_sync(0xffffffff,s,offset,16);
    if (!lane) static_cast<float *>(a.y)[row] = s;
}
