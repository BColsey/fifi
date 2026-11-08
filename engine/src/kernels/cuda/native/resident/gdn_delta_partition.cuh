// Fifi GDN state-partition experiment. Serving keeps the vendored kernel.
#ifndef INFERENCE_CUDA_GDN_DELTA_PARTITION_CUH
#define INFERENCE_CUDA_GDN_DELTA_PARTITION_CUH

#include "ops.h"

namespace gdn_partition {

__device__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float red[8];
    v = warp_sum(v);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();  // prior calls may still be reading red
    if (lane == 0) red[warp] = v;
    __syncthreads();
    return warp_sum(lane < 8 ? red[lane] : 0.f);
}

template <bool Snap>
__global__ void __launch_bounds__(256) delta_k(kern::DeltaArgs a) {
    constexpr int D = kern::kStateDim, Rows = D / 2;
    static_assert(D == 128, "partition is qualified only for D128");
    const int h = blockIdx.x, tid = threadIdx.x, j = tid % D, first = (tid / D) * Rows;
    const int kh = h % a.n_k_heads;
    __shared__ float qs[D], ks[D], partial[2 * D];
    float *st = a.state + (long)h * D * D;
    float s[Rows];
#pragma unroll
    for (int i = 0; i < Rows; ++i) s[i] = st[(first + i) * D + j];
    for (int t = 0; t < a.m; ++t) {
        const float *c = a.conv + (long)t * a.conv_ld;
        const float *p = a.proj + (long)t * a.ld;
        const float qv = tid < D ? c[kh * D + j] : 0.f;
        const float kv = tid < D ? c[(a.n_k_heads + kh) * D + j] : 0.f;
        const float qn = rsqrtf(block_sum(qv * qv) + a.eps) * rsqrtf((float)D);
        const float kn = rsqrtf(block_sum(kv * kv) + a.eps);
        if (tid < D) {
            qs[j] = qv * qn;
            ks[j] = kv * kn;
        }
        const float alpha = p[a.alpha_off + h] + a.dt_bias[h];
        const float sp = alpha > 20.f ? alpha : log1pf(__expf(alpha));
        const float decay = __expf(sp * a.a[h]);
        const float beta = 1.f / (1.f + __expf(-p[a.beta_off + h]));
        const float vv = c[(2 * a.n_k_heads + h) * D + j];
        __syncthreads();
        float sk = 0.f;
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            s[i] *= decay;
            sk += s[i] * ks[first + i];
        }
        partial[tid] = sk;
        __syncthreads();
        sk = partial[j] + partial[D + j];
        __syncthreads();  // both halves must read sk before partial is reused for o
        const float d = (vv - sk) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            s[i] += ks[first + i] * d;
            o += s[i] * qs[first + i];
        }
        partial[tid] = o;
        __syncthreads();
        o = partial[j] + partial[D + j];
        const float rstd = rsqrtf(block_sum(tid < D ? o * o : 0.f) / (float)D + a.eps);
        if (tid < D) {
            const float z = p[a.z_off + h * D + j];
            a.out[(long)t * a.n_v_heads * D + h * D + j] =
                o * rstd * a.norm_w[j] * (z / (1.f + __expf(-z)));
        }
        if (Snap && t < a.m - 1) {
            float *snap = (float *)((char *)a.snap + (size_t)t * a.snap_stride) + (long)h * D * D;
#pragma unroll
            for (int i = 0; i < Rows; ++i) snap[(first + i) * D + j] = s[i];
        }
    }
#pragma unroll
    for (int i = 0; i < Rows; ++i) st[(first + i) * D + j] = s[i];
}

inline cudaError_t launch(const kern::DeltaArgs &a, cudaStream_t stream) {
    if (a.snap) delta_k<true><<<a.n_v_heads, 256, 0, stream>>>(a);
    else delta_k<false><<<a.n_v_heads, 256, 0, stream>>>(a);
    return cudaGetLastError();
}

}  // namespace gdn_partition
#endif
