// Each of nine warps covers 256 columns of a Q8_0 row. Four lanes
// share a scale and each handles two aligned float4 activation vectors.
__global__ void decode_q8_kblock(inference_cuda_dense_gemv_args a) {
    __shared__ float partial[9];
    const int tile = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int block = 8 * tile + lane / 4, j = lane % 4;
    const auto *w = static_cast<const uint8_t *>(a.w) + (size_t(blockIdx.x) * 72 + block) * 34;
    const auto *q = reinterpret_cast<const uint16_t *>(w + 2 + 8*j);
    const auto *x = static_cast<const float4 *>(a.x) + block * 8 + 2*j;
    const float4 u = x[0], h = x[1];
    const unsigned l = unsigned(q[0]) | (unsigned(q[1]) << 16);
    const unsigned r = unsigned(q[2]) | (unsigned(q[3]) << 16);
    float acc = 0;
#pragma unroll
    for (int p = 0; p < 4; p++) {
        acc = fmaf(float(int8_t(l >> (8*p))), reinterpret_cast<const float *>(&u)[p], acc);
        acc = fmaf(float(int8_t(r >> (8*p))), reinterpret_cast<const float *>(&h)[p], acc);
    }
    acc *= __half2float(*reinterpret_cast<const __half *>(w));
    acc = warp_sum(acc);
    if (!lane) partial[tile] = acc;
    __syncthreads();
    if (!tile) {
        float s = lane < 9 ? partial[lane] : 0.f;
        s = warp_sum(s);
        if (!lane) static_cast<float *>(a.y)[blockIdx.x] = s;
    }
}
