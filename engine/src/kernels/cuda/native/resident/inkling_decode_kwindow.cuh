// Each thread owns eight contiguous activations, so the window scale and
// minimum apply once per partial sum. Three warps split the K blocks evenly.
__device__ __forceinline__ unsigned kw_read32(const uint8_t *p) {
    const auto *v = reinterpret_cast<const uint16_t *>(p);
    return unsigned(v[0]) | (unsigned(v[1]) << 16);
}

template<int Type, int Bytes, int Cols>
__global__ void decode_kwindow(inference_cuda_dense_gemv_args a) {
    __shared__ float part[2][3];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = 2*blockIdx.x + warp/3, win = lane/4, c = 8*(lane%4);
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row)*(Cols/256)*Bytes;
    const auto *x = static_cast<const float4 *>(a.x) + 2*lane;
    float acc = 0;
#pragma unroll
    for (int k = warp%3; k < Cols/256; k += 3) {
        const auto *b = w + k*Bytes;
        unsigned codes[2];
        float scale, minimum;
        if constexpr (Type == 14) {
            const int half = win / 4, p = win % 4;
            const auto *l = b + 64*half + 32*(p%2) + c;
            const auto *h = b + 128 + 32*half + c;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                unsigned lo = (kw_read32(l+4*j) >> (4*(p/2))) & 0x0f0f0f0f;
                unsigned hi = ((kw_read32(h+4*j) >> (2*p)) & 0x03030303) << 4;
                codes[j] = __vsub4(lo|hi, 0x20202020u);
            }
            scale = __half2float(*reinterpret_cast<const __half *>(b+208))*float(int8_t(b[192+lane/2]));
            minimum = 0;
        } else {
            const auto *sc = b + 4;
            unsigned s = win < 4 ? sc[win]&63u : (sc[win+4]&15u)|((sc[win-4]>>6)<<4);
            unsigned m = win < 4 ? sc[win+4]&63u : (sc[win+4]>>4)|((sc[win]>>6)<<4);
            scale = __half2float(*reinterpret_cast<const __half *>(b))*float(s);
            minimum = __half2float(*reinterpret_cast<const __half *>(b+2))*float(m);
            const auto *l = b + (Type==13 ? 48 : 16) + 32*(win/2) + c;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                codes[j] = (*reinterpret_cast<const unsigned *>(l+4*j) >> (4*(win%2))) & 0x0f0f0f0f;
                if constexpr (Type==13) codes[j] |= ((*reinterpret_cast<const unsigned *>(b+16+c+4*j) >> win) & 0x01010101) << 4;
            }
        }
        float dot=0, sum=0;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            float4 v=x[k*64+j];
#pragma unroll
            for (int i=0;i<4;i++) {
                float u=reinterpret_cast<float *>(&v)[i];
                dot=fmaf(float(int8_t(codes[j]>>(8*i))),u,dot);
                if constexpr(Type!=14) sum+=u;
            }
        }
        acc=fmaf(dot,scale,acc);
        if constexpr(Type!=14) acc=fmaf(-minimum,sum,acc);
    }
    acc=warp_sum(acc);
    if(!lane) part[warp/3][warp%3]=acc;
    __syncthreads();
    if(warp%3==0 && !lane) {
        auto *p=part[warp/3]; static_cast<float *>(a.y)[row]=(p[0]+p[1])+p[2];
    }
}
