// Decode only: assign two rows to each warp, loading each activation digit
// once for both independent chains. Keep the existing gate pack and dot order.
template<int Type, int Bytes>
__global__ void decode_gate_pair(const GateDigits *packed, const uint8_t *gate,
                                 float *out, DeviceSelectedWeights index) {
    constexpr int width = kInklingGateWidth, blocks = width / 256;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * 16 + threadIdx.x / 32;
    const auto *w = gate + index.offset(blockIdx.y) + size_t(row)*blocks*Bytes;
    float sums[2] = {};
    for (int k=0; k<blocks; ++k) {
        const GateDigits a = packed[k*32 + lane];
#pragma unroll
        for (int r=0; r<2; ++r) {
            const auto g = inkling_grid<Type>(w+(k+r*8*blocks)*Bytes,lane);
            int dot=0;
#pragma unroll
            for (int j=0;j<3;++j)
                dot=128*dot + __dp4a(g.hi,a.hi[j],__dp4a(g.lo,a.lo[j],0));
            sums[r]=fmaf(float(dot),a.scale*g.scale,sums[r]);
        }
    }
#pragma unroll
    for (int r=0; r<2; ++r) {
        float v=warp_sum(sums[r]);
        if (!lane) out[size_t(blockIdx.y)*width+row+r*8]=v;
    }
}

__global__ void decode_q2_pair(const uint8_t *down,const float *x,float *out,DeviceSelectedWeights idx) {
    constexpr int rows=2304, cols=1152;
    int row=blockIdx.x*32+threadIdx.x/8, lane=threadIdx.x%8;
    auto *w=down+idx.offset(blockIdx.y)+size_t(row)*(cols/64)*18;
    const auto *v=reinterpret_cast<const float4 *>(x+size_t(blockIdx.y)*cols);
    float acc[4]={};
#pragma unroll 3
    for (int k=0;k<cols/64;++k) {
        const uint8_t *b=w+k*18;
        unsigned q=b[2+lane], q1=b[10+lane];
        float d=__half2float(*reinterpret_cast<const __half *>(b));
        float4 lo=v[16*k+lane], hi=v[16*k+lane+8];
#pragma unroll
        for(int j=0;j<4;++j) {
            acc[j]=fmaf(d*float(int((q>>(2*j))&3)-1),reinterpret_cast<float *>(&lo)[j],acc[j]);
            acc[j]=fmaf(d*float(int((q1>>(2*j))&3)-1),reinterpret_cast<float *>(&hi)[j],acc[j]);
        }
    }
#pragma unroll
    for (int o=4;o;o/=2) {
#pragma unroll
        for (int j=0;j<4;++j) acc[j]+=__shfl_xor_sync(0xffffffff,acc[j],o,8);
    }
    if(!lane) out[size_t(blockIdx.y)*rows+row]=(acc[0]+acc[2])+(acc[1]+acc[3]);
}
