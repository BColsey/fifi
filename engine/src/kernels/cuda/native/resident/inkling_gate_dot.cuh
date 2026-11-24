// Included by dense_gemv.cu inside its implementation namespace.
// Pack each group of eight activations once, then reuse it across all expert rows.
struct GateDigits {
    int lo[3], hi[3];
    float scale;
};

__global__ void inkling_gate_pack(const float *x, GateDigits *out) {
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= kInklingGateWidth / 8) return;
    const float *v = x + group * 8;
    float peak = 0;
    bool finite = true;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        finite &= isfinite(v[j]);
        peak = fmaxf(peak, fabsf(v[j]));
    }
    GateDigits result{};
    if (!finite) {
        result.scale = __int_as_float(0x7fc00000);
    } else if (peak > 0) {
        result.scale = peak * (1.0f / (127.0f * 16384.0f));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float lo = (v[j] / peak) * 127.0f;
            float hi = (v[4 + j] / peak) * 127.0f;
#pragma unroll
            for (int digit = 0; digit < 3; ++digit) {
                const int qlo = __float2int_rn(lo), qhi = __float2int_rn(hi);
                result.lo[digit] |= int(unsigned(uint8_t(qlo)) << (8 * j));
                result.hi[digit] |= int(unsigned(uint8_t(qhi)) << (8 * j));
                lo = (lo - qlo) * 128.0f;
                hi = (hi - qhi) * 128.0f;
            }
        }
    }
    out[group] = result;
}

__device__ __forceinline__ int inkling_signed_grid(uint32_t grid, unsigned signs) {
    const uint32_t mask = __byte_perm(0, 0xffffffff,
        ((signs & 1) << 2) | ((signs & 2) << 5) | ((signs & 4) << 8) | ((signs & 8) << 11));
    // The codebook magnitudes are positive, so bytewise +1 cannot carry.
    return int((grid ^ mask) + (mask & 0x01010101));
}

struct GateGrid {
    int lo, hi;
    float scale;
};

template<int Type>
__device__ __forceinline__ GateGrid inkling_grid(const uint8_t *block, int lane) {
    const float d = fifi_gguf::half_value(block);
    const int row = lane / 4, segment = lane % 4;
    if constexpr (Type == 19) {
        const unsigned high = fifi_gguf::read16(block + 34 + 2 * row);
        const unsigned code = block[2 + lane] | (((high >> (3 * segment)) & 7) << 8);
        const auto grid = iq1s_grid_gpu[code];
        // The decoded value is (code - 1) ± 1/8. Scale by eight before DP4A.
        const unsigned delta = high & 0x8000 ? 0x09090909u : 0x07070707u;
        return {int(__vsub4((grid & 0x0f0f0f0fu) << 3, delta)),
                int(__vsub4((grid & 0xf0f0f0f0u) >> 1, delta)),
                d * (0.125f * float(1 + 2 * ((high >> 12) & 7)))};
    } else {
        const auto g = fifi_gguf::iq2_group<Type>(block, row, segment, d);
        return {inkling_signed_grid(uint32_t(g.grid), g.signs),
                inkling_signed_grid(uint32_t(g.grid >> 32), g.signs >> 4), g.scale};
    }
}

template<int Type, int Bytes>
__global__ void inkling_gate_dot(const GateDigits *packed, const uint8_t *gate,
                                 float *out, DeviceSelectedWeights index) {
    constexpr int width = kInklingGateWidth, blocks = width / 256;
    __shared__ GateDigits activations[width / 8];
    const int tid = threadIdx.x;
    for (int group = tid; group < width / 8; group += blockDim.x)
        activations[group] = packed[group];
    __syncthreads();

    const int lane = tid & 31;
    const int row = blockIdx.x * kWarps + (tid >> 5);
    const auto *w = gate + index.offset(blockIdx.y) + size_t(row) * blocks * Bytes;
    float sum = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto g = inkling_grid<Type>(w + block * Bytes, lane);
        const auto a = activations[block * 32 + lane];
        int dot = 0;
        // Radix-128 residuals have magnitude <=64; even with magnitude 127 for
        // the leading digit, eight weights <=62 cannot overflow the signed sum.
#pragma unroll
        for (int digit = 0; digit < 3; ++digit) {
            const int term = __dp4a(g.hi, a.hi[digit], __dp4a(g.lo, a.lo[digit], 0));
            dot = dot * 128 + term;
        }
        sum = fmaf(float(dot), a.scale * g.scale, sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) out[size_t(blockIdx.y) * width + row] = sum;
}

#include "resident/inkling_decode_experts.cuh"
#include "resident/inkling_gate_cached.cuh"

template<int Type, int Bytes>
void launch_decode_gate(const GateDigits *packed, const uint8_t *gate, float *out,
                        DeviceSelectedWeights index, cudaStream_t stream) {
    const dim3 grid(2304 / 16, 8);
    if constexpr (Type == 19) decode_gate_pair<Type, Bytes><<<grid,256,0,stream>>>(packed,gate,out,index);
    else decode_gate_cached<Type, Bytes><<<dim3(48,8),256,0,stream>>>(packed,gate,out,index);
}

bool launch_inkling_gate_dot(const inference_cuda_gguf_moe_args &a, DeviceSelectedWeights index,
                             float *temporary, cudaStream_t stream, Failure &failure) {
    constexpr int width = kInklingGateWidth;
    if (a.hidden != width || 2 * a.middle != width || a.count != 8) return false;
    const int type = a.gate_dtype - fifi_gguf::kDtypeBase;
    if (type != 16 && type != 17 && type != 19 && type != 22) return false;
    // The middle rows are not live until activate_moe, after this gate completes.
    static_assert((width / 8) * sizeof(GateDigits) <= 8 * (width / 2) * sizeof(float));
    auto *packed = reinterpret_cast<GateDigits *>(temporary);
    inkling_gate_pack<<<3,128,0,stream>>>(a.x, packed);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return true;
    const dim3 grid(width / kWarps, a.count);
#define GATE_DOT(id, bytes) \
    case id: if (decode_split_enabled()) \
        launch_decode_gate<id,bytes>(packed, static_cast<const uint8_t *>(a.gate), a.scratch, index, stream); \
        else inkling_gate_dot<id, bytes><<<grid, kWarps * 32, 0, stream>>>( \
            packed, static_cast<const uint8_t *>(a.gate), a.scratch, index); break;
    switch (type) {
        GATE_DOT(16,66)
        GATE_DOT(17,74)
        GATE_DOT(19,50)
        GATE_DOT(22,82)
    }
#undef GATE_DOT
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return true;
}
