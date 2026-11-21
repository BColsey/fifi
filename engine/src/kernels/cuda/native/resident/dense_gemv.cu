#include "../expert_math.cuh"
// Dense matvec for floating-point or packed GGUF weights (see resident/dense_gemv.h).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/dense_gemv.h"
#include "resident/gguf_gemv.cuh"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxRows = 1 << 20;
constexpr int kMaxCols = 12288;
constexpr int kMaxLd = 1 << 20;
constexpr int kWarps = 8;
constexpr int kDtypeF16 = 0;
constexpr int kDtypeBf16 = 1;
constexpr int kDtypeGgufNvfp4 = 2;
constexpr int kDtypeF32 = 3;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__half v) { return __half2float(v); }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// grid (ceil(rows / kWarps), m); dynamic shared memory holds one row of x.
template <typename W, bool kVector>
__global__ void dense_gemv_k(const W *__restrict__ w, const float *__restrict__ x,
                             float *__restrict__ y, int rows, int cols, int ld_x, int ld_y) {
    extern __shared__ float staged[];
    const int t = blockIdx.y;
    const float *in = x + (size_t)t * ld_x;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) staged[c] = in[c];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (r >= rows) return;
    const W *row = w + (size_t)r * cols;
    float acc = 0.0f;
    if (kVector) {
        constexpr int width = sizeof(uint4) / sizeof(W);
        const uint4 *wide = reinterpret_cast<const uint4 *>(row);
        for (int v = lane; v < cols / width; v += 32) {
            const uint4 raw = wide[v];
            const W *e = reinterpret_cast<const W *>(&raw);
#pragma unroll
            for (int j = 0; j < width; ++j) acc = fmaf(widen(e[j]), staged[v * width + j], acc);
        }
    } else {
        for (int c = lane; c < cols; c += 32) acc = fmaf(widen(row[c]), staged[c], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[(size_t)t * ld_y + r] = acc;
}

// Each lane reads one packed byte and consumes both nibbles of its GGUF block.
__device__ float gguf_scale(unsigned code) {
    if (code == 0x7f) return 0.0f;
    const unsigned exp = (code >> 3) & 15u, frac = code & 7u;
    return exp == 0 ? float(frac) / 512.0f
                    : __int_as_float(((exp + 120u) << 23) | (frac << 20));
}

__device__ float gguf_fp4(unsigned code) {
    const unsigned m = code & 7u;
    const float v = m < 2 ? float(m) * 0.5f :
        __int_as_float(((126u + (m >> 1)) << 23) | ((m & 1u) << 22));
    return code > 8 ? -v : v;
}

struct SharedWeights {
    __device__ size_t offset(unsigned) const { return 0; }
};

struct SelectedWeights {
    size_t stride;
    uint32_t ids[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];

    __device__ size_t offset(unsigned slot) const { return stride * ids[slot]; }
};

struct DeviceSelectedWeights {
    size_t stride;
    const int *ids;
    unsigned experts;

    __device__ size_t offset(unsigned slot) const {
        const unsigned id = unsigned(ids[slot]);
        // Faulty router rows publish -1. Bind them to a safe matrix; the sum reports NaN.
        return id < experts ? stride * id : 0;
    }
};

struct SeparateWeights {
    const void *matrices[INFERENCE_CUDA_GGUF_MOE_MAX_PICKS];
};

template <typename Index>
__device__ __forceinline__ const uint8_t *matrix_base(const void *base, Index index, unsigned slot) {
    return static_cast<const uint8_t *>(base) + index.offset(slot);
}

__device__ __forceinline__ const uint8_t *matrix_base(const void *, SeparateWeights index, unsigned slot) {
    return static_cast<const uint8_t *>(index.matrices[slot]);
}

template <typename Index>
__global__ void gguf_nvfp4_gemv(inference_cuda_dense_gemv_args a, Index index) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const float *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    const int blocks = a.cols / 64;
    const auto *w = matrix_base(a.w, index, blockIdx.y) + size_t(row) * blocks * 36;
    const int group = lane / 8;
    float sum = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto *b = w + block * 36;
        const float scale = gguf_scale(b[group]);
        const unsigned pair = b[4 + lane];
        const int col = block * 64 + group * 16 + lane % 8;
        sum = fmaf(gguf_fp4(pair & 15u) * scale, x[col], sum);
        sum = fmaf(gguf_fp4(pair >> 4) * scale, x[col + 8], sum);
    }
    sum = warp_sum(sum);
    if (lane == 0) static_cast<float *>(a.y)[size_t(blockIdx.y) * a.ld_y + row] = sum;
}

template <int Type, int Block, int Bytes, typename Index>
__global__ void gguf_packed_gemv(inference_cuda_dense_gemv_args a, Index index) {
    extern __shared__ float staged[];
    const auto *x = static_cast<const float *>(a.x) + size_t(blockIdx.y) * a.ld_x;
    for (int col = threadIdx.x; col < a.cols; col += blockDim.x) staged[col] = x[col];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / Block;
    const auto *w = matrix_base(a.w, index, blockIdx.y) + size_t(row) * blocks * Bytes;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const auto *weights = w + block * Bytes;
#pragma unroll
        for (int col = lane; col < Block; col += 32) {
            sum = fmaf(fifi_gguf::decode<Type>(weights, col), staged[block * Block + col], sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) static_cast<float *>(a.y)[size_t(blockIdx.y) * a.ld_y + row] = sum;
}

constexpr int kInklingGateWidth = 2304;

// One lane owns each eight-value IQ2 group. Transpose each activation block
// once per CTA so the eight inner FMA steps read shared memory without bank
// conflicts. Weight values stay F32; only the dot-product reduction order changes.
template <int Type, int Bytes>
__global__ void inkling_iq2_gate_gemv(const float *x, const uint8_t *gate, float *y,
                                      DeviceSelectedWeights index) {
    constexpr int cols = kInklingGateWidth, rows = kInklingGateWidth, blocks = cols / 256;
    __shared__ float staged[cols];
    const int tid = threadIdx.x;
    for (int col = tid; col < cols; col += kWarps * 32) {
        staged[(col / 256) * 256 + (col % 8) * 32 + (col % 256) / 8] = x[col];
    }
    __syncthreads();

    const int lane = tid & 31;
    const int row = blockIdx.x * kWarps + (tid >> 5);
    const auto *w = gate + index.offset(blockIdx.y) + size_t(row) * blocks * Bytes;
    float sum = 0;
    for (int block = 0; block < blocks; ++block) {
        const auto *b = w + block * Bytes;
        const auto group = fifi_gguf::iq2_group<Type>(b, lane / 4, lane % 4, fifi_gguf::half_value(b));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            sum = fmaf(fifi_gguf::grid_value(group.grid, j, group.scale, group.signs),
                       staged[block * 256 + j * 32 + lane], sum);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) y[size_t(blockIdx.y) * rows + row] = sum;
}

bool launch_inkling_iq2_gate(const inference_cuda_gguf_moe_args &a, DeviceSelectedWeights gate,
                             cudaStream_t stream, Failure &failure) {
    if (a.hidden != kInklingGateWidth || 2 * a.middle != kInklingGateWidth || a.count != 8) return false;
    const dim3 grid(kInklingGateWidth / kWarps, a.count);
    switch (a.gate_dtype - fifi_gguf::kDtypeBase) {
#define LAUNCH_GATE(id, bytes) \
    case id: \
        inkling_iq2_gate_gemv<id, bytes><<<grid, kWarps * 32, 0, stream>>>( \
            a.x, static_cast<const uint8_t *>(a.gate), a.scratch, gate); \
        break;
        LAUNCH_GATE(16, 66)
        LAUNCH_GATE(17, 74)
        LAUNCH_GATE(22, 82)
#undef LAUNCH_GATE
    default: return false;
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return true;
}

size_t weight_row_bytes(int dtype, int cols) {
    switch (dtype) {
    case kDtypeF16:
    case kDtypeBf16:
        return size_t(cols) * 2;
    case kDtypeF32:
        return size_t(cols) * sizeof(float);
    case kDtypeGgufNvfp4:
        return cols % 64 == 0 ? size_t(cols / 64) * 36 : 0;
    default:
        return fifi_gguf::row_bytes(dtype, cols);
    }
}

bool valid_operands(const inference_cuda_dense_gemv_args &a, void *stream) {
    if (!stream || !a.w || !a.x || !a.y ||
        a.m < 1 || a.rows < 1 || a.cols < 1 || a.ld_x < a.cols || a.ld_y < a.rows) {
        return false;
    }
    const size_t weight_alignment = a.weight_dtype == kDtypeF32 ? 4 : 2;
    if (!aligned(a.w, weight_alignment) || !aligned(a.x, 4) || !aligned(a.y, 4)) return false;
    const size_t m = (size_t)a.m;
    const size_t row_bytes = weight_row_bytes(a.weight_dtype, a.cols);
    if (row_bytes == 0) return false;
    const Span weights = {a.w, size_t(a.rows) * row_bytes};
    const Span in = {a.x, ((m - 1) * a.ld_x + a.cols) * 4};
    const Span out = {a.y, ((m - 1) * a.ld_y + a.rows) * 4};
    return apart(out, in) && apart(out, weights);
}

bool valid(const inference_cuda_dense_gemv_args &a, void *stream) {
    return a.m <= kMaxM && a.rows <= kMaxRows && a.cols <= kMaxCols &&
           a.ld_x <= kMaxLd && a.ld_y <= kMaxLd && valid_operands(a, stream);
}

constexpr int kWmmaTile = 32;
constexpr int kWmmaStride = kWmmaTile + 8;
constexpr int kWmmaThreads = 128;

// Native GGUF blocks are decoded once per tile and rounded to F16. External
// tensor scales remain the caller's responsibility, as in the F32 matvec.
template <int Type, int Block, int Bytes>
__device__ __forceinline__ __half wmma_weight(const uint8_t *row, int col) {
    const uint8_t *block = row + size_t(col / Block) * Bytes;
    const int i = col % Block;
    if constexpr (Type == 40) {
        const int group = i / 16;
        const unsigned pair = block[4 + 8 * group + i % 8];
        return __float2half_rn(gguf_fp4((pair >> (4 * ((i / 8) % 2))) & 15) *
                                 gguf_scale(block[group]));
    } else {
        return __float2half_rn(fifi_gguf::decode<Type>(block, i));
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
using WmmaAccumulator = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;

template <int Type, int Block, int Bytes>
__device__ void accumulate_packed_tile(inference_cuda_dense_gemv_args a,
                                       size_t token, size_t output, int warp,
                                       __half *x_tile, __half *w_tile, WmmaAccumulator &accum) {
    using namespace nvcuda::wmma;
    const auto *x = static_cast<const float *>(a.x);
    const auto *w = static_cast<const uint8_t *>(a.w);
    const size_t row_bytes = size_t(a.cols / Block) * Bytes;
    fill_fragment(accum, 0.0f);
    for (int base = 0; base < a.cols; base += kWmmaTile) {
        for (int i = threadIdx.x; i < kWmmaTile * kWmmaTile; i += kWmmaThreads) {
            const int row = i / kWmmaTile;
            const int col = base + i % kWmmaTile;
            const int at = row * kWmmaStride + i % kWmmaTile;
            x_tile[at] = token + row < size_t(a.m) ?
                __float2half_rn(x[(token + row) * a.ld_x + col]) : __float2half_rn(0.0f);
            w_tile[at] = output + row < size_t(a.rows) ?
                wmma_weight<Type, Block, Bytes>(w + (output + row) * row_bytes, col) :
                __float2half_rn(0.0f);
        }
        __syncthreads();
#pragma unroll
        for (int step = 0; step < kWmmaTile; step += 16) {
            fragment<matrix_a, 16, 16, 16, __half, row_major> left;
            fragment<matrix_b, 16, 16, 16, __half, col_major> right;
            load_matrix_sync(left, x_tile + (warp / 2) * 16 * kWmmaStride + step, kWmmaStride);
            load_matrix_sync(right, w_tile + (warp % 2) * 16 * kWmmaStride + step, kWmmaStride);
            mma_sync(accum, left, right, accum);
        }
        __syncthreads();
    }
}
#endif

template <int Type, int Block, int Bytes>
__global__ void gguf_wmma_kernel(inference_cuda_dense_gemv_args a) {
    const size_t output = size_t(blockIdx.x) * kWmmaTile;
    auto *y = static_cast<float *>(a.y);
#if __CUDA_ARCH__ >= 700
    __shared__ __align__(32) __half x_tile[kWmmaTile * kWmmaStride];
    __shared__ __align__(32) __half w_tile[kWmmaTile * kWmmaStride];
    __shared__ __align__(32) float results[kWmmaTile * kWmmaTile];
    const int warp = threadIdx.x / 32;
#endif
    // Grid Y is hardware-limited. Return rows in the original order even if
    // one CTA must process multiple token tiles.
    for (size_t token = size_t(blockIdx.y) * kWmmaTile; token < size_t(a.m);
         token += size_t(gridDim.y) * kWmmaTile) {
#if __CUDA_ARCH__ >= 700
        WmmaAccumulator accum;
        accumulate_packed_tile<Type, Block, Bytes>(a, token, output, warp, x_tile, w_tile, accum);
        nvcuda::wmma::store_matrix_sync(
            results + (warp / 2) * 16 * kWmmaTile + (warp % 2) * 16, accum,
            kWmmaTile, nvcuda::wmma::mem_row_major);
        __syncthreads();
#endif
        for (int i = threadIdx.x; i < kWmmaTile * kWmmaTile; i += kWmmaThreads) {
            const size_t t = token + i / kWmmaTile;
            const size_t r = output + i % kWmmaTile;
            if (t < size_t(a.m) && r < size_t(a.rows)) {
#if __CUDA_ARCH__ >= 700
                y[t * a.ld_y + r] = results[i];
#else
                // This path requires Tensor Cores. An unsupported raw launch
                // must not leave stale results in the output.
                y[t * a.ld_y + r] = __int_as_float(0x7fc00000);
#endif
            }
        }
        __syncthreads();
    }
}

int launch_wmma(inference_cuda_dense_gemv_args a, void *stream, int *out_stage) {
    const int kind = a.weight_dtype == kDtypeGgufNvfp4 ? 40 :
        a.weight_dtype - fifi_gguf::kDtypeBase;
    const unsigned token_tiles = (a.m - 1) / kWmmaTile + 1;
    const dim3 grid((a.rows - 1) / kWmmaTile + 1, token_tiles < 65535 ? token_tiles : 65535);
    const auto cuda_stream = (cudaStream_t)stream;
    switch (kind) {
#define WMMA_CASE(id, block, bytes) case id: \
        gguf_wmma_kernel<id, block, bytes><<<grid, kWmmaThreads, 0, cuda_stream>>>(a); \
        break;
        WMMA_CASE(16, 256, 66)
        WMMA_CASE(17, 256, 74)
        WMMA_CASE(19, 256, 50)
        WMMA_CASE(22, 256, 82)
        WMMA_CASE(29, 256, 56)
        WMMA_CASE(40, 64, 36)
        WMMA_CASE(42, 64, 18)
#undef WMMA_CASE
    default:
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

template <typename W>
void launch(const inference_cuda_dense_gemv_args &a, cudaStream_t s, Failure &failure) {
    const dim3 grid((a.rows + kWarps - 1) / kWarps, a.m);
    const size_t shared = (size_t)a.cols * sizeof(float);
    const bool vector = a.cols % (sizeof(uint4) / sizeof(W)) == 0 && aligned(a.w, 16);
    if (vector) {
        dense_gemv_k<W, true><<<grid, kWarps * 32, shared, s>>>(
            (const W *)a.w, (const float *)a.x, (float *)a.y, a.rows, a.cols, a.ld_x, a.ld_y);
    } else {
        dense_gemv_k<W, false><<<grid, kWarps * 32, shared, s>>>(
            (const W *)a.w, (const float *)a.x, (float *)a.y, a.rows, a.cols, a.ld_x, a.ld_y);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

template <typename Index>
void launch_packed(const inference_cuda_dense_gemv_args &a, cudaStream_t stream,
                   Failure &failure, Index index) {
    const dim3 grid((a.rows + kWarps - 1) / kWarps, a.m);
    const size_t shared = size_t(a.cols) * sizeof(float);
    if (a.weight_dtype == kDtypeGgufNvfp4) {
        gguf_nvfp4_gemv<Index><<<grid, kWarps * 32, 0, stream>>>(a, index);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        return;
    }
    switch (a.weight_dtype - fifi_gguf::kDtypeBase) {
#define LAUNCH(id, block, bytes) \
    case id: \
        gguf_packed_gemv<id, block, bytes, Index><<<grid, kWarps * 32, shared, stream>>>(a, index); \
        break;
        FIFI_GGUF_GEMV_TYPES(LAUNCH)
#undef LAUNCH
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

struct MoeSizes {
    size_t gate_stride;
    size_t down_stride;
};

bool good_scale(float value) { return std::isfinite(value) && value > 0.0f; }

bool valid_moe(const inference_cuda_gguf_moe_args &a,
               const inference_cuda_gguf_moe_matrices *matrices, void *stream, MoeSizes &sizes) {
    if (!stream || !a.x || !a.scratch || !a.y) return false;
    if (a.hidden < 1 || a.hidden > kMaxCols || a.middle < 1 || a.middle > kMaxCols ||
        a.experts < 1 || a.experts > kMaxRows ||
        a.count < 1 || a.count > INFERENCE_CUDA_GGUF_MOE_MAX_PICKS ||
        !good_scale(a.gate_cap) || !good_scale(a.up_cap)) return false;
    if (a.gate_dtype < kDtypeGgufNvfp4 || a.down_dtype < kDtypeGgufNvfp4 ||
        a.gate_dtype == kDtypeF32 || a.down_dtype == kDtypeF32) return false;
    sizes = {2 * size_t(a.middle) * weight_row_bytes(a.gate_dtype, a.hidden),
             size_t(a.hidden) * weight_row_bytes(a.down_dtype, a.middle)};
    if (!sizes.gate_stride || !sizes.down_stride || !aligned(a.x, 4) ||
        !aligned(a.scratch, 4) || !aligned(a.y, 4)) return false;

    const Span x = {a.x, size_t(a.hidden) * sizeof(float)};
    const Span scratch = {a.scratch, size_t(a.count) * (3 * a.middle + a.hidden) * sizeof(float)};
    const Span result = {a.y, size_t(a.hidden) * sizeof(float)};
    if (!apart(scratch, result) || !apart(x, scratch) || !apart(x, result)) return false;
    const auto weights_fit = [&](Span gate, Span down) {
        const Span weights[] = {gate, down};
        for (const Span &weight : weights) {
            if (!weight.p || !aligned(weight.p, 2) ||
                !apart(weight, scratch) || !apart(weight, result)) return false;
        }
        return true;
    };
    if (!matrices && !weights_fit({a.gate, sizes.gate_stride * a.experts},
                                  {a.down, sizes.down_stride * a.experts})) return false;
    if (a.device_ids || a.device_weights || a.gate_scales || a.down_scales) {
        if (matrices || a.count > a.experts) return false;
        const size_t picks = size_t(a.count) * sizeof(float);
        const size_t scales = size_t(a.experts) * sizeof(float);
        const Span inputs[] = {{a.device_ids, picks}, {a.device_weights, picks},
                               {a.gate_scales, scales}, {a.down_scales, scales}};
        for (const Span &input : inputs) {
            if (!input.p || !aligned(input.p, 4) || !apart(input, scratch) ||
                !apart(input, result)) return false;
        }
        return true;
    }
    for (int i = 0; i < a.count; ++i) {
        const auto pick = a.picks[i];
        if (pick.expert >= uint32_t(a.experts) || !good_scale(pick.gate_scale) ||
            !good_scale(pick.down_scale) || !std::isfinite(pick.weight)) return false;
        if (matrices && !weights_fit({matrices->gate[i], sizes.gate_stride},
                                     {matrices->down[i], sizes.down_stride})) return false;
    }
    return true;
}

template <bool DevicePicks>
__global__ void activate_moe(inference_cuda_gguf_moe_args a, float *middle) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.middle) return;
    const size_t offset = size_t(blockIdx.y) * a.middle;
    float scale;
    if constexpr (DevicePicks) {
        const unsigned id = unsigned(a.device_ids[blockIdx.y]);
        scale = a.gate_scales[id < unsigned(a.experts) ? id : 0];
    } else {
        scale = a.picks[blockIdx.y].gate_scale;
    }
    const float gate = a.scratch[2 * offset + col] * scale;
    const float up = a.scratch[2 * offset + a.middle + col] * scale;
    middle[offset + col] = clipped_expert_activation(gate, up, a.gate_cap, a.up_cap);
}

template <bool DevicePicks>
__global__ void sum_moe(inference_cuda_gguf_moe_args a, const float *down) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.hidden) return;
    float out = 0.0f;
    for (int slot = 0; slot < a.count; ++slot) {
        float row = down[size_t(slot) * a.hidden + col];
        float weight;
        if constexpr (DevicePicks) {
            const unsigned id = unsigned(a.device_ids[slot]);
            if (id >= unsigned(a.experts)) {
                a.y[col] = __int_as_float(0x7fc00000);
                return;
            }
            row = __fmul_rn(row, a.down_scales[id]);
            weight = a.device_weights[slot];
        } else {
            row = __fmul_rn(row, a.picks[slot].down_scale);
            weight = a.picks[slot].weight;
        }
        out = __fadd_rn(out, __fmul_rn(weight, row));
    }
    a.y[col] = out;
}

template <bool DevicePicks = false, typename Index>
void launch_moe(const inference_cuda_gguf_moe_args &a, Index gate, Index selected_down,
                cudaStream_t stream, Failure &failure, int *grouped_iq2_gate) {
    float *middle = a.scratch + size_t(a.count) * 2 * a.middle;
    float *down = middle + size_t(a.count) * a.middle;
    bool grouped = false;
    if constexpr (DevicePicks) {
        if (grouped_iq2_gate && *grouped_iq2_gate == 1) {
            grouped = launch_inkling_iq2_gate(a, gate, stream, failure);
        }
    }
    if (grouped_iq2_gate) *grouped_iq2_gate = grouped;
    if (!grouped) {
        launch_packed({a.gate, a.x, a.scratch, a.count, 2 * a.middle, a.hidden,
                       0, 2 * a.middle, a.gate_dtype}, stream, failure, gate);
    }
    if (!failure.ok()) return;

    constexpr int threads = 256;
    const dim3 grid((a.middle + threads - 1) / threads, a.count);
    activate_moe<DevicePicks><<<grid, threads, 0, stream>>>(a, middle);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;

    launch_packed({a.down, middle, down, a.count, a.hidden, a.middle,
                   a.middle, a.hidden, a.down_dtype}, stream, failure, selected_down);
    if (!failure.ok()) return;

    sum_moe<DevicePicks><<<(a.hidden + threads - 1) / threads, threads, 0, stream>>>(a, down);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

}  // namespace

extern "C" int inference_cuda_dense_gemv_launch(const inference_cuda_dense_gemv_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        const auto cuda_stream = (cudaStream_t)stream;
        if (args->weight_dtype == kDtypeF16) {
            launch<__half>(*args, cuda_stream, failure);
        } else if (args->weight_dtype == kDtypeBf16) {
            launch<__nv_bfloat16>(*args, cuda_stream, failure);
        } else if (args->weight_dtype == kDtypeF32) {
            launch<float>(*args, cuda_stream, failure);
        } else {
            launch_packed(*args, cuda_stream, failure, SharedWeights{});
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_wmma_launch(const inference_cuda_dense_gemv_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!args || !valid_operands(*args, stream) || args->cols % kWmmaTile) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch_wmma(*args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_moe_launch(const inference_cuda_gguf_moe_args *args,
                                               const inference_cuda_gguf_moe_matrices *matrices,
                                               void *stream, int *out_stage,
                                               int *grouped_iq2_gate) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        MoeSizes sizes;
        if (!args || !valid_moe(*args, matrices, stream, sizes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        if (matrices) {
            SeparateWeights gate{}, down{};
            for (int slot = 0; slot < args->count; ++slot) {
                gate.matrices[slot] = matrices->gate[slot];
                down.matrices[slot] = matrices->down[slot];
            }
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, grouped_iq2_gate);
        } else if (args->device_ids) {
            const auto experts = unsigned(args->experts);
            const DeviceSelectedWeights gate{sizes.gate_stride, args->device_ids, experts};
            const DeviceSelectedWeights down{sizes.down_stride, args->device_ids, experts};
            launch_moe<true>(*args, gate, down, (cudaStream_t)stream, failure, grouped_iq2_gate);
        } else {
            SelectedWeights gate = {sizes.gate_stride, {}};
            for (int slot = 0; slot < args->count; ++slot) gate.ids[slot] = args->picks[slot].expert;
            SelectedWeights down = gate;
            down.stride = sizes.down_stride;
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, grouped_iq2_gate);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
