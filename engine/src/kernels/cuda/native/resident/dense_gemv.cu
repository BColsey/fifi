#include "../expert_math.cuh"
// Dense matvec for floating-point or packed GGUF weights (see resident/dense_gemv.h).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <type_traits>
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

// One shared matrix per call. Each warp owns the same weight row and lane
// columns as gguf_packed_gemv, but reuses each decoded weight in Tile separate
// F32 accumulators. Do not transpose the lane mapping or split K: the exact
// packed-GEMV lane FMA sequence and XOR butterfly are the reference.
template <int Type, int Block, int Bytes, int Tile>
__global__ void gguf_exact_expert_tile(inference_cuda_dense_gemv_args a) {
    extern __shared__ float staged[];
    const int first = blockIdx.y * Tile;
    const auto *x = static_cast<const float *>(a.x);
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        // The tail stages a valid row in unused slots; no invalid reads or stores.
        const auto *in = x + size_t(first + (first + t < a.m ? t : 0)) * a.ld_x;
        for (int col = threadIdx.x; col < a.cols; col += blockDim.x) {
            staged[t * a.cols + col] = in[col];
        }
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / Block;
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * blocks * Bytes;
    float sums[Tile] = {};
    for (int block = 0; block < blocks; ++block) {
        const auto *weights = w + block * Bytes;
#pragma unroll
        for (int col = lane; col < Block; col += 32) {
            const float value = fifi_gguf::decode<Type>(weights, col);
#pragma unroll
            for (int t = 0; t < Tile; ++t) {
                sums[t] = fmaf(value, staged[t * a.cols + block * Block + col], sums[t]);
            }
        }
    }
    auto *y = static_cast<float *>(a.y);
#pragma unroll
    for (int t = 0; t < Tile; ++t) {
        const float sum = warp_sum(sums[t]);
        if (lane == 0 && first + t < a.m) y[size_t(first + t) * a.ld_y + row] = sum;
    }
}

constexpr int kInklingGateWidth = 2304;

// Called only after the ordinary GEMV operand and shape checks.
constexpr int exact_expert_tile(int type, int m, int rows, int cols) {
    if (type == 14) {
        // Inkling attention projections at the measured narrow and wide tile shapes.
        return rows == 3072 && cols == 2304 && (m == 8 || m == 63 || m == 64) ? 2 : 0;
    }
    if (m < 2 || m > 512 || rows > kInklingGateWidth || cols > kInklingGateWidth ||
        (type != 16 && type != 42)) return 0;
    // At most 18 KiB of activation staging per CTA.
    return type == 42 && cols <= kInklingGateWidth / 2 && m >= 4 ? 4 : 2;
}

static_assert(exact_expert_tile(16, 512, 2304, 2304) == 2 &&
              exact_expert_tile(42, 5, 2304, 1152) == 4 &&
              exact_expert_tile(42, 3, 2304, 1152) == 2 &&
              exact_expert_tile(42, 5, 2304, 1216) == 2);
static_assert(exact_expert_tile(16, 1, 2304, 2304) == 0 &&
              exact_expert_tile(16, 513, 2304, 2304) == 0 &&
              exact_expert_tile(16, 2, 2305, 2304) == 0 &&
              exact_expert_tile(16, 2, 2304, 2560) == 0 &&
              exact_expert_tile(8, 2, 2304, 2304) == 0);
static_assert(exact_expert_tile(14, 8, 3072, 2304) == 2 &&
              exact_expert_tile(14, 63, 3072, 2304) == 2 &&
              exact_expert_tile(14, 64, 3072, 2304) == 2 &&
              exact_expert_tile(14, 1, 3072, 2304) == 0 &&
              exact_expert_tile(14, 8, 3072, 2560) == 0 &&
              exact_expert_tile(14, 64, 2304, 2304) == 0);

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

#include "resident/inkling_decode_weights.cuh"
#include "resident/inkling_decode_control_pair.cuh"
#include "resident/inkling_decode_attention_group.cuh"
#include "resident/inkling_gate_dot.cuh"
#include "resident/inkling_q2_down.cuh"

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
    if (a.m == 1 && ((a.cols == 4608 && a.rows == 64 && a.weight_dtype == kDtypeF32) ||
                     (a.cols == 2304 && (a.rows == 24 || a.rows == 32) && a.weight_dtype == kDtypeBf16)) &&
        decode_split_enabled() && vector && aligned(a.x,16)) {
        if constexpr (std::is_same_v<W,float>) decode_controller_down_exact<<<a.rows,256,0,s>>>(a);
        else decode_bf16_exact<<<a.rows,32,0,s>>>(a);
    } else if (vector) {
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
    if (decode_split_enabled() && launch_decode_split(a, index, stream)) {
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        return;
    }
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

bool launch_inkling_q2_down(const inference_cuda_gguf_moe_args &a, DeviceSelectedWeights index,
                            const float *middle, float *down, cudaStream_t stream, Failure &failure) {
    if (!inkling_q2_down_shape(a.hidden, a.middle, a.count, a.down_dtype)) return false;
    const dim3 grid(kInklingDownRows / kQ2RowsPerCta, kInklingDownPicks);
    if (decode_split_enabled() && aligned(middle, 16)) {
        decode_q2_pair<<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const uint8_t *>(a.down), middle, down, index);
    } else {
        inkling_q2_down<<<grid, kWarps * 32, 0, stream>>>(
            static_cast<const uint8_t *>(a.down), middle, down, index);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return true;
}

template <bool DevicePicks = false, typename Index>
void launch_moe(const inference_cuda_gguf_moe_args &a, Index gate, Index selected_down,
                cudaStream_t stream, Failure &failure, int *schedule) {
    float *middle = a.scratch + size_t(a.count) * 2 * a.middle;
    float *down = middle + size_t(a.count) * a.middle;
    const int request = schedule ? *schedule : 0;
    const int gate_request = request & INFERENCE_CUDA_GGUF_MOE_GATE_MASK;
    int selected_gate = 0;
    if constexpr (DevicePicks) {
        if (gate_request == 2 && launch_inkling_gate_dot(a, gate, middle, stream, failure)) {
            selected_gate = 2;
        } else if (gate_request != 0 && launch_inkling_iq2_gate(a, gate, stream, failure)) {
            selected_gate = 1;
        }
    }
    if (schedule) *schedule = selected_gate;
    if (!selected_gate) {
        launch_packed({a.gate, a.x, a.scratch, a.count, 2 * a.middle, a.hidden,
                       0, 2 * a.middle, a.gate_dtype}, stream, failure, gate);
    }
    if (!failure.ok()) return;

    constexpr int threads = 256;
    const dim3 grid((a.middle + threads - 1) / threads, a.count);
    activate_moe<DevicePicks><<<grid, threads, 0, stream>>>(a, middle);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;

    bool exact_down = false;
    if constexpr (DevicePicks) {
        exact_down = (request & INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN) &&
            launch_inkling_q2_down(a, selected_down, middle, down, stream, failure);
        if (schedule && exact_down && failure.ok()) *schedule |= INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN;
    }
    if (!exact_down) {
        launch_packed({a.down, middle, down, a.count, a.hidden, a.middle,
                       a.middle, a.hidden, a.down_dtype}, stream, failure, selected_down);
    }
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

extern "C" int inference_cuda_decode_attention_group(const inference_cuda_dense_gemv_args *calls,
        int count, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!calls || (count != 6 && count != 8)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        DecodeAttentionGroup group{};
        int blocks=0;
        for (int i=0;i<count;++i) {
            const auto &a=calls[i];
            int n=decode_attention_blocks(a);
            if (!n || !valid(a,stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            group.calls[i]=a;
            group.limits[i]=blocks+=n;
        }
        for (int i=count;i<8;++i) group.limits[i]=blocks;
        for (int i=0;i<count;++i) {
            Span dst={calls[i].y,size_t(calls[i].rows)*4};
            for (int j=0;j<count;++j) if (j!=i) {
                const auto &b=calls[j];
                if (!apart(dst,{b.y,size_t(b.rows)*4}) || !apart(dst,{b.x,2304*4}) ||
                    !apart(dst,{b.w,size_t(b.rows)*weight_row_bytes(b.weight_dtype,2304)}))
                    return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
        }
        Failure f;
        decode_attention_group<<<blocks,128,0,(cudaStream_t)stream>>>(group);
        f.record(cudaGetLastError(),INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage,f.stage);
        return (int)f.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_decode_control_pair(const inference_cuda_dense_gemv_args *a,
        const inference_cuda_dense_gemv_args *b, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const auto eligible = [stream](const inference_cuda_dense_gemv_args *p) {
            return p && p->weight_dtype == kDtypeF32 && p->m == 1 && p->rows == 2304 &&
                   p->cols == 64 && aligned(p->w, 16) && valid(*p, stream);
        };
        if (!eligible(a) || !eligible(b) || a->x != b->x ||
            !apart({a->y,2304*4},{b->y,2304*4}) ||
            !apart({a->y,2304*4},{b->w,2304*64*4}) ||
            !apart({b->y,2304*4},{a->w,2304*64*4})) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure f;
        decode_control_pair<<<dim3(144,2),256,0,(cudaStream_t)stream>>>(*a,*b);
        f.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, f.stage);
        return (int)f.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

namespace {

// Two adjacent input rows share a decoded Q8_0 weight. Keep lane c on
// column block*32+c and preserve the baseline FMA order within each row.
__global__ void inkling_q8_reuse_pair(inference_cuda_dense_gemv_args a) {
    extern __shared__ float2 paired[];
    const int first = blockIdx.y * 2;
    const bool has_second = first + 1 < a.m;
    const auto *x = static_cast<const float *>(a.x) + size_t(first) * a.ld_x;
    const auto *second = has_second ? x + a.ld_x : x;
    for (int c = threadIdx.x; c < a.cols; c += blockDim.x) {
        paired[c] = make_float2(x[c], second[c]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= a.rows) return;
    const int blocks = a.cols / 32;
    const auto *weights = static_cast<const uint8_t *>(a.w) + size_t(row) * blocks * 34;
    float lo = 0.0f, hi = 0.0f;
    for (int k = 0; k < blocks; ++k) {
        const float w = fifi_gguf::decode<8>(weights + k * 34, lane);
        const float2 v = paired[k * 32 + lane];
        lo = fmaf(w, v.x, lo);
        hi = fmaf(w, v.y, hi);
    }
    lo = warp_sum(lo);
    hi = warp_sum(hi);
    if (lane == 0) {
        auto *y = static_cast<float *>(a.y) + size_t(first) * a.ld_y + row;
        y[0] = lo;
        if (has_second) y[a.ld_y] = hi;
    }
}

// Only the 2304 x 3072 attention-output projection uses this path. Stage
// a third of K at a time (16 KiB/CTA), retaining four independent F32 chains.
__global__ void inkling_q8_reuse_four(inference_cuda_dense_gemv_args a) {
    constexpr int chunk = 1024;
    constexpr int cols = 3072;
    constexpr int block_bytes = 34;
    __shared__ float4 staged[chunk];
    const int first = blockIdx.y * 4;
    const int live = min(a.m - first, 4);
    const auto *x = static_cast<const float *>(a.x) + size_t(first) * a.ld_x;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarps + (threadIdx.x >> 5);
    const auto *w = static_cast<const uint8_t *>(a.w) + size_t(row) * (cols / 32) * block_bytes;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int base = 0; base < cols; base += chunk) {
        for (int i = threadIdx.x; i < chunk; i += blockDim.x) {
            const float *p = x + base + i;
            staged[i] = make_float4(p[0], live > 1 ? p[a.ld_x] : 0.f,
                                     live > 2 ? p[2 * a.ld_x] : 0.f,
                                     live > 3 ? p[3 * a.ld_x] : 0.f);
        }
        __syncthreads();
        for (int k = 0; k < chunk / 32; ++k) {
            const float value = fifi_gguf::decode<8>(w + (base / 32 + k) * block_bytes, lane);
            const float4 input = staged[k * 32 + lane];
            a0 = fmaf(value, input.x, a0);
            a1 = fmaf(value, input.y, a1);
            a2 = fmaf(value, input.z, a2);
            a3 = fmaf(value, input.w, a3);
        }
        __syncthreads();
    }
    a0 = warp_sum(a0);
    a1 = warp_sum(a1);
    a2 = warp_sum(a2);
    a3 = warp_sum(a3);
    if (lane == 0) {
        auto *y = static_cast<float *>(a.y) + size_t(first) * a.ld_y + row;
        y[0] = a0;
        if (live > 1) y[a.ld_y] = a1;
        if (live > 2) y[2 * a.ld_y] = a2;
        if (live > 3) y[3 * a.ld_y] = a3;
    }
}

// Restrict production to the measured attention prefill M windows.
constexpr bool q8_row_reuse_shape(int dtype, int m, int rows, int cols) {
    if (dtype != fifi_gguf::kDtypeBase + 8) return false;
    return (cols == 2304 && (rows == 1024 || rows == 3072) && m >= 7 && m <= 12) ||
           (cols == 3072 && rows == 2304 && m >= 63 && m <= 64);
}

constexpr int kQ8 = fifi_gguf::kDtypeBase + 8;
static_assert(q8_row_reuse_shape(kQ8, 7, 1024, 2304) &&
              q8_row_reuse_shape(kQ8, 12, 1024, 2304) &&
              q8_row_reuse_shape(kQ8, 7, 3072, 2304) &&
              q8_row_reuse_shape(kQ8, 12, 3072, 2304) &&
              q8_row_reuse_shape(kQ8, 63, 2304, 3072) &&
              q8_row_reuse_shape(kQ8, 64, 2304, 3072));
static_assert(!q8_row_reuse_shape(kQ8 + 1, 7, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 7, 1023, 2304) &&
              !q8_row_reuse_shape(kQ8, 7, 1024, 2336) &&
              !q8_row_reuse_shape(kQ8, 1, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 2, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 6, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 13, 1024, 2304) &&
              !q8_row_reuse_shape(kQ8, 6, 3072, 2304) &&
              !q8_row_reuse_shape(kQ8, 13, 3072, 2304) &&
              !q8_row_reuse_shape(kQ8, 62, 2304, 3072) &&
              !q8_row_reuse_shape(kQ8, 65, 2304, 3072));

cudaError_t current_is_sm75(bool &sm75) {
    sm75 = false;
    int device = 0, major = 0, minor = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (err == cudaSuccess && major == 7) {
        err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        sm75 = err == cudaSuccess && minor == 5;
    }
    return err;
}

}  // namespace

extern "C" int inference_cuda_gguf_q8_row_reuse_width_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_width,
    int prefer_four) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_width) *out_width = 0;
        if (!args || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const auto &a = *args;
        if (!q8_row_reuse_shape(a.weight_dtype, a.m, a.rows, a.cols)) {
            return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        }
        Failure failure;
        bool sm75;
        // Rust selects the stream's adapter under the dispatch guard before entering here.
        failure.record(current_is_sm75(sm75), INFERENCE_CUDA_STAGE_LAUNCH);
        if (!failure.ok()) {
            report_stage(out_stage, failure.stage);
            return (int)failure.err;
        }
        if (!sm75) return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        constexpr int output_rows = 2304;
        static_assert(output_rows % kWarps == 0);
        const bool four = prefer_four && a.rows == output_rows && a.cols == 3072;
        const int width = four ? 4 : 2;
        const dim3 grid((a.rows + kWarps - 1) / kWarps, (a.m + width - 1) / width);
        const auto s = static_cast<cudaStream_t>(stream);
        if (four) {
            inkling_q8_reuse_four<<<grid, kWarps * 32, 0, s>>>(a);
        } else {
            inkling_q8_reuse_pair<<<grid, kWarps * 32, size_t(a.cols) * sizeof(float2), s>>>(a);
        }
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok() && out_width) *out_width = width;
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gguf_q8_row_reuse_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_pair) {
    int width = 0;
    const int error = inference_cuda_gguf_q8_row_reuse_width_launch(args, stream, out_stage, &width, 0);
    if (out_pair) *out_pair = width == 2;
    return error;
}

extern "C" int inference_cuda_gguf_exact_expert_tile_launch(
    const inference_cuda_dense_gemv_args *args, void *stream, int *out_stage, int *out_tile) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (out_tile) *out_tile = 0;
        if (!args || !valid(*args, stream)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const auto &a = *args;
        const int type = a.weight_dtype - fifi_gguf::kDtypeBase;
        const int tile = exact_expert_tile(type, a.m, a.rows, a.cols);
        if (tile == 0) return inference_cuda_dense_gemv_launch(args, stream, out_stage);
        const dim3 grid((a.rows + kWarps - 1) / kWarps, (a.m + tile - 1) / tile);
        const size_t shared = size_t(tile) * a.cols * sizeof(float);
        const auto s = (cudaStream_t)stream;
        if (type == 14) {
            gguf_exact_expert_tile<14, 256, 210, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (type == 16) {
            gguf_exact_expert_tile<16, 256, 66, 2><<<grid, kWarps * 32, shared, s>>>(a);
        } else if (tile == 4) {
            gguf_exact_expert_tile<42, 64, 18, 4><<<grid, kWarps * 32, shared, s>>>(a);
        } else {
            gguf_exact_expert_tile<42, 64, 18, 2><<<grid, kWarps * 32, shared, s>>>(a);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        if (failure.ok() && out_tile) *out_tile = tile;
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
                                               int *schedule) {
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
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        } else if (args->device_ids) {
            const auto experts = unsigned(args->experts);
            const DeviceSelectedWeights gate{sizes.gate_stride, args->device_ids, experts};
            const DeviceSelectedWeights down{sizes.down_stride, args->device_ids, experts};
            launch_moe<true>(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        } else {
            SelectedWeights gate = {sizes.gate_stride, {}};
            for (int slot = 0; slot < args->count; ++slot) gate.ids[slot] = args->picks[slot].expert;
            SelectedWeights down = gate;
            down.stride = sizes.down_stride;
            launch_moe(*args, gate, down, (cudaStream_t)stream, failure, schedule);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
