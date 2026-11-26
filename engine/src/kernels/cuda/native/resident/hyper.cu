// The hyper-connection read and apply of Qwen3.8 Flash-Next (see resident/hyper.h for the
// formulas and the argument contract). Every dense product accumulates in plain f32, one warp per
// output row; the five steps are not fused. The products of many tokens reuse each weight across
// a tile of tokens (hyper_down_tiled_k, hyper_up_tiled_k) with the arithmetic order of the
// one-token kernels, so both give the same bits.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/hyper.h"

namespace {

constexpr int kMaxM = 32768;
constexpr int kMaxBranches = 16;
constexpr int kMaxHidden = 65536;
// The up projection launches width / kWarps blocks in grid y, whose limit is 65535.
constexpr int kMaxWidth = 1 << 18;
constexpr int kMaxRank = 4096;
constexpr int kWarps = 8;
// Tiled products: output rows per warp, row groups per block of the up product, and the token
// counts from which each product takes the tiled kernel (below them the one-token kernel is
// faster: too few blocks to hide the staging of the activations).
constexpr int kRowsPerWarp = 4;
constexpr int kUpGroups = 4;
constexpr int kDownTiledMinM = 32;
constexpr int kUpTiledMinM = 4;
constexpr int kNormThreads = 256;
constexpr int kDtypeF16 = 0;
constexpr int kDtypeBf16 = 1;
// GGUF Q8_0 matrices (`down`, `up`), decoded on load, with F16 vectors (`norm_w`, `inject`).
constexpr int kDtypeQ80 = 2;

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

struct Span {
    const void *p;
    size_t bytes;
};

bool apart(const Span &a, const Span &b) {
    const uintptr_t x = (uintptr_t)a.p, y = (uintptr_t)b.p;
    return x <= y ? a.bytes <= y - x : b.bytes <= x - y;
}

__device__ __forceinline__ float load(const __half *w, size_t i) { return __half2float(w[i]); }
__device__ __forceinline__ float load(const __nv_bfloat16 *w, size_t i) {
    return __bfloat162float(w[i]);
}

// A GGUF Q8_0 matrix addressed like a row of floats: element i is its block's fp16 scale (block
// i >> 5) times its int8 lane (i & 31), a product f32 holds exactly, so no weight is rounded on the
// way in, where an F16 copy of the decoded matrix rounds every one. The offsets the kernels add
// are multiples of 32 elements, which `valid_read` requires of this dtype.
struct Q80 {
    const uint8_t *p;
};
constexpr int kQ80Elems = 32;
constexpr int kQ80Bytes = 34;

__device__ __forceinline__ Q80 operator+(Q80 w, size_t elements) {
    return Q80{w.p + (elements >> 5) * kQ80Bytes};
}

__device__ __forceinline__ float load(Q80 w, size_t i) {
    const uint8_t *b = w.p + (i >> 5) * kQ80Bytes;
    __half_raw scale;
    scale.x = (uint16_t)b[0] | (uint16_t)((uint16_t)b[1] << 8);
    return __half2float(__half(scale)) * (float)(int8_t)b[2 + (int)(i & 31)];
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One block per (token, branch): the RMS of the branch, then the (1 + w)-weighted row.
template <typename W>
__global__ void hyper_norm_k(const float *__restrict__ x, const W *__restrict__ norm_w,
                             float *__restrict__ normed, int branches, int hidden, float eps) {
    const int t = blockIdx.x / branches;
    const int b = blockIdx.x % branches;
    const size_t width = (size_t)branches * hidden;
    const float *row = x + (size_t)t * width + (size_t)b * hidden;
    float *out = normed + (size_t)t * width + (size_t)b * hidden;
    float sum = 0.0f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) sum += row[d] * row[d];
    __shared__ float partial[kNormThreads / 32];
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < kNormThreads / 32 ? partial[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) partial[0] = v;
    }
    __syncthreads();
    const float denom = sqrtf(partial[0] / (float)hidden + eps);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        const float w = load(norm_w, (size_t)b * hidden + d);
        out[d] = row[d] * (1.0f + w) / denom;
    }
}

// One block per token, one warp per output row r of the down projection (r < rank) and of inject
// (r >= rank). The reference arithmetic of hyper_down_tiled_k.
template <typename W>
__global__ void hyper_down_k(const float *__restrict__ normed, const W *__restrict__ down,
                             const W *__restrict__ inject, float *__restrict__ low,
                             float *__restrict__ gate, int branches, int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * kWarps + (threadIdx.x >> 5);
    const int rows = rank + (inject != NULL ? branches : 0);
    if (r >= rows) return;
    const int t = blockIdx.x;
    const float *in = normed + (size_t)t * width;
    const W *w = r < rank ? down + (size_t)r * width : inject + (size_t)(r - rank) * width;
    float acc = 0.0f;
    for (int i = lane; i < width; i += 32) acc += load(w, i) * in[i];
    acc = warp_sum(acc);
    if (lane != 0) return;
    const float scaled = acc / (float)branches;
    if (r < rank) {
        low[(size_t)t * rank + r] = scaled * sigmoid_f(scaled);
    } else {
        gate[(size_t)t * branches + (r - rank)] = 2.0f * sigmoid_f(scaled);
    }
}

// One block per token, one warp per output row i of the up projection. The reference arithmetic
// of hyper_up_tiled_k.
template <typename W>
__global__ void hyper_up_k(const float *__restrict__ low, const W *__restrict__ up,
                           float *__restrict__ mix, int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (i >= width) return;
    const int t = blockIdx.x;
    const float *in = low + (size_t)t * rank;
    const W *w = up + (size_t)i * rank;
    float acc = 0.0f;
    for (int j = lane; j < rank; j += 32) acc += load(w, j) * in[j];
    acc = warp_sum(acc);
    if (lane == 0) mix[(size_t)t * width + i] = acc;
}

// Weight reuse across tokens. A warp forms R output rows for a tile of T tokens at once: lane l
// walks i = l, l + 32, ... in increasing order exactly as the one-token kernels above do, loads
// each weight once and adds its product into the R * T accumulators, so every (row, token) sum
// is built in the same order as hyper_down_k / hyper_up_k and is bit-identical to theirs.
// The T activation rows of the block are staged in shared memory, one chunk of the input at a
// time; chunks are multiples of 32 and taken in increasing order, so a lane's index order is
// unchanged.

constexpr int kStageFloats = 4096;

// Floats of one token row staged at a time; a multiple of 32.
template <int T>
constexpr int kChunk = kStageFloats / T;

// Reduces each of the N accumulators of the warp with the xor tree of warp_sum (offsets 16, 8,
// 4, 2, 1 in that order, every add `own + partner`), but exchanges half of the live values at each
// of the first log2(N) steps, so the lane that ends up holding accumulator a is lane >> (5 - log2 N)
// and the shuffles number N - 1 + (5 - log2 N) instead of 5 N. Each accumulator sees the same adds
// in the same order as under warp_sum, so the result is the same bits.
template <int N, int OFF>
__device__ __forceinline__ float fold(const float (&v)[N], int lane) {
    if constexpr (N == 1) {
        float x = v[0];
#pragma unroll
        for (int o = OFF; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
        return x;
    } else {
        const bool hi = (lane & OFF) != 0;
        float w[N / 2];
#pragma unroll
        for (int k = 0; k < N / 2; ++k) {
            const float send = hi ? v[k] : v[k + N / 2];
            const float keep = hi ? v[k + N / 2] : v[k];
            w[k] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
        }
        return fold<N / 2, OFF / 2>(w, lane);
    }
}

template <int N>
constexpr int kLog2 = N <= 1 ? 0 : 1 + kLog2<N / 2>;

// Block-wide copy of `len` floats from each of T token rows (`in` + (tok0 + t) * stride + at)
// into act[t * chunk ...]; tokens at or past m stage zeros.
template <int T>
__device__ __forceinline__ void stage_tile(float *act, const float *in, int tok0, int m,
                                           size_t stride, int at, int len) {
    constexpr int chunk = kChunk<T>;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const bool live = tok0 + t < m;
        const float *src = in + (size_t)(live ? tok0 + t : 0) * stride + at;
        for (int j = threadIdx.x; j < len; j += blockDim.x) {
            act[t * chunk + j] = live ? src[j] : 0.0f;
        }
    }
}

// acc[r * T + t] += w_r[at + j] * act[t][j] for this lane's j = lane, lane + 32, ... < len.
template <typename W, int T, int R>
__device__ __forceinline__ void tile_dot(const float *act, const W *const (&row)[R], int at,
                                         int len, int lane, float (&acc)[R * T]) {
    constexpr int chunk = kChunk<T>;
#pragma unroll 4
    for (int j = lane; j < len; j += 32) {
        float wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = load(row[r], (size_t)at + j);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float a = act[t * chunk + j];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r * T + t] += wv[r] * a;
        }
    }
}

// One warp per R rows of the down projection (row < rank) and of inject (row >= rank), for the T
// tokens of blockIdx.y. Rows past the last one run on a clamped row and store nothing.
template <typename W, int T, int R>
__global__ void __launch_bounds__(kWarps * 32)
    hyper_down_tiled_k(const float *__restrict__ normed, const W *__restrict__ down,
                       const W *__restrict__ inject, float *__restrict__ low,
                       float *__restrict__ gate, int m, int branches, int width, int rank) {
    constexpr int chunk = kChunk<T>;
    constexpr int kAcc = R * T;
    __shared__ float act[T * chunk];
    const int lane = threadIdx.x & 31;
    const int rows = rank + (inject != NULL ? branches : 0);
    const int tok0 = blockIdx.y * T;
    const int row0 = (blockIdx.x * kWarps + (threadIdx.x >> 5)) * R;
    const W *rp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int at = min(row0 + r, rows - 1);
        rp[r] = at < rank ? down + (size_t)at * width : inject + (size_t)(at - rank) * width;
    }
    float acc[kAcc];
#pragma unroll
    for (int a = 0; a < kAcc; ++a) acc[a] = 0.0f;
    for (int at = 0; at < width; at += chunk) {
        const int len = min(chunk, width - at);
        __syncthreads();
        stage_tile<T>(act, normed, tok0, m, (size_t)width, at, len);
        __syncthreads();
        tile_dot<W, T, R>(act, rp, at, len, lane, acc);
    }
    const float sum = fold<kAcc, 16>(acc, lane);
    constexpr int kLow = 5 - kLog2<kAcc>;
    const int a = lane >> kLow;
    const int r = row0 + a / T;
    const int t = tok0 + a % T;
    if ((lane & ((1 << kLow) - 1)) != 0 || r >= rows || t >= m) return;
    const float scaled = sum / (float)branches;
    if (r < rank) {
        low[(size_t)t * rank + r] = scaled * sigmoid_f(scaled);
    } else {
        gate[(size_t)t * branches + (r - rank)] = 2.0f * sigmoid_f(scaled);
    }
}

// One warp per R rows of the up projection, `groups` times per block, for the T tokens of
// blockIdx.y. A rank that fits one chunk is staged once per block.
template <typename W, int T, int R>
__global__ void __launch_bounds__(kWarps * 32)
    hyper_up_tiled_k(const float *__restrict__ low, const W *__restrict__ up,
                     float *__restrict__ mix, int m, int width, int rank, int groups) {
    constexpr int chunk = kChunk<T>;
    constexpr int kAcc = R * T;
    __shared__ float act[T * chunk];
    const int lane = threadIdx.x & 31;
    const int tok0 = blockIdx.y * T;
    const bool once = rank <= chunk;
    for (int g = 0; g < groups; ++g) {
        const int row0 = ((blockIdx.x * groups + g) * kWarps + (threadIdx.x >> 5)) * R;
        const W *rp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rp[r] = up + (size_t)min(row0 + r, width - 1) * rank;
        float acc[kAcc];
#pragma unroll
        for (int a = 0; a < kAcc; ++a) acc[a] = 0.0f;
        for (int at = 0; at < rank; at += chunk) {
            const int len = min(chunk, rank - at);
            if (!once || g == 0) {
                __syncthreads();
                stage_tile<T>(act, low, tok0, m, (size_t)rank, at, len);
                __syncthreads();
            }
            tile_dot<W, T, R>(act, rp, at, len, lane, acc);
        }
        const float sum = fold<kAcc, 16>(acc, lane);
        constexpr int kLow = 5 - kLog2<kAcc>;
        const int a = lane >> kLow;
        const int r = row0 + a / T;
        const int t = tok0 + a % T;
        if ((lane & ((1 << kLow) - 1)) == 0 && r < width && t < m) {
            mix[(size_t)t * width + r] = sum;
        }
    }
}

__global__ void hyper_combine_k(const float *__restrict__ normed, const float *__restrict__ mix,
                                float *__restrict__ mixed, int branches, int hidden) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= hidden) return;
    const int t = blockIdx.x;
    const size_t base = (size_t)t * branches * hidden + d;
    float sum = 0.0f;
    for (int b = 0; b < branches; ++b) {
        const size_t at = base + (size_t)b * hidden;
        sum += sigmoid_f(mix[at]) * normed[at];
    }
    mixed[(size_t)t * hidden + d] = sum / (float)branches;
}

// ------------------------------------------------------------------ Q8_0 matrices
// The read of dtype `kDtypeQ80`: `down` and `up` are Q8_0 (`Q80`), `norm_w` and `inject` F16. Each
// kernel keeps the arithmetic of its F16 counterpart row for row, so tiled and one-token give the
// same bits here too.

// One token's down projection: the arithmetic of hyper_down_k row for row.
template <typename V>
__global__ void hyper_down_q80_k(const float *__restrict__ normed, Q80 down,
                                 const V *__restrict__ inject, float *__restrict__ low,
                                 float *__restrict__ gate, int branches, int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * kWarps + (threadIdx.x >> 5);
    const int rows = rank + (inject != NULL ? branches : 0);
    if (r >= rows) return;
    const int t = blockIdx.x;
    const float *in = normed + (size_t)t * width;
    float acc = 0.0f;
    if (r < rank) {
        const Q80 w = down + (size_t)r * width;
        for (int i = lane; i < width; i += 32) acc += load(w, i) * in[i];
    } else {
        const V *w = inject + (size_t)(r - rank) * width;
        for (int i = lane; i < width; i += 32) acc += load(w, i) * in[i];
    }
    acc = warp_sum(acc);
    if (lane != 0) return;
    const float scaled = acc / (float)branches;
    if (r < rank) {
        low[(size_t)t * rank + r] = scaled * sigmoid_f(scaled);
    } else {
        gate[(size_t)t * branches + (r - rank)] = 2.0f * sigmoid_f(scaled);
    }
}

// One token's up projection: the arithmetic of hyper_up_k row for row.
__global__ void hyper_up_q80_k(const float *__restrict__ low, Q80 up, float *__restrict__ mix,
                               int width, int rank) {
    const int lane = threadIdx.x & 31;
    const int i = blockIdx.y * kWarps + (threadIdx.x >> 5);
    if (i >= width) return;
    const int t = blockIdx.x;
    const float *in = low + (size_t)t * rank;
    const Q80 w = up + (size_t)i * rank;
    float acc = 0.0f;
    for (int j = lane; j < rank; j += 32) acc += load(w, j) * in[j];
    acc = warp_sum(acc);
    if (lane == 0) mix[(size_t)t * width + i] = acc;
}

template <typename W, int T, int R>
__device__ __forceinline__ void tile_dot(const float *act, const W (&row)[R], int at, int len,
                                         int lane, float (&acc)[R * T]) {
    constexpr int chunk = kChunk<T>;
#pragma unroll 4
    for (int j = lane; j < len; j += 32) {
        float wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) wv[r] = load(row[r], (size_t)at + j);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float a = act[t * chunk + j];
#pragma unroll
            for (int r = 0; r < R; ++r) acc[r * T + t] += wv[r] * a;
        }
    }
}

// The weight type of one row in a `rp` array of the tiled helpers.
template <typename Rows>
struct RowWeight;
template <int R>
struct RowWeight<Q80[R]> {
    using type = Q80;
};
template <typename V, int R>
struct RowWeight<const V *[R]> {
    using type = V;
};

// One warp's `rp`: R rows of one weight for the T tokens of blockIdx.y, the staging and the
// barriers of hyper_down_tiled_k. `row0` is the first row's global index; inject rows count from
// `rank`.
template <typename Rows, int T, int R>
__device__ __forceinline__ void down_rows(float *act, const float *normed, const Rows &rp, int row0,
                                          int rows, int rank, int branches, int width, int m,
                                          int lane, float *__restrict__ low,
                                          float *__restrict__ gate) {
    constexpr int chunk = kChunk<T>;
    constexpr int kAcc = R * T;
    using W = typename RowWeight<Rows>::type;
    const int tok0 = blockIdx.y * T;
    float acc[kAcc];
#pragma unroll
    for (int a = 0; a < kAcc; ++a) acc[a] = 0.0f;
    for (int at = 0; at < width; at += chunk) {
        const int len = min(chunk, width - at);
        __syncthreads();
        stage_tile<T>(act, normed, tok0, m, (size_t)width, at, len);
        __syncthreads();
        tile_dot<W, T, R>(act, rp, at, len, lane, acc);
    }
    const float sum = fold<kAcc, 16>(acc, lane);
    constexpr int kLow = 5 - kLog2<kAcc>;
    const int a = lane >> kLow;
    const int r = row0 + a / T;
    const int t = tok0 + a % T;
    if ((lane & ((1 << kLow) - 1)) != 0 || r >= rows || t >= m) return;
    const float scaled = sum / (float)branches;
    if (r < rank) {
        low[(size_t)t * rank + r] = scaled * sigmoid_f(scaled);
    } else {
        gate[(size_t)t * branches + (r - rank)] = 2.0f * sigmoid_f(scaled);
    }
}

// The tiled down projection. Every block's rows are of one kind, the first `down_blocks` blocks
// the matrix and the rest the inject rows, so a block's warps never part at a barrier, as they
// would where a warp's R rows straddle `rank`.
template <typename V, int T, int R>
__global__ void __launch_bounds__(kWarps * 32)
    hyper_down_q80_tiled_k(const float *__restrict__ normed, Q80 down,
                           const V *__restrict__ inject, float *__restrict__ low,
                           float *__restrict__ gate, int m, int branches, int width, int rank,
                           int down_blocks) {
    constexpr int chunk = kChunk<T>;
    __shared__ float act[T * chunk];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int rows = rank + (inject != NULL ? branches : 0);
    if (blockIdx.x < down_blocks) {
        const int row0 = (blockIdx.x * kWarps + warp) * R;
        Q80 rp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rp[r] = down + (size_t)min(row0 + r, rank - 1) * width;
        down_rows<Q80[R], T, R>(act, normed, rp, row0, rows, rank, branches, width, m, lane, low,
                                gate);
    } else {
        const int row0 = rank + ((blockIdx.x - down_blocks) * kWarps + warp) * R;
        const V *rp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            rp[r] = inject + (size_t)(min(row0 + r, rows - 1) - rank) * width;
        }
        down_rows<const V *[R], T, R>(act, normed, rp, row0, rows, rank, branches, width, m, lane,
                                      low, gate);
    }
}

// One warp's R rows of the up matrix for the T tokens of blockIdx.y: the staging and the barriers
// of hyper_up_tiled_k; a rank that fits one chunk is staged once per block, on the first pass.
template <int T, int R>
__device__ __forceinline__ void up_rows(float *act, const float *low, const Q80 (&rp)[R], int row0,
                                        int rank, int width, int m, int lane, bool first,
                                        float *__restrict__ mix) {
    constexpr int chunk = kChunk<T>;
    constexpr int kAcc = R * T;
    const int tok0 = blockIdx.y * T;
    const bool once = rank <= chunk;
    float acc[kAcc];
#pragma unroll
    for (int a = 0; a < kAcc; ++a) acc[a] = 0.0f;
    for (int at = 0; at < rank; at += chunk) {
        const int len = min(chunk, rank - at);
        if (!once || first) {
            __syncthreads();
            stage_tile<T>(act, low, tok0, m, (size_t)rank, at, len);
            __syncthreads();
        }
        tile_dot<Q80, T, R>(act, rp, at, len, lane, acc);
    }
    const float sum = fold<kAcc, 16>(acc, lane);
    constexpr int kLow = 5 - kLog2<kAcc>;
    const int a = lane >> kLow;
    const int r = row0 + a / T;
    const int t = tok0 + a % T;
    if ((lane & ((1 << kLow) - 1)) == 0 && r < width && t < m) {
        mix[(size_t)t * width + r] = sum;
    }
}

template <int T, int R>
__global__ void __launch_bounds__(kWarps * 32)
    hyper_up_q80_tiled_k(const float *__restrict__ low, Q80 up, float *__restrict__ mix, int m,
                         int width, int rank, int groups) {
    constexpr int chunk = kChunk<T>;
    __shared__ float act[T * chunk];
    const int lane = threadIdx.x & 31;
    for (int g = 0; g < groups; ++g) {
        const int row0 = ((blockIdx.x * groups + g) * kWarps + (threadIdx.x >> 5)) * R;
        Q80 rp[R];
#pragma unroll
        for (int r = 0; r < R; ++r) rp[r] = up + (size_t)min(row0 + r, width - 1) * rank;
        up_rows<T, R>(act, low, rp, row0, rank, width, m, lane, g == 0, mix);
    }
}

/* One apply kernel for both uses. With Mix = false it is the in-place gated write
 * x[t][b][d] += gate[t][b] * block[t][d] (`x_in` is not read; the instruction stream is that of
 * the original kernel). With Mix = true it writes out[t][dst][d] = sum_src mix[t][dst + src *
 * branches] * x_in[t][src][d] + gate[t][dst] * block[t][d], out and x_in apart. */
template <bool Mix>
__global__ void hyper_apply_k(float *__restrict__ x, const float *__restrict__ x_in,
                              const float *__restrict__ mix, const float *__restrict__ gate,
                              const float *__restrict__ block, int branches, int hidden) {
    const int i = blockIdx.y * blockDim.x + threadIdx.x;
    const int width = branches * hidden;
    if (i >= width) return;
    const int t = blockIdx.x;
    const int b = i / hidden;
    const int d = i - b * hidden;
    if constexpr (Mix) {
        float acc = 0.0f;
        for (int src = 0; src < branches; ++src)
            acc += x_in[(size_t)t * width + src * hidden + d] *
                   mix[(size_t)t * branches * branches + b + src * branches];
        acc += block[(size_t)t * hidden + d] * gate[(size_t)t * branches + b];
        x[(size_t)t * width + i] = acc;
    } else {
        (void)x_in;
        (void)mix;
        x[(size_t)t * width + i] += gate[(size_t)t * branches + b] * block[(size_t)t * hidden + d];
    }
}

bool shape_ok(int m, int branches, int hidden) {
    return m >= 1 && m <= kMaxM && branches >= 1 && branches <= kMaxBranches && hidden >= 1 &&
           hidden <= kMaxHidden && (long long)branches * hidden <= kMaxWidth;
}

// Mirrors the Rust wrapper term by term; the Rust side refuses everything this refuses.
bool valid_read(const inference_cuda_hyper_read_args &a, void *stream) {
    if (stream == NULL || a.x == NULL || a.norm_w == NULL || a.down == NULL || a.up == NULL ||
        a.normed == NULL || a.low == NULL || a.mix == NULL || a.mixed == NULL) {
        return false;
    }
    if ((a.inject == NULL) != (a.gate == NULL)) return false;
    if (!shape_ok(a.m, a.branches, a.hidden) || a.rank < 1 || a.rank > kMaxRank) return false;
    if (a.weight_dtype != kDtypeF16 && a.weight_dtype != kDtypeBf16 &&
        a.weight_dtype != kDtypeQ80) {
        return false;
    }
    if (!(a.eps >= 0.0f) || !(a.eps < INFINITY)) return false;
    const void *f32s[] = {a.x, a.normed, a.low, a.mix, a.mixed};
    for (const void *p : f32s) {
        if (!aligned(p, 4)) return false;
    }
    if (a.gate != NULL && !aligned(a.gate, 4)) return false;
    const void *weights[] = {a.norm_w, a.down, a.up};
    for (const void *p : weights) {
        if (!aligned(p, 2)) return false;
    }
    if (a.inject != NULL && !aligned(a.inject, 2)) return false;
    const size_t m = (size_t)a.m, width = (size_t)a.branches * a.hidden, rank = (size_t)a.rank;
    // Q8_0 rows start on a block, so the width and the rank take whole blocks.
    if (a.weight_dtype == kDtypeQ80 && (width % kQ80Elems != 0 || rank % kQ80Elems != 0)) {
        return false;
    }
    // A Q8_0 matrix takes 34 bytes a block of 32 elements.
    const size_t matrix_bytes = a.weight_dtype == kDtypeQ80
                                    ? rank * width / kQ80Elems * kQ80Bytes
                                    : rank * width * 2;
    const Span written[] = {{a.normed, m * width * 4},
                            {a.low, m * rank * 4},
                            {a.mix, m * width * 4},
                            {a.mixed, m * (size_t)a.hidden * 4},
                            {a.gate, a.gate != NULL ? m * (size_t)a.branches * 4 : 0}};
    const Span read[] = {{a.x, m * width * 4},
                         {a.norm_w, width * 2},
                         {a.down, matrix_bytes},
                         {a.up, matrix_bytes},
                         {a.inject, a.inject != NULL ? (size_t)a.branches * width * 2 : 0}};
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 5; ++j) {
            if (!apart(written[i], written[j])) return false;
        }
        for (const Span &r : read) {
            if (!apart(written[i], r)) return false;
        }
    }
    return true;
}

bool valid_mix_apply(const inference_cuda_hyper_mix_apply_args &a, void *stream) {
    if (stream == NULL || a.x == NULL || a.out == NULL || a.mix == NULL || a.gate == NULL ||
        a.block == NULL) {
        return false;
    }
    if (!shape_ok(a.m, a.branches, a.hidden)) return false;
    if (!aligned(a.x, 4) || !aligned(a.out, 4) || !aligned(a.mix, 4) || !aligned(a.gate, 4) ||
        !aligned(a.block, 4)) {
        return false;
    }
    const size_t m = (size_t)a.m, width = (size_t)a.branches * a.hidden;
    const Span out = {a.out, m * width * 4};
    return apart(out, {a.x, m * width * 4}) &&
           apart(out, {a.mix, m * (size_t)a.branches * a.branches * 4}) &&
           apart(out, {a.gate, m * (size_t)a.branches * 4}) &&
           apart(out, {a.block, m * (size_t)a.hidden * 4});
}

bool valid_apply(const inference_cuda_hyper_apply_args &a, void *stream) {
    if (stream == NULL || a.x == NULL || a.gate == NULL || a.block == NULL) return false;
    if (!shape_ok(a.m, a.branches, a.hidden)) return false;
    if (!aligned(a.x, 4) || !aligned(a.gate, 4) || !aligned(a.block, 4)) return false;
    const size_t m = (size_t)a.m, width = (size_t)a.branches * a.hidden;
    const Span x = {a.x, m * width * 4};
    return apart(x, {a.gate, m * (size_t)a.branches * 4}) &&
           apart(x, {a.block, m * (size_t)a.hidden * 4});
}

template <typename W, int T>
void launch_down_tiled(const inference_cuda_hyper_read_args &a, cudaStream_t s, int width) {
    const int rows = a.rank + (a.inject != NULL ? a.branches : 0);
    const int per_block = kWarps * kRowsPerWarp;
    const dim3 grid((rows + per_block - 1) / per_block, (a.m + T - 1) / T);
    hyper_down_tiled_k<W, T, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
        (const float *)a.normed, (const W *)a.down, (const W *)a.inject, (float *)a.low,
        (float *)a.gate, a.m, a.branches, width, a.rank);
}

template <typename W, int T>
void launch_up_tiled(const inference_cuda_hyper_read_args &a, cudaStream_t s, int width) {
    const int per_block = kWarps * kRowsPerWarp * kUpGroups;
    const dim3 grid((width + per_block - 1) / per_block, (a.m + T - 1) / T);
    hyper_up_tiled_k<W, T, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
        (const float *)a.low, (const W *)a.up, (float *)a.mix, a.m, width, a.rank, kUpGroups);
}

// `tiled` false runs the one-token kernels for every m: the reference of the equality test.
template <typename W>
void launch_read(const inference_cuda_hyper_read_args &a, cudaStream_t s, bool tiled,
                 Failure &failure) {
    const int width = a.branches * a.hidden;
    const W *norm_w = (const W *)a.norm_w;
    hyper_norm_k<W><<<a.m * a.branches, kNormThreads, 0, s>>>(
        (const float *)a.x, norm_w, (float *)a.normed, a.branches, a.hidden, a.eps);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    if (tiled && a.m >= kDownTiledMinM) {
        if (a.m >= 128) {
            launch_down_tiled<W, 8>(a, s, width);
        } else {
            launch_down_tiled<W, 4>(a, s, width);
        }
    } else {
        const int down_rows = a.rank + (a.inject != NULL ? a.branches : 0);
        const dim3 down_grid(a.m, (down_rows + kWarps - 1) / kWarps);
        hyper_down_k<W><<<down_grid, kWarps * 32, 0, s>>>(
            (const float *)a.normed, (const W *)a.down, (const W *)a.inject, (float *)a.low,
            (float *)a.gate, a.branches, width, a.rank);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    if (tiled && a.m >= kUpTiledMinM) {
        if (a.m >= 32) {
            launch_up_tiled<W, 8>(a, s, width);
        } else {
            launch_up_tiled<W, 4>(a, s, width);
        }
    } else {
        const dim3 up_grid(a.m, (width + kWarps - 1) / kWarps);
        hyper_up_k<W><<<up_grid, kWarps * 32, 0, s>>>((const float *)a.low, (const W *)a.up,
                                                       (float *)a.mix, width, a.rank);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const dim3 combine_grid(a.m, (a.hidden + 255) / 256);
    hyper_combine_k<<<combine_grid, 256, 0, s>>>((const float *)a.normed, (const float *)a.mix,
                                                 (float *)a.mixed, a.branches, a.hidden);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

// The read of dtype `kDtypeQ80`; `tiled` false runs the one-token kernels for every m, as in
// `launch_read`.
template <typename V>
void launch_read_q80(const inference_cuda_hyper_read_args &a, cudaStream_t s, bool tiled,
                     Failure &failure) {
    const int width = a.branches * a.hidden;
    const V *norm_w = (const V *)a.norm_w;
    const V *inject = (const V *)a.inject;
    const Q80 down{(const uint8_t *)a.down};
    const Q80 up{(const uint8_t *)a.up};
    const float *normed = (const float *)a.normed;
    float *low = (float *)a.low, *gate = (float *)a.gate, *mix = (float *)a.mix;
    hyper_norm_k<V><<<a.m * a.branches, kNormThreads, 0, s>>>(
        (const float *)a.x, norm_w, (float *)a.normed, a.branches, a.hidden, a.eps);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const int per_block = kWarps * kRowsPerWarp;
    if (tiled && a.m >= kDownTiledMinM) {
        const int down_blocks = (a.rank + per_block - 1) / per_block;
        const int inject_blocks = a.inject != NULL ? (a.branches + per_block - 1) / per_block : 0;
        if (a.m >= 128) {
            const dim3 grid(down_blocks + inject_blocks, (a.m + 7) / 8);
            hyper_down_q80_tiled_k<V, 8, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
                normed, down, inject, low, gate, a.m, a.branches, width, a.rank, down_blocks);
        } else {
            const dim3 grid(down_blocks + inject_blocks, (a.m + 3) / 4);
            hyper_down_q80_tiled_k<V, 4, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
                normed, down, inject, low, gate, a.m, a.branches, width, a.rank, down_blocks);
        }
    } else {
        const int rows = a.rank + (a.inject != NULL ? a.branches : 0);
        const dim3 down_grid(a.m, (rows + kWarps - 1) / kWarps);
        hyper_down_q80_k<V><<<down_grid, kWarps * 32, 0, s>>>(
            normed, down, inject, low, gate, a.branches, width, a.rank);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    if (tiled && a.m >= kUpTiledMinM) {
        const int up_per_block = per_block * kUpGroups;
        if (a.m >= 32) {
            const dim3 grid((width + up_per_block - 1) / up_per_block, (a.m + 7) / 8);
            hyper_up_q80_tiled_k<8, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
                low, up, mix, a.m, width, a.rank, kUpGroups);
        } else {
            const dim3 grid((width + up_per_block - 1) / up_per_block, (a.m + 3) / 4);
            hyper_up_q80_tiled_k<4, kRowsPerWarp><<<grid, kWarps * 32, 0, s>>>(
                low, up, mix, a.m, width, a.rank, kUpGroups);
        }
    } else {
        const dim3 up_grid(a.m, (width + kWarps - 1) / kWarps);
        hyper_up_q80_k<<<up_grid, kWarps * 32, 0, s>>>(low, up, mix, width, a.rank);
    }
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    if (!failure.ok()) return;
    const dim3 combine_grid(a.m, (a.hidden + 255) / 256);
    hyper_combine_k<<<combine_grid, 256, 0, s>>>(normed, (const float *)a.mix, (float *)a.mixed,
                                                 a.branches, a.hidden);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
}

int read_entry(const inference_cuda_hyper_read_args *args, void *stream, int *out_stage,
               bool tiled) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_read(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Failure failure;
        if (args->weight_dtype == kDtypeQ80) {
            launch_read_q80<__half>(*args, (cudaStream_t)stream, tiled, failure);
        } else if (args->weight_dtype == kDtypeF16) {
            launch_read<__half>(*args, (cudaStream_t)stream, tiled, failure);
        } else {
            launch_read<__nv_bfloat16>(*args, (cudaStream_t)stream, tiled, failure);
        }
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

}  // namespace

extern "C" int inference_cuda_hyper_read_launch(const inference_cuda_hyper_read_args *args,
                                                void *stream, int *out_stage) {
    return read_entry(args, stream, out_stage, true);
}

extern "C" int inference_cuda_hyper_read_reference_launch(
    const inference_cuda_hyper_read_args *args, void *stream, int *out_stage) {
    return read_entry(args, stream, out_stage, false);
}

extern "C" int inference_cuda_hyper_apply_launch(const inference_cuda_hyper_apply_args *args,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_apply(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int width = args->branches * args->hidden;
        const dim3 grid(args->m, (width + 255) / 256);
        Failure failure;
        hyper_apply_k<false><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (float *)args->x, NULL, NULL, (const float *)args->gate,
            (const float *)args->block, args->branches, args->hidden);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_hyper_mix_apply_launch(
    const inference_cuda_hyper_mix_apply_args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || !valid_mix_apply(*args, stream)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int width = args->branches * args->hidden;
        const dim3 grid(args->m, (width + 255) / 256);
        Failure failure;
        hyper_apply_k<true><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (float *)args->out, (const float *)args->x, (const float *)args->mix,
            (const float *)args->gate, (const float *)args->block, args->branches, args->hidden);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
