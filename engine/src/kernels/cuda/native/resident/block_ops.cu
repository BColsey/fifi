/* Entry and kernels of block_ops. See block_ops.h for the slot table of each operation.
 *
 * Numerics: f32 throughout with IEEE expf/tanhf/logf (this unit is built without
 * -use_fast_math). Reductions run in a fixed order (warp shuffle tree, then the warp
 * partials summed in index order), so a launch repeats bit for bit. q8_1 blocks follow the
 * vendored quantize rule: d = amax / 127, q = roundf(x / d), s = the raw sum of the inputs. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/block_ops.h"

// The vendored header defines block_q8_1 and keeps static host helpers that are unreferenced in
// this glue TU (the quantize_q8_1 glue includes it the same way).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "quantize.cuh"
#pragma GCC diagnostic pop
#include "resident/q8_1_store.cuh"

namespace {

constexpr int kQ8Block = 36;     /* bytes of one block_q8_1: half2 ds + 32 codes */
constexpr int kQ8Padding = 512;  /* values per row are padded to a multiple of this */

constexpr int kHcStreams = 4;
constexpr int kHcMixes = 17;     /* 4 pre + 4 post + 9 comb */

constexpr int kMlaRank = 512;
constexpr int kMlaRope = 64;
constexpr int kMlaWidth = kMlaRank + kMlaRope;
constexpr int kMlaGroup = 8;     /* query heads per attention block */
constexpr int kMlaTile = 32;     /* key positions per tile */
constexpr int kMlaRowHalves = kMlaWidth + 2;  /* padded smem row: 289 words, no bank conflicts */
constexpr int kMlaPartStride = kMlaRank + 4;  /* acc[512], m, l, two spare */

constexpr int kRdaDim = 128;

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplus_f(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* Sum over the block; `scratch` holds one float per warp. Every thread gets the total. */
__device__ float block_sum(float v, float *scratch) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.0f;
    const int warps = blockDim.x >> 5;
    for (int i = 0; i < warps; ++i) total += scratch[i];
    return total;
}

__host__ __device__ size_t q8_row_bytes(long long k) {
    const long long padded = (k + kQ8Padding - 1) / kQ8Padding * kQ8Padding;
    return (size_t)(padded / 32) * kQ8Block;
}

/* Zeroes the padding blocks of one row after its `k / 32` live blocks. */
__device__ void zero_q8_tail(unsigned char *row, long long k) {
    const long long live = k / 32;
    const long long total = (k + kQ8Padding - 1) / kQ8Padding * kQ8Padding / 32;
    for (long long i = (long long)threadIdx.x; i < (total - live) * (kQ8Block / 4); i += blockDim.x)
        reinterpret_cast<unsigned *>(row + live * kQ8Block)[i] = 0u;
}

/* ---- mixed residual ---------------------------------------------------------------- */

__global__ void __launch_bounds__(256) hc_read_kernel(
    const float *resid, const __half *fn, const float *base, const float *scale, float *branch,
    float *post_out, float *comb_out, int h, float eps, float magnitude) {
    __shared__ float red[kHcMixes + 1][8];
    __shared__ float s_pre[kHcStreams];
    const int t = blockIdx.x;
    const int width = kHcStreams * h;
    const float *inp = resid + (size_t)t * width;
    float dots[kHcMixes];
#pragma unroll
    for (int m = 0; m < kHcMixes; ++m) dots[m] = 0.0f;
    float ss = 0.0f;
    for (int r = threadIdx.x * 2; r < width; r += blockDim.x * 2) {
        const float2 v = *reinterpret_cast<const float2 *>(inp + r);
        ss += v.x * v.x + v.y * v.y;
#pragma unroll
        for (int m = 0; m < kHcMixes; ++m) {
            const float2 f =
                __half22float2(*reinterpret_cast<const __half2 *>(fn + (size_t)m * width + r));
            dots[m] += f.x * v.x + f.y * v.y;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    ss = warp_sum(ss);
#pragma unroll
    for (int m = 0; m < kHcMixes; ++m) dots[m] = warp_sum(dots[m]);
    if (lane == 0) {
        red[kHcMixes][warp] = ss;
#pragma unroll
        for (int m = 0; m < kHcMixes; ++m) red[m][warp] = dots[m];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot[kHcMixes + 1];
        for (int m = 0; m <= kHcMixes; ++m) {
            float s = 0.0f;
            for (int w = 0; w < 8; ++w) s += red[m][w];
            tot[m] = s;
        }
        const float rms = sqrtf(tot[kHcMixes] / (float)width + eps);
        float mixes[kHcMixes];
        for (int m = 0; m < kHcMixes; ++m) mixes[m] = tot[m] / rms;
        float post[kHcStreams];
        for (int s = 0; s < kHcStreams; ++s) {
            s_pre[s] = sigmoid_f(mixes[s] * scale[0] + base[s]);
            post[s] = magnitude * sigmoid_f(mixes[kHcStreams + s] * scale[1] + base[kHcStreams + s]);
            post_out[(size_t)t * kHcStreams + s] = post[s];
        }
        /* z[j][q] of the 3x3 block, then e = H^T z H with the Helmert basis rows. */
        float z[3][3];
        for (int j = 0; j < 3; ++j)
            for (int q = 0; q < 3; ++q)
                z[j][q] = mixes[2 * kHcStreams + j * 3 + q] * scale[2] + base[2 * kHcStreams + j * 3 + q];
        float helmert[kHcStreams][3];
        for (int i = 0; i < kHcStreams; ++i)
            for (int j = 0; j < 3; ++j) {
                const float v = i <= j ? 1.0f : (i == j + 1 ? -(float)(j + 1) : 0.0f);
                helmert[i][j] = v / sqrtf((float)((j + 1) * (j + 2)));
            }
        float e[kHcStreams][kHcStreams];
        float rho = 0.0f;
        for (int i = 0; i < kHcStreams; ++i)
            for (int x = 0; x < kHcStreams; ++x) {
                float acc = 0.0f;
                for (int j = 0; j < 3; ++j)
                    for (int q = 0; q < 3; ++q) acc += helmert[x][q] * z[j][q] * helmert[i][j];
                e[i][x] = acc;
                const float chart = 0.5f * (i == x ? 1.0f : 0.0f) + 0.5f / (float)kHcStreams;
                rho = fmaxf(rho, -acc / chart);
            }
        rho = 1.0f + fmaxf(rho, 0.0f);
        const float gamma = sigmoid_f(scale[3]);
        for (int i = 0; i < kHcStreams; ++i)
            for (int x = 0; x < kHcStreams; ++x) {
                const float ident = i == x ? 1.0f : 0.0f;
                const float phi = e[i][x] / rho + 0.5f * ident + 0.5f / (float)kHcStreams;
                comb_out[(size_t)t * 16 + i + x * kHcStreams] = ident + gamma * (phi - ident);
            }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < h; k += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < kHcStreams; ++s) acc += inp[s * h + k] * s_pre[s];
        branch[(size_t)t * h + k] = acc;
    }
}

__global__ void hc_mean_kernel(const float *resid, float *out, int row, int h) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= h) return;
    const float *inp = resid + (size_t)row * kHcStreams * h;
    float acc = 0.0f;
    for (int s = 0; s < kHcStreams; ++s) acc += inp[s * h + k];
    out[k] = acc / (float)kHcStreams;
}

/* ---- quantize -------------------------------------------------------------------------- */

__global__ void __launch_bounds__(256) gated_quant_kernel(const float *x, const float *g,
                                                          unsigned char *q8, float *y, int w) {
    const int t = blockIdx.x;
    unsigned char *qrow = q8 == NULL ? NULL : q8 + (size_t)t * q8_row_bytes(w);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int b = warp; b < w / 32; b += warps) {
        const size_t at = (size_t)t * w + (size_t)b * 32 + lane;
        float v = x[at];
        if (g != NULL) v *= sigmoid_f(g[at]);
        if (y != NULL) y[at] = v;
        if (qrow != NULL) store_q8_1_block(reinterpret_cast<block_q8_1 *>(qrow + (size_t)b * kQ8Block), v);
    }
    if (qrow != NULL) zero_q8_tail(qrow, w);
}

__global__ void __launch_bounds__(256) situ_quant_kernel(const float *gate, const float *up,
                                                         unsigned char *q8, float *y, int w,
                                                         int ld_gate, int ld_up, float beta,
                                                         float lb) {
    const int t = blockIdx.x;
    unsigned char *qrow = q8 == NULL ? NULL : q8 + (size_t)t * q8_row_bytes(w);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int b = warp; b < w / 32; b += warps) {
        const int i = b * 32 + lane;
        const float g = gate[(size_t)t * ld_gate + i];
        const float u = up[(size_t)t * ld_up + i];
        const float act = tanhf(g / beta) * beta * sigmoid_f(g);
        const float upq = lb > 0.0f ? tanhf(u / lb) * lb : u;
        if (y != NULL) y[(size_t)t * w + i] = act * upq;
        if (qrow != NULL) store_q8_1_block(reinterpret_cast<block_q8_1 *>(qrow + (size_t)b * kQ8Block), act * upq);
    }
    if (qrow != NULL) zero_q8_tail(qrow, w);
}

__global__ void fill_kernel(float *p, int count, float value) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) p[i] = value;
}

/* ---- delta-rule scan with a decay per key channel ----------------------------------- */

/* One block per head, two threads per value channel w: each holds half of the state column
 * S[:][w] in registers. The per-token q, k and decay vectors are shared through smem; the
 * two halves of a pair add their partial dot products with one shuffle. */
__global__ void __launch_bounds__(2 * kRdaDim) rda_scan_kernel(
    const float *q, const float *k, const float *v, const float *g1, const float *dt_bias,
    const float *a, const float *beta, const float *delta, const float *norm_w, const float *g2,
    float *state, float *out, int n, int heads, float lower_bound, float norm_eps) {
    constexpr int kHalf = kRdaDim / 2;
    __shared__ float qs[kRdaDim], ks[kRdaDim], decay[kRdaDim];
    __shared__ float scratch[32];
    const int head = blockIdx.x;
    const int w = threadIdx.x >> 1;
    const int half = threadIdx.x & 1;
    const int width = heads * kRdaDim;
    const float qscale = 1.0f / sqrtf((float)kRdaDim);
    float s[kHalf];
    float *col = state + ((size_t)head * kRdaDim + w) * kRdaDim + half * kHalf;
#pragma unroll
    for (int u = 0; u < kHalf; ++u) s[u] = col[u];
    for (int t = 0; t < n; ++t) {
        const size_t at = (size_t)t * width + (size_t)head * kRdaDim + w;
        const float qv = q[at], kv = k[at];
        const float own = half == 0 ? 1.0f : 0.0f;
        const float qn = block_sum(own * qv * qv, scratch);
        const float kn = block_sum(own * kv * kv, scratch);
        if (half == 0) {
            qs[w] = qv * (1.0f / fmaxf(sqrtf(qn), 1e-6f));
            ks[w] = kv * (1.0f / fmaxf(sqrtf(kn), 1e-6f));
            const float raw = (g1[at] + dt_bias[head * kRdaDim + w]) * a[head];
            decay[w] = expf(lower_bound * sigmoid_f(-raw));
        }
        const float bi = beta[(size_t)t * heads + head];
        const float de = bi + delta[(size_t)t * heads + head];
        const float fold = expf(softplus_f(-bi) - softplus_f(-de));
        const float vs = v[at] * fold;
        const float bscan = sigmoid_f(bi);
        __syncthreads();
        float sk = 0.0f;
#pragma unroll
        for (int u = 0; u < kHalf; ++u) {
            s[u] *= decay[half * kHalf + u];
            sk += s[u] * ks[half * kHalf + u];
        }
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        const float dw = (vs - sk) * bscan;
        float acc = 0.0f;
#pragma unroll
        for (int u = 0; u < kHalf; ++u) {
            s[u] += ks[half * kHalf + u] * dw;
            acc += s[u] * qs[half * kHalf + u];
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        const float o = acc * qscale;
        const float ss = block_sum(own * o * o, scratch);
        const float rn = 1.0f / sqrtf(ss / (float)kRdaDim + norm_eps);
        if (half == 0) out[at] = o * rn * norm_w[w] * sigmoid_f(g2[at]);
        __syncthreads();
    }
#pragma unroll
    for (int u = 0; u < kHalf; ++u) col[u] = s[u];
}

/* ---- absorbed latent attention ------------------------------------------------------ */

/* qeff[t][h] = [ wk_b[h] . q_nope | q_rope ]: one block per (token, head), one warp per
 * latent row. */
__global__ void __launch_bounds__(128) mla_q_absorb_kernel(const float *qb, const float *wk_b,
                                                           float *qeff, int heads, int nope,
                                                           int rope, int rank) {
    extern __shared__ float qn[];
    const int t = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *qrow = qb + ((size_t)t * heads + h) * (nope + rope);
    for (int i = threadIdx.x; i < nope; i += blockDim.x) qn[i] = qrow[i];
    float *dst = qeff + ((size_t)t * heads + h) * (rank + rope);
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dst[rank + i] = qrow[nope + i];
    __syncthreads();
    const float *wh = wk_b + (size_t)h * rank * nope;
    for (int c = warp; c < rank; c += blockDim.x >> 5) {
        const float4 wv = *reinterpret_cast<const float4 *>(wh + (size_t)c * nope + lane * 4);
        const float4 qv = *reinterpret_cast<const float4 *>(qn + lane * 4);
        float acc = wv.x * qv.x + wv.y * qv.y + wv.z * qv.z + wv.w * qv.w;
        acc = warp_sum(acc);
        if (lane == 0) dst[c] = acc;
    }
}

/* cache[pos0 + t] = [ rms_norm(kv[:rank]) * w | kv[rank:] ] as f16. */
__global__ void __launch_bounds__(256) mla_kv_prep_kernel(const float *kv, const float *w,
                                                          __half *cache, int rank, int rope,
                                                          int pos0, float eps) {
    __shared__ float scratch[32];
    const int t = blockIdx.x;
    const float *row = kv + (size_t)t * (rank + rope);
    float ss = 0.0f;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) ss += row[i] * row[i];
    ss = block_sum(ss, scratch);
    const float scale = 1.0f / sqrtf(ss / (float)rank + eps);
    __half *dst = cache + (size_t)(pos0 + t) * (rank + rope);
    for (int i = threadIdx.x; i < rank; i += blockDim.x) dst[i] = __float2half(row[i] * scale * w[i]);
    for (int i = threadIdx.x; i < rope; i += blockDim.x) dst[rank + i] = __float2half(row[rank + i]);
}

/* Partial softmax attention of one (token, head group, key split). Writes the unnormalized
 * accumulator, the running max and the running sum per head. */
__global__ void __launch_bounds__(256) mla_attn_kernel(const float *qeff, const __half *cache,
                                                       float *part, int heads, int pos0,
                                                       int splits, int keys_per_split,
                                                       float scale) {
    extern __shared__ unsigned char smem_raw[];
    float *qs = reinterpret_cast<float *>(smem_raw);                 /* [8][576] */
    __half *kv = reinterpret_cast<__half *>(qs + kMlaGroup * kMlaWidth); /* [32][578] */
    float *pbuf = reinterpret_cast<float *>(kv + kMlaTile * kMlaRowHalves); /* [8][32] */
    const int t = blockIdx.x, g = blockIdx.y, sp = blockIdx.z;
    const int groups = heads / kMlaGroup;
    const int visible = pos0 + t + 1;
    const int lo = sp * keys_per_split;
    const int hi = min(lo + keys_per_split, visible);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float *qsrc = qeff + ((size_t)t * heads + (size_t)g * kMlaGroup) * kMlaWidth;
    for (int i = tid; i < kMlaGroup * kMlaWidth; i += 256) qs[i] = qsrc[i];
    float m = -INFINITY, l = 0.0f;
    float acc[kMlaRank / 32];
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) acc[i] = 0.0f;
    for (int base = lo; base < hi; base += kMlaTile) {
        const int cnt = min(kMlaTile, hi - base);
        __syncthreads();
        for (int idx = tid; idx < cnt * (kMlaWidth / 2); idx += 256) {
            const int p = idx / (kMlaWidth / 2), wd = idx % (kMlaWidth / 2);
            const unsigned v = reinterpret_cast<const unsigned *>(cache + (size_t)(base + p) * kMlaWidth)[wd];
            reinterpret_cast<unsigned *>(kv + p * kMlaRowHalves)[wd] = v;
        }
        __syncthreads();
        float sc = -INFINITY;
        if (lane < cnt) {
            const float *qh = qs + warp * kMlaWidth;
            const __half2 *kr = reinterpret_cast<const __half2 *>(kv + lane * kMlaRowHalves);
            float a = 0.0f;
#pragma unroll 4
            for (int d2 = 0; d2 < kMlaWidth / 2; ++d2) {
                const float2 kf = __half22float2(kr[d2]);
                a += qh[2 * d2] * kf.x + qh[2 * d2 + 1] * kf.y;
            }
            sc = a * scale;
        }
        const float mt = warp_max(sc);
        const float mnew = fmaxf(m, mt);
        const float corr = m == -INFINITY ? 0.0f : expf(m - mnew);
        const float p = lane < cnt ? expf(sc - mnew) : 0.0f;
        l = l * corr + warp_sum(p);
        m = mnew;
        pbuf[warp * kMlaTile + lane] = p;
#pragma unroll
        for (int i = 0; i < kMlaRank / 32; ++i) acc[i] *= corr;
        __syncwarp();
        for (int pp = 0; pp < cnt; ++pp) {
            const float pw = pbuf[warp * kMlaTile + pp];
            const __half *lat = kv + pp * kMlaRowHalves;
#pragma unroll
            for (int i = 0; i < kMlaRank / 32; ++i) acc[i] += pw * __half2float(lat[lane + 32 * i]);
        }
    }
    float *dst = part + ((((size_t)t * groups + g) * splits + sp) * kMlaGroup + warp) * kMlaPartStride;
#pragma unroll
    for (int i = 0; i < kMlaRank / 32; ++i) dst[lane + 32 * i] = acc[i];
    if (lane == 0) {
        dst[kMlaRank] = m;
        dst[kMlaRank + 1] = l;
    }
}

__global__ void __launch_bounds__(128) mla_reduce_kernel(const float *part, float *mix, int heads,
                                                         int splits) {
    const int t = blockIdx.x, h = blockIdx.y;
    const int groups = heads / kMlaGroup;
    const int g = h / kMlaGroup, hh = h % kMlaGroup;
    const float *base = part + (((size_t)t * groups + g) * splits) * kMlaGroup * kMlaPartStride +
                        (size_t)hh * kMlaPartStride;
    const size_t split_stride = (size_t)kMlaGroup * kMlaPartStride;
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s) mx = fmaxf(mx, base[s * split_stride + kMlaRank]);
    float total = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float ms = base[s * split_stride + kMlaRank];
        if (ms != -INFINITY) total += base[s * split_stride + kMlaRank + 1] * expf(ms - mx);
    }
    for (int c = threadIdx.x; c < kMlaRank; c += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float ms = base[s * split_stride + kMlaRank];
            if (ms != -INFINITY) acc += base[s * split_stride + c] * expf(ms - mx);
        }
        mix[((size_t)t * heads + h) * kMlaRank + c] = acc / total;
    }
}

/* out[t][h][v] = wv_b[h][v] . mix[t][h]: one block per (token, head), one warp per row. */
__global__ void __launch_bounds__(128) mla_project_kernel(const float *mix, const float *wv_b,
                                                          float *out, int heads, int rank, int v) {
    extern __shared__ float mx[];
    const int t = blockIdx.x, h = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float *m = mix + ((size_t)t * heads + h) * rank;
    for (int i = threadIdx.x; i < rank; i += blockDim.x) mx[i] = m[i];
    __syncthreads();
    const float *wh = wv_b + (size_t)h * v * rank;
    for (int r = warp; r < v; r += blockDim.x >> 5) {
        float acc = 0.0f;
        for (int c = lane * 4; c < rank; c += 128) {
            const float4 wv = *reinterpret_cast<const float4 *>(wh + (size_t)r * rank + c);
            acc += wv.x * mx[c] + wv.y * mx[c + 1] + wv.z * mx[c + 2] + wv.w * mx[c + 3];
        }
        acc = warp_sum(acc);
        if (lane == 0) out[((size_t)t * heads + h) * v + r] = acc;
    }
}

/* ---- biased sigmoid router ---------------------------------------------------------- */

/* One block per token, one thread per expert slot (up to 256). The routed experts are the
 * top `k` of sigmoid(logit) + bias, ties to the lower index; the weights are the unbiased
 * sigmoids, divided by their sum (clamped below) when `normalize` is set. */
__global__ void __launch_bounds__(256) moe_route_kernel(const float *x, const float *gate_w,
                                                        const float *bias, int *ids,
                                                        float *weights, int h, int e, int k,
                                                        int normalize, float min_sum) {
    extern __shared__ float xs[];  /* [h] */
    __shared__ float probs[256];
    __shared__ float sel[256];
    __shared__ float red_v[8];
    __shared__ int red_i[8];
    __shared__ int pick[8];
    __shared__ float pick_p[8];
    const int t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int i = threadIdx.x; i < h; i += blockDim.x) xs[i] = x[(size_t)t * h + i];
    __syncthreads();
    for (int ex = warp; ex < e; ex += warps) {
        const float *wr = gate_w + (size_t)ex * h;
        float acc = 0.0f;
        for (int i = lane * 4; i < h; i += 128) {
            const float4 wv = *reinterpret_cast<const float4 *>(wr + i);
            acc += wv.x * xs[i] + wv.y * xs[i + 1] + wv.z * xs[i + 2] + wv.w * xs[i + 3];
        }
        acc = warp_sum(acc);
        if (lane == 0) {
            const float p = sigmoid_f(acc);
            probs[ex] = p;
            sel[ex] = p + bias[ex];
        }
    }
    __syncthreads();
    for (int round = 0; round < k; ++round) {
        float best = -INFINITY;
        int best_i = 0x7fffffff;
        if (threadIdx.x < e) {
            best = sel[threadIdx.x];
            best_i = threadIdx.x;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
            if (ov > best || (ov == best && oi < best_i)) {
                best = ov;
                best_i = oi;
            }
        }
        if (lane == 0) {
            red_v[warp] = best;
            red_i[warp] = best_i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bv = red_v[0];
            int bi = red_i[0];
            for (int wi = 1; wi < warps; ++wi)
                if (red_v[wi] > bv || (red_v[wi] == bv && red_i[wi] < bi)) {
                    bv = red_v[wi];
                    bi = red_i[wi];
                }
            pick[round] = bi;
            pick_p[round] = probs[bi];
            sel[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int r = 0; r < k; ++r) sum += pick_p[r];
        sum = fmaxf(sum, min_sum);
        for (int r = 0; r < k; ++r) {
            ids[(size_t)t * k + r] = pick[r];
            weights[(size_t)t * k + r] = normalize ? pick_p[r] / sum : pick_p[r];
        }
    }
}

/* ---- entry ------------------------------------------------------------------------- */

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }

bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

struct Slots {
    const inference_cuda_block_ops_args *args;
    bool ok = true;
    bool written[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};
    bool used[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};
    size_t need[INFERENCE_CUDA_BLOCK_OPS_PTRS] = {};

    explicit Slots(const inference_cuda_block_ops_args *a) : args(a) {}

    /* A required slot of `bytes` bytes. */
    void use(int slot, long long bytes, bool write) {
        if (bytes <= 0 || args->ptr[slot] == NULL || args->bytes[slot] < (size_t)bytes ||
            !aligned(args->ptr[slot], 4)) {
            ok = false;
            return;
        }
        used[slot] = true;
        written[slot] = write;
        need[slot] = (size_t)bytes;
    }

    /* A slot that may be absent. */
    void maybe(int slot, long long bytes, bool write) {
        if (args->ptr[slot] != NULL) use(slot, bytes, write);
    }

    /* Slots that are not used must be NULL; written slots share no byte with another. */
    bool valid() const {
        if (!ok) return false;
        for (int i = 0; i < INFERENCE_CUDA_BLOCK_OPS_PTRS; ++i) {
            if (!used[i]) {
                if (args->ptr[i] != NULL) return false;
                continue;
            }
            if (!written[i]) continue;
            for (int j = 0; j < INFERENCE_CUDA_BLOCK_OPS_PTRS; ++j)
                if (j != i && used[j] && !apart(args->ptr[i], need[i], args->ptr[j], need[j]))
                    return false;
        }
        return true;
    }
};

int launch_status(int *out_stage) {
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_block_ops_launch(const inference_cuda_block_ops_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const cudaStream_t s = (cudaStream_t)stream;
        const int *i = args->ints;
        const float *f = args->floats;
        void *const *p = args->ptr;
        Slots slots(args);
        switch (args->op) {
        case INFERENCE_CUDA_BLOCK_OP_HC_READ: {
            const long long n = i[0], h = i[1], w = 4LL * h;
            if (n < 1 || n > 65536 || h < 64 || h > 8192 || h % 2 != 0)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * w * 4, false);
            slots.use(1, 17 * w * 2, false);
            slots.use(2, 17 * 4, false);
            slots.use(3, 4 * 4, false);
            slots.use(4, n * h * 4, true);
            slots.use(5, n * 4 * 4, true);
            slots.use(6, n * 16 * 4, true);
            if (!slots.valid() || !aligned(p[0], 8) || !aligned(p[1], 4))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            hc_read_kernel<<<(unsigned)n, 256, 0, s>>>((const float *)p[0], (const __half *)p[1],
                                                       (const float *)p[2], (const float *)p[3],
                                                       (float *)p[4], (float *)p[5], (float *)p[6],
                                                       (int)h, f[0], f[1]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_HC_MEAN: {
            const long long row = i[0], h = i[1];
            if (row < 0 || h < 32 || h > 8192) return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, (row + 1) * 4 * h * 4, false);
            slots.use(1, h * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            hc_mean_kernel<<<(unsigned)((h + 255) / 256), 256, 0, s>>>(
                (const float *)p[0], (float *)p[1], (int)row, (int)h);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_RDA_SCAN: {
            const long long n = i[0], heads = i[1], hd = i[2];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || hd != kRdaDim)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            const long long w = heads * hd;
            for (int slot = 0; slot < 4; ++slot) slots.use(slot, n * w * 4, false);
            slots.use(4, w * 4, false);
            slots.use(5, heads * 4, false);
            slots.use(6, n * heads * 4, false);
            slots.use(7, n * heads * 4, false);
            slots.use(8, hd * 4, false);
            slots.use(9, n * w * 4, false);
            slots.use(10, heads * hd * hd * 4, true);
            slots.use(11, n * w * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            rda_scan_kernel<<<(unsigned)heads, 2 * kRdaDim, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (const float *)p[2], (const float *)p[3],
                (const float *)p[4], (const float *)p[5], (const float *)p[6], (const float *)p[7],
                (const float *)p[8], (const float *)p[9], (float *)p[10], (float *)p[11], (int)n,
                (int)heads, f[0], f[1]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_Q_ABSORB: {
            const long long n = i[0], heads = i[1], nope = i[2], rope = i[3], rank = i[4];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || nope != 128 || rope != kMlaRope ||
                rank != kMlaRank)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * (nope + rope) * 4, false);
            slots.use(1, heads * rank * nope * 4, false);
            slots.use(2, n * heads * (rank + rope) * 4, true);
            if (!slots.valid() || !aligned(p[1], 16)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_q_absorb_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, (size_t)nope * 4, s>>>(
                (const float *)p[0], (const float *)p[1], (float *)p[2], (int)heads, (int)nope,
                (int)rope, (int)rank);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP: {
            const long long n = i[0], rank = i[1], rope = i[2], pos0 = i[3], cells = i[4];
            if (n < 1 || n > 65536 || rank != kMlaRank || rope != kMlaRope || pos0 < 0 ||
                pos0 + n > cells)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * (rank + rope) * 4, false);
            slots.use(1, rank * 4, false);
            slots.use(2, cells * (rank + rope) * 2, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_kv_prep_kernel<<<(unsigned)n, 256, 0, s>>>((const float *)p[0], (const float *)p[1],
                                                           (__half *)p[2], (int)rank, (int)rope,
                                                           (int)pos0, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_ATTN: {
            const long long n = i[0], heads = i[1], rank = i[2], rope = i[3], pos0 = i[4];
            const long long splits = i[5], kps = i[6];
            if (n < 1 || n > 65536 || heads < kMlaGroup || heads > 256 || heads % kMlaGroup != 0 ||
                rank != kMlaRank || rope != kMlaRope || pos0 < 0 || splits < 1 || splits > 64 ||
                kps < 32 || kps % 32 != 0 || splits * kps < pos0 + n)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * (rank + rope) * 4, false);
            slots.use(1, (pos0 + n) * (rank + rope) * 2, false);
            slots.use(2, n * (heads / kMlaGroup) * splits * kMlaGroup * kMlaPartStride * 4, true);
            if (!slots.valid() || !aligned(p[1], 4)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            const size_t smem = (size_t)kMlaGroup * kMlaWidth * 4 +
                                (size_t)kMlaTile * kMlaRowHalves * 2 +
                                (size_t)kMlaGroup * kMlaTile * 4;
            // The dynamic shared-memory limit is a per-device attribute of the function.
            static bool configured[16] = {};
            int device = 0;
            if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) device = 0;
            if (!configured[device]) {
                Failure failure;
                failure.record(cudaFuncSetAttribute(mla_attn_kernel,
                                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                    (int)smem),
                               INFERENCE_CUDA_STAGE_LAUNCH);
                report_stage(out_stage, failure.stage);
                if (!failure.ok()) return (int)failure.err;
                configured[device] = true;
            }
            mla_attn_kernel<<<dim3((unsigned)n, (unsigned)(heads / kMlaGroup), (unsigned)splits),
                              256, smem, s>>>((const float *)p[0], (const __half *)p[1],
                                              (float *)p[2], (int)heads, (int)pos0, (int)splits,
                                              (int)kps, f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_REDUCE: {
            const long long n = i[0], heads = i[1], rank = i[2], splits = i[3];
            if (n < 1 || n > 65536 || heads < kMlaGroup || heads > 256 || heads % kMlaGroup != 0 ||
                rank != kMlaRank || splits < 1 || splits > 64)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * (heads / kMlaGroup) * splits * kMlaGroup * kMlaPartStride * 4, false);
            slots.use(1, n * heads * rank * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_reduce_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, 0, s>>>(
                (const float *)p[0], (float *)p[1], (int)heads, (int)splits);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MLA_PROJECT: {
            const long long n = i[0], heads = i[1], rank = i[2], v = i[3];
            if (n < 1 || n > 65536 || heads < 1 || heads > 256 || rank != kMlaRank || v < 1 ||
                v > 1024)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * heads * rank * 4, false);
            slots.use(1, heads * v * rank * 4, false);
            slots.use(2, n * heads * v * 4, true);
            if (!slots.valid() || !aligned(p[1], 16)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            mla_project_kernel<<<dim3((unsigned)n, (unsigned)heads), 128, (size_t)rank * 4, s>>>(
                (const float *)p[0], (const float *)p[1], (float *)p[2], (int)heads, (int)rank,
                (int)v);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_SITU_QUANT: {
            const long long rows = i[0], w = i[1], ldg = i[2], ldu = i[3];
            if (rows < 1 || rows > 1048576 || w < 32 || w > 32768 || w % 32 != 0 || ldg < w ||
                ldu < w)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, ((rows - 1) * ldg + w) * 4, false);
            slots.use(1, ((rows - 1) * ldu + w) * 4, false);
            slots.maybe(2, rows * (long long)q8_row_bytes(w), true);
            slots.maybe(3, rows * w * 4, true);
            if (!slots.valid() || (p[2] == NULL && p[3] == NULL))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            situ_quant_kernel<<<(unsigned)rows, 256, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (unsigned char *)p[2], (float *)p[3],
                (int)w, (int)ldg, (int)ldu, f[0], f[1]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_GATED_QUANT: {
            const long long rows = i[0], w = i[1];
            if (rows < 1 || rows > 1048576 || w < 32 || w > 32768 || w % 32 != 0)
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, rows * w * 4, false);
            slots.maybe(1, rows * w * 4, false);
            slots.maybe(2, rows * (long long)q8_row_bytes(w), true);
            slots.maybe(3, rows * w * 4, true);
            if (!slots.valid() || (p[2] == NULL && p[3] == NULL))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            gated_quant_kernel<<<(unsigned)rows, 256, 0, s>>>(
                (const float *)p[0], (const float *)p[1], (unsigned char *)p[2], (float *)p[3],
                (int)w);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_MOE_ROUTE: {
            const long long n = i[0], h = i[1], e = i[2], k = i[3];
            if (n < 1 || n > 65536 || h < 128 || h > 8192 || h % 128 != 0 || e < 1 || e > 256 ||
                k < 1 || k > 8 || k > e || (i[4] != 0 && i[4] != 1))
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, n * h * 4, false);
            slots.use(1, e * h * 4, false);
            slots.use(2, e * 4, false);
            slots.use(3, n * k * 4, true);
            slots.use(4, n * k * 4, true);
            if (!slots.valid() || !aligned(p[1], 16)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            moe_route_kernel<<<(unsigned)n, 256, (size_t)h * 4, s>>>(
                (const float *)p[0], (const float *)p[1], (const float *)p[2], (int *)p[3],
                (float *)p[4], (int)h, (int)e, (int)k, (int)i[4], f[0]);
            return launch_status(out_stage);
        }
        case INFERENCE_CUDA_BLOCK_OP_FILL: {
            const long long count = i[0];
            if (count < 1 || count > (1LL << 30)) return INFERENCE_CUDA_INVALID_ARGUMENT;
            slots.use(0, count * 4, true);
            if (!slots.valid()) return INFERENCE_CUDA_INVALID_ARGUMENT;
            fill_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>((float *)p[0], (int)count,
                                                                       f[0]);
            return launch_status(out_stage);
        }
        default:
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
