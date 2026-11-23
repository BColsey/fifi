/*
 * The in-house GGUF decode matvec for the codebook and 2-bit types (see matvec_gguf.h and
 * gguf_decode.cuh). No vendored includes: the weights are the file's own blocks, the
 * activation is the shared quantize_q8_1 output (36 bytes per 32 values: fp16 d, fp16 s,
 * 32 int8 quants; s is unused here), the dot products are __dp4a over int8 and the scales
 * apply in f32.
 *
 * Mapping. One warp owns one weight row; lane i of step t owns the 32-value window
 * 32 * t + i of the row, so a warp covers 1024 weights per step and every lane decodes
 * and dots one window. The CTA is eight warps and persistent: it walks rows with a stride
 * of the whole grid, staging once at the start
 *   - the type's lookup table (if any), copied from the by-value kernel argument to shared
 *     memory, each thread moving a few 16-byte chunks, and
 *   - the activation windows of every column, re-laid out so one lane reads its window with
 *     two 16-byte shared loads and a 4-byte one without bank conflicts: the 32 int8 quants
 *     at a 48-byte stride and the f32 scale in its own array.
 * When the staged activation would pass the 48 KiB static shared limit the kernel reads the
 * q8_1 blocks from global memory instead (same arithmetic).
 *
 * The step loop is unrolled by two with both windows' global loads issued before either is
 * decoded, so a warp keeps two windows of weights in flight.
 *
 * The exact f32 entries (inference_cuda_matvec_gguf_f32_launch and its expert twin) run the
 * same row walk over the same weights with an unquantized activation: every product is the
 * decoded weight of window_value times the f32 activation in one fused multiply-add. A
 * kernel's activation policy (Q8Act or F32Act) supplies the rows, so the mapping, the expert
 * id contract and the gate fusion each exist once.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "abi.h"
#include "device.h"
#include "resident/gguf_decode.cuh"
#include "resident/matvec_gguf.h"
#include "resident/step_params.h"  // INFERENCE_CUDA_STATUS_INVALID_STEP_STATE

namespace {

using namespace gguf;

const int kWarps = 8;
const int kThreads = kWarps * 32;
const int kXBlock = 36;       /* bytes per q8_1 block of 32 values */
const int kXQuantStride = 48; /* shared bytes per window of 32 int8 quants */
const int kXWindowSmem = kXQuantStride + 4; /* quants plus the f32 scale */
const int kMaxSmem = 48 * 1024;
const int kMaxCols = 8;
const int kMaxUsed = 32;
const int kCtasPerSm = 4;

/* Shared bytes of one staged activation column of nwin windows (nwin rounded up to 4 so every
 * array stays 16-byte aligned). */
__host__ __device__ inline int staged_windows(int nwin) { return (nwin + 3) & ~3; }

/* Copies one activation column of q8_1 blocks to the shared layout: the f32 scales first
 * (nwp of them), then the 48-byte-stride quant windows. Consecutive threads copy consecutive
 * words of the row, so the global loads coalesce. */
__device__ __forceinline__ void stage_x_column(const unsigned char *x, int nwin, int nwp,
                                               unsigned char *sm) {
    const unsigned *src = reinterpret_cast<const unsigned *>(x);
    float *scales = reinterpret_cast<float *>(sm);
    unsigned char *quants = sm + 4 * nwp;
    for (int i = (int)threadIdx.x; i < nwin * 9; i += kThreads) {
        const int win = i / 9;
        const int word = i - win * 9;
        const unsigned v = src[i];
        if (word == 0) {
            scales[win] = __half2float(__ushort_as_half((unsigned short)(v & 0xFFFFu)));
        } else {
            *reinterpret_cast<unsigned *>(quants + win * kXQuantStride + 4 * (word - 1)) = v;
        }
    }
}

/* Where a lane finds the activation window of column c. */
template <bool Staged>
struct XRef {
    const unsigned char *base;
    size_t col_stride; /* bytes between columns */
    int nwp;

    __device__ __forceinline__ void load(int c, int win, int (&q)[8], float &d) const {
        const unsigned char *col = base + (size_t)c * col_stride;
        if (Staged) {
            d = reinterpret_cast<const float *>(col)[win];
            const int4 *p = reinterpret_cast<const int4 *>(col + 4 * nwp + win * kXQuantStride);
            const int4 a = p[0];
            const int4 b = p[1];
            q[0] = a.x;
            q[1] = a.y;
            q[2] = a.z;
            q[3] = a.w;
            q[4] = b.x;
            q[5] = b.y;
            q[6] = b.z;
            q[7] = b.w;
        } else {
            const unsigned char *blk = col + (size_t)win * kXBlock;
            d = rd_half(blk);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q[i] = *reinterpret_cast<const int *>(blk + 4 + 4 * i);
            }
        }
    }
};

/* The f32 activation rows: column c at base + c * col_floats, packed windows of 32 floats. */
struct XRefF32 {
    const float *base;
    size_t col_floats; /* floats between columns, a multiple of 4 */

    __device__ __forceinline__ const float *col(int c) const {
        return base + (size_t)c * col_floats;
    }
};

/* How a kernel reaches its activation. Both policies give the windows per row (nwin) and
 * row j; stage copies a row into its shared area (a no-op for f32), ref is what the dot
 * products read. */
template <bool Staged>
struct Q8Act {
    const unsigned char *x;
    size_t x_row_bytes;
    int nwin;
    int nwp;

    /* q8_1 rows of k values, x_row_bytes apart. */
    static Q8Act of(const unsigned char *x, size_t x_row_bytes, int k) {
        return Q8Act{x, x_row_bytes, k / 32, staged_windows(k / 32)};
    }
    __device__ __forceinline__ const unsigned char *row(int j) const {
        return x + (size_t)j * x_row_bytes;
    }
    __device__ __forceinline__ void stage(unsigned char *sm, int j, const unsigned char *r) const {
        if constexpr (Staged) {
            stage_x_column(r, nwin, nwp, sm + (size_t)j * (size_t)nwp * kXWindowSmem);
        }
    }
    /* The staged rows (column j at sm + j * stride) or the global row r. */
    __device__ __forceinline__ XRef<Staged> ref(const unsigned char *sm,
                                                const unsigned char *r) const {
        return XRef<Staged>{Staged ? sm : r,
                            Staged ? (size_t)nwp * kXWindowSmem : x_row_bytes, nwp};
    }
};

struct F32Act {
    const float *x;
    int nwin;

    /* Packed f32 rows of k values. */
    static F32Act of(const float *x, int k) { return F32Act{x, k / 32}; }
    __device__ __forceinline__ const float *row(int j) const {
        return x + (size_t)j * (size_t)nwin * 32;
    }
    __device__ __forceinline__ void stage(unsigned char *, int, const float *) const {}
    __device__ __forceinline__ XRefF32 ref(const unsigned char *, const float *r) const {
        return XRefF32{r, (size_t)nwin * 32};
    }
};

/* Adds one decoded window's dot products against the first ncols columns into acc. */
template <class T, bool Staged>
__device__ __forceinline__ void accumulate(const typename T::Raw &raw, int win,
                                           const XRef<Staged> &xr, int ncols,
                                           const uint32_t *tab, float (&acc)[kMaxCols]) {
    Win wn;
    T::decode(raw, tab, wn);
#pragma unroll
    for (int c = 0; c < kMaxCols; ++c) {
        if (c < ncols) {
            int q[8];
            float d;
            xr.load(c, win, q, d);
            int s0 = 0;
            int s1 = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                s0 = __dp4a(wn.w[i], q[i], s0);
                s1 = __dp4a(wn.w[4 + i], q[4 + i], s1);
            }
            float t = fmaf(wn.s0, (float)s0, __fmul_rn(wn.s1, (float)s1));
            if constexpr (T::kHasAdd || T::kHasMin) {
                /* The shifted and offset forms: sums of the activation quants per group of
                 * eight (a dp4a against ones), then s_h * a_g * sum_g and m_h * sum_h. */
                int xs[4];
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    xs[g] = __dp4a(0x01010101, q[2 * g + 1], __dp4a(0x01010101, q[2 * g], 0));
                }
                if constexpr (T::kHasAdd) {
                    const float h0 = fmaf(wn.a[0], (float)xs[0], __fmul_rn(wn.a[1], (float)xs[1]));
                    const float h1 = fmaf(wn.a[2], (float)xs[2], __fmul_rn(wn.a[3], (float)xs[3]));
                    t = fmaf(wn.s0, h0, fmaf(wn.s1, h1, t));
                }
                if constexpr (T::kHasMin) {
                    t -= fmaf(wn.m0, (float)(xs[0] + xs[1]), __fmul_rn(wn.m1, (float)(xs[2] + xs[3])));
                }
            }
            acc[c] = fmaf(d, t, acc[c]);
        }
    }
}

/* The exact form for f32 activations: one fused multiply-add per value, the weight as
 * window_value gives it. A window's values go in four chunks of eight, so a lane issues its
 * activation loads two 16-byte words at a time. */
template <class T>
__device__ __forceinline__ void accumulate(const typename T::Raw &raw, int win, const XRefF32 &xr,
                                           int ncols, const uint32_t *tab,
                                           float (&acc)[kMaxCols]) {
    Win wn;
    T::decode(raw, tab, wn);
    const float *xv[kMaxCols];
#pragma unroll
    for (int c = 0; c < kMaxCols; ++c) {
        if (c < ncols) {
            xv[c] = xr.col(c) + (size_t)win * 32;
        }
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        float4 a[kMaxCols] = {};
        float4 b[kMaxCols] = {};
#pragma unroll
        for (int c = 0; c < kMaxCols; ++c) {
            if (c < ncols) {
                const float4 *p = reinterpret_cast<const float4 *>(xv[c] + 8 * h);
                a[c] = __ldg(p);
                b[c] = __ldg(p + 1);
            }
        }
        const float w[8] = {window_value<T>(wn, 8 * h),     window_value<T>(wn, 8 * h + 1),
                            window_value<T>(wn, 8 * h + 2), window_value<T>(wn, 8 * h + 3),
                            window_value<T>(wn, 8 * h + 4), window_value<T>(wn, 8 * h + 5),
                            window_value<T>(wn, 8 * h + 6), window_value<T>(wn, 8 * h + 7)};
#pragma unroll
        for (int c = 0; c < kMaxCols; ++c) {
            if (c < ncols) {
                float t = acc[c];
                t = fmaf(w[0], a[c].x, t);
                t = fmaf(w[1], a[c].y, t);
                t = fmaf(w[2], a[c].z, t);
                t = fmaf(w[3], a[c].w, t);
                t = fmaf(w[4], b[c].x, t);
                t = fmaf(w[5], b[c].y, t);
                t = fmaf(w[6], b[c].z, t);
                t = fmaf(w[7], b[c].w, t);
                acc[c] = t;
            }
        }
    }
}

/* One lane's partial dot products of a row over its windows; the activation reference picks
 * the q8_1 or the exact f32 accumulate. */
template <class T, class XR>
__device__ __forceinline__ void row_dot(const unsigned char *wrow, const XR &xr,
                                        int nwin, int ncols, const uint32_t *tab, int lane,
                                        float (&acc)[kMaxCols]) {
    for (int base = 0; base < nwin; base += 64) {
        const int wa = base + lane;
        const int wb = wa + 32;
        const bool va = wa < nwin;
        const bool vb = wb < nwin;
        typename T::Raw ra{};
        typename T::Raw rb{};
        if (va) {
            ra = T::load(wrow + (size_t)(wa / T::kWinPerBlock) * T::kBlockBytes,
                         wa % T::kWinPerBlock);
        }
        if (vb) {
            rb = T::load(wrow + (size_t)(wb / T::kWinPerBlock) * T::kBlockBytes,
                         wb % T::kWinPerBlock);
        }
        if (va) {
            accumulate<T>(ra, wa, xr, ncols, tab, acc);
        }
        if (vb) {
            accumulate<T>(rb, wb, xr, ncols, tab, acc);
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

template <class T, class Act>
__global__ void __launch_bounds__(kThreads)
    matvec_dense(const unsigned char *w, float *y, int rows, int ncols, Act act,
                 Tab<T::kTabWords> tab) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t *stab = reinterpret_cast<uint32_t *>(smem);
    unsigned char *sx = smem + 4 * T::kTabWords;
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    stage_table<T::kTabWords, kThreads>(tab, stab);
    for (int c = 0; c < ncols; ++c) {
        act.stage(sx, c, act.row(c));
    }
    __syncthreads();
    const auto xr = act.ref(sx, act.row(0));
    const size_t row_bytes = (size_t)(act.nwin / T::kWinPerBlock) * T::kBlockBytes;
    for (int row = (int)blockIdx.x * kWarps + warp; row < rows; row += (int)gridDim.x * kWarps) {
        float acc[kMaxCols];
#pragma unroll
        for (int c = 0; c < kMaxCols; ++c) {
            acc[c] = 0.0f;
        }
        row_dot<T>(w + (size_t)row * row_bytes, xr, act.nwin, ncols, stab, lane, acc);
#pragma unroll
        for (int c = 0; c < kMaxCols; ++c) {
            if (c < ncols) {
                const float sum = warp_sum(acc[c]);
                if (lane == 0) {
                    y[(size_t)c * rows + row] = sum;
                }
            }
        }
    }
}

/* Grid (row groups, slots): block row group g of slot s walks the rows g * 8 + warp, with a
 * stride of gridDim.x * 8, of expert ids[s]'s matrix. */
template <class T, class Act>
__global__ void __launch_bounds__(kThreads)
    matvec_expert(const unsigned char *w_stack, const int *ids, float *y,
                  const unsigned char *gate_stack, int rows, int n_expert, int x_per_slot,
                  int y_slot_stride, Act act, Tab<T::kTabWords> tab) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int slot = (int)blockIdx.y;
    const int id = ids[slot];
    float *yslot = y + (size_t)slot * y_slot_stride;
    if (id < 0 || id >= n_expert) {
        for (int r = (int)(blockIdx.x * kThreads + threadIdx.x); r < rows;
             r += (int)(gridDim.x * kThreads)) {
            yslot[r] = __int_as_float(0x7fc00000);
        }
        return;
    }
    uint32_t *stab = reinterpret_cast<uint32_t *>(smem);
    unsigned char *sx = smem + 4 * T::kTabWords;
    const int warp = (int)(threadIdx.x >> 5);
    const int lane = (int)(threadIdx.x & 31);
    const int xrow = x_per_slot ? slot : 0;
    stage_table<T::kTabWords, kThreads>(tab, stab);
    act.stage(sx, 0, act.row(xrow));
    __syncthreads();
    const auto xr = act.ref(sx, act.row(xrow));
    const size_t row_bytes = (size_t)(act.nwin / T::kWinPerBlock) * T::kBlockBytes;
    const size_t expert_off = (size_t)id * (size_t)rows * row_bytes;
    for (int row = (int)blockIdx.x * kWarps + warp; row < rows; row += (int)gridDim.x * kWarps) {
        float acc[kMaxCols];
#pragma unroll
        for (int c = 0; c < kMaxCols; ++c) {
            acc[c] = 0.0f;
        }
        row_dot<T>(w_stack + expert_off + (size_t)row * row_bytes, xr, act.nwin, 1, stab, lane,
                   acc);
        float out = warp_sum(acc[0]);
        if (gate_stack != NULL) {
            float gate[kMaxCols];
#pragma unroll
            for (int c = 0; c < kMaxCols; ++c) {
                gate[c] = 0.0f;
            }
            row_dot<T>(gate_stack + expert_off + (size_t)row * row_bytes, xr, act.nwin, 1, stab,
                       lane, gate);
            const float g = warp_sum(gate[0]);
            out *= g / (1.0f + expf(-g));
        }
        if (lane == 0) {
            yslot[row] = out;
        }
    }
}

/* Decodes whole blocks through the window decoders: thread t decodes window t. */
template <class T>
__global__ void __launch_bounds__(kThreads)
    dequant(const unsigned char *w, float *y, size_t n_windows, Tab<T::kTabWords> tab) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t *stab = reinterpret_cast<uint32_t *>(smem);
    stage_table<T::kTabWords, kThreads>(tab, stab);
    __syncthreads();
    for (size_t win = (size_t)blockIdx.x * kThreads + threadIdx.x; win < n_windows;
         win += (size_t)gridDim.x * kThreads) {
        const typename T::Raw raw =
            T::load(w + (win / T::kWinPerBlock) * T::kBlockBytes, (int)(win % T::kWinPerBlock));
        Win wn;
        T::decode(raw, stab, wn);
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            y[win * 32 + j] = window_value<T>(wn, j);
        }
    }
}

/* The embedding row gather: block t gathers row ids[t] of the native table into x[t][0..k), one
 * window per thread step. An id outside [0, n_rows) writes a row of NaN, reads no weight and ORs
 * the invalid-step-state bit into the status word from one thread (the contract of embed.h). */
const int kEmbedThreads = 128;

template <class T>
__global__ void __launch_bounds__(kEmbedThreads)
    embed_rows(const unsigned char *w, const int *ids, float *x, int *status, int n_rows, int k,
               Tab<T::kTabWords> tab) {
    extern __shared__ __align__(16) unsigned char smem[];
    uint32_t *stab = reinterpret_cast<uint32_t *>(smem);
    stage_table<T::kTabWords, kEmbedThreads>(tab, stab);
    __syncthreads();
    const int id = ids[blockIdx.x];
    float *row_out = x + (size_t)blockIdx.x * (size_t)k;
    if (id < 0 || id >= n_rows) {
        for (int j = (int)threadIdx.x; j < k; j += kEmbedThreads) {
            row_out[j] = __int_as_float(0x7fc00000);
        }
        if (threadIdx.x == 0) {
            atomicOr(status, INFERENCE_CUDA_STATUS_INVALID_STEP_STATE);
        }
        return;
    }
    const size_t row_bytes = (size_t)(k / (32 * T::kWinPerBlock)) * T::kBlockBytes;
    const unsigned char *row = w + (size_t)id * row_bytes;
    for (int win = (int)threadIdx.x; win < k / 32; win += kEmbedThreads) {
        const typename T::Raw raw =
            T::load(row + (size_t)(win / T::kWinPerBlock) * T::kBlockBytes, win % T::kWinPerBlock);
        Win wn;
        T::decode(raw, stab, wn);
        float4 *out4 = reinterpret_cast<float4 *>(row_out + (size_t)win * 32);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out4[j] = make_float4(window_value<T>(wn, 4 * j), window_value<T>(wn, 4 * j + 1),
                                  window_value<T>(wn, 4 * j + 2), window_value<T>(wn, 4 * j + 3));
        }
    }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

int sm_count() {
    static int cached[16] = {0};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) {
        return 1;
    }
    if (cached[device] == 0) {
        int count = 0;
        if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) !=
                cudaSuccess ||
            count < 1) {
            return 1;
        }
        cached[device] = count;
    }
    return cached[device];
}

/* Host bytes of the table of a type: Q2_0 has none. */
template <class T>
constexpr size_t host_table_bytes() {
    return (size_t)T::kHostTabWords * 4;
}

template <class T>
bool load_table(Tab<T::kTabWords> *out, const void *table, size_t table_bytes) {
    if (table_bytes != host_table_bytes<T>() || (table_bytes != 0 && table == NULL)) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    if (table_bytes != 0) {
        memcpy(out, table, table_bytes);
    }
    return true;
}

/* The tag a type switch hands to its callback. */
template <class T>
struct TypeTag {
    using Type = T;
};

template <class F>
int with_type(int type, F &&f) {
    switch (type) {
    case kIq2Xxs: return f(TypeTag<Iq2Xxs>());
    case kIq2Xs: return f(TypeTag<Iq2Xs>());
    case kIq3Xxs: return f(TypeTag<Iq3Xxs>());
    case kIq4Nl: return f(TypeTag<Iq4Nl>());
    case kIq3S: return f(TypeTag<Iq3S>());
    case kIq2S: return f(TypeTag<Iq2S>());
    case kIq4Xs: return f(TypeTag<Iq4Xs>());
    case kQ2_0: return f(TypeTag<Q2_0>());
    case kQ2K: return f(TypeTag<Q2K>());
    case kIq1S: return f(TypeTag<Iq1S>());
    case kIq1M: return f(TypeTag<Iq1M>());
    case kQ4_0: return f(TypeTag<Q4_0>());
    case kQ5_0: return f(TypeTag<Q5_0>());
    case kQ8_0: return f(TypeTag<Q8_0>());
    case kQ3K: return f(TypeTag<Q3K>());
    case kQ4K: return f(TypeTag<Q4K>());
    case kQ5K: return f(TypeTag<Q5K>());
    case kQ6K: return f(TypeTag<Q6K>());
    default: return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
}

int finish_launch(int *out_stage, size_t argument_bytes) {
    Failure failure;
    const cudaError_t err = cudaGetLastError();
    note_argument_failure(err, argument_bytes);
    failure.record(err, INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_matvec_gguf_launch(const inference_cuda_matvec_gguf_args *args,
                                                 const void *table, size_t table_bytes,
                                                 size_t x_row_bytes, void *stream,
                                                 int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0 || args->w == NULL ||
            args->x_q8 == NULL || args->y == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || rows <= 0 || ncols < 1 || ncols > kMaxCols) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (!aligned(args->w, 4) || !aligned(args->x_q8, 4) || !aligned(args->y, 4) ||
            x_row_bytes % 4 != 0 || x_row_bytes < (size_t)(k / 32) * kXBlock) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(args->type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            if (k % (32 * T::kWinPerBlock) != 0) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const size_t w_bytes =
                (size_t)rows * (size_t)(k / (32 * T::kWinPerBlock)) * T::kBlockBytes;
            const size_t x_bytes = x_row_bytes * (size_t)ncols;
            const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
            if (!apart(args->w, w_bytes, args->y, y_bytes) ||
                !apart(args->x_q8, x_bytes, args->y, y_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const int nwin = k / 32;
            const size_t staged = (size_t)ncols * staged_windows(nwin) * kXWindowSmem;
            const bool use_smem = 4 * (size_t)T::kTabWords + staged <= (size_t)kMaxSmem;
            const size_t smem = 4 * (size_t)T::kTabWords + (use_smem ? staged : 0);
            const int groups = (rows + kWarps - 1) / kWarps;
            const int blocks = groups < sm_count() * kCtasPerSm ? groups : sm_count() * kCtasPerSm;
            const unsigned char *w = (const unsigned char *)args->w;
            const unsigned char *x = (const unsigned char *)args->x_q8;
            if (use_smem) {
                matvec_dense<T><<<blocks, kThreads, smem, (cudaStream_t)stream>>>(
                    w, args->y, rows, ncols, Q8Act<true>::of(x, x_row_bytes, k), tab);
            } else {
                matvec_dense<T><<<blocks, kThreads, smem, (cudaStream_t)stream>>>(
                    w, args->y, rows, ncols, Q8Act<false>::of(x, x_row_bytes, k), tab);
            }
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_matvec_gguf_expert_launch(
    const inference_cuda_matvec_gguf_expert_args *args, const void *table, size_t table_bytes,
    size_t x_row_bytes, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || x_row_bytes == 0 || args->w_stack == NULL ||
            args->ids == NULL || args->x_q8 == NULL || args->y == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int n_expert = args->n_expert;
        const int n_used = args->n_used;
        const int per_slot = args->x_per_slot;
        if (k <= 0 || rows < 1 || n_expert < 1 || n_used < 1 || n_used > kMaxUsed ||
            n_used > n_expert || (per_slot != 0 && per_slot != 1) || args->reserved != 0 ||
            args->y_slot_stride < rows || !aligned(args->w_stack, 4) || !aligned(args->ids, 4) ||
            !aligned(args->x_q8, 4) || !aligned(args->y, 4) || x_row_bytes % 4 != 0 ||
            x_row_bytes < (size_t)(k / 32) * kXBlock) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (args->glu_gate != NULL && (per_slot != 0 || !aligned(args->glu_gate, 4))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(args->type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            if (k % (32 * T::kWinPerBlock) != 0) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const size_t expert_bytes =
                (size_t)rows * (size_t)(k / (32 * T::kWinPerBlock)) * T::kBlockBytes;
            const size_t w_bytes = expert_bytes * (size_t)n_expert;
            const size_t x_bytes = x_row_bytes * (size_t)(per_slot ? n_used : 1);
            const size_t y_bytes =
                ((size_t)(n_used - 1) * (size_t)args->y_slot_stride + (size_t)rows) *
                sizeof(float);
            if (!apart(args->y, y_bytes, args->w_stack, w_bytes) ||
                !apart(args->y, y_bytes, args->x_q8, x_bytes) ||
                !apart(args->y, y_bytes, args->ids, (size_t)n_used * sizeof(int32_t)) ||
                (args->glu_gate != NULL && !apart(args->y, y_bytes, args->glu_gate, w_bytes))) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const int nwin = k / 32;
            const size_t staged = (size_t)staged_windows(nwin) * kXWindowSmem;
            const bool use_smem = 4 * (size_t)T::kTabWords + staged <= (size_t)kMaxSmem;
            const size_t smem = 4 * (size_t)T::kTabWords + (use_smem ? staged : 0);
            const int groups = (rows + kWarps - 1) / kWarps;
            const int want = (sm_count() * kCtasPerSm + n_used - 1) / n_used;
            const dim3 grid(groups < want ? groups : want, n_used);
            const unsigned char *w = (const unsigned char *)args->w_stack;
            const int *ids = (const int *)args->ids;
            const unsigned char *x = (const unsigned char *)args->x_q8;
            const unsigned char *gate = (const unsigned char *)args->glu_gate;
            if (use_smem) {
                matvec_expert<T><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
                    w, ids, args->y, gate, rows, n_expert, per_slot, args->y_slot_stride,
                    Q8Act<true>::of(x, x_row_bytes, k), tab);
            } else {
                matvec_expert<T><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
                    w, ids, args->y, gate, rows, n_expert, per_slot, args->y_slot_stride,
                    Q8Act<false>::of(x, x_row_bytes, k), tab);
            }
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_dequant_gguf_launch(int type, const void *w, float *y,
                                                  size_t n_blocks, const void *table,
                                                  size_t table_bytes, void *stream,
                                                  int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (w == NULL || y == NULL || stream == NULL || n_blocks == 0 || !aligned(w, 4) ||
            !aligned(y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const size_t n_windows = n_blocks * T::kWinPerBlock;
            const size_t blocks_needed = (n_windows + kThreads - 1) / kThreads;
            const int blocks = (int)(blocks_needed < 1024 ? blocks_needed : 1024);
            dequant<T><<<blocks, kThreads, 4 * (size_t)T::kTabWords, (cudaStream_t)stream>>>(
                (const unsigned char *)w, y, n_windows, tab);
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_embed_gguf_launch(const inference_cuda_embed_gguf_args *args,
                                                const void *table, size_t table_bytes,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->blocks == NULL || args->ids == NULL ||
            args->x == NULL || args->status == NULL || !aligned(args->blocks, 2) ||
            !aligned(args->ids, 4) || !aligned(args->x, 16) || !aligned(args->status, 4) ||
            args->m < 1 || args->m > 4096 || args->n_rows < 1 || args->n_rows > (1 << 28) ||
            args->k < 32 || args->k > 32768) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(args->type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            if (args->k % (32 * T::kWinPerBlock) != 0) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            embed_rows<T><<<args->m, kEmbedThreads, 4 * (size_t)T::kTabWords,
                            (cudaStream_t)stream>>>((const unsigned char *)args->blocks, args->ids,
                                                    args->x, args->status, args->n_rows, args->k,
                                                    tab);
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_matvec_gguf_f32_launch(
    const inference_cuda_matvec_gguf_f32_args *args, const void *table, size_t table_bytes,
    void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int ncols = args->ncols;
        if (k <= 0 || rows <= 0 || ncols < 1 || ncols > kMaxCols || !aligned(args->w, 4) ||
            !aligned(args->x, 16) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(args->type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            if (k % (32 * T::kWinPerBlock) != 0) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const size_t w_bytes =
                (size_t)rows * (size_t)(k / (32 * T::kWinPerBlock)) * T::kBlockBytes;
            const size_t x_bytes = (size_t)k * (size_t)ncols * sizeof(float);
            const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
            if (!apart(args->w, w_bytes, args->y, y_bytes) ||
                !apart(args->x, x_bytes, args->y, y_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const int groups = (rows + kWarps - 1) / kWarps;
            const int blocks = groups < sm_count() * kCtasPerSm ? groups : sm_count() * kCtasPerSm;
            matvec_dense<T><<<blocks, kThreads, 4 * (size_t)T::kTabWords,
                            (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, args->y, rows, ncols, F32Act::of(args->x, k),
                tab);
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_matvec_gguf_f32_expert_launch(
    const inference_cuda_matvec_gguf_f32_expert_args *args, const void *table,
    size_t table_bytes, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w_stack == NULL || args->ids == NULL ||
            args->x == NULL || args->y == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const int n_expert = args->n_expert;
        const int n_used = args->n_used;
        const int per_slot = args->x_per_slot;
        if (k <= 0 || rows < 1 || n_expert < 1 || n_used < 1 || n_used > kMaxUsed ||
            n_used > n_expert || (per_slot != 0 && per_slot != 1) || args->reserved != 0 ||
            args->y_slot_stride < rows || !aligned(args->w_stack, 4) ||
            !aligned(args->ids, 4) || !aligned(args->x, 16) || !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        if (args->glu_gate != NULL && (per_slot != 0 || !aligned(args->glu_gate, 4))) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_type(args->type, [&](auto tag) -> int {
            using T = typename decltype(tag)::Type;
            if (k % (32 * T::kWinPerBlock) != 0) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            Tab<T::kTabWords> tab;
            if (!load_table<T>(&tab, table, table_bytes)) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const size_t expert_bytes =
                (size_t)rows * (size_t)(k / (32 * T::kWinPerBlock)) * T::kBlockBytes;
            const size_t w_bytes = expert_bytes * (size_t)n_expert;
            const size_t x_bytes = (size_t)k * (size_t)(per_slot ? n_used : 1) * sizeof(float);
            const size_t y_bytes =
                ((size_t)(n_used - 1) * (size_t)args->y_slot_stride + (size_t)rows) *
                sizeof(float);
            if (!apart(args->y, y_bytes, args->w_stack, w_bytes) ||
                !apart(args->y, y_bytes, args->x, x_bytes) ||
                !apart(args->y, y_bytes, args->ids, (size_t)n_used * sizeof(int32_t)) ||
                (args->glu_gate != NULL && !apart(args->y, y_bytes, args->glu_gate, w_bytes))) {
                return INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            const int groups = (rows + kWarps - 1) / kWarps;
            const int want = (sm_count() * kCtasPerSm + n_used - 1) / n_used;
            const dim3 grid(groups < want ? groups : want, n_used);
            const unsigned char *w = (const unsigned char *)args->w_stack;
            const int *ids = (const int *)args->ids;
            const unsigned char *gate = (const unsigned char *)args->glu_gate;
            matvec_expert<T><<<grid, kThreads, 4 * (size_t)T::kTabWords,
                              (cudaStream_t)stream>>>(
                w, ids, args->y, gate, rows, n_expert, per_slot, args->y_slot_stride,
                F32Act::of(args->x, k), tab);
            return finish_launch(out_stage, sizeof(tab));
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
