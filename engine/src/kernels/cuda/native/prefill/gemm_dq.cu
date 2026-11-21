/*
 * The in-house dequantizing prefill GEMM. y = W . x: the weight is the file's own native
 * quant blocks, decoded into shared memory one 64 x 64 tile at a time by the per-type
 * decoder (gemm_dq_decoders.cuh) and multiplied against an f16 activation tile, f32
 * accumulation. No repack, no f16 copy of the weights, no vendored code.
 *
 * Tile: BM = BN = BK = 64, 256 threads (eight warps). Per k-step of 64 values: threads 0..63
 * each decode one weight-tile row — a single decode_chunk call writes that row's 64
 * dequantized values straight into shared memory, no local staging — while threads 64..255
 * load the activation tile; then each warp multiplies its 16 x 32 slice of the output tile on
 * the tensor cores (16 x 16 x 16 half fragments, f32 accumulators) from shared memory. The A
 * and B tiles carry a +8-half row-stride pad, so a fragment load spreads over the banks, and
 * keep every fragment pointer 32-byte aligned. The accumulators go through shared memory to
 * the output so that a warp writes consecutive rows of one column together.
 *
 * The 64-value k-step is the decoder's natural slice for the K-quants (a Q5_K chunk is
 * exactly 64 values), which is why BK is 64 for every type. Tail rows and columns are
 * bounds-checked (zero-filled loads, guarded stores), so any positive rows/ncols works.
 *
 * Every exported function has C linkage, takes and returns C types only, and wraps its body
 * in try/catch, like matvec_q5k.cu.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <climits>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "prefill/gemm_dq.h"
#include <string.h>

#include "prefill/gemm_dq_decoders.cuh"

namespace {

const int kBM = 64;
const int kBN = 64;
const int kBK = 64;
const int kThreads = 256; /* eight warps */
/* Row strides of the shared tiles in halves: a multiple of 8 (the fragment loads need 16-byte
 * rows) that is not a multiple of 64, so the rows of a fragment fall on different banks. */
const int kAStride = kBK + 8;
const int kBStride = kBN + 8;

/* Wide staging of one tile's raw chunk bytes (kChunkBytes per row, the kChunkBlocks blocks
 * the chunk decodes): threads stride the (row, 16-byte piece) space in address order, so
 * adjacent lanes hit adjacent 16-byte words — a warp-cooperative sweep of 16-byte loads
 * when the chunk size is a multiple of 16, else of byte loads over the same layout (one
 * warp's 32 consecutive bytes still land in one sector). Tail rows beyond `rows` stage
 * zeros, which decode to zeroes like the old explicit fill. The quad path needs `w` itself
 * 16-byte aligned; the entry enforces it for the types that take that path. */
template <int kBlockBytes, int kChunkBytes>
__device__ __forceinline__ void stage_blocks(const unsigned char *w, int tile_m, int rows,
                                             int w_blocks, int c, int chunk_block_off,
                                             unsigned char *stage) {
    const size_t chunk_off = (size_t)chunk_block_off * kBlockBytes;
    if constexpr ((kChunkBytes & 15) == 0) {
        static_assert(kChunkBytes % 16 == 0, "uint4 staging needs a 16-byte-multiple chunk");
        constexpr int kQuads = kBM * kChunkBytes / 16;
        const int quads_per_row = kChunkBytes / 16;
        #pragma unroll 4
        for (int idx = threadIdx.x; idx < kQuads; idx += kThreads) {
            const int row = idx / quads_per_row;
            const int q = idx - row * quads_per_row;
            uint4 v = make_uint4(0, 0, 0, 0);
            const int grow = tile_m + row;
            if (grow < rows) {
                v = *reinterpret_cast<const uint4 *>(
                    w + (size_t)grow * w_blocks * kBlockBytes + chunk_off + q * 16);
            }
            *reinterpret_cast<uint4 *>(stage + idx * 16) = v;
        }
    } else {
        #pragma unroll 8
        for (int idx = threadIdx.x; idx < kBM * kChunkBytes; idx += kThreads) {
            const int row = idx / kChunkBytes;
            const int o = idx - row * kChunkBytes;
            unsigned char v = 0;
            const int grow = tile_m + row;
            if (grow < rows) {
                v = w[(size_t)grow * w_blocks * kBlockBytes + chunk_off + o];
            }
            stage[idx] = v;
        }
    }
}

template <class D>
constexpr int stage_bytes_of = kBM * D::kChunkBlocks * D::kBlockBytes;

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

/* The grouped (per-expert) launch: the compact columns of expert e are the entries
 * expert_bounds[e] .. expert_bounds[e + 1] of ids_dst, each the token slot whose activation row
 * and output row the column reads and writes. Slot s reads activation row s / x_div and writes
 * output row s. */
struct GroupedArgs {
    const int *ids_dst;
    const int *bounds;
    size_t expert_stride; /* bytes between the weights of consecutive experts */
    int n_experts;
    int x_div;
};

/* Tile `tile` of the grouped launch: tiles are numbered expert after expert, ceil(n_e / kBN)
 * for expert e. One warp finds the expert and the tile inside it (out[0] = -1 when `tile` is
 * past the last): each lane totals the tiles of its own run of experts, a shuffle scan gives
 * the run starts, and the lane whose run holds `tile` walks it. */
__device__ void find_tile(const int *bounds, int n_experts, int tile, int *out) {
    const int lane = (int)threadIdx.x;
    const int per = (n_experts + 31) / 32;
    const int e0 = min(lane * per, n_experts);
    const int e1 = min(e0 + per, n_experts);
    int count = 0;
    for (int e = e0; e < e1; ++e) {
        count += (bounds[e + 1] - bounds[e] + kBN - 1) / kBN;
    }
    int inclusive = count;
    for (int off = 1; off < 32; off <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inclusive, off);
        if (lane >= off) {
            inclusive += v;
        }
    }
    const int exclusive = inclusive - count;
    const unsigned owner = __ballot_sync(0xffffffffu, tile >= exclusive && tile < inclusive);
    if (owner == 0) {
        if (lane == 0) {
            out[0] = -1;
        }
        return;
    }
    if (lane == __ffs(owner) - 1) {
        int t = tile - exclusive;
        for (int e = e0; e < e1; ++e) {
            const int tiles = (bounds[e + 1] - bounds[e] + kBN - 1) / kBN;
            if (t < tiles) {
                out[0] = e;
                out[1] = t;
                return;
            }
            t -= tiles;
        }
    }
}

/* One 64 x 64 output tile of y = W . x. Dense: tile (blockIdx.x, blockIdx.y) of the rows x
 * ncols product. Grouped: blockIdx.x is the row tile and blockIdx.y a tile number of the
 * grouped launch (find_tile); the weights are that expert's, the columns its compact columns,
 * and the activation and output rows are addressed through the token slots (GroupedArgs). */
template <class D, bool kGrouped>
__global__ void __launch_bounds__(kThreads, 2)
    gemm_dq_kernel(const unsigned char *w, const __half *x, float *y, int k, int rows, int ncols,
                   int w_blocks, int x_col_stride, int y_col_stride, int x_f32,
                   gguf::Tab<D::kTabWords> tab, GroupedArgs grouped) {
    extern __shared__ __align__(128) __half smem[]; /* A [kBM][kAStride] then B [kBK][kBStride] then raw blocks */
    __half *a = smem;                  /* the decoded weight tile */
    __half *b = smem + kBM * kAStride; /* the activation tile */
    unsigned char *stage =
        reinterpret_cast<unsigned char *>(smem + kBM * kAStride + kBK * kBStride);
    /* The type's lookup table (a 4-word dummy for the K-quant decoders), after the raw blocks. */
    uint32_t *stab = reinterpret_cast<uint32_t *>(stage + stage_bytes_of<D>);
    gguf::stage_table<D::kTabWords, kThreads>(tab, stab);
    const int tile_m = blockIdx.x * kBM;
    int tile_n = blockIdx.y * kBN;
    __shared__ int s_xoff[kGrouped ? kBN : 1]; /* grouped: per tile column, its activation row offset or -1 */
    __shared__ int s_yoff[kGrouped ? kBN : 1]; /* grouped: its output row offset */
    __shared__ int s_tile[2];
    if constexpr (kGrouped) {
        if (threadIdx.x < 32) {
            find_tile(grouped.bounds, grouped.n_experts, (int)blockIdx.y, s_tile);
        }
        __syncthreads();
        const int expert = s_tile[0];
        if (expert < 0) {
            return; /* uniform: every thread of the block reads the same word */
        }
        const int lo = grouped.bounds[expert];
        const int count = grouped.bounds[expert + 1] - lo;
        tile_n = s_tile[1] * kBN;
        w += (size_t)expert * grouped.expert_stride;
        if (threadIdx.x < kBN) {
            const int j = tile_n + (int)threadIdx.x;
            if (j < count) {
                const int slot = grouped.ids_dst[lo + j];
                s_xoff[threadIdx.x] = (slot / grouped.x_div) * x_col_stride;
                s_yoff[threadIdx.x] = slot * y_col_stride;
            } else {
                s_xoff[threadIdx.x] = -1;
                s_yoff[threadIdx.x] = 0;
            }
        }
        ncols = count;
    }
    const int tx = threadIdx.x;      /* 0..255 */
    const int warp = tx / 32;
    const int warp_r = warp % 4;     /* rows 16 * warp_r .. +16 of the tile */
    const int warp_c = warp / 4;     /* columns 32 * warp_c .. +32 of the tile */

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[2];
    nvcuda::wmma::fill_fragment(acc[0], 0.0f);
    nvcuda::wmma::fill_fragment(acc[1], 0.0f);
    const int chunks = k / 64;
    for (int c = 0; c < chunks; ++c) {
        __syncthreads();
        /* Weight tile: the chunk's raw blocks first move into shared memory as one
         * warp-cooperative sweep, then one thread per tile row decodes its row's 64-value
         * chunk slice from shared into the f16 tile. */
        stage_blocks<D::kBlockBytes, D::kChunkBlocks * D::kBlockBytes>(
            w, tile_m, rows, w_blocks, c, D::chunk_block_off(c), stage);
        __syncthreads();
        constexpr int kDecodeThreads = kBM * D::kParts;
        if (tx < kDecodeThreads) {
            /* One row of the tile per thread and part: thread tx decodes part tx / 64 of row
             * tx % 64 (a part is a 32-value window for the window decoders). */
            const int row = tx % kBM;
            D::decode_chunk(stage + (size_t)row * D::kChunkBlocks * D::kBlockBytes, c,
                            a + row * kAStride, stab, tx / kBM);
        } else {
            /* Activation tile: the remaining threads load 4096 halves, row-major by k,
             * written at the padded stride. */
            for (int idx = tx - kDecodeThreads; idx < kBK * kBN; idx += kThreads - kDecodeThreads) {
                const int r = idx / kBN;
                const int col = tile_n + idx % kBN;
                const int row_k = c * 64 + r;
                if constexpr (kGrouped) {
                    const int xoff = s_xoff[idx % kBN];
                    b[r * kBStride + idx % kBN] =
                        (row_k < k && xoff >= 0)
                            ? __float2half(reinterpret_cast<const float *>(x)[(size_t)xoff + row_k])
                            : __float2half(0.0f);
                } else {
                    const size_t x_at = x_col_stride ? (size_t)col * x_col_stride + row_k
                                                    : (size_t)row_k * ncols + col;
                    b[r * kBStride + idx % kBN] =
                        (row_k < k && col < ncols)
                            ? (x_f32 ? __float2half(reinterpret_cast<const float *>(x)[x_at])
                                     : x[x_at])
                            : __float2half(0.0f);
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < kBK / 16; ++kk) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half,
                                   nvcuda::wmma::row_major>
                fa;
            nvcuda::wmma::load_matrix_sync(fa, a + warp_r * 16 * kAStride + kk * 16, kAStride);
            #pragma unroll
            for (int f = 0; f < 2; ++f) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half,
                                       nvcuda::wmma::row_major>
                    fb;
                nvcuda::wmma::load_matrix_sync(
                    fb, b + kk * 16 * kBStride + warp_c * 32 + f * 16, kBStride);
                nvcuda::wmma::mma_sync(acc[f], fa, fb, acc[f]);
            }
        }
    }
    /* The accumulators to shared memory, column-major (element [row][col] at col * 64 + row, over
     * the A and B tiles, which no warp reads any more), then out with consecutive threads on
     * consecutive rows of one column. */
    __syncthreads();
    float *tile_out = reinterpret_cast<float *>(smem);
    static_assert(kBM * kBN * sizeof(float) <= (kBM * kAStride + kBK * kBStride) * sizeof(__half),
                  "the output tile reuses the A and B tiles");
    #pragma unroll
    for (int f = 0; f < 2; ++f) {
        nvcuda::wmma::store_matrix_sync(tile_out + (warp_c * 32 + f * 16) * kBM + warp_r * 16,
                                        acc[f], kBM, nvcuda::wmma::mem_col_major);
    }
    __syncthreads();
    for (int idx = tx; idx < kBM * kBN; idx += kThreads) {
        const int col_in = idx / kBM;
        const int row = tile_m + idx % kBM;
        if (row >= rows) {
            continue;
        }
        if constexpr (kGrouped) {
            if (s_xoff[col_in] >= 0) {
                y[(size_t)s_yoff[col_in] + row] = tile_out[idx];
            }
        } else {
            const int col = tile_n + col_in;
            if (col < ncols) {
                y[y_col_stride ? (size_t)col * y_col_stride + row
                               : (size_t)row * ncols + col] = tile_out[idx];
            }
        }
    }
}

template <class D>
int launch(const inference_cuda_gemm_dq_args *args, void *stream, int *out_stage,
           const void *table = NULL, size_t table_bytes = 0) {
    const int k = args->k;
    const int rows = args->rows;
    const int ncols = args->ncols;
    const int x_col_stride = args->x_col_stride;
    const int y_col_stride = args->y_col_stride;
    const int x_f32 = args->x_f32;
    if (k <= 0 || k % D::kQK != 0 || k % 64 != 0 || rows < 1 || ncols < 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if (args->x_col_stride != 0 &&
        (args->x_col_stride < k || (long long)ncols * args->x_col_stride > INT_MAX)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if (args->y_col_stride != 0 &&
        (args->y_col_stride < rows || (long long)ncols * args->y_col_stride > INT_MAX)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const int w_blocks = k / D::kQK;
    const size_t w_bytes = (size_t)w_blocks * (size_t)rows * D::kBlockBytes;
    const size_t x_elems =
        args->x_col_stride ? (size_t)(ncols - 1) * args->x_col_stride + k
                           : (size_t)k * (size_t)ncols;
    if (args->x_f32 != 0 && args->x_f32 != 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t y_elems =
        args->y_col_stride ? (size_t)(ncols - 1) * args->y_col_stride + rows
                           : (size_t)rows * (size_t)ncols;
    const size_t x_unit = args->x_f32 ? sizeof(float) : sizeof(__half);
    if (!apart(args->y, y_elems * sizeof(float), args->w, w_bytes) ||
        !apart(args->y, y_elems * sizeof(float), args->x, x_elems * x_unit)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const dim3 grid((unsigned)((rows + kBM - 1) / kBM), (unsigned)((ncols + kBN - 1) / kBN));
    const size_t smem = (size_t)(kBM * kAStride + kBK * kBStride) * sizeof(__half) +
                        (size_t)stage_bytes_of<D> + 4 * (size_t)D::kTabWords;
    gguf::Tab<D::kTabWords> tab;
    memset(&tab, 0, sizeof(tab));
    if (table_bytes > sizeof(tab) || (table_bytes != 0 && table == NULL)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if (table_bytes != 0) {
        memcpy(&tab, table, table_bytes);
    }
    // The lookup table rides a kernel argument, never a per-TU __constant__ array (see
    // native/constant_bank_guard.rs).
    gemm_dq_kernel<D, false><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
        (const unsigned char *)args->w, (const __half *)args->x, args->y, k, rows, ncols,
        w_blocks, x_col_stride, y_col_stride, x_f32, tab, GroupedArgs{});
    Failure failure;
    const cudaError_t err = cudaGetLastError();
    gguf::note_argument_failure(err, sizeof(tab));
    failure.record(err, INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

/* The tag a type switch hands to its callback. */
template <class D>
struct DecoderTag {
    using Type = D;
};

/* Calls f(DecoderTag<decoder>) for the GGUF type `type`: the K-quant and Q4_0 decoders of
 * gemm_dq_decoders.cuh, or the window decoder of every codebook type, Q2_K and IQ4_XS. */
template <class F>
int with_decoder(int type, F &&f) {
    using namespace gguf;
    using namespace gemm_dq;
    switch (type) {
    case kQ4_0: return f(DecoderTag<Q40Decoder>());
    case kQ3K: return f(DecoderTag<Q3kDecoder>());
    case kQ5K: return f(DecoderTag<Q5kDecoder>());
    case kIq2Xxs: return f(DecoderTag<WindowDecoder<Iq2Xxs>>());
    case kIq2Xs: return f(DecoderTag<WindowDecoder<Iq2Xs>>());
    case kIq2S: return f(DecoderTag<WindowDecoder<Iq2S>>());
    case kIq3Xxs: return f(DecoderTag<WindowDecoder<Iq3Xxs>>());
    case kIq3S: return f(DecoderTag<WindowDecoder<Iq3S>>());
    case kIq4Nl: return f(DecoderTag<WindowDecoder<Iq4Nl>>());
    case kIq4Xs: return f(DecoderTag<WindowDecoder<Iq4Xs>>());
    case kQ2K: return f(DecoderTag<WindowDecoder<Q2K>>());
    case kIq1S: return f(DecoderTag<WindowDecoder<Iq1S>>());
    case kIq1M: return f(DecoderTag<WindowDecoder<Iq1M>>());
    case kQ2_0: return f(DecoderTag<WindowDecoder<Q2_0>>());
    default: return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
}

/* The grouped launch of decoder D: one block per (row tile, tile of the grouped columns). The
 * tile count is bounded by ceil(total_cols / kBN) + n_experts (every expert adds at most one
 * partial tile); blocks past the real count return at once. */
template <class D>
int launch_grouped(const inference_cuda_gemm_dq_mmid_args *args, const void *table,
                   size_t table_bytes, void *stream, int *out_stage) {
    const int k = args->k;
    const int rows = args->rows;
    if (k <= 0 || k % D::kQK != 0 || k % 64 != 0 || rows < 1 || args->n_experts < 1 ||
        args->total_cols < 1 || args->x_div < 1 || args->x_row_stride < k ||
        args->y_row_stride < rows) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if ((long long)args->total_cols * args->y_row_stride > INT_MAX ||
        (long long)args->total_cols * args->x_row_stride > INT_MAX) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const long long tiles = ((long long)args->total_cols + kBN - 1) / kBN + args->n_experts;
    if (tiles > 65535) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    if (table_bytes != 4 * (size_t)D::kHostTabWords || (table_bytes != 0 && table == NULL)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    gguf::Tab<D::kTabWords> tab;
    memset(&tab, 0, sizeof(tab));
    if (table_bytes != 0) {
        memcpy(&tab, table, table_bytes);
    }
    GroupedArgs grouped;
    grouped.ids_dst = args->ids_dst;
    grouped.bounds = args->expert_bounds;
    grouped.expert_stride = (size_t)args->expert_stride;
    grouped.n_experts = args->n_experts;
    grouped.x_div = args->x_div;
    const dim3 grid((unsigned)((rows + kBM - 1) / kBM), (unsigned)tiles);
    const size_t smem = (size_t)(kBM * kAStride + kBK * kBStride) * sizeof(__half) +
                        (size_t)stage_bytes_of<D> + 4 * (size_t)D::kTabWords;
    gemm_dq_kernel<D, true><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
        (const unsigned char *)args->w, (const __half *)args->x, args->y, k, rows, 0,
        k / D::kQK, args->x_row_stride, args->y_row_stride, 1, tab, grouped);
    Failure failure;
    const cudaError_t err = cudaGetLastError();
    gguf::note_argument_failure(err, sizeof(tab));
    failure.record(err, INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return (int)failure.err;
}

}  // namespace

extern "C" int inference_cuda_gemm_dq_q5k_launch(const inference_cuda_gemm_dq_args *args,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || !aligned(args->w, 16) || !aligned(args->x, 2) ||
            !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch<gemm_dq::Q5kDecoder>(args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gemm_dq_q3k_launch(const inference_cuda_gemm_dq_args *args,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || !aligned(args->w, 2) || !aligned(args->x, 2) ||
            !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch<gemm_dq::Q3kDecoder>(args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_gemm_dq_q40_launch(const inference_cuda_gemm_dq_args *args,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || !aligned(args->w, 2) || !aligned(args->x, 2) ||
            !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch<gemm_dq::Q40Decoder>(args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}


/* The decoders of the decode matvec (resident/gguf_decode.cuh) behind the same GEMM, for every
 * codebook type, Q2_K and IQ4_XS. The host passes the lookup table of the type as the exact
 * words of weights/gguf_tables.rs (none for Q2_0 and Q2_K); they ride the launch by value. */
extern "C" int inference_cuda_gemm_dq_codebook_launch(const inference_cuda_gemm_dq_args *args,
                                                      int type, const void *table,
                                                      size_t table_bytes, void *stream,
                                                      int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || !aligned(args->w, 2) || !aligned(args->x, 2) ||
            !aligned(args->y, 4) || type == gguf::kQ4_0 || type == gguf::kQ3K ||
            type == gguf::kQ5K) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_decoder(type, [&](auto tag) {
            using D = typename decltype(tag)::Type;
            if (table_bytes != 4 * (size_t)D::kHostTabWords) {
                return (int)INFERENCE_CUDA_INVALID_ARGUMENT;
            }
            return launch<D>(args, stream, out_stage, table, table_bytes);
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* The grouped (per-expert) launch of the same GEMM for every type the dequantizing GEMM
 * serves. w holds n_experts stacked weights expert_stride bytes apart (each rows x k of the
 * type), x f32 token rows, y f32 slot rows. */
extern "C" int inference_cuda_gemm_dq_mmid_launch(const inference_cuda_gemm_dq_mmid_args *args,
                                                  int type, const void *table,
                                                  size_t table_bytes, void *stream,
                                                  int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        /* Q5_K stages 16-byte words of its 176-byte blocks; every other type reads 2-byte fields. */
        const size_t w_align = type == gguf::kQ5K ? 16 : 2;
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || args->ids_dst == NULL || args->expert_bounds == NULL ||
            !aligned(args->w, w_align) || !aligned(args->x, 4) || !aligned(args->y, 4) ||
            !aligned(args->ids_dst, 4) || !aligned(args->expert_bounds, 4) ||
            args->expert_stride % w_align != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return with_decoder(type, [&](auto tag) {
            using D = typename decltype(tag)::Type;
            return launch_grouped<D>(args, table, table_bytes, stream, out_stage);
        });
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
