/*
 * The in-house dequantizing prefill GEMM. y = W . x: the weight is the file's own native
 * quant blocks, decoded into shared memory one 64 x 64 tile at a time by the per-type
 * decoder (gemm_dq_decoders.cuh) and multiplied against an f16 activation tile, f32
 * accumulation. No repack, no f16 copy of the weights, no vendored code.
 *
 * Tile: BM = BN = BK = 64, 256 threads. Each thread owns one 4 x 4 output micro-tile (the
 * block covers a 16 x 16 grid of them). Per k-step of 64 values: threads 0..63 each decode
 * one weight-tile row — a single decode_chunk call writes that row's 64 dequantized values
 * straight into shared memory, no local staging — while threads 64..255 load the activation
 * tile; then the micro-tiles accumulate from shared memory. The A and B tiles carry a
 * +8-half row-stride pad so the column-major reads of the accumulate loop do not collide
 * on the same shared-memory banks.
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

#include <climits>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "prefill/gemm_dq.h"
#include "prefill/gemm_dq_decoders.cuh"

namespace {

const int kBM = 64;
const int kBN = 64;
const int kBK = 64;
const int kThreads = 256; /* 16 x 16 grid of 4 x 4 micro-tiles */
/* Row strides of the shared tiles, padded to break bank conflicts.
 *
 * A stores: a warp writes a[tx * kAStride + l] for one l, i.e. 4-byte words
 * (tx * kAStride / 2 + l / 2); a stride of 74 halves = 37 words and 37 mod 32 = 5 is
 * coprime with the 32 banks, so the 32 lanes of the warp hit 32 distinct banks (a 72-half
 * stride would be 36 mod 32 = 4 and collide 4-way). A reads broadcast within a row group
 * and 5 * delta-row mod 32 is nonzero for every row distance below 32, so they stay clean.
 * B stores/reads spread along the (unpadded-conflicting) column index; 72 halves = 36
 * words, 36 mod 32 = 4, and the warp's columns differ by 1 half each, so only rows collide
 * -- 8 halves of pad spreads those row groups apart.
 */
const int kAStride = kBK + 10;
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

template <class D>
__global__ void __launch_bounds__(kThreads, 2)
    gemm_dq_kernel(const unsigned char *w, const __half *x, float *y, int k, int rows, int ncols,
                   int w_blocks, int x_col_stride, int y_col_stride, int x_f32,
                   iq4nl_table iq4nl) {
    extern __shared__ __align__(16) __half smem[]; /* A [kBM][kAStride] then B [kBK][kBStride] then raw blocks */
    __half *a = smem;                  /* the decoded weight tile */
    __half *b = smem + kBM * kAStride; /* the activation tile */
    unsigned char *stage =
        reinterpret_cast<unsigned char *>(smem + kBM * kAStride + kBK * kBStride);
    const int tile_m = blockIdx.x * kBM;
    const int tile_n = blockIdx.y * kBN;
    const int tx = threadIdx.x;      /* 0..255 */
    const int out_r = tx / 16;       /* micro-tile row group */
    const int out_c = tx % 16;       /* micro-tile column group */

    float acc[4][4] = {};
    const int chunks = k / 64;
    for (int c = 0; c < chunks; ++c) {
        __syncthreads();
        /* Weight tile: the chunk's raw blocks first move into shared memory as one
         * warp-cooperative sweep, then one thread per tile row decodes its row's 64-value
         * chunk slice from shared into the f16 tile. */
        stage_blocks<D::kBlockBytes, D::kChunkBlocks * D::kBlockBytes>(
            w, tile_m, rows, w_blocks, c, D::chunk_block_off(c), stage);
        __syncthreads();
        if (tx < kBM) {
            D::decode_chunk(stage + (size_t)tx * D::kChunkBlocks * D::kBlockBytes, c,
                            a + tx * kAStride, iq4nl.v);
        } else {
            /* Activation tile: the remaining 192 threads load 4096 halves, row-major by k,
             * written at the padded stride. */
            for (int idx = tx - kBM; idx < kBK * kBN; idx += kThreads - kBM) {
                const int r = idx / kBN;
                const int col = tile_n + idx % kBN;
                const int row_k = c * 64 + r;
                const size_t x_at = x_col_stride ? (size_t)col * x_col_stride + row_k
                                                : (size_t)row_k * ncols + col;
                b[r * kBStride + idx % kBN] =
                    (row_k < k && col < ncols)
                        ? (x_f32 ? __float2half(reinterpret_cast<const float *>(x)[x_at])
                                 : x[x_at])
                        : __float2half(0.0f);
            }
        }
        __syncthreads();
        for (int t = 0; t < kBK; ++t) {
            const float av[4] = {__half2float(a[(out_r * 4 + 0) * kAStride + t]),
                                 __half2float(a[(out_r * 4 + 1) * kAStride + t]),
                                 __half2float(a[(out_r * 4 + 2) * kAStride + t]),
                                 __half2float(a[(out_r * 4 + 3) * kAStride + t])};
            const float bv[4] = {__half2float(b[t * kBStride + out_c * 4 + 0]),
                                 __half2float(b[t * kBStride + out_c * 4 + 1]),
                                 __half2float(b[t * kBStride + out_c * 4 + 2]),
                                 __half2float(b[t * kBStride + out_c * 4 + 3])};
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
                }
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        const int row = tile_m + out_r * 4 + i;
        if (row >= rows) {
            continue;
        }
        for (int j = 0; j < 4; ++j) {
            const int col = tile_n + out_c * 4 + j;
            if (col < ncols) {
                y[y_col_stride ? (size_t)col * y_col_stride + row
                               : (size_t)row * ncols + col] = acc[i][j];
            }
        }
    }
}

template <class D>
int launch(const inference_cuda_gemm_dq_args *args, void *stream, int *out_stage) {
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
    const size_t smem =
        (size_t)(kBM * kAStride + kBK * kBStride) * sizeof(__half) + (size_t)stage_bytes_of<D>;
    // The IQ4_NL table rides a kernel argument, never a per-TU __constant__
    // array: the at-load constant bank of this multi-cubin .so aliases those
    // by position (2026-11-18, 3090). The tree-wide guard that keeps new
    // initialized __constant__ tables out is native/constant_bank_guard.rs.
    gemm_dq_kernel<D><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
        (const unsigned char *)args->w, (const __half *)args->x, args->y, k, rows, ncols,
        w_blocks, x_col_stride, y_col_stride, x_f32, make_iq4nl_table());
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
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

extern "C" int inference_cuda_gemm_dq_iq4xs_launch(const inference_cuda_gemm_dq_args *args,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->w == NULL || args->x == NULL ||
            args->y == NULL || !aligned(args->w, 2) || !aligned(args->x, 2) ||
            !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch<gemm_dq::Iq4xsDecoder>(args, stream, out_stage);
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
