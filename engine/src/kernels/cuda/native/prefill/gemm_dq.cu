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

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

template <class D>
__global__ void __launch_bounds__(kThreads, 2)
    gemm_dq_kernel(const unsigned char *w, const __half *x, float *y, int k, int rows, int ncols,
                   int w_blocks) {
    extern __shared__ __half smem[]; /* [kBM][kAStride] A then [kBK][kBStride] B */
    __half *a = smem;                /* the decoded weight tile */
    __half *b = smem + kBM * kAStride; /* the activation tile */
    const int tile_m = blockIdx.x * kBM;
    const int tile_n = blockIdx.y * kBN;
    const int tx = threadIdx.x;      /* 0..255 */
    const int out_r = tx / 16;       /* micro-tile row group */
    const int out_c = tx % 16;       /* micro-tile column group */

    float acc[4][4] = {};
    const int chunks = k / 64;
    for (int c = 0; c < chunks; ++c) {
        __syncthreads();
        /* Weight tile: one thread per tile row decodes the row's whole 64-value chunk slice
         * in a single decode_chunk call, straight into shared memory (zero-filled past the
         * weight's tail rows). */
        if (tx < kBM) {
            const int row = tile_m + tx;
            if (row < rows) {
                /* chunk c of block c / 4 of the row. */
                const unsigned char *block =
                    w + ((size_t)row * w_blocks + (c / 4)) * D::kBlockBytes;
                D::decode_chunk(block, c % 4, a + tx * kAStride);
            } else {
                for (int e = 0; e < kBK; ++e) {
                    a[tx * kAStride + e] = __float2half(0.0f);
                }
            }
        } else {
            /* Activation tile: the remaining 192 threads load 4096 halves, row-major by k,
             * written at the padded stride. */
            for (int idx = tx - kBM; idx < kBK * kBN; idx += kThreads - kBM) {
                const int r = idx / kBN;
                const int col = tile_n + idx % kBN;
                const int row_k = c * 64 + r;
                b[r * kBStride + idx % kBN] =
                    (row_k < k && col < ncols) ? x[(size_t)row_k * ncols + col]
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
                y[(size_t)row * ncols + col] = acc[i][j];
            }
        }
    }
}

template <class D>
int launch(const inference_cuda_gemm_dq_args *args, void *stream, int *out_stage) {
    const int k = args->k;
    const int rows = args->rows;
    const int ncols = args->ncols;
    if (k <= 0 || k % D::kQK != 0 || rows < 1 || ncols < 1) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const int w_blocks = k / D::kQK;
    const size_t w_bytes = (size_t)w_blocks * (size_t)rows * D::kBlockBytes;
    const size_t x_bytes = (size_t)k * (size_t)ncols * sizeof(__half);
    const size_t y_bytes = (size_t)rows * (size_t)ncols * sizeof(float);
    if (!apart(args->y, y_bytes, args->w, w_bytes) || !apart(args->y, y_bytes, args->x, x_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const dim3 grid((unsigned)((rows + kBM - 1) / kBM), (unsigned)((ncols + kBN - 1) / kBN));
    const size_t smem = (size_t)(kBM * kAStride + kBK * kBStride) * sizeof(__half);
    gemm_dq_kernel<D><<<grid, kThreads, smem, (cudaStream_t)stream>>>(
        (const unsigned char *)args->w, (const __half *)args->x, args->y, k, rows, ncols,
        w_blocks);
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
            args->y == NULL || !aligned(args->w, 4) || !aligned(args->x, 2) ||
            !aligned(args->y, 4)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        return launch<gemm_dq::Q5kDecoder>(args, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
