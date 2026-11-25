/*
 * The decode dq matvec: y = W . x over the file's native Q4_K and Q6_K blocks, with x one f32
 * activation row. No vendored includes, no repack and no q8_1 quantize: the weight side is the
 * native block decoded per value, the activation is read as it lies, and the arithmetic is
 * plain f32.
 *
 * Geometry: one warp serves four weight rows, eight lanes per row (a block of 128 threads
 * serves 16 rows). Lane l = lane mod 8 owns values [32 l, 32 l + 32) of every K block, so its
 * 32 x values are contiguous (eight float4 loads, 16-byte aligned), and the four lanes at one
 * position of the four rows ask for the same x values in one warp load: every x value is read
 * once per four weight rows, with no shared memory.
 *
 * The block layouts are the CPU oracle's (weights/gguf_dequant.rs). Every code goes to f32
 * through its mantissa (code_f), which runs on the FMA pipe where __int2float_rn costs a
 * convert.
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
#include "resident/matvec_dq.h"

namespace {

const int kQK = 256;         /* values per K block */
const int kRowsPerWarp = 4;  /* weight rows one warp serves */
const int kLanesPerRow = 8;  /* lanes per weight row: 32 values of a K block each */
const int kWarps = 4;        /* warps per CTA: 128 threads, 16 rows */

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* An integer code 0..2^23-1 as an f32 through its mantissa: 0x4B000000 | code is 2^23 + code
 * and the subtraction is exact. */
__device__ __forceinline__ float code_f(unsigned code) {
    return __int_as_float(0x4B000000u | code) - 8388608.0f;
}

/* The 32 f32 values a lane owns in one K block, at a 16-byte aligned address. */
struct Run {
    float4 v[8];
};

__device__ __forceinline__ Run load_run(const float *x) {
    Run r;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r.v[i] = *reinterpret_cast<const float4 *>(x + 4 * i);
    }
    return r;
}

__device__ __forceinline__ float run_sum(const Run &r) {
    float4 s = r.v[0];
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        s.x += r.v[i].x;
        s.y += r.v[i].y;
        s.z += r.v[i].z;
        s.w += r.v[i].w;
    }
    return (s.x + s.y) + (s.z + s.w);
}

/* ggml's get_scale_min_k4: the 6-bit multiplier and offset of Q4_K sub-block s (0..7) from the
 * 12 packed factor bytes. */
__device__ __forceinline__ void k4_factors(const unsigned char *scales, int s, float *mult,
                                           float *offs) {
    if (s < 4) {
        *mult = code_f(scales[s] & 63u);
        *offs = code_f(scales[s + 4] & 63u);
    } else {
        *mult = code_f((scales[s + 4] & 0x0Fu) | ((scales[s - 4] >> 6) << 4));
        *offs = code_f((scales[s + 4] >> 4) | ((scales[s] >> 6) << 4));
    }
}

/* Q4_K block, 144 bytes: f16 d at +0, f16 dmin at +2, 12 packed factor bytes at +4, 128 code
 * bytes at +16. The 256 values are eight sub-blocks of 32. Sub-block l is one nibble of code
 * unit l / 2 (the low nibble for an even l, the high nibble for an odd one) and has a 6-bit
 * multiplier and offset. Value i of sub-block l weighs d * mult * code - dmin * offs, so a row
 * is the sum over blocks of d * mult * sum(code * x) - dmin * offs * sum(x): the two sums stay
 * apart because d and dmin are per block. */
struct Q4kSide {
    static const bool kOffsetSum = true;
    static const int kBlockBytes = 144;
    float sum_d;
    float sum_m;

    __device__ __forceinline__ void start() {
        sum_d = 0.0f;
        sum_m = 0.0f;
    }

    /* One K block of the lane's row: `raw` is 16-byte aligned (144 = 16 * 9). */
    __device__ __forceinline__ void add(const unsigned char *raw, int l, const Run &x, float sx) {
        const unsigned char *unit = raw + 16 + 32 * (l >> 1);
        const unsigned nib = 4u * (l & 1);
        float sd = 0.0f;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            uint4 packed;
            memcpy(&packed, unit + 16 * i, 16);
            const unsigned codes[4] = {packed.x, packed.y, packed.z, packed.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float4 xv = x.v[4 * i + j];
                sd = fmaf(code_f((codes[j] >> nib) & 0xFu), xv.x, sd);
                sd = fmaf(code_f((codes[j] >> (nib + 8)) & 0xFu), xv.y, sd);
                sd = fmaf(code_f((codes[j] >> (nib + 16)) & 0xFu), xv.z, sd);
                sd = fmaf(code_f((codes[j] >> (nib + 24)) & 0xFu), xv.w, sd);
            }
        }
        float mult;
        float offs;
        k4_factors(raw + 4, l, &mult, &offs);
        sum_d = fmaf(half_at(raw) * mult, sd, sum_d);
        sum_m = fmaf(half_at(raw + 2) * offs, sx, sum_m);
    }

    __device__ __forceinline__ void reduce(int bit) {
        sum_d += __shfl_xor_sync(0xffffffffu, sum_d, bit);
        sum_m += __shfl_xor_sync(0xffffffffu, sum_m, bit);
    }

    __device__ __forceinline__ float result() const { return sum_d - sum_m; }
};

/* A signed byte as an f32 through its mantissa: s + 128 is the byte with its top bit flipped. */
__device__ __forceinline__ float signed_f(unsigned byte) {
    return code_f(byte ^ 0x80u) - 128.0f;
}

/* One Q6_K weight from the value's code bytes and its signed scale. */
__device__ __forceinline__ float q6_weight(unsigned lo, unsigned hi, float scale, unsigned nib,
                                           unsigned pair) {
    const unsigned code = ((lo >> nib) & 0xFu) | (((hi >> pair) & 3u) << 4);
    return scale * (code_f(code) - 32.0f);
}

/* Q6_K block, 210 bytes, 2-byte aligned in the weight file (so every load is 16-bit): ql[128]
 * at +0, qh[64] at +128, 16 signed scales at +192 (one per 16 values), f16 d at +208. The 256
 * values are two halves of 128, each four quarters of 32. Lane l owns quarter q = l % 4 of half
 * h = l / 4 (values [32 l, 32 l + 32)). Value m of that quarter weighs d * scale * (code - 32)
 * with
 *   code = ((ql[64 h + m + 32 (q & 1)] >> 4 (q >> 1)) & 0xF) | ((qh[32 h + m] >> 2 q & 3) << 4)
 *   scale = scales[8 h + 2 q + m / 16]
 * (the CPU oracle, weights/gguf_dequant.rs q6_k). */
struct Q6kSide {
    static const bool kOffsetSum = false;
    static const int kBlockBytes = 210;
    float sum;

    __device__ __forceinline__ void start() { sum = 0.0f; }

    __device__ __forceinline__ void add(const unsigned char *raw, int l, const Run &x, float) {
        const int q = l & 3;
        const int h = l >> 2;
        const unsigned nib = 4u * (unsigned)(q >> 1);
        const unsigned pair = 2u * (unsigned)q;
        const unsigned char *ql = raw + 64 * h + 32 * (q & 1);
        const unsigned char *qh = raw + 128 + 32 * h;
        const unsigned char *sc = raw + 192 + 8 * h + 2 * q;
        const float scales[2] = {signed_f(sc[0]), signed_f(sc[1])};
        float sd = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float w[4];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                /* Values m = 4 i + 2 j and m + 1: the low and the high byte of each pair. */
                unsigned short lo;
                unsigned short hi;
                memcpy(&lo, ql + 4 * i + 2 * j, 2);
                memcpy(&hi, qh + 4 * i + 2 * j, 2);
                const float scale = scales[i >> 2];
                w[2 * j] = q6_weight(lo, hi, scale, nib, pair);
                w[2 * j + 1] = q6_weight(lo >> 8, hi >> 8, scale, nib, pair);
            }
            const float4 xv = x.v[i];
            sd = fmaf(w[0], xv.x, sd);
            sd = fmaf(w[1], xv.y, sd);
            sd = fmaf(w[2], xv.z, sd);
            sd = fmaf(w[3], xv.w, sd);
        }
        sum = fmaf(half_at(raw + 208), sd, sum);
    }

    __device__ __forceinline__ void reduce(int bit) {
        sum += __shfl_xor_sync(0xffffffffu, sum, bit);
    }

    __device__ __forceinline__ float result() const { return sum; }
};

template <class Side>
__global__ void __launch_bounds__(kWarps * 32) matvec_dq_kernel(const unsigned char *w,
                                                                const float *x, float *y, int rows,
                                                                int blocks) {
    const int lane = (int)(threadIdx.x & 31u);
    const int row = (int)blockIdx.x * (kWarps * kRowsPerWarp) +
                    (int)(threadIdx.x >> 5) * kRowsPerWarp + lane / kLanesPerRow;
    const int l = lane % kLanesPerRow;
    /* The last block of a row count below a multiple of 16 has rows that are not there: their
     * lanes read x and join the reduction but read no weight and write nothing. */
    const bool live = row < rows;
    const unsigned char *wrow = live ? w + (size_t)row * blocks * Side::kBlockBytes : w;
    Side side;
    side.start();
    for (int b = 0; b < blocks; ++b) {
        const Run xv = load_run(x + (size_t)b * kQK + 32 * l);
        const float sx = Side::kOffsetSum ? run_sum(xv) : 0.0f;
        if (live) {
            side.add(wrow + (size_t)b * Side::kBlockBytes, l, xv, sx);
        }
    }
    /* Eight lanes hold one row: fold the lane index's low three bits. Every lane joins. */
    #pragma unroll
    for (int bit = 1; bit < kLanesPerRow; bit <<= 1) {
        side.reduce(bit);
    }
    if (live && l == 0) {
        y[row] = side.result();
    }
}

bool aligned(const void *p, size_t alignment) { return (uintptr_t)p % alignment == 0; }

bool apart(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    return (uintptr_t)a + a_bytes <= (uintptr_t)b || (uintptr_t)b + b_bytes <= (uintptr_t)a;
}

}  // namespace

extern "C" int inference_cuda_matvec_dq_launch(const inference_cuda_matvec_dq_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int k = args->k;
        const int rows = args->rows;
        const bool q4k = args->type == INFERENCE_CUDA_TYPE_Q4_K;
        if (args->w == NULL || args->x == NULL || args->y == NULL || k <= 0 || k % kQK != 0 ||
            rows <= 0 || !(q4k || args->type == INFERENCE_CUDA_TYPE_Q6_K)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const size_t blocks = (size_t)k / kQK;
        const int block_bytes = q4k ? Q4kSide::kBlockBytes : Q6kSide::kBlockBytes;
        const size_t w_bytes = (size_t)rows * blocks * (size_t)block_bytes;
        const size_t x_bytes = (size_t)k * sizeof(float);
        const size_t y_bytes = (size_t)rows * sizeof(float);
        if (!aligned(args->w, 16) || !aligned(args->x, 16) || !aligned(args->y, 4) ||
            !apart(args->y, y_bytes, args->w, w_bytes) ||
            !apart(args->y, y_bytes, args->x, x_bytes)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const unsigned int grid = (unsigned int)(((size_t)rows + 15) / 16);
        if (q4k) {
            matvec_dq_kernel<Q4kSide><<<grid, kWarps * 32, 0, (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, (const float *)args->x, args->y, rows, (int)blocks);
        } else {
            matvec_dq_kernel<Q6kSide><<<grid, kWarps * 32, 0, (cudaStream_t)stream>>>(
                (const unsigned char *)args->w, (const float *)args->x, args->y, rows, (int)blocks);
        }
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
