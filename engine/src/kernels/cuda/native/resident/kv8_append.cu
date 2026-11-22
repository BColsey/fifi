/* The KV8 append kernel of the resident decode step: quantizes the post-norm, post-rope k rows
 * and the v rows of one step to the frozen 8-bit KV format and appends them to the code and scale
 * planes.
 *
 * The layout (separate K and V code and scale arrays, cell-major per KV head, one quantizer for
 * decode and prefill) follows Strata's k34-kv8 layout (`laguna_kv8_view.hpp`); the code
 * is original CUDA C written from the shapes of this repository. The poison rule mirrors the
 * one this repository added to the vendored attn_prep.
 *
 * One thread per value, one warp per group of 32: the warp reduces |x| to the group amax, stores
 * the fp16 scale from lane 0 and the int8 code of its own value from every lane. The code is
 * quantized against the explicitly selected stored-half or unrounded-f32 scale (a group whose
 * stored fp16 scale is 0 keeps codes 0). Both policies reconstruct with the stored scale.
 * The kernel is one launch of grid (n_kv_head, rows) and head_dim threads; all geometry is run-time.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/kv8_append.h"
#include "resident/multi_row.h"

namespace {

constexpr int kStoredHalf = 1;
constexpr int kUnroundedF32 = 2;
constexpr int kGroup = 32;        // values per fp16 scale: the frozen group of the format
constexpr int kMaxM = 16;         // decode rows (the step's look-ahead)
constexpr int kMaxChunk = 65535;  // prefill rows (the grid's y extent)
constexpr int kMaxKvHeads = 64;
constexpr int kMinHead = 32;      // one warp quantizes one group
constexpr int kMaxHead = 1024;
constexpr int kMaxCells = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *k;             // [m][n_kv_head][head_dim]
    const float *v;
    signed char *k_codes;       // [n_kv_head][cells][head_dim]
    unsigned short *k_scales;   // [n_kv_head][cells][head_dim / 32]
    signed char *v_codes;
    unsigned short *v_scales;
    const int *pos_dev;         // the decode entry: the device word; NULL for the chunk entry
    int pos0;                   // the chunk entry: the host base position
    int *status;
    int m, n_kv_head, head_dim, cells;
    int code_scale;
    const inference_cuda_kv_row *rows;  // the many-rows entry: row t appends to rows[t]'s planes; else null
    int ring;                   // 0: cell = position; R > 0: cell = position % R (rows: each row's own position word)
};

// NaN-propagating max: `fmaxf` returns the other operand when one is NaN, so a group holding a
// NaN would lose its poison at the first reduction round. A NaN operand `!=` itself and is
// returned as that NaN; otherwise the pair is `fmaxf`. The two selects carry no branch, so the
// reduction stays straight-line between the shuffles. Precondition: every caller passes `fabsf`
// results (never a signed zero); the `fmaxf` fallback is bit-identical to `fmaxf` for every
// non-NaN pair anyway. No extra shuffle round.
__device__ inline float fmax_nan(float a, float b) {
    const float m = fmaxf(a, b);
    const float n = (b != b) ? b : m;
    return (a != a) ? a : n;
}

__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmax_nan(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// scale = amax / 127 rounded to fp16 (half even). An all-zero group stores 0.
// Poison rule: a NaN anywhere among the group's 32 values reaches `warp_max_a` and makes amax
// NaN, so the stored scale is an fp16 NaN. The codes may be anything `quant_code` gives, but
// every dequantized value `code * f32(scale)` of that group is then NaN, so the poison cannot
// vanish from the int8 KV (the fp16 KV path keeps the NaN, and the argmax status mask 2 catches
// it downstream). A group whose values are all ±Inf keeps the old behavior: scale +Inf, and every
// dequantized value is non-finite (0 * Inf = NaN).
__device__ inline unsigned short quant_scale(float d) {
    return __half_as_ushort(__float2half_rn(d));
}

// Both policies require rounded division, not a rounded reciprocal followed by multiplication.
__device__ inline int quant_code(float x, float s) {
    if (s == 0.f) {
        return 0;
    }
    return max(-127, min(127, __float2int_rn(__fdiv_rn(x, s))));
}

// kMany: row t is the single token of its own sequence, so the position word, the status word and
// the four planes come from rows[t] and the token's cell is the position itself (m is the row
// count). kMany = false is the kernel of the single sequence, unchanged.
template <bool kMany = false>
__global__ void kv8_append_k(Args a) {
    const int kvh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    // Poison rule, as in attn_prep: the position of token 0 is read from the device (decode) or
    // validated here (chunk) and tested, in 64 bits, against the cells before any other memory
    // access. The answer is block uniform (while no other stream writes the word), so the early
    // return happens before any barrier. A refused launch writes no code, no scale, and ORs
    // mask 4 into status (one thread, integer atomic). A valid launch touches neither.
    signed char *k_codes = a.k_codes, *v_codes = a.v_codes;
    unsigned short *k_scales = a.k_scales, *v_scales = a.v_scales;
    int *status = a.status;
    long long base;
    if constexpr (kMany) {
        const inference_cuda_kv_row &row = a.rows[t];
        k_codes = (signed char *)row.k_codes;
        k_scales = (unsigned short *)row.k_scales;
        v_codes = (signed char *)row.v_codes;
        v_scales = (unsigned short *)row.v_scales;
        status = (int *)((char *)row.step + INFERENCE_CUDA_STEP_STATUS_OFFSET);
        base = *(const int *)((const char *)row.step + INFERENCE_CUDA_STEP_POS_OFFSET);
    } else {
        base = a.pos_dev ? (long long)*a.pos_dev : (long long)a.pos0;
    }
    if (base < 0 || base + (kMany ? 1 : a.m) > (a.ring > 0 ? (long long)kMaxCells : (long long)a.cells)) {
        if (status && blockIdx.x == 0 && threadIdx.x == 0 && (kMany || blockIdx.y == 0))
            atomicOr(status, kStatusInvalidStepState);
        return;
    }
    const int groups = a.head_dim / kGroup;
    const size_t cell = (size_t)(kMany ? (a.ring > 0 ? base % a.ring : base)
                                       : (a.ring > 0 ? (base + t) % a.ring : base + t));
    const size_t in_at = ((size_t)t * a.n_kv_head + kvh) * a.head_dim + d;
    const size_t code_at = (size_t)kvh * a.cells * a.head_dim + cell * a.head_dim + d;
    const size_t scale_at = (size_t)kvh * a.cells * groups + cell * groups + d / kGroup;
    const float kx = a.k[in_at], vx = a.v[in_at];
    const float kmax = warp_max_a(fabsf(kx));
    const float vmax = warp_max_a(fabsf(vx));
    const float kd = __fdiv_rn(kmax, 127.f);
    const float vd = __fdiv_rn(vmax, 127.f);
    const unsigned short ks = quant_scale(kd);
    const unsigned short vs = quant_scale(vd);
    const float kden = a.code_scale == kStoredHalf ? h2f(ks) : kd;
    const float vden = a.code_scale == kStoredHalf ? h2f(vs) : vd;
    // Stored-half zero forces zero codes even when the unrounded f32 scale is nonzero.
    k_codes[code_at] = (signed char)(ks == 0 ? 0 : quant_code(kx, kden));
    v_codes[code_at] = (signed char)(vs == 0 ? 0 : quant_code(vx, vden));
    if ((d & (kGroup - 1)) == 0) {
        k_scales[scale_at] = ks;
        v_scales[scale_at] = vs;
    }
}

/* The K-only chunk variant of kv8_append_k for the mixed profile ("q8k": K
 * quantized, V kept f16): identical math on the K plane, no V read or write.
 * The frozen both-plane entry above is untouched. */
__global__ void kv8_chunk_k_k(Args a) {
    const int kvh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const long long base = (long long)a.pos0;
    if (base < 0 || base + a.m > (a.ring > 0 ? (long long)kMaxCells : (long long)a.cells)) {
        if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int groups = a.head_dim / kGroup;
    const size_t cell = (size_t)(a.ring > 0 ? (base + t) % a.ring : base + t);
    const size_t in_at = ((size_t)t * a.n_kv_head + kvh) * a.head_dim + d;
    const size_t code_at = (size_t)kvh * a.cells * a.head_dim + cell * a.head_dim + d;
    const size_t scale_at = (size_t)kvh * a.cells * groups + cell * groups + d / kGroup;
    const float kx = a.k[in_at];
    const float kmax = warp_max_a(fabsf(kx));
    const float kd = __fdiv_rn(kmax, 127.f);
    const unsigned short ks = quant_scale(kd);
    const float kden = a.code_scale == kStoredHalf ? h2f(ks) : kd;
    a.k_codes[code_at] = (signed char)(ks == 0 ? 0 : quant_code(kx, kden));
    if ((d & (kGroup - 1)) == 0) {
        a.k_scales[scale_at] = ks;
    }
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

// The six data pointers every entry takes: f32 rows in, one code array and one scale array out,
// per K and V, each 4-byte aligned.
bool valid_data(const void *k, const void *v, const void *k_codes, const void *k_scales,
                const void *v_codes, const void *v_scales) {
    return k != NULL && v != NULL && k_codes != NULL && k_scales != NULL && v_codes != NULL &&
           v_scales != NULL && aligned_to(k, 4) && aligned_to(v, 4) && aligned_to(k_codes, 4) &&
           aligned_to(k_scales, 4) && aligned_to(v_codes, 4) && aligned_to(v_scales, 4);
}

// rows tokens, n_kv_head kv heads, head_dim values per head (a multiple of the group), cells
// plane cells per head with room for the rows.
bool valid_geometry(int rows, int n_kv_head, int head_dim, int cells) {
    return rows >= 1 && n_kv_head >= 1 && n_kv_head <= kMaxKvHeads && head_dim >= kMinHead &&
           head_dim <= kMaxHead && head_dim % kGroup == 0 && cells >= rows && cells <= kMaxCells;
}

// ring 0 is the linear plane; R > 0 needs R <= cells and every row of one launch in its own cell.
bool valid_ring(int ring, int rows, int cells) {
    return ring == 0 || (ring >= rows && ring <= cells);
}

bool valid_code_scale(int code_scale) {
    return code_scale == kStoredHalf || code_scale == kUnroundedF32;
}

Args make_args(const void *k, const void *v, void *k_codes, void *k_scales, void *v_codes,
               void *v_scales, int rows, int n_kv_head, int head_dim, int cells, int code_scale,
               int ring) {
    Args a = {};
    a.k = (const float *)k;
    a.v = (const float *)v;
    a.k_codes = (signed char *)k_codes;
    a.k_scales = (unsigned short *)k_scales;
    a.v_codes = (signed char *)v_codes;
    a.v_scales = (unsigned short *)v_scales;
    a.m = rows;
    a.n_kv_head = n_kv_head;
    a.head_dim = head_dim;
    a.cells = cells;
    a.code_scale = code_scale;
    a.ring = ring;
    return a;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    kv8_append_k<><<<dim3(a.n_kv_head, a.m), a.head_dim, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

Failure launch_many(const Args &a, cudaStream_t s) {
    Failure failure;
    kv8_append_k<true><<<dim3(a.n_kv_head, a.m), a.head_dim, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

Failure launch_k(const Args &a, cudaStream_t s) {
    Failure failure;
    kv8_chunk_k_k<<<dim3(a.n_kv_head, a.m), a.head_dim, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_kv8_append_launch(const inference_cuda_kv8_append_args *in,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers = in != NULL && valid_data(in->k, in->v, in->k_codes, in->k_scales,
                                                       in->v_codes, in->v_scales) &&
                              in->pos != NULL && in->status != NULL && aligned_to(in->pos, 4) &&
                              aligned_to(in->status, 4);
        if (!pointers || !valid_geometry(in->m, in->n_kv_head, in->head_dim, in->cells) ||
            in->m > kMaxM || !valid_code_scale(in->code_scale) ||
            !valid_ring(in->ring, in->m, in->cells)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->v, in->k_codes, in->k_scales, in->v_codes, in->v_scales,
                           in->m, in->n_kv_head, in->head_dim, in->cells, in->code_scale, in->ring);
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_kv8_append_multi_launch(const inference_cuda_kv8_append_multi_args *in,
                                                      void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (in == NULL || stream == NULL || in->rows == NULL || !aligned_to(in->rows, 8) ||
            in->k == NULL || in->v == NULL || !aligned_to(in->k, 4) || !aligned_to(in->v, 4) ||
            in->n < 1 || in->n > INFERENCE_CUDA_MULTI_ROW_MAX ||
            !valid_geometry(1, in->n_kv_head, in->head_dim, in->cells) ||
            !valid_code_scale(in->code_scale) || !valid_ring(in->ring, 1, in->cells)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->v, NULL, NULL, NULL, NULL, in->n, in->n_kv_head,
                           in->head_dim, in->cells, in->code_scale, in->ring);
        a.rows = in->rows;
        const Failure failure = launch_many(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_kv8_chunk_launch(const inference_cuda_kv8_chunk_args *in,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers =
            in != NULL && valid_data(in->k, in->v, in->k_codes, in->k_scales, in->v_codes,
                                     in->v_scales);
        const bool position = in == NULL || (in->pos0 >= 0 && (long long)in->pos0 + in->t <= (in->ring > 0 ? (long long)kMaxCells : (long long)in->cells));
        if (!pointers || !position ||
            !valid_geometry(in->t, in->n_kv_head, in->head_dim, in->cells) ||
            in->t > kMaxChunk || !valid_code_scale(in->code_scale) || in->reserved != 0 ||
            !valid_ring(in->ring, in->t, in->cells)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->v, in->k_codes, in->k_scales, in->v_codes, in->v_scales,
                           in->t, in->n_kv_head, in->head_dim, in->cells, in->code_scale, in->ring);
        a.pos0 = in->pos0;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* The K-only chunk entry of the mixed profile: the frozen both-plane args
 * struct, but v, v_codes and v_scales must all be NULL (V stays f16). */
extern "C" int inference_cuda_kv8_chunk_k_launch(const inference_cuda_kv8_chunk_args *in,
                                                 void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool k_only = in != NULL && in->v == NULL && in->v_codes == NULL &&
                            in->v_scales == NULL;
        const bool pointers = k_only && in->k != NULL && in->k_codes != NULL &&
                              in->k_scales != NULL && aligned_to(in->k, 4) &&
                              aligned_to(in->k_codes, 4) && aligned_to(in->k_scales, 4);
        const bool position = in == NULL || (in->pos0 >= 0 && (long long)in->pos0 + in->t <= (in->ring > 0 ? (long long)kMaxCells : (long long)in->cells));
        if (!pointers || !position ||
            !valid_geometry(in->t, in->n_kv_head, in->head_dim, in->cells) ||
            in->t > kMaxChunk || !valid_code_scale(in->code_scale) || in->reserved != 0 ||
            !valid_ring(in->ring, in->t, in->cells)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, NULL, in->k_codes, in->k_scales, NULL, NULL,
                           in->t, in->n_kv_head, in->head_dim, in->cells, in->code_scale, in->ring);
        a.pos0 = in->pos0;
        const Failure failure = launch_k(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
