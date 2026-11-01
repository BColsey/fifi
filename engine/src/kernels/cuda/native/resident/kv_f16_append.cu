/* The standalone fp16 KV append kernel of the multi-device card path: the store arm of the
 * vendored attn_prep (vendor/xavier-qwen/src/ops.cu, attn_prep_k) on its own launch, so the
 * card's gathered full rows reach the same fp16 planes the fused single-device path writes.
 *
 * One thread per value, one block per (kv head, token): grid (2 * n_kv_head, m), head_dim
 * threads. Blocks below n_kv_head store the k rows, blocks above them the v rows. The
 * conversion is __float2half, the plane layout [(pos + t) * n_kv_head + head] * head_dim + d -
 * both bit-identical to the fused kernel's cache stores, which is the parity the card rig
 * asserts. The poison rule (the position word tested in 64 bits before any other memory access,
 * no cache byte on a refusal, mask 4 OR-ed into the status word by one thread) mirrors
 * attn_prep's and kv8_append's.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/kv_f16_append.h"

namespace {

constexpr int kHead = 256;  // the fused kernel's only head size (attn_prep's supported_configuration)
constexpr int kMaxM = 16;   // decode rows (the step's look-ahead; kv8_append's bound)
constexpr int kMaxKvHeads = 64;
constexpr int kMaxCells = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *k;         // [m][n_kv_head][head_dim]
    const float *v;
    __half *k_cache;        // [cells][n_kv_head][head_dim]
    __half *v_cache;
    const int *pos_dev;
    int *status;
    int m, n_kv_head, head_dim, cells;
};

__global__ void kv_f16_append_k(Args a) {
    // Poison guard: block-uniform (while no other stream writes the word) and before the first
    // global read of the rows, exactly as attn_prep_k orders it.
    const long long first = *a.pos_dev;
    if (first < 0 || first + a.m > a.cells) {
        if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int head = blockIdx.x;  // 0..2*n_kv_head-1: the K heads, then the V heads
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const bool is_k = head < a.n_kv_head;
    const int h = is_k ? head : head - a.n_kv_head;
    const float *src = is_k ? a.k : a.v;
    __half *dst = is_k ? a.k_cache : a.v_cache;
    const float value = src[((size_t)t * a.n_kv_head + h) * a.head_dim + d];
    dst[((size_t)((int)first + t) * a.n_kv_head + h) * a.head_dim + d] = __float2half(value);
}

bool aligned_to(const void *p, uintptr_t a) { return ((uintptr_t)p & (a - 1)) == 0; }

bool valid_args(const inference_cuda_kv_f16_append_args *a) {
    return a != NULL && a->k != NULL && a->v != NULL && a->k_cache != NULL &&
           a->v_cache != NULL && a->pos != NULL && a->status != NULL &&
           aligned_to(a->k, 4) && aligned_to(a->v, 4) && aligned_to(a->k_cache, 16) &&
           aligned_to(a->v_cache, 16) && aligned_to(a->pos, 4) && aligned_to(a->status, 4) &&
           a->head_dim == kHead && a->m >= 1 && a->m <= kMaxM && a->n_kv_head >= 1 &&
           a->n_kv_head <= kMaxKvHeads && a->cells >= a->m && a->cells <= kMaxCells;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    kv_f16_append_k<<<dim3(2 * a.n_kv_head, a.m), a.head_dim, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_kv_f16_append_launch(
    const inference_cuda_kv_f16_append_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!valid_args(in)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = {};
        a.k = (const float *)in->k;
        a.v = (const float *)in->v;
        a.k_cache = (__half *)in->k_cache;
        a.v_cache = (__half *)in->v_cache;
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        a.m = in->m;
        a.n_kv_head = in->n_kv_head;
        a.head_dim = in->head_dim;
        a.cells = in->cells;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
