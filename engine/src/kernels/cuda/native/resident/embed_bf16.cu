/* Entry of embed_bf16 on an explicit stream (contract: resident/embed_bf16.h). */
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/embed_bf16.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaskInvalidState = 4;  // the step status bit of an id outside the table

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

/* One block per token row; the threads stride over the k columns. */
__global__ void embed_bf16_k(const unsigned short *w, const int *ids, float *x, int *status,
                             int n_rows, int k) {
    const int token = blockIdx.x;
    const int id = ids[token];
    float *out = x + (long long)token * k;
    if (id < 0 || id >= n_rows) {
        for (int c = threadIdx.x; c < k; c += blockDim.x) out[c] = nanf("");
        if (threadIdx.x == 0) atomicOr(status, kMaskInvalidState);
        return;
    }
    const unsigned short *in = w + (long long)id * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x)
        out[c] = __uint_as_float(((unsigned)in[c]) << 16);
}

bool valid(const inference_cuda_embed_bf16_args *a) {
    if (a == NULL || a->w == NULL || a->ids == NULL || a->x == NULL || a->status == NULL)
        return false;
    if (a->m < 1 || a->m > 4096 || a->n_rows < 1 || a->n_rows > (1 << 28) || a->k < 8 ||
        a->k > 32768 || a->k % 8 != 0)
        return false;
    if (!aligned(a->w, 4) || !aligned(a->ids, 4) || !aligned(a->x, 16) || !aligned(a->status, 4))
        return false;
    const size_t x_bytes = (size_t)a->m * (size_t)a->k * sizeof(float);
    const size_t w_bytes = (size_t)a->n_rows * (size_t)a->k * 2;
    const size_t ids_bytes = (size_t)a->m * sizeof(int);
    return apart(a->x, x_bytes, a->w, w_bytes) && apart(a->x, x_bytes, a->ids, ids_bytes) &&
           apart(a->x, x_bytes, a->status, 4) && apart(a->status, 4, a->ids, ids_bytes) &&
           apart(a->status, 4, a->w, w_bytes);
}

}  // namespace

extern "C" int inference_cuda_embed_bf16_launch(const inference_cuda_embed_bf16_args *args,
                                                void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (stream == NULL || !valid(args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Failure failure;
        embed_bf16_k<<<(unsigned)args->m, kThreads, 0, (cudaStream_t)stream>>>(
            (const unsigned short *)args->w, args->ids, args->x, args->status, args->n_rows,
            args->k);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
