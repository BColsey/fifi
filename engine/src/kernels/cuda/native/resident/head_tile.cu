#include <cuda_runtime.h>
#include <stdint.h>
#include "abi.h"
#include "device.h"
#include "resident/head_tile.h"

namespace {
bool apart(const void *a, size_t bytes, const void *b) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? bytes <= y - x : bytes <= x - y;
}
__global__ void head_tile_kernel(uint32_t *dst, const uint32_t *src, int keys, int reps,
                                 int dim, int total) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= total) return;
    const int h = i / dim;
    const int tiled = (h % reps) * keys + h / reps;
    dst[tiled * dim + i % dim] = src[i];
}
}

extern "C" int inference_cuda_head_tile_launch(const inference_cuda_head_tile_args *args,
                                               void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == NULL || stream == NULL || args->src == NULL || args->dst == NULL ||
            args->key_heads <= 0 || args->value_heads <= 0 || args->head_dim <= 0 ||
            args->value_heads % args->key_heads != 0 ||
            (uintptr_t)args->src % 4 != 0 || (uintptr_t)args->dst % 4 != 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        const int64_t total = (int64_t)args->value_heads * args->head_dim;
        if (total > (1 << 24) || !apart(args->src, (size_t)total * 4, args->dst)) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        constexpr int block = 256;
        head_tile_kernel<<<(unsigned)((total + block - 1) / block), block, 0,
                           (cudaStream_t)stream>>>((uint32_t *)args->dst,
                                                   (const uint32_t *)args->src,
                                                   args->key_heads,
                                                   args->value_heads / args->key_heads,
                                                   args->head_dim, (int)total);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
