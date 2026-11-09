#include <cuda_runtime.h>

#include "abi.h"
#include "abi_device.h"

extern "C" int inference_cuda_device_integrated(int *out_integrated) {
    try {
        if (out_integrated == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_integrated = 0;
        (void)cudaGetLastError();
        int device = 0;
        int value = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(&value, cudaDevAttrIntegrated, device);
        }
        (void)cudaGetLastError();
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out_integrated = value != 0 ? 1 : 0;
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

/* One-thread store of a host-known i32, stream-ordered, never blocking:
 * the KV8 tile loop's causal-length words ride this instead of a
 * blocking write_at, so the host stops draining the stream between
 * attention tiles (deep prompts paid ~hundreds of syncs per chunk). */
__global__ void store_i32_k(int *dst, int value) { *dst = value; }

extern "C" int inference_cuda_store_i32(void *stream, int *dst, int value) {
    try {
        if (dst == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        store_i32_k<<<1, 1, 0, (cudaStream_t)stream>>>(dst, value);
        const cudaError_t err = cudaGetLastError();
        return err == cudaSuccess ? INFERENCE_CUDA_OK : (int)err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
