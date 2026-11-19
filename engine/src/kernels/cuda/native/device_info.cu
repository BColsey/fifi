#include <cuda_runtime.h>

#include "abi.h"
#include "abi_device.h"
#include "device.h"

extern "C" int inference_cuda_device_integrated(int *out_integrated) {
    try {
        if (out_integrated == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_integrated = 0;
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, NULL))
            return static_cast<int>(sticky);
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

extern "C" int inference_cuda_device_attributes(int *out_sm_count,
                                                 int *out_cc_major,
                                                 int *out_cc_minor,
                                                 char *out_bus_id) {
    try {
        if (out_sm_count == NULL || out_cc_major == NULL ||
            out_cc_minor == NULL || out_bus_id == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_sm_count = 0;
        *out_cc_major = 0;
        *out_cc_minor = 0;
        out_bus_id[0] = '\0';
        const cudaError_t sticky = cudaPeekAtLastError();
        if (sticky != cudaSuccess && sticky_prior(sticky, NULL))
            return static_cast<int>(sticky);
        int device = 0;
        cudaError_t err = cudaGetDevice(&device);
        if (err == cudaSuccess)
            err = cudaDeviceGetAttribute(out_sm_count,
                                         cudaDevAttrMultiProcessorCount, device);
        if (err == cudaSuccess)
            err = cudaDeviceGetAttribute(out_cc_major,
                                         cudaDevAttrComputeCapabilityMajor, device);
        if (err == cudaSuccess)
            err = cudaDeviceGetAttribute(out_cc_minor,
                                         cudaDevAttrComputeCapabilityMinor, device);
        if (err == cudaSuccess)
            err = cudaDeviceGetPCIBusId(out_bus_id, INFERENCE_CUDA_BUS_ID_BYTES,
                                        device);
        (void)cudaGetLastError();
        if (err != cudaSuccess) {
            out_bus_id[0] = '\0';
            return (int)err;
        }
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
