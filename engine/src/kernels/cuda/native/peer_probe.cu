#include <cuda_runtime.h>

#include "abi.h"
#include "abi_peer.h"

namespace {

int finish(cudaError_t err) {
    (void)cudaGetLastError();
    return err == cudaSuccess ? INFERENCE_CUDA_OK : (int)err;
}

} // namespace

extern "C" int inference_cuda_set_device(int device) {
    try {
        if (device < 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        return finish(cudaSetDevice(device));
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_get_device(int *out_device) {
    try {
        if (out_device == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_device = 0;
        (void)cudaGetLastError();
        int device = 0;
        const int status = finish(cudaGetDevice(&device));
        if (status == INFERENCE_CUDA_OK) {
            *out_device = device;
        }
        return status;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_device_cc(int device, int *out_major, int *out_minor) {
    try {
        if (device < 0 || out_major == NULL || out_minor == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_major = 0;
        *out_minor = 0;
        (void)cudaGetLastError();
        int major = 0;
        int minor = 0;
        cudaError_t err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        if (err == cudaSuccess) {
            err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        }
        const int status = finish(err);
        if (status == INFERENCE_CUDA_OK) {
            *out_major = major;
            *out_minor = minor;
        }
        return status;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_peer_can_access(int device, int peer, int *out_can) {
    try {
        if (device < 0 || peer < 0 || out_can == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out_can = 0;
        (void)cudaGetLastError();
        int can = 0;
        const int status = finish(cudaDeviceCanAccessPeer(&can, device, peer));
        if (status == INFERENCE_CUDA_OK) {
            *out_can = can != 0 ? 1 : 0;
        }
        return status;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_peer_enable(int device, int peer) {
    try {
        if (device < 0 || peer < 0) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        (void)cudaGetLastError();
        cudaError_t err = cudaSetDevice(device);
        if (err == cudaSuccess) {
            err = cudaDeviceEnablePeerAccess(peer, 0);
        }
        return finish(err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
