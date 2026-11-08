/*
 * The device facts of the vendored MMQ entries (see mmq_facts.h). The first translation unit of
 * the MMQ glue build group (the vendor group's flags, warnings as errors). The query follows
 * device_info.cu and the MMVQ wrappers' pattern in this tree.
 */
#include <cuda_runtime.h>

#include "abi.h"

// The vendored headers keep static host helpers their own objects reference; in this glue TU they
// are unreferenced, which -Wall reports although nothing here is wrong.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "common.cuh"
#pragma GCC diagnostic pop

#include "prefill/mmq_facts.h"
#include "resident/mmvq_arch.h"

// The live query. The four attributes cannot change while the process holds the device.
static cudaError_t mmq_facts_live(fifi_device_facts *out) {
    int device = 0, major = 0, minor = 0, nsm = 0, smpbo = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, device);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    }
    if (err != cudaSuccess) {
        return err;
    }
    const int built = inference_cuda_mmvq_built_arch(100 * major + 10 * minor);
    if (built == 0) {
        return cudaErrorNoKernelImageForDevice;
    }
    *out = fifi_device_facts{device, built, nsm, (size_t)smpbo};
    return cudaSuccess;
}

// Every entry must use the calling thread's current device. Cache the four immutable
// attributes separately for each device and thread: a process can alternate contexts
// on different GPUs, and independent threads must not race to populate a shared slot.
cudaError_t inference_cuda_mmq_query_facts(fifi_device_facts *out) {
    int device = 0;
    const cudaError_t selected = cudaGetDevice(&device);
    if (selected != cudaSuccess) {
        return selected;
    }
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) {
        // Preserve the live-query behavior for ordinals outside the cache.
        return mmq_facts_live(out);
    }
    static thread_local fifi_device_facts cached[GGML_CUDA_MAX_DEVICES]{};
    if (cached[device].cc == 0) {
        fifi_device_facts facts{};
        const cudaError_t err = mmq_facts_live(&facts);
        if (err != cudaSuccess) {
            return err;
        }
        cached[device] = facts;
    }
    *out = cached[device];
    return cudaSuccess;
}

extern "C" int inference_cuda_mmq_device_facts(inference_cuda_mmq_facts *out) {
    try {
        if (out == NULL) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        *out = inference_cuda_mmq_facts{};
        (void)cudaGetLastError();
        fifi_device_facts facts{};
        const cudaError_t err = mmq_facts_live(&facts);
        (void)cudaGetLastError();
        if (err != cudaSuccess) {
            return (int)err;
        }
        *out = inference_cuda_mmq_facts{facts.device, facts.cc, facts.nsm, 0, facts.smpbo};
        return INFERENCE_CUDA_OK;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
