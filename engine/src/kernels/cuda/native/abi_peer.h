/*
 * Device selection and peer-access entries of the CUDA adapter.
 *
 * Like abi.h, a function returns an int status (INFERENCE_CUDA_OK, a negative
 * adapter status, or a positive cudaError_t) and clears the thread's pending
 * non-sticky CUDA error before it returns. Nothing here allocates device
 * memory, copies or launches; peer enablement does change the calling
 * thread's device mappings and is why every entry takes explicit ordinals.
 * A negative ordinal is INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call.
 */
#ifndef INFERENCE_CUDA_ABI_PEER_H
#define INFERENCE_CUDA_ABI_PEER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Makes `device` the calling thread's current CUDA device (cudaSetDevice). */
int inference_cuda_set_device(int device);

/* Writes the calling thread's current device ordinal to *out_device
 * (cudaGetDevice). *out_device is 0 on failure. */
int inference_cuda_get_device(int *out_device);

/* Writes the compute capability of `device` to *out_major and *out_minor
 * (cudaDevAttrComputeCapabilityMajor/Minor). Both are 0 on failure. */
int inference_cuda_device_cc(int device, int *out_major, int *out_minor);

/* Writes 1 to *out_can when `device` may directly access memory of `peer`
 * (cudaDeviceCanAccessPeer), else 0. *out_can is 0 on failure. */
int inference_cuda_peer_can_access(int device, int peer, int *out_can);

/* Enables direct access from `device` to `peer` for this process
 * (cudaSetDevice(device), then cudaDeviceEnablePeerAccess(peer, 0)).
 * Enabling an already-enabled pair reports the runtime status
 * (cudaErrorPeerAccessAlreadyEnabled) instead of success, so a repeated
 * enable is visible to the caller, never silently absorbed here. The
 * calling thread's current device is `device` when the call returns. */
int inference_cuda_peer_enable(int device, int peer);

/* Writes the placement facts of `device`: its name into out_name (a caller
 * buffer of name_cap bytes, NUL-terminated), the total and currently free
 * global memory (cudaMemGetInfo under a temporary device selection, with
 * the thread's previous device restored), the SM count, the memory bus
 * width in bits, and the memory clock in Hz (the runtime reports kHz).
 * Zeroes every out value on failure. The free figure is a snapshot at call
 * time; no allocation is made here. */
int inference_cuda_device_profile(
    int device,
    char *out_name,
    int name_cap,
    unsigned long long *out_total,
    unsigned long long *out_free,
    int *out_multiprocessors,
    int *out_bus_width,
    unsigned long long *out_memory_clock_hz);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_PEER_H */
