/*
 * Device-property queries of the CUDA adapter. Host only: nothing here
 * allocates, copies or launches. Like abi.h, a function returns an int status
 * (INFERENCE_CUDA_OK, a negative adapter status, or a positive cudaError_t) and
 * clears the thread's pending non-sticky CUDA error before it returns.
 */
#ifndef INFERENCE_CUDA_ABI_DEVICE_H
#define INFERENCE_CUDA_ABI_DEVICE_H

#ifdef __cplusplus
extern "C" {
#endif

/* A PCI bus id buffer: "0000:01:00.0" is 12 chars + NUL; 16 is the size
 * cudaDeviceGetPCIBusId documents. */
#define INFERENCE_CUDA_BUS_ID_BYTES 16

/* Writes 1 to *out_integrated when the calling thread's current device shares
 * physical memory with the host (cudaDevAttrIntegrated: a Jetson), else 0.
 * *out_integrated is 0 on failure. */
int inference_cuda_device_integrated(int *out_integrated);

/* Writes the calling thread's current device's multiprocessor count and
 * compute capability (major, minor), and the device's PCI bus id as a
 * NUL-terminated string into out_bus_id, a caller buffer of
 * INFERENCE_CUDA_BUS_ID_BYTES bytes. The int outs are 0 on failure; the
 * string is an empty one. */
int inference_cuda_device_attributes(int *out_sm_count, int *out_cc_major,
                                     int *out_cc_minor, char *out_bus_id);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_DEVICE_H */
