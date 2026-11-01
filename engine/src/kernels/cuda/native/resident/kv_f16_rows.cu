/*
 * Entry of kv_f16_rows on an explicit stream. Glue only: the kernel is below; every
 * exported function has C linkage, takes and returns C types only, and wraps its
 * body in try/catch, like row_scale.
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <stddef.h>
#include <stdint.h>
#include <cmath>

#include "abi.h"
#include "device.h"
#include "resident/kv_f16_rows.h"

namespace {

bool aligned(const void *p, size_t a) { return (uintptr_t)p % a == 0; }
bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? ab <= y - x : bb <= x - y;
}

/* The device twin of the host f16_row guard: accept only finite F32 values whose
 * low 16 mantissa bits are zero and that stay finite in F16. Normal-range
 * accepted values are exactly representable; clean subnormal-range values
 * round RNE on both sides, identically. Rejects zero the half-word and set
 * the sticky flag; the caller rejects the token with the host error text. */
__global__ void kv_f16_rows_kernel(inference_cuda_kv_f16_args a) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= a.rows * a.columns) return;
    const float value = a.x[i];
    const unsigned bits = __float_as_uint(value);
    const __half stored = __float2half_rn(value);
    const bool ok = isfinite(value) && (bits & 0xffffu) == 0u &&
                    isfinite(__half2float(stored));
    if (!ok) atomicOr(a.flag, 1u);
    if (a.y != NULL)
        reinterpret_cast<unsigned short *>(a.y)[i] = ok ? __half_as_ushort(stored) : 0;
}

bool valid_kv_f16(const inference_cuda_kv_f16_args *a) {
    if (!a || a->rows < 1 || a->rows > 65536 || a->columns < 1 ||
        a->columns > (1u << 22) || a->columns > SIZE_MAX / a->rows) return false;
    const size_t values = a->rows * a->columns;
    if (values > (1ull << 32) || values > SIZE_MAX / sizeof(float)) return false;
    const size_t in_bytes = values * sizeof(float);
    const size_t out_bytes = values * sizeof(unsigned short);
    const bool guard_only = a->y == NULL;
    const void *ptrs[3] = {a->x, a->y, a->flag};
    const size_t have[3] = {a->x_bytes, a->y_bytes, a->flag_bytes};
    const size_t need[3] = {in_bytes, guard_only ? 0 : out_bytes, sizeof(unsigned)};
    for (unsigned i = 0; i < 3; ++i) {
        if (i == 1 && guard_only) continue;
        if (!ptrs[i] || !aligned(ptrs[i], 4) || have[i] < need[i] ||
            reinterpret_cast<uintptr_t>(ptrs[i]) > UINTPTR_MAX - need[i]) return false;
        for (unsigned j = 0; j < i; ++j) {
            if (j == 1 && guard_only) continue;
            if (!apart(ptrs[i], need[i], ptrs[j], need[j])) return false;
        }
    }
    return true;
}
}  // namespace

extern "C" int inference_cuda_kv_f16_rows(const inference_cuda_kv_f16_args *args,
                                          void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid_kv_f16(args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const unsigned grid =
            unsigned((args->rows * args->columns + 255) / 256);
        kv_f16_rows_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(*args);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
