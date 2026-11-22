#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "abi.h"
#include "device.h"
#include "resident/correct_draft_logits_bf16.h"

namespace {

using Args = inference_cuda_correct_draft_logits_bf16_args;
constexpr size_t kMaxLogits = 1u << 30;
constexpr size_t kLutBytes = (1u << 16) * sizeof(uint16_t);
constexpr unsigned kThreads = 256;

bool aligned(const void *ptr, size_t alignment) {
    return ptr != nullptr && uintptr_t(ptr) % alignment == 0;
}

bool apart(const void *a, size_t ab, const void *b, size_t bb) {
    const uintptr_t x = uintptr_t(a), y = uintptr_t(b);
    return x <= y ? ab <= y - x : bb <= x - y;
}

__global__ void correct_draft_logits(Args a) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= a.len) return;

    const float base = __bfloat162float(__float2bfloat16_rn(a.base[i]));
    const float correction = __bfloat162float(__float2bfloat16_rn(a.correction[i]));
    const __nv_bfloat16 merged = __float2bfloat16_rn(__fadd_rn(base, correction));
    // Nonfinite operands cannot produce a finite sum. Check after the BF16
    // conversion as a finite FP32 result can round to BF16 infinity.
    float capped = __uint_as_float(0x7fc00000u);
    if (isfinite(__bfloat162float(merged))) {
        const uint16_t bits = a.softcap_lut[__bfloat16_as_ushort(merged)];
        const float value = __bfloat162float(__ushort_as_bfloat16(bits));
        if (isfinite(value)) capped = value;
    }
    a.base[i] = capped;
}

bool valid(const Args &a) {
    if (a.len == 0 || a.len > kMaxLogits || !aligned(a.base, alignof(float)) ||
        !aligned(a.correction, alignof(float)) || !aligned(a.softcap_lut, alignof(uint16_t))) {
        return false;
    }
    const size_t bytes = a.len * sizeof(float);
    return apart(a.base, bytes, a.correction, bytes) &&
           apart(a.base, bytes, a.softcap_lut, kLutBytes);
}

}  // namespace

extern "C" int inference_cuda_correct_draft_logits_bf16_launch(
    const Args *args, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (args == nullptr || stream == nullptr || !valid(*args)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        const unsigned grid = unsigned((args->len + kThreads - 1) / kThreads);
        Failure failure;
        correct_draft_logits<<<grid, kThreads, 0, (cudaStream_t)stream>>>(*args);
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return int(failure.err);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
