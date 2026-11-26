// Rows of 32-bit words zeroed or copied in one launch. The contract is resident/word_rows.h.
#include <stdint.h>

#include "resident/word_rows.h"

#include "abi.h"
#include "device.h"

namespace {
constexpr int kThreads = 128;
constexpr int kMaxRows = INFERENCE_CUDA_WORD_ROWS_MAX;

// The row lists of one launch, passed by value so they live in the kernel's parameter space;
// only the row pointers refer to device memory.
struct WordRows {
    const uint32_t *source[kMaxRows];
    uint32_t *destination[kMaxRows];
    int words[kMaxRows];
};

template <bool Copy>
__global__ void __launch_bounds__(kThreads) word_rows(WordRows rows) {
    const int r = blockIdx.x;
    const uint32_t *source = rows.source[r];
    uint32_t *destination = rows.destination[r];
    for (int i = threadIdx.x; i < rows.words[r]; i += kThreads) {
        destination[i] = Copy ? source[i] : 0u;
    }
}

bool aligned(const void *pointer) {
    return pointer && reinterpret_cast<uintptr_t>(pointer) % sizeof(uint32_t) == 0;
}

template <bool Copy>
int launch(const void *const *source, void *const *destination, const int *words, int count,
           void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!destination || !words || !stream || count < 1 || count > kMaxRows ||
        (Copy && !source)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    WordRows rows{};
    for (int r = 0; r < count; ++r) {
        if (!aligned(destination[r]) || (Copy && !aligned(source[r])) || words[r] < 1) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        rows.source[r] = Copy ? static_cast<const uint32_t *>(source[r]) : nullptr;
        rows.destination[r] = static_cast<uint32_t *>(destination[r]);
        rows.words[r] = words[r];
    }
    word_rows<Copy><<<count, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(rows);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
}  // namespace

extern "C" int inference_cuda_zero_word_rows(void *const *rows, const int *words, int count,
                                             void *stream, int *out_stage) {
    try {
        return launch<false>(nullptr, rows, words, count, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_copy_word_rows(const void *const *source, void *const *destination,
                                             const int *words, int count, void *stream,
                                             int *out_stage) {
    try {
        return launch<true>(source, destination, words, count, stream, out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
