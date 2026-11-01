#include "flint_qk.h"
#include "status.h"
#include "../abi.h"
#include "../device.h"

#include <cuda_runtime.h>
#include <initializer_list>
#include <stdint.h>

namespace {
struct Region {
    const void *pointer;
    size_t available;
    size_t needed;
    bool written;
};

bool valid_regions(std::initializer_list<Region> regions) {
    for (auto a = regions.begin(); a != regions.end(); ++a) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(a->pointer);
        if (!a->pointer || address % 4 || a->available < a->needed ||
            a->needed > UINTPTR_MAX - address) return false;
        for (auto b = regions.begin(); b != a; ++b) {
            const uintptr_t other = reinterpret_cast<uintptr_t>(b->pointer);
            const bool apart = address + a->needed <= other || other + b->needed <= address;
            if ((a->written || b->written) && !apart) return false;
        }
    }
    return true;
}

bool valid(const inference_cuda_flint_qk_args *a) {
    if (!a || a->rows < 1 || a->rows > 4096 || a->reserved != 0 ||
        a->position > UINT32_MAX - (a->rows - 1)) return false;
    const size_t q_bytes = a->rows * 32 * 128 * sizeof(float);
    const size_t k_bytes = a->rows * 4 * 128 * sizeof(float);
    return valid_regions({
        {a->q, a->q_bytes, q_bytes, false},
        {a->k, a->k_bytes, k_bytes, false},
        {a->gamma_q, a->gamma_q_bytes, 128 * sizeof(float), false},
        {a->gamma_k, a->gamma_k_bytes, 128 * sizeof(float), false},
        {a->out_q, a->out_q_bytes, q_bytes, true},
        {a->out_k, a->out_k_bytes, k_bytes, true},
        {a->flag, a->flag_bytes, sizeof(unsigned), true}});
}

__device__ float publication_input(double value, unsigned *flag) {
    const float narrowed = __double2float_rn(value);
    // RN-even BF16 would overflow at this midpoint; NaN/Inf also exceed it.
    if ((__float_as_uint(narrowed) & 0x7fffffffu) >= 0x7f7f8000u) {
        atomicOr(flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
        return 0.0f;
    }
    return narrowed;
}

__global__ void norm_k(inference_cuda_flint_qk_args a) {
    const size_t head = blockIdx.x % 36;
    const size_t row = blockIdx.x / 36;
    const bool query = head < 32;
    const size_t offset = (row * (query ? 32 : 4) + (query ? head : head - 32)) * 128;
    const float *x = (query ? a.q : a.k) + offset;
    const float *gamma = query ? a.gamma_q : a.gamma_k;
    float *out = (query ? a.out_q : a.out_k) + offset;
    const unsigned d = threadIdx.x;
    const float value = x[d];
    const float weight = gamma[d];
    __shared__ double input[128];
    __shared__ unsigned finite[128];
    __shared__ double inverse;
    input[d] = double(value);
    finite[d] = isfinite(value) && isfinite(weight);
    __syncthreads();
    if (d == 0) {
        double squares = 0.0;
        bool good = true;
        // Keep the independent scalar oracle's dimension order.
        for (unsigned i = 0; i < 128; ++i) {
            good = good && finite[i];
            squares = __dadd_rn(squares, __dmul_rn(input[i], input[i]));
        }
        finite[0] = good;
        inverse = good ? __ddiv_rn(1.0, __dsqrt_rn(
            __dadd_rn(__ddiv_rn(squares, 128.0), double(1e-6f)))) : 0.0;
        if (!good) atomicOr(a.flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
    }
    __syncthreads();
    out[d] = finite[0] ? publication_input(
        __dmul_rn(__dmul_rn(input[d], inverse), double(weight)), a.flag) : 0.0f;
}

/* The same arithmetic as norm_k with one thread per (row, head). norm_k keeps the oracle's
 * dimension order with one thread summing 128 squares in a block, and on a GeForce part a warp
 * instruction of F64 occupies the pipe for 16 cycles whether one lane or 32 use it, so a tall
 * piece spends most of its Q/K time in those one-lane chains. Here the 32 lanes of a warp run 32
 * vectors' chains side by side, each in the same order, so every output is the value norm_k
 * writes, bit for bit, and the flag latches under the same conditions. */
__global__ void norm_rows_k(inference_cuda_flint_qk_args a) {
    const size_t vector = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector >= a.rows * 36) return;
    const size_t head = vector % 36;
    const size_t row = vector / 36;
    const bool query = head < 32;
    const size_t offset = (row * (query ? 32 : 4) + (query ? head : head - 32)) * 128;
    const float *x = (query ? a.q : a.k) + offset;
    const float *gamma = query ? a.gamma_q : a.gamma_k;
    float *out = (query ? a.out_q : a.out_k) + offset;
    double squares = 0.0;
    bool good = true;
    for (unsigned i = 0; i < 128; ++i) {
        const float value = x[i];
        good = good && isfinite(value) && isfinite(gamma[i]);
        const double widened = double(value);
        squares = __dadd_rn(squares, __dmul_rn(widened, widened));
    }
    if (!good) {
        atomicOr(a.flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
        for (unsigned d = 0; d < 128; ++d) out[d] = 0.0f;
        return;
    }
    const double inverse = __ddiv_rn(1.0, __dsqrt_rn(
        __dadd_rn(__ddiv_rn(squares, 128.0), double(1e-6f))));
    for (unsigned d = 0; d < 128; ++d) {
        out[d] = publication_input(
            __dmul_rn(__dmul_rn(double(x[d]), inverse), double(gamma[d])), a.flag);
    }
}

__global__ void rope_k(inference_cuda_flint_qk_args a) {
    const size_t head = blockIdx.x % 36;
    const size_t row = blockIdx.x / 36;
    const bool query = head < 32;
    const size_t offset = (row * (query ? 32 : 4) + (query ? head : head - 32)) * 128;
    const float *x = (query ? a.q : a.k) + offset;
    float *out = (query ? a.out_q : a.out_k) + offset;
    const unsigned pair = threadIdx.x;
    const double left = double(x[pair]);
    const double right = double(x[pair + 64]);
    if (!isfinite(left) || !isfinite(right)) {
        out[pair] = out[pair + 64] = 0.0f;
        atomicOr(a.flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
        return;
    }
    const double position = double(a.position) + double(row);
    const double angle = __dmul_rn(position, pow(10000.0, -double(pair) / 64.0));
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    out[pair] = publication_input(
        __dsub_rn(__dmul_rn(left, cosine), __dmul_rn(right, sine)), a.flag);
    out[pair + 64] = publication_input(
        __dadd_rn(__dmul_rn(left, sine), __dmul_rn(right, cosine)), a.flag);
}

/* The same arithmetic as rope_k with the angle shared by the heads of a row: one block per row,
 * a thread per rotation pair, the 36 heads in a loop. rope_k recomputes pow and sincos in F64 for
 * every head of a row, which on a GeForce part (F64 at 1/64 of F32) costs milliseconds for the
 * thousand rows of a tall piece. Each output element is the value rope_k writes, bit for bit. */
__global__ void rope_rows_k(inference_cuda_flint_qk_args a) {
    const size_t row = blockIdx.x;
    const unsigned pair = threadIdx.x;
    const double position = double(a.position) + double(row);
    const double angle = __dmul_rn(position, pow(10000.0, -double(pair) / 64.0));
    double sine, cosine;
    sincos(angle, &sine, &cosine);
    for (unsigned head = 0; head < 36; ++head) {
        const bool query = head < 32;
        const size_t offset = (row * (query ? 32 : 4) + (query ? head : head - 32)) * 128;
        const float *x = (query ? a.q : a.k) + offset;
        float *out = (query ? a.out_q : a.out_k) + offset;
        const double left = double(x[pair]);
        const double right = double(x[pair + 64]);
        if (!isfinite(left) || !isfinite(right)) {
            out[pair] = out[pair + 64] = 0.0f;
            atomicOr(a.flag, unsigned(INFERENCE_CUDA_PREFILL_STATUS_BOUND_VIOLATION));
            continue;
        }
        out[pair] = publication_input(
            __dsub_rn(__dmul_rn(left, cosine), __dmul_rn(right, sine)), a.flag);
        out[pair + 64] = publication_input(
            __dadd_rn(__dmul_rn(left, sine), __dmul_rn(right, cosine)), a.flag);
    }
}

int launch_status(int *out_stage) {
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return int(failure.err);
}
} // namespace

extern "C" int inference_cuda_flint_qk_norm(const inference_cuda_flint_qk_args *a,
                                           void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        if (a->rows > 8) {
            norm_rows_k<<<unsigned((a->rows * 36 + 127) / 128), 128, 0, (cudaStream_t)stream>>>(*a);
        } else {
            norm_k<<<unsigned(a->rows * 36), 128, 0, (cudaStream_t)stream>>>(*a);
        }
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_flint_qk_rope(const inference_cuda_flint_qk_args *a,
                                           void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!stream || !valid(a)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        // Eight rows or fewer are a decode or narrow-chunk step: 36 blocks per row spread the work
        // over the device. A tall piece shares each row's angle across its heads instead.
        if (a->rows > 8) {
            rope_rows_k<<<unsigned(a->rows), 64, 0, (cudaStream_t)stream>>>(*a);
        } else {
            rope_k<<<unsigned(a->rows * 36), 64, 0, (cudaStream_t)stream>>>(*a);
        }
        return launch_status(out_stage);
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
