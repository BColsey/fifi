// Depth-residual stream of Inkling 2 on the device: mix, seal and add.
// The contract is resident/inkling_stream.h. Every arithmetic intrinsic is _rn, so no build flag
// can contract a multiply and an add or turn a divide or a square root into an approximation.
#include <math.h>
#include <stdint.h>

#include "resident/inkling_stream.h"

#include "abi.h"
#include "device.h"

namespace {
constexpr int threads = 256;
constexpr int warps = threads / 32;
constexpr int max_blocks = 63;
constexpr int max_width = 8192;
constexpr int max_rows = max_blocks + 1;
// The most states one launch takes: one block per state.
constexpr int max_states = 65535;

// Ordered sum of one warp, lane 0 holds the result.
__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// Sum of the 256 per-thread partials: thread 0 adds them in lane order.
__device__ float block_sum(float value, float *scratch) {
    const int lane = threadIdx.x;
    scratch[lane] = value;
    __syncthreads();
    if (lane == 0) {
        float total = 0.0f;
        for (int i = 0; i < threads; ++i) total = __fadd_rn(total, scratch[i]);
        scratch[0] = total;
    }
    __syncthreads();
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// Rust rms_norm_with_scale: scale = (dot(x, x) / len + eps).sqrt().recip().
__device__ float rms_scale(const float *row, int width, float eps, float *scratch, int &invalid) {
    const int lane = threadIdx.x;
    float squares = 0.0f;
    for (int i = lane; i < width; i += threads) {
        const float value = row[i];
        invalid |= !isfinite(value);
        squares = __fadd_rn(squares, __fmul_rn(value, value));
    }
    const float total = block_sum(squares, scratch);
    __shared__ float shared;
    if (lane == 0) {
        const float mean = __fdiv_rn(total, static_cast<float>(width));
        const float root = __fsqrt_rn(__fadd_rn(mean, eps));
        const float scale = __frcp_rn(root);
        invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
        shared = scale;
    }
    __syncthreads();
    const float scale = shared;
    __syncthreads();
    return scale;
}

// Row r of the mix order: the sealed rows newest first, then the current row.
__device__ const float *mix_row(const float *state, int blocks, int rows, int r, int width) {
    if (r == rows - 1) return state;
    return state + (static_cast<size_t>(blocks) - r) * width;
}

// e^x from IEEE double additions and multiplications only: the host's portable_exp
// (kernels/portable_exp.rs), bit for bit, whatever the host's libm.
__device__ float portable_exp(float x) {
    if (isnan(x)) return x;
    if (!(x > -110.0f)) return 0.0f;
    if (x > 89.0f) return INFINITY;
    // 2^r = sum of (r ln 2)^i / i! for |r| <= 1/2, to the ninth power.
    const long long terms[10] = {0x3ff0000000000000LL, 0x3fe62e42fefa39efLL, 0x3fcebfbdff82c58fLL,
                                 0x3fac6b08d704a0c0LL, 0x3f83b2ab6fba4e77LL, 0x3f55d87fe78a6731LL,
                                 0x3f2430912f86c787LL, 0x3eeffcbfc588b0c7LL, 0x3eb62c0223a5c824LL,
                                 0x3e7b5253d395e7c4LL};
    // Adding and removing 1.5 * 2^52 rounds a double of magnitude below 2^51 to an integer.
    const double shift = 6755399441055744.0;
    const double z = __dmul_rn(static_cast<double>(x), __longlong_as_double(0x3ff71547652b82feLL));
    const double k = __dadd_rn(__dadd_rn(z, shift), -shift);
    const double r = __dadd_rn(z, -k);
    double p = __longlong_as_double(terms[9]);
#pragma unroll
    for (int i = 8; i >= 0; --i) p = __dadd_rn(__dmul_rn(p, r), __longlong_as_double(terms[i]));
    const double scale = __longlong_as_double((static_cast<long long>(k) + 1023) << 52);
    return __double2float_rn(__dmul_rn(p, scale));
}

// Quiet NaN in the health word of a block whose inputs or outputs were not finite.
__device__ void report(float *health, int invalid) {
    if (__syncthreads_or(invalid) != 0 && threadIdx.x == 0) health[0] = __int_as_float(0x7fc00000);
}

// Depth mix in one block: the scale, the scores and the weights are block-wide reductions in a
// fixed order, and one block cannot hide a memory latency behind its threads. Each lane
// therefore issues several loads ahead of its dependent adds (scores 16, mixed channels 9,
// squares 9); a lane still adds the same elements in the same order, so every rounding is that
// of the kernel that loaded one element per add. Scratch timing with one to 63 rows on a
// RTX 2080 Ti: 7.95 to 12.5 us down to 5.5 to 9.6 us for 2 to 8 rows, output bit for bit.
constexpr int kScoreLoads = 16;
constexpr int kChannelLoads = 9;
constexpr int kSquareLoads = 9;

__global__ void inkling_stream_mix(const float *state, const float *query, float *out,
                                   float *health, int first_block, int blocks, int width,
                                   float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    out += static_cast<size_t>(blockIdx.x) * width;
    __shared__ float scratch[threads];
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float shared_scale;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int sub = lane & 31;
    const int rows = blocks - first_block + 1;
    int invalid = 0;
    // Rust rms_norm_with_scale on the current row: scale = (dot(x, x) / len + eps).sqrt().recip().
    {
        float squares = 0.0f;
        int i = lane;
        for (; i + threads * (kSquareLoads - 1) < width; i += threads * kSquareLoads) {
            float value[kSquareLoads];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) value[u] = state[i + threads * u];
            #pragma unroll
            for (int u = 0; u < kSquareLoads; ++u) {
                invalid |= !isfinite(value[u]);
                squares = __fadd_rn(squares, __fmul_rn(value[u], value[u]));
            }
        }
        for (; i < width; i += threads) {
            const float value = state[i];
            invalid |= !isfinite(value);
            squares = __fadd_rn(squares, __fmul_rn(value, value));
        }
        const float total = block_sum(squares, scratch);
        if (lane == 0) {
            const float mean = __fdiv_rn(total, static_cast<float>(width));
            const float root = __fsqrt_rn(__fadd_rn(mean, eps));
            const float scale = __frcp_rn(root);
            invalid |= !isfinite(total) || !isfinite(root) || !isfinite(scale);
            shared_scale = scale;
        }
        __syncthreads();
    }
    const float current_scale = shared_scale;
    __syncthreads();
    // The host scores dot(row, query) * (len as f32).sqrt().recip().
    const float norm = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
    for (int r = warp; r < rows; r += warps) {
        const float *row = mix_row(state, blocks, rows, r, width);
        const float scale = r == rows - 1 ? current_scale : 1.0f;
        float sum = -0.0f;
        int i = sub;
        for (; i + 32 * (kScoreLoads - 1) < width; i += 32 * kScoreLoads) {
            float value[kScoreLoads], q[kScoreLoads];
            #pragma unroll
            for (int u = 0; u < kScoreLoads; ++u) {
                value[u] = row[i + 32 * u];
                q[u] = query[i + 32 * u];
            }
            #pragma unroll
            for (int u = 0; u < kScoreLoads; ++u) {
                invalid |= !isfinite(value[u]) || !isfinite(q[u]);
                sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(value[u], scale), q[u]));
            }
        }
        for (; i < width; i += 32) {
            const float value = row[i];
            const float q = query[i];
            invalid |= !isfinite(value) || !isfinite(q);
            sum = __fadd_rn(sum, __fmul_rn(__fmul_rn(value, scale), q));
        }
        sum = warp_sum(sum);
        if (sub == 0) scores[r] = __fmul_rn(sum, norm);
    }
    __syncthreads();
    if (lane == 0) {
        float peak = -INFINITY;
        for (int r = 0; r < rows; ++r) peak = fmaxf(peak, scores[r]);
        float mass = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float weight = portable_exp(__fsub_rn(scores[r], peak));
            weights[r] = weight;
            mass = __fadd_rn(mass, weight);
        }
        shared_scale = mass;
    }
    __syncthreads();
    const float mass = shared_scale;
    __syncthreads();
    int i = lane;
    for (; i + threads * (kChannelLoads - 1) < width; i += threads * kChannelLoads) {
        float value[kChannelLoads];
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) value[u] = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float *row = mix_row(state, blocks, rows, r, width);
            float x[kChannelLoads];
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) {
                const float raw = row[i + threads * u];
                x[u] = r == rows - 1 ? __fmul_rn(raw, current_scale) : raw;
            }
            #pragma unroll
            for (int u = 0; u < kChannelLoads; ++u) {
                value[u] = __fadd_rn(value[u], __fmul_rn(weights[r], x[u]));
            }
        }
        #pragma unroll
        for (int u = 0; u < kChannelLoads; ++u) {
            value[u] = __fdiv_rn(value[u], mass);
            out[i + threads * u] = value[u];
            invalid |= !isfinite(value[u]);
        }
    }
    for (; i < width; i += threads) {
        float value = 0.0f;
        for (int r = 0; r < rows; ++r) {
            const float x = r == rows - 1
                                ? __fmul_rn(state[i], current_scale)
                                : mix_row(state, blocks, rows, r, width)[i];
            value = __fadd_rn(value, __fmul_rn(weights[r], x));
        }
        value = __fdiv_rn(value, mass);
        out[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

__global__ void inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                    int width, float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    next += static_cast<size_t>(blockIdx.x) * width;
    __shared__ float scratch[threads];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = rms_scale(state, width, eps, scratch, invalid);
    float *sealed = state + (static_cast<size_t>(blocks) + 1) * width;
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(state[i], scale);
        const float replacement = next[i];
        sealed[i] = value;
        state[i] = replacement;
        invalid |= !isfinite(value) || !isfinite(replacement);
    }
    report(health, invalid);
}

__global__ void inkling_stream_add(float *state, const float *row, float *health, int width,
                                   int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    row += static_cast<size_t>(blockIdx.x) * width;
    const int lane = threadIdx.x;
    int invalid = 0;
    for (int i = lane; i < width; i += threads) {
        const float value = __fadd_rn(state[i], row[i]);
        state[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

bool aligned(const void *pointer) { return reinterpret_cast<uintptr_t>(pointer) % alignof(float) == 0; }

__global__ void inkling_stream_norm(const float *src, float *dst, float *health, int width,
                                    float eps) {
    __shared__ float scratch[threads];
    const int lane = threadIdx.x;
    int invalid = 0;
    const float scale = rms_scale(src, width, eps, scratch, invalid);
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(src[i], scale);
        dst[i] = value;
        invalid |= !isfinite(value);
    }
    report(health, invalid);
}

// Host-ordered variants of the mix and the seal (`Stream::mix`, `Stream::seal` and
// `CpuForward::norm` of the host): every reduction runs on one lane in the host's element order,
// so the results are the host's bits. One block per state; the other lanes stage the normalized
// current row and write the outputs, and the states of a prompt keep the card busy.
constexpr int ordered_max_width = 4096;

// Products per batch of an ordered sum. One lane adds in element order, so each add waits for
// the one before it; the batch's loads and products are independent of the sum and are issued
// together, so a row read from global memory costs one memory latency per batch, not per element.
constexpr int ordered_batch = 16;

// sum(x[i] * y[i]) in element order from -0.0, the host's `dot`.
__device__ float ordered_dot(const float *__restrict__ x, const float *__restrict__ y, int width) {
    float sum = -0.0f;
    int i = 0;
    for (; i + ordered_batch <= width; i += ordered_batch) {
        float products[ordered_batch];
        #pragma unroll
        for (int j = 0; j < ordered_batch; ++j) products[j] = __fmul_rn(x[i + j], y[i + j]);
        #pragma unroll
        for (int j = 0; j < ordered_batch; ++j) sum = __fadd_rn(sum, products[j]);
    }
    for (; i < width; ++i) sum = __fadd_rn(sum, __fmul_rn(x[i], y[i]));
    return sum;
}

// The host's rms_norm_with_scale scale, summed in element order on the calling lane.
__device__ float ordered_scale(const float *row, int width, float eps) {
    const float squares = ordered_dot(row, row, width);
    return __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(squares, static_cast<float>(width)), eps)));
}

// out = norm(mix) * gamma when `gamma` is set, else the mix. Without a query the mix is the raw
// current row (the first visit's input). Output rows are `out_stride` floats apart.
__global__ void inkling_stream_mix_ordered(const float *state, const float *query,
                                           const float *gamma, float *out, float *health,
                                           int first_block, int blocks, int width, float mix_eps,
                                           float norm_eps, int stride, int out_stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    out += static_cast<size_t>(blockIdx.x) * out_stride;
    extern __shared__ float ordered[];
    float *current = ordered;
    float *mixed = ordered + width;
    __shared__ float scores[max_rows];
    __shared__ float weights[max_rows];
    __shared__ float scalars[2];
    const int lane = threadIdx.x;
    int invalid = 0;
    if (query) {
        const int rows = blocks - first_block + 1;
        // Lane 0 scales the current row while warp 1 scores the sealed rows, one lane each.
        if (lane == 0) {
            scalars[0] = ordered_scale(state, width, mix_eps);
        } else if (lane >= 32 && lane - 32 < rows - 1) {
            scores[lane - 32] = ordered_dot(mix_row(state, blocks, rows, lane - 32, width), query, width);
        }
        __syncthreads();
        const float scale = scalars[0];
        for (int i = lane; i < width; i += threads) {
            const float value = state[i];
            invalid |= !isfinite(value);
            current[i] = __fmul_rn(value, scale);
        }
        __syncthreads();
        if (lane == 0) {
            scores[rows - 1] = ordered_dot(current, query, width);
            const float inverse = __frcp_rn(__fsqrt_rn(static_cast<float>(width)));
            float peak = -INFINITY;
            for (int r = 0; r < rows; ++r) {
                scores[r] = __fmul_rn(scores[r], inverse);
                peak = fmaxf(peak, scores[r]);
            }
            float mass = 0.0f;
            for (int r = 0; r < rows; ++r) {
                const float weight = portable_exp(__fsub_rn(scores[r], peak));
                weights[r] = weight;
                mass = __fadd_rn(mass, weight);
            }
            scalars[1] = mass;
            invalid |= !isfinite(mass) || !isfinite(peak);
        }
        __syncthreads();
        const float mass = scalars[1];
        for (int c = lane; c < width; c += threads) {
            float sum = 0.0f;
            for (int r = 0; r < rows - 1; ++r) {
                sum = __fadd_rn(sum, __fmul_rn(weights[r], mix_row(state, blocks, rows, r, width)[c]));
            }
            sum = __fadd_rn(sum, __fmul_rn(weights[rows - 1], current[c]));
            mixed[c] = __fdiv_rn(sum, mass);
        }
    } else {
        for (int c = lane; c < width; c += threads) mixed[c] = state[c];
    }
    __syncthreads();
    if (gamma) {
        if (lane == 0) scalars[0] = ordered_scale(mixed, width, norm_eps);
        __syncthreads();
        const float scale = scalars[0];
        for (int c = lane; c < width; c += threads) {
            const float value = __fmul_rn(__fmul_rn(mixed[c], scale), gamma[c]);
            out[c] = value;
            invalid |= !isfinite(value);
        }
    } else {
        for (int c = lane; c < width; c += threads) {
            out[c] = mixed[c];
            invalid |= !isfinite(mixed[c]);
        }
    }
    report(health, invalid);
}

__global__ void inkling_stream_seal_ordered(float *state, const float *next, float *health,
                                            int blocks, int width, float eps, int stride) {
    state += static_cast<size_t>(blockIdx.x) * stride * width;
    next += static_cast<size_t>(blockIdx.x) * width;
    __shared__ float shared_scale;
    const int lane = threadIdx.x;
    if (lane == 0) shared_scale = ordered_scale(state, width, eps);
    __syncthreads();
    const float scale = shared_scale;
    float *sealed = state + (static_cast<size_t>(blocks) + 1) * width;
    int invalid = !isfinite(scale);
    for (int i = lane; i < width; i += threads) {
        const float value = __fmul_rn(state[i], scale);
        const float replacement = next[i];
        sealed[i] = value;
        state[i] = replacement;
        invalid |= !isfinite(value) || !isfinite(replacement);
    }
    report(health, invalid);
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a);
    const uintptr_t y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

bool valid_shape(int width, float eps) {
    return width >= 1 && width <= max_width && isfinite(eps) && eps > 0.0f;
}

// `rows` states of `used` rows each, `stride >= used` rows apart; 0 when the shape is invalid.
size_t states_bytes(int rows, int stride, int used, int width) {
    if (rows < 1 || rows > max_states || stride < used) return 0;
    return ((static_cast<size_t>(rows) - 1) * stride + used) * width * sizeof(float);
}
}  // namespace

extern "C" int inference_cuda_inkling_stream_mix(const float *state, const float *query, float *out,
                                                 float *health, int first_block, int blocks,
                                                 int width, float eps, int rows, int stride,
                                                 void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !query || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(out) || !aligned(health) || !valid_shape(width, eps) || first_block < 0 ||
        blocks < first_block || blocks > max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 1, width);
    const size_t out_bytes = static_cast<size_t>(rows) * row;
    if (state_bytes == 0 || overlaps(out, out_bytes, state, state_bytes) ||
        overlaps(out, out_bytes, query, row) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), query, row) ||
        overlaps(health, sizeof(float), out, out_bytes) ||
        overlaps(state, state_bytes, query, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_mix<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, query, out, health, first_block, blocks, width, eps, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_seal(float *state, const float *next, float *health,
                                                  int blocks, int width, float eps, int rows,
                                                  int stride, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !next || !health || !stream || !aligned(state) || !aligned(next) ||
        !aligned(health) || !valid_shape(width, eps) || blocks < 0 || blocks >= max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 2, width);
    const size_t next_bytes = static_cast<size_t>(rows) * row;
    if (state_bytes == 0 || overlaps(state, state_bytes, next, next_bytes) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), next, next_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_seal<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, next, health, blocks, width, eps, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_add(float *state, const float *row, float *health,
                                                 int width, int rows, int stride, void *stream,
                                                 int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !row || !health || !stream || !aligned(state) || !aligned(row) ||
        !aligned(health) || !valid_shape(width, 1.0f)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t state_bytes = states_bytes(rows, stride, 1, width);
    const size_t row_bytes = static_cast<size_t>(rows) * width * sizeof(float);
    if (state_bytes == 0 || overlaps(state, state_bytes, row, row_bytes) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), row, row_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_add<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, row, health, width, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_norm(const float *src, float *dst, float *health,
                                                   int width, float eps, void *stream,
                                                   int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!src || !dst || !health || !stream || !aligned(src) || !aligned(dst) ||
        !aligned(health) || !valid_shape(width, eps)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    if (overlaps(dst, row, src, row) || overlaps(health, sizeof(float), src, row) ||
        overlaps(health, sizeof(float), dst, row)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_norm<<<1, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        src, dst, health, width, eps);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_mix_ordered(
    const float *state, const float *query, const float *gamma, float *out, float *health,
    int first_block, int blocks, int width, float mix_eps, float norm_eps, int rows, int stride,
    int out_stride, void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !out || !health || !stream || !aligned(state) || !aligned(query) ||
        !aligned(gamma) || !aligned(out) || !aligned(health) || width > ordered_max_width ||
        !valid_shape(width, query ? mix_eps : 1.0f) || !valid_shape(width, gamma ? norm_eps : 1.0f) ||
        out_stride < width || first_block < 0 || blocks < first_block || blocks > max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 1, width);
    const size_t out_bytes =
        ((static_cast<size_t>(rows) - 1) * out_stride + width) * sizeof(float);
    if (state_bytes == 0 || overlaps(out, out_bytes, state, state_bytes) ||
        (query && (overlaps(out, out_bytes, query, row) ||
                   overlaps(health, sizeof(float), query, row))) ||
        (gamma && (overlaps(out, out_bytes, gamma, row) ||
                   overlaps(health, sizeof(float), gamma, row))) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), out, out_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_mix_ordered<<<rows, threads, 2 * row, static_cast<cudaStream_t>(stream)>>>(
        state, query, gamma, out, health, first_block, blocks, width, mix_eps, norm_eps, stride,
        out_stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}

extern "C" int inference_cuda_inkling_stream_seal_ordered(float *state, const float *next,
                                                          float *health, int blocks, int width,
                                                          float eps, int rows, int stride,
                                                          void *stream, int *out_stage) {
    report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
    if (!state || !next || !health || !stream || !aligned(state) || !aligned(next) ||
        !aligned(health) || !valid_shape(width, eps) || blocks < 0 || blocks >= max_blocks) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    const size_t row = static_cast<size_t>(width) * sizeof(float);
    const size_t state_bytes = states_bytes(rows, stride, blocks + 2, width);
    const size_t next_bytes = static_cast<size_t>(rows) * row;
    if (state_bytes == 0 || overlaps(state, state_bytes, next, next_bytes) ||
        overlaps(health, sizeof(float), state, state_bytes) ||
        overlaps(health, sizeof(float), next, next_bytes)) {
        return INFERENCE_CUDA_INVALID_ARGUMENT;
    }
    inkling_stream_seal_ordered<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        state, next, health, blocks, width, eps, stride);
    Failure failure;
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    report_stage(out_stage, failure.stage);
    return static_cast<int>(failure.err);
}
