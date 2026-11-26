// Sums in the host's element order, for kernels whose results must be the host's bits.
// One lane adds; every intrinsic is _rn, so no build flag can contract or approximate them.
#ifndef INFERENCE_CUDA_RESIDENT_ORDERED_SUM_CUH
#define INFERENCE_CUDA_RESIDENT_ORDERED_SUM_CUH

// Products per batch of an ordered sum. Each add waits for the one before it; the batch's loads
// and products are independent of the sum and are issued together, so a row costs one memory
// latency per batch, not per element.
constexpr int ordered_batch = 16;

// sum(x[i] * y[i]) in element order from -0.0, the host's `dot`. Both rows are only read and
// may be the same row.
static __device__ __forceinline__ float ordered_dot(const float *x, const float *y, int width) {
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

// Elements per chunk of `ordered_dots`. A chunk's products are staged in shared memory, so the
// adding lanes wait on shared loads, not on device memory.
constexpr int ordered_chunk = 120;

// The shared floats `ordered_dots` stages for `count` sums: a padded chunk per sum, so the adding
// lanes read different banks.
__host__ __device__ constexpr int ordered_staged_floats(int count) { return count * (ordered_chunk + 1); }

// `count` sums in element order from -0.0, each the bits of `ordered_dot`: lane k < count
// receives sum k of `product(k, i)` over i < width. The block's lanes compute every sum's
// products one chunk at a time into `staged` (`ordered_staged_floats(count)` floats), then each
// adding lane adds its sum's chunk in element order. Every lane of the block calls it; it
// synchronizes the block before it returns.
template <typename Product>
static __device__ __forceinline__ float ordered_dots(Product product, int count, int width,
                                                     float *staged) {
    constexpr int pitch = ordered_chunk + 1;
    float sum = -0.0f;
    for (int base = 0; base < width; base += ordered_chunk) {
        const int span = min(ordered_chunk, width - base);
        for (int at = threadIdx.x; at < count * ordered_chunk; at += blockDim.x) {
            const int k = at / ordered_chunk;
            const int j = at % ordered_chunk;
            if (j < span) staged[k * pitch + j] = product(k, base + j);
        }
        __syncthreads();
        if (static_cast<int>(threadIdx.x) < count) {
            const float *chunk = staged + threadIdx.x * pitch;
#pragma unroll 16
            for (int j = 0; j < span; ++j) sum = __fadd_rn(sum, chunk[j]);
        }
        __syncthreads();
    }
    return sum;
}

// The host's `rms_norm_with_scale` scale from a row's ordered sum of squares.
static __device__ __forceinline__ float ordered_rms_scale(float squares, int width, float eps) {
    return __frcp_rn(__fsqrt_rn(__fadd_rn(__fdiv_rn(squares, static_cast<float>(width)), eps)));
}

#endif  // INFERENCE_CUDA_RESIDENT_ORDERED_SUM_CUH
