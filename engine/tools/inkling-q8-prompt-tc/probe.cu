#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include "resident/dense_gemv.h"
#include "resident/gguf_gemv.cuh"
#include "baseline.inc"

#define CK(expr) do { const auto status = (expr); if (status != 0) { \
    fprintf(stderr, "%s failed: %d\n", #expr, int(status)); exit(2); } } while (0)

__global__ void cast_x(const float *x, half *h, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) h[i] = __float2half_rn(x[i]);
}

__global__ void unpack_w(const uint8_t *w, half *h, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) h[i] = __float2half_rn(fifi_gguf::decode<8>(w + (i / 32) * 34, i % 32));
}

// A CTA owns 16 prompt rows and Warps adjacent 16-output tiles. Pad only M;
// the trace geometries all have N divisible by 64 and K divisible by 32.
template<int Warps>
__global__ void packed_wmma(const uint8_t *w, const float *x, float *y, int m, int n, int k) {
    namespace tc = nvcuda::wmma;
    __shared__ __align__(32) half sx[16 * 32], sw[Warps * 16 * 32];
    __shared__ __align__(32) float sy[Warps * 16 * 16];
    const int t = threadIdx.x, warp = t / 32;
    const int first_m = blockIdx.y * 16, first_n = blockIdx.x * Warps * 16;
    tc::fragment<tc::accumulator, 16, 16, 16, float> result;
    tc::fill_fragment(result, 0.0f);
    for (int col = 0; col < k; col += 32) {
        for (int i = t; i < 16 * 32; i += 32 * Warps) {
            int row = first_m + i / 32;
            sx[i] = __float2half_rn(row < m ? x[row * k + col + i % 32] : 0.0f);
        }
        for (int i = t; i < Warps * 16 * 32; i += 32 * Warps) {
            const auto *q = w + ((first_n + i / 32) * (k / 32) + col / 32) * 34;
            sw[i] = __float2half_rn(fifi_gguf::decode<8>(q, i % 32));
        }
        __syncthreads();
        #pragma unroll
        for (int off = 0; off < 32; off += 16) {
            tc::fragment<tc::matrix_a, 16, 16, 16, half, tc::row_major> a;
            tc::fragment<tc::matrix_b, 16, 16, 16, half, tc::col_major> b;
            tc::load_matrix_sync(a, sx + off, 32);
            tc::load_matrix_sync(b, sw + warp * 16 * 32 + off, 32);
            tc::mma_sync(result, a, b, result);
        }
        __syncthreads();
    }
    tc::store_matrix_sync(sy + warp * 256, result, 16, tc::mem_row_major);
    __syncwarp();
    for (int i = t % 32; i < 256; i += 32) {
        const int row = first_m + i / 16;
        if (row < m) y[row * n + first_n + warp * 16 + i % 16] = sy[warp * 256 + i];
    }
}

enum Mode { packed, pair, fused, cached, unpack_each };
const char *names[] = {"packed", "pair", "fused_wmma", "blas_cached_fp16", "blas_unpack_each"};
constexpr int modes = sizeof(names) / sizeof(names[0]);

struct Device {
    inference_cuda_dense_gemv_args a{};
    uint8_t *w;
    float *x, *y;
    half *wh, *xh;
    cublasHandle_t blas;

    void unpack() const {
        unpack_w<<<(a.rows * a.cols + 255) / 256, 256>>>(w, wh, a.rows * a.cols);
    }

    void launch(int mode) const {
        const int m = a.m, n = a.rows, k = a.cols;
        if (mode == packed) {
            gguf_packed_gemv<8, 32, 34><<<dim3(n / 8, m), 256, k * sizeof(float)>>>(a, SharedWeights{});
        } else if (mode == pair) {
            inkling_q8_reuse_pair<<<dim3(n / 8, (m + 1) / 2), 256, k * sizeof(float2)>>>(a);
        } else if (mode == fused) {
            if (m >= 63) packed_wmma<4><<<dim3(n / 64, (m + 15) / 16), 128>>>(w, x, y, m, n, k);
            else packed_wmma<1><<<dim3(n / 16, (m + 15) / 16), 32>>>(w, x, y, m, n, k);
        } else {
            if (mode == unpack_each) unpack();
            cast_x<<<(m * k + 255) / 256, 256>>>(x, xh, m * k);
            const float one = 1, zero = 0;
            CK(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one,
                wh, CUDA_R_16F, k, xh, CUDA_R_16F, k, &zero, y, CUDA_R_32F, n,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
};

// Deterministic synthetic signed Q8 codes, variable binary16 scales, and
// F32 activations (not model values and not pre-rounded to FP16).
uint32_t rng(uint32_t &s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

void check_cpu(const std::vector<uint8_t> &w, const std::vector<float> &x,
               const std::vector<float> &ref, int m, int n, int k) {
    for (int r : {0, m / 2, m - 1}) for (int c : {0, 7, n / 2, n - 1}) {
        double dot = 0, sum_abs = 0;
        for (int i = 0; i < k; ++i) {
            const uint8_t *b = w.data() + (c * (k / 32) + i / 32) * 34;
            __half_raw h; h.x = uint16_t(b[0] | (b[1] << 8));
            double prod = __half2float(half(h)) * int8_t(b[2 + i % 32]) * double(x[r * k + i]);
            dot += prod;
            sum_abs += std::abs(prod);
        }
        if (!std::isfinite(ref[r * n + c]) || std::abs(ref[r * n + c] - dot) > 5e-7 * sum_abs) {
            fprintf(stderr, "CPU oracle mismatch at %d,%d\n", r, c); exit(3);
        }
    }
}

void check_all(const Device &d, const std::vector<uint8_t> &w, const std::vector<float> &x) {
    const int m = d.a.m, n = d.a.rows, k = d.a.cols, len = m * n;
    std::vector<float> ref(len), out(len + 32);
    for (int mode = 0; mode < modes; ++mode) {
        CK(cudaMemset(d.y, 0xff, out.size() * sizeof(float)));
        d.launch(mode);
        CK(cudaMemcpy(out.data(), d.y, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = len; i < int(out.size()); ++i) {
            uint32_t guard; memcpy(&guard, &out[i], sizeof(guard));
            if (guard != UINT32_MAX) { fprintf(stderr, "output canary changed\n"); exit(3); }
        }
        if (mode == packed) {
            std::copy_n(out.begin(), len, ref.begin());
            check_cpu(w, x, ref, m, n, k);
        }
        double l2 = 0, ref2 = 0, max_abs = 0, peak = 0;
        for (int i = 0; i < len; ++i) {
            double e = double(out[i]) - ref[i];
            if (!std::isfinite(e)) { fprintf(stderr, "non-finite/unwritten output\n"); exit(3); }
            l2 += e * e;
            ref2 += double(ref[i]) * ref[i];
            max_abs = std::max(max_abs, std::abs(e));
            peak = std::max(peak, std::abs(double(ref[i])));
        }
        const double rel_l2 = sqrt(l2 / ref2);
        printf("error,%d,%d,%d,%s,%.9g,%.9g,%.9g\n", m,n,k,names[mode],max_abs,rel_l2,max_abs/peak);
        if (mode == pair && memcmp(ref.data(), out.data(), len * sizeof(float))) exit(3);
        if (rel_l2 > 0.0008 || max_abs/peak > 0.0015) exit(3);
    }
}

void time_all(const Device &d, bool smoke) {
    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
    const int reps = smoke ? 3 : 40;
    for (int trial = 0; trial < (smoke ? 1 : 10); ++trial) {
        for (int slot = 0; slot < modes; ++slot) {
            int mode = (trial / 2 + (trial % 2 ? modes - 1 - slot : slot)) % modes;
            for (int warm = 0; warm < 8; ++warm) d.launch(mode);
            CK(cudaEventRecord(start));
            for (int rep = 0; rep < reps; ++rep) d.launch(mode);
            CK(cudaGetLastError()); CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
            float ms; CK(cudaEventElapsedTime(&ms, start, stop));
            printf("time,%d,%d,%d,%s,%d,%.6f\n", d.a.m,d.a.rows,d.a.cols,names[mode],trial,ms*1000/reps);
        }
    }
    CK(cudaEventDestroy(stop)); CK(cudaEventDestroy(start));
}

void run(cublasHandle_t handle, int m, int n, int k, bool smoke) {
    Device d{}; d.blas = handle;
    std::vector<uint8_t> w(size_t(n) * (k / 32) * 34);
    std::vector<float> x(m * k);
    uint32_t state = 0x0c2248f5u;
    for (size_t i = 0; i < w.size(); i += 34) {
        uint16_t h = 0x1400 + (rng(state) % 0x1000); // positive normal scales
        w[i] = h & 255; w[i+1] = h >> 8;
        for (int j = 2; j < 34; ++j) w[i+j] = rng(state) >> 24;
    }
    for (auto &v : x) v = float(int(rng(state) >> 8) - 8388608) / 8388608.0f;
    CK(cudaMalloc(&d.w, w.size())); CK(cudaMalloc(&d.x, x.size() * sizeof(float)));
    CK(cudaMalloc(&d.y, (m * n + 32) * sizeof(float)));
    CK(cudaMalloc(&d.wh, n * k * sizeof(half))); CK(cudaMalloc(&d.xh, m * k * sizeof(half)));
    CK(cudaMemcpy(d.w, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    d.a.m = m; d.a.rows = n; d.a.cols = k; d.a.ld_x = k; d.a.ld_y = n;
    d.a.x = d.x; d.a.w = d.w; d.a.y = d.y;
    d.unpack(); check_all(d, w, x); time_all(d, smoke);
    CK(cudaFree(d.xh)); CK(cudaFree(d.wh)); CK(cudaFree(d.y)); CK(cudaFree(d.x)); CK(cudaFree(d.w));
}

int main(int argc, char **argv) {
    const bool smoke = argc == 2 && !strcmp(argv[1], "--smoke");
    if (argc != 1 && !smoke) return 2;
    cudaDeviceProp p; int count;
    CK(cudaGetDeviceCount(&count)); CK(cudaGetDeviceProperties(&p, 0));
    if (count != 1 || p.major != 7 || p.minor != 5 || !strstr(p.name, "2080 Ti")) return 2;
    size_t free, total; CK(cudaMemGetInfo(&free, &total));
    if (free < 512ul * 1024 * 1024) return 2;
    cublasHandle_t handle; CK(cublasCreate(&handle));
    CK(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    int blas_version; CK(cublasGetVersion(handle, &blas_version));
    printf("device=%s sm=%d%d free_bytes=%zu cublas=%d weights=synthetic_Q8_0 x=synthetic_F32\n",
        p.name,p.major,p.minor,free,blas_version);
    for (int n : {384, 1024, 3072, 2304}) {
        int k = n == 2304 ? 3072 : 2304;
        for (int m : {1,7,8,63,64}) {
            if (m == 7 && n != 384 && n != 2304) continue;
            if (m == 63 && n == 2304) continue;
            if (smoke && !((n == 384 && m == 7) || (n == 2304 && m == 64))) continue;
            run(handle, m, n, k, smoke);
        }
    }
    CK(cublasDestroy(handle));
}
