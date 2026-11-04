/* The sparse-attention sketch append kernel of the resident decode step: projects the
 * post-norm, post-rope k rows of one step onto the fixed per-KV-head orthonormal basis and
 * appends the fp16 projections to the sketch plane of the query-aware sparse KV path
 * (`sketch [n_kv_head][cells][rank]`, `cell = position`; the contract is in
 * `resident/sparse_sketch.h`). The poison rule mirrors the one this repository added to the
 * vendored attn_prep and kv8_append.
 *
 * One warp per (kv head, row): the warp holds the row in registers (one thread per 32-strided
 * value) and loops the rank, reducing each inner product with a fixed 32-lane shuffle tree, so
 * the result is bit-deterministic per launch geometry. Lane 0 stores the fp16 value. The kernel
 * is one launch of grid (n_kv_head, rows) and 32 threads; all geometry is run-time.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"
#include "resident/sparse_sketch.h"

namespace {

constexpr int kWarp = 32;
constexpr int kMaxM = 16;         // decode rows (the step's look-ahead)
constexpr int kMaxChunk = 65535;  // prefill rows (the grid's y extent)
constexpr int kMaxKvHeads = 64;
constexpr int kMinHead = 32;  // one warp holds the row
constexpr int kMaxHead = 1024;
constexpr int kMaxRank = 1024;
constexpr int kMaxCells = (1 << 30) - 1;
// Mask 4 of StepParams.status (invalid step state); equals
// INFERENCE_CUDA_STATUS_INVALID_STEP_STATE of resident/step_params.h, as attn_prep.cu.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *k;       // [m][n_kv_head][head_dim]
    const float *basis;   // [n_kv_head][rank][head_dim]
    __half *sketch;       // [n_kv_head][cells][rank]
    const int *pos_dev;   // the decode entry: the device word; NULL for the chunk entry
    int pos0;             // the chunk entry: the host base position
    int *status;
    int m, n_kv_head, head_dim, rank, cells;
};

__global__ void sparse_sketch_k(Args a) {
    const int kvh = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    // Poison rule, as in attn_prep and kv8_append: the position of token 0 is read from the
    // device (decode) or validated here (chunk) and tested, in 64 bits, against the cells
    // before any other memory access. The answer is block uniform (while no other stream
    // writes the word), so the early return happens before any barrier. A refused launch
    // writes nothing and ORs mask 4 into status (one thread, integer atomic).
    const long long base = a.pos_dev ? (long long)*a.pos_dev : (long long)a.pos0;
    if (base < 0 || base + a.m > a.cells) {
        if (a.status && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    const int steps = a.head_dim / kWarp;
    const size_t cell = (size_t)(base + t);
    const size_t in_at = ((size_t)t * a.n_kv_head + kvh) * a.head_dim + d;
    // The row in registers: thread d holds values d, d + 32, .. (head_dim <= 1024 => <= 32).
    float row[32];
#pragma unroll 4
    for (int i = 0; i < steps; ++i) row[i] = a.k[in_at + (size_t)i * kWarp];
    const float *b = a.basis + (size_t)kvh * a.rank * a.head_dim;
    __half *cell_out = a.sketch + (size_t)kvh * a.cells * a.rank + cell * a.rank;
    for (int r = 0; r < a.rank; ++r, b += a.head_dim) {
        // Fixed tree order (fma rounds once per step, the shuffle tree in a fixed lane order),
        // so every launch of the same geometry stores the same fp16 value.
        float acc = 0.f;
#pragma unroll 4
        for (int i = 0; i < steps; ++i) acc = __fmaf_rn(row[i], b[(size_t)i * kWarp + d], acc);
#pragma unroll
        for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (d == 0) cell_out[r] = __float2half_rn(acc);
    }
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

bool valid_data(const void *k, const void *basis, const void *sketch) {
    return k != NULL && basis != NULL && sketch != NULL && aligned_to(k, 4) &&
           aligned_to(basis, 4) && aligned_to(sketch, 4);
}

// rows tokens, n_kv_head kv heads, head_dim values per head (a multiple of the warp), rank
// projection outputs, cells plane cells per head with room for the rows.
bool valid_geometry(int rows, int n_kv_head, int head_dim, int rank, int cells) {
    return rows >= 1 && n_kv_head >= 1 && n_kv_head <= kMaxKvHeads && head_dim >= kMinHead &&
           head_dim <= kMaxHead && head_dim % kWarp == 0 && rank >= 2 && rank <= kMaxRank &&
           rank <= head_dim && cells >= rows && cells <= kMaxCells;
}

Args make_args(const void *k, const void *basis, void *sketch, int rows, int n_kv_head,
               int head_dim, int rank, int cells) {
    Args a = {};
    a.k = (const float *)k;
    a.basis = (const float *)basis;
    a.sketch = (__half *)sketch;
    a.m = rows;
    a.n_kv_head = n_kv_head;
    a.head_dim = head_dim;
    a.rank = rank;
    a.cells = cells;
    return a;
}

Failure launch(const Args &a, cudaStream_t s) {
    Failure failure;
    sparse_sketch_k<<<dim3(a.n_kv_head, a.m), kWarp, 0, s>>>(a);
    failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
    return failure;
}

}  // namespace

extern "C" int inference_cuda_sparse_sketch_launch(const inference_cuda_sparse_sketch_args *in,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers = in != NULL && valid_data(in->k, in->basis, in->sketch) &&
                              in->pos != NULL && in->status != NULL && aligned_to(in->pos, 4) &&
                              aligned_to(in->status, 4);
        if (!pointers || !valid_geometry(in->m, in->n_kv_head, in->head_dim, in->rank,
                                         in->cells) ||
            in->m > kMaxM) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->basis, in->sketch, in->m, in->n_kv_head, in->head_dim,
                           in->rank, in->cells);
        a.pos_dev = (const int *)in->pos;
        a.status = (int *)in->status;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}

extern "C" int inference_cuda_sparse_sketch_chunk_launch(
    const inference_cuda_sparse_sketch_chunk_args *in, void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        const bool pointers = in != NULL && valid_data(in->k, in->basis, in->sketch);
        const bool position =
            in == NULL || (in->pos0 >= 0 && (long long)in->pos0 + in->t <= in->cells);
        if (!pointers || !position || !valid_geometry(in->t, in->n_kv_head, in->head_dim,
                                                      in->rank, in->cells) ||
            in->t > kMaxChunk) {
            return INFERENCE_CUDA_INVALID_ARGUMENT;
        }
        Args a = make_args(in->k, in->basis, in->sketch, in->t, in->n_kv_head, in->head_dim,
                           in->rank, in->cells);
        a.pos0 = in->pos0;
        const Failure failure = launch(a, (cudaStream_t)stream);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
