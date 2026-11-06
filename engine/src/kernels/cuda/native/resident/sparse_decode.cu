/* Sparse-attention decode: attn_decode_q8.cu's arithmetic over the SELECTED cells only.
 * The contract is in resident/sparse_decode.h; the numerics deliberately mirror
 * attn_q8_part_k of attn_decode_q8.cu line for line (same dequant, same fmaf chains, same
 * reduce8 order, same 64-tile online softmax) so that subset-exactness against the
 * independent reference of family/qwen36/sparse_reference.rs is a property of the SELECTION,
 * not of a second attention arithmetic. Single block per KV head (grid n_kv_head); no
 * splits, no part buffer, no combine - the baseline kernel of phase 4c.
 */
#include "sparse_decode.h"

#include <cuda_fp16.h>
#include <stdint.h>

#include "abi.h"
#include "device.h"

namespace {

constexpr int kTile = 64;    // the tile of attn_decode_q8.cu; counts need not be multiples
constexpr int kQuant = 32;   // values per fp16 scale: the frozen group of the Q8 layout
constexpr int kThreads = 256;
// Mask 4 of StepParams.status (invalid step state), as sparse_topk.cu and the poison rule of
// attn_prep / kv8_append.
constexpr int kStatusInvalidStepState = 4;

struct Args {
    const float *q;                // [n_head][D] f32, one decode token, unscaled
    const signed char *k_codes;    // [n_kv_head][cells][D]
    const unsigned short *k_scales;  // [n_kv_head][cells][D / 32]
    const signed char *v_codes;
    const unsigned short *v_scales;
    const unsigned *indices;       // [n_kv_head][row]
    const unsigned *counts;        // [n_kv_head]
float *out;                    // [n_head * D]
const float *proj;             // NULL (no gate) or [n_head][2D] f32: sigmoid of the D half gates out
int *status;
    int n_kv_head, n_head, head_dim, cells, row;
    float scale;
};

__device__ inline float h2f(unsigned short bits) { return __half2float(__ushort_as_half(bits)); }

// Reduces eight per-lane partial sums (one per query head) over the 32 lanes with 9
// shuffles, verbatim from attn_decode_q8.cu: lane l with (l & 3) == 0 holds the full sum of
// head ((l >> 4) & 1) * 4 + ((l >> 3) & 1) * 2 + ((l >> 2) & 1); the order is fixed.
__device__ inline float reduce8(const float (&a)[8], int lane, int &head) {
    const unsigned full = 0xffffffffu;
    const bool h16 = lane & 16, h8 = lane & 8, h4 = lane & 4;
    float b[4], c[2];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float keep = h16 ? a[i + 4] : a[i], give = h16 ? a[i] : a[i + 4];
        b[i] = keep + __shfl_xor_sync(full, give, 16);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float keep = h8 ? b[i + 2] : b[i], give = h8 ? b[i] : b[i + 2];
        c[i] = keep + __shfl_xor_sync(full, give, 8);
    }
    const float keep = h4 ? c[1] : c[0], give = h4 ? c[0] : c[1];
    float d = keep + __shfl_xor_sync(full, give, 4);
    d += __shfl_xor_sync(full, d, 2);
    d += __shfl_xor_sync(full, d, 1);
    head = (h16 ? 4 : 0) + (h8 ? 2 : 0) + (h4 ? 1 : 0);
    return d;
}

__device__ inline float warp_max_a(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ inline float warp_sum_a(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// One block per KV head: threads 0 up to D (exclusive) own one output dimension, warps 0 up to
// G (exclusive) each score one query head, G = n_head / n_kv_head (1..8, the production GQA mapping).
__global__ void __launch_bounds__(kThreads) sparse_decode_k(Args a) {
    const int kvh = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int D = a.head_dim, G = a.n_head / a.n_kv_head;
    const int groups = D / kQuant;

    // Poison rule: the device word decides. The head's count and indices are validated
    // against the capacity BEFORE any write. The decision is made block-uniformly AFTER a
    // barrier (a divergent early return in front of __syncthreads is undefined), and the
    // loop bound is min(count, row) so a count above row cannot read past the indices row
    // nor sign-flip through an int cast. An upstream refusal (score or topk ORs mask 4 and
    // writes nothing) must cascade: the counts are then stale, and garbage that happens to
    // pass the bounds check would silently compute a wrong row instead of refusing.
    const unsigned count = a.counts[kvh];
    const unsigned *sel = a.indices + (size_t)kvh * a.row;
    __shared__ int s_bad;
    if (tid == 0) s_bad = (*a.status & kStatusInvalidStepState) ? 1 : 0;
    __syncthreads();
    if (count > (unsigned)a.row) {
        s_bad = 1;
    } else {
        const int n_check = (int)count;
        for (int i = tid; i < n_check; i += kThreads) {
            if (sel[i] >= (unsigned)a.cells) s_bad = 1;
        }
    }
    __syncthreads();
    if (s_bad) {
        if (tid == 0) atomicOr(a.status, kStatusInvalidStepState);
        return;
    }
    // Count is now provably <= row and every cell is in range; reread it as the tile bound.
    const int n_keys = (int)count;

    __shared__ float s_sc[8][kTile + 1];           // scores, head-major, padded like production
    __shared__ __align__(16) float s_p[kTile][8];  // probabilities, key-major (float4 reads)
    __shared__ float s_corr[8];
    __shared__ float s_inv_l[8];                   // 1 / l per head for the final normalize

    const size_t code_stride = (size_t)a.cells * D;
    const size_t scale_stride = (size_t)a.cells * groups;
    const bool own = tid < D;

    // This lane's dims are 4 * lane .. +3 and for D 256 also 128 + 4 * lane .. +3; the group
    // of chunk c is 4c + lane / 8. q is pre-scaled once, exactly as attn_q8_part_k.
    float4 qv[8][2];
    const int nch = D / 128;
    for (int g = 0; g < G; ++g) {
        const float *qh = a.q + ((size_t)(kvh * G + g)) * D;
        for (int c = 0; c < nch; ++c) {
            qv[g][c] = *reinterpret_cast<const float4 *>(qh + 128 * c + 4 * lane);
            qv[g][c].x *= a.scale; qv[g][c].y *= a.scale;
            qv[g][c].z *= a.scale; qv[g][c].w *= a.scale;
        }
    }
    float acc[8];
    for (int g = 0; g < G; ++g) acc[g] = 0.f;
    float m_run = -INFINITY, l_run = 0.f;  // head `warp`'s running max and sum

    for (int base = 0; base < n_keys; base += kTile) {
        const int nvalid = min(kTile, n_keys - base);
        // V of this tile: codes and scale bits in registers, issued first to overlap the
        // scores' latency; threads that own no dimension still join every barrier.
        signed char vc[kTile];
        unsigned short vs[kTile];
        if (own) {
            const signed char *vcol = a.v_codes + (size_t)kvh * code_stride + tid;
            const unsigned short *vrow =
        a.v_scales + (size_t)kvh * scale_stride + (tid / kQuant);
#pragma unroll 8
            for (int j = 0; j < kTile; ++j) {
                vc[j] = j < nvalid ? vcol[(size_t)sel[base + j] * D] : (signed char)0;
                vs[j] = j < nvalid ? vrow[(size_t)sel[base + j] * groups] : (unsigned short)0;
            }
        }
        // Phase 1: scores. Warp w scores selected keys w, w + 8, ...; each lane dequantizes
        // its 4 dims of each key against the key's stored fp16 scale.
        for (int i = 0; i < kTile / 8; ++i) {
            const int j = warp + 8 * i;
            if (j < nvalid) {
                const int cell = (int)sel[base + j];
                const signed char *kc =
        a.k_codes + (size_t)kvh * code_stride + (size_t)cell * D;
                const unsigned short *ks =
        a.k_scales + (size_t)kvh * scale_stride + (size_t)cell * groups;
                float kf[8];
                for (int c = 0; c < nch; ++c) {
                    const float sv = h2f(ks[4 * c + (lane >> 3)]);
                    const int word = *reinterpret_cast<const int *>(kc + 128 * c + 4 * lane);
                    kf[4 * c] = sv * (float)(signed char)(word & 0xff);
                    kf[4 * c + 1] = sv * (float)(signed char)((word >> 8) & 0xff);
                    kf[4 * c + 2] = sv * (float)(signed char)((word >> 16) & 0xff);
                    kf[4 * c + 3] = sv * (float)(signed char)((word >> 24) & 0xff);
                }
                float dots[8] = {};  // heads >= G stay unused; zeroed for determinism
                for (int g = 0; g < G; ++g) {
                    float p = qv[g][0].x * kf[0];
                    p = fmaf(qv[g][0].y, kf[1], p); p = fmaf(qv[g][0].z, kf[2], p);
                    p = fmaf(qv[g][0].w, kf[3], p);
                    for (int c = 1; c < nch; ++c) {
                        p = fmaf(qv[g][c].x, kf[4 * c], p);
                        p = fmaf(qv[g][c].y, kf[4 * c + 1], p);
                        p = fmaf(qv[g][c].z, kf[4 * c + 2], p);
                        p = fmaf(qv[g][c].w, kf[4 * c + 3], p);
                    }
                    dots[g] = p;
                }
                int head;
                const float sum = reduce8(dots, lane, head);
                if ((lane & 3) == 0) s_sc[head][j] = sum;
            }
        }
        __syncthreads();
        // Phase 2: online softmax of head `warp` over the tile, two 32-lane passes, the
        // exact max first - the fixed order of attn_q8_part_k.
        {
            float m_new = m_run;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                m_new = fmaxf(m_new, warp_max_a(sc));
            }
            const float corr = expf(m_run - m_new);
            float psum = 0.f;
            for (int half = 0; half < kTile / 32; ++half) {
                const int j = half * 32 + lane;
                const float sc = j < nvalid ? s_sc[warp][j] : -INFINITY;
                const float p = j < nvalid ? expf(sc - m_new) : 0.f;
                psum += p;
                s_p[j][warp] = p;
            }
            l_run = l_run * corr + warp_sum_a(psum);
            m_run = m_new;
            if (lane == 0) s_corr[warp] = corr;
        }
        __syncthreads();
        // Phase 3: acc[g] (dimension tid of head g) += p[g][j] * V[j][tid], dequantized here.
        for (int g = 0; g < G; ++g) acc[g] *= s_corr[g];
        if (own) {
#pragma unroll 4
            for (int j = 0; j < kTile; ++j) {
                const float v = h2f(vs[j]) * (float)vc[j];
                const float4 pa = *reinterpret_cast<const float4 *>(&s_p[j][0]);
                const float4 pb = *reinterpret_cast<const float4 *>(&s_p[j][4]);
                acc[0] = fmaf(pa.x, v, acc[0]); acc[1] = fmaf(pa.y, v, acc[1]);
                acc[2] = fmaf(pa.z, v, acc[2]); acc[3] = fmaf(pa.w, v, acc[3]);
                acc[4] = fmaf(pb.x, v, acc[4]); acc[5] = fmaf(pb.y, v, acc[5]);
                acc[6] = fmaf(pb.z, v, acc[6]); acc[7] = fmaf(pb.w, v, acc[7]);
            }
        }
    }
    // Normalize per head: single split, so out = acc / l (the combine of one partial).
    if (lane == 0) s_inv_l[warp] = l_run > 0.f ? 1.f / l_run : 0.f;
    __syncthreads();
    if (own) {
        for (int g = 0; g < G; ++g) {
            // The production gated-attention output of attn_decode_q8.cu: the elementwise
            // sigmoid of proj's second half per head (one token, so ld = n_head * 2 * D).
            const float sig =
        a.proj == NULL
                    ? 1.f
                    : 1.f / (1.f + expf(-a.proj[(size_t)(kvh * G + g) * 2 * D + D + tid]));
        a.out[(size_t)(kvh * G + g) * D + tid] = acc[g] * s_inv_l[g] * sig;
        }
    }
}

bool aligned_to(const void *p, uintptr_t n) { return ((uintptr_t)p & (n - 1)) == 0; }

bool valid_args(const inference_cuda_sparse_decode_args *in) {
    if (!in->q || !in->k_codes || !in->k_scales || !in->v_codes || !in->v_scales ||
        !in->indices || !in->counts || !in->out || !in->status) {
        return false;
    }
    if (!(in->n_kv_head >= 1 && in->n_kv_head <= 64)) return false;
    if (in->n_head % in->n_kv_head != 0) return false;
    const int g = in->n_head / in->n_kv_head;
    if (!(g >= 1 && g <= 8)) return false;
    if (in->head_dim != 128 && in->head_dim != 256) return false;
    if (!(in->cells >= 1 && (long long)in->cells <= 0x3fffffffLL)) return false;
    if (!(in->row >= 1 && in->row <= (1 << 24))) return false;
    // The fp16 scale plane needs D / 32 groups per cell. The gate, when present, must be
    // 4-byte aligned; its span is n_head * 2 * head_dim floats (one token).
    if (in->proj && !aligned_to(in->proj, 4)) return false;
    return aligned_to(in->q, 16) && aligned_to(in->out, 16) &&
           aligned_to(in->k_codes, 4) && aligned_to(in->v_codes, 4) &&
           aligned_to(in->k_scales, 2) && aligned_to(in->v_scales, 2) &&
           aligned_to(in->indices, 4) && aligned_to(in->counts, 4) &&
           aligned_to(in->status, 4);
}

}  // namespace

extern "C" int inference_cuda_sparse_decode_launch(const inference_cuda_sparse_decode_args *in,
                                                   void *stream, int *out_stage) {
    try {
        report_stage(out_stage, INFERENCE_CUDA_STAGE_NONE);
        if (!in || !valid_args(in)) return INFERENCE_CUDA_INVALID_ARGUMENT;
        Args a{};
        a.q = (const float *)in->q;
        a.k_codes = (const signed char *)in->k_codes;
        a.k_scales = (const unsigned short *)in->k_scales;
        a.v_codes = (const signed char *)in->v_codes;
        a.v_scales = (const unsigned short *)in->v_scales;
        a.indices = (const unsigned *)in->indices;
        a.counts = (const unsigned *)in->counts;
        a.out = (float *)in->out;
        a.proj = (const float *)in->proj;
        a.status = (int *)in->status;
        a.n_kv_head = in->n_kv_head;
        a.n_head = in->n_head;
        a.head_dim = in->head_dim;
        a.cells = in->cells;
        a.row = in->row;
        a.scale = in->scale;
        sparse_decode_k<<<in->n_kv_head, kThreads, 0, (cudaStream_t)stream>>>(a);
        Failure failure;
        failure.record(cudaGetLastError(), INFERENCE_CUDA_STAGE_LAUNCH);
        report_stage(out_stage, failure.stage);
        return (int)failure.err;
    } catch (...) {
        return INFERENCE_CUDA_NATIVE_EXCEPTION;
    }
}
