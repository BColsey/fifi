/* The head_dim 512 part kernel of attn_prefill for Q8 affine KV planes (sm_80+). Included by
 * attn_prefill.cu inside its anonymous namespace, after the shared helpers it builds on
 * (Args, Block, sees, cache_row, swz, ldmatrix and mma wrappers); it writes the same partial
 * cells as the other part kernels, so the combine kernel and the split plan are shared.
 *
 * Shape. One block owns a query tile of 8 tokens x 4 query heads = 32 rows of one kv head and
 * one split's keys. Its 16 warps are 2 row groups (16 rows each) x 8 dim parts (64 of the 512
 * dimensions each): a warp keeps the Q fragments of its dim part, the O accumulators of its
 * 16 rows x 64 dims (32 floats per thread) and the online-softmax state of its rows. The eight
 * warps of a row group each compute a partial S = Q.K^T over their 64 dims, exchange the
 * partials through shared memory, sum all eight in one fixed order and so hold identical
 * scores, maxima and P. The C fragment of S is the A fragment of P.V, so P never leaves the
 * registers.
 *
 * Keys come in tiles of 16. A thread loads the int8 codes and f16 scales of its share of tile
 * n + 1 into registers before tile n is consumed, then dequantizes them to fp16 into the
 * other of two swizzled smem stages (K plane then V plane, 16 KB each): one block barrier per
 * tile. Shared memory is 2 x 32 KB stages + a 16 KB exchange = 80 KB; registers are capped at
 * 128 by 512 threads x 1 block per SM.
 *
 * Numerics: q, K, V and P are rounded to fp16 before the MMAs, the accumulators, scores and
 * softmax are fp32; K and V values are scale * code rounded once to fp16. */

constexpr int kD512 = 512;
constexpr int kD512Group = 4;                                   // query heads per kv head
constexpr int kD512KeyTile = 16;                                // keys per smem tile
constexpr int kD512Threads = 512;                               // 2 row groups x 8 dim parts x 32
constexpr int kD512DimPart = kD512 / 8;                         // dims per warp
constexpr int kD512PlaneBytes = kD512KeyTile * kD512 * 2;       // one fp16 K or V tile
constexpr int kD512StageBytes = 2 * kD512PlaneBytes;
constexpr int kD512ExchangeBytes = 2 * 8 * 2 * 32 * 16;         // [row group][dim part][n8][lane] f32x4
constexpr int kD512SmemBytes = 2 * kD512StageBytes + kD512ExchangeBytes;
static_assert(kD512SmemBytes == 81920, "d512 shared memory");
static_assert(kTile % kD512KeyTile == 0, "the d512 key tile divides the split granule");

/* One thread's share of a key tile before dequantization: the thread owns key (tid >> 5)
 * and the 8-dim runs at 8 * lane and 256 + 8 * lane of both planes. */
struct D512Raw {
    uint2 k[2], v[2];
    float ks[2], vs[2];
};

__device__ inline void d512_load_raw(const Args &a, const Block &b,
                                     const inference_cuda_attn_kv_pair &views, int kvh,
                                     int base, D512Raw &r) {
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
    r = D512Raw{};
    if (base + j >= b.k_end) return;  // beyond the split: zero codes, zero scales
    const int row = cache_row(a, b.first, base + j);
    const unsigned char *kp = (const unsigned char *)views.k.payload +
                              (size_t)row * views.k.payload_time_stride_bytes +
                              (size_t)kvh * views.k.payload_head_stride_bytes + 8 * lane;
    const unsigned char *vp = (const unsigned char *)views.v.payload +
                              (size_t)row * views.v.payload_time_stride_bytes +
                              (size_t)kvh * views.v.payload_head_stride_bytes + 8 * lane;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int group = 8 * h + (lane >> 2);  // dims 256 h + 8 lane lie in group size 32 group
        r.k[h] = *reinterpret_cast<const uint2 *>(kp + 256 * h);
        r.v[h] = *reinterpret_cast<const uint2 *>(vp + 256 * h);
        r.ks[h] = affine_scalar(views.k.scales, views.k.scale_dtype,
                                (size_t)row * views.k.scale_time_stride_bytes +
                                    (size_t)kvh * views.k.scale_head_stride_bytes +
                                    (size_t)group * views.k.scale_group_stride_bytes);
        r.vs[h] = affine_scalar(views.v.scales, views.v.scale_dtype,
                                (size_t)row * views.v.scale_time_stride_bytes +
                                    (size_t)kvh * views.v.scale_head_stride_bytes +
                                    (size_t)group * views.v.scale_group_stride_bytes);
    }
}

/* Eight int8 codes (two words) times one scale, as eight fp16 values packed in a uint4. */
__device__ inline uint4 d512_dequant8(uint2 codes, float scale, bool is_signed) {
    unsigned words[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const unsigned w = (p < 2 ? codes.x : codes.y) >> (16 * (p & 1));
        const int c0 = is_signed ? (int)(signed char)(w & 0xff) : (int)(w & 0xff);
        const int c1 = is_signed ? (int)(signed char)((w >> 8) & 0xff) : (int)((w >> 8) & 0xff);
        const __half2 h = __floats2half2_rn(scale * (float)c0, scale * (float)c1);
        words[p] = *reinterpret_cast<const unsigned *>(&h);
    }
    return make_uint4(words[0], words[1], words[2], words[3]);
}

#if __CUDA_ARCH__ >= 800
/* The named barrier of one row group's eight warps (ids 1 and 2; 0 is __syncthreads). */
__device__ inline void d512_bar_row_group(int rg) {
    asm volatile("bar.sync %0, 256;\n" ::"r"(1 + rg) : "memory");
}

/* Dequantizes the register tile into one stage: key j = tid >> 5, 16-byte chunk 32 h + lane of
 * its row, XOR-swizzled by the key like the ldmatrix reads expect (consecutive lanes write
 * consecutive chunks, so the 16-byte stores are conflict free). */
__device__ inline void d512_store_stage(const D512Raw &r, bool k_signed, bool v_signed,
                                        unsigned char *stage) {
    const int j = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int chunk = 32 * h + lane;
        const unsigned off = swz<kD512>(j, chunk);
        *reinterpret_cast<uint4 *>(stage + off) = d512_dequant8(r.k[h], r.ks[h], k_signed);
        *reinterpret_cast<uint4 *>(stage + kD512PlaneBytes + off) =
            d512_dequant8(r.v[h], r.vs[h], v_signed);
    }
}
#endif

__global__ void __launch_bounds__(kD512Threads, 1)
    attn_prefill_part_q8_d512_k(Args a, inference_cuda_attn_kv_pair views) {
#if __CUDA_ARCH__ >= 800
    if (!bounds_ok(a)) {
        report_bound_violation(a);
        return;  // no partial
    }
    extern __shared__ __align__(16) unsigned char d512smem[];
    const int q_tiles = (a.m + kQTile - 1) / kQTile;
    const int tile = q_tiles - 1 - (int)blockIdx.x, kvh = blockIdx.y, s = blockIdx.z;
    Block b;
    if (!block_of<kD512, kD512Group>(a, s, kvh, tile, b)) return;
    const bool k_signed = views.k.q8_signed != 0, v_signed = views.v.q8_signed != 0;
    float4 *const xch = reinterpret_cast<float4 *>(d512smem + 2 * kD512StageBytes);
    const unsigned stages = smem_addr(d512smem);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp >> 3, dp = warp & 7;
    const int g = lane >> 2, c = (lane & 3) * 2;

    /* Rows r0 = 16 rg + g and r1 = r0 + 8 of the 32-row tile: row r is token r / 4, head r % 4. */
    const int tok0 = 4 * rg + (g >> 2), head = g & 3;
    const int t_r0 = b.t0 + tok0, t_r1 = t_r0 + 2;
    const int p_row0 = a.p0 + t_r0, p_row1 = a.p0 + t_r1;

    unsigned qf[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int t = i == 0 ? t_r0 : t_r1;
#pragma unroll
            for (int hi = 0; hi < 2; ++hi) {
                float2 v = make_float2(0.f, 0.f);
                if (t < b.tile_end) {
                    const float *q = a.q + ((size_t)t * a.n_head + kvh * kD512Group + head) * kD512 +
                                     kD512DimPart * dp + 16 * ks + 8 * hi + c;
                    v = *reinterpret_cast<const float2 *>(q);
                }
                const __half2 h2 = __floats2half2_rn(v.x, v.y);
                qf[ks][i + 2 * hi] = *reinterpret_cast<const unsigned *>(&h2);
            }
        }

    /* ldmatrix lane roles: matrix mi = lane >> 3, row rr = lane & 7. K: key 8 (mi >> 1) + rr,
     * 8-dim chunk 8 dp + 2 ks + (mi & 1). V: key 8 (mi & 1) + rr, chunk 8 dp + 2 jp + (mi >> 1). */
    const int mi = lane >> 3, rr = lane & 7;
    const int k_key = 8 * (mi >> 1) + rr, k_chunk = 8 * dp + (mi & 1);
    const int v_key = 8 * (mi & 1) + rr, v_chunk = 8 * dp + (mi >> 1);

    float o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

    D512Raw raw;
    const bool has_tile = b.k_begin < b.k_end && b.first + b.k_begin <= b.max_p;
    if (has_tile) {
        d512_load_raw(a, b, views, kvh, b.k_begin, raw);
        d512_store_stage(raw, k_signed, v_signed, d512smem);
    }
    __syncthreads();

    int n = 0;
    for (int base = b.k_begin; base < b.k_end && b.first + base <= b.max_p;
         base += kD512KeyTile, ++n) {
        const int next = base + kD512KeyTile;
        const bool more = next < b.k_end && b.first + next <= b.max_p;
        if (more) d512_load_raw(a, b, views, kvh, next, raw);  // in flight during the MMAs
        const unsigned kst = stages + (n & 1) * kD512StageBytes, vst = kst + kD512PlaneBytes;

        /* This dim part's S over 16 keys: 4 k16 steps x 2 n8 key blocks. */
        float sq[2][4];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) sq[nb][e] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            unsigned kb[4];
            ldsm_x4(kst + swz<kD512>(k_key, k_chunk + 2 * ks), kb);
            mma_16816(sq[0], qf[ks], kb[0], kb[1]);
            mma_16816(sq[1], qf[ks], kb[2], kb[3]);
        }

        /* Exchange: every warp of the row group publishes its partial, then sums all eight in
         * dim part order, so the eight warps hold bit-identical scores. */
#pragma unroll
        for (int nb = 0; nb < 2; ++nb)
            xch[((rg * 8 + dp) * 2 + nb) * 32 + lane] =
                make_float4(sq[nb][0], sq[nb][1], sq[nb][2], sq[nb][3]);
        d512_bar_row_group(rg);
        float sc[2][4];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            float4 sum = xch[((rg * 8) * 2 + nb) * 32 + lane];
#pragma unroll
            for (int p = 1; p < 8; ++p) {
                const float4 v = xch[((rg * 8 + p) * 2 + nb) * 32 + lane];
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                sum.w += v.w;
            }
            sc[nb][0] = sum.x;
            sc[nb][1] = sum.y;
            sc[nb][2] = sum.z;
            sc[nb][3] = sum.w;
        }

        /* Scale and mask in natural units; a tile every row fully sees skips the mask. */
        const bool full = b.first + base + kD512KeyTile - 1 <= a.p0 + b.t0 &&
                          (a.window == 0 || b.first + base > b.max_p - a.window) &&
                          (a.mode != 1 || base + kD512KeyTile <= b.k_end) &&
                          base + kD512KeyTile <= b.k_end;
        bool vis[2][4];
        float x[2][4];
#pragma unroll
        for (int nb = 0; nb < 2; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int key = base + 8 * nb + c + (e & 1);
                const int pos = b.first + key, p = (e >> 1) == 0 ? p_row0 : p_row1;
                vis[nb][e] = full || (key < b.k_end && sees(a, pos, p));
                x[nb][e] = vis[nb][e] ? __fmul_rn(sc[nb][e], a.scale) : -INFINITY;
            }
        float bm[2] = {fmaxf(fmaxf(x[0][0], x[0][1]), fmaxf(x[1][0], x[1][1])),
                       fmaxf(fmaxf(x[0][2], x[0][3]), fmaxf(x[1][2], x[1][3]))};
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            bm[0] = fmaxf(bm[0], __shfl_xor_sync(0xffffffffu, bm[0], o2));
            bm[1] = fmaxf(bm[1], __shfl_xor_sync(0xffffffffu, bm[1], o2));
        }
        const float m_new[2] = {fmaxf(m_run[0], bm[0]), fmaxf(m_run[1], bm[1])};
        /* The no-key-yet rule: exponentiate against 0 while the max is still -inf. */
        const float m_ref[2] = {m_new[0] == -INFINITY ? 0.f : m_new[0],
                                m_new[1] == -INFINITY ? 0.f : m_new[1]};
        float pw[2][4], ps[2] = {0.f, 0.f};
#pragma unroll
        for (int nb = 0; nb < 2; ++nb)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = e >> 1;
                pw[nb][e] = exp_score_delta(x[nb][e] - m_ref[r]);
                ps[r] += normalizer_weight(vis[nb][e], x[nb][e], pw[nb][e]);
            }
#pragma unroll
        for (int o2 = 1; o2 <= 2; o2 <<= 1) {
            ps[0] += __shfl_xor_sync(0xffffffffu, ps[0], o2);
            ps[1] += __shfl_xor_sync(0xffffffffu, ps[1], o2);
        }
        const float corr[2] = {exp_score_delta(m_run[0] - m_ref[0]),
                               exp_score_delta(m_run[1] - m_ref[1])};
        l_run[0] = l_run[0] * corr[0] + ps[0];
        l_run[1] = l_run[1] * corr[1] + ps[1];
        m_run[0] = m_new[0];
        m_run[1] = m_new[1];

        /* O = O * corr + P.V over this dim part: P is the A fragment (the two n8 blocks of S
         * are its k = 0..7 and 8..15 halves), V arrives by ldmatrix.trans, two n8 blocks per
         * load. */
        unsigned pa[4];
        {
            const __half2 p0h = __floats2half2_rn(pw[0][0], pw[0][1]);
            const __half2 p1h = __floats2half2_rn(pw[0][2], pw[0][3]);
            const __half2 p2h = __floats2half2_rn(pw[1][0], pw[1][1]);
            const __half2 p3h = __floats2half2_rn(pw[1][2], pw[1][3]);
            pa[0] = *reinterpret_cast<const unsigned *>(&p0h);
            pa[1] = *reinterpret_cast<const unsigned *>(&p1h);
            pa[2] = *reinterpret_cast<const unsigned *>(&p2h);
            pa[3] = *reinterpret_cast<const unsigned *>(&p3h);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
#pragma unroll
        for (int jp = 0; jp < 4; ++jp) {
            unsigned vb[4];
            ldsm_x4_trans(vst + swz<kD512>(v_key, v_chunk + 2 * jp), vb);
            mma_16816(o[2 * jp], pa, vb[0], vb[1]);
            mma_16816(o[2 * jp + 1], pa, vb[2], vb[3]);
        }

        if (more) d512_store_stage(raw, k_signed, v_signed, d512smem + ((n + 1) & 1) * kD512StageBytes);
        __syncthreads();  // tile n + 1 visible; this stage and the exchange free
    }

    /* The partial cell: rows r0 and r1 of the tile, stats once per row, 64 dims per warp. */
    const int cell_r0 = 16 * rg + g, cell_r1 = cell_r0 + 8;
    if (dp == 0 && (lane & 3) == 0) {
        b.stats[cell_r0 * 2] = m_run[0];
        b.stats[cell_r0 * 2 + 1] = l_run[0];
        b.stats[cell_r1 * 2] = m_run[1];
        b.stats[cell_r1 * 2 + 1] = l_run[1];
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dim = kD512DimPart * dp + 8 * j + c;
        *reinterpret_cast<float2 *>(b.accs + (size_t)cell_r0 * kD512 + dim) = make_float2(o[j][0], o[j][1]);
        *reinterpret_cast<float2 *>(b.accs + (size_t)cell_r1 * kD512 + dim) = make_float2(o[j][2], o[j][3]);
    }
#else
    __trap();
#endif
}
