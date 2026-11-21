/*
 * Window decoders of the in-house GGUF matvec (matvec_gguf.cu). A window is 32 consecutive
 * weights of one row. Every type here turns one window into the same shape: 32 signed int8
 * weights packed four per int (Win::w) and two f32 scales, s0 for weights 0..15 and s1 for
 * weights 16..31, so that
 *
 *     dequantized weight j  =  (j < 16 ? s0 : s1) * (float)int8_j
 *
 * is the exact f32 value of weights/gguf_dequant.rs (the CPU oracle). The matvec multiplies
 * the int8 weights against the int8 activation quants with __dp4a and applies the scales in
 * f32; the device dequant test applies them to the int8 weights directly.
 *
 * The scale expressions use __fmul_rn/__fadd_rn in the oracle's evaluation order, so the
 * compiler cannot contract them and the device value equals the CPU value bit for bit.
 *
 * A type is a struct with
 *   kBlockBytes   bytes of one block on disk
 *   kWinPerBlock  windows per block (1 for the 32-value blocks, 8 for the 256-value blocks)
 *   kTabWords     32-bit words of the lookup table the kernel stages into shared memory
 *   kHostTabWords words the host passes (kTabWords; Q2_0 has no table: the kernel gets a
 *                 4-word zero dummy)
 *   Raw           the bytes of one window as loaded from global memory
 *   load(block, wi)        loads window wi of the block; issues only global loads
 *   decode(raw, tab, win)  decodes without touching global memory; tab is the shared copy of
 *                          the table the host passed by value
 *
 * Lookup tables are never __constant__ storage (see iq4_nl_values.cuh and
 * constant_bank_guard.rs): the host passes the table words from weights/gguf_tables.rs as a
 * by-value kernel argument and the kernel copies them to shared memory once per CTA. The
 * sign byte of the IQ2/IQ3 families is not a table: the oracle's KSIGNS_IQ2XS[i] is i with
 * the parity of its seven bits in bit 7 (pinned by a host test), computed with __popc.
 *
 * Every grid byte of the IQ2 and IQ3 grids is in 1..=62, which the sign step relies on: it
 * negates four packed bytes as (g ^ m) + (m & 0x01010101) with m = 0xFF in the negated bytes,
 * and the +1 never carries out of a byte because (~g) is at most 254. A host test scans the
 * grids for that.
 */
#ifndef INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH
#define INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH

#include <cuda_fp16.h>

#include <stdint.h>
#include <string.h>

namespace gguf {

struct Win {
    int w[8];
    float s0;
    float s1;
};

/* GGUF type ids (ggml_type values) the in-house matvec serves. */
enum {
    kIq2Xxs = 16,
    kIq2Xs = 17,
    kIq3Xxs = 18,
    kIq4Nl = 20,
    kIq3S = 21,
    kIq2S = 22,
    kIq4Xs = 23,
    kQ2_0 = 42,
};

__device__ __forceinline__ unsigned rd16(const unsigned char *p) {
    unsigned short v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Little-endian u32 from a 2-byte-aligned address. */
__device__ __forceinline__ unsigned rd32(const unsigned char *p) {
    return rd16(p) | (rd16(p + 2) << 16);
}

__device__ __forceinline__ float rd_half(const unsigned char *p) {
    __half h;
    memcpy(&h, p, sizeof(h));
    return __half2float(h);
}

/* The oracle's d * (0.5 + n) * q, evaluated left to right without contraction. */
__device__ __forceinline__ float half_step(float d, unsigned n, float q) {
    return __fmul_rn(__fmul_rn(d, __fadd_rn(0.5f, (float)n)), q);
}

/* Bytes 0xFF where bit j (0..3) of nib is set. */
__device__ __forceinline__ unsigned sign_mask4(unsigned nib) {
    return ((nib * 0x00204081u) & 0x01010101u) * 0xFFu;
}

/* The four packed bytes of g, negated where the sign mask m is set. */
__device__ __forceinline__ int apply_signs(unsigned g, unsigned nib) {
    const unsigned m = sign_mask4(nib);
    return (int)((g ^ m) + (m & 0x01010101u));
}

/* The eight sign bits of a 7-bit sign index: bit 7 is the parity of the seven bits. */
__device__ __forceinline__ unsigned sign_byte(unsigned s7) {
    return s7 | ((__popc(s7) & 1u) << 7);
}

/* Four table entries from four nibbles. sel holds nibble j in bits 4j..4j+3 (j = 0..3), t0..t3
 * the sixteen table bytes as four words. Output byte j is table byte nibble_j. PRMT reads
 * its selector as four nibbles whose low three bits pick one of eight source bytes, so the
 * first two PRMTs resolve the low three bits against table bytes 0..7 and 8..15, and the last
 * picks between the two results by bit 3. */
__device__ __forceinline__ int lookup16(unsigned sel, unsigned t0, unsigned t1, unsigned t2,
                                        unsigned t3) {
    const unsigned low3 = sel & 0x7777u;
    const unsigned a = __byte_perm(t0, t1, low3);
    const unsigned b = __byte_perm(t2, t3, low3);
    const unsigned pick = ((sel >> 1) & 0x4444u) | 0x3210u;
    return (int)__byte_perm(a, b, pick);
}

/* The four low nibbles of the four bytes of q, as one 16-bit selector (byte i -> nibble i). */
__device__ __forceinline__ unsigned low_nibbles(unsigned q) {
    const unsigned x = q & 0x0F0F0F0Fu;
    const unsigned t = (x | (x >> 4)) & 0x00FF00FFu;
    return (t | (t >> 8)) & 0xFFFFu;
}

/* Windows of 16 code bytes holding two nibbles each (IQ4_NL, IQ4_XS): weight j (0..15) is
 * the table entry of the low nibble of byte j, weight 16 + j the entry of its high nibble. */
__device__ __forceinline__ void decode_nibbles(const unsigned (&q)[4], const uint32_t *tab,
                                               Win &out) {
    const unsigned t0 = tab[0], t1 = tab[1], t2 = tab[2], t3 = tab[3];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        out.w[r] = lookup16(low_nibbles(q[r]), t0, t1, t2, t3);
        out.w[4 + r] = lookup16(low_nibbles(q[r] >> 4), t0, t1, t2, t3);
    }
}

/* IQ4_NL: 18-byte block, f16 d then 16 code bytes. */
struct Iq4Nl {
    static constexpr int kBlockBytes = 18;
    static constexpr int kWinPerBlock = 1;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned q[4];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int) {
        Raw r;
        r.d = rd_half(blk);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.q[i] = rd32(blk + 2 + 4 * i);
        }
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        decode_nibbles(r.q, tab, o);
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* IQ4_XS: 136-byte super-block of eight windows: f16 d, u16 scales_h, scales_l[4], 128 code
 * bytes. Window ib has the 6-bit scale (scales_l nibble ib | 2 bits of scales_h) - 32. */
struct Iq4Xs {
    static constexpr int kBlockBytes = 136;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned q[4];
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        const unsigned lo = (blk[4 + (wi >> 1)] >> (4 * (wi & 1))) & 0x0Fu;
        const unsigned hi = (rd16(blk + 2) >> (2 * wi)) & 3u;
        r.scale = lo | (hi << 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r.q[i] = *reinterpret_cast<const unsigned *>(blk + 8 + 16 * wi + 4 * i);
        }
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        decode_nibbles(r.q, tab, o);
        const float dl = __fmul_rn(r.d, (float)((int)r.scale - 32));
        o.s0 = dl;
        o.s1 = dl;
    }
};

/* Q2_0: 18-byte block of 64 weights, f16 d then 64 two-bit codes (four per byte, low bits
 * first); the weight is (code - 1) * d. Window wi is the 8 code bytes at 2 + 8 * wi. The four
 * codes map to the bytes -1, 0, 1, 2 with one PRMT. */
struct Q2_0 {
    static constexpr int kBlockBytes = 18;
    static constexpr int kWinPerBlock = 2;
    static constexpr int kTabWords = 4;
    static constexpr int kHostTabWords = 0;
    struct Raw {
        unsigned q[4];
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned lo = rd16(blk + 2 + 8 * wi + 4 * i);
            const unsigned hi = rd16(blk + 4 + 8 * wi + 4 * i);
            r.q[i] = lo | (hi << 16);
        }
        r.q[2] = 0;
        r.q[3] = 0;
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *, Win &o) {
        const unsigned codes = 0x020100FFu; /* bytes -1, 0, 1, 2 */
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned word = r.q[i];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const unsigned v = (word >> (8 * b)) & 0xFFu;
                const unsigned sel = (v & 0x03u) | ((v & 0x0Cu) << 2) | ((v & 0x30u) << 4) |
                                     ((v & 0xC0u) << 6);
                o.w[4 * i + b] = (int)__byte_perm(codes, 0u, sel);
            }
        }
        o.s0 = r.d;
        o.s1 = r.d;
    }
};

/* IQ2_XXS: 66-byte super-block of eight windows: f16 d then, per window, four grid indices
 * (one byte each) and a word holding four 7-bit sign indices and a 4-bit scale in its top
 * nibble. Each grid entry is eight bytes. */
struct Iq2Xxs {
    static constexpr int kBlockBytes = 66;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 512;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned a0, a1;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.a0 = rd32(blk + 2 + 8 * wi);
        r.a1 = rd32(blk + 6 + 8 * wi);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[(r.a0 >> (8 * l)) & 0xFFu];
            const unsigned sg = sign_byte((r.a1 >> (7 * l)) & 127u);
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        const float db = half_step(r.d, r.a1 >> 28, 0.25f);
        o.s0 = db;
        o.s1 = db;
    }
};

/* IQ2_XS: 74-byte super-block: f16 d, 32 u16 (9-bit grid index, 7-bit sign index), eight
 * scale bytes. Window ib reads four of the u16 and scale byte ib, whose low nibble scales
 * weights 0..15 and high nibble weights 16..31. */
struct Iq2Xs {
    static constexpr int kBlockBytes = 74;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 1024;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned v0, v1;
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.v0 = rd32(blk + 2 + 8 * wi);
        r.v1 = rd32(blk + 6 + 8 * wi);
        r.scale = blk[66 + wi];
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned v = ((l < 2 ? r.v0 : r.v1) >> (16 * (l & 1))) & 0xFFFFu;
            const uint2 g = grid[v & 511u];
            const unsigned sg = sign_byte(v >> 9);
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        o.s0 = half_step(r.d, r.scale & 0x0Fu, 0.25f);
        o.s1 = half_step(r.d, r.scale >> 4, 0.25f);
    }
};

/* IQ2_S: 82-byte super-block: f16 d, 32 grid index bytes, 32 sign bytes, eight bytes of
 * grid index high bits (two bits per index), eight scale bytes (nibbles as in IQ2_XS). */
struct Iq2S {
    static constexpr int kBlockBytes = 82;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 2048;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned qs, signs;
        unsigned qh, scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.qs = rd32(blk + 2 + 4 * wi);
        r.signs = rd32(blk + 34 + 4 * wi);
        r.qh = blk[66 + wi];
        r.scale = blk[74 + wi];
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
        const uint2 *grid = reinterpret_cast<const uint2 *>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = ((r.qs >> (8 * l)) & 0xFFu) | ((r.qh << (8 - 2 * l)) & 0x300u);
            const uint2 g = grid[idx];
            const unsigned sg = (r.signs >> (8 * l)) & 0xFFu;
            o.w[2 * l] = apply_signs(g.x, sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(g.y, sg >> 4);
        }
        o.s0 = half_step(r.d, r.scale & 0x0Fu, 0.25f);
        o.s1 = half_step(r.d, r.scale >> 4, 0.25f);
    }
};

/* IQ3_XXS: 98-byte super-block: f16 d, 64 grid index bytes, then per window a word with four
 * 7-bit sign indices and a 4-bit scale. Each grid entry is four bytes; a window reads eight
 * index bytes. */
struct Iq3Xxs {
    static constexpr int kBlockBytes = 98;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 256;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned q0, q1;
        unsigned a;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.q0 = rd32(blk + 2 + 8 * wi);
        r.q1 = rd32(blk + 6 + 8 * wi);
        r.a = rd32(blk + 66 + 4 * wi);
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = (l < 2 ? r.q0 : r.q1) >> (16 * (l & 1));
            const unsigned sg = sign_byte((r.a >> (7 * l)) & 127u);
            o.w[2 * l] = apply_signs(tab[idx & 0xFFu], sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(tab[(idx >> 8) & 0xFFu], sg >> 4);
        }
        const float db = half_step(r.d, r.a >> 28, 0.5f);
        o.s0 = db;
        o.s1 = db;
    }
};

/* IQ3_S: 110-byte super-block: f16 d, 64 grid index bytes, eight bytes of index bit 8 (one
 * byte per window), 32 sign bytes, four bytes of scale nibbles (window ib: nibble ib & 1 of
 * byte ib >> 1; scale 1 + 2 * nibble). */
struct Iq3S {
    static constexpr int kBlockBytes = 110;
    static constexpr int kWinPerBlock = 8;
    static constexpr int kTabWords = 512;
    static constexpr int kHostTabWords = kTabWords;
    struct Raw {
        unsigned q0, q1;
        unsigned qh, signs;
        unsigned scale;
        float d;
    };
    static __device__ __forceinline__ Raw load(const unsigned char *blk, int wi) {
        Raw r;
        r.d = rd_half(blk);
        r.q0 = rd32(blk + 2 + 8 * wi);
        r.q1 = rd32(blk + 6 + 8 * wi);
        r.qh = blk[66 + wi];
        r.signs = rd32(blk + 74 + 4 * wi);
        r.scale = (blk[106 + (wi >> 1)] >> (4 * (wi & 1))) & 0x0Fu;
        return r;
    }
    static __device__ __forceinline__ void decode(const Raw &r, const uint32_t *tab, Win &o) {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const unsigned idx = (l < 2 ? r.q0 : r.q1) >> (16 * (l & 1));
            const unsigned i1 = (idx & 0xFFu) | ((r.qh << (8 - 2 * l)) & 256u);
            const unsigned i2 = ((idx >> 8) & 0xFFu) | ((r.qh << (7 - 2 * l)) & 256u);
            const unsigned sg = (r.signs >> (8 * l)) & 0xFFu;
            o.w[2 * l] = apply_signs(tab[i1], sg & 0xFu);
            o.w[2 * l + 1] = apply_signs(tab[i2], sg >> 4);
        }
        const float db = __fmul_rn(r.d, (float)(1 + 2 * (int)r.scale));
        o.s0 = db;
        o.s1 = db;
    }
};

}  // namespace gguf

#endif /* INFERENCE_CUDA_RESIDENT_GGUF_DECODE_CUH */
