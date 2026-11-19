/*
 * The per-type block decoders of the in-house dequantizing GEMM (gemm_dq.cu). A decoder is a
 * struct with the block geometry of its GGUF type and `decode_chunk`, which writes the 64
 * dequantized halves of one 64-value slice (chunk) of one on-disk block, straight into the
 * shared-memory weight tile. The layouts are the GGUF block layouts, read out in this
 * engine's own structure (the CPU oracle that checks them is weights/gguf_dequant.rs).
 *
 * A new quant type adds a decoder here, not a kernel: gemm_dq.cu is generic over the decoder.
 */
#ifndef INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH
#define INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH

#include <cuda_fp16.h>

#include <stdint.h>

/* The IQ4_NL decode table rides a by-value kernel argument whose carrier and host
 * factory live in the shared iq4_nl_values.cuh — never a __constant__ definition in
 * any TU of this .so: the at-load constant-bank layout aliases per-TU initialized
 * constants (observed on the 3090, 2026-11-18: this TU's uniquely named table read
 * nvfp4 kE2M1's initializer, and renaming and external linkage both still aliased).
 * Kernel parameters are copied at launch, outside that machinery. The values stay
 * pinned to weights/gguf_tables.rs KVALUES_IQ4NL by the matvec host pin test, and
 * native/constant_bank_guard.rs keeps new __constant__ tables out of the tree. */
#include "iq4_nl_values.cuh"

namespace gemm_dq {

__device__ __forceinline__ float half_at(const unsigned char *p) {
    __half raw;
    memcpy(&raw, p, sizeof(raw));
    return __half2float(raw);
}

/* Q5_K block layout (176 bytes for 256 values, split into 8 groups of 32): an f16 value
 * scale at +0, an f16 offset scale at +2, 12 packed 6-bit group factors at +4, 32 high-bit
 * bytes at +16, 128 4-bit code bytes at +48. Group g carries a 6-bit multiplier and 6-bit
 * offset, and value i of the block weighs
 *   (value_scale * mult[g]) * (code_i + 16 * high_i) - offset_scale * offs[g].
 * The factor bytes unpack as: groups 0..3 take the low 6 bits of bytes [g] (multiplier)
 * and [g+4] (offset); groups 4..7 take the low nibble of byte [g+4] spliced with the top
 * 2 bits of byte [g-4] (multiplier), and the high nibble of byte [g+4] spliced with the
 * top 2 bits of byte [g] (offset). */
__device__ __forceinline__ void q5k_group_factors(const unsigned char *raw, float *mult,
                                                  float *offs) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        mult[g] = (float)(raw[g] & 63);
        offs[g] = (float)(raw[g + 4] & 63);
    }
    #pragma unroll
    for (int g = 4; g < 8; ++g) {
        mult[g] = (float)((raw[g + 4] & 0x0f) | ((raw[g - 4] >> 6) << 4));
        offs[g] = (float)((raw[g + 4] >> 4) | ((raw[g] >> 6) << 4));
    }
}

/* Chunk c (0..3) of a Q5_K block carries values [64c, 64c+64): the low code nibbles of
 * bytes [48+32c, 48+32c+32) with high bit 2c, then the high nibbles with high bit 2c+1,
 * each half against its own group pair from q5k_group_factors. A chunk is a quarter block,
 * so the row's chunk-c blocks start at block c/4. */
struct Q5kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 176;
    static const int kChunkBlocks = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* c is the raw 64-value chunk index along k (0..k/64); the slice inside this block is
     * c mod 4, because four chunks share one 256-value block. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst, const float *iq4nl_kv) {
        c &= 3;
        const float d = half_at(block);
        const float dmin = half_at(block + 2);
        float mult[8];
        float offs[8];
        q5k_group_factors(block + 4, mult, offs);
        const float mul1 = d * mult[2 * c];
        const float off1 = dmin * offs[2 * c];
        const float mul2 = d * mult[2 * c + 1];
        const float off2 = dmin * offs[2 * c + 1];
        const unsigned bit1 = 1u << (2 * c);
        const unsigned bit2 = 2u << (2 * c);
        const unsigned char *high = block + 16;
        const unsigned char *codes = block + 48 + 32 * c;
        for (int l = 0; l < 32; ++l) {
            const unsigned char code = codes[l];
            const unsigned char h = high[l];
            const float v1 = (float)((code & 0x0f) + ((h & bit1) ? 16 : 0));
            const float v2 = (float)((code >> 4) + ((h & bit2) ? 16 : 0));
            dst[l] = __float2half(fmaf(mul1, v1, -off1));
            dst[32 + l] = __float2half(fmaf(mul2, v2, -off2));
        }
    }
};

/* Q4_0 block layout (18 bytes for 32 values): an f16 scale d at +0, then 16 code bytes at
 * +2 whose byte j holds the codes of values j (low nibble) and j + 16 (high nibble); value i
 * of the block is d * (code_i - 8). A 64-value chunk is exactly two whole blocks, so the
 * row's chunk-c blocks start at block 2c. */
struct Q40Decoder {
    static const int kQK = 32;
    static const int kBlockBytes = 18;
    static const int kChunkBlocks = 2;

    __device__ static int chunk_block_off(int c) { return 2 * c; }

    /* A block's 32 values, in GGUF order: value j (0..16) is the LOW nibble of
     * qs[j], value j + 16 the HIGH nibble (the layout weights/gguf_dequant.rs
     * q4_0 decodes and the ggml reference pins). */
    __device__ static void decode_chunk(const unsigned char *chunk, int c, __half *dst, const float *iq4nl_kv) {
        (void)c;
        for (int b = 0; b < 2; ++b) {
            const unsigned char *blk = chunk + 18 * b;
            const float d = half_at(blk);
            for (int j = 0; j < 16; ++j) {
                const unsigned char code = blk[2 + j];
                dst[32 * b + j] = __float2half(d * (float)((code & 0x0f) - 8));
                dst[32 * b + 16 + j] = __float2half(d * (float)((code >> 4) - 8));
            }
        }
    }
};

/* Q3_K block layout (110 bytes for 256 values), per the CPU oracle of
 * weights/gguf_dequant.rs (the ggml reference `dequantize_row_q3_K`): 32
 * hmask bytes at +0, 64 qs bytes at +32, 12 packed 6-bit sub-scale bytes at
 * +96, the f16 super scale at +108. A 256-value block is two 128-value
 * halves; each half reads qs[32*half..) with lane j (0..4) at bit pair 2*j,
 * groups of 16 at +16, and its hmask bit 4*half + j of hmask[r] (set
 * subtracts 0, clear 4); the
 * 16 sub-scales come from the aux shuffle (see `scale_of`) biased by 32. */
struct Q3kDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 110;
    static const int kChunkBlocks = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    /* One of the 16 packed 6-bit sub-scales of the ggml aux shuffle (the CPU
     * oracle's aux[0..3] expansion, addressed directly): scale `is` is byte
     * is&3 of row is>>2, low nibble of ks[(row&1)*4 + byte] at shift
     * 4*(row>>1), plus the 2-bit high field of ks[8+byte] at shift 2*row. */
    __device__ static int scale_of(const unsigned char *ks, int is) {
        const int row = is >> 2;
        const int byte = is & 3;
        const int lo = (ks[(row & 1) * 4 + byte] >> (4 * (row >> 1))) & 0x0f;
        const int hi = ((ks[8 + byte] >> (2 * row)) & 0x03) << 4;
        return lo | hi;
    }

    /* Chunk c carries block values [64c, 64c + 64) in the oracle's output
     * order: half = c >> 1 owns qs[32*half..); within the half, lane j covers
     * 32 values (qs shift 2*j, hmask bit 4*half + j: one mask byte carries
     * all eight lanes of both halves), groups of 16 at qs+16 and hm+16; a
     * CLEAR hmask bit subtracts 4. */
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst, const float *iq4nl_kv) {
        c &= 3;
        const float d = half_at(block + 108);
        const unsigned char *hm = block;
        const unsigned char *qs = block + 32 + 32 * (c >> 1);
        const unsigned char *ks = block + 96;
        #pragma unroll
        for (int s = 0; s < 64; ++s) {
            const int j = 2 * (c & 1) + (s >> 5);
            const int g2 = (s >> 4) & 1;
            const int l = s & 15;
            const int qv =
                (int)((qs[16 * g2 + l] >> (2 * j)) & 3) -
                ((hm[16 * g2 + l] & (1 << (4 * (c >> 1) + j))) == 0 ? 4 : 0);
            const float dl = d * (float)(scale_of(ks, (c >> 1) * 8 + 2 * j + g2) - 32);
            dst[s] = __float2half(dl * (float)qv);
        }
    }
};

/* IQ4_XS block layout (136 bytes for 256 values), per the CPU oracle of
 * weights/gguf_dequant.rs iq4_xs: an f16 super scale at +0, a u16 of 2-bit
 * high scale fields at +2, four bytes of 4-bit low scale nibbles at +4
 * (sub-block ib reads nibble ib%2 of byte ib/2), 128 qs bytes at +8.
 * Sub-block ib (0..7) covers values [32*ib, 32*ib+32): value j of the
 * sub-block reads the low nibble of qs[16*ib + j] (j < 16) or the high
 * nibble of qs[16*ib + j - 16], weighs d * (6-bit scale - 32) against the
 * shared value table (iq4_nl_values.cuh, the same 16 values IQ4_NL reads).
 * A 256-value block is four 64-value chunks: chunk c carries sub-blocks
 * 2c and 2c+1. */
struct Iq4xsDecoder {
    static const int kQK = 256;
    static const int kBlockBytes = 136;
    static const int kChunkBlocks = 1;

    __device__ static int chunk_block_off(int c) { return c >> 2; }

    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst, const float *iq4nl_kv) {
        c &= 3;
        const float d = half_at(block);
        unsigned short scales_h_raw;
        memcpy(&scales_h_raw, block + 2, sizeof(scales_h_raw));
        const unsigned int scales_h = scales_h_raw;
        const unsigned char *scales_l = block + 4;
        const unsigned char *qs = block + 8;
        #pragma unroll
        for (int s = 0; s < 64; ++s) {
            const int ib = 2 * c + (s >> 5);
            const int j = s & 31;
            const int lo = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0f;
            const int hi = (int)((scales_h >> (2 * ib)) & 3u);
            const float dl = d * (float)((lo | (hi << 4)) - 32);
            const unsigned char q = qs[16 * ib + (j & 15)];
            const int nib = j < 16 ? (q & 0x0f) : (q >> 4);
            dst[s] = __float2half(dl * iq4nl_kv[nib]);
        }
    }
};

}  // namespace gemm_dq

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH */
