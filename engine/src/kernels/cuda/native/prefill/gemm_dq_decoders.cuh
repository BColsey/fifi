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
    __device__ static void decode_chunk(const unsigned char *block, int c, __half *dst) {
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
    __device__ static void decode_chunk(const unsigned char *chunk, int c, __half *dst) {
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

}  // namespace gemm_dq

#endif /* INFERENCE_CUDA_PREFILL_GEMM_DQ_DECODERS_CUH */
