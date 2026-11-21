/*
 * Launch entry of block_ops: the row operations of a hybrid decoder block that no other
 * resident kernel covers: the four-stream mixed residual, a causal short convolution, a
 * delta-rule scan with a decay per key channel, absorbed latent attention over one shared
 * key/value head, the SiTU activation with its q8_1 quantize, and a biased sigmoid router.
 * Launch-only: nothing here allocates, copies or synchronizes. Every operation works on
 * `n` token rows at once, so decode (n = 1) and prefill chunks share the entry.
 *
 * One generic argument block carries up to 12 device ranges, 12 integers and 4 floats; the
 * meaning of each slot is fixed per operation and listed beside the operation code below.
 * The entry checks every slot an operation uses (non-NULL, 4-byte aligned, at least the
 * bytes the integers imply) before any CUDA call and returns INFERENCE_CUDA_INVALID_ARGUMENT
 * otherwise. Ranges the operation writes may not share a byte with any other slot.
 */
#ifndef INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H
#define INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_BLOCK_OPS_PTRS 12
#define INFERENCE_CUDA_BLOCK_OPS_INTS 12
#define INFERENCE_CUDA_BLOCK_OPS_FLOATS 4

/* Operation codes. Slots are `p0..p11`, integers `i0..i11`, floats `f0..f3`.
 *
 * 1 HC_READ   mixed-residual read of one site. p0 resid f32 [n][4*h]; p1 fn f16 [17][4*h];
 *             p2 base f32 [17]; p3 scale f32 [4]; p4 branch f32 [n][h] (W); p5 post f32
 *             [n][4] (W); p6 comb f32 [n][16] (W). i0 n; i1 h. f0 epsilon; f1 magnitude.
 * 2 HC_APPLY  p0 resid f32 [n][4*h]; p1 branch f32 [n][h]; p2 post f32 [n][4]; p3 comb f32
 *             [n][16]; p4 out f32 [n][4*h] (W). i0 n; i1 h.
 * 3 RMS_ROWS  p0 x f32 [n][ld_x]; p1 weight f32 [d]; p2 y f32 [n][d] (W, optional); p3 q8_1
 *             rows (W, optional, row stride q8_1_row_bytes(d)). i0 n; i1 d; i2 ld_x.
 *             f0 epsilon.
 * 4 HC_MEAN   p0 resid f32 [rows][4*h]; p1 out f32 [h] (W). i0 row; i1 h.
 * 5 CONV_SILU p0 x f32 [n][c]; p1 weight f32 [c][4]; p2 hist f32 [3][c] (RW); p3 y f32
 *             [n][c] (W). i0 n; i1 c.
 * 6 RDA_SCAN  p0 q; p1 k; p2 v; p3 g1; each f32 [n][heads*hd]; p4 dt_bias f32 [heads*hd];
 *             p5 a f32 [heads]; p6 beta f32 [n][heads]; p7 delta f32 [n][heads]; p8
 *             norm_w f32 [hd]; p9 g2 f32 [n][heads*hd]; p10 state f32 [heads][hd][hd]
 *             (RW); p11 out f32 [n][heads*hd] (W). i0 n; i1 heads; i2 hd (128).
 *             f0 gate lower bound; f1 norm epsilon.
 * 7 MLA_Q_ABSORB p0 qb f32 [n][heads*(nope+rope)]; p1 wk_b f32 [heads][rank][nope];
 *             p2 qeff f32 [n][heads][rank+rope] (W). i0 n; i1 heads; i2 nope; i3 rope;
 *             i4 rank.
 * 8 MLA_KV_PREP p0 kv f32 [n][rank+rope]; p1 norm_w f32 [rank]; p2 cache f16
 *             [cells][rank+rope] (W). i0 n; i1 rank; i2 rope; i3 pos0; i4 cells.
 *             f0 epsilon.
 * 9 MLA_ATTN  p0 qeff f32 [n][heads][rank+rope]; p1 cache f16 [cells][rank+rope]; p2
 *             part f32 (W). i0 n; i1 heads (multiple of 8); i2 rank; i3 rope; i4 pos0;
 *             i5 splits; i6 keys_per_split (multiple of 32); f0 score scale. rank is 512
 *             and rope 64.
 * 10 MLA_REDUCE p0 part f32; p1 mix f32 [n][heads][rank] (W). i0 n; i1 heads; i2 rank;
 *             i3 splits.
 * 11 MLA_PROJECT p0 mix f32 [n][heads][rank]; p1 wv_b f32 [heads][v][rank]; p2 out f32
 *             [n][heads*v] (W). i0 n; i1 heads; i2 rank; i3 v.
 * 12 SITU_QUANT p0 gate f32 rows at ld_gate; p1 up f32 rows at ld_up; p2 q8_1 rows (W,
 *             optional); p3 y f32 [rows][w] (W, optional; one of p2 and p3 is present).
 *             i0 rows; i1 w; i2 ld_gate; i3 ld_up. f0 beta; f1 linear beta.
 * 13 GATED_QUANT p0 x f32 [rows][w]; p1 gate f32 [rows][w] (optional); p2 q8_1 rows (W,
 *             optional); p3 y f32 [rows][w] (W, optional; one of p2 and p3 is present).
 *             i0 rows; i1 w.
 * 14 MOE_ROUTE p0 x f32 [n][h]; p1 gate_w f32 [e][h]; p2 bias f32 [e]; p3 ids i32 [n][k]
 *             (W); p4 weights f32 [n][k] (W). i0 n; i1 h; i2 e (<= 256); i3 k (<= 8);
 *             i4 normalize. f0 smallest weight sum.
 * 15 FILL     p0 f32 [i0] (W). i0 count. f0 value.
 */
enum {
    INFERENCE_CUDA_BLOCK_OP_HC_READ = 1,
    INFERENCE_CUDA_BLOCK_OP_HC_APPLY = 2,
    INFERENCE_CUDA_BLOCK_OP_RMS_ROWS = 3,
    INFERENCE_CUDA_BLOCK_OP_HC_MEAN = 4,
    INFERENCE_CUDA_BLOCK_OP_CONV_SILU = 5,
    INFERENCE_CUDA_BLOCK_OP_RDA_SCAN = 6,
    INFERENCE_CUDA_BLOCK_OP_MLA_Q_ABSORB = 7,
    INFERENCE_CUDA_BLOCK_OP_MLA_KV_PREP = 8,
    INFERENCE_CUDA_BLOCK_OP_MLA_ATTN = 9,
    INFERENCE_CUDA_BLOCK_OP_MLA_REDUCE = 10,
    INFERENCE_CUDA_BLOCK_OP_MLA_PROJECT = 11,
    INFERENCE_CUDA_BLOCK_OP_SITU_QUANT = 12,
    INFERENCE_CUDA_BLOCK_OP_GATED_QUANT = 13,
    INFERENCE_CUDA_BLOCK_OP_MOE_ROUTE = 14,
    INFERENCE_CUDA_BLOCK_OP_FILL = 15
};

typedef struct inference_cuda_block_ops_args {
    int op;
    int ints[INFERENCE_CUDA_BLOCK_OPS_INTS];
    float floats[INFERENCE_CUDA_BLOCK_OPS_FLOATS];
    void *ptr[INFERENCE_CUDA_BLOCK_OPS_PTRS]; /* NULL for an unused slot */
    size_t bytes[INFERENCE_CUDA_BLOCK_OPS_PTRS]; /* bytes of the range behind each slot */
} inference_cuda_block_ops_args;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_block_ops_args) == 264 && offsetof(inference_cuda_block_ops_args, ints) == 4 && offsetof(inference_cuda_block_ops_args, floats) == 52 && offsetof(inference_cuda_block_ops_args, ptr) == 72 && offsetof(inference_cuda_block_ops_args, bytes) == 168, "layout");
#else
_Static_assert(sizeof(inference_cuda_block_ops_args) == 264 && offsetof(inference_cuda_block_ops_args, ints) == 4 && offsetof(inference_cuda_block_ops_args, floats) == 52 && offsetof(inference_cuda_block_ops_args, ptr) == 72 && offsetof(inference_cuda_block_ops_args, bytes) == 168, "layout");
#endif

/* One launch on stream (must not be NULL). Returns INFERENCE_CUDA_INVALID_ARGUMENT before
 * any CUDA call for a bad argument, else the launch status. */
int inference_cuda_block_ops_launch(const inference_cuda_block_ops_args *args, void *stream,
                                    int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_BLOCK_OPS_H */
