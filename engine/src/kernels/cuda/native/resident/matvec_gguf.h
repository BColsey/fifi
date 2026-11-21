/*
 * Launch entries of the in-house GGUF decode matvec for the codebook and 2-bit types (IQ2_XXS,
 * IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_0): the dense matvec, the expert matvec
 * (one launch for the used slots of an expert stack, expert ids read on the device when the
 * kernel runs) and a device dequant used by the tests. The weights are the file's native
 * blocks (no repack, no vendored code); the activation is the shared quantize_q8_1 output.
 * Launch-only: nothing here allocates, copies or synchronizes.
 *
 * The lookup table of a type rides the launch BY VALUE (see gguf_decode.cuh): the caller
 * passes a host pointer to the table words of weights/gguf_tables.rs and their byte count, the
 * entry copies them into the kernel arguments. The byte count must be exactly the table size
 * of the type (0 for Q2_0).
 *
 * Every includer is a C++ TU of the native adapter, so the layouts are pinned with C++
 * static_asserts here and with the same numbers by the Rust mirror (matvec_gguf.rs).
 */
#ifndef INFERENCE_CUDA_RESIDENT_MATVEC_GGUF_H
#define INFERENCE_CUDA_RESIDENT_MATVEC_GGUF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_matvec_gguf_args {
    int type;         /* GGUF tensor type id: 16, 17, 18, 20, 21, 22, 23 or 42 */
    int k;            /* a positive multiple of the type's block size */
    int rows;         /* any positive count (the kernel owns whole rows) */
    int ncols;        /* 1..8 */
    const void *w;    /* native blocks [rows][k / block_size] */
    const void *x_q8; /* block_q8_1 [ncols][pad(k, 512) / 32], 36 bytes each */
    float *y;         /* f32 out [ncols][rows], column j at y + j * rows */
} inference_cuda_matvec_gguf_args;

/* One dense launch on stream (must not be NULL): y[j * rows + r] = sum_i x_q8[j][i] * w[r][i].
 * x_row_bytes is one activation row's stride (padded blocks x 36). Column j is computed from
 * x_q8 row j alone. Returns INFERENCE_CUDA_INVALID_ARGUMENT before any CUDA call for a bad
 * argument, else the launch status. */
int inference_cuda_matvec_gguf_launch(const inference_cuda_matvec_gguf_args *args,
                                      const void *table, size_t table_bytes,
                                      size_t x_row_bytes, void *stream, int *out_stage);

typedef struct inference_cuda_matvec_gguf_expert_args {
    int type;
    int k;
    int rows;
    int n_expert;
    int n_used;          /* 1..32 */
    int x_per_slot;      /* 0: one shared activation row, 1: one per slot */
    int y_slot_stride;   /* elements between slots, >= rows */
    int reserved;        /* zero */
    const void *w_stack; /* native blocks [n_expert][rows][k / block_size] */
    const void *ids;     /* device i32 [n_used], read when the kernel runs */
    const void *x_q8;    /* block_q8_1: one padded row (x_per_slot 0) or n_used rows */
    float *y;            /* f32 out [n_used][y_slot_stride], rows used per slot */
    const void *glu_gate; /* NULL, or the gate stack (same type, k, rows, n_expert as w_stack):
                             y = up * silu(gate) per element, w_stack being the UP stack;
                             x_per_slot must be 0 */
} inference_cuda_matvec_gguf_expert_args;

/* One expert launch on stream: slot s computes y[s * y_slot_stride .. + rows) against expert
 * ids[s]'s weights and activation row (s if x_per_slot else 0). An id outside [0, n_expert)
 * makes the kernel write the quiet NaN 0x7fc00000 to that slot's rows and read no weight. */
int inference_cuda_matvec_gguf_expert_launch(const inference_cuda_matvec_gguf_expert_args *args,
                                             const void *table, size_t table_bytes,
                                             size_t x_row_bytes, void *stream, int *out_stage);

/* Test support: decodes n_blocks blocks of type from w into f32 y (block_size values each)
 * through the same window decoders the matvec uses. */
int inference_cuda_dequant_gguf_launch(int type, const void *w, float *y, size_t n_blocks,
                                       const void *table, size_t table_bytes, void *stream,
                                       int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_matvec_gguf_args) == 40 &&
                  offsetof(inference_cuda_matvec_gguf_args, type) == 0 &&
                  offsetof(inference_cuda_matvec_gguf_args, k) == 4 &&
                  offsetof(inference_cuda_matvec_gguf_args, rows) == 8 &&
                  offsetof(inference_cuda_matvec_gguf_args, ncols) == 12 &&
                  offsetof(inference_cuda_matvec_gguf_args, w) == 16 &&
                  offsetof(inference_cuda_matvec_gguf_args, x_q8) == 24 &&
                  offsetof(inference_cuda_matvec_gguf_args, y) == 32,
              "inference_cuda_matvec_gguf_args layout");
static_assert(sizeof(inference_cuda_matvec_gguf_expert_args) == 72 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, type) == 0 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, k) == 4 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, rows) == 8 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, n_expert) == 12 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, n_used) == 16 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, x_per_slot) == 20 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, y_slot_stride) == 24 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, reserved) == 28 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, w_stack) == 32 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, ids) == 40 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, x_q8) == 48 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, y) == 56 &&
                  offsetof(inference_cuda_matvec_gguf_expert_args, glu_gate) == 64,
              "inference_cuda_matvec_gguf_expert_args layout");
#endif

#endif /* INFERENCE_CUDA_RESIDENT_MATVEC_GGUF_H */
