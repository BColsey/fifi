/*
 * The mHC stream routing of the SenseNova Loop family (native/resident/mhc_route.h).
 * Launch-only, like the entries of abi_resident.h. Include after abi.h in C++; Rust mirrors the
 * structs in native/resident/mhc_route.rs (child module of native/resident.rs).
 *
 * Three small kernels over the two mHC residual streams `a` and `b` (each [hidden] f32, held
 * adjacent as [a || b] by the caller):
 *   route   reads the 8 router logits (the router matvec of the unit-RMS'd [a || b] row plus its
 *           bias) and writes the routed input row, the residual gains and the Sinkhorn-normalized
 *           2x2 mixing matrix;
 *   merge   applies one routing decision to (a, b) with a branch output, in place;
 *   average writes 0.5 * (a + b), the value the trunk takes back at the end of an mHC visit.
 */
#ifndef INFERENCE_CUDA_MHC_ROUTE_H
#define INFERENCE_CUDA_MHC_ROUTE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The routing math, per the CPU reference (family/qwen/sensenova/step.rs):
 *   input[j] = (a[j] * sigmoid(l0) + b[j] * sigmoid(l1)) * pre_scale
 *   post     = (2 * sigmoid(l2), 2 * sigmoid(l3))
 *   mix      = identity when fixed_identity, else exp of `iterations` Sinkhorn passes over
 *              (l4, l5, l6, l7): normalize(0,1); normalize(2,3); normalize(0,2); normalize(1,3),
 *              where normalize(i, j) sets m[i] = -softplus(-(m[i]-m[j])) and
 *              m[j] = -softplus(m[i]-m[j]) with the ORIGINAL difference.
 * a and b are [hidden] f32, input is [hidden] f32, logits are 8 f32, route is 6 f32 written:
 * post at 0..2 and mix at 2..6. hidden is a positive multiple of 256 (grid math) and at most
 * 2^20. bias is NULL or 8 f32 added to the logits once, before any sigmoid or Sinkhorn step.
 * Returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 *
 * rows is 1 (one decode step) or a prefill chunk's row count, at most 4096. Every buffer then
 * holds that many rows back to back (logits 8 and route 6 floats per row) except a and b: they
 * are the halves of one ld_ab-wide [a || b] row with ld_ab >= 2 * hidden, so consecutive rows
 * step ld_ab. The entry cannot see allocation extents or overlap: the launch boundary proves
 * that no element of a's rows is an element of b's rows (merge writes both in place). */
typedef struct inference_cuda_mhc_route_args {
    const float *logits;
    const float *bias; /* NULL, or 8 f32 added to the logits before every use */
    const float *a;
    const float *b;
    float *input;
    float *route; /* [rows][6]: [2] post then [4] mix */
    int hidden;
    float pre_scale;
    int iterations;
    int fixed_identity;
    int rows;
    int ld_ab;
} inference_cuda_mhc_route_args;
int inference_cuda_mhc_route_launch(const inference_cuda_mhc_route_args *args, void *stream,
                                    int *out_stage);

/* merge: a and b in place with the branch output; route is the 6-float cell of the matching
 * route call (post at 0..2, mix at 2..6). */
typedef struct inference_cuda_mhc_merge_args {
    float *a;
    float *b;
    const float *branch; /* [rows][hidden] f32 */
    const float *route;  /* [rows][6] */
    int hidden;
    int rows;
    int ld_ab;
} inference_cuda_mhc_merge_args;
int inference_cuda_mhc_merge_launch(const inference_cuda_mhc_merge_args *args, void *stream,
                                    int *out_stage);

/* average: out[j] = 0.5f * (a[j] + b[j]). */
typedef struct inference_cuda_mhc_average_args {
    const float *a;
    const float *b;
    float *out; /* [rows][hidden] f32 */
    int hidden;
    int rows;
    int ld_ab;
} inference_cuda_mhc_average_args;
int inference_cuda_mhc_average_launch(const inference_cuda_mhc_average_args *args, void *stream,
                                      int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
