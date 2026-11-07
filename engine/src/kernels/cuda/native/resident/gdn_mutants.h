/* Deliberately wrong variants of the conv prefill kernel (conv_silu_rows_k) and the narrow-block scan kernel
 * (gdn_scan_seq_cols_k), for the mutation tests only. They exist when the build defines
 * INFERENCE_CUDA_GDN_MUTANTS (Cargo feature cuda-gdn-mutants); without it every GDN_MUT(id) is the constant false,
 * so no kernel contains a mutant branch and the compiled kernels are the ones without this header.
 *
 * One mutant is active at a time, chosen at run time by inference_cuda_gdn_mutant_select(id); 0 is none. The id is
 * read at launch and travels to the kernels as an argument. The ids are mirrored by kernels/cuda/gdn_mutants.rs.
 *
 *   conv:  1 plain expression (the compiler's own contraction)   2 outgoing window words 0 and 1 swapped
 *          3 incoming state rows read in reverse                 4 outgoing window written by every CTA row but the first
 *          5 the last row of each tile not stored
 *   scan:  6 column offset of a block uses 32 per z step         7 v offset of a block uses 32 per z step
 *          8 v staged for the first slot of a warp only          9 grid z uses 32 columns per block
 *          10 sk sum combined in another order                   11 width 16: head 0 state stored as zero
 *   Rust (kernels/cuda/gdn_mutants.rs only): 12 conv routing ignored   13 conv threshold 1   14 conv "1" selects 16
 *          15 conv value not trimmed   16 conv accepts "8"   17 scan accepts "4"   18 scan value not trimmed. */
#ifndef INFERENCE_CUDA_RESIDENT_GDN_MUTANTS_H
#define INFERENCE_CUDA_RESIDENT_GDN_MUTANTS_H

#ifdef INFERENCE_CUDA_GDN_MUTANTS
#include <atomic>
extern std::atomic<int> g_gdn_mutant;
static inline int gdn_mutant_current() { return g_gdn_mutant.load(std::memory_order_relaxed); }
extern "C" int inference_cuda_gdn_mutant_select(int id);
#define GDN_MUTANT_PARAM , int mutant
#define GDN_MUTANT_ARG , gdn_mutant_current()
#else
static inline int gdn_mutant_current() { return 0; }
#define GDN_MUTANT_PARAM
#define GDN_MUTANT_ARG
#endif

#endif
