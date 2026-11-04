/* The block selection of Qwen3.8 Flash-Next QSA attention, per-kernel header
 * (native/resident/qsa_select.h). Included by qsa_select.cu and read by the Rust FFI mirror;
 * nothing else.
 *
 * For the token at position p, with blocks = (p + 1) / ratio completed blocks and
 * take = min(blocks, budget_blocks), the selection is the `take` blocks with the largest scores
 * (qsa_score's output), a tie of equal f32 scores going to the lower block, as the host's
 * ranking does. The host ranks f64 sums of the same dots, so two blocks whose scores differ by a
 * few f32 ulps at the budget boundary may rank the other way: a test can demand the host's
 * selection only on data without such near ties. The entry writes the chosen indices to
 * blocks[0 .. take) in ascending block order; the attention reads the
 * positions of those blocks and then the open tail, positions blocks * ratio .. p, and derives
 * `take` from the position itself. A score is compared by its f32 value; a negative or NaN score
 * (qsa_score writes neither) counts as +0.
 *
 * The position is read from the device word `pos` when it is given (a decode graph replays it),
 * else it is the launch argument pos0. Poison rule (attn_prep's): if it is negative or p + 1 >
 * cap_blocks * ratio (64-bit), nothing is written and status bits 4 are OR-ed into *status by one
 * thread when status is given.
 *
 * Limits: ratio in [1, 64]; cap_blocks in [1, 2^24]; budget_blocks in [1, cap_blocks]. scores
 * is f32 [cap_blocks], blocks is i32 [budget_blocks]; every range is 4-byte aligned and the
 * written ranges (blocks, status) overlap no other range. The entry enqueues one kernel on
 * stream (not NULL) and returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad
 * argument. */
#ifndef INFERENCE_CUDA_RESIDENT_QSA_SELECT_H
#define INFERENCE_CUDA_RESIDENT_QSA_SELECT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_qsa_select_args {
    const void *scores; /* f32 [cap_blocks], [0, (p + 1) / ratio) read */
    void *blocks;       /* i32 [budget_blocks], [0, take) written */
    const void *pos;    /* device i32 position, or NULL for pos0 */
    void *status;       /* device i32, or NULL */
    int ratio;
    int cap_blocks;
    int budget_blocks;
    int pos0;
} inference_cuda_qsa_select_args;

int inference_cuda_qsa_select_launch(const inference_cuda_qsa_select_args *args, void *stream,
                                     int *out_stage);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <cstddef>
static_assert(sizeof(inference_cuda_qsa_select_args) == 48 &&
                  offsetof(inference_cuda_qsa_select_args, status) == 24 &&
                  offsetof(inference_cuda_qsa_select_args, ratio) == 32 &&
                  offsetof(inference_cuda_qsa_select_args, pos0) == 44,
              "inference_cuda_qsa_select_args layout");
#endif

#endif
