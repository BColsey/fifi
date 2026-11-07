/* The per-step parameter block of the resident decode step, and the kernels that begin
 * a step, choose the greedy token and advance the position. One header per kernel file;
 * abi_resident.h stays closed.
 *
 * inference_cuda_step_params is the device image of StepParams: 32 bytes, 256-byte aligned in
 * the arena (the kernels need only 4-byte alignment). Other kernels take a 4-byte DeviceRange
 * at one field offset, not the struct:
 *   token_id  offset  0  read by the embedding gather; written by argmax (token chaining)
 *   pos       offset  4  read by attn_prep (rope angle)
 *   kv_len    offset  8  read by attn_decode; kv_len == pos + 1 once the step has appended
 *   kv_slot   offset 12  reserved; equals pos
 *   status    offset 16  flags by mask: 1 router fault; 2 non-finite logits row (argmax);
 *                        4 invalid step state (step_begin; a defect when it appears)
 * A valid state is 0 <= pos <= 2^30 - 2, kv_len == pos + 1, kv_slot == pos.
 * token_id and status stay the device source of truth; the result pair of argmax is a copy.
 *
 * All three launch entries are launch-only and enqueue on stream (NULL: the default stream).
 * Each returns INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument.
 */
#ifndef INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H
#define INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_STATUS_ROUTER_FAULT 1
#define INFERENCE_CUDA_STATUS_NONFINITE_LOGITS 2
#define INFERENCE_CUDA_STATUS_INVALID_STEP_STATE 4
/* Mask 16: an EXL3 projection of the step saw a nonfinite value (moved in by exl3_flag_fold). */
#define INFERENCE_CUDA_STATUS_EXL3_ARITHMETIC 16

typedef struct inference_cuda_step_params {
    int token_id;
    int pos;
    int kv_len;
    int kv_slot;
    int status;
    int pad[3];
} inference_cuda_step_params;

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_step_params) == 32, "StepParams is 32 bytes");
#else
_Static_assert(sizeof(inference_cuda_step_params) == 32, "StepParams is 32 bytes");
#endif

/* Clears params->status: the status is zeroed at each step start, then mask 4
 * (invalid step state) is set when the state is not valid (see above). First kernel of a step; the bits of the step
 * before were published to the host (result pair of argmax) at its end. One launch. */
int inference_cuda_step_begin_launch(void *params, void *stream, int *out_stage);

/* The sticky arithmetic flag of the EXL3 projections (a caller-owned u32) rides the result pair:
 * exl3_flag_clear zeroes the word and runs right after step_begin; exl3_flag_fold runs after the
 * last EXL3 projection and before argmax: it zeroes the word and ORs mask 16 into
 * params->status when bit 0 was set, so the one result read carries the verdict. One launch each. */
int inference_cuda_exl3_flag_clear_launch(void *flag, void *stream, int *out_stage);
int inference_cuda_exl3_flag_fold_launch(void *params, void *flag, void *stream, int *out_stage);

/* Last kernel of a step: pos += m, kv_len += m, and kv_slot += m (kv_slot is reserved and
 * equals pos) when the state is valid and pos + m <= 2^30 - 2, so kv_len == pos + 1
 * holds after every advance. Otherwise all three words become -1 (never clamped to row 0) and
 * the next step_begin sets mask 4. token_id and status are never touched (argmax is the last
 * kernel that writes status). m in [1, 4096]. One launch. */
typedef struct inference_cuda_step_advance_args {
    void *params;
    int m;
} inference_cuda_step_advance_args;
int inference_cuda_step_advance_launch(const inference_cuda_step_advance_args *args, void *stream,
                                       int *out_stage);
/* The Rust mirror (native/resident/step_params.rs) pins the same size and offsets with const asserts. */
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_step_advance_args) == 16 && offsetof(inference_cuda_step_advance_args, params) == 0 && offsetof(inference_cuda_step_advance_args, m) == 8,
              "inference_cuda_step_advance_args layout");
#else
_Static_assert(sizeof(inference_cuda_step_advance_args) == 16 && offsetof(inference_cuda_step_advance_args, params) == 0 && offsetof(inference_cuda_step_advance_args, m) == 8,
               "inference_cuda_step_advance_args layout");
#endif

/* Greedy argmax over one logits row of n f32 values.
 *   token rule: the largest value, the lowest index on a tie; a NaN never wins; +infinity wins
 *     like any value; -infinity is the smallest value (an all -infinity row gives index 0, as
 *     the host argmax of family/sampler.rs does); a row with no comparable value (all NaN)
 *     gives -1.
 *   status rule: mask 2 is OR-ed into params->status when ANY of the n values is not finite
 *     (NaN, +infinity or -infinity). This is the predicate of check_logits in sampler.rs,
 *     which fails the host sampler on such a row. The token of such a row is defined above
 *     but the host must not use it.
 *   token chaining: params->token_id = the token (also -1), so that the next replay of the
 *     step graph embeds it. No other word of params is written.
 *   result (may be NULL, else 8-byte aligned): two ints {token, status}, where status is
 *     params->status after the OR. One 8-byte device-to-host copy gives the host both words. It
 *     is a copy: params->token_id and params->status stay the source of truth. It equals the
 *     final status only if no kernel writes status after argmax.
 * part is scratch of 3072 bytes (ARGMAX_PART_BYTES), 4-byte aligned, needs no initialization.
 * n in [1, 2^30]. Enqueues two kernels (256 blocks of 256 threads, then one block). */
typedef struct inference_cuda_argmax_args {
    const void *logits;
    void *part;
    void *params;
    void *result;
    int n;
} inference_cuda_argmax_args;
int inference_cuda_argmax_launch(const inference_cuda_argmax_args *args, void *stream,
                                 int *out_stage);
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_argmax_args) == 40 && offsetof(inference_cuda_argmax_args, logits) == 0 && offsetof(inference_cuda_argmax_args, part) == 8 && offsetof(inference_cuda_argmax_args, params) == 16 && offsetof(inference_cuda_argmax_args, result) == 24 && offsetof(inference_cuda_argmax_args, n) == 32,
              "inference_cuda_argmax_args layout");
#else
_Static_assert(sizeof(inference_cuda_argmax_args) == 40 && offsetof(inference_cuda_argmax_args, logits) == 0 && offsetof(inference_cuda_argmax_args, part) == 8 && offsetof(inference_cuda_argmax_args, params) == 16 && offsetof(inference_cuda_argmax_args, result) == 24 && offsetof(inference_cuda_argmax_args, n) == 32,
               "inference_cuda_argmax_args layout");
#endif

/* The row-batched greedy argmax of a K-row verify window, and the kernel that folds the window's
 * status once. The single-row argmax above is unchanged.
 *   argmax_rows: row r of the m logits rows (n f32 values each, row r at logits + r * ld floats)
 *     gets the token of the single-row argmax (the largest value, the lowest index on a tie, a NaN
 *     never wins, a row with no comparable value gives -1: the same code and the same grid shape per
 *     row, so the same token for every row) in result[2 r], and its OWN status bits in result[2 r + 1]:
 *     mask 2 when any of the row's n values is not finite, otherwise 0. The pair is stored whole
 *     (not OR-ed), so every window rewrites every pair. Nothing is written to params: the single-row
 *     kernel stores `before | bits`, the return value of atomicOr on the shared status word, which
 *     with m rows in flight depends on block scheduling; here no atomic runs and a pair can show no
 *     other row's bits. params->token_id is not written either (the rows take their ids from the
 *     window-ids buffer). part is scratch of m * 3072 bytes (m * ARGMAX_PART_BYTES), 4-byte aligned,
 *     no initialization. result is 8-byte aligned and holds 2 m ints. m in [1, 16], n in [1, 2^30],
 *     ld >= n. Enqueues two kernels (256 x m blocks of 256 threads, then m blocks).
 *   window_status_fold: one thread, after all rows and after every other status writer. ORs
 *     result[2 r + 1] for r < m and the EXL3 arithmetic flag (bit 0 of the u32 flag word, which it
 *     zeroes) into params->status exactly once (mask 16 for the flag, as exl3_flag_fold does), then
 *     stores the window word w = params->status after the OR in result[2 m] and 0 in result[2 m + 1].
 *     It only reads the pairs and clears none: the pairs and w are rewritten whole by every window,
 *     so the host reads 2 m + 2 ints (8 m + 8 bytes) after one synchronization and needs no
 *     preparation of the buffer. Mask 1 (router), mask 4 (invalid state) and mask 16 are window-level
 *     bits that live in w only. params, flag and result must overlap no other range. */
typedef struct inference_cuda_argmax_rows_args {
    const void *logits;
    void *part;
    void *result;
    long long ld;
    int n;
    int m;
} inference_cuda_argmax_rows_args;
int inference_cuda_argmax_rows_launch(const inference_cuda_argmax_rows_args *args, void *stream,
                                      int *out_stage);

typedef struct inference_cuda_window_fold_args {
    void *params;
    void *flag;
    void *result;
    int m;
} inference_cuda_window_fold_args;
int inference_cuda_window_status_fold_launch(const inference_cuda_window_fold_args *args, void *stream,
                                             int *out_stage);
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_argmax_rows_args) == 40 && offsetof(inference_cuda_argmax_rows_args, logits) == 0 && offsetof(inference_cuda_argmax_rows_args, part) == 8 && offsetof(inference_cuda_argmax_rows_args, result) == 16 && offsetof(inference_cuda_argmax_rows_args, ld) == 24 && offsetof(inference_cuda_argmax_rows_args, n) == 32 && offsetof(inference_cuda_argmax_rows_args, m) == 36,
              "inference_cuda_argmax_rows_args layout");
static_assert(sizeof(inference_cuda_window_fold_args) == 32 && offsetof(inference_cuda_window_fold_args, params) == 0 && offsetof(inference_cuda_window_fold_args, flag) == 8 && offsetof(inference_cuda_window_fold_args, result) == 16 && offsetof(inference_cuda_window_fold_args, m) == 24,
              "inference_cuda_window_fold_args layout");
#else
_Static_assert(sizeof(inference_cuda_argmax_rows_args) == 40 && offsetof(inference_cuda_argmax_rows_args, logits) == 0 && offsetof(inference_cuda_argmax_rows_args, part) == 8 && offsetof(inference_cuda_argmax_rows_args, result) == 16 && offsetof(inference_cuda_argmax_rows_args, ld) == 24 && offsetof(inference_cuda_argmax_rows_args, n) == 32 && offsetof(inference_cuda_argmax_rows_args, m) == 36,
               "inference_cuda_argmax_rows_args layout");
_Static_assert(sizeof(inference_cuda_window_fold_args) == 32 && offsetof(inference_cuda_window_fold_args, params) == 0 && offsetof(inference_cuda_window_fold_args, flag) == 8 && offsetof(inference_cuda_window_fold_args, result) == 16 && offsetof(inference_cuda_window_fold_args, m) == 24,
               "inference_cuda_window_fold_args layout");
#endif

/* The device-side comparison of a greedy K-row window (see verify_greedy_k in step_params.cu).
 *   After argmax_rows: a = the number of leading rows i < m - 1 with result[2 i] == ids[i + 1], and
 *   bonus = result[2 a]. Stored whole in result[2 m + 2] (a) and result[2 m + 3] (bonus), after the
 *   window word and its pad (result[2 m], result[2 m + 1]) that window_status_fold stores. The pairs
 *   are only read and nothing else is written: no status, no token chaining. ids holds the window
 *   ids (m int32, 4-byte aligned, the first is the pending token, the others the drafts); result is
 *   8-byte aligned and holds 2 m + 4 ints. The two ranges overlap no other. m in [1, 16].
 *   One launch of one thread. */
typedef struct inference_cuda_verify_greedy_args {
    const void *ids;
    void *result;
    int m;
} inference_cuda_verify_greedy_args;
int inference_cuda_verify_greedy_launch(const inference_cuda_verify_greedy_args *args, void *stream,
                                        int *out_stage);
#ifdef __cplusplus
static_assert(sizeof(inference_cuda_verify_greedy_args) == 24 && offsetof(inference_cuda_verify_greedy_args, ids) == 0 && offsetof(inference_cuda_verify_greedy_args, result) == 8 && offsetof(inference_cuda_verify_greedy_args, m) == 16,
              "inference_cuda_verify_greedy_args layout");
#else
_Static_assert(sizeof(inference_cuda_verify_greedy_args) == 24 && offsetof(inference_cuda_verify_greedy_args, ids) == 0 && offsetof(inference_cuda_verify_greedy_args, result) == 8 && offsetof(inference_cuda_verify_greedy_args, m) == 16,
               "inference_cuda_verify_greedy_args layout");
#endif

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_STEP_PARAMS_H */
