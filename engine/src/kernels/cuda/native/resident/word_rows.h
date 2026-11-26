/*
 * Rows of 32-bit words zeroed or copied in one launch, one block per row (word_rows.cu): the
 * batched form of a device memset or copy of each row, for the many small bookkeeping rows a
 * step clears and gathers.
 *
 * inference_cuda_zero_word_rows
 *   `rows[r]` holds `words[r]` words; every word becomes zero.
 * inference_cuda_copy_word_rows
 *   `destination[r]` takes the `words[r]` words at `source[r]`. Words move as bits.
 *
 * `count` is 1 to INFERENCE_CUDA_WORD_ROWS_MAX, every `words[r]` is positive and every pointer
 * is 4-byte aligned. The caller keeps every written row apart from every other row of the launch
 * (sources may share bytes with each other). The pointer lists are read before the call returns.
 */
#ifndef INFERENCE_CUDA_WORD_ROWS_H
#define INFERENCE_CUDA_WORD_ROWS_H

#ifdef __cplusplus
extern "C" {
#endif

#define INFERENCE_CUDA_WORD_ROWS_MAX 64

int inference_cuda_zero_word_rows(void *const *rows, const int *words, int count, void *stream,
                                  int *out_stage);

int inference_cuda_copy_word_rows(const void *const *source, void *const *destination,
                                  const int *words, int count, void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif
