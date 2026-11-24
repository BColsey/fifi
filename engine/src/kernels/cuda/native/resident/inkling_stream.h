/* Depth-residual stream of Inkling 2, kept on the device: mix, seal and add.
 * One header per kernel file; abi_resident.h stays closed.
 *
 * State layout (f32, device memory), `width` floats per row:
 *   state = [ current | sealed[0] | sealed[1] | ... | sealed[blocks - 1] ]
 * A sealed row is the RMS-normalized copy of the row that was current when it was sealed.
 * `blocks` and `first_block` are kernel arguments: the launch sequence knows them.
 *
 * inference_cuda_inkling_stream_mix
 *   Rows in this order: sealed[blocks - 1], sealed[blocks - 2], ..., sealed[first_block], then
 *   the normalized current row. Each row scores dot(row, query) / sqrt(width). The weights are the
 *   softmax of the scores (the peak is subtracted first) and `out` is their weighted mean. This is
 *   `Stream::mix(first_block, query)` of the host.
 *
 * inference_cuda_inkling_stream_seal
 *   sealed[blocks] = current * rms_scale(current), then current = next. This is `Stream::seal(next)`.
 *
 * inference_cuda_inkling_stream_add
 *   current = current + row, each sum rounded to nearest even. This is `Stream::add_current(row)`.
 *
 * RMS scale: (dot(x, x) / width + eps).sqrt().recip(), every operation rounded to nearest even.
 * Reductions are reordered against the host's ordered sums, so results agree with the host
 * within floating-point error and are not bit-identical.
 * `health` is one float: left alone, or quiet NaN when an input or an output is not finite.
 *
 * Every entry takes the stream explicitly, returns INFERENCE_CUDA_INVALID_ARGUMENT for a bad
 * argument before any CUDA call, never falls back to the default stream and does not synchronize.
 * `out` and `health` must not overlap any other range.
 */
#ifndef INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H
#define INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H

#ifdef __cplusplus
extern "C" {
#endif

int inference_cuda_inkling_stream_mix(const float *state, const float *query, float *out,
                                      float *health, int first_block, int blocks, int width,
                                      float eps, void *stream, int *out_stage);

int inference_cuda_inkling_stream_seal(float *state, const float *next, float *health, int blocks,
                                       int width, float eps, void *stream, int *out_stage);

int inference_cuda_inkling_stream_add(float *state, const float *row, float *health, int width,
                                      void *stream, int *out_stage);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_RESIDENT_INKLING_STREAM_H */
