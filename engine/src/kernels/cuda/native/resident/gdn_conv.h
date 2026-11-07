/* The DeltaNet causal conv step, per-kernel header (native/resident/gdn_conv.h). Included by
 * gdn_conv.cu and read by the Rust FFI mirror; nothing else. Wraps the vendored kern::conv_silu
 * (engine/vendor/xavier-qwen/src/ops.cu) and adds the prefill variant conv_silu_rows_k.
 *
 * out[t][c] = silu(w[c][0]*s0 + w[c][1]*s1 + w[c][2]*s2 + w[c][3]*x) with x = proj[t][c] and
 * (s0, s1, s2) the three previous inputs of channel c, oldest first; the window then shifts. The
 * state is planar f32 [3][channels] (row 0 the oldest input) and is read and written in place.
 * ggml keeps the same ring as [channel][tap] (ne0 = 3 fastest), the transpose of this layout: a state
 * import or export (layout tag qwen36-gguf-xq-1 keeps the device order) must transpose it.
 * proj is f32 [m][ld] (only the first `channels` columns of a row are read), w is the GGUF
 * ssm_conv1d tensor as stored, f32 [channels][4] (tap 0 multiplies the oldest input), 16-byte
 * aligned. out is f32 [m][channels]. Limits: m in [1, 32768], channels in [1, 65536],
 * ld in [channels, 1048576]. state and out must overlap no other range (proj and w are read only).
 * Enqueues one kernel on stream (not NULL); returns INFERENCE_CUDA_INVALID_ARGUMENT before any
 * launch for a bad argument. The kernel reads no position, length or slot word. */
#ifndef INFERENCE_CUDA_RESIDENT_GDN_CONV_H
#define INFERENCE_CUDA_RESIDENT_GDN_CONV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_gdn_conv_args {
    const void *proj;
    const void *w;
    void *state;
    void *out;
    int m;
    int ld;
    int channels;
} inference_cuda_gdn_conv_args;

int inference_cuda_gdn_conv_launch(const inference_cuda_gdn_conv_args *args, void *stream,
                                   int *out_stage);

/* The same step on the prefill kernel (conv_silu_rows_k of gdn_conv.cu): bytes of out and state
 * identical to the entry above for every valid call. Each thread owns one channel for
 * rows_per_thread rows (16, 32 or 64; any other value is INVALID_ARGUMENT), so a call of many
 * rows runs on many more CTAs than the original kernel's one thread per channel. It takes no
 * snapshot range. The arguments, limits and overlap rules are those of the entry above. */
int inference_cuda_gdn_conv_rows_launch(const inference_cuda_gdn_conv_args *args, void *stream,
                                        int *out_stage, int rows_per_thread);

/* The same step with per-token state snapshots stored at a stride (the K-row verify window). After
 * token t < m - 1 the planar state [3][channels] is also stored at snap + t * snap_stride bytes, so
 * a K-row window writes the state after each row straight into its rollback slot. The kernel is the
 * kSnap = true instantiation of the vendored conv_silu_k (patch GDN-snap-stride); the entry above
 * keeps launching the original instantiation. The snapshot rows are stores only: out and state are
 * the bytes of the entry above. Needs m >= 2, snap 4-byte aligned, snap_stride a multiple of 4 and
 * at least 12 * channels bytes, and snap_bytes >= (m - 2) * snap_stride + 12 * channels. snap must
 * overlap no other range. */
typedef struct inference_cuda_gdn_conv_snap_args {
    inference_cuda_gdn_conv_args conv;
    void *snap;
    size_t snap_stride;
    size_t snap_bytes;
} inference_cuda_gdn_conv_snap_args;

int inference_cuda_gdn_conv_snap_launch(const inference_cuda_gdn_conv_snap_args *args, void *stream,
                                        int *out_stage);

#ifdef __cplusplus
static_assert(sizeof(inference_cuda_gdn_conv_snap_args) == 72 &&
                  offsetof(inference_cuda_gdn_conv_snap_args, snap) == 48 &&
                  offsetof(inference_cuda_gdn_conv_snap_args, snap_stride) == 56 &&
                  offsetof(inference_cuda_gdn_conv_snap_args, snap_bytes) == 64,
              "inference_cuda_gdn_conv_snap_args layout");
}
#endif

#endif
