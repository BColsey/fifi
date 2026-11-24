/* The device router, per-kernel header (native/resident/router.h). Included by router.cu
 * and by the Rust FFI mirror's reader; nothing else. x is the f32 normalized row [m][hidden] (the
 * norm_out of the add+RMSNorm kernel), w the router weight f32 [n_expert][hidden] (expert-major,
 * 16-byte aligned), w_shared the shared-expert gate f32 [hidden], or NULL together with
 * shared_gate for a model without a shared expert. logits is scratch f32 [m][n_expert + 1] (the
 * last column is unused without a shared gate). Outputs: ids i32 [m][top_k] (-1 on a faulty row),
 * weights f32 [m][top_k], shared_gate f32 [m] = sigmoid(dot(x, w_shared)) (NULL: not computed).
 * status is a device i32 word; bit 1 is OR-ed in when any row is faulty (a non-finite logit, or a
 * softmax or selected sum that is not finite and positive, or a non-finite gate dot). Limits: m in
 * [1, 4096], hidden a multiple of 4 up to 65536, n_expert in [1, 512], top_k in
 * [1, min(n_expert, 32)]. Enqueues two kernels on stream (NULL: the default stream); returns
 * INFERENCE_CUDA_INVALID_ARGUMENT before any launch for a bad argument. The logits kernel is a
 * tiled one for m >= 8 and the decode one below; both give the same logit bits (but for the sign
 * of an exact zero when hidden / 4 is not a multiple of 512, see router.cu). */
#ifndef INFERENCE_CUDA_RESIDENT_ROUTER_H
#define INFERENCE_CUDA_RESIDENT_ROUTER_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct inference_cuda_router_args {
    const void *x;
    const void *w;
    const void *w_shared;
    void *logits;
    void *ids;
    void *weights;
    void *shared_gate;
    void *status;
    int m;
    int hidden;
    int n_expert;
    int top_k;
} inference_cuda_router_args;

int inference_cuda_router_launch(const inference_cuda_router_args *args, void *stream,
                                 int *out_stage);

/* Split path: tightly packed logits [m][n_expert] without a shared gate.
 * The caller owns any model-specific publication. Both entries require an explicit stream.
 * Every written active extent is disjoint from all other active extents.
 */
typedef struct inference_cuda_router_logits_args {
    const float *x;
    size_t x_bytes;
    const float *w;
    size_t w_bytes;
    float *logits;
    size_t logits_bytes;
    int m;
    int hidden;
    int n_expert;
} inference_cuda_router_logits_args;

typedef struct inference_cuda_router_sigmoid_args {
    const float *logits;
    size_t logits_bytes;
    int *ids;
    size_t ids_bytes;
    float *weights;
    size_t weights_bytes;
    int *status;
    size_t status_bytes;
    int m;
    int n_expert;
    int top_k;
    const float *bias;
    size_t bias_bytes;
    float floor;
    float scale;
    int bias_profile;
} inference_cuda_router_sigmoid_args;

/* bias_profile, the contract of a non-NULL bias. Both rank the experts by sigmoid(logit) + bias
 * and weight the selected experts by their unbiased scores.
 *  KERNEL_ROUNDED: the kernel's own F32 sigmoid, ties broken by expert index, weights p / sum, and
 *                  an empty sum is a fault. floor 0 and scale 1.
 *  HOST_ROUNDED:   the sigmoid rounded as the Rust host rounds it, ties broken like total_cmp,
 *                  weights scale * p / max(sum, floor) with a finite floor > 0. */
#define INFERENCE_CUDA_ROUTER_BIAS_KERNEL_ROUNDED 0
#define INFERENCE_CUDA_ROUTER_BIAS_HOST_ROUNDED 1

int inference_cuda_router_logits_launch(const inference_cuda_router_logits_args *args,
                                        void *stream, int *out_stage);

/* Strided logits of a K-row verify window: the decode kernel (the one-warp-per-dot router_logits_k)
 * with the row stride of the output as a parameter, so a plain F32 projection (the GDN beta and
 * alpha gates) writes row t's n_expert values at logits[t * logit_stride] of a wider row, next to
 * the columns other launches write, with no copy. The arithmetic of every logit is that of the
 * tight entry above: only the address of the store differs. m in [1, 7]: the tiled kernel of
 * m >= 8 is never launched here (a caller with 8 rows makes two 4-row launches). logit_stride >=
 * n_expert (floats); logits_bytes >= ((m - 1) * logit_stride + n_expert) * 4. Columns of a row
 * outside [0, n_expert) of the stride are not written. Same ranges and alignment as the tight
 * entry. */
typedef struct inference_cuda_router_logits_strided_args {
    const float *x;
    size_t x_bytes;
    const float *w;
    size_t w_bytes;
    float *logits;
    size_t logits_bytes;
    int m;
    int hidden;
    int n_expert;
    int logit_stride;
} inference_cuda_router_logits_strided_args;

int inference_cuda_router_logits_strided_launch(const inference_cuda_router_logits_strided_args *args,
                                                void *stream, int *out_stage);
/* Raw-logit health precedes F32 sigmoid. Rank sigmoid scores (lower-ID ties), then
 * normalize only the selected F32 scores. Bad rows get IDs -1, weights 0, status OR1.
 * With bias, rank sigmoid(logit) + bias in descending total order under bias_profile. Without
 * bias, pass NULL, floor 0, scale 1 and bias_profile INFERENCE_CUDA_ROUTER_BIAS_KERNEL_ROUNDED.
 * Never clears status.
 */
int inference_cuda_router_sigmoid_select_launch(const inference_cuda_router_sigmoid_args *args,
                                                void *stream, int *out_stage);

#ifdef __cplusplus
}
static_assert(sizeof(inference_cuda_router_logits_args) == 64 &&
              offsetof(inference_cuda_router_logits_args, m) == 48 &&
              offsetof(inference_cuda_router_logits_args, n_expert) == 56, "router logits layout");
static_assert(sizeof(inference_cuda_router_logits_strided_args) == 64 &&
              offsetof(inference_cuda_router_logits_strided_args, m) == 48 &&
              offsetof(inference_cuda_router_logits_strided_args, n_expert) == 56 &&
              offsetof(inference_cuda_router_logits_strided_args, logit_stride) == 60,
              "router logits strided layout");
static_assert(sizeof(inference_cuda_router_sigmoid_args) == 112 &&
              offsetof(inference_cuda_router_sigmoid_args, status) == 48 &&
              offsetof(inference_cuda_router_sigmoid_args, top_k) == 72 &&
              offsetof(inference_cuda_router_sigmoid_args, bias) == 80 &&
              offsetof(inference_cuda_router_sigmoid_args, scale) == 100 &&
              offsetof(inference_cuda_router_sigmoid_args, bias_profile) == 104,
              "router sigmoid layout");
#endif

#endif
