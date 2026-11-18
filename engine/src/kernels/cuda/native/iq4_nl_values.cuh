/*
 * The 16 non-linear dequant values of the IQ4_NL family, as f32. These are the bytes of
 * weights/gguf_tables.rs KVALUES_IQ4NL, and the host pin test in
 * kernels/cuda/resident/matvec_q/tests.rs compares this text against that constant, so the
 * CPU oracle and every native table cannot drift apart. IQ4_XS and IQ4_NL read the same
 * table.
 *
 * The values reach kernels BY VALUE through make_iq4nl_table(), never through a
 * __constant__ definition: in this .so's multi-cubin image the at-load constant-bank
 * layout aliases per-TU initialized __constant__ arrays by position (2026-11-18, 3090:
 * the gemm_dq TU's uniquely named table read nvfp4 kE2M1's initializer, and renaming and
 * external linkage both still aliased). Kernel parameters are copied at launch, outside
 * that machinery. The tree-wide guard against new initialized __constant__ tables is
 * kernels/cuda/native/constant_bank_guard.rs.
 */
#ifndef INFERENCE_CUDA_IQ4_NL_VALUES_CUH
#define INFERENCE_CUDA_IQ4_NL_VALUES_CUH

#include <string.h>

struct iq4nl_table {
    float v[16];
};

static inline iq4nl_table make_iq4nl_table() {
    iq4nl_table t;
    const float values[16] = {
        -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
        1.0f,    13.0f,   25.0f,   38.0f,   53.0f,   69.0f,   89.0f,   113.0f,
    };
    memcpy(t.v, values, sizeof t.v);
    return t;
}

#endif /* INFERENCE_CUDA_IQ4_NL_VALUES_CUH */
