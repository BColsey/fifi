#ifndef INFERENCE_CUDA_EXPERT_MATH_CUH
#define INFERENCE_CUDA_EXPERT_MATH_CUH

#include <cuda_runtime.h>

__device__ inline float clipped_expert_activation(float gate, float up,
                                                   float gate_cap, float up_cap) {
    const float g = gate_cap * tanhf(gate / gate_cap);
    const float u = up_cap * tanhf(up / up_cap);
    // Preserve projection overflow even if the soft caps would hide it.
    return isfinite(gate) && isfinite(up)
        ? g * (1.0f / (1.0f + expf(-gate))) * u
        : __int_as_float(0x7fc00000);
}

#endif
