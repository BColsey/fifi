// A CTA works on just one projection. The host checks all ranges first, so
// these CTAs can run in any order. Arguments are immutable kernel parameters;
// no device-side descriptor allocations, copies or dependent launches.
struct DecodeAttentionGroup {
    inference_cuda_dense_gemv_args calls[8];
    int limits[8];  // Exclusive CTA boundaries; pad unused entries to total.
};

template<int Type, int Bytes>
__device__ __forceinline__ void decode_group_kw(inference_cuda_dense_gemv_args a, int block) {
    // Keep the original three K warps on narrow shapes. The extra warp still
    // participates in the CTA barrier, but has no row to emit.
    if (a.rows == 384) decode_kw_rows<Type,Bytes,2304,3,true>(a,block);
    else decode_kw_rows<Type,Bytes,2304,2>(a,block);
}

__global__ void decode_attention_group(const __grid_constant__ DecodeAttentionGroup group) {
    const int index = blockIdx.x;
    int call = 0;
#pragma unroll
    for (int i=0; i<7; ++i) call += index >= group.limits[i];
    const auto &a = group.calls[call];
    const int block = index - (call ? group.limits[call-1] : 0);
    switch (a.weight_dtype) {
        case fifi_gguf::kDtypeBase+8: decode_q8_rows<2304>(a,block); break;
        case fifi_gguf::kDtypeBase+12: decode_group_kw<12,144>(a,block); break;
        case fifi_gguf::kDtypeBase+13: decode_group_kw<13,176>(a,block); break;
        case fifi_gguf::kDtypeBase+14: decode_group_kw<14,210>(a,block); break;
        case fifi_gguf::kDtypeBase+18: decode_iq3_rows<true>(a,block); break;
        case kDtypeBf16: decode_bf16_exact_row(a,block); break;
    }
}

// Called after validating compact matrix extents, before any kernel launch.
int decode_attention_blocks(const inference_cuda_dense_gemv_args &a) {
    if (a.m != 1 || a.cols != 2304 || !aligned(a.x,16)) return 0;
    if (a.weight_dtype == kDtypeBf16) return aligned(a.w,16) && (a.rows == 24 || a.rows == 32) ? a.rows : 0;
    if (a.rows != 384 && a.rows != 1024 && a.rows != 3072) return 0;
    switch (a.weight_dtype-fifi_gguf::kDtypeBase) {
        case 8: return a.rows/2;
        case 12: case 13: case 14: return !aligned(a.w,4) ? 0 : a.rows == 384 ? a.rows : a.rows/2;
        case 18: return a.rows == 3072 ? a.rows : 0;
        default: return 0;
    }
}
