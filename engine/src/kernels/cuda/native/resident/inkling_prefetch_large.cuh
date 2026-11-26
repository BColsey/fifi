// Bounded 128/256-row controller tiles and the final partial chunk of 511.
// Never vector-read unaligned activations; use the old exact kernel instead.
inline bool inkling_large_select(inference_cuda_dense_gemv_args a,int c,cudaStream_t s){
    if(!(a.m==127 || a.m==128 || a.m==255 || a.m==256) || !aligned(a.x,16) || a.ld_x%4 || a.rows%64 || a.cols%256)return false;
#define LARGE(C,K) inkling_prefetch<C,16,8,K,true><<<dim3(a.rows/64,(a.m+15)/16),256,0,s>>>(a)
    switch(c){case 8:LARGE(8,256);break;case 12:LARGE(12,128);break;case 13:LARGE(13,128);break;case 14:LARGE(14,64);break;case 18:LARGE(18,64);break;default:return false;}
#undef LARGE
    return true;
}
