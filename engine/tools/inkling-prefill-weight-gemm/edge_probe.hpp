// IQ2_XS misaligned F32 subrange and odd strides must use scalar staging.
void expert_edge(const char *dir){
    constexpr int m=35,n=2304,k=2304,sx=k+3,sy=n+5;constexpr size_t bytes=size_t(n)*k/256*74;
    auto data=load(std::string(dir)+"/extra-17-2304-2304.bin",bytes);Stream s;
    Buffer<uint8_t> w(bytes+2);Buffer<float> x(m*sx+1);Buffer<uint32_t> y(m*sy+2);
    check(cudaMemcpy(w.p+2,data.data(),bytes,cudaMemcpyHostToDevice));
    std::vector<float> h(m*sx+1);for(size_t i=0;i<h.size();i++)h[i]=float(int(i%39)-20)/32.f;
    check(cudaMemcpy(x.p,h.data(),h.size()*4,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{w.p+2,x.p+1,y.p+1,m,n,k,sx,sy,256+17};
    std::vector<uint32_t> out(m*sy+2),ref;
    for(int arm=0;arm<2;arm++){
        check(cudaMemsetAsync(y.p,255,out.size()*4,s.id));int stage;
        require(!(arm?inference_cuda_inkling_iq2_visit_launch(&a,s.id,&stage):inference_cuda_gguf_wmma_launch(&a,s.id,&stage)),"strided WMMA");
        check(cudaMemcpyAsync(out.data(),y.p,out.size()*4,cudaMemcpyDeviceToHost,s.id));check(cudaStreamSynchronize(s.id));if(!arm)ref=out;compare_bits(a,ref,out);
    }
    Buffer<uint8_t> scratch(size_t(n)*k*2);
    int stage=0,used=1;
    require(!inference_cuda_inkling_panel_launch(&a,scratch.p,n*k*2,s.id,&stage,&used)&&!used,"strided panel should decline");
    a.x=x.p;a.ld_x=k;
    require(!inference_cuda_inkling_panel_launch(&a,scratch.p,n*k*2-1,s.id,&stage,&used)&&!used,"short panel should decline");
    require(!inference_cuda_inkling_panel_launch(&a,x.p,n*k*2,s.id,&stage,&used)&&!used,"overlapping panel should decline");
    fprintf(stderr,"expert_edge M=%d sx=%d sy=%d exact=%d guard=pass\n",m,sx,sy,m*n);
}
