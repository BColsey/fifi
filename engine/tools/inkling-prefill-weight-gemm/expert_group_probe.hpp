extern "C" int expert_group_probe(const inference_cuda_dense_gemv_args *,int,cudaStream_t);
void expert_batch(const char *dir,int codec,int count){
    constexpr int n=2304;int k=codec==40?1152:2304;size_t bytes=size_t(n)*k/(codec==40?64:256)*(codec==40?36:codec==17?74:codec==19?50:82);
    auto w=load(std::string(dir)+"/bank-"+std::to_string(codec)+".bin",8*bytes);int all=0;constexpr int ms[]={32,22,54,96,17,61,28,32};for(int i=0;i<count;i++)all+=ms[i];
    Stream s;Buffer<uint8_t> dw(bytes*count);Buffer<float> x(all*k),y(all*n);std::vector<float> in(all*k);for(size_t i=0;i<in.size();i++)in[i]=float(int(i%59)-22)/23.f;
    check(cudaMemcpy(dw.p,w.data(),bytes*count,cudaMemcpyHostToDevice));check(cudaMemcpy(x.p,in.data(),in.size()*4,cudaMemcpyHostToDevice));
    std::vector<inference_cuda_dense_gemv_args> calls;size_t base=0,sweeps=0;for(int i=0;i<count;i++){calls.push_back({dw.p+i*bytes,x.p+base*k,y.p+base*n,ms[i],n,k,k,n,codec==40?2:256+codec});base+=ms[i];sweeps+=(ms[i]+31)/32;}
    std::vector<float> out(all*n),ref;
    for(int arm=0;arm<3;arm++){
        auto launch=[&]{int stage=0;if(arm==2)require(!expert_group_probe(calls.data(),count,s.id),"grouped WMMA");else for(auto &a:calls)require(!(arm?inference_cuda_inkling_iq2_visit_launch(&a,s.id,&stage):inference_cuda_gguf_wmma_launch(&a,s.id,&stage)),"expert batch serial");};
        for(int i=0;i<5;i++)launch();Event st,ed;check(cudaEventRecord(st.id,s.id));for(int i=0;i<32;i++)launch();check(cudaEventRecord(ed.id,s.id));check(cudaEventSynchronize(ed.id));float ms;check(cudaEventElapsedTime(&ms,st.id,ed.id));double us=ms*1000./32;
        check(cudaMemcpy(out.data(),y.p,out.size()*4,cudaMemcpyDeviceToHost));if(!arm)ref=out;require(memcmp(out.data(),ref.data(),out.size()*4)==0,"expert group mismatch");
        size_t logical=bytes*sweeps;printf("%d,%d,%d,expert_group%d_%d,%zu,%.3f,%.2f,%.4f,%d\n",all,n,k,codec,arm,logical,us,logical/us/1000,2.*all*n*k/us/1e6,arm?all*n:0);
    }
}
