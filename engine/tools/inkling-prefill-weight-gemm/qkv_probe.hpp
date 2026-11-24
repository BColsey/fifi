extern "C" int qkv_prefetch_probe(const inference_cuda_dense_gemv_args *,int,cudaStream_t);
void qkv(const char *dir,int m,int count){
    constexpr int k=2304,sx=k*3;int sy=3;std::vector<inference_cuda_dense_gemv_args> args;size_t bytes=0;
    for(int i=0;i<count;i++){int n=(count>=3 && i==0)||i==3?3072:1024;sy+=n;bytes+=size_t(n)*k/32*34;}
    Stream s;Buffer<uint8_t> w(bytes);Buffer<float> x(m*sx);Buffer<uint32_t> y(m*sy+2);
    std::vector<float> host(m*sx);for(size_t i=0;i<host.size();i++)host[i]=float(int(i%31)-15)/17.f;
    check(cudaMemcpy(x.p,host.data(),host.size()*4,cudaMemcpyHostToDevice));size_t offset=0,output=0;
    for(int i=0;i<count;i++){int n=(count>=3 && i==0)||i==3?3072:1024;size_t len=size_t(n)*k/32*34;
        auto v=load(std::string(dir)+"/q8-2304-"+std::to_string(n)+".bin",len);check(cudaMemcpy(w.p+offset,v.data(),len,cudaMemcpyHostToDevice));
        args.push_back({w.p+offset,x.p+(i%3)*k,y.p+1+output,m,n,k,sx,sy,kQ8});output+=n;offset+=len;}
    std::vector<uint32_t> out(m*sy+2),ref;
    auto joined=args[0];joined.rows=sy-3;
    for(int arm=0;arm<4;arm++){
        auto launch=[&]{int stage=0,used=0;
            if(arm==3)require(!qkv_prefetch_probe(args.data(),count,s.id),"qkv prefetch");
            else if(arm==2) require(!inference_cuda_inkling_qkv_visit_launch(args.data(),count,s.id,&stage,&used)&&used,"qkv dispatch");
            else for(auto &a:args) require(!(arm?inference_cuda_gguf_q8_row_reuse_width_launch(&a,s.id,&stage,&used,-1):inference_cuda_dense_gemv_launch(&a,s.id,&stage)),"qkv baseline");};
        check(cudaMemsetAsync(y.p,255,out.size()*4,s.id));for(int i=0;i<5;++i) launch();Event start,stop;check(cudaEventRecord(start.id,s.id));
        for(int i=0;i<32;++i) launch();check(cudaEventRecord(stop.id,s.id));check(cudaEventSynchronize(stop.id));float ms;check(cudaEventElapsedTime(&ms,start.id,stop.id));double us=ms*1000./32;
        check(cudaMemcpy(out.data(),y.p,out.size()*4,cudaMemcpyDeviceToHost));if(!arm)ref=out;compare_bits(joined,ref,out);
        size_t logical=bytes*(arm?(m+7)/8:m);printf("%d,%d,%d,%s,%zu,%.3f,%.2f,%.4f,%d\n",m,sy-3,k,arm==3?"qkv_prefetch":arm==2?"qkv_group":arm==1?"qkv_serial":"qkv_ref",logical,us,logical/us/1000,2.*m*(sy-3)*k/us/1e6,arm?m*(sy-3):0);
    }
    auto invalid=args;invalid[1].y=invalid[0].y;int stage=0,used=1;require(!inference_cuda_inkling_qkv_visit_launch(invalid.data(),count,s.id,&stage,&used)&&!used,"overlapping QKV writes accepted");
    invalid=args;invalid[1].x=invalid[0].y;used=1; // May be refused as an invalid single call or a cross-call hazard.
    int rc=inference_cuda_inkling_qkv_visit_launch(invalid.data(),count,s.id,&stage,&used);require(rc || !used,"dependent QKV read accepted");
}
