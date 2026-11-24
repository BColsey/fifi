extern "C" int codebook_exact_probe(const inference_cuda_dense_gemv_args *,cudaStream_t);
extern "C" int expert_octet_probe(const inference_cuda_dense_gemv_args *,cudaStream_t,int);
void mixed(const char *dir,int codec,int block,int size,int n,int k,int m){
    size_t bytes=size_t(n)*k/block*size;
    std::string path=std::string(dir)+(codec==8 && n!=384?"/q8-":"/extra-"+std::to_string(codec)+"-")+std::to_string(k)+"-"+std::to_string(n)+".bin";
    auto w=load(path,bytes); Stream s;Buffer<uint8_t> dw(bytes);Buffer<float> dx(m*k);Buffer<uint32_t> dy(m*n+2);
    std::vector<float> x(m*k); for(size_t i=0;i<x.size();++i) x[i]=float(int(i%113)-52)/79.f;
    check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{dw.p,dx.p,dy.p+1,m,n,k,k,n,256+codec};std::vector<uint32_t> ref(m*n+2),out(ref.size());
    bool oct=(codec==16 || codec==17 || codec==19 || codec==22) && k==2304 && n==2304;
    for(int arm=0;arm<=6;++arm){
        if(arm==6 && (m==1 || codec==16))continue;
        if(!oct && arm>=3 && arm<=5)continue;
        if((m==1 || codec==16) && (arm==1||arm==2)) continue;
        auto launch=[&]{int stage;int rc=arm==6?inference_cuda_inkling_packed_visit_launch(&a,s.id,&stage,3):arm==5?codebook_exact_probe(&a,s.id):arm>=3?expert_octet_probe(&a,s.id,arm==3?2:4):arm?inference_cuda_inkling_packed_visit_launch(&a,s.id,&stage,arm):inference_cuda_dense_gemv_launch(&a,s.id,&stage); require(!rc,"mixed dispatch");};
        check(cudaMemsetAsync(dy.p,255,out.size()*4,s.id));for(int i=0;i<5;i++) launch();
        Event start,stop;check(cudaEventRecord(start.id,s.id));for(int i=0;i<32;++i) launch();check(cudaEventRecord(stop.id,s.id));check(cudaEventSynchronize(stop.id));
        float ms;check(cudaEventElapsedTime(&ms,start.id,stop.id));double us=ms*1000./32;
        check(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));if(!arm) ref=out;compare_bits(a,ref,out);
        size_t logical=bytes*(arm==3?(m+1)/2:arm==4?(m+3)/4:arm?(m+7)/8:m);
        if(arm==6){int bm=probe_prefetch_bm(&a);logical=bytes*((m+bm-1)/bm);}
        printf("%d,%d,%d,c%d_%d,%zu,%.3f,%.2f,%.4f,%d\n",m,n,k,codec,arm,logical,us,logical/us/1000,2.*m*n*k/us/1e6,arm?m*n:0);
    }
}
