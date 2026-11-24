extern "C" int weight_panel_probe(const inference_cuda_dense_gemv_args *,void *,cudaStream_t,int);
extern "C" int interleave_probe(const inference_cuda_dense_gemv_args *,void *,cudaStream_t,int);
void interleave(const char *dir,int m,int n,int k,bool q8){
    int block=q8?32:256;size_t bytes=size_t(k)*n/block*(q8?34:210);size_t cached=size_t(k)*n+size_t(k)*n/(q8?32:16)*4;
    auto w=load(std::string(dir)+(q8?"/q8-":"/extra-14-")+std::to_string(k)+"-"+std::to_string(n)+".bin",bytes);Stream s;
    Buffer<uint8_t> dw(bytes),copy(cached);Buffer<float> dx(m*k);Buffer<uint32_t> dy(m*n+2);std::vector<uint32_t> out(m*n+2),ref;
    std::vector<float>x(m*k);for(size_t i=0;i<x.size();++i)x[i]=float(int(i%71)-25)/31.f;check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{dw.p,dx.p,dy.p+1,m,n,k,k,n,256+(q8?8:14)};
    for(int arm=0;arm<(q8?8:4);arm++){
        auto launch=[&]{int stage=0;int rc=arm>=4?weight_panel_probe(&a,copy.p,s.id,arm-4):arm?interleave_probe(&a,copy.p,s.id,arm-1):inference_cuda_dense_gemv_launch(&a,s.id,&stage);require(rc==0,"interleave dispatch");};
        check(cudaMemsetAsync(dy.p,255,out.size()*4,s.id));for(int i=0;i<5;i++)launch();Event start,stop;check(cudaEventRecord(start.id,s.id));for(int i=0;i<32;i++)launch();check(cudaEventRecord(stop.id,s.id));check(cudaEventSynchronize(stop.id));float ms;check(cudaEventElapsedTime(&ms,start.id,stop.id));double us=ms*1000./32;
        if(arm!=1 && arm!=4){check(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));if(arm==0)ref=out;compare_bits(a,ref,out);}
        size_t logical=arm==0?bytes*m:arm==1?bytes+cached:cached*((m+(arm==2?7:15))/(arm==2?8:16));
        if(arm>=4)logical=arm==4?2*bytes:bytes*((m+(arm==7?15:7))/(arm==7?16:8));
        printf("%d,%d,%d,%s_%s,%zu,%.3f,%.2f,%.4f,%d\n",m,n,k,q8?"q8":"q6",arm==0?"orig":arm==1?"encode":arm==2?"cache8":arm==3?"cache16":arm==4?"compact_encode":arm==5?"compact8x16":arm==6?"compact8x32":"compact16x32",logical,us,logical/us/1000,arm==1 || arm==4?0:2.*m*n*k/us/1e6,arm>=2 && arm!=4?m*n:0);
    }
}
