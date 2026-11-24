extern "C" int ragged_exact_probe(const inference_cuda_dense_gemv_args *,int,int,cudaStream_t);
void ragged_exact(const char *dir,int c,int num,int bin){
    const int n=2304,k=c==16?2304:1152;size_t bytes=size_t(n)*k/(c==16?256:64)*(c==16?66:18);
    auto w=load(std::string(dir)+"/bank-"+std::to_string(c)+".bin",8*bytes);int ms[8];int all=0;for(int i=0;i<num;i++){ms[i]=bin==0?4:bin==1?5+i%4:9+i%(c==16?8:24);all+=ms[i];}
    Stream s;Buffer<uint8_t> dw(bytes*num);Buffer<float> x(all*k),y(all*n);std::vector<float> in(all*k);for(size_t i=0;i<in.size();i++)in[i]=float(int(i%73)-37)/67.f;
    check(cudaMemcpy(dw.p,w.data(),bytes*num,cudaMemcpyHostToDevice));check(cudaMemcpy(x.p,in.data(),in.size()*4,cudaMemcpyHostToDevice));
    std::vector<inference_cuda_dense_gemv_args> calls;int off=0;for(int i=0;i<num;i++){calls.push_back({dw.p+i*bytes,x.p+off*k,y.p+off*n,ms[i],n,k,k,n,256+c});off+=ms[i];}
    std::vector<float> ref,out(all*n);for(int arm=0;arm<3;arm++){auto launch=[&]{int st=0,used=0;if(arm==2 && num==2)require(!inference_cuda_inkling_ragged_pair_launch(calls.data(),s.id,&st,&used)&&used,"admitted ragged pair");else if(arm==2)require(!ragged_exact_probe(calls.data(),num,bin,s.id),"ragged exact");else for(auto &a:calls){int rc=arm?inference_cuda_gguf_q8_row_reuse_width_launch(&a,s.id,&st,&used,-1):inference_cuda_gguf_exact_expert_tile_launch(&a,s.id,&st,&used);require(!rc,"serial exact");}};
        for(int i=0;i<5;i++)launch();Event st,ed;check(cudaEventRecord(st.id,s.id));for(int i=0;i<24;i++)launch();check(cudaEventRecord(ed.id,s.id));check(cudaEventSynchronize(ed.id));float ms0;check(cudaEventElapsedTime(&ms0,st.id,ed.id));check(cudaMemcpy(out.data(),y.p,out.size()*4,cudaMemcpyDeviceToHost));if(!arm)ref=out;require(!memcmp(out.data(),ref.data(),4*out.size()),"ragged exact mismatch");fprintf(stderr,"ragged codec=%d bin=%d count=%d m_total=%d arm=%d us=%.3f words=%zu\n",c,bin,num,all,arm,ms0*1000./24,arm?out.size():0);}
    if(num==2){int st=0,used=1;auto bad=calls;bad[1].y=bad[0].y;require(!inference_cuda_inkling_ragged_pair_launch(bad.data(),s.id,&st,&used)&&!used,"ragged alias");
        bad=calls;bad[1].x=bad[0].y;used=1;require(inference_cuda_inkling_ragged_pair_launch(bad.data(),s.id,&st,&used)||!used,"ragged dependency");
        bad=calls;bad[1].m=3;used=1;require(!inference_cuda_inkling_ragged_pair_launch(bad.data(),s.id,&st,&used)&&!used,"ragged incompatible bin");}
}
