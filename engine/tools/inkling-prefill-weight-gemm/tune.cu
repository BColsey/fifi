// Randomized standalone shape sweep. A separate translation unit keeps the
// unselected template instantiations out of the engine build.
#define main full_probe_main
#include "probe.cu"
#undef main
#include <cuda_fp16.h>
#include "resident/gguf_gemv.cuh"
#include "resident/inkling_prefetch.cuh"
#include <algorithm>
#include <cstdlib>
struct Config{int bm,r,bk,pipe;void(*call)(inference_cuda_dense_gemv_args,cudaStream_t);};
template<int C,int BM,int R,int BK,bool P> void submit(inference_cuda_dense_gemv_args a,cudaStream_t s){inkling_prefetch<C,BM,R,BK,P><<<dim3(a.rows/(8*R),(a.m+BM-1)/BM),256,0,s>>>(a);}
template<int C> std::vector<Config> variants(){std::vector<Config> res;
#define T(M,R,K,P) res.push_back({M,R,K,P,submit<C,M,R,K,P>})
#define K(M,R,k) T(M,R,k,false); T(M,R,k,true)
#define R(M,r) K(M,r,32);K(M,r,64);K(M,r,128);K(M,r,256)
#define M(m) R(m,1);R(m,2);R(m,4);R(m,8)
#ifdef EXPERT_TUNE
 T(2,1,64,true);T(2,1,128,true);T(4,1,64,true);T(4,1,128,true);T(4,2,128,true);T(8,1,64,true);T(8,1,128,true);T(8,2,128,true);T(8,4,128,true);T(16,1,128,true);T(16,2,128,true);T(16,4,128,true);
#elif defined(EXTRA_TUNE)
 T(4,1,64,true);T(8,1,64,true); T(8,1,128,true);T(8,2,128,true);T(8,2,256,true);T(8,4,128,true);T(8,4,256,true);T(16,4,128,true);T(16,8,64,true);T(16,8,128,true);T(16,8,256,true);
#else
 M(4);M(8);M(16);
#endif
#undef M
#undef R
#undef K
#undef T
 return res;}
template<int C> void sweep(const char *dir,int m,int n,int k,int seed,const char *filter){
    size_t bytes=size_t(n)*k/(C==8?32:C==42?64:256)*(C==8?34:C==12?144:C==13?176:C==14?210:C==16?66:C==42?18:98);
    std::string filename=(C==8 && n!=384?"q8-":("extra-"+std::to_string(C)+"-"))+std::to_string(k)+"-"+std::to_string(n)+".bin";
    auto w=load(std::string(dir)+"/"+filename,bytes);Stream s;Buffer<uint8_t> dw(bytes);Buffer<float> dx(size_t(m)*k),dy(size_t(m)*n);
    std::mt19937 rng(seed);std::vector<float> x(size_t(m)*k),ref(size_t(m)*n),out(ref.size());for(auto &v:x)v=float(int(rng()%8193)-4096)/2048.f;
    check(cudaMemcpy(dx.p,x.data(),4*x.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));inference_cuda_dense_gemv_args a{dw.p,dx.p,dy.p,m,n,k,k,n,256+C};
    auto cs=variants<C>();std::shuffle(cs.begin(),cs.end(),rng);cs.insert(cs.begin(),{8,4,128,0,nullptr});
    for(size_t z=0;z<cs.size();z++){auto conf=cs[z];std::string id=std::to_string(conf.bm)+":"+std::to_string(conf.r)+":"+std::to_string(conf.bk)+":"+std::to_string(conf.pipe);
        if(z && *filter && std::string(filter).find(","+id+",")==std::string::npos)continue;
        auto launch=[&]{if(conf.call)conf.call(a,s.id);else{int stage=0;
#ifdef EXPERT_TUNE
            bool use=C==16?(m>=5 && m<=16):(m>=5 && m<=8)||(m>=13 && m<=32);int tile=0;int rc=use?inference_cuda_gguf_q8_row_reuse_width_launch(&a,s.id,&stage,&tile,-1):inference_cuda_gguf_exact_expert_tile_launch(&a,s.id,&stage,&tile);require(!rc,"deployed expert tile");
#else
            int variant=2;
#ifdef EXTRA_TUNE
            variant=(n<=1024 && m<=8)?1:2;
#endif
            require(!inference_cuda_inkling_packed_visit_launch(&a,s.id,&stage,variant),"previous exact tile");
#endif
            }};
        for(int i=0;i<4;i++)launch();Event st,ed;check(cudaEventRecord(st.id,s.id));for(int i=0;i<24;i++)launch();check(cudaEventRecord(ed.id,s.id));check(cudaEventSynchronize(ed.id));check(cudaGetLastError());float ms;check(cudaEventElapsedTime(&ms,st.id,ed.id));
        check(cudaMemcpy(out.data(),dy.p,4*out.size(),cudaMemcpyDeviceToHost));if(!z)ref=out;require(!memcmp(out.data(),ref.data(),4*out.size()),"sweep bit mismatch");printf("%d,%d,%d,%d,%d,%s,%.3f,%zu\n",seed,C,m,n,k,z?id.c_str():"deployed",ms*1000./24,z?out.size():0);}
}
int main(int argc,char **argv){try{require(argc>=3,"tune WEIGHTS SEED [filter]");int seed=atoi(argv[2]);const char *filter=argc==4?argv[3]:"";puts("seed,codec,m,n,k,tile,us,exact");
#ifdef EXPERT_TUNE
 for(int m=2;m<=32;m++){if(m<=16)sweep<16>(argv[1],m,2304,2304,seed,filter);sweep<42>(argv[1],m,2304,1152,seed,filter);}
#elif defined(EXTRA_TUNE)
 for(int m:{7,8,12,63,64}){for(int n:{384,1024,3072}){sweep<8>(argv[1],m,n,2304,seed,filter);sweep<12>(argv[1],m,n,2304,seed,filter);sweep<13>(argv[1],m,n,2304,seed,filter);sweep<14>(argv[1],m,n,2304,seed,filter);}sweep<18>(argv[1],m,3072,2304,seed,filter);sweep<8>(argv[1],m,2304,3072,seed,filter);sweep<13>(argv[1],m,2304,3072,seed,filter);sweep<14>(argv[1],m,2304,3072,seed,filter);}
#else
 for(auto s:kShapes)for(int m:{8,64})sweep<8>(argv[1],m,s.rows,s.cols,seed,filter);for(int m:{8,64})sweep<14>(argv[1],m,3072,2304,seed,filter);
#endif
 return 0;}catch(const std::exception &e){fprintf(stderr,"%s\n",e.what());return 1;}}
