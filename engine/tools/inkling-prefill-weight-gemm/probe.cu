#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "resident/dense_gemv.h"

constexpr int kQ8 = 264;
struct Shape { int cols, rows; };
constexpr Shape kShapes[] = {{2304,1024},{2304,3072},{3072,2304}};
void require(bool ok,const char *msg) { if (!ok) throw std::runtime_error(msg); }
void check(cudaError_t e) { require(e==cudaSuccess,cudaGetErrorString(e)); }
struct Stream {
    cudaStream_t id; Stream(){check(cudaStreamCreate(&id));} ~Stream(){cudaStreamDestroy(id);}
};
struct Event {
    cudaEvent_t id; Event(){check(cudaEventCreate(&id));} ~Event(){cudaEventDestroy(id);}
};
template<typename T> struct Buffer {
    T *p; explicit Buffer(size_t n){check(cudaMalloc(&p,n*sizeof(T)));} ~Buffer(){cudaFree(p);}
};
std::vector<uint8_t> load(const std::string &path,size_t bytes) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(f && f.tellg()==std::streamoff(bytes),"checkpoint tensor length");
    std::vector<uint8_t> v(bytes);
    require(bool(f.seekg(0).read(reinterpret_cast<char *>(v.data()),bytes)),"checkpoint tensor read");
    return v;
}
void compare_bits(const inference_cuda_dense_gemv_args &a,const std::vector<uint32_t> &ref,
                  const std::vector<uint32_t> &got) {
    require(got==ref,"nonzero error vs packed kernel");
    for(size_t i=0;i<got.size();++i){
        bool live=i>0 && (i-1)/a.ld_y<size_t(a.m) && (i-1)%a.ld_y<size_t(a.rows);
        require(live ? (got[i]&0x7f800000u)!=0x7f800000u : got[i]==0xffffffffu,"guard or finiteness");
    }
}

extern "C" int down_visit_probe(const inference_cuda_dense_gemv_args *,cudaStream_t,int);
extern "C" int probe_prefetch_bm(const inference_cuda_dense_gemv_args *);
extern "C" int ablate_weight_probe(const inference_cuda_dense_gemv_args *,cudaStream_t);
extern "C" int codebook_visit_probe(const inference_cuda_dense_gemv_args *,cudaStream_t);
extern "C" int octet_big_probe(const inference_cuda_dense_gemv_args *,cudaStream_t,int);
extern "C" int q6_imma_probe(const inference_cuda_dense_gemv_args *,void *,cudaStream_t,int);
extern "C" int residual_probe(const inference_cuda_dense_gemv_args *,void *,cudaStream_t);
extern "C" int panel_probe(const inference_cuda_dense_gemv_args *,void *,cudaStream_t,int);
extern "C" int octet_vector_probe(const inference_cuda_dense_gemv_args *,cudaStream_t);
extern "C" int stream_visit_probe(const inference_cuda_dense_gemv_args *, cudaStream_t,int);
extern "C" int split_visit_probe(const inference_cuda_dense_gemv_args *, cudaStream_t);
void compare_approx(const void *rv,const void *gv,int m,int n,int k,int ld,bool guards){
    auto *ref=static_cast<const unsigned char *>(rv), *got=static_cast<const unsigned char *>(gv);
    double err=0,norm=0,peak=0,maxerr=0; int exact=0;
    for (int i=0;i<m*ld+(guards?2:0);++i){
        int j=i-(guards?1:0);
        if(j<0 || j>=m*ld || j%ld>=n){uint32_t v;memcpy(&v,got+4*i,4);require(v==0xffffffffu,"guard");continue;}
        float r,v;memcpy(&r,ref+4*i,4);memcpy(&v,got+4*i,4);require(std::isfinite(v),"finite");
        double e=fabs(double(v)-r);err+=e*e;norm+=double(r)*r;peak=fmax(peak,fabs(r));maxerr=fmax(e,maxerr);exact+=!memcmp(ref+4*i,got+4*i,4);
    }
    fprintf(stderr,"split %d %d %d ld=%d maxAbs=%.8g relL2=%.8g relPeak=%.8g exact=%d/%d\n",m,n,k,ld,maxerr,sqrt(err/norm),maxerr/peak,exact,m*n);
    require(sqrt(err/norm)<0.001 && maxerr/peak<0.002,"FP16 tolerance");
}
void measure(const char *dir, int m, int n, int k, bool padded) {
    size_t bytes = size_t(n) * k / 32 * 34;
    auto w = load(std::string(dir) + "/q8-" + std::to_string(k) + "-" + std::to_string(n) + ".bin", bytes);
    int sx = k + (padded ? 3 : 0), sy = n + (padded ? 7 : 0);
    Stream stream;
    Buffer<uint8_t> dw(bytes);
    Buffer<float> dx(size_t(m)*sx);
    Buffer<uint32_t> dy(size_t(m)*sy+2);
    std::mt19937 rng(345);
    std::vector<float> x(size_t(m)*sx);
    for (float &v : x) v = float(int(rng()%8193)-4096)/2048.f;
    check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{dw.p,dx.p,dy.p+1,m,n,k,sx,sy,kQ8};
    std::vector<uint32_t> ref(size_t(m)*sy+2), result(ref.size());
    if(!padded){
        auto run=[&]{check(cudaError_t(ablate_weight_probe(&a,stream.id)));};
        for(int i=0;i<5;i++)run();Event st,ed;check(cudaEventRecord(st.id,stream.id));for(int i=0;i<32;i++)run();check(cudaEventRecord(ed.id,stream.id));check(cudaEventSynchronize(ed.id));float ms;check(cudaEventElapsedTime(&ms,st.id,ed.id));fprintf(stderr,"ablate_decode %d %d %d us=%.3f\n",m,n,k,ms*1000./32);
    }
    Buffer<uint8_t> quant(size_t(m)*k/32*68);
    for (int arm=0; arm<23; ++arm) {
        // Only the currently admitted pair windows have an actual pair baseline.
        if (arm==1 && !(k==2304 ? (m>=7 && m<=12) : (m==63 || m==64))) continue;
        auto launch = [&] {
            int stage=0, width=0;
            if(arm==12){check(cudaError_t(residual_probe(&a,quant.p,stream.id)));return;}
            if(arm>=4){check(cudaError_t(stream_visit_probe(&a,stream.id,arm-3)));return;}
            if(arm==3){check(cudaError_t(split_visit_probe(&a,stream.id)));return;}
            int rc=arm ? inference_cuda_gguf_q8_row_reuse_width_launch(&a,stream.id,&stage,&width,arm==2 ? -1 : 0)
                       : inference_cuda_dense_gemv_launch(&a,stream.id,&stage);
            require(rc==0 && (!arm || width==(arm==2 ? m : 2)),"unexpected dispatch");
        };
        check(cudaMemsetAsync(dy.p,255,result.size()*4,stream.id));
        launch();
        check(cudaMemcpyAsync(result.data(),dy.p,result.size()*4,cudaMemcpyDeviceToHost,stream.id));
        check(cudaStreamSynchronize(stream.id));
        if (!arm) ref=result;
        if(arm==3 || arm==6 || arm==12 || arm==15 || arm==16) compare_approx(ref.data(),result.data(),m,n,k,sy,true); else compare_bits(a,ref,result);
        if (padded) continue;
        for (int i=0;i<5;++i) launch();
        Event start,stop;
        check(cudaEventRecord(start.id,stream.id));
        for (int i=0;i<32;++i) launch();
        check(cudaEventRecord(stop.id,stream.id));
        check(cudaEventSynchronize(stop.id));
        float ms;
        check(cudaEventElapsedTime(&ms,start.id,stop.id));
        double us=ms*1000./32;
        size_t sweeps=arm==0 ? m : arm==1 ? (m+1)/2 : arm>=19 ? (arm==22?(m+7)/8:(m+3)/4) : arm>=17 ? (arm==17?(m+31)/32:(m+63)/64) : arm>=15 ? (arm==15?(m+15)/16:(m+63)/64) : arm==14 ? (m+31)/32 : arm>=8 ? (m+7)/8 : arm==7 ? (m+15)/16 : arm>=4 ? 1 : arm==3 ? (m+15)/16 : (m+7)/8;
        if(arm==2){int bm=probe_prefetch_bm(&a);sweeps=(m+bm-1)/bm;}
        printf("%d,%d,%d,%s,%zu,%.3f,%.2f,%.4f,%d\n",m,n,k,(arm==0?"packed":arm==1?"pair":arm==2?"visit":arm==3?"split":arm==4?"stream8":arm==5?"stream16":arm==6?"splitn":arm==7?"grid16":arm==8?"warp2":arm==9?"warp4":arm==10?"oct8":arm==11?"oct32":arm==12?"residual":arm==13?"token8":arm==14?"token32":arm==15?"simt16":arm==16?"simt64":arm==17?"group32":arm==18?"group64":arm==19?"occ0":arm==20?"occ1":arm==21?"occ2":"occ3"),bytes*sweeps,us,bytes*sweeps/us/1000,2.*m*n*k/us/1e6,arm==3 || arm==6 || arm==12 || arm==15 || arm==16?0:m*n);
    }
}
void other(const char *dir,int m,int n,int k,int codec,int block,int bytes_per_block) {
    size_t bytes=size_t(n)*k/block*bytes_per_block;
    auto w=load(std::string(dir)+"/extra-"+std::to_string(codec)+"-"+std::to_string(k)+"-"+std::to_string(n)+".bin",bytes);
    Stream s; Buffer<unsigned char> dw(bytes); Buffer<float> dx(m*k), dy(m*n);
    std::vector<float> input(m*k); for(size_t i=0;i<input.size();++i) input[i]=float(int(i%127)-60)/64.f;
    check(cudaMemcpy(dw.p,w.data(),bytes,cudaMemcpyHostToDevice));
    check(cudaMemcpy(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{dw.p,dx.p,dy.p,m,n,k,k,n,codec==0?3:codec==40?2:256+codec};
    std::vector<float> ref(m*n), out(m*n);
    bool expert=codec==16 || codec==42, wmma=codec==17 || codec==19 || codec==22 || codec==40;
    Buffer<uint8_t> digits(codec==14?size_t(m)*k/32*68:1);
    Buffer<uint16_t> panels(wmma?size_t(k)*n:1);
    for(int arm=0;arm<=(expert?1:codec==14?24:wmma?11:codec==0 && k==4608?8:0);++arm){
        if(wmma && codec==40 && arm==8)continue;
        if(wmma && codec!=17 && arm==1) continue; // split-K experiment only built for IQ2_XS
        auto launch=[&] {
            int stage=-1,tile=0, rc;
            if(codec==0 && arm==8){require(!inference_cuda_inkling_down_launch(&a,s.id,&stage),"down dispatch");return;}
            if(codec==0 && arm){check(cudaError_t(down_visit_probe(&a,s.id,arm)));return;}
            if(codec==14 && arm>=21){check(cudaError_t(q6_imma_probe(&a,digits.p,s.id,arm-20)));return;}
            if(codec==14 && arm>=3){check(cudaError_t(stream_visit_probe(&a,s.id,arm>=11?arm-1:arm-2)));return;}
            if(wmma && arm>=9){check(cudaError_t(panel_probe(&a,panels.p,s.id,arm-7)));return;}
            if(wmma && arm==8){check(cudaError_t(codebook_visit_probe(&a,s.id)));return;}
            if(wmma && arm>=6){check(cudaError_t(octet_big_probe(&a,s.id,arm-5)));return;}
            if(wmma && arm==4 && m>32){int used=0;require(!inference_cuda_inkling_panel_launch(&a,panels.p,size_t(k)*n*2,s.id,&stage,&used) && used,"borrowed panel admission");return;}
            if(wmma && arm>=4){check(cudaError_t(panel_probe(&a,panels.p,s.id,arm==5)));return;}
            if(wmma && arm==3){check(cudaError_t(octet_vector_probe(&a,s.id)));return;}
            if(wmma && arm==2){require(!inference_cuda_inkling_iq2_visit_launch(&a,s.id,&stage),"IQ2 visit");return;}
            if(arm && (codec==17 || arm==2)){check(cudaError_t(split_visit_probe(&a,s.id)));return;}
            if(arm) rc=inference_cuda_gguf_q8_row_reuse_width_launch(&a,s.id,&stage,&tile,-1);
            else if(codec==14 || expert) rc=inference_cuda_gguf_exact_expert_tile_launch(&a,s.id,&stage,&tile);
            else if(wmma) rc=inference_cuda_gguf_wmma_launch(&a,s.id,&stage);
            else rc=inference_cuda_dense_gemv_launch(&a,s.id,&stage);
            require(rc==0 && (!(codec==14 || expert) || tile==(arm?m:codec==42 && m>=4?4:2)),"extra dispatch");
        };
        for(int i=0;i<5;++i) launch();
        Event start,stop;check(cudaEventRecord(start.id,s.id));
        for(int i=0;i<32;++i) launch();
        check(cudaEventRecord(stop.id,s.id));check(cudaEventSynchronize(stop.id));
        float ms;check(cudaEventElapsedTime(&ms,start.id,stop.id)); double us=ms*1000./32;
        check(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));
        for(float v:out) require(std::isfinite(v),"nonfinite extra output");
        bool split=wmma?arm==1:codec==14 && (arm==2 || arm==5 || arm==13 || arm==14 || arm>=21);
        if(!arm) ref=out; else if(split) compare_approx(ref.data(),out.data(),m,n,k,n,false); else require(memcmp(ref.data(),out.data(),ref.size()*4)==0,"exact tile mismatch");
        size_t sweeps=codec==14 && arm>=17?(arm==20?(m+7)/8:(m+3)/4):codec==14 && arm>=15?(arm==15?(m+31)/32:(m+63)/64):codec==14 && arm>=13?(arm==13?(m+15)/16:(m+63)/64):codec==14 && arm==12?(m+31)/32:codec==14 && arm>=7?(m+7)/8:codec==14 && arm==6?(m+15)/16:codec==14 && arm>=3?1:wmma && arm==6?(m+63)/64:wmma && arm>=2?(m+31)/32:split?(m+15)/16:arm?(m+7)/8:codec==42 && m>=4?(m+3)/4:codec==14 || expert?(m+1)/2:wmma?(m+31)/32:m;
        size_t logical=bytes*sweeps;
        if(expert && arm==1 && m>=4){int bm=m==4?4:codec==16 && m>8?16:8;logical=bytes*((m+bm-1)/bm);}
        if(codec==0 && arm){const int bs[]={1,2,4,8,8,2,4,8,m<=21?2:8};logical=bytes*((m+bs[arm]-1)/bs[arm]);}
        if(codec==14 && arm>=21){int bm=arm>=23?16:8;logical=bytes*((m+bm-1)/bm)*(arm>=23?2:1);}
        if(codec==14 && arm==1){int bm=probe_prefetch_bm(&a);logical=bytes*((m+bm-1)/bm);}
        if(wmma && arm>=9){int bm=arm==9?32:64;size_t panel=size_t(k)*n*2;logical=bytes+panel+panel*((m+bm-1)/bm);fprintf(stderr,"widepanel codec=%d m=%d variant=%d us=%.3f\n",codec,m,arm-7,us);}
        if(wmma && (arm==4 || arm==5)){size_t panel_bytes=size_t(k)*n*2; logical=sweeps*panel_bytes+(arm==4?bytes+panel_bytes:0);fprintf(stderr,"panel codec=%d m=%d cached=%d us=%.3f logical=%zu\n",codec,m,arm==5,us,logical);}
        printf("%d,%d,%d,%s,%zu,%.3f,%.2f,%.4f,%d\n",m,n,k,codec==0 && arm?(arm==1?"down2":arm==2?"down4":arm==3?"down8":arm==4?"down8r2":arm==5?"direct2":arm==6?"direct4":arm==7?"direct8":"down_selected"):wmma && arm>=9?(arm==9?"panel32x64":arm==10?"panel64x32":"panel64x64"):wmma && arm==8?"lut_shared":wmma && arm>=6?(arm==6?"wide_m":"wide_k"):wmma && arm>=4?(arm==4?"panel_each":"panel_cached"):wmma && arm==3?(codec==17?"iqvec":codec==19?"iq1vec":codec==22?"iq2svec":"nvvec"):codec==14 && arm>=21?(arm==21?"q6imma0":arm==22?"q6imma1":arm==23?"q6imma2":"q6imma3"):codec==14 && arm>=3?(arm==3?"q6stream8":arm==4?"q6stream16":arm==5?"q6splitn":arm==6?"q6grid16":arm==7?"q6warp2":arm==8?"q6warp4":arm==9?"q6oct8":arm==10?"q6oct32":arm==11?"q6token8":arm==12?"q6token32":arm==13?"q6simt16":arm==14?"q6simt64":arm==15?"q6group32":arm==16?"q6group64":arm==17?"q6occ0":arm==18?"q6occ1":arm==19?"q6occ2":"q6occ3"):expert?(arm?"expert_visit":"expert_off"):wmma && arm==2?(codec==17?"iqvisit":codec==19?"iq1visit":codec==22?"iq2svisit":"nvvisit"):split?(codec==14?"q6split":"iqsplit"):arm?"q6visit":codec==14?"q6tile":wmma?"wmma":"dense",logical,us,logical/us/1000,2.*m*n*k/us/1e6,arm&&!split?m*n:0);
    }
}
void controller(const char *dir,int m){
    constexpr size_t bytes=64*2304*4;
    auto gain=load(std::string(dir)+"/gain.bin",bytes), forget=load(std::string(dir)+"/extra-0-64-2304.bin",bytes);
    Stream s; Buffer<uint8_t> wa(bytes),wb(bytes);Buffer<float> dx(m*64), dy(m*4608+2);
    std::vector<float> input(m*64); for(size_t i=0;i<input.size();++i) input[i]=float(int(i%127)-60)/64.f;
    check(cudaMemcpy(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(wa.p,gain.data(),bytes,cudaMemcpyHostToDevice));check(cudaMemcpy(wb.p,forget.data(),bytes,cudaMemcpyHostToDevice));
    inference_cuda_dense_gemv_args a{wa.p,dx.p,dy.p+1,m,2304,64,64,4608,3}, b=a; b.w=wb.p;b.y=dy.p+2305;
    std::vector<uint32_t> out(m*4608+2),ref;
    for(int arm=0;arm<2;++arm){
        auto launch=[&]{int stage=0,used=0;if(arm) require(!inference_cuda_inkling_controller_pair_launch(&a,&b,s.id,&stage,&used)&&used,"controller dispatch");
            else require(!inference_cuda_dense_gemv_launch(&a,s.id,&stage)&&!inference_cuda_dense_gemv_launch(&b,s.id,&stage),"baseline heads");};
        check(cudaMemsetAsync(dy.p,255,out.size()*4,s.id));for(int i=0;i<5;++i) launch();
        Event start,stop;check(cudaEventRecord(start.id,s.id));for(int i=0;i<32;++i) launch();check(cudaEventRecord(stop.id,s.id));check(cudaEventSynchronize(stop.id));
        float ms;check(cudaEventElapsedTime(&ms,start.id,stop.id));double us=ms*1000./32;
        check(cudaMemcpy(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost));if(!arm) ref=out;
        auto checkargs=a;checkargs.rows=4608;compare_bits(checkargs,ref,out);
        size_t logical=2*bytes*(arm?(m+7)/8:m);
        printf("%d,4608,64,%s,%zu,%.3f,%.2f,%.4f,%d\n",m,arm?"heads_pair":"heads_off",logical,us,logical/us/1000,2.*m*4608*64/us/1e6,arm?m*4608:0);
    }
}
#include "cap_probe.hpp"
#include "ragged_probe.hpp"
#include "mixed_probe.hpp"
#include "edge_probe.hpp"
#include "qkv_probe.hpp"
#include "interleave_probe.hpp"
#include "expert_group_probe.hpp"
int main(int argc,char **argv) {
    try {
        require(argc==2,"expected extracted weight directory");
        printf("m,n,k,arm,logical_weight_bytes,us,logical_weight_GBps,TFLOPs,exact_words\n");
        std::ifstream cases(std::string(argv[1])+"/mixed.txt"); int codec,block,size,n,k,mixed_m;
        while(cases>>codec>>block>>size>>n>>k>>mixed_m) mixed(argv[1],codec,block,size,n,k,mixed_m);
        for(int c:{16,17,19,22}) mixed(argv[1],c,256,c==16?66:c==17?74:c==19?50:82,2304,2304,1);
        for(int rows=2;rows<=16;rows++) mixed(argv[1],16,256,66,2304,2304,rows);
        for (auto s:kShapes) for (int m:{7,8,12,63,64}) {
            if (m==12 && s.cols==3072) continue;
            measure(argv[1],m,s.rows,s.cols,false);
            measure(argv[1],m,s.rows,s.cols,true);
        }
        for(int m:{8,63,64}) other(argv[1],m,3072,2304,14,256,210);
        for(int m:{1,32,33,64,65,96,97,128,160,192,224,256,288,320,352,384,448,480,512}) other(argv[1],m,2304,2304,17,256,74);
        for(int m:{1,32,64,96,128,160,192,224,256,288,320,352,384,416,448,480,512}) {
            other(argv[1],m,2304,2304,19,256,50); other(argv[1],m,2304,2304,22,256,82); other(argv[1],m,2304,1152,40,64,36);
        }
        for(int m=2;m<=32;++m){if(m<=16) other(argv[1],m,2304,2304,16,256,66); other(argv[1],m,2304,1152,42,64,18);}
        for(int m:{7,8,12}) for(int count:{2,3,4}) qkv(argv[1],m,count);
        for(auto shape:kShapes)for(int m:{7,8,63,64})interleave(argv[1],m,shape.rows,shape.cols,true);
        for(int m:{8,63,64}) interleave(argv[1],m,3072,2304,false);
        for(int c:{17,19,22,40}) for(int count:{2,4,8}) expert_batch(argv[1],c,count);
        for(int m:{127,128,255,256}){
            for(int c:{8,12,13,14})for(int n:{384,1024,3072})cap_probe(argv[1],c,n,2304,m);
            for(int c:{8,13,14})cap_probe(argv[1],c,2304,3072,m);
            cap_probe(argv[1],18,3072,2304,m);
        }
        for(int c:{16,42}) for(int b:{0,1,2}) for(int count:{2,4,8})ragged_exact(argv[1],c,count,b);
        expert_edge(argv[1]);
        for(int m:{6,7,8,21,63,64,127,128,255,256}) controller(argv[1],m);
        for(int m:{7,8,21,63,64,127,128,255,256}) { other(argv[1],m,2304,64,0,1,4); other(argv[1],m,64,4608,0,1,4); }
    } catch(const std::exception &e) { fprintf(stderr,"%s\n",e.what()); return 1; }
    return 0;
}
