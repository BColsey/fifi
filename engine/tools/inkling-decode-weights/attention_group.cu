#include <array>
#include <map>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "resident/dense_gemv.cu"
// ABBA: separate split calls versus one fused launch; identical per-row arithmetic.
#define CK(op) do {auto err=(op); if(err!=cudaSuccess){std::cerr<<#op<<": "<<cudaGetErrorString(err)<<'\n';return 2;}}while(0)
struct Spec{int ty,c,n,e;size_t off;};
int main(int argc,char **argv){
    if(argc!=3)return 1;
    std::ifstream f(argv[1],std::ios::binary),man(argv[2]);
    std::map<std::string,Spec> specs;{std::string name;Spec p;while(man>>name>>p.ty>>p.c>>p.n>>p.e>>p.off)specs[name]=p;}
    std::array<std::string,8> names={"attn_q","attn_k","attn_v","attn_rel","attn_sink_proj","attn_gate","attn_k_conv_proj","attn_v_conv_proj"};
    constexpr int copies=2,cols=2304;
    cudaStream_t s;cudaEvent_t t0,t1;CK(cudaStreamCreate(&s));CK(cudaEventCreate(&t0));CK(cudaEventCreate(&t1));
    setenv("FIFI_INKLING_CUDA_DECODE_SPLIT","1",1);
    for(int layer=0;layer<24;layer++){
        int count=specs.count("blk."+std::to_string(layer)+".attn_k_conv_proj.weight") ? 8 : 6;
        std::vector<uint8_t *> ws(count);std::array<float *,3> xs{};std::vector<float *> ys(count);std::vector<size_t> bytes(count);std::vector<int> rows(count),types(count);
        size_t total=0;bool valid=true;
        for(int j=0;j<count;j++) {auto p=specs["blk."+std::to_string(layer)+"."+names[j]+".weight"];rows[j]=p.n;types[j]=p.ty;
            if(p.c!=cols||p.e!=1){valid=false;break;}
            int rb=p.ty==8?cols/32*34:p.ty==12?cols/256*144:p.ty==13?cols/256*176:p.ty==14?cols/256*210:p.ty==18?cols/256*98:p.ty==30?cols*2:0;
            if(!rb){valid=false;break;}bytes[j]=size_t(rb)*p.n;total+=bytes[j];}
        if(!valid)return 6;
        for(int j=0;j<3;j++){CK(cudaMalloc(&xs[j],cols*4));std::vector<float>x(cols);for(int i=0;i<cols;i++)x[i]=sinf((i+j)*.013f)+cosf((i-j)*.097f);CK(cudaMemcpy(xs[j],x.data(),cols*4,cudaMemcpyHostToDevice));}
        for(int j=0;j<count;j++) {auto p=specs.at("blk."+std::to_string(layer)+"."+names[j]+".weight");std::vector<uint8_t>w(bytes[j]);f.seekg(p.off);if(!f.read((char*)w.data(),bytes[j]))return 3;
            CK(cudaMalloc(&ws[j],copies*bytes[j]));for(int k=0;k<copies;k++)CK(cudaMemcpy(ws[j]+k*bytes[j],w.data(),bytes[j],cudaMemcpyHostToDevice));CK(cudaMalloc(&ys[j],p.n*4));}
        std::vector<std::vector<float>> ref(count);
        for(int arm:{0,1,1,0}){
            for(int j=0;j<count;j++)CK(cudaMemsetAsync(ys[j],0xff,rows[j]*4,s));
            for(int i=0;i<110;i++){if(i==10)CK(cudaEventRecord(t0,s));inference_cuda_dense_gemv_args a[8];int stage;
                for(int j=0;j<count;j++) a[j]={ws[j]+(i%copies)*bytes[j],xs[j==1||j==6?1:j==2||j==7?2:0],ys[j],1,rows[j],cols,cols,rows[j],types[j]==30?kDtypeBf16:fifi_gguf::kDtypeBase+types[j]};
                if(!arm){for(int j=0;j<count;j++)if(inference_cuda_dense_gemv_launch(&a[j],s,&stage))return 4;}
                else if(inference_cuda_decode_attention_group(a,count,s,&stage))return 4;
            }
            CK(cudaEventRecord(t1,s));CK(cudaEventSynchronize(t1));float ms;CK(cudaEventElapsedTime(&ms,t0,t1));int exact=0,words=0,bad=0;double worst=0;
            for(int j=0;j<count;j++){std::vector<float> y(rows[j]);CK(cudaMemcpy(y.data(),ys[j],rows[j]*4,cudaMemcpyDeviceToHost));if(ref[j].empty())ref[j]=y;for(int k=0;k<rows[j];k++){double d=std::abs(y[k]-ref[j][k]);worst=std::max(worst,d);bad+=!std::isfinite(y[k])||d>1e-4+2e-5*std::abs(ref[j][k]);exact+=!memcmp(&y[k],&ref[j][k],4);words++;}}
            std::cout<<"layer="<<layer<<" arm="<<arm<<" us="<<ms*10<<" bytes="<<total<<" exact="<<exact<<'/'<<words<<" max="<<worst<<" bad="<<bad<<'\n';if(bad||exact!=words)return 5;
        }
        for(auto p:ws) CK(cudaFree(p));for(auto p:ys)CK(cudaFree(p));for(auto p:xs)CK(cudaFree(p));
    }
}
