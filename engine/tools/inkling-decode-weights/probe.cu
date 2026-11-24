// Compile this fixture alone: the production kernels have internal linkage.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>
#include "resident/dense_gemv.cu"

#define OK(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { std::cerr << #expr << ": " << cudaGetErrorString(err) << '\n'; exit(2); } } while (0)

template<class T> T *upload(const std::vector<T> &v) {
    T *p; OK(cudaMalloc(&p, v.size()*sizeof(T)));
    OK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    std::ifstream model(argv[1],std::ios::binary), manifest(argv[2]);
    cudaStream_t stream; OK(cudaStreamCreate(&stream));
    cudaEvent_t start,end; OK(cudaEventCreate(&start)); OK(cudaEventCreate(&end));
    std::string name; int t,c,r,e; size_t offset;
    std::set<std::tuple<int,int,int>> seen;
    int failures = 0;
    const char *variant=std::getenv("FIFI_PROBE_CASE");
    const std::string test_case=variant ? variant : "waves";
    const bool unaligned=test_case=="unaligned";
    const int iters=std::getenv("FIFI_PROBE_CHECK_ONLY") ? 1 : 100;
    if (!unaligned && test_case!="waves" && test_case!="random" && test_case!="impulses" && test_case!="zeros") return 2;
    std::cerr << "case=" << test_case << " experts=" << (test_case=="waves"?"first":"last") << '\n';
    std::cout << "tensor type cols rows picks bytes off_us on_us exact total max_abs failed\n";
    while (manifest >> name >> t >> c >> r >> e >> offset) {
        if (t==0 && !(c==4608 || c==64)) continue;
        if (t==30 && c!=2304) continue;
        if (t==23) r=std::getenv("FIFI_PROBE_UNPADDED") ? 200058 : 200064; // model width vs rounded trace geometry
        if (!seen.emplace(t,c,r).second) continue;
        int picks=e>1?8:1, dtype=t==0?3:t==30?1:t==40?2:t+256;
        size_t stride=weight_row_bytes(dtype,c)*r, bytes=stride*picks;
        std::vector<uint8_t> weights(bytes);
        const size_t selection=(test_case!="waves" && e>picks) ? stride*(e-picks) : 0;
        model.seekg(offset+selection); if (!model.read((char*)weights.data(),bytes)) return 3;
        // Rotate identical matrix copies beyond L2 to avoid a cache-only GEMV benchmark.
        const int copies = (12*1024*1024 + bytes - 1)/bytes;
        // The public entry permits two-byte packed weights and four-byte F32.
        const size_t pad=unaligned ? (t==0 ? 4 : 2) : 0;
        uint8_t *allocation; OK(cudaMalloc(&allocation,copies*bytes+pad));
        auto *base=allocation+pad;
        for(int n=0;n<copies;++n) OK(cudaMemcpy(base+n*bytes,weights.data(),bytes,cudaMemcpyHostToDevice));
        auto *ids=upload(std::vector<int>{7,1,5,0,6,3,2,4});
        DeviceSelectedWeights idx{stride,ids,8};
        bool gate=e>1 && c==2304;
        std::vector<float> host_x(c*(gate?1:picks));
        uint32_t rng=0x8913501u;
        for (size_t i=0;i<host_x.size();++i) {
            rng^=rng<<13; rng^=rng>>17; rng^=rng<<5;
            if (test_case=="random") host_x[i]=float(rng>>8)*(4.f/16777216.f)-2.f;
            else if (test_case=="impulses") host_x[i]=(i%53==0 ? -1.25f : i%71==0 ? 2.5f : 0.f);
            else if (test_case=="zeros") host_x[i]=0.f;
            else host_x[i]=sinf(i*0.017f)+cosf(i*1.324f);
        }
        if(unaligned) host_x.insert(host_x.begin(),0.f);
        float *x_base=upload(host_x), *x=x_base+int(unaligned), *y=upload(std::vector<float>(r*picks));
        GateDigits *digits=nullptr;
        if (gate) { OK(cudaMalloc(&digits,288*sizeof(*digits))); inkling_gate_pack<<<3,128,0,stream>>>(x,digits); }
        std::vector<float> reference(r*picks), candidate(r*picks);
        float times[2];
        const bool reverse=std::getenv("FIFI_PROBE_REVERSE")!=nullptr;
        for(int step=0;step<2;++step) {
            const int arm=reverse ? 1-step : step;
            setenv("FIFI_INKLING_CUDA_DECODE_SPLIT",arm?"1":"0",1);
            int iteration=0;
            auto run=[&]() {
                auto *w=base + bytes*(iteration++ % copies);
                inference_cuda_dense_gemv_args a{w,x,y,picks,r,c,gate?0:c,r,dtype};
                Failure f;
                if (gate) {
#define G(id,n) case id: if (arm) launch_decode_gate<id,n>(digits,w,y,idx,stream); else inkling_gate_dot<id,n><<<dim3(r/8,8),256,0,stream>>>(digits,w,y,idx); break;
                    switch(t) { G(16,66) G(17,74) G(19,50) G(22,82) default: exit(3); }
#undef G
                } else if (t==42 && arm && aligned(x,16)) decode_q2_pair<<<dim3(r/32,8),256,0,stream>>>(w,x,y,idx);
                else if (t==42) inkling_q2_down<<<dim3(r/kQ2RowsPerCta,8),256,0,stream>>>(w,x,y,idx);
                else if (t==30) launch<__nv_bfloat16>(a,stream,f);
                else if (t==0) launch<float>(a,stream,f);
                else if (picks>1) launch_packed(a,stream,f,idx);
                else launch_packed(a,stream,f,SharedWeights{});
                if (!f.ok()) exit(2);
            };
            OK(cudaMemsetAsync(y,0xff,reference.size()*4,stream));
            for(int i=0;i<(iters==1?0:10);++i) run();
            OK(cudaEventRecord(start,stream));
            for(int i=0;i<iters;++i) run();
            OK(cudaEventRecord(end,stream)); OK(cudaEventSynchronize(end));
            OK(cudaEventElapsedTime(&times[arm],start,end)); times[arm]*=1000.f/iters;
            OK(cudaMemcpy((arm?candidate:reference).data(),y,reference.size()*4,cudaMemcpyDeviceToHost));
        }
        size_t exact=0, bad=0; float max_abs=0;
        for(size_t i=0;i<reference.size();++i) {
            float delta=fabsf(candidate[i]-reference[i]); max_abs=fmaxf(max_abs,delta);
            exact+=memcmp(&candidate[i],&reference[i],4)==0;
            bad+=!std::isfinite(delta) || delta>1e-4f+2e-5f*fabsf(reference[i]);
        }
        failures+=bad!=0;
        std::cout << name << ' ' << t << ' ' << c << ' ' << r << ' ' << picks << ' ' << bytes << ' ' << times[0] << ' ' << times[1] << ' ' << exact << ' ' << reference.size() << ' ' << max_abs << ' ' << bad << std::endl;
        OK(cudaFree(digits)); OK(cudaFree(ids)); OK(cudaFree(y)); OK(cudaFree(x_base)); OK(cudaFree(allocation));
    }
    std::cerr << "shapes=" << seen.size() << " failures=" << failures << '\n';
    return failures || seen.size()!=27;
}
