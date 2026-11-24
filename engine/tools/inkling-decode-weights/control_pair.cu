// Compare two sequential real-weight calls with the fused controller pair.
#include <array>
#include <fstream>
#include <iostream>
#include <vector>
#include "resident/dense_gemv.cu"
#define REQUIRE(x) do { if (!(x)) { std::cerr << "failed: " #x << '\n'; return 2; } } while (0)
#define OK(x) REQUIRE((x) == cudaSuccess)

int main(int argc, char **argv) {
    REQUIRE(argc == 3);
    std::ifstream model(argv[1],std::ios::binary), manifest(argv[2]);
    constexpr int n = 2304, c = 64, copies = 12;
    constexpr size_t stride = n*c*sizeof(float);
    std::array<float *,2> w{}, y{};
    float *x;
    OK(cudaMalloc(&x,c*4));
    std::array<float,c> hx;
    for(int i=0;i<c;++i) hx[i]=sinf(i*.017f)+cosf(i*1.324f);
    OK(cudaMemcpy(x,hx.data(),c*4,cudaMemcpyHostToDevice));
    for(int j=0;j<2;j++) {
        std::string name; int dtype,cols,rows,experts; size_t offset;
        REQUIRE(manifest >> name >> dtype >> cols >> rows >> experts >> offset);
        REQUIRE(dtype==0 && cols==c && rows==n && experts==1);
        std::vector<float> weights(n*c); model.seekg(offset);
        REQUIRE(model.read(reinterpret_cast<char *>(weights.data()),stride));
        OK(cudaMalloc(&w[j],stride*copies));
        for(int i=0;i<copies;i++) OK(cudaMemcpy(w[j]+i*n*c,weights.data(),stride,cudaMemcpyHostToDevice));
        OK(cudaMalloc(&y[j],n*4));
    }
    cudaStream_t s; cudaEvent_t t0,t1;
    OK(cudaStreamCreate(&s)); OK(cudaEventCreate(&t0)); OK(cudaEventCreate(&t1));
    std::array<std::vector<float>,2> ref{std::vector<float>(n),std::vector<float>(n)};
    for(int arm : {0,1,1,0}) {
        setenv("FIFI_INKLING_CUDA_DECODE_SPLIT",arm?"1":"0",1);
        for(int j=0;j<2;j++) OK(cudaMemsetAsync(y[j],0xff,n*4,s));
        for(int i=0;i<110;i++) {
            if(i==10) OK(cudaEventRecord(t0,s));
            inference_cuda_dense_gemv_args a[2]; int stage;
            for(int j=0;j<2;j++) a[j]={w[j]+(i%copies)*n*c,x,y[j],1,n,c,c,n,kDtypeF32};
            if(arm) REQUIRE(inference_cuda_decode_control_pair(&a[0],&a[1],s,&stage)==0);
            else for(auto &v:a) REQUIRE(inference_cuda_dense_gemv_launch(&v,s,&stage)==0);
        }
        OK(cudaEventRecord(t1,s)); OK(cudaEventSynchronize(t1)); float ms;
        OK(cudaEventElapsedTime(&ms,t0,t1));
        int same=0;
        for(int j=0;j<2;j++) {
            std::vector<float> host(n); OK(cudaMemcpy(host.data(),y[j],n*4,cudaMemcpyDeviceToHost));
            if(!arm) ref[j]=host;
            for(int k=0;k<n;k++) same+=std::isfinite(host[k]) && memcmp(&host[k],&ref[j][k],4)==0;
        }
        std::cout << "arm=" << arm << " pair_us=" << 10*ms << " bytes=" << 2*stride << " GB/s=" << 2*stride/(ms*10000) << " exact=" << same << '/' << 2*n << '\n';
        REQUIRE(same==2*n);
    }
}
