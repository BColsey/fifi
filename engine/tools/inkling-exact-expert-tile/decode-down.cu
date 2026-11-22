// Standalone native MoE fixture: real packed weights, device picks, synthetic activations.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "abi.h"
#include "resident/dense_gemv.h"

namespace {
constexpr int kWidth = 2304, kMiddle = 1152, kExperts = 128, kCount = 8;
constexpr int kGate = 256 + 16, kDown = 256 + 42;
constexpr int kExact = INFERENCE_CUDA_GGUF_MOE_EXACT_Q2_DOWN;
constexpr size_t kWork = kCount * (3 * kMiddle + kWidth);
static_assert(kWork == 46080);

void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}
void check(cudaError_t err) { require(err == cudaSuccess, cudaGetErrorString(err)); }

void host_refusals() {
    // Nonoverlapping mock address ranges. None of these invalid calls reaches CUDA.
    const auto addr = [](std::uintptr_t n) { return reinterpret_cast<void *>(n); };
    const auto floats = [&](std::uintptr_t n) { return static_cast<float *>(addr(n)); };
    inference_cuda_gguf_moe_args a{};
    a.gate = addr(0x10000000); a.down = addr(0x20000000);
    a.x = floats(0x40000000); a.scratch = floats(0x41000000); a.y = floats(0x42000000);
    a.device_ids = static_cast<int *>(addr(0x43000000));
    a.device_weights = floats(0x44000000);
    a.gate_scales = floats(0x45000000); a.down_scales = floats(0x46000000);
    a.hidden = kWidth; a.middle = kMiddle; a.count = kCount; a.experts = kExperts;
    a.gate_dtype = kGate; a.down_dtype = kDown; a.gate_cap = 2.0f; a.up_cap = 3.0f;
    auto *stream = addr(0x47000000);
    const auto refused = [&](const inference_cuda_gguf_moe_args *call, void *s) {
        int stage = -1, schedule = kExact;
        require(inference_cuda_gguf_moe_launch(call, nullptr, s, &stage, &schedule) ==
                    INFERENCE_CUDA_INVALID_ARGUMENT && stage == INFERENCE_CUDA_STAGE_NONE,
                "bad host-argument result");
    };
    refused(nullptr, stream);
    refused(&a, nullptr);
    auto bad = a;
    bad.middle = kMiddle - 1; refused(&bad, stream); // Fractional Q2_0 block.
    bad = a; bad.down = addr(0x20000001); refused(&bad, stream);
    bad = a; bad.down = a.scratch; refused(&bad, stream);
    bad = a; bad.down_scales = nullptr; refused(&bad, stream);
    bad = a; bad.device_ids = static_cast<int *>(addr(0x43000001)); refused(&bad, stream);
    bad = a; bad.up_cap = 0.0f; refused(&bad, stream);
    std::puts("{\"host_refusals\":8}");
}

struct Stream {
    cudaStream_t s;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { check(cudaStreamDestroy(s)); }
};

template<class T> struct Device {
    T *p;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    Device(const Device &) = delete;
    ~Device() { check(cudaFree(p)); }
    void upload(cudaStream_t s, const std::vector<T> &host) {
        check(cudaMemcpyAsync(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, s));
        check(cudaStreamSynchronize(s));
    }
    std::vector<T> read(cudaStream_t s, size_t n) {
        std::vector<T> host(n);
        check(cudaMemcpyAsync(host.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost, s));
        check(cudaStreamSynchronize(s));
        return host;
    }
};

std::vector<unsigned char> tensor(const char *model, std::streamoff offset, size_t bytes) {
    // Exercise valid 2-byte, not 4-byte, base alignment.
    std::vector<unsigned char> result(bytes + 2);
    std::ifstream in(model, std::ios::binary);
    require(bool(in.seekg(offset).read(reinterpret_cast<char *>(result.data() + 2), bytes)),
            "GGUF tensor read failed");
    return result;
}

void launch(const inference_cuda_gguf_moe_args &a, cudaStream_t s, int request, int want) {
    int stage = -1, got = request;
    int status = inference_cuda_gguf_moe_launch(&a, nullptr, s, &stage, request ? &got : nullptr);
    if (status || stage != INFERENCE_CUDA_STAGE_NONE || got != want) {
        std::fprintf(stderr, "request=%d got=%d want=%d stage=%d status=%d\n", request, got, want, stage, status);
        std::exit(1);
    }
}

void same_finite(const std::vector<float> &a, const std::vector<float> &b) {
    require(a.size() == b.size(), "unexpected comparison size");
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || std::memcmp(&a[i], &b[i], sizeof(float))) {
            std::fprintf(stderr, "nonfinite or nonbitwise result at %zu: %.9g / %.9g\n", i, a[i], b[i]);
            std::exit(1);
        }
    }
}

struct Inputs {
    std::vector<float> x, weights, gate_scales, down_scales;
    std::vector<int> ids;
    explicit Inputs(unsigned seed) : x(kWidth), weights(kCount), gate_scales(kExperts),
                                      down_scales(kExperts), ids(kExperts) {
        std::mt19937 rng(seed);
        for (float &v : x) v = (int(rng() % 200001) - 100000) * (1.0f / 65536.0f);
        for (int i = 0; i < kExperts; ++i) {
            gate_scales[i] = 0.25f + (i % 11) * 0.125f;
            down_scales[i] = 0.5f + (i % 7) * 0.125f;
            ids[i] = i;
        }
        for (int i = 0; i < kCount; ++i) {
            std::swap(ids[i], ids[i + rng() % (kExperts - i)]);
            weights[i] = (i + 1) / 36.0f;
        }
        ids.resize(kCount);
    }
};

struct Runner {
    Stream stream;
    Device<float> x{kWidth}, y{kWidth}, work{kWork}, weights{kCount}, gs{kExperts}, ds{kExperts};
    Device<int> ids{kCount};
    Device<unsigned char> gate, down;
    inference_cuda_gguf_moe_args a{};
    Runner(const std::vector<unsigned char> &gate_host, const std::vector<unsigned char> &down_host,
           const Inputs &in) : gate(gate_host.size()), down(down_host.size()) {
        auto s = stream.s;
        gate.upload(s, gate_host); down.upload(s, down_host);
        x.upload(s, in.x); ids.upload(s, in.ids); weights.upload(s, in.weights);
        gs.upload(s, in.gate_scales); ds.upload(s, in.down_scales);
        a.gate = gate.p + 2; a.down = down.p + 2;
        a.x = x.p; a.scratch = work.p; a.y = y.p;
        a.hidden = kWidth; a.middle = kMiddle; a.experts = kExperts; a.count = kCount;
        a.gate_dtype = kGate; a.down_dtype = kDown; a.gate_cap = 2.0f; a.up_cap = 3.0f;
        a.device_ids = ids.p; a.device_weights = weights.p;
        a.gate_scales = gs.p; a.down_scales = ds.p;
    }
    void parity(int gate_schedule, int exact_used) {
        auto s = stream.s;
        // Sentinels detect incomplete writes even when a prior run filled the buffer.
        check(cudaMemsetAsync(work.p, 0xff, kWork * sizeof(float), s));
        check(cudaMemsetAsync(y.p, 0xff, kWidth * sizeof(float), s));
        launch(a, s, gate_schedule, gate_schedule);
        const auto reference = work.read(s, a.count * (3 * a.middle + a.hidden));
        const auto output = y.read(s, kWidth);
        check(cudaMemsetAsync(work.p, 0xff, kWork * sizeof(float), s));
        check(cudaMemsetAsync(y.p, 0xff, kWidth * sizeof(float), s));
        launch(a, s, gate_schedule | kExact, gate_schedule | exact_used);
        same_finite(reference, work.read(s, reference.size()));
        same_finite(output, y.read(s, kWidth));
    }
    void health() {
        for (int arm : {2, 2 | kExact}) {
            launch(a, stream.s, arm, arm);
            const auto v = y.read(stream.s, kWidth);
            require(std::all_of(v.begin(), v.end(), [](float x) { return std::isnan(x); }),
                    "health case did not report NaN");
        }
    }
    void timing(int iters) {
        if (iters == 0) return;
        auto s = stream.s;
        cudaEvent_t begin, end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        for (int i = 0; i < 32; ++i) {
            launch(a, s, 2, 2); launch(a, s, 2 | kExact, 2 | kExact);
        }
        for (int candidate : {0, 1, 1, 0, 1, 0, 0, 1}) {
            const int schedule = 2 | (candidate ? kExact : 0);
            check(cudaEventRecord(begin, s));
            for (int i = 0; i < iters; ++i) launch(a, s, schedule, schedule);
            check(cudaEventRecord(end, s)); check(cudaEventSynchronize(end));
            float ms;
            check(cudaEventElapsedTime(&ms, begin, end));
            std::printf("{\"exact_q2_down\":%s,\"moe_us\":%.3f,\"iterations\":%d}\n",
                        candidate ? "true" : "false", ms * 1000 / iters, iters);
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
    }
};

void verify(Runner &r, Inputs in) {
    for (int gate : {0, 1, 2}) r.parity(gate, kExact);
    r.a.count = kCount - 1;
    r.parity(0, 0); // Device picks, valid unsupported shape.
    r.a.count = kCount;
    const auto save = r.a;
    r.a.device_ids = nullptr; r.a.device_weights = nullptr;
    r.a.gate_scales = nullptr; r.a.down_scales = nullptr;
    for (int i = 0; i < kCount; ++i) {
        const unsigned id = in.ids[i];
        r.a.picks[i] = {id, in.gate_scales[id], in.down_scales[id], in.weights[i]};
    }
    r.parity(0, 0); // Host-selected stacks fall back even at target geometry.
    r.a = save;
    auto s = r.stream.s;
    auto invalid = in.ids;
    invalid[3] = -1; r.ids.upload(s, invalid); r.health();
    invalid[3] = kExperts; r.ids.upload(s, invalid); r.health();
    r.ids.upload(s, in.ids);
    auto bad_x = in.x;
    bad_x[17] = std::numeric_limits<float>::infinity(); r.x.upload(s, bad_x); r.health();
    r.x.upload(s, in.x);
    auto bad_scale = in.down_scales;
    bad_scale[in.ids[2]] = std::numeric_limits<float>::quiet_NaN(); r.ds.upload(s, bad_scale); r.health();
    r.ds.upload(s, in.down_scales);
    std::puts("{\"bitwise_outputs_per_gate\":2304,\"bitwise_scratch_per_gate\":46080,"
              "\"gate_schedules\":3,\"fallbacks\":2,\"health_cases\":4}");
}
} // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        host_refusals();
        return 0;
    }
    require(argc == 6, "usage: decode-down --host-only | model gate-offset down-offset seed iterations");
    const int iters = std::stoi(argv[5]);
    require(iters >= 0 && iters <= 5000, "expected 0..5000 timing iterations");
    int gpu;
    cudaDeviceProp prop{};
    check(cudaGetDevice(&gpu)); check(cudaGetDeviceProperties(&prop, gpu));
    require(prop.major == 7 && prop.minor == 5, "SM75 required for this fixture");
    const Inputs in(std::stoul(argv[4]));
    Runner r(tensor(argv[1], std::stoll(argv[2]), size_t(kExperts) * kWidth * (kWidth / 256) * 66),
             tensor(argv[1], std::stoll(argv[3]), size_t(kExperts) * kWidth * (kMiddle / 64) * 18), in);
    verify(r, in);
    r.timing(iters);
}
