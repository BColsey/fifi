// Standalone native fixture. It never calls a router, an approximate projection, or Rust dispatch.
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
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
constexpr int kExperts = 128, kRows = 2304, kRank = 8, kGate = 256 + 16, kDown = 256 + 42;

void require(bool ok, const char *what) {
    if (!ok) { std::fprintf(stderr, "%s\n", what); std::exit(1); }
}

void check(cudaError_t err) { require(err == cudaSuccess, cudaGetErrorString(err)); }

struct Stream {
    cudaStream_t s;
    Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { check(cudaStreamDestroy(s)); }
};

template <class T> struct Device {
    T *p;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    Device(const Device &) = delete;
    ~Device() { check(cudaFree(p)); }
    void upload(const std::vector<T> &v) {
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        // Pageable H2D can return before DMA completes. The nonblocking probe
        // stream does not wait for the default stream, so finish the upload here.
        check(cudaStreamSynchronize(nullptr));
    }
    void download(std::vector<T> &v) { check(cudaMemcpy(v.data(), p, v.size() * sizeof(T), cudaMemcpyDeviceToHost)); }
};

int launch(const inference_cuda_dense_gemv_args &a, cudaStream_t s, bool tiled) {
    int stage = -1, tile = 0;
    const int status = tiled ? inference_cuda_gguf_exact_expert_tile_launch(&a, s, &stage, &tile)
                             : inference_cuda_dense_gemv_launch(&a, s, &stage);
    if (status || stage != INFERENCE_CUDA_STAGE_NONE) {
        std::fprintf(stderr, "launch (tiled=%d, m=%d) status=%d stage=%d\n", tiled, a.m, status, stage);
        std::exit(1);
    }
    return tile;
}

void host_refusals() {
    alignas(16) unsigned char w[1024]{};
    alignas(16) float x[512]{}, y[32]{};
    const inference_cuda_dense_gemv_args good{w, x, y, 2, 9, 256, 256, 9, kGate};
    auto stream = reinterpret_cast<cudaStream_t>(w); // Never dereferenced: malformed calls only.
    auto refused = [&](const inference_cuda_dense_gemv_args *a, cudaStream_t s) {
        int stage = -1, tile = -1;
        require(inference_cuda_gguf_exact_expert_tile_launch(a, s, &stage, &tile) == INFERENCE_CUDA_INVALID_ARGUMENT &&
                stage == INFERENCE_CUDA_STAGE_NONE && tile == 0, "bad host-argument result");
    };
    refused(nullptr, stream);
    refused(&good, nullptr);
    auto bad = good;
    bad.cols = 255; refused(&bad, stream); // Fractional IQ2 block.
    bad = good; bad.w = w + 1; refused(&bad, stream);
    bad = good; bad.ld_x = 255; refused(&bad, stream);
    bad = good; bad.ld_y = 8; refused(&bad, stream);
    bad = good; bad.y = x + 511; refused(&bad, stream); // Crosses the last input element.
    bad = good; bad.y = w + 16; refused(&bad, stream);
    bad = good; bad.m = 0; refused(&bad, stream);
    bad = good; bad.m = 32769; refused(&bad, stream);
    bad = good; bad.weight_dtype = -1; refused(&bad, stream);
    require(std::all_of(y, y + 32, [](float v) { return v == 0.0f; }), "refusal wrote output");
    std::puts("host: 11 ABI refusals passed without CUDA calls");
}

uint32_t bits(float v) { uint32_t u; std::memcpy(&u, &v, sizeof(u)); return u; }

void compare(const std::vector<float> &ref, const std::vector<float> &new_y, int rows, int ld_y) {
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto a = bits(ref[i]), b = bits(new_y[i]);
        const bool pad = i % ld_y >= size_t(rows);
        if (a != b || (pad ? a != UINT32_MAX : !std::isfinite(ref[i]))) {
            std::fprintf(stderr, "mismatch t=%zu row=%zu bits=%08x/%08x padding=%d\n", i / ld_y, i % ld_y, a, b, pad);
            std::exit(1);
        }
    }
}

float activation(std::mt19937 &rng) {
    return float(int(rng() % 2000001) - 1000000) * (1.0f / 524288.0f);
}

int block_bytes(int kind, int cols) {
    if (kind == kGate) return cols / 256 * 66;
    if (kind == kDown) return cols / 64 * 18;
    return cols / 32 * 34; // Q8_0 fallback fixture.
}

void edge_case(cudaStream_t s, const std::vector<unsigned char> &real, int kind,
               int m, int rows, int cols, int want_tile) {
    const size_t wbytes = size_t(rows) * block_bytes(kind, cols);
    require(wbytes <= real.size(), "edge fixture weight slice too short");
    std::vector<unsigned char> weights(2 + wbytes, 0xa5); // 2-byte aligned, not 4/16.
    std::copy_n(real.begin(), wbytes, weights.begin() + 2);
    const int ld_x = cols + 5, ld_y = rows + 7;
    const size_t nx = size_t(m - 1) * ld_x + cols, ny = size_t(m) * ld_y;
    std::vector<float> x(nx, std::numeric_limits<float>::quiet_NaN()), left(ny + ld_y), right(ny + ld_y);
    std::mt19937 rng(125);
    for (int t = 0; t < m; ++t)
        for (int c = 0; c < cols; ++c) x[size_t(t) * ld_x + c] = activation(rng);
    Device<unsigned char> w(weights.size()); w.upload(weights);
    Device<float> dx(x.size()), a(ny + ld_y), b(ny + ld_y); dx.upload(x);
    // Leading guard row and trailing (per-row) stride padding must stay unwritten.
    check(cudaMemsetAsync(a.p, 0xff, (ny + ld_y) * sizeof(float), s));
    check(cudaMemsetAsync(b.p, 0xff, (ny + ld_y) * sizeof(float), s));
    inference_cuda_dense_gemv_args call{w.p + 2, dx.p, a.p + ld_y, m, rows, cols, ld_x, ld_y, kind};
    launch(call, s, false);
    call.y = b.p + ld_y;
    require(launch(call, s, true) == want_tile, "unexpected edge dispatch");
    check(cudaStreamSynchronize(s));
    a.download(left); b.download(right);
    for (int i = 0; i < ld_y; ++i)
        require(bits(left[i]) == UINT32_MAX && bits(right[i]) == UINT32_MAX, "output leading guard changed");
    left.erase(left.begin(), left.begin() + ld_y); right.erase(right.begin(), right.begin() + ld_y);
    compare(left, right, rows, ld_y);
    std::vector<float> read_x(nx); dx.download(read_x);
    std::vector<unsigned char> read_w(weights.size()); w.download(read_w);
    require(std::memcmp(x.data(), read_x.data(), nx * sizeof(float)) == 0 && weights == read_w, "input modified");
}

struct Routing {
    std::array<std::vector<int>, kExperts> groups;
    Routing(int tokens, unsigned seed, bool skew) {
        std::mt19937 rng(seed);
        for (int t = 0; t < tokens; ++t) {
            bool used[kExperts]{};
            for (int rank = 0; rank < kRank; ++rank) {
                unsigned id;
                do { id = rng() % (skew && rank < 3 ? 24 : kExperts); } while (used[id]);
                used[id] = true;
                groups[id].push_back(t * kRank + rank);
            }
        }
    }
};

void run(const char *model, std::streamoff offset, int kind, int tokens, unsigned seed, bool skew, int iters) {
    const bool gate = kind == kGate;
    const int cols = gate ? 2304 : 1152, picks = tokens * kRank;
    const size_t expert_bytes = size_t(kRows) * block_bytes(kind, cols);
    std::vector<unsigned char> raw(kExperts * expert_bytes);
    std::ifstream in(model, std::ios::binary);
    require(bool(in.seekg(offset).read(reinterpret_cast<char *>(raw.data()), raw.size())), "GGUF tensor read failed");
    Stream stream;
    for (int m : {1, 2, 3, 4, 5, 7})
        edge_case(stream.s, raw, kind, m, 9, cols, m == 1 ? 0 : !gate && m >= 4 ? 4 : 2);
    if (gate) {
        edge_case(stream.s, raw, kind, 512, 9, 256, 2);
        edge_case(stream.s, raw, kind, 513, 9, 256, 0);
        edge_case(stream.s, raw, kind, 2, 2305, 256, 0);
        edge_case(stream.s, raw, kind, 2, 9, 2560, 0);
        std::vector<unsigned char> q8(9 * 34, 0x03);
        for (int r = 0; r < 9; ++r) { q8[r * 34] = 0; q8[r * 34 + 1] = 0x3c; } // d=1
        edge_case(stream.s, q8, 256 + 8, 2, 9, 32, 0);
    } else {
        edge_case(stream.s, raw, kind, 5, 11, 1216, 2); // 4 rows would exceed the smem bound.
    }
    const Routing route(tokens, seed, skew);
    std::mt19937 rng(seed + 1);
    std::vector<float> x(size_t(gate ? tokens : picks) * cols), sorted(size_t(picks) * cols);
    for (auto &v : x) v = activation(rng);
    int at = 0, min_m = tokens, max_m = 0;
    for (const auto &group : route.groups) {
        min_m = std::min(min_m, int(group.size())); max_m = std::max(max_m, int(group.size()));
        for (int slot : group) {
            std::copy_n(x.data() + size_t(gate ? slot / kRank : slot) * cols, cols,
                        sorted.data() + size_t(at++) * cols);
        }
    }
    require(at == picks, "incorrect route count");
    const size_t n = size_t(picks) * kRows;
    Device<unsigned char> w(raw.size()); w.upload(raw);
    Device<float> dx(sorted.size()), a(n), b(n); dx.upload(sorted);
    int tiled = 0, fallback = 0;
    auto enqueue = [&](bool candidate, bool verify_path) {
        int first = 0;
        for (int e = 0; e < kExperts; ++e) {
            const int m = route.groups[e].size();
            if (!m) continue;
            const inference_cuda_dense_gemv_args call{
                w.p + size_t(e) * expert_bytes, dx.p + size_t(first) * cols,
                (candidate ? b.p : a.p) + size_t(first) * kRows, m, kRows, cols, cols, kRows, kind};
            const int tile = launch(call, stream.s, candidate);
            if (verify_path) {
                require(tile == (m == 1 ? 0 : !gate && m >= 4 ? 4 : 2), "unexpected real-weight dispatch");
                if (tile) ++tiled; else ++fallback;
            }
            first += m;
        }
    };
    check(cudaMemsetAsync(a.p, 0xff, n * sizeof(float), stream.s));
    check(cudaMemsetAsync(b.p, 0xff, n * sizeof(float), stream.s));
    enqueue(false, false); enqueue(true, true); check(cudaStreamSynchronize(stream.s));
    {
        std::vector<float> left(n), right(n); a.download(left); b.download(right);
        compare(left, right, kRows, kRows);
    }
    std::printf("{\"codec\":\"%s\",\"tokens\":%d,\"seed\":%u,\"skew\":%d,\"min_group\":%d,\"max_group\":%d,"
                "\"edge_cases\":%d,\"tiled\":%d,\"fallback\":%d,\"bitwise\":%zu,\"iters\":%d,\"ms\":[",
                gate ? "IQ2_XXS" : "Q2_0", tokens, seed, skew, min_m, max_m, gate ? 11 : 7,
                tiled, fallback, n, iters);
    if (iters) {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        const bool order[] = {false, true, true, false, true, false, false, true};
        bool comma = false;
        for (bool candidate : order) {
            check(cudaEventRecord(start, stream.s));
            for (int i = 0; i < iters; ++i) enqueue(candidate, false);
            check(cudaEventRecord(stop, stream.s)); check(cudaEventSynchronize(stop));
            float ms;
            check(cudaEventElapsedTime(&ms, start, stop));
            std::printf("%s[\"%s\",%.7g]", comma ? "," : "", candidate ? "exact_tile" : "packed_gemv", ms / iters);
            comma = true;
        }
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    }
    std::puts("]}");
}
}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
        host_refusals();
        return 0;
    }
    require(argc == 8, "usage: probe MODEL GATE_OFFSET DOWN_OFFSET TOKENS SEED SKEW ITERS (0 verifies only)");
    const auto gate_offset = std::stoll(argv[2]), down_offset = std::stoll(argv[3]);
    const int tokens = std::stoi(argv[4]), skew = std::stoi(argv[6]), iters = std::stoi(argv[7]);
    const unsigned seed = std::stoul(argv[5]);
    require(gate_offset >= 0 && down_offset >= 0 && tokens > 0 && tokens <= 512 &&
            (skew == 0 || skew == 1) && iters >= 0 && iters <= 100, "invalid fixture arguments");
    run(argv[1], gate_offset, kGate, tokens, seed, skew, iters);
    run(argv[1], down_offset, kDown, tokens, seed, skew, iters);
}
