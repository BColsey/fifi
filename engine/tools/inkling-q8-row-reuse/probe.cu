// Same original packed GGUF weight tensor and F32 inputs in both arms.
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "abi.h"
#include "resident/dense_gemv.h"

namespace {

constexpr int kQ8 = 256 + 8;
constexpr uint32_t kUnset = 0xffffffff;

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void check(cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template <class T> struct Buffer {
    T *p = nullptr;
    explicit Buffer(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

struct Stream {
    cudaStream_t id = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&id, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(id); }
};

struct Event {
    cudaEvent_t id = nullptr;
    Event() { check(cudaEventCreate(&id)); }
    ~Event() { cudaEventDestroy(id); }
};

struct Shape {
    int cols, rows, min_m, max_m;
};
constexpr Shape kShapes[] = {{2304, 1024, 7, 12}, {2304, 3072, 7, 12}, {3072, 2304, 63, 64}};

struct Case {
    int m, rows, pad_x, pad_y;
    bool expect_pair;
    int repeats = 0;
};

struct Options {
    bool compare;
    int repeats, seed;
};

struct Counts {
    unsigned paired = 0, baseline = 0;
};

struct Run {
    cudaStream_t stream;

    void launch(const inference_cuda_dense_gemv_args &args, bool compare, Counts &counts) const {
        int stage = -1, paired = 0;
        const int rc = compare
            ? inference_cuda_gguf_q8_row_reuse_launch(&args, stream, &stage, &paired)
            : inference_cuda_dense_gemv_launch(&args, stream, &stage);
        if (rc != 0) {
            std::fprintf(stderr, "dispatch failed: code=%d stage=%d\n", rc, stage);
            throw std::runtime_error("projection launch");
        }
        if (paired) ++counts.paired;
        else ++counts.baseline;
    }
};

void host_checks() {
    constexpr inference_cuda_dense_gemv_args a{nullptr, nullptr, nullptr, 7, 1024, 2304,
                                              2304, 1024, kQ8};
    auto valid_shape = a;
    valid_shape.w = reinterpret_cast<void *>(uintptr_t{0x1000000});
    valid_shape.x = reinterpret_cast<void *>(uintptr_t{0x2000000});
    valid_shape.y = reinterpret_cast<void *>(uintptr_t{0x3000000});
    void *fake_stream = reinterpret_cast<void *>(uintptr_t{1});
    const auto rejected = [&](const inference_cuda_dense_gemv_args *args, void *stream) {
        int stage = 99, pair = 99;
        require(inference_cuda_gguf_q8_row_reuse_launch(args, stream, &stage, &pair) ==
                    INFERENCE_CUDA_INVALID_ARGUMENT &&
                stage == INFERENCE_CUDA_STAGE_NONE && pair == 0, "bad operand accepted");
    };
    rejected(nullptr, fake_stream);
    rejected(&a, fake_stream);
    rejected(&valid_shape, nullptr);
    constexpr int mutations = 14;
    for (int i = 0; i < mutations; ++i) {
        auto args = valid_shape;
        switch (i) {
        case 0: args.y = const_cast<void *>(args.w); break;
        case 1: args.y = const_cast<void *>(args.x); break;
        case 2: args.w = reinterpret_cast<void *>(uintptr_t{0x1000001}); break;
        case 3: args.cols--; args.ld_x--; break;
        case 4: args.ld_y--; break;
        case 5: args.m = 32769; break;
        case 6: args.weight_dtype = 999; break;
        case 7: args.ld_x--; break;
        case 8: args.ld_y = (1 << 20) + 1; break;
        case 9: args.x = reinterpret_cast<void *>(uintptr_t{0x2000002}); break;
        case 10: args.y = reinterpret_cast<void *>(uintptr_t{0x3000002}); break;
        case 11: args.x = nullptr; break;
        case 12: args.y = nullptr; break;
        case 13: // y overlaps only the last strided activation row.
            args.ld_x += 3;
            args.y = reinterpret_cast<void *>(uintptr_t{0x2000000} +
                     ((args.m - 1) * args.ld_x + args.cols - 1) * sizeof(float));
            break;
        }
        rejected(&args, fake_stream);
    }
    std::printf("host-only: %d production ABI refusals passed\n", 3 + mutations);
}

std::vector<uint8_t> load(const std::string &path, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(bytes)) throw std::runtime_error("invalid size: " + path);
    std::vector<uint8_t> data(bytes);
    if (!file.seekg(0).read(reinterpret_cast<char *>(data.data()), bytes)) {
        throw std::runtime_error("read failed: " + path);
    }
    return data;
}

void compare_bits(const inference_cuda_dense_gemv_args &a, const std::vector<uint32_t> &base,
                  const std::vector<uint32_t> &candidate) {
    for (size_t i = 0; i < base.size(); ++i) {
        // Output starts after a leading guard. Check every bit, including both guards and padding.
        const bool live = i > 0 && (i - 1) / a.ld_y < size_t(a.m) && (i - 1) % a.ld_y < size_t(a.rows);
        if (base[i] != candidate[i] || (live ? (base[i] & 0x7f800000u) == 0x7f800000u
                                            : base[i] != kUnset)) {
            std::fprintf(stderr, "index=%zu live=%d A=%08x B=%08x\n", i, live, base[i], candidate[i]);
            throw std::runtime_error("output bits, finiteness or guard mismatch");
        }
    }
}

void timing(const Run &run, const inference_cuda_dense_gemv_args &args, int repeats) {
    Counts warm;
    for (int i = 0; i < 16; ++i) run.launch(args, i % 2 != 0, warm);
    check(cudaStreamSynchronize(run.stream));
    Event start, stop;
    Counts counts;
    std::array<std::array<float, 4>, 2> samples{};
    int n[2] = {};
    // ABBA BAAB: both arms occupy each position once across the symmetric waves.
    for (bool pair : {false, true, true, false, true, false, false, true}) {
        check(cudaEventRecord(start.id, run.stream));
        for (int i = 0; i < repeats; ++i) run.launch(args, pair, counts);
        check(cudaEventRecord(stop.id, run.stream));
        check(cudaEventSynchronize(stop.id));
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start.id, stop.id));
        samples[pair][n[pair]++] = ms * 1000.0f / repeats;
    }
    require(counts.paired == unsigned(repeats) * 4 && counts.baseline == unsigned(repeats) * 4,
            "timed launch counts did not match ABBA BAAB");
    const auto &a = samples[0];
    const auto &b = samples[1];
    std::printf("{\"kind\":\"timing\",\"m\":%d,\"rows\":%d,\"cols\":%d,\"repeats\":%d,"
                "\"baseline_launches\":%u,\"pair_launches\":%u,"
                "\"a_us\":[%.3f,%.3f,%.3f,%.3f],\"b_us\":[%.3f,%.3f,%.3f,%.3f]}\n",
                args.m, args.rows, args.cols, repeats, counts.baseline, counts.paired,
                a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
}

struct Weights {
    const uint8_t *p;
    int cols;
};

void verify_case(const Run &run, Weights w, Case test, Options options) {
    inference_cuda_dense_gemv_args args{w.p, nullptr, nullptr, test.m, test.rows, w.cols,
                                       w.cols + test.pad_x, test.rows + test.pad_y, kQ8};
    std::vector<float> x(size_t(args.ld_x) * args.m, -123456.0f);
    std::mt19937 rng(options.seed);
    const float uncommon[] = {0.0f, -0.0f, 0x1p-149f, -0x1p-149f, 0x1p-126f, -0x1p-126f};
    for (int i = 0; i < args.m; ++i) {
        for (int c = 0; c < args.cols; ++c) {
            x[size_t(i) * args.ld_x + c] = (c + i) % 97 < int(std::size(uncommon)) ?
                uncommon[(c + i) % 97] : float(int(rng() % 8193) - 4096) / 1024.0f;
        }
    }
    const size_t out_size = size_t(args.m) * args.ld_y + 2;
    Buffer<float> dx(x.size());
    Buffer<uint32_t> dy(out_size);
    std::vector<uint32_t> results[2] = {std::vector<uint32_t>(out_size), std::vector<uint32_t>(out_size)};
    check(cudaMemcpyAsync(dx.p, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice, run.stream));
    args.x = dx.p;
    args.y = dy.p + 1;
    Counts counts;
    for (int arm = 0; arm < 2; ++arm) {
        check(cudaMemsetAsync(dy.p, 255, out_size * sizeof(uint32_t), run.stream));
        if (arm == 0) {
            // Reference the direct production entry, not the fallback wrapper.
            int stage = -1;
            require(inference_cuda_dense_gemv_launch(&args, run.stream, &stage) == 0, "baseline dispatch");
        } else {
            run.launch(args, options.compare, counts);
        }
        check(cudaMemcpyAsync(results[arm].data(), dy.p, out_size * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, run.stream));
        check(cudaStreamSynchronize(run.stream));
    }
    compare_bits(args, results[0], results[1]);
    std::vector<float> after(x.size());
    check(cudaMemcpyAsync(after.data(), dx.p, after.size() * sizeof(float),
                          cudaMemcpyDeviceToHost, run.stream));
    check(cudaStreamSynchronize(run.stream));
    require(std::memcmp(x.data(), after.data(), x.size() * sizeof(float)) == 0, "input changed");
    require(counts.paired == unsigned(options.compare && test.expect_pair) &&
            counts.baseline + counts.paired == 1, "wrong dispatch");
    const bool padded = test.pad_x || test.pad_y;
    std::printf("{\"kind\":\"bits\",\"m\":%d,\"rows\":%d,\"cols\":%d,\"padded\":%s,"
                "\"bitwise\":%d,\"pair_launches\":%u,\"baseline_launches\":%u}\n",
                args.m, args.rows, args.cols, padded ? "true" : "false", args.m * args.rows,
                counts.paired, counts.baseline);
    if (test.repeats) timing(run, args, test.repeats);
}

void run_shape(const Run &run, const std::string &directory, Shape shape, Options options) {
    const size_t bytes = size_t(shape.cols / 32) * 34 * shape.rows;
    const auto name = "/q8-" + std::to_string(shape.cols) + "-" + std::to_string(shape.rows) + ".bin";
    const auto payload = load(directory + name, bytes);
    Buffer<uint8_t> dw(bytes), offset_dw(bytes + 2);
    check(cudaMemcpyAsync(dw.p, payload.data(), bytes, cudaMemcpyHostToDevice, run.stream));
    // Deliberately use the ABI's minimum 2-byte weight alignment in padded tests.
    check(cudaMemcpyAsync(offset_dw.p + 2, payload.data(), bytes, cudaMemcpyHostToDevice, run.stream));
    check(cudaStreamSynchronize(run.stream));
    for (int m = shape.min_m; m <= shape.max_m; ++m) {
        const int repeats = (m == shape.min_m || m == shape.max_m) ? options.repeats : 0;
        verify_case(run, {dw.p, shape.cols}, {m, shape.rows, 0, 0, true, repeats}, options);
        if (m % 2) {
            verify_case(run, {offset_dw.p + 2, shape.cols}, {m, shape.rows, 3, 7, true}, options);
        }
    }
    for (auto c : {Case{1, shape.rows, 7, 7, false}, Case{65, shape.rows, 0, 0, false},
                   Case{7, shape.rows - 1, 7, 7, false}, Case{2, shape.rows, 0, 0, false}}) {
        verify_case(run, {dw.p, shape.cols}, c, options);
    }
}

int number(const char *arg, int lo, int hi) {
    size_t end = 0;
    int value = std::stoi(arg, &end);
    require(end == std::strlen(arg) && value >= lo && value <= hi, "bad numeric option");
    return value;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
            host_checks();
            return 0;
        }
        const bool compare = argc >= 3 && std::strcmp(argv[2], "--compare") == 0;
        require(argc == 2 || (compare && argc <= 5),
                "usage: probe --host-only | probe WEIGHTS_DIR [--compare [REPEATS [SEED]]]");
        const int repeats = argc > 3 ? number(argv[3], 0, 5000) : 0;
        const int seed = argc > 4 ? number(argv[4], 1, 2147483647) : 123;
        int device;
        cudaDeviceProp prop;
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&prop, device));
        require(prop.major == 7 && prop.minor == 5, "fixture built for SM75 only");
        Stream stream;
        for (auto shape : kShapes) run_shape({stream.id}, argv[1], shape, {compare, repeats, seed});
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
