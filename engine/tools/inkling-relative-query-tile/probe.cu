// Standalone TU: exercise the production launchers with packed input planes.
#include "resident/relative_attention.cu"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void check(bool ok, const char *description) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

void cuda(cudaError_t error) {
    check(error == cudaSuccess, cudaGetErrorString(error));
}

template <typename T> struct Buffer : DeviceBuffer<T> {
    explicit Buffer(size_t n) : count(n) { cuda(this->allocate(n * sizeof(T))); }

    void upload(const std::vector<T> &src) {
        check(src.size() == count, "upload extent");
        cuda(cudaMemcpy(this->ptr, src.data(), bytes(), cudaMemcpyHostToDevice));
    }
    std::vector<T> download() const {
        std::vector<T> data(count);
        cuda(cudaMemcpy(data.data(), this->ptr, bytes(), cudaMemcpyDeviceToHost));
        return data;
    }
    size_t bytes() const { return count * sizeof(T); }
    size_t count;
};

struct Shape {
    int q, kv, rel, extent, capacity;
};

// This probe only supplies disjoint, packed F32 planes and consecutive host-known
// positions, with status initially zero. All writers to the per-visit ring are
// ordered on this stream. The production single-row device-position ABI is intact.
bool enqueue(Args a, int first, int certified_cursor, int count, bool enabled, cudaStream_t s) {
    check(certified_cursor == first, "ring must be reseeded before either path");
    check(valid(a) && snapshot::valid_interval(first, count), "invalid packed probe geometry or position");
    if (snapshot::eligible(enabled, certified_cursor, first, count, a.capacity)) {
        int stage;
        cuda(static_cast<cudaError_t>(
            inference_cuda_relative_attention_tile_launch(&a, first, count, 0, 0, s, &stage)));
        return true;
    }
    for (int i = 0; i < count; ++i) {
        int stage;
        cuda(static_cast<cudaError_t>(inference_cuda_relative_attention_launch(&a, s, &stage)));
        a.out += a.query_heads * kHeadDim;
        a.q += a.query_heads * kHeadDim;
        a.k += a.kv_heads * kHeadDim;
        a.v += a.kv_heads * kHeadDim;
        a.relative += a.query_heads * a.relative_dims;
        a.sinks += a.query_heads;
        a.gates += a.query_heads * kHeadDim;
        ++a.status;
        ++a.pos;
    }
    return false;
}

float sample(uint32_t i) {
    i = (i ^ (i >> 16)) * 0x7feb352du;
    i = (i ^ (i >> 15)) * 0x846ca68bu;
    return float(int(i >> 24) - 128) * (1.f / 128.f);
}

void fill(Buffer<float> &buffer, uint32_t seed, int infinite_gate = -1) {
    std::vector<float> data(buffer.count);
    for (size_t i = 0; i < data.size(); ++i) data[i] = sample(i + seed);
    if (infinite_gate >= 0) data[infinite_gate] = INFINITY;
    buffer.upload(data);
}

std::vector<float> seed_ring(int capacity, int channels, int first, uint32_t seed) {
    std::vector<float> ring(size_t(capacity) * channels, std::numeric_limits<float>::quiet_NaN());
    for (int pos = std::max(0, first - capacity); pos < first; ++pos) {
        for (int d = 0; d < channels; ++d) {
            ring[(pos % capacity) * channels + d] = sample(seed + uint32_t(pos) * channels + d);
        }
    }
    return ring;
}

template <typename T> bool same(const std::vector<T> &a, const std::vector<T> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

struct Result {
    std::vector<float> output, keys, values;
    std::vector<int> status;
    float ms;
};

struct Fixture {
    Shape shape;
    int rows, qch, kvch;
    Buffer<float> out, keys, values, q, k, v, relative, table, sinks, gates;
    Buffer<int> status, pos;

    Fixture(Shape shape, int rows, int bad_gate_row)
        : shape(shape), rows(rows), qch(shape.q * kHeadDim), kvch(shape.kv * kHeadDim),
          out(rows * qch), keys(shape.capacity * kvch), values(keys.count), q(out.count),
          k(rows * kvch), v(k.count), relative(rows * shape.q * shape.rel),
          table(shape.rel * shape.extent), sinks(rows * shape.q), gates(out.count),
          status(rows), pos(rows) {
        fill(q, 1234);
        fill(k, 219345);
        fill(v, 27543);
        fill(relative, 24874);
        fill(table, 184479);
        fill(sinks, 49358);
        fill(gates, 49383, bad_gate_row < 0 ? -1 : bad_gate_row * qch);
    }

    Result run(cudaStream_t stream, int first, int width, bool tiled,
               const std::vector<float> &seed_k, const std::vector<float> &seed_v, int bad_gate_row) {
        keys.upload(seed_k);
        values.upload(seed_v);
        std::vector<int> positions(rows);
        for (int i = 0; i < rows; ++i) positions[i] = first + i;
        pos.upload(positions);
        cuda(cudaMemsetAsync(status.ptr, 0, status.bytes(), stream));
        cuda(cudaMemsetAsync(out.ptr, 0xff, out.bytes(), stream));
        cudaEvent_t begin, end;
        cuda(cudaEventCreate(&begin));
        cuda(cudaEventCreate(&end));
        cuda(cudaEventRecord(begin, stream));
        int cursor = first;
        for (int i = 0; i < rows; i += width) {
            const int count = std::min(width, rows - i);
            const Args args{out.ptr + i * qch, keys.ptr, values.ptr, status.ptr + i,
                            q.ptr + i * qch, k.ptr + i * kvch, v.ptr + i * kvch,
                            relative.ptr + i * shape.q * shape.rel, table.ptr, sinks.ptr + i * shape.q,
                            gates.ptr + i * qch, pos.ptr + i, shape.q, shape.kv,
                            shape.rel, shape.extent, shape.capacity};
            const bool used = enqueue(args, first + i, cursor, count, tiled, stream);
            check(used == (tiled && count > 1), "unexpected dispatch or fallback");
            // Only this fixture certifies prevalidated rows across multiple tiles.
            // The deliberately invalid fixture is one tile; it never advances the cursor.
            if (bad_gate_row < 0) cursor += count;
        }
        cuda(cudaEventRecord(end, stream));
        cuda(cudaEventSynchronize(end));
        float ms;
        cuda(cudaEventElapsedTime(&ms, begin, end));
        cuda(cudaEventDestroy(begin));
        cuda(cudaEventDestroy(end));
        return {out.download(), keys.download(), values.download(), status.download(), ms};
    }
};

struct Case {
    const char *name;
    Shape shape;
    int first, rows, width, branches, bad_gate_row = -1, samples = 1;
};

void run_case(cudaStream_t stream, const Case &c, bool tiled) {
    Fixture fixture(c.shape, c.rows, c.bad_gate_row);
    std::vector<float> prior_output, serial_ms, tiled_ms;
    for (int branch = 0; branch < c.branches; ++branch) {
        // A same-position divergent prefix must invalidate the previous cursor.
        // Reuse the allocation after overwrite, but explicitly seed each branch's own history.
        const auto keys = seed_ring(c.shape.capacity, fixture.kvch, c.first, 900001 + branch * 100000);
        const auto values = seed_ring(c.shape.capacity, fixture.kvch, c.first, 500013 + branch * 200000);
        for (int run = 0; run < c.samples; ++run) {
            const auto serial = fixture.run(stream, c.first, c.width, false, keys, values, c.bad_gate_row);
            check(std::all_of(serial.output.begin(), serial.output.end(),
                              [](float x) { return std::isfinite(x); }), "unwritten or nonfinite output");
            for (int i = 0; i < c.rows; ++i) {
                check(serial.status[i] == (i == c.bad_gate_row ? INFERENCE_CUDA_STATUS_INVALID_STEP_STATE : 0),
                      "wrong serial status");
            }
            if (run == 0) {
                check(branch == 0 || !same(prior_output, serial.output), "divergent prefix had no effect");
                prior_output = serial.output;
            }
            serial_ms.push_back(serial.ms);
            if (!tiled) continue;
            const auto current = fixture.run(stream, c.first, c.width, true, keys, values, c.bad_gate_row);
            check(same(current.status, serial.status), "FP32 tile status differs");
            check(same(current.output, serial.output), "FP32 tile output differs bitwise");
            if (c.bad_gate_row < 0) {
                check(same(current.keys, serial.keys), "key ring differs bitwise");
                check(same(current.values, serial.values), "value ring differs bitwise");
            } else {
                check(same(current.keys, keys) && same(current.values, values), "failed tile modified snapshot");
            }
            tiled_ms.push_back(current.ms);
        }
    }
    std::printf("PASS %s (first=%d rows=%d tile=%d capacity=%d branches=%d) %s\n",
                c.name, c.first, c.rows, c.width, c.shape.capacity, c.branches,
                tiled ? "bitwise A/B" : "serial only");
    if (c.samples > 1) {
        // Drop the first sample per arm as warmup. Includes cache commit and launch
        // scheduling gaps, but not preparation, snapshot seeding, or transfers.
        const auto median = [](std::vector<float> ms) {
            std::sort(ms.begin() + 1, ms.end());
            return ms[1 + (ms.size() - 1) / 2];
        };
        std::printf("  component event ms (%d measured): serial=%.3f", c.samples - 1, median(serial_ms));
        if (tiled) std::printf(" tiled=%.3f", median(tiled_ms));
        std::puts("");
    }
}

}  // namespace

int main(int argc, char **argv) {
    const bool tiled = argc == 2 && std::strcmp(argv[1], "--query-tiles") == 0;
    if (argc != 1 && !tiled) {
        std::fprintf(stderr, "usage: %s [--query-tiles]\n", argv[0]);
        return 2;
    }
    cudaDeviceProp properties;
    cuda(cudaGetDeviceProperties(&properties, 0));
    std::printf("%s sm_%d%d, %d SMs; query tiling %s\n", properties.name,
                properties.major, properties.minor, properties.multiProcessorCount, tiled ? "ON" : "OFF");
    cudaStream_t stream;
    // Blocking default-stream copies seed immutable input planes before enqueue.
    cuda(cudaStreamCreate(&stream));
    for (const Case &c : {
             Case{"single query fallback", {3, 1, 2, 4, 3}, 0, 1, 1, 1},
             Case{"multiwrap, divergent same-position restore", {3, 1, 2, 1, 3}, 4, 9, 4, 2},
             Case{"current row only", {2, 2, 1, 1, 1}, 7, 8, 8, 1},
             Case{"failed tile does not publish", {3, 1, 2, 4, 4}, 7, 5, 5, 1, 1},
             Case{"upper position bound", {2, 1, 1, 2, 3}, snapshot::kLastPosition - 1, 2, 2, 1},
             Case{"Mini geometry, 512-token prompt", {24, 8, 16, 1024, 768}, 0, 512, 64, 1, -1, 6},
             Case{"Mini geometry, SWA wrap", {24, 8, 16, 1024, 1024}, 1007, 64, 64, 1, -1, 6},
         }) {
        run_case(stream, c, tiled);
    }
    cuda(cudaStreamDestroy(stream));
}
