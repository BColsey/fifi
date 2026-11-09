"""Exercise the real native MMQ facts cache with a simulated CUDA runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine/src/kernels/cuda/native/prefill/mmq_facts.cu"

HEADERS = {
    "cuda_runtime.h": r"""
#pragma once
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaErrorInvalidDevice = 10, cudaErrorNoKernelImageForDevice = 209;
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
                      cudaDevAttrMultiProcessorCount, cudaDevAttrMaxSharedMemoryPerBlockOptin };
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
cudaError_t cudaGetLastError();
""",
    "abi.h": r"""
#pragma once
constexpr int INFERENCE_CUDA_INVALID_ARGUMENT = -1, INFERENCE_CUDA_OK = 0,
              INFERENCE_CUDA_NATIVE_EXCEPTION = -2;
""",
    "common.cuh": r"""
#pragma once
#include <cstddef>
#define GGML_CUDA_MAX_DEVICES 16
struct fifi_device_facts { int device, cc, nsm; std::size_t smpbo; };
""",
    "prefill/mmq_facts.h": r"""
#pragma once
#include <cstddef>
struct inference_cuda_mmq_facts { int device, cc, nsm, reserved; std::size_t smpbo; };
""",
    "resident/mmvq_arch.h": r"""
#pragma once
int inference_cuda_mmvq_built_arch(int capability);
""",
}

HARNESS = r"""
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <thread>
#include "cuda_runtime.h"
#include "common.cuh"

thread_local int current_device = 0;
std::atomic<int> attributes[2]{};
cudaError_t cudaGetDevice(int *device) {
    if (current_device < 0) return cudaErrorInvalidDevice;
    *device = current_device;
    return cudaSuccess;
}
cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    if (device < 0 || device > 1) return cudaErrorInvalidDevice;
    ++attributes[device];
    switch (attr) {
        case cudaDevAttrComputeCapabilityMajor: *value = 8; break;
        case cudaDevAttrComputeCapabilityMinor: *value = 6; break;
        case cudaDevAttrMultiProcessorCount: *value = device == 0 ? 28 : 68; break;
        case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = 101376; break;
    }
    return cudaSuccess;
}
cudaError_t cudaGetLastError() { return cudaSuccess; }
int inference_cuda_mmvq_built_arch(int capability) { return capability; }
cudaError_t inference_cuda_mmq_query_facts(fifi_device_facts *out);

bool facts_match(int device) {
    current_device = device;
    fifi_device_facts facts{};
    const int status = inference_cuda_mmq_query_facts(&facts);
    if (status != cudaSuccess || facts.device != device || facts.cc != 860 ||
        facts.nsm != (device == 0 ? 28 : 68) || facts.smpbo != 101376) {
        std::cerr << "wanted device " << device << ", got status=" << status
                  << " device=" << facts.device << " nsm=" << facts.nsm << '\n';
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    switch (std::atoi(argv[1])) {
        case 0:
            if (!facts_match(0) || !facts_match(1) || !facts_match(0) || !facts_match(1)) return 1;
            if (attributes[0] != 4 || attributes[1] != 4) return 3;
            return 0;
        case 1: {
            std::atomic<bool> ok{true};
            auto work = [&ok](int device) {
                for (int i = 0; i < 20; ++i) if (!facts_match(device)) ok = false;
            };
            std::thread a(work, 0), b(work, 1);
            a.join(); b.join();
            return ok ? 0 : 1;
        }
        case 2: {
            current_device = -1;
            fifi_device_facts facts{};
            if (inference_cuda_mmq_query_facts(&facts) != cudaErrorInvalidDevice) return 1;
            if (!facts_match(1) || !facts_match(1) || attributes[1] != 4) return 3;
            return 0;
        }
    }
    return 2;
}
"""


class MmqDeviceFactsCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++-13") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("a C++ compiler is required for the MMQ facts cache test")
        cls.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.tmp.name)
        for name, content in HEADERS.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        harness = root / "harness.cc"
        harness.write_text(HARNESS)
        cls.binary = root / "mmq-facts-cache"
        subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(root),
             "-x", "c++", str(SOURCE), str(harness), "-o", str(cls.binary)],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_switching_devices_returns_their_own_cached_facts(self):
        subprocess.run([self.binary, "0"], check=True)

    def test_threads_can_query_different_devices(self):
        subprocess.run([self.binary, "1"], check=True)

    def test_initial_query_failure_does_not_poison_the_cache(self):
        subprocess.run([self.binary, "2"], check=True)


if __name__ == "__main__":
    unittest.main()
