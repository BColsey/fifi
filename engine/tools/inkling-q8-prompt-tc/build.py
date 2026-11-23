"""Build only the standalone CUDA translation unit and exact baseline excerpts."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path, help='fresh build directory on an executable filesystem')
out = parser.parse_args().output.resolve()
w = here.parents[2]
rev = 'bf72f7d9541c21fb41bf9f35924db298cf67dca2'
relative = 'engine/src/kernels/cuda/native/resident/dense_gemv.cu'
native = w / 'engine/src/kernels/cuda/native'
s = subprocess.check_output(['git', '-C', str(w), 'show', f'{rev}:{relative}']).decode()
assert s == (w / relative).read_text()

def piece(start, end):
    first = s.index(start)
    return s[first:s.index(end, first)]

out.mkdir()
(out / 'baseline.inc').write_text('constexpr int kWarps = 8;\n' +
    piece('__device__ __forceinline__ float warp_sum', '// grid') +
    piece('struct SharedWeights', 'struct SelectedWeights') +
    piece('template <typename Index>\n__device__ __forceinline__ const uint8_t *matrix_base',
          '__device__ __forceinline__ const uint8_t *matrix_base(const void *, SeparateWeights') +
    piece('template <int Type, int Block, int Bytes, typename Index>\n__global__ void gguf_packed_gemv',
          '// One expert matrix') +
    piece('__global__ void inkling_q8_reuse_pair', 'constexpr int kQ8 ='))
print(rev, hashlib.sha256(s.encode()).hexdigest(), relative, flush=True)
subprocess.run(['/usr/bin/nvcc', '-ccbin', '/usr/bin/g++-13', '-O3', '-std=c++17',
    '-arch=sm_75', '-I' + str(native), '-I' + str(w / 'engine/vendor/ggml-cuda-mmq/src'),
    '-I' + str(out), str(here / 'probe.cu'), '-lcublas', '-o', str(out / 'probe')],
    check=True, env=dict(os.environ, TMPDIR=str(out), TMP=str(out), TEMP=str(out)))
