# Inkling strict-F32 relative-attention query tiles

Standalone, default-off CUDA component probe. It does not change serving, decode,
workspace admission, draft verification, or prefix handling. The only native
engine change extracts the serial kernel's K/V addressing behind a compile-time
reader; the probe includes that translation unit to reuse the production FP32
arithmetic and test against the actual serial launch.

## Why this experiment

In the prior RTX 2080 Ti trace, FiFi spent about 0.625 s in relative attention with
24 CTAs per row on 68 SMs. Eight warps per block provide at most 192 warps for that
entire grid, versus 32 concurrent warps per SM on Turing. A 64-query tile exposes
1,536 CTAs of the same shape. This increases the available parallel work; it is not
a measurement of achieved occupancy or speed. [NVIDIA's Turing tuning guide][turing]
lists the hardware occupancy limits, and [FlashAttention-2][fa2] describes parallel
work across the sequence dimension. The same-file [llama.cpp banded attention
reference][banded] spent about 14.37 ms across 48 attention launches, but used FP16
K/V and a Tensor Core algorithm with a different arithmetic order. That time is
not a parity target or an isolated apples-to-apples comparison.

## Snapshot contract and fallback

Simply launching query rows concurrently using the mutating step is wrong. In a
capacity-3 ring a query at position 4 needs the key at position 2 (slot 2); a
concurrent later query at position 5 writes the same slot. Conversely the later
query may read a stale slot if the earlier CTA has not run. Prescattering the
entire tile also overwrites still-visible history when the ring wraps. CUDA
[requires independent CTAs; it orders commands on an explicit stream][cuda].

For a tile starting at position `p`, query `i` has visible distances
`0 <= d < min(p + i + 1, capacity)` in the order already used by the serial
kernel. If `d <= i`, its source is staged row `i - d` (including the current
K/V). Otherwise, its source is `(p + i - d) % capacity` in the **pre-tile** ring.
No future row can be read; the earliest possible historical row was still
visible at the start of the tile. Only the addresses change: QK FMA, relative
bias extent, warp interleave and shuffle order, softmax/sink, partial merge and
per-channel gate are shared with the serial kernel. There is no downcast,
sequence reduction, or floating-point atomic.

The same-stream commit kernel runs after all query CTAs and checks the tile's
status words before writing. It scatters only the final `min(capacity, count)`
rows; those have distinct slots even when the tile wraps multiple times. This
produces the serial ring on success without copying it, even for width greater
than capacity. On a status failure it leaves the snapshot untouched. A CUDA
execution error is not recoverable by scheduling further work against that ring.

The probe opts in only for 2–64 consecutive queries at a valid host-known
position with a certified pre-tile cursor, exclusive ownership of **one visit's**
ring, and validated disjoint packed F32 input/output/cache storage. Competing
streams, graph-replayed positions, and arbitrary model layouts are outside this
probe. Off or ordinary valid non-tile sizes use the unchanged serial launch.
Never fall back by reading a stale ring: a missing cursor, gap or rewind needs
its visit's authoritative history reseeded first. A same-position divergent
restore must invalidate lineage before comparing position numbers. The probe
reseeds both arms explicitly and enforces matching certification in `enqueue`;
it runs no attention against an unverified cursor.

A future integration would keep the original host history and prefix/rejection
protocol. It must take/invalidate its per-visit cursor before enqueuing, repair
history on the same stream, and publish its next cursor only after completion,
status, and any downstream projection/readback validation. Tile application to a
restored prompt would require the same proof as fresh prefill. This probe is not
serving dispatch or speculative transaction support.

## Build and qualify

From the checkout root; `build.sh` runs the CPU topology test and compiles one
CUDA TU. Neither step uses a GPU. Use a compiler supported by the toolkit (for
example, `g++-13` with CUDA 12.4 on this host).

```sh
CXX=g++-13 bash engine/tools/inkling-relative-query-tile/build.sh ../tmp/query-sm75 75
```

After coordinator GPU admission, use the established device runner to run
`../tmp/query-sm75/probe --query-tiles`. Without `--query-tiles`, the reference
path runs alone. With it, the program compares every output F32 bit and status,
compares final rings (or the untouched seed for a failed tile), and reports
CUDA-event component timing including commit but excluding preparation,
seeding, and transfers. The multiwrap case reuses an overwritten allocation
for a divergent same-position prefix and ends with a serial step. Other cases
cover capacity one, relative extent shorter than the visible window, an
infinite resident gate, and the upper legal position. The realistic fixtures
use synthetic prepared operands with Mini's measured geometry: 24 query heads,
8 K/V heads, head width 128, 16 relative coefficients, extent 1,024. They test a
512-row prefill at context 768 and a 64-row tile starting just before the
1,024-row SWA ring wraps. No model weights or serving claim.

### Checks completed without a GPU

- 29,488 CPU source-identity and final-ring cases against a sequential oracle,
  plus cursor/extent eligibility tests; also passed with UBSan.
- Targeted nvcc 12.4.131 build/link succeeded for SM75 and SM86 with warnings as
  errors. On SM75, serial uses 39 registers and the query tile 41, both with
  4,256 shared bytes per CTA and no spills. Commit uses 22 registers and 4
  shared bytes, also with no spills. SM86 query tile uses 36 registers, no spills.
- Standalone production translation unit's serial kernel instructions and raw
  encodings (`cuobjdump --dump-sass`) matched base `9567f77e` on both SM75 and SM86.
- Exact **device** parity and timings are pending GPU admission. No full engine
  build, integration, GPU execution, or flag default change was performed.

[turing]: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy
[fa2]: https://arxiv.org/abs/2307.08691
[banded]: https://github.com/unslothai/llama.cpp/pull/386
[cuda]: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html
