# Inkling strict-F32 relative-attention query tiles

The standalone CUDA probe calls the production launchers with packed synthetic
inputs. The default-off GGUF prefill bridge additionally uses strided inputs and
can borrow the existing resident sink/gate sidecar. It reuses the batch workspace
and does not change decode or draft verification.

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

The production bridge (`attention/prefill.rs`) takes its per-visit cursor before
any repair or launch. The ordinary grouped route in `cpu_forward/prefill.rs`
excludes draft windows and journaled rows; `CpuForward::restore` and `reset`
invalidate before replacing host history. A missing, behind or ahead cursor
requires a full blocking reseed from the host deques on the attention stream.
An equal cursor may only be reused within that visit and lineage. The sidecar
producer fences before returning; the consumer drains before the next tile can
reuse it. Upload, queries, guarded commit, output projection and readback are
ordered on the attention stream. Cursor publication requires a successful fence,
zero attention status and finite projected output. On any later failure the
cursor stays unknown, even if physical commit ran; there is no in-place serial
retry. The enclosing grouped forward remains failed until reset/restore. It
still appends each successful row's prepared host K/V and applies output
convolution in token order.

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

### Focused coverage and remaining qualification

- `build.sh` runs 29,488 CPU source-identity and final-ring cases against a
  sequential oracle, plus cursor/extent checks, and compiles/links the SM75 probe.
  The standalone production CUDA TU also compiles for SM75 and SM86 with nvcc
  12.4.131 (`--Werror all-warnings`). Neither command runs any GPU code.
- `query_tile_validates_strided_bounds_and_positions` checks byte extents,
  alignment, tile width and the upper legal position without a GPU.
- Under a GPU lease, run `cargo test --manifest-path engine/Cargo.toml --features
  native-cuda restored_attention_replaces_a_divergent_ring_at_the_same_position`.
  It compares
  strided host and resident-sidecar routes against serial output bits and physical
  rings at wrap, checks a singleton continuation, rewinds, rejects missing host
  history and bad sidecar extents, and checks that status failure suppresses
  commit and projection failure invalidates an already-written ring. It then
  retries from authoritative host rows. This test was not run for the bridge.
  For a real-model A/B, keep all other flags fixed, check the successful-use log
  (`Inkling CUDA used relative attention query tiles`), then compare fresh and
  restored prefixes, and the first decode row after a local ring wrap.
- Component commit `8d59b740` passed synthetic bitwise output, status and ring
  parity on a leased RTX 2080 Ti (`../evidence/query-device/probe.log` relative to
  the lane checkout). The 512-token component measured 18.672 ms serial vs
  1.911 ms tiled, including commit but excluding host preparation and transfers.
  This predates the production bridge and is not a serving result. Production
  GPU parity, real-model prefix/continuation checks, end-to-end timings and a
  full engine build remain outstanding.

[turing]: https://docs.nvidia.com/cuda/turing-tuning-guide/index.html#occupancy
[fa2]: https://arxiv.org/abs/2307.08691
[banded]: https://github.com/unslothai/llama.cpp/pull/386
[cuda]: https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html
