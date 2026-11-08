# Chunked P2P prefill: the path from 40 to thousands of tokens/s

Measured on the 3060+3080 pair, Q4_K_M, q8 KV: the eager card path runs
every prompt token through the full distributed step (~25 ms/token). The
ring is no longer the cost (121 sequences after the MoE fusion and the
gather drop, ~1 ms); the step is WEIGHT-TRAFFIC bound - each token's
matvecs re-read its top-8 routed experts plus the dense/mixer weights,
~1.5-2 GB of reads, at matvec (memory-bound, low arithmetic intensity)
efficiency. No per-token optimization crosses ~45 tok/s.

Chunking changes the arithmetic: one 2048-token pass reads the same
weight bytes ONCE and runs dense MMQ GEMMs instead of one-row matvecs.
MEASURED (mmq_prefill_ceiling_probe, Q4_K, k=n=2048, release build, the
deployment's cards): the 3060 sustains 14.7 / 23.9 / 28.8 / 33.0 / 33.7
TFLOPS at m = 64 / 256 / 512 / 1024 / 2048 - a whole-model projection of
8,400 to 19,300 tok/s on that card alone; the 3080 measures 16.2 TFLOPS
at m=64 (its m>=256 sweep hits a launch-config refusal in the vendored
MMQ wrapper on 68 SMs - a wrapper quirk to file separately, the kernel
family is identical). Even at a quarter of the m=512 efficiency
end-to-end (MoE gather, attention, quantize passes, ring sums), the pair
sustains >3000 tok/s. The target is real with wide margin; m = 512 is
the knee.

## What already exists (do not rebuild)

- `kernels/cuda/prefill/`: `mmq_dense`, `mmq_mmid`, `mmq_quant`,
  `mmq_size`, `attn_prep_chunk`, `attn_prefill`, `kv8_dequant`,
  `gdn_scan_seq`, `flint_qk` - the single-device chunk pipeline's
  kernels, all shape-parameterized by chunk rows.
- `prefill_plan.rs` / `prefill_driver.rs`: chunk planning, admission
  (M_sess ladder), staging - single-device wired, card-agnostic math.
- `TokenStepper::feed_rows`: the seam a chunked executor overrides.
- The ring ABI counts payloads as `rows * width` (`abi_ring.h`); the
  peer-ring wait kernels are count-strided. Multi-row exchanges need
  only a larger mailbox cap (the MoE fusion already proved the pattern:
  hidden*top_k values).
- Prefix save/restore (card path, bitwise-qualified): chunked prefill
  must add `CudaPrefillPath::Chunked(n)` so a chunk-built state never
  restores into a steps-built one (the path tag exists for exactly this).

## Measured transport (chunk_sized_exchange probe)

A 256x2048 f32 exchange_sum (2.1 MB payload) on the physical pair:
**1.333 ms per exchange** with the original single-block publish/wait
kernels (1.6 GB/s effective - 256 threads copying over PCIe). A
40-layer chunk pays 80 such sums: 107 ms - MORE than the chunk's
compute at m=256 (~20 ms/card), so the transport would dominate. The
multi-block rework of the ring's payload kernels (grid-strided copies,
64-block ceiling, publish split into flags-zero + data + seq-commit)
addresses exactly this; the probe re-run after it is the acceptance
number. PCIe 3.0 x4-ish effective ceiling would put 2.1 MB at ~0.3 ms.

## Keystone finding (card_views_construct_chunk_models)

The card session's per-card views construct ChunkModel under the CARD
geometry for every mixer, attention and DeltaNet role - the stage
functions are reusable unchanged. The ONE refusal is the MoE expert
stacks: `check_model` wants the full 131072-row stack, each card holds
its 65536-row slice. So the card chunk driver needs exactly one
purpose-built stage - `chunk_moe` over the card's expert slice (the
mmq_mmid grouped GEMM behind a per-card pick guard: a row's top-8 picks
scatter across both cards, each card computes the picks it owns and the
multi-row MoeSum exchange combines them) - plus the driver loop itself.
MoeLayerWeights/StackWeights/ExpertStack are now Clone+Copy so a
[card][layer] transpose of the session's views exists.

## The build, in stages

1. **Card chunk geometry** (`StepDims`/`StepPlan`): a `rows`-wide card
   plan variant - per-card slot widths scale by rows; `AttnProj` and the
   prep/append kernels already take rows (the single-device prefill uses
   them at 8-2048 rows).
2. **Multi-row ring exchanges**: mixer sum and MoE sum over `rows*hidden`
   values (mailbox cap grows; PCIe bytes/token amortize by the chunk
   factor - at rows=2048 the residual sums are 8 MB/layer, ~1 ms each,
   still cheap against the GEMM time).
3. **The card chunk op list**: per layer - project (MMQ over the chunk),
   prep/rope (rows kernel), append (kv8 chunk kernel, card's head slice),
   attention (attn_prefill chunk kernel against the card's own planes -
   fully local since the gather drop), mixer sum, MoE (route per row,
   grouped MMQ over the card's expert slice), moe sum.
4. **Admission and rollback**: the prefill_driver ladder per card (KV
   bytes halve with sliced caches); head-skip for prompt-only feeds
   already exists.
5. **Rigs**: chunk-vs-single-device-reference logits over a real GGUF
   (tolerance like the single-device chunk tests), plus the prefix
   turn-chain rig under the chunked path tag.

## Staging order that ships value early

Stage 1+2 alone (rows=256) with everything else eager would already
batch the dominant matvec traffic ~4-6x if only the projections run
row-batched; but the clean win is doing stages 1-3 together at rows=512
(the single-device ladder's sweet spot for this shape) and treating
rows=2048 as the tuning step after correctness.
