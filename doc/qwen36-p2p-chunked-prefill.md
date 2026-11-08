# Chunked P2P prefill: the path from 40 to thousands of tokens/s

Measured on the 3060+3080 pair, Q4_K_M, q8 KV: the eager card path runs
every prompt token through the full distributed step (~25 ms/token). The
ring is no longer the cost (121 sequences after the MoE fusion and the
gather drop, ~1 ms); the step is WEIGHT-TRAFFIC bound - each token's
matvecs re-read its top-8 routed experts plus the dense/mixer weights,
~1.5-2 GB of reads, at matvec (memory-bound, low arithmetic intensity)
efficiency. No per-token optimization crosses ~45 tok/s.

Chunking changes the arithmetic: one 2048-token pass reads the same
weight bytes ONCE and runs dense MMQ GEMMs at 20-30 TFLOPS instead of
one-row matvecs. Weight traffic per token collapses by the chunk factor;
the pair's prefill becomes compute-bound at an estimated 1500-3000 tok/s
(the single-device 3090 path already demonstrates ~2500 tok/s with the
same kernels on one card).

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
