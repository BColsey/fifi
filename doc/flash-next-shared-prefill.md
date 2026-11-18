# Flash-Next shared prefill

## Dispatch

The shared EXL3 prefill path is the only chunked prefill path (the legacy chunk
path and its `INFERENCE_FLASH_NEXT_SHARED_PREFILL` rollback switch were deleted
after the flip was qualified). The per-token `prefill_rows = 1` path is the token
driver and remains.
An unset `prefill_rows` resolves to 256; only width 256 is qualified (the measurements
below). Explicit widths from 2 up to the shared bound run the shared path at that width,
unqualified.

During an open prefill window, the shared steps use the common Projector and MoE pass.
Decode and verify use the existing launches with the same arena owner.

See the [environment reference](reference/environment.md#flash-next-on-nvidia-cuda).

## Memory charge

The startup charge calculation constructs `FlashGeometry`.
The charge is the full aligned scratch size from `StepPlan::prefill_bytes`.
`TierInputs::with_prefill_arena` adds this charge before the tier plan selects expert slots.

The existing `RUNTIME_OVERHEAD_FLOOR_BYTES` reserve stays in the arena ledger.
The floor covers measured kernel-arena and CUDA-context costs.
These terms conservatively overlap because the runtime floor does not separate every scratch allocation.
The sum does not represent a second allocation of the shared arena.
Other admission reserves stay unchanged.
No runtime floor subtraction or scratch-difference credit occurs.

## Qualification

Measured 2026-11-17 on the RTX 3090 (binary from main c1cc32892, 3.05 bpw h5 ng5 pack,
slots 196 both paths, `INFERENCE_FLASH_NEXT_RING_BUDGET` unset so the bank's
union-admit path serves every chunk):

- Correctness: shared256 (and, before its deletion, legacy64) completed greedy
  128-token runs at 2.9K, 10.6K and 30K prompt tokens with no refusal, no non-finite
  logits and non-empty output; texts diverge between widths early, as any chunk-width
  arithmetic change expects.
- Prefill speed: shared256 measured 1.47x to 1.50x faster end-to-end at those lengths
  (60.4/65.0/62.0 vs 40.6/43.4/42.1 prefill tok/s); at 117K depth 48.3 tok/s
  (40.4 min cold prefill), with prefix resume reusing 116,992 of 117,062 tokens
  (2.0 s to first token).
- Served cost: the ON arena charge holds the per-layer slot budget down - 196 vs 202
  slots at context 32768, 168 slots at context 131072 - and decode after a shared
  prefill ran 2 to 3.5 percent slower per token at the same 196 slots.

FP16 staging remains as implemented. No served performance gain is claimed here
beyond the measurements above.
