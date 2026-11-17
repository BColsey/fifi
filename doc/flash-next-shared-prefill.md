# Flash-Next shared prefill

## Switch and dispatch

The switch `INFERENCE_FLASH_NEXT_SHARED_PREFILL` selects the shared EXL3 prefill path.
Unset selects the shared path (the default since the flip).
`0`, `false`, and `off` roll back to the existing legacy path.
`1`, `true`, and `on` select shared prefill.
Any other value refuses startup before CUDA initialization.
The switch and the legacy chunk path it guards die together in the deletion commit;
the per-token `prefill_rows = 1` path is the token driver and survives that deletion.
Until the deletion, `INFERENCE_FLASH_NEXT_SHARED_PREFILL=0` is the rollback, and it
reproduces the pre-flip behavior exactly: an unset `prefill_rows` resolves to the old
default 1, and the legacy path accepts at most 64 rows.
Forcing the switch ON (`1`, `true`, `on`) refuses `prefill_rows = 1`: width 1 is the token
 driver, not a shared chunk width. Under the default (unset), `prefill_rows = 1` keeps
that token driver silently.
An unset `prefill_rows` resolves to 256; only width 256 is qualified (the measurements
below). Explicit widths from 2 up to the shared bound run the shared path at that width,
unqualified.

During an open prefill window, shared-owned steps use the common Projector and MoE pass.
Decode and verify use the existing launches with either arena owner.
Startup selects one `FlashStep` constructor and allocates one scratch arena.

See the [environment reference](reference/environment.md#flash-next-on-nvidia-cuda).

## Memory charge

The startup charge calculation constructs `FlashGeometry` only for ON.
The ON charge is the full aligned scratch size from `StepPlan::prefill_bytes`, not the difference from the legacy plan.
The OFF charge is zero.
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

- Correctness: shared256 and legacy64 completed greedy 128-token runs at 2.9K, 10.6K
  and 30K prompt tokens with no refusal, no non-finite logits and non-empty output;
  texts diverge early, as any chunk-width arithmetic change expects.
- Prefill speed: shared256 measured 1.47x to 1.50x faster end-to-end at those lengths
  (60.4/65.0/62.0 vs 40.6/43.4/42.1 prefill tok/s); at 117K depth 48.3 tok/s
  (40.4 min cold prefill), with prefix resume reusing 116,992 of 117,062 tokens
  (2.0 s to first token).
- Served cost: the ON arena charge holds the per-layer slot budget down - 196 vs 202
  slots at context 32768, 168 slots at context 131072 - and decode after a shared
  prefill ran 2 to 3.5 percent slower per token at the same 196 slots.

FP16 staging remains as implemented. No served performance gain is claimed here
beyond the measurements above.
