# Flash-Next shared prefill

## Switch and dispatch

The switch `INFERENCE_FLASH_NEXT_SHARED_PREFILL` selects the shared EXL3 prefill path.
Unset, `0`, `false`, and `off` keep the existing path.
`1`, `true`, and `on` select shared prefill.
Any other value refuses startup before CUDA initialization.
When ON, startup refuses `prefill_rows <= 1` before CUDA initialization.
The default chunk width stays 64 rows: `prefill_rows` defaults to 64, the legacy path accepts
at most 64 rows, and the shared path plans up to the shared bound of 256 rows.

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

ON uses FP16 staging and remains opt-in.
No device qualification or served performance gain is claimed here.
