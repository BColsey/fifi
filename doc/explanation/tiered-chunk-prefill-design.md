# Batched tiered prefill — design

Status: design, not implemented. Everything below was verified against the tree at
`a81ebdf` by a three-lane source read (chunk driver, tiered host, KV/state). The
motivating measurements live on MRs zbrdc/fifi#14 (per-token tiered prefill is
decode-rate: 50–75 tok/s) and #15 (prefix reuse makes cold prefill a
once-per-unique-prompt cost, so cutting it ~2x is still worth real money).

## Why this is smaller than it looks

Three ingredients already shipped with tiered chunking in mind:

- **The grouped MMQ kernel takes bank stacks and host-built ids.**
  `MmidWeight` is documented as "a NativeWeight or a bank of slots with
  `n_experts = slots`" (`kernels/cuda/prefill/mmq_mmid.rs:346`), and
  `MmidIds::from_host` is "the tiered planner's checked host constructor"
  (`mmq_mmid.rs:129`). The device guard (`IdsGuardCall`) rewrites duplicate and
  out-of-range ids, so tiered misses (duplicate slot 0) must NOT flow through
  `mmq_build_ids`; the host builds ids, exactly as decode does
  (`cuda/moe/tiered.rs:326`).
- **The `ChunkMoe::Tiered` arm already exists** as an anticipated refusal stub
  carrying `runtime` and `layer` (`cuda/prefill.rs:2223`), and the router seam
  documents the one-line change (`prefill.rs:2429`).
- **M-row KV, attention, state publication and prefix export are MoE-mode
  agnostic** (`prefill.rs:1761-2120`, the `revalidate` protocol). Zero work there.

## The honest ceiling

Batching removes the per-layer-per-token host round trips and job overhead, not
the CPU expert arithmetic: a 512-row chunk still owes ~345 GMAC of CPU expert
work through 8 threads. Expect **~2x cold prefill (90–100 tok/s)**, not the
chunk-path GPU rate. The plan should say this up front so nobody measures 300
tok/s and calls the implementation broken.

## Slices (each independently landable)

1. **Kernel seam — no code.** Verify `MmidIds::from_host` + bank `MmidWeight`
   compile for the qwen36 shapes (`MmidShape { n_experts: slots, m, top_k }`,
   m ≤ 2048 per `mmq_mmid.rs:449`).
2. **Runtime accessors + scratch** (~80 lines, `tiered_step.rs`,
   `moe/tiered.rs`): `TieredRuntime::layer_weights(layer)` building
   `MoeLayerWeights` from parts + bank stacks; a `[m][hidden]` CPU-partial
   buffer and an `[m]` ones buffer (the M-row twins of the decode scratch).
3. **Host M-row protocol** (~250 lines, `tiered/host.rs`; the biggest piece):
   `begin_chunk_layer(layer, ChunkRouteRead)` / `finish_chunk_layer` beside the
   single-row twins — validate all m rows first (`check_route` per row, the
   `account_token` validate-then-act pattern), classify m×top_k picks against
   the table, misses get device weight 0.0, routes stashed per row for the
   chunk boundary. Decode is untouched.
4. **Grouped CPU jobs** (~250 lines, `tiered/cpu_pool.rs`, host-only tests): one
   job per (layer, distinct missed expert) carrying its row list and per-row
   router weights over a shared `[m][hidden]` activation block;
   `PendingChunk::finish` merges in submission-time per-row pick order.
5. **The prefill Tiered arm** (~250 lines, `prefill.rs`): the batched route
   readback (one pinned batch per layer, the twin of decode's), host-built
   grouped ids with **miss columns spread round-robin over the slots**
   (the `≤ m columns per expert` invariant of `check_host_ids`,
   `mmq_mmid.rs:207`, can refuse adversarial small-m routing — the spread rule
   needs its own check and error), the miss-weight patch on `TopkWeight`, and
   one M-row `moe_combine` per layer for the CPU partial (the kernel is already
   M-row, `prefill.rs:665`).
6. **Gate + admission** (~100 lines): teach `needs_arena`/`chunk_verdict` to
   admit tiered (behind a flag, default off), charge the existing banks as an
   admission Extra, and validate stacks via `check_bank` facts with
   `ModelFacts.experts = slots` (D5/D7/D8 then hold for banks).
7. **Tests** (~150 lines): flip the refusal pin (`prefill_device_tests.rs:2303`)
   to a run-pin, add host-arithmetic pins for the batch classify and the
   grouped-pool merge.

## Ordering traps (the state lane's list — each is a bug if missed)

- **Status contract**: the tiered MoE launches must OR faults into
  `Word::PrefillStatus` exactly like the native stages; host-side tiered errors
  must poison through the same one-read-per-chunk rule.
- **Publication order**: body → read status → `revalidate` → end-of-chunk tiered
  accounting (routes, policy, fills on the owning stream) — the chunk twin of
  commit-then-`end_tiered_token`.
- **Error cleanup**: any body error must run the `abort_token` analog, or a
  layer that never reached `finish_chunk_layer` leaves `in_flight` set and every
  later token refuses with `HostError::Order`.
- **One pair per layer**: a chunk processes one layer's m rows as ONE
  begin/finish pair — which is also where the speedup comes from.
- **Bank stability**: policy swaps stay at chunk boundaries; a chunk classifies
  all rows against residency at chunk start, so the grouped launches read a
  stable bank. Per-layer syncs drop from m×(1 wait + 2-3 writes) to 1 wait +
  ~4 writes.

## Known follow-on effects

- The run report's per-token tiered evidence moves to chunk granularity; the
  job line flips from `verdict Tiered path per_token` to `Ready/chunk`. Any
  parser keyed on those labels needs the update.
- Chunk-boundary policy sees m rows at once (`boundary_chunk`, credit
  m×swaps_per_token once) — swap pressure per boundary rises with m; the
  guess-mode study on MR #14 measured how sensitive decode is to bank churn.
