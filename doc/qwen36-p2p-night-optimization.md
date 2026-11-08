# Night optimization: 3060+3080 P2P decode and prefill (2026-11-08)

Goal: raise decode and prefill tokens/s of the two-device Qwen3.6-35B-A3B
Q4_K_M deployment for 128-256k coding sessions, keeping outputs identical.

All numbers: same machine, `q8` KV, 131072-token context, `bench_p2p_prefill.py`
3-sample medians (524-token prompts) and SSE-timed decode over 120 numbered
lines, unless noted. Day-to-day noise on this rig is about 4%.

## Results

| Configuration | prefill 524 (tok/s) | prefill 4274 (tok/s) | decode short (tok/s) | decode @4.3k |
| --- | ---: | ---: | ---: | ---: |
| baseline (deployment, 6 GiB host experts) | 32.72 | 32.7 | 21.46 | 21.35 |
| fused MoE ring exchange (d0569ca) | 34.36 | - | 22.58 | 22.49 |
| fused + 4.5 GiB host budgets (deployed tonight) | 39.38 | 40.33 | 26.0 | 26.1 |

First full-context run on this rig (deployed config, 130,981 prompt tokens,
streaming): TTFT 3,736.9 s = 62.3 min (~35.1 tok/s prefill at full depth -
attention prep costs ~14% versus the 4.3k rate), decode ~30 tok/s by the
server's own metric at 131k depth (19.4 measured client-side through the
SSE proxy), and the model still counted 1..120 correctly at full depth.

- Fusing the MoE all-reduce from `top_k+1` ring exchanges per layer to 2
  (the whole `expert_down` block, then `shared_out`) is worth ~5% on every
  token: ~320 serialized PCIe round trips removed; the measured 1.45 ms/token
  saving matches ~4.5 us per removed sequence.
- Rebalancing the host-expert budgets from 2+4 GiB to 1.25+3.25 GiB (1.5 GiB
  more weights GPU-resident, found by the resident-reserve lane the same
  night) stacks another ~15%: the per-token cost is dominated by host-expert
  reads over PCIe, so every GiB moved back to VRAM is worth ~3 tok/s.
- Correctness: 11 card parity device rigs, the full card-session rig, the
  prefill-head-skip peer rig, and every benchmark answer match the baseline.
  Evidence: `qwen36-local/night-opt/*.json` on the deployment host.

## What still blocks the 256k goal

Resolved and re-scoped (second night session): `Q36_P2P_WATERFILL=1` now
serves end to end (grain fixes 9267ca0 + 10b8e52: native-matvec rows move
in multiples of four, and value heads in whole key-head groups - the GDN
kernel's `n_v_heads % n_k_heads == 0` pairing rule). Measured 40.34 tok/s
prefill vs 39.38 even: within noise, because at Q4_K every heavy dimension
is geometry-pinned (experts at the 256-value grain, query heads in 8-groups
over 16 heads, value heads 16+16 as the only legal deal) and the plan can
tilt only vocabulary (~286 MB of mass, ~100 MB moved). Placement cannot
rescue 256k at this quant; the planner flag stays default-off, verified
and available for quants whose experts are expressible (Q8_0's 32-value
grain).

The real 256k path is **KV slicing**: today every card replicates the full
KV planes (10,880 B/position each; 2.85 GB/card at 262144) even though
each card only decodes against its own K/V head groups. Slicing the cache
by kv-head ownership frees ~1.43 GB per card at 256k; with that and the
existing host-expert offload at its floor (card 1 tensors 6.75 GB vs a
9.48 - 1.43 - 0.27 = 7.78 GB budget), 256k admission fits with ~1 GB of
margin. The step's K/V gather already broadcasts full rows, so the
append kernel's plane layout and the resident-state allocation are the
affected pieces (card_step.rs:145-150, placement.rs:395-415 budget notes).

## Ranked remaining levers (from the subsystem map, evidence-based)

1. Host-expert traffic: the dominant per-token cost. Options: move budgets
   back toward VRAM as free memory allows (measured ~3 tok/s per GiB);
   stage/overlap host reads with compute (`expert_tier/overlap.rs` pattern).
2. Per-token host syncs: ~12 blocking stream syncs per token (gather gate
   per full-attention layer, 4-byte status reads, ~497 KB logits readback).
   Device-side argmax merge over the ring (8 B/card) + gating once per token
   is the next structural win.
3. Chunked prefill: the ring ABI already counts `rows * width`; M-row
   exchanges + 32-64-row prompt chunks amortize per-exchange latency ~50x.
   This is the 128k+ TTFT lever (a 131k prompt is ~54 min at 40 tok/s).
4. Prefix save/restore for `CardSession` (refused today, `card_session.rs:13`):
   Flint's `cuda_prefix.rs` is the in-repo reference. For coding sessions
   this beats every per-token optimization: no re-prefill on follow-up turns.
5. Fuse the 3 route-broadcast word exchanges per layer into one packed
   broadcast (needs contiguous route scratch), ~80 sequences/token.

## Artifacts

- Branch `task/qwen36-ring-fuse-moe`: d0569ca (fused exchange), e3d286c
  (waterfill wiring, default-off, known gap documented above).
- Deployment: `fifi-qwen36-engine.service` runs the fused binary with the
  4.5 GiB host budgets; rollback is `start.sh.pre-night-opt`.
- Benchmarks and telemetry: `qwen36-local/night-opt/` (baseline/fused/
  combined json, dmon csv, engine logs).
- The full-128k characterization (first ever on this rig) is recorded in
  `qwen36-local/night-opt/full128k-result.json`.

## Second session addendum (same night)

- `Q36_P2P_WATERFILL=1` now serves end to end after two planner grain
  fixes (9267ca0: native-matvec rows in %4 blocks; 10b8e52: value heads in
  whole key-head groups - the GDN kernel's `n_v_heads % n_k_heads == 0`
  pairing rule, caught by serving the real model, where 12+20 heads
  refused the first token). At Q4_K all heavy dims are geometry-pinned, so
  the flag tilts only vocabulary: 40.34 vs 39.38 tok/s, noise. Deployment
  stays on the even split; the flag is verified for quants whose experts
  are expressible (Q8_0).
- Robustness: the engine service now self-heals (systemd drop-in
  `Restart=on-failure`, 15 s, 3 per 10 min) because the P2P failure model
  poisons sessions irrevocably by design. Verified live: kill -9 ->
  restarted in 15 s -> ready in ~135 s -> correct answer.
- The per-attention-layer gather gate (10 host syncs/token) is
  load-bearing: its doc pins "refusal BEFORE the append queues, so a dead
  gather cannot write a stale row into the replicated cache". The correct
  removal is device-side status guards in the native append kernel - the
  mechanism multi.rs:74-76 already names as the future non-fatal path.

## Third session: KV slicing - 256k context achieved (3bd5ac7)

Each card's KV planes now hold only its own K/V head groups (the shard
table already deals them with the query heads): the pair stores exactly
one replica of cache instead of two, the append quantizes the card's
slice of the gathered row (new k_at/v_at op offsets), the card decode
addresses its planes from local head 0, the state identity records the
slice, and the multi-device load reserve swaps the full-replica state
term for the largest card's. The gathers and the both-halves gate are
unchanged.

- 131k, same budgets: prefill 40.30 tok/s, decode 26.3 (+2% over
  combined) - the smaller planes also help L2/append work.
- **262,144-token context now admits and serves** (was structurally
  impossible: full-replica KV left card 1 exactly 385 MB short even at
  its host-offload floor): 41.3 tok/s prefill at the same budgets
  (card 0 nudged +50 MB host), answers verified. Not yet exercised
  end-to-end: a >131k prompt at depth.
- Validation: 1408 host tests; 11 card parity rigs byte-exact over each
  card's head slice (codes and scales, K and V, both cards, whole-40-
  layer model vs reference); session and peer rigs; the shards rig now
  pins "the sliced pools hold exactly one replica".
- Deployment: the service runs the sliced 256k build; rollback is
  `start.sh.pre-kv-slicing` (or `.pre-night-opt` for the original).

Phase 2 (next): drop the K/V gathers entirely - each card's prep already
computes its rows locally, so appending from AttnProj removes 2 ring
sequences per full-attention layer (~20/token) and shrinks the KRows/VRows
slots to the card's width. After that, prefix save/restore (the multi-turn
capability) and chunked prefill remain the top levers.

## Fourth session: card prefix save/restore (wired, gated off)

The CardSession now implements the ExecutorOps prefix hooks over the
shared session rules: a save exports one state per card (its sliced KV
planes plus its DeltaNet slots), a restore imports them in order, and
the card states' books are enqueued/committed per token so exports carry
the true count; after an import the next feed rewrites the full step
params (the import zeroes the word block, and the position must restart
at the restored count). Q36_P2P_PREFIX=1 opts in; the default is OFF
because the feature is NOT yet qualified:

- Repeat requests restore and answer 46x faster (0.28 s vs 13.2 s,
  reused_tokens 517/524) - but that bench's answer is filler-agnostic,
  and it initially passed while the restore was in fact broken (the
  export carried tokens=0 because the card books were never committed;
  the fix adds the enqueue/commit pair).
- The deeper turn-chain check (recall question across a warm multi-turn
  conversation) passes on answers today, but the end-of-turn save
  refused with a fed/key mismatch in one instrumented run and the
  follow-up validation was blocked by GPU contention from the other
  automation lanes. The bookkeeping arithmetic (fed vs key length with
  the pending-token rule) needs a device rig before this ships - the
  session.rs rules suite is single-device-calibrated and the card path
  reports a manual counter.
