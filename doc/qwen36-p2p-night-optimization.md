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
