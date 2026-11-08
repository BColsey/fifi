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

KV is replicated full-size on both cards (10,880 B/position, 2.85 GB per
card at 262144). Admission arithmetic at 256k needs each card's tensors
below `free - 2.85 GB - floors`; the even 50/50 shard deal puts card 1 at
its floor (only ~6.05 GiB of its expert mass is host-eligible), 385 MB
short. So 256k requires an uneven shard split - the waterfill planner.

`Q36_P2P_WATERFILL=1` (e3d286c, default off) wires `plan_placement` in with
host budgets as free-byte relief; on this pair it now chooses
"bandwidth-waterfill" but still fails closed at load: at Q4_K the DeltaNet
value-head dimension can be dealt into row counts the native matvec refuses
(`rows % 4 != 0`, observed on `blk.0.ssm_beta`). Fix: dim-0 grain becomes
`lcm(head grain, 4)` in `placement.rs::dims`, and the remainder dealing in
`runs_from_counts` needs a test for it. Also note the planner geometry-caps
attention query heads and expert intermediate at this quant, so the expected
256k win comes from moving vocabulary/delta-head mass, not experts.

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
