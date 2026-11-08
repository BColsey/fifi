# Running a second engine on an idle GPU

This guide shows how to run a second, independent `enginesrv` on a spare GPU
next to an existing deployment (for example a two-card P2P pair), and how to
serve both models through one LAN web GUI. Everything below was deployed and
measured on 2026-11-08 on ben-X570 (P2P pair: RTX 3080 + RTX 3060, second
instance: RTX 2080 Ti 11 GB, Qwen3.6-35B-A3B Q4_K_M GGUF).

## Why a second instance, not a third ring member

An idle GPU of a *different* compute capability cannot join a multi-device
device set at all: `check_peer_facts` refuses mixed compute capabilities
(`engine/src/kernels/cuda/peer.rs`). Independent of that, the Qwen3.6 shard
geometry refuses 3 cards at Q4_K_M: the 16 attention query heads form exactly
two 8-per-KV-head decode groups, and the expert down projection's K width of
512 is exactly two Q4_K superblocks, so any 3-way cut splits a quant block
(`engine/src/family/qwen36/cuda/shard.rs`). A card behind a chipset link is
also strictly worse than host RAM for expert traffic. A second process is the
only shape that works — and it is fully independent: separate KV cache,
separate sessions, separate failure domain.

## Build

Use the standard scratch worktree and a task-local `build-env.sh`. The one
new requirement is the arch list: a Turing card (sm_75) needs
`INFERENCE_CUDA_ARCHS=75`, because the build emits PTX for the newest listed
arch only and PTX does not JIT downward:

```bash
export INFERENCE_CUDA_ARCHS=75        # or "75,86" to cover both generations
export CUDA_VISIBLE_DEVICES=GPU-<uuid-of-the-spare-card>
cargo build --release --locked --features native-cuda --bin enginesrv
```

The kernels on the single-device serving path are kept sm_72-clean by
construction (the aarch64 default), so sm_75 compiles; the pre-Ampere arms
had never been board-run before 2026-11-08, when a full smoke (auth, models,
chat, streaming) passed on a 2080 Ti.

## Config

Single-device serving is first-class: omit `cuda_devices` (it defaults to
`[0]`, which is the one visible device). Tiered experts are what fit the
20 GB checkpoint on an 11 GB card: dense weights are resident, experts
stream from the model file's page cache. Reuse the primary deployment's
`model_dir` (read-only mmap, shared page cache — the second instance then
becomes ready in seconds).

```json
{
  "listen": "127.0.0.1:18793",
  "api_key_env": "ENGINESRV_API_KEY",
  "engines": [{
    "id": "q36-spare",
    "family": "qwen3_6_35b_a3b",
    "model": "fifi-q36-spare",
    "model_dir": "/path/to/shared/model",
    "tokenizer_dir": "/path/to/shared/model",
    "qwen36": {
      "context_limit": 131072,
      "kv_dtype": "q8",
      "capture": false,
      "tiered": { "slots": 85, "threads": 8, "preload": true }
    }
  }]
}
```

Give the engine a distinct `model` id (not the primary's `fifi-q36`) so a
merged endpoint can route by model name. Measured on the 2080 Ti at the
262144/q8 shape: 9.7 of 11.3 GB VRAM (~1.5 GB margin), ready in 7 s with a
warm page cache, ~26 tok/s decode on short contexts (f32 KV measured ~16).

Two engine-level facts to know before tuning further:

- **`kv_dtype: "q8"` over `"f32"` is a large win** — 4x smaller KV and a
  cheaper attention path; it is what makes 256k context fit an 11 GB card.
- **`kv_offload_host_bytes` (host-pinned KV spill) is refused with tiered
  experts** — the two are mutually exclusive, and tiered is what fits the
  model, so the spill knob is unusable in this shape. 256k fits in VRAM at
  q8 anyway.

## Prefill is per-token in tiered mode

The chunked prefill path requires non-tiered q8 with fixed admission and
context staging; tiered mode forces per-token feeding (the P2P pair is
per-token for its own reason — multi-device also refuses chunked prefill).
Measured on the 2080 Ti, per-token prefill still out-runs the pair: 45 tok/s
at ~3k prompt tokens rising to 64 tok/s at ~22k, versus the pair's flat
39 tok/s. The very first prompt after a restart can be much slower (~8 tok/s
at first touch) while the tiered banks fault cold experts into the page
cache. Budget ~16 s per 1000 tokens for large pastes; continued chats on the
same conversation only feed new turns.

## One LAN GUI, both models

The web GUI already lists models from `/v1/models` and sends the selected
`model` on every request, so no GUI change is needed. Put a loopback-only
model router between the vite proxy and the engines (`scripts/web-model-router.mjs`):
it merges `GET /v1/models` across backends and routes every other request by
its `model` field, injecting the matching backend's API key. Point the proxy
at it with the existing env hook (`export FIFI_BACKEND=http://127.0.0.1:18795`
in `start-web.sh`) and add `Wants=`/`After=` on the router unit. Requests
without a model field (monitor panels, `/v1/system`) report the primary
engine by design.

## GPU test coordination

A permanently resident process on the spare GPU breaks any machine-wide
"no compute processes" gate. Two policy fixes, both deployed here:

- **Scope the lease runner by UUID**: in `gpu-run-local`, check
  `nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps` when
  `CUDA_VISIBLE_DEVICES` is in UUID form (indices enumerate differently for
  nvidia-smi vs CUDA children — do not scope on bare indices), falling back
  to the machine-wide check otherwise. This is never looser than the
  original guard.
- **Fix the other lanes' gates the same way**: any script with a
  machine-wide `--query-compute-apps` idle check must either scope to the
  GPUs it actually uses or allowlist the second instance's `enginesrv`
  path. Mark such edits and leave a note in each lane's coordination
  ledger; measurement lanes that need a PCIe-quiet window stop the second
  instance for the window (stop → lease → test → restore, as before).

Measured interference: the primary pair decoded 46.7 tok/s alone vs 44.5
tok/s while the second instance generated full-rate concurrently (~5%
worst case, ~0 when idle-resident).

## Measured summary (2026-11-08, short contexts unless noted)

| engine | prompt_t | prefill | decode | notes |
|---|---:|---:|---:|---|
| P2P pair (3080+3060) | 747 | 39 tok/s | 35 tok/s | warm-prefix TTFT 0.26 s |
| 2080 Ti 64-slot, threads 6, 256k | 22,447 | 64 tok/s | 45 tok/s | |
| 2080 Ti 64-slot, threads 8, 256k | 15,762 | 62 tok/s | 48-50 tok/s | |
| 2080 Ti 85-slot, threads 8, preload, 131k | 22,447 | 75 tok/s | 50-53 tok/s | no first-touch crawl |

Measured 2026-11-08, temperature 0, one run per point; see
`scripts/bench_engines.py` to reproduce. The second instance is
competitive-with-or-faster than the pair at matched context — the 5%
interference figure above was measured on short-context decode.
