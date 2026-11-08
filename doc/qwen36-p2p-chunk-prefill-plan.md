# Qwen3.6 P2P chunked prefill: implementation plan

Status: design only. The multi-device session currently feeds one token at a
time; the single-device chunk runner cannot be enabled for it unchanged.

## Boundaries

1. Admit one chunk size on every participating card using that card's free
   memory, current decode arena, replicated resident state, workspace and all
   peer mailboxes. Allocate only after the same chunk size passes on all cards.
   Keep per-token prefill as the default and fallback.
2. Embed once, broadcast the rows, and run the layer stages in lockstep across
   the cards. Exchange the mixer partials before the following residual/norm.
3. For full attention, prepare each card's K/V slices, gather complete K/V rows
   on both cards, and check peer statuses before appending either replica.
   Adapt the chunk Q8 attention shape: its current checks require 16 query
   heads and two K/V heads on a single card.
4. Broadcast routing choices and weights from the routing card. Run each
   expert's sharded intermediate on each card, reduce the routed and shared
   partials, then combine. Do not change the order of floating-point
   additions without separate numerical evidence.
5. Commit session position and expose logits only when all card statuses and
   persistent states pass. An aborted chunk poisons the session until reset;
   it must never publish a partially appended KV cache as usable state.

The peer ring currently caps each payload at 1,048,576 f32 values. A Qwen3.6
chunk of 128 rows with hidden width 2048 and eight picks has 2,097,152 routed
output values; adding the shared output gives 2,359,296. The transfer therefore
needs bounded tiles or a different reduction strategy. Account for both
mailbox slots per peer and per card; never treat the single-device arena alone
as the full P2P allocation.

The chunk DeltaNet scan also assumes that its local value-head count is a
multiple of the key-head count. Its tiled key lookup uses local value-head
indices. The P2P layout can use asymmetric head ranges while replicating key
heads, so the scan needs checked global offsets and valid shapes before such
a shard can use it.

## Verification gates

- CPU: chunk sizes and tails, overflow and mailbox tiling, disjoint scratch
  lifetimes, admission on the tighter GPU, exact 131,072 total-token limit.
- Two peer GPUs: per-stage partials, routing, mixer and expert reductions,
  DeltaNet state, K/V rows and cache replicas against the one-token path.
  Include asymmetric placement, chunk boundaries, one-row tails, reset and
  injected peer failures before append.
- Full model: match greedy outputs and compare logits across short, boundary,
  long and decode-after-prefill prompts with the same context and placement.
  Measure prefill and decode separately, record per-card free VRAM, and keep
  raw, reproducible baseline and candidate samples.

## Initial sizing reference

The existing single-device Qwen3.6 sizing test reports a 20.12 MiB arena and
64 MiB attention-part charge for 128 rows using an 82-SM reference device.
This excludes the P2P-specific staging and mailboxes and is not an admission
result for either installed GPU. A two-card ring at the maximum f32 payload
uses four payload slots per card: about 16 MiB plus headers per card. Measure
or calculate each actual allocation on the 28-SM RTX 3060 and 68-SM RTX 3080
before deciding whether a chunk fits their configured 128k sessions.
