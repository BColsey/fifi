# Qwen3.6 P2P chunked prefill: implementation plan

Status: design only. The multi-device session currently feeds one token at a
time; the single-device chunk runner cannot be enabled for it unchanged.

This plan was first written against a build with replicated K/V caches. The
currently deployed experimental branch includes KV slicing (`3bd5ac7`):
each card now owns only its dealt K/V head groups. Rebase and revalidate the
transfer primitive and the admission calculation against that branch before
using this plan. The production P2P chunk path is still unimplemented.

## Boundaries

1. Admit one chunk size on every participating card using that card's free
   memory, current decode arena, that card's resident state, workspace and all
   peer mailboxes. Allocate only after the same chunk size passes on all cards.
   Keep per-token prefill as the default and fallback.
2. Embed once, broadcast the rows, and run the layer stages in lockstep across
   the cards. Exchange the mixer partials before the following residual/norm.
3. For full attention on the sliced-cache branch, prepare and append each
   card's own K/V head groups. Preserve a cross-card peer-status check before
   either persistent cache is changed; removing a gather does not remove this
   failure boundary. The older replicated-cache branch needs complete K/V rows
   on every card instead. Adapt the chunk Q8 attention shape: its current
   checks require 16 query heads and two K/V heads on a single card.
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

## Performance accounting

Batching can reduce the number of launches and synchronization points per
token, and can reuse weights across rows. It does not by itself reduce the
payload bytes per token: a full-width f32 row still contributes 2,048 values
for each token. For example, a 2,048-row, full-width f32 sum carries 16 MiB
of values, before transport overhead. At the current 4 MiB payload limit, a
contiguous sum of that size needs at least four tiles. A fused buffer holding
eight routed outputs and one shared output at the same shape contains 144 MiB
and needs at least 36 such tiles. These are payload sizes, not total mailbox
allocations or measured PCIe traffic.

Treat projected throughput as a hypothesis until the complete card path is
implemented and measured against the same binary and configuration. The
experimental card-prefix path also still needs physical-peer and full-model
qualification; a prefill path tag alone does not establish restore parity.

## Verification gates

- CPU: chunk sizes and tails, overflow and mailbox tiling, disjoint scratch
  lifetimes, admission on the tighter GPU, exact 131,072 total-token limit.
- Two peer GPUs: per-stage partials, routing, mixer and expert reductions,
  DeltaNet state, K/V rows and each card's cache-head slice against the
  one-token path.
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

## Transfer primitive status

A staged `PeerRing::exchange_sum_tiled` divides an oversized contiguous sum
into mailbox-sized transfers. It checks sequence capacity, buffer ownership,
lengths and unsafe overlaps before queuing the first tile. No prefill caller
uses it yet. CPU tests cover the transfer boundaries and overflow checks; the
physical two-GPU stress test is still required before integrating the method.
