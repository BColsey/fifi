# Qwen3.6 multi-GPU P2P execution

The multi-device Qwen3.6 path shards each token's work across GPUs and uses a
peer mailbox ring to share intermediate rows. Host-backed expert slices let a
card read selected expert weights from system memory when its shard would not
otherwise fit in device memory. The mechanisms and their admission limits are
described in [host-backed experts](qwen36-p2p-mapped-experts.md).

## Routed expert reductions

Each card writes a contiguous block of routed expert down-projection rows.
This branch exchanges the block in one ring operation, then exchanges the
shared expert row separately. Every card combines the reduced results with its
residual. The mailbox must be large enough for all routed rows. This branch
does not include a runtime switch back to the per-expert exchange path.

## Placement

`Q36_P2P_WATERFILL=1` opts into the bandwidth-aware placement planner. The
planner considers per-card host-expert budgets and deals tensor dimensions in
grains accepted by native matrix-vector kernels. DeltaNet value heads remain
in whole key-head groups. Without the option, the existing placement is used.

## K/V cache and failure boundary

Each participating card keeps a full replica of the attention K/V cache in
this branch. The two K/V gathers complete before the cache append; a host-side
ring-status check refuses a failed exchange before persistent state is used.

## Performance considerations

Host-backed expert reads use PCIe bandwidth on every token. Placement can trade
host traffic against device-memory headroom, so compare placements on the
target machine with the same model and runtime settings. The placement option
does not remove the need to check the memory floors on every participating
card.

The K/V status gate synchronizes the cards before persistent cache writes.
Removing or moving that boundary requires an equivalent failure check; a
failed exchange must not leave a cache that can be reused as valid state.

Possible follow-up work includes batching prompt rows across cards, overlapping
host expert reads with compute, and combining route-broadcast payloads. These
are separate changes with their own memory, ordering and correctness checks.

## Evidence and limits

The [measurement record](../scripts/benchmarks/qwen36-p2p-night-results.json)
contains raw samples, provenance and limitations. In particular, the earlier
streaming decode script counted content chunks rather than output tokens, and
the long-answer checker verified only the first five requested numbers. Neither
is evidence of complete answer correctness or a token-based decode rate.
