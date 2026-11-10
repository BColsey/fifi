# Experimental chunked P2P prefill

The `Q36_P2P_CHUNK=<rows>` setting is an experimental, default-off path
for batching prompt processing across the P2P cards. Unset or zero uses the
existing per-token path. A positive row count is capped by the peer ring's
maximum payload. A malformed setting is rejected at startup.

The chunk path is not qualified for production use. Its arena is admitted
only when each card can retain the configured free-VRAM floor; a failed
allocation or construction falls back to the eager path. Prefix reuse is
disabled while chunking is active until its numeric path and turn-chain
behavior have been qualified. The default eager path and its prefix handling
are unchanged.

The present card chunk arenas do not attach Q8 K/V staging. Q8 attention
therefore uses the unstaged tiled fallback, which performs a blocking length
update for each query tile. Its impact at long contexts has not been measured
on this port. Staged attention on card geometry needs separate investigation
and device validation before it can be enabled.

When this experimental path is active, the card executor advertises the
admitted chunk size to the serving worker. With chunking off, the worker keeps
its existing fallback. The chunk driver leaves one row to the eager path, and
the session keeps one pending token across pieces. Large pieces need physical
device correctness, memory-admission, cancellation and long-context checks;
they are not yet qualified on this port.

## Design

Each card builds a chunk arena and model using its own mixer geometry and
expert-intermediate slices. The embedding and routing run on card 0 and are
broadcast to the others. The cards sum their partial mixer outputs before the
feed-forward normalization, and sum their partial expert outputs before the
next layer. The mailbox is sized to carry the larger of a chunk sum and an
eager decode exchange. Requests larger than the selected chunk size are
split into bounded pieces; a failed piece poisons the session. Each of the
three consecutive routing broadcasts waits for every card to consume it
before the next broadcast. This protects mailbox reuse and has an unmeasured
performance cost.

Both cards compute the selected experts over their intermediate slices. The
existing eager K/V gathers are retained. A physical peer test must check both
single- and multiple-chunk output against per-token execution, using the
intended devices selected through `CUDA_VISIBLE_DEVICES`.

## Evidence needed

An earlier, different 3060+3080 prototype produced one short cold sample
of about 106 prompt tokens/s and an eager sample of about 40 tokens/s. Those
observations are not measurements of this port and do not establish a
repeatable speedup or long-context behavior.

Before this path is proposed for use, build the exact commit with native
CUDA, pass the host and physical-peer tests, and compare repeated cold
served prompts against the eager path with identical weights and settings.
Record prompt and output token counts, prefix reuse, prefill and decode
rates, VRAM usage, device identities, and failures. Separately validate
128k and near-256k prompts with reserved output and marker recall; synthetic
recall alone is not evidence of coding quality.
