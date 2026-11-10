# Experimental chunked P2P prefill

The `Q36_P2P_CHUNK=<rows>` setting is an experimental, default-off path
for batching prompt processing across the P2P cards. Unset or zero uses the
existing per-token path. A positive row count is capped by the peer ring's
maximum payload. A malformed setting is rejected at startup.

The FP32 peer payload limit permits up to 1,024 rows at hidden width 2,048.
This larger chunk is experimental and can require more VRAM; use 512 rows
to return to the earlier limit. FP16 ring exchanges retain their previous
payload limit.

The chunk path is not qualified for production use. Its arena is admitted
only when each card can retain the configured free-VRAM floor; a failed
allocation or construction falls back to the eager path. Prefix reuse is
disabled while chunking is active until its numeric path and turn-chain
behavior have been qualified. The default eager path and its prefix handling
are unchanged.

For Q8 K/V, each card can attach an FP16 attention staging buffer after all
base chunk arenas are allocated. The buffer uses that card's K/V head count
and the same context-based size rule as single-device fixed staging. It is
admitted only if the card can retain its free-VRAM floor; otherwise that card
uses the existing tiled Q8 attention. Staging uses segmented attention beyond
its capacity when the checked partial-buffer plan fits. An allocation race can
also fall back to tiled attention; other CUDA failures refuse the chunk arena.
Tiled attention performs a blocking length update for each query tile.
Neither staged path has yet been qualified or measured on physical P2P cards
with this integration. Earlier measurements used a different prototype.

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

For internal diagnostics, the existing `Q36_TIERED_TRACE=1` switch also
records gate, up and down routed-projection stream intervals for the first and
last layer on each card's first prefill chunk (`p0 = 0`), deduplicating a
single-layer model. It allocates at most twelve timing events per card for that
chunk. A completed batch identifies card index, layer, row
count, quantization and allocation backing (`device`, `host-mapped`, or
`unknown`); card indices follow the serving process's device order. It emits
no batch unless all cards have completed chunk close and every interval is
valid. Other chunks and layers are not measured, and the selected layers need
not contain both backing classes. The batch has no request ID or device UUID;
run an isolated request and separately record the process's device order.

The intervals include scheduling, host enqueue gaps and other-stream
contention, and a projection can include fixup work. They do not establish
PCIe bytes, CPU utilization or the identity of individual selected experts.
Compare tracing on and off with the same binary and workload to measure its
overhead and check output parity. The same switch also enables the existing
`tiered-chunk: launching swiglu` log for every routed layer on both cards in
every chunk. Those writes may materially perturb a long-context run, so
trace-on overhead includes more than the selected timing events. Diagnostic
output is not a performance qualification.

An earlier, different 3060+3080 prototype produced one short cold sample
of about 106 prompt tokens/s and an eager sample of about 40 tokens/s. Those
observations are not measurements of this port and do not establish a
repeatable speedup or long-context behavior.

Before this path is proposed for use, build the exact commit with native
CUDA, pass the host and physical-peer full and segmented staging tests, and
compare repeated cold served prompts against the eager path with identical
weights and settings.
Record prompt and output token counts, prefix reuse, prefill and decode
rates, VRAM usage, device identities, and failures. Separately validate
128k and near-256k prompts with reserved output and marker recall; synthetic
recall alone is not evidence of coding quality.
