# Experimental combined P2P expert sums

The two-device Qwen3.6 GGUF path computes a partial down-projection row for
each selected expert on every card. By default it sends and sums each row
separately. With `Q36_P2P_FUSE_EXPERT_SUM=1`, the contiguous expert rows are
sent and summed as one larger f32 payload. The shared expert is still summed
separately. Every output element keeps the same per-card addition order, and
the normal ring status checks still apply.

The option is off by default and is read at engine startup. It allocates a
larger per-card ring mailbox, sized with checked arithmetic, before weight
admission. Remove the variable or set it to `0` and restart to restore the
original per-row exchanges. Measure on the target system before relying on
the experimental path.

For a controlled comparison, use the same model, context and host-expert
placement for both runs. [The prefill driver](../scripts/bench_p2p_prefill.py)
reports prompt tokens divided by the time through a short reply. The
[decode driver](../scripts/bench_p2p_decode.py) matches each reply to the
server's request log and uses its token-based decode rate. Both drivers
support `--self-test`; run `--help` for their endpoint and output options.
