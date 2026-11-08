# Experimental local K/V rows for sliced P2P caches

Status: unmeasured and default-off. This option requires the multi-device
Qwen3.6 engine with per-card sliced K/V caches.

Every card prepares the K/V rows for its own head groups at their existing
offsets. The sliced-cache append reads exactly those local offsets, so the
other cards' rows are not needed for that append or for the card's attention
decode. Set `Q36_P2P_LOCAL_KV=1` at engine startup to omit the K and V
all-gathers. The existing host check of every card's ring status still runs
before either card queues its cache append. No K/V arena layout changes.

For Qwen3.6-35B-A3B, the intended effect is 20 fewer ring exchanges per token
(two in each of ten full-attention layers). This is an exchange count, not a
measured speedup. The option needs two-peer-GPU cache and logits parity,
peer-failure checks, and a controlled same-binary benchmark before it is relied
on. The session fixture compares Q8 and FP16 cache prefixes after five tokens,
both first and later logits, exchange counts, and the pre-append fault gate;
full-model and long-context validation remain separate requirements.

To roll back, remove `Q36_P2P_LOCAL_KV` and restart the engine. The original
K/V gathers are then used.
