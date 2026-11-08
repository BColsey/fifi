# Combined shared and routed expert sums

The multi-device Qwen3.6 path reduces each card's routed expert outputs and
shared expert output across peers. Set `Q36_P2P_FUSE_EXPERT_SUM=1` and
`Q36_P2P_FUSE_SHARED_SUM=1` before starting the engine to send these outputs
in one reduction per MoE layer. The shared output occupies an additional hidden
row in each card's step arena; the peer mailbox is extended by the same row.

This path is experimental and off by default. The engine refuses the shared
option when the routed-expert option is absent. Remove
`Q36_P2P_FUSE_SHARED_SUM` and restart the engine to restore the separate
shared reduction. Existing peer-access and memory admission requirements still
apply.
