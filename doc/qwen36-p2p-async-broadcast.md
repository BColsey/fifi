# Experimental P2P broadcast event barriers

The Qwen3.6 multi-device path uses a double-buffered peer mailbox. Consecutive
broadcasts need a barrier so a publisher cannot reuse a slot before every
consumer has copied its contents. `Q36_P2P_SYNC_BROADCAST=1` provides that
barrier by synchronizing each GPU stream on the host.

Set `Q36_P2P_ASYNC_BROADCAST=1` before starting the engine to use CUDA events
for this barrier instead. Each consumer records an event after its copy, and
the publisher queues a wait for every peer before submitting later work. This
option takes precedence over `Q36_P2P_SYNC_BROADCAST` and is off by default.
It is intended for the non-captured multi-device path; cross-device event
waits are refused during graph capture.

To roll back, remove `Q36_P2P_ASYNC_BROADCAST`, retain
`Q36_P2P_SYNC_BROADCAST=1`, and restart the engine. Measure on the target
system before relying on the experimental path.
