# Experimental multi-block P2P ring

The peer ring normally publishes and reduces each payload with one CUDA
block. Setting `INFERENCE_CUDA_P2P_RING_MULTIBLOCK=1` before starting the
engine allows larger f32 publishes and f32/FP16 reductions to use multiple blocks.
Payloads of 2,048 values or fewer retain the single-block path. FP16 publishes
also retain one block: their overflow flag lives in peer memory, and not all
P2P links support remote atomics. Only the exact value `1` enables the
experiment; unset the variable and restart to roll back.

The switch is off by default and is not qualified. It does not implement
chunked P2P prefill by itself. Before enabling it for normal serving, compare
the opt-in and default paths on the exact build with two physical peer-enabled
GPUs. Cover f32 and f16 payloads, tails, overflow and timeout status, repeated
mailbox reuse, and complete model logits and replies. Measure prefill and
decode separately using the same binary and configuration, and retain the raw
samples and process/device identities.

An earlier device log and a failed attempt are in the [observation
record](../scripts/benchmarks/p2p-ring-multiblock-observations.json). They do
not identify the exact binaries or supply a matched baseline, so they do not
establish a speedup or qualify this implementation.
