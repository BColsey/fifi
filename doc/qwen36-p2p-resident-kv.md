# Experimental P2P resident KV admission

The multi-device GGUF path allocates each card's resident KV cache and step
buffers before loading weights. By default, weight admission also reserves a
second copy of the full KV charge, leaving that space unused. Setting
`Q36_P2P_PREALLOCATED_KV=1` before startup removes the duplicate KV charge from
weight admission. The option is off by default, and applies only to the
multi-device path.

The weight loader still admits against free memory measured after the state
allocations. It reserves at least 1 GiB on each card, and the existing check
after weight loading still rejects a card below that floor. The option does not
shrink the allocated KV cache or change the configured context. Free device
memory and other processes can change; the floor is not a guarantee that all
future allocations will fit.

On a configuration using host-backed experts, a lower
`Q36_P2P_HOST_EXPERT_BYTES_0` or `Q36_P2P_HOST_EXPERT_BYTES` value can use
some of the recovered room to keep more expert weights on the GPUs. Those
settings specify minimum bytes to map into host memory; the loader selects
complete slices and can map slightly more. Measure the resulting speed and
remaining VRAM on the target system before relying on the configuration.
Remove `Q36_P2P_PREALLOCATED_KV` or set it to `0`, then restore the former
host-mapping budgets and restart to roll back.

## One measured placement

On an RTX 3060 12 GB and RTX 3080 10 GB, with the Q4_K_M GGUF model,
q8 KV, a 131,072-token configured context, synchronized broadcasts, and
prompt-only head skipping enabled, the following used the same binary and
three 524-token requests per placement:

| Host-mapping requests, card 0 / card 1 | Prompt tokens/s | Median |
| --- | --- | --- |
| 2 GiB / 4 GiB | 32.751, 32.761, 32.757 | 32.757 |
| 1.25 GiB / 3.25 GiB | 37.847, 37.837, 35.691 | 37.837 |

The tuned placement raised the median by 15.51%. The engine reported
1,536,753,664 and 1,726,611,456 free bytes after loading the tuned placement;
four deterministic replies matched the control. Rates divide prompt tokens by
elapsed time through a two-token reply. A single 4,274-token request took
130.515 s with the previous binary and placement and 112.730 s with the tuned
candidate; both returned the same answer. The longer comparison has one sample
per arm. [Individual results, first-token telemetry, and test summaries](qwen36-p2p-resident-kv-results.json)
record the hardware and limitations. A full 131,072-token prompt was not run.
