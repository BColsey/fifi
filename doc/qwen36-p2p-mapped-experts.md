# Experimental host-backed experts for two-device Qwen3.6

The native GGUF two-device path can place some of either device's sliced
expert weights in page-locked host RAM. The GPU reads those weights over PCIe;
both GPUs still execute their shards and use peer access. This option is off by
default and has not been qualified for production use.

Set `Q36_P2P_HOST_EXPERT_BYTES` to the minimum number of expert bytes to place
in host RAM before starting `enginesrv`, for example `2147483648` for 2 GiB.
The loader selects complete expert slices, so the actual amount can exceed the
requested minimum by at most one slice. Only sliced quantized routed expert
stacks on the second device in `cuda_devices` are eligible under that setting.
Set `Q36_P2P_HOST_EXPERT_BYTES_0` independently to map slices on the first
device. Both settings default to zero. A request that cannot be met fails
before that device's weights are uploaded.

The loader checks available host RAM and leaves an additional 1 GiB in its
admission calculation. It also retains the usual per-device reservations and
the free-VRAM check after loading. Available host RAM can change during a
load, and pinned pages cannot be swapped. Direct host-memory reads can reduce
inference throughput; measure on the target hardware before relying on this
option. Remove the environment variable to restore the existing GPU-only
placement.

On systems where peers can advance at different rates, set
`Q36_P2P_SYNC_BROADCAST=1` as well. This experimental option waits for every
card to consume a broadcast before reusing the ring's two mailbox slots;
it prevents a faster publisher from overwriting a slower peer's message.
The additional synchronizations can reduce throughput.
