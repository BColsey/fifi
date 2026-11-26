# CUDA conventions the types do not enforce

Four rules govern sharing device memory across streams and lifetimes. Parts
are enforced in the types; the rest hold only by convention, and nothing in
a type, an assertion, or a test rejects a change that breaks them. This page
is the durable statement of each rule so a future change knows it is breaking
a contract, not reshaping an accident. Every rule names its code site and
what enforces it today.

Debug tools that help while working against these rules:
`FIFI_CUDA_POISON_ALLOC` and `FIFI_CUDA_LINEINFO` (`environment.md`), and
NVIDIA compute-sanitizer (memcheck, racecheck, synccheck) over a build made
with `FIFI_CUDA_LINEINFO`.

## Hidden synchronization sites

Every place the engine blocks on the device without a caller-visible
`synchronize` call, with why it exists and where the details live:

- `complete_pageable_copy` (`kernels/cuda/native/resident.rs`) blocks the
  host on `stream_synchronize` so an owned pageable buffer is handed back
  (readback) or released (upload) only after its copy completes; the wait
  is charged to the page-io counters when they are armed.
- `download` (`kernels/cuda/context.rs`) calls the synchronous
  `cudaMemcpy`: it provides the stream ordering rule 4 describes, and it
  also blocks the host until the copy completes — every readback pays this
  hidden sync.
- `DeviceBuffer` upload/download (`kernels/cuda/context/buffer_io.rs`) — a
  pageable `cudaMemcpyAsync` costs a full staged copy the caller cannot
  see; the impl block's doc states the consumption-at-enqueue rule.
- The allocation cache's return path (`kernels/cuda/context/alloc_cache.rs`)
  synchronizes the device before a slot re-enters the pool: the lazy return
  policy defers that sync until the slot is reused or drained, but it is a
  full device sync when it fires.
- The old projector's default-stream sync no longer exists: `engine/src/project/`
  contains no synchronize call; the current projector build path has no
  hidden sync.

## 1. Lazy return runs on one stream (alloc_cache.rs)

`AllocCache` may defer the device sync when a block goes idle if lazy return
is armed (`set_lazy_alloc_return`; no start arms it now) and the lease has at most one
stream (`on_return`, `kernels/cuda/context/alloc_cache.rs`). Reissuing that
block is safe only because every other enqueued user of its bytes is on the
same stream, where ordering is program order.

What holds it today: `Lease::streams` counts `CudaContext::stream()` calls and
never decrements (`guard.rs`, `stream.rs`), and creating a second stream first
calls `cache.settle()`. Every host-synchronous write into a cached block
settles first (`host_write`, `AllocCache::copy`).

What would break it: a stream created outside `CudaContext::stream()` (the
counter would not see it — today `cudaStreamCreate*` appears once, in
`stream.cu`), a consumer on the legacy NULL stream (the counter ignores it), or
a raw `native::upload_at` call that bypasses `host_write`.

Possible enforcement: assertions on the stream counter at return and reissue,
or an event recorded at return and queried at reissue (`cudaEventQuery`, a
not-ready answer treated as a cache miss).

## 2. A capture pins the buffers its body uses (context/graph.rs)

A CUDA graph bakes in device addresses. If a captured body reads or writes a
cached block that is later returned and reissued to someone else, the graph's
next launch touches the new tenant's bytes. `CapturePlan::pins` is the list of
buffers a capture claims to use — and the launch window does not compare it
with the buffers the body actually touches (`kernels/cuda/context/graph.rs`).

What holds it today: the capture window refuses allocation, free, sync and
graph launch, and defers drops to `free_after_capture`; the pin list is filled
by the capture author by convention.

What would break it: a capture body that touches a cached buffer without
listing it in `pins`.

Possible enforcement: in debug builds, record the buffers `launch_on` sees
during capture and compare with `pins` at `capture_end`.

## 3. The refill source outlives the refill (family/exl3/bank.rs)

The expert-bank refill copies from the page-locked store into device memory
(`family/exl3/bank.rs`). The copy is asynchronous, so its source must stay
valid and unchanged while it runs. Most of this rule is now enforced in the
types:

- `pin()` seals the store, and `put()` on a sealed store refuses — and the
  store's own finalize calls `pin()` on every part (bank.rs), so no write
  path exists after the load that could overwrite an in-flight copy's
  source.
- `Drop` calls `device_synchronize()` before `host_unregister`, so the
  unregister cannot race a refill DMA still reading the page-locked range,
  even on a drop path the streams have not drained.

What is still by convention: nothing material. Every call site keeps the
store process-long-lived; a shorter-lived store would remain correct through
`Drop`'s sync, at the cost of one device sync on its drop path.

## 4. The split refill's readback ordering rests on quiescence (family/exl3/bank.rs)

The split refill copies misses on a second stream. Two orderings are involved:

- Slot reads after the copy are event-ordered: the copy records an event and
  the launches that read the slot wait on it. This part is enforced by the
  code.
- The reverse direction — the refill's writes must not pass the readback whose
  picks produced them — rests on "both streams were quiescent at the
  readback": a host-side argument, not a device dependency.

What holds it today: the readback path waits on the launch stream before the
refill is planned.

What would break it: removing that sync, or enqueueing the refill before the
readback completes.

Possible enforcement: an event on the readback stream that the copy stream
waits on, removing the quiescence argument. The path is opt-in
(`Q36_CUDA_BANK_OVERLAP=1`), so no served default depends on it.
