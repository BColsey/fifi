/*
 * Peer mailbox exchange entries of the CUDA adapter: the decode-time sum
 * that turns per-device partial rows into the replicated residual row.
 *
 * A mailbox is a device allocation of slots, one per source device and
 * sequence parity. A slot is a 128-byte header (unsigned sequence word at
 * byte 0, unsigned flags word at byte 4) followed by the payload. Sequence
 * 0 means empty; sequences start at 1. The layout arithmetic is owned by
 * the Rust layer (`kernels::cuda::peer_ring`); these entries receive slot
 * base addresses already computed from it.
 *
 * Protocol per exchange, all in GPU threads on the caller's stream:
 *   1. Every device publishes its partial row into every peer's slot for
 *      (source = itself, parity = sequence & 1): payload first, flags,
 *      a system fence, then the sequence word last.
 *   2. Every device waits until each peer slot in its own mailbox carries
 *      the sequence, fences, then sums the payloads with its own partial
 *      in f32, in a fixed order, so every device's result is identical.
 * Payloads travel as fp16 by default; a source value outside the fp16
 * range (or NaN) sets the slot overflow flag instead of saturating, and
 * the receiving sum reports it in the status word. The f32 variants carry
 * the values unconverted, for diagnosis.
 *
 * Ordering: the sequence word is stored with system-scope release after
 * payload and flags, and loaded with system-scope acquire before any
 * payload read (st.release.sys / ld.acquire.sys). That pair, not the
 * fences around it, is what orders the payload across devices.
 *
 * Slot lifetime: the slot for sequence S is reused at S+2. That is safe
 * only under the caller's stream contract — every publish and wait of one
 * device is queued in exchange order on ONE stream, and exchanges are
 * never overlapped. Then a publisher's S+2 publish is stream-ordered
 * after its own S+1 wait, which observed the consumer's S+1 publish,
 * which is stream-ordered after the consumer's S wait finished reading.
 * A caller that overlaps exchanges (prefill against decode, say) must
 * keep them on that same per-device stream or this guarantee is void.
 *
 * Status word bits written by the wait-sum entries (OR-ed, never cleared
 * by these entries): 1 = timed out waiting for a peer, 2 = a received
 * slot carried the overflow flag. On timeout the output row is unwritten
 * and still holds whatever it held before — a previous valid row. A
 * caller that reads the row without first checking the status word will
 * silently consume that stale row; checking is part of the contract, and
 * the step treats any nonzero status as a poisoned token.
 *
 * Like abi.h, a function launches nothing on failure, returns an int
 * status, and clears the thread's pending non-sticky CUDA error before it
 * returns. `count` is a number of f32 values (rows * width), at most
 * INFERENCE_CUDA_RING_MAX_VALUES. Peer slot addresses are device
 * addresses of other devices' allocations; peer access must already be
 * enabled. The native side can only repeat shape checks on them.
 */
#ifndef INFERENCE_CUDA_ABI_RING_H
#define INFERENCE_CUDA_ABI_RING_H

#ifdef __cplusplus
extern "C" {
#endif

/* Converts `count` f32 values at `src` to fp16 in the payload of the slot
 * at `dst_slot`, writes the overflow flag, fences, and stores `seq` into
 * the slot's sequence word last. One block; ordered behind earlier work
 * on `stream` only. */
int inference_cuda_ring_publish_f16(void *stream, const float *src,
                                    void *dst_slot, int count,
                                    unsigned seq);

/* The f32-payload form of the publish above. The overflow flag is 0. */
int inference_cuda_ring_publish_f32(void *stream, const float *src,
                                    void *dst_slot, int count,
                                    unsigned seq);

/* Waits until the `peer_count` (at most 3) slots at `peer_slot0..2` all
 * carry sequence `seq`, then writes `own + sum(fp16 payloads)` as f32 to
 * `out`. Slots past `peer_count` are ignored and may be NULL. Spins at
 * most `timeout_cycles` of the SM clock; on expiry ORs status bit 1 into
 * `*status` and writes nothing. A peer slot's overflow flag ORs bit 2. */
int inference_cuda_ring_wait_sum_f16(void *stream, float *out,
                                     const float *own,
                                     const void *peer_slot0,
                                     const void *peer_slot1,
                                     const void *peer_slot2,
                                     int peer_count, int count,
                                     unsigned seq,
                                     unsigned long long timeout_cycles,
                                     unsigned *status);

/* The f32-payload form of the wait-sum above. */
int inference_cuda_ring_wait_sum_f32(void *stream, float *out,
                                     const float *own,
                                     const void *peer_slot0,
                                     const void *peer_slot1,
                                     const void *peer_slot2,
                                     int peer_count, int count,
                                     unsigned seq,
                                     unsigned long long timeout_cycles,
                                     unsigned *status);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_CUDA_ABI_RING_H */
