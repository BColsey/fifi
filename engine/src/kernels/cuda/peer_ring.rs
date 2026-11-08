//! The peer mailbox ring: one mailbox allocation per device, and the
//! exchange that turns per-device partial rows into the replicated row.
//!
//! The protocol and slot layout are stated in `native/abi_ring.h`; this
//! module owns the layout arithmetic (pure, in [`MailboxLayout`]) and the
//! mailbox allocations. An exchange publishes every device's partial into
//! every peer's slot, then each device waits and sums locally, so a
//! finished exchange leaves the identical full row on every device. The
//! sequence advances once per exchange and the slot parity follows it, so
//! a slot is only reused two exchanges later and never needs clearing.
//!
//! Slot reuse requires the stream contract stated in `native/abi_ring.h`:
//! all ring work for one device is queued in exchange order on one stream.
//! For sums and gathers, each device's own wait supplies a two-exchange
//! dependency chain. Consecutive broadcasts have only one publisher, so
//! they also need a consumer barrier before a slot can be reused. The
//! Qwen card stack offers an opt-in host synchronization for this case.
//!
//! [`PeerRing::reset`] re-zeroes every mailbox and restarts the sequence;
//! it is the session boundary. It also serves the (practically unreachable)
//! sequence exhaustion, which `exchange_sum` refuses rather than wrapping
//! into a slot a peer might still read. Construction does not enable peer
//! access: `peer::enable_peer_access` runs first, at engine start. Two
//! contexts on one physical device are a legal ring (the addresses are
//! ordinary allocations then), which is how the protocol is tested on a
//! single-card box.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "staged for the multi-device decode step, its first production caller"
    )
)]

use std::cell::Cell;

use super::context::{CudaContext, DeviceBuffer, DeviceStream};
use super::native;
use crate::EngError;
use crate::kernels::cuda::DeviceRange;

/// Bytes of slot header before the payload. Matches `kHeaderBytes` in
/// `native/peer_ring.cu`.
pub(crate) const HEADER_BYTES: usize = 128;
/// Payloads are allocated at the f32 stride so the fp32 diagnostic mode
/// needs no second mailbox.
const VALUE_BYTES: usize = 4;
/// The spin budget of one wait, in SM clock cycles: about a second on a
/// 1.7 GHz part. A peer that has not published within it has failed, and
/// the status word says so instead of hanging the device.
pub(crate) const TIMEOUT_CYCLES: u64 = 1_700_000_000;

/// The status bit a wait sets when a peer never published.
pub(crate) const STATUS_TIMEOUT: u32 = native::peer_ring::STATUS_TIMEOUT;
/// The status bit a wait sets when a received fp16 payload overflowed.
pub(crate) const STATUS_OVERFLOW: u32 = native::peer_ring::STATUS_OVERFLOW;

fn bad(message: String) -> EngError {
    EngError::BadRequest(format!("peer ring: {message}"))
}

/// Where every slot of one device's mailbox lies. Pure arithmetic over the
/// device count and the largest payload (in f32 values) an exchange carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MailboxLayout {
    cards: usize,
    max_values: usize,
}

impl MailboxLayout {
    pub(crate) fn new(cards: usize, max_values: usize) -> Result<Self, EngError> {
        if !(1..=super::peer::MAX_DEVICES).contains(&cards) {
            return Err(bad(format!(
                "{cards} devices is outside 1..={}",
                super::peer::MAX_DEVICES
            )));
        }
        if max_values == 0 || max_values > native::peer_ring::MAX_VALUES {
            return Err(bad(format!(
                "a payload of {max_values} values is outside 1..={}",
                native::peer_ring::MAX_VALUES
            )));
        }
        Ok(Self { cards, max_values })
    }

    /// Bytes of one slot: header plus the payload at the f32 stride.
    pub(crate) fn slot_bytes(&self) -> usize {
        HEADER_BYTES + self.max_values * VALUE_BYTES
    }

    /// Bytes of one device's whole mailbox: a slot per source and parity.
    pub(crate) fn total_bytes(&self) -> usize {
        self.slot_bytes() * self.cards * 2
    }

    /// Byte offset of the slot that `source` publishes into for `seq`.
    pub(crate) fn slot_offset(&self, source: usize, seq: u32) -> Result<usize, EngError> {
        if source >= self.cards {
            return Err(bad(format!(
                "source device {source} is outside the {}-device ring",
                self.cards
            )));
        }
        Ok((source * 2 + (seq as usize & 1)) * self.slot_bytes())
    }
}

/// Which payload representation an exchange uses. `F32` is the diagnostic
/// mode that keeps ring numerics attributable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RingMode {
    F16,
    F32,
}

impl RingMode {
    fn f32_payload(self) -> bool {
        matches!(self, RingMode::F32)
    }
}

/// The mailboxes and sequence state of one device set. Not `Send`: the
/// mailbox buffers share their contexts' leases and thread.
pub(crate) struct PeerRing {
    layout: MailboxLayout,
    mailboxes: Vec<DeviceBuffer>,
    /// Device address of each mailbox allocation, in device order, recorded
    /// at construction. The allocations live in `mailboxes` for the ring's
    /// whole life, so these stay valid.
    bases: Vec<u64>,
    next_seq: Cell<u32>,
}

impl PeerRing {
    /// Allocates and zeroes one mailbox per context. `contexts[i]` owns the
    /// mailbox that device `i` reads its peer slots from. Zeroing writes go
    /// through a short-lived stream per context and return only when landed.
    pub(crate) fn new(contexts: &[CudaContext], max_values: usize) -> Result<Self, EngError> {
        let layout = MailboxLayout::new(contexts.len(), max_values)?;
        let zeros = vec![0u8; layout.total_bytes()];
        let mut mailboxes = Vec::with_capacity(contexts.len());
        let mut bases = Vec::with_capacity(contexts.len());
        for context in contexts {
            let buffer = context.allocate(layout.total_bytes())?;
            let stream = context.stream()?;
            buffer.write_at(&stream, 0, &zeros)?;
            bases.push(context.owned(&buffer)?.address());
            mailboxes.push(buffer);
        }
        Ok(Self {
            layout,
            mailboxes,
            bases,
            next_seq: Cell::new(1),
        })
    }

    pub(crate) fn cards(&self) -> usize {
        self.layout.cards
    }

    /// Re-zeroes every mailbox and restarts the sequence. The caller queues
    /// this on `streams` it knows are ordered after earlier exchanges.
    pub(crate) fn reset(
        &self,
        contexts: &[CudaContext],
        streams: &[DeviceStream],
    ) -> Result<(), EngError> {
        self.check_members(contexts, streams.len())?;
        let zeros = vec![0u8; self.layout.total_bytes()];
        for (context, (stream, mailbox)) in contexts.iter().zip(streams.iter().zip(&self.mailboxes))
        {
            context.owned(mailbox)?;
            mailbox.write_at(stream, 0, &zeros)?;
        }
        self.next_seq.set(1);
        Ok(())
    }

    fn check_members(&self, contexts: &[CudaContext], streams: usize) -> Result<(), EngError> {
        if contexts.len() != self.layout.cards || streams != self.layout.cards {
            return Err(bad(format!(
                "the ring spans {} devices, got {} contexts and {streams} streams",
                self.layout.cards,
                contexts.len()
            )));
        }
        Ok(())
    }

    /// The next sequence of the shared stream and its successor. Every ring
    /// operation (sum, broadcast, gather) takes exactly one, so slot parity
    /// alternates across operation kinds too.
    fn take_seq(&self) -> Result<(u32, u32), EngError> {
        let seq = self.next_seq.get();
        let Some(next) = seq.checked_add(1) else {
            return Err(bad("the ring sequence is exhausted; reset the ring".into()));
        };
        Ok((seq, next))
    }

    fn slot_address(&self, owner: usize, source: usize, seq: u32) -> Result<u64, EngError> {
        let offset = self.layout.slot_offset(source, seq)?;
        self.bases[owner]
            .checked_add(u64::try_from(offset).map_err(|_| bad("slot offset overflow".into()))?)
            .ok_or_else(|| bad("slot address overflow".into()))
    }

    /// Queues one exchange: every device publishes `partials[i]` to its
    /// peers, then waits and writes the full sum to `outs[i]`. All slices are
    /// per-device, in ring order; every buffer must hold `count` f32 values
    /// (and each `statuses[i]` one u32, zeroed by the caller, which the wait
    /// ORs its outcome into). The caller MUST read every status word before
    /// using any `outs[i]`: on timeout the wait writes nothing, so the row
    /// still holds a previous exchange's valid result, and an overflow bit
    /// means a peer partial saturated in fp16. Any nonzero status poisons
    /// the token; there is no valid row. On one device the exchange
    /// degenerates to the wait-sum over zero peers, an in-order copy of the
    /// partial row.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exchange_sum(
        &self,
        contexts: &[CudaContext],
        streams: &[DeviceStream],
        partials: &[DeviceRange<'_>],
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        count: usize,
        mode: RingMode,
    ) -> Result<(), EngError> {
        self.check_members(contexts, streams.len())?;
        if partials.len() != self.layout.cards
            || outs.len() != self.layout.cards
            || statuses.len() != self.layout.cards
        {
            return Err(bad(
                "partials, outs and statuses are one range per device".into()
            ));
        }
        if count == 0 || count > self.layout.max_values {
            return Err(bad(format!(
                "an exchange of {count} values exceeds the mailbox payload of {}",
                self.layout.max_values
            )));
        }
        let (seq, next) = self.take_seq()?;
        let bytes = count * VALUE_BYTES;
        if (partials
            .iter()
            .chain(outs)
            .any(|range| range.bytes() < bytes))
            || statuses.iter().any(|range| range.bytes() < 4)
        {
            return Err(bad("an exchange range is shorter than its payload".into()));
        }
        for (source, context) in contexts.iter().enumerate() {
            let src = context.raw_range(&partials[source])?;
            for owner in 0..self.layout.cards {
                if owner == source {
                    continue;
                }
                let dst = self.slot_address(owner, source, seq)?;
                context.launch_on(&streams[source], |stream| {
                    native::peer_ring::publish(stream, src, dst, count, seq, mode.f32_payload())
                })?;
            }
        }
        for (owner, context) in contexts.iter().enumerate() {
            let out = context.raw_range(&outs[owner])?;
            let own = context.raw_range(&partials[owner])?;
            let word = context.raw_range(&statuses[owner])?;
            let mut slots = Vec::with_capacity(self.layout.cards - 1);
            for source in 0..self.layout.cards {
                if source != owner {
                    slots.push(self.slot_address(owner, source, seq)?);
                }
            }
            context.launch_on(&streams[owner], |stream| {
                native::peer_ring::wait_sum(
                    stream,
                    out,
                    own,
                    &slots,
                    count,
                    seq,
                    mode.f32_payload(),
                    TIMEOUT_CYCLES,
                    word,
                )
            })?;
        }
        self.next_seq.set(next);
        Ok(())
    }

    /// Queues one broadcast: device `source_card` publishes `count` f32
    /// values of `src` into every device's mailbox (its own included), and
    /// every device copies them to `outs[i]`. The copy is bit-exact, so
    /// integer payloads (router picks) ride bit-cast through f32. Status
    /// handling is the exchange's: read every `statuses[i]` before using
    /// any output; a timeout leaves the previous contents in place.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn broadcast(
        &self,
        contexts: &[CudaContext],
        streams: &[DeviceStream],
        source_card: usize,
        src: DeviceRange<'_>,
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        count: usize,
    ) -> Result<(), EngError> {
        self.check_members(contexts, streams.len())?;
        if source_card >= self.layout.cards {
            return Err(bad(format!(
                "broadcast source {source_card} is outside the {}-device ring",
                self.layout.cards
            )));
        }
        if outs.len() != self.layout.cards || statuses.len() != self.layout.cards {
            return Err(bad("outs and statuses are one range per device".into()));
        }
        if count == 0 || count > self.layout.max_values {
            return Err(bad(format!(
                "a broadcast of {count} values exceeds the mailbox payload of {}",
                self.layout.max_values
            )));
        }
        let (seq, next) = self.take_seq()?;
        let bytes = count * VALUE_BYTES;
        if src.bytes() < bytes
            || outs.iter().any(|range| range.bytes() < bytes)
            || statuses.iter().any(|range| range.bytes() < 4)
        {
            return Err(bad("a broadcast range is shorter than its payload".into()));
        }
        let publisher = &contexts[source_card];
        let src_range = publisher.raw_range(&src)?;
        for owner in 0..self.layout.cards {
            let dst = self.slot_address(owner, source_card, seq)?;
            publisher.launch_on(&streams[source_card], |stream| {
                native::peer_ring::publish(stream, src_range, dst, count, seq, true)
            })?;
        }
        for (owner, context) in contexts.iter().enumerate() {
            let out = context.raw_range(&outs[owner])?;
            let word = context.raw_range(&statuses[owner])?;
            let slot = self.slot_address(owner, source_card, seq)?;
            context.launch_on(&streams[owner], |stream| {
                native::peer_ring::wait_copy(stream, out, slot, count, seq, TIMEOUT_CYCLES, word)
            })?;
        }
        self.next_seq.set(next);
        Ok(())
    }

    /// Queues one all-gather of a row dealt across devices: device `i`
    /// computed the values `ranges[i]` of a `total` row (its new K or V
    /// slice, say) in `parts[i]`; afterwards every `outs[i]` holds the full
    /// row, each slice at its range offset. Only the new row moves —
    /// anything accumulated from earlier rows already lives in every
    /// replica. Ranges must be in ring order and exactly tile `total`.
    /// Status handling is the exchange's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn all_gather_row(
        &self,
        contexts: &[CudaContext],
        streams: &[DeviceStream],
        parts: &[DeviceRange<'_>],
        outs: &[DeviceRange<'_>],
        statuses: &[DeviceRange<'_>],
        ranges: &[std::ops::Range<usize>],
        total: usize,
    ) -> Result<(), EngError> {
        self.check_members(contexts, streams.len())?;
        if parts.len() != self.layout.cards
            || outs.len() != self.layout.cards
            || statuses.len() != self.layout.cards
            || ranges.len() != self.layout.cards
        {
            return Err(bad(
                "parts, outs, statuses and ranges are one per device".into()
            ));
        }
        if statuses.iter().any(|range| range.bytes() < 4)
            || outs.iter().any(|range| range.bytes() < total * VALUE_BYTES)
        {
            return Err(bad("a gather range is shorter than its payload".into()));
        }
        let mut next_start = 0usize;
        for range in ranges {
            if range.start != next_start || range.is_empty() || range.len() > self.layout.max_values
            {
                return Err(bad(format!(
                    "gather ranges {ranges:?} do not tile 0..{total} within the mailbox payload"
                )));
            }
            next_start = range.end;
        }
        if next_start != total {
            return Err(bad(format!(
                "gather ranges end at {next_start}, the row is {total} values"
            )));
        }
        let (seq, next) = self.take_seq()?;
        for (source, context) in contexts.iter().enumerate() {
            let count = ranges[source].len();
            if parts[source].bytes() < count * VALUE_BYTES {
                return Err(bad("a gather part is shorter than its slice".into()));
            }
            let src = context.raw_range(&parts[source])?;
            for owner in 0..self.layout.cards {
                let dst = self.slot_address(owner, source, seq)?;
                context.launch_on(&streams[source], |stream| {
                    native::peer_ring::publish(stream, src, dst, count, seq, true)
                })?;
            }
        }
        for (owner, context) in contexts.iter().enumerate() {
            let word = context.raw_range(&statuses[owner])?;
            for (source, range) in ranges.iter().enumerate() {
                let (offset, len) = (range.start * VALUE_BYTES, range.len() * VALUE_BYTES);
                let out = context.raw_range(
                    &outs[owner]
                        .buffer()
                        .range(outs[owner].offset() + offset, len)?,
                )?;
                let slot = self.slot_address(owner, source, seq)?;
                let count = range.len();
                context.launch_on(&streams[owner], |stream| {
                    native::peer_ring::wait_copy(
                        stream,
                        out,
                        slot,
                        count,
                        seq,
                        TIMEOUT_CYCLES,
                        word,
                    )
                })?;
            }
        }
        self.next_seq.set(next);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_tiles_slots_by_source_then_parity() {
        let layout = MailboxLayout::new(3, 2048).expect("layout");
        assert_eq!(layout.slot_bytes(), 128 + 2048 * 4);
        assert_eq!(layout.total_bytes(), layout.slot_bytes() * 6);
        assert_eq!(layout.slot_offset(0, 1).expect("slot"), layout.slot_bytes());
        assert_eq!(layout.slot_offset(0, 2).expect("slot"), 0);
        assert_eq!(
            layout.slot_offset(2, 3).expect("slot"),
            layout.slot_bytes() * 5
        );
        // Consecutive sequences alternate slots for every source.
        for source in 0..3 {
            assert_ne!(
                layout.slot_offset(source, 7).expect("slot"),
                layout.slot_offset(source, 8).expect("slot")
            );
            assert_eq!(
                layout.slot_offset(source, 7).expect("slot"),
                layout.slot_offset(source, 9).expect("slot")
            );
        }
    }

    #[test]
    fn the_layout_refuses_bad_shapes() {
        assert!(MailboxLayout::new(0, 8).is_err());
        assert!(MailboxLayout::new(5, 8).is_err());
        assert!(MailboxLayout::new(2, 0).is_err());
        assert!(MailboxLayout::new(2, native::peer_ring::MAX_VALUES + 1).is_err());
        let layout = MailboxLayout::new(2, 8).expect("layout");
        assert!(layout.slot_offset(2, 1).is_err());
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use crate::family::project::cuda::served_admission;
    use crate::kernels::cuda::testkit::device_present;

    fn upload_f32(
        context: &CudaContext,
        stream: &DeviceStream,
        values: &[f32],
    ) -> Result<DeviceBuffer, EngError> {
        let buffer = context.allocate(values.len() * 4)?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        buffer.write_at(stream, 0, &bytes)?;
        Ok(buffer)
    }

    fn download_f32(
        buffer: &DeviceBuffer,
        stream: &DeviceStream,
        count: usize,
    ) -> Result<Vec<f32>, EngError> {
        let mut bytes = vec![0u8; count * 4];
        buffer.read_at(stream, 0, &mut bytes)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }

    fn status_of(buffer: &DeviceBuffer, stream: &DeviceStream) -> Result<u32, EngError> {
        Ok(download_f32(buffer, stream, 1)?[0].to_bits())
    }

    /// The whole buffer as one range: what the step-level callers pass for
    /// the ring operations that take arena slots (a `DeviceRange`).
    fn whole(buffer: &DeviceBuffer) -> DeviceRange<'_> {
        buffer.range(0, buffer.bytes()).expect("whole range")
    }

    /// Two contexts on the one physical device of this box: the mailbox
    /// addresses are ordinary allocations, so the whole protocol runs and
    /// only the PCIe hop is absent. A real peer pair needs a second card.
    #[test]
    fn two_contexts_exchange_a_sum_and_repeat_it() {
        if !device_present("two_contexts_exchange_a_sum_and_repeat_it") {
            return;
        }
        let admission = served_admission();
        let contexts = [
            CudaContext::with_admission_on(0, &admission).expect("context 0"),
            CudaContext::with_admission_on(0, &admission).expect("context 1"),
        ];
        let streams = [
            contexts[0].stream().expect("stream 0"),
            contexts[1].stream().expect("stream 1"),
        ];
        let ring = PeerRing::new(&contexts, 2048).expect("ring");
        let a: Vec<f32> = (0..2048).map(|i| (i as f32) * 0.25 - 100.0).collect();
        let b: Vec<f32> = (0..2048).map(|i| 50.0 - (i as f32) * 0.125).collect();
        let partials = [
            upload_f32(&contexts[0], &streams[0], &a).expect("partial 0"),
            upload_f32(&contexts[1], &streams[1], &b).expect("partial 1"),
        ];
        let outs = [
            contexts[0].allocate(2048 * 4).expect("out 0"),
            contexts[1].allocate(2048 * 4).expect("out 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        assert_eq!(ring.cards(), 2);
        for round in 0..3 {
            ring.exchange_sum(
                &contexts,
                &streams,
                &[whole(&partials[0]), whole(&partials[1])],
                &[whole(&outs[0]), whole(&outs[1])],
                &[whole(&statuses[0]), whole(&statuses[1])],
                2048,
                RingMode::F32,
            )
            .expect("exchange");
            for (index, stream) in streams.iter().enumerate() {
                stream.synchronize().expect("sync");
                assert_eq!(
                    status_of(&statuses[index], stream).expect("status"),
                    0,
                    "round {round} device {index}"
                );
                let got = download_f32(&outs[index], stream, 2048).expect("download");
                for (i, (&g, (&x, &y))) in got.iter().zip(a.iter().zip(&b)).enumerate() {
                    assert_eq!(g, x + y, "round {round} device {index} value {i}");
                }
            }
        }
    }

    #[test]
    fn a_reset_ring_exchanges_again_from_sequence_one() {
        if !device_present("a_reset_ring_exchanges_again_from_sequence_one") {
            return;
        }
        let admission = served_admission();
        let contexts = [
            CudaContext::with_admission_on(0, &admission).expect("context 0"),
            CudaContext::with_admission_on(0, &admission).expect("context 1"),
        ];
        let streams = [
            contexts[0].stream().expect("stream 0"),
            contexts[1].stream().expect("stream 1"),
        ];
        let ring = PeerRing::new(&contexts, 4).expect("ring");
        let partials = [
            upload_f32(&contexts[0], &streams[0], &[1.0, 2.0, 3.0, 4.0]).expect("partial 0"),
            upload_f32(&contexts[1], &streams[1], &[10.0, 20.0, 30.0, 40.0]).expect("partial 1"),
        ];
        let outs = [
            contexts[0].allocate(16).expect("out 0"),
            contexts[1].allocate(16).expect("out 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        let exchange = |ring: &PeerRing| {
            ring.exchange_sum(
                &contexts,
                &streams,
                &[whole(&partials[0]), whole(&partials[1])],
                &[whole(&outs[0]), whole(&outs[1])],
                &[whole(&statuses[0]), whole(&statuses[1])],
                4,
                RingMode::F32,
            )
            .expect("exchange");
            for stream in &streams {
                stream.synchronize().expect("sync");
            }
            download_f32(&outs[0], &streams[0], 4).expect("download")
        };
        assert_eq!(exchange(&ring), [11.0, 22.0, 33.0, 44.0]);
        ring.reset(&contexts, &streams).expect("reset");
        assert_eq!(exchange(&ring), [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn the_fp16_exchange_sums_within_half_precision_and_flags_overflow() {
        if !device_present("the_fp16_exchange_sums_within_half_precision_and_flags_overflow") {
            return;
        }
        let admission = served_admission();
        let contexts = [
            CudaContext::with_admission_on(0, &admission).expect("context 0"),
            CudaContext::with_admission_on(0, &admission).expect("context 1"),
        ];
        let streams = [
            contexts[0].stream().expect("stream 0"),
            contexts[1].stream().expect("stream 1"),
        ];
        let ring = PeerRing::new(&contexts, 16).expect("ring");
        // Device 1 publishes 70000.0, outside fp16: device 0 must see the
        // overflow status bit; device 1 sums only device 0's in-range row.
        let a = [1.5f32; 16];
        let mut b = [2.25f32; 16];
        b[3] = 70_000.0;
        let partials = [
            upload_f32(&contexts[0], &streams[0], &a).expect("partial 0"),
            upload_f32(&contexts[1], &streams[1], &b).expect("partial 1"),
        ];
        let outs = [
            contexts[0].allocate(16 * 4).expect("out 0"),
            contexts[1].allocate(16 * 4).expect("out 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        ring.exchange_sum(
            &contexts,
            &streams,
            &[whole(&partials[0]), whole(&partials[1])],
            &[whole(&outs[0]), whole(&outs[1])],
            &[whole(&statuses[0]), whole(&statuses[1])],
            16,
            RingMode::F16,
        )
        .expect("exchange");
        for stream in &streams {
            stream.synchronize().expect("sync");
        }
        assert_eq!(
            status_of(&statuses[0], &streams[0]).expect("status 0"),
            native::peer_ring::STATUS_OVERFLOW
        );
        assert_eq!(status_of(&statuses[1], &streams[1]).expect("status 1"), 0);
        let got = download_f32(&outs[1], &streams[1], 16).expect("download");
        assert_eq!(got[0], 3.75);
        assert_eq!(got[3], 70_000.0 + 1.5);
    }

    fn two_context_ring(max_values: usize) -> ([CudaContext; 2], [DeviceStream; 2], PeerRing) {
        let admission = served_admission();
        let contexts = [
            CudaContext::with_admission_on(0, &admission).expect("context 0"),
            CudaContext::with_admission_on(0, &admission).expect("context 1"),
        ];
        let streams = [
            contexts[0].stream().expect("stream 0"),
            contexts[1].stream().expect("stream 1"),
        ];
        let ring = PeerRing::new(&contexts, max_values).expect("ring");
        (contexts, streams, ring)
    }

    #[test]
    fn a_broadcast_lands_bit_exact_on_every_device() {
        if !device_present("a_broadcast_lands_bit_exact_on_every_device") {
            return;
        }
        let (contexts, streams, ring) = two_context_ring(64);
        // Router-shaped payload: top-8 (id, weight) pairs, ids bit-cast.
        let mut picks = [0f32; 16];
        for i in 0..8 {
            picks[2 * i] = f32::from_bits(100 + i as u32 * 37);
            picks[2 * i + 1] = 0.125 * (i as f32 + 1.0);
        }
        let src = upload_f32(&contexts[1], &streams[1], &picks).expect("src");
        let outs = [
            upload_f32(&contexts[0], &streams[0], &[-9.0; 16]).expect("out 0"),
            upload_f32(&contexts[1], &streams[1], &[-9.0; 16]).expect("out 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        // Interleave with a sum exchange: both operations share one sequence.
        let partials = [
            upload_f32(&contexts[0], &streams[0], &[1.0; 16]).expect("partial 0"),
            upload_f32(&contexts[1], &streams[1], &[2.0; 16]).expect("partial 1"),
        ];
        let sums = [
            contexts[0].allocate(64).expect("sum 0"),
            contexts[1].allocate(64).expect("sum 1"),
        ];
        ring.exchange_sum(
            &contexts,
            &streams,
            &[whole(&partials[0]), whole(&partials[1])],
            &[whole(&sums[0]), whole(&sums[1])],
            &[whole(&statuses[0]), whole(&statuses[1])],
            16,
            RingMode::F32,
        )
        .expect("exchange");
        ring.broadcast(
            &contexts,
            &streams,
            1,
            whole(&src),
            &[whole(&outs[0]), whole(&outs[1])],
            &[whole(&statuses[0]), whole(&statuses[1])],
            16,
        )
        .expect("broadcast");
        ring.exchange_sum(
            &contexts,
            &streams,
            &[whole(&partials[0]), whole(&partials[1])],
            &[whole(&sums[0]), whole(&sums[1])],
            &[whole(&statuses[0]), whole(&statuses[1])],
            16,
            RingMode::F32,
        )
        .expect("exchange after broadcast");
        for (index, stream) in streams.iter().enumerate() {
            stream.synchronize().expect("sync");
            assert_eq!(status_of(&statuses[index], stream).expect("status"), 0);
            let got = download_f32(&outs[index], stream, 16).expect("download picks");
            for (g, w) in got.iter().zip(&picks) {
                assert_eq!(g.to_bits(), w.to_bits(), "device {index}");
            }
            assert_eq!(
                download_f32(&sums[index], stream, 16).expect("download sum"),
                [3.0; 16],
                "device {index}"
            );
        }
    }

    #[test]
    fn an_all_gather_row_replicates_the_full_row_on_every_device() {
        if !device_present("an_all_gather_row_replicates_the_full_row_on_every_device") {
            return;
        }
        let (contexts, streams, ring) = two_context_ring(512);
        // Uneven KV-row slices: 192 values from device 0, 320 from device 1.
        let full: Vec<f32> = (0..512).map(|i| i as f32 * -0.5).collect();
        let parts = [
            upload_f32(&contexts[0], &streams[0], &full[0..192]).expect("part 0"),
            upload_f32(&contexts[1], &streams[1], &full[192..512]).expect("part 1"),
        ];
        let outs = [
            upload_f32(&contexts[0], &streams[0], &[777.0; 512]).expect("out 0"),
            upload_f32(&contexts[1], &streams[1], &[777.0; 512]).expect("out 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        ring.all_gather_row(
            &contexts,
            &streams,
            &[whole(&parts[0]), whole(&parts[1])],
            &[whole(&outs[0]), whole(&outs[1])],
            &[whole(&statuses[0]), whole(&statuses[1])],
            &[0..192, 192..512],
            512,
        )
        .expect("gather");
        for (index, stream) in streams.iter().enumerate() {
            stream.synchronize().expect("sync");
            assert_eq!(status_of(&statuses[index], stream).expect("status"), 0);
            assert_eq!(
                download_f32(&outs[index], stream, 512).expect("download"),
                full,
                "device {index}"
            );
        }
        // Ranges that do not tile refuse before any launch.
        assert!(
            ring.all_gather_row(
                &contexts,
                &streams,
                &[whole(&parts[0]), whole(&parts[1])],
                &[whole(&outs[0]), whole(&outs[1])],
                &[whole(&statuses[0]), whole(&statuses[1])],
                &[0..192, 200..512],
                512,
            )
            .is_err()
        );
    }

    #[test]
    fn a_wait_with_no_publisher_times_out_in_the_status_word() {
        if !device_present("a_wait_with_no_publisher_times_out_in_the_status_word") {
            return;
        }
        let admission = served_admission();
        let context = CudaContext::with_admission_on(0, &admission).expect("context");
        let stream = context.stream().expect("stream");
        let own = upload_f32(&context, &stream, &[1.0, 2.0, 3.0, 4.0]).expect("own");
        let out = upload_f32(&context, &stream, &[-1.0; 4]).expect("out");
        let status = upload_f32(&context, &stream, &[0.0]).expect("status");
        let mailbox = context.allocate(HEADER_BYTES + 16).expect("mailbox");
        let zeros = vec![0u8; HEADER_BYTES + 16];
        mailbox.write_at(&stream, 0, &zeros).expect("zero mailbox");
        let slot = context.owned(&mailbox).expect("owned").address();
        let out_range = context
            .raw_range(&out.range(0, 16).expect("out range"))
            .expect("raw out");
        let own_range = context
            .raw_range(&own.range(0, 16).expect("own range"))
            .expect("raw own");
        let word_range = context
            .raw_range(&status.range(0, 4).expect("word range"))
            .expect("raw word");
        context
            .launch_on(&stream, |stream_raw| {
                native::peer_ring::wait_sum(
                    stream_raw,
                    out_range,
                    own_range,
                    &[slot],
                    4,
                    1,
                    true,
                    1_000_000,
                    word_range,
                )
            })
            .expect("launch wait");
        stream.synchronize().expect("sync");
        assert_eq!(
            status_of(&status, &stream).expect("status"),
            native::peer_ring::STATUS_TIMEOUT
        );
        assert_eq!(download_f32(&out, &stream, 4).expect("out"), [-1.0; 4]);
    }

    /// The coordinator invariant, pinned: ONE stream per card, all ring work
    /// in exchange order, and NO host synchronization between exchanges.
    /// Each round's PARTIAL is the previous round's `wait_sum` OUTPUT on the
    /// same stream, so a publish that could overtake its stream's wait, or a
    /// wait that could sum a slot before the peer's publish landed, shows up
    /// as a wrong value: the peer's round-R publish reads a buffer the peer's
    /// round-(R-1) wait wrote on that same stream. Sixteen rounds queue
    /// back-to-back on both cards; verification happens once at the end.
    /// Round R's expected row is 2^(R-1) * (a + b): exact in f32, the scaling
    /// is a power of two.
    ///
    /// Receipt limit: both contexts sit on the ONE physical device of this
    /// box, so this pins local stream sequencing and the mailbox protocol,
    /// not the PCIe hop or peer visibility of a real card pair.
    #[test]
    fn chained_exchanges_keep_producer_and_consumer_order_without_host_syncs() {
        if !device_present("chained_exchanges_keep_producer_and_consumer_order") {
            return;
        }
        let (contexts, streams, ring) = two_context_ring(64);
        let n = 64usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 8.0).collect();
        let b: Vec<f32> = (0..n).map(|i| 16.0 - i as f32 * 0.25).collect();
        // Round 0's partials are preloaded; every later round's partial is a
        // previous round's out, so no host write joins the queue phase.
        let partials = [
            upload_f32(&contexts[0], &streams[0], &a).expect("partial 0"),
            upload_f32(&contexts[1], &streams[1], &b).expect("partial 1"),
        ];
        let statuses = [
            upload_f32(&contexts[0], &streams[0], &[0.0]).expect("status 0"),
            upload_f32(&contexts[1], &streams[1], &[0.0]).expect("status 1"),
        ];
        let rounds = 16usize;
        let outs: Vec<[DeviceBuffer; 2]> = (0..rounds)
            .map(|_| {
                [
                    contexts[0].allocate(n * 4).expect("out 0"),
                    contexts[1].allocate(n * 4).expect("out 1"),
                ]
            })
            .collect();
        let exchange = |partial: &[DeviceRange<'_>; 2], out: &[DeviceRange<'_>; 2]| {
            ring.exchange_sum(
                &contexts,
                &streams,
                partial,
                out,
                &[whole(&statuses[0]), whole(&statuses[1])],
                n,
                RingMode::F32,
            )
        };
        exchange(
            &[whole(&partials[0]), whole(&partials[1])],
            &[whole(&outs[0][0]), whole(&outs[0][1])],
        )
        .expect("round 0");
        for round in 1..rounds {
            exchange(
                &[whole(&outs[round - 1][0]), whole(&outs[round - 1][1])],
                &[whole(&outs[round][0]), whole(&outs[round][1])],
            )
            .expect("chained round");
        }
        let mut want = a.iter().zip(&b).map(|(&x, &y)| x + y).collect::<Vec<_>>();
        for (round, pair) in outs.iter().enumerate() {
            for (card, buffer) in pair.iter().enumerate() {
                let got = download_f32(buffer, &streams[card], n).expect("download");
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(g, w, "round {round} card {card} value {i}");
                }
            }
            if round + 1 < rounds {
                for w in &mut want {
                    *w += *w;
                }
            }
        }
        for (card, stream) in streams.iter().enumerate() {
            assert_eq!(
                status_of(&statuses[card], stream).expect("status"),
                0,
                "card {card}: every chained exchange succeeded"
            );
        }
    }
}
