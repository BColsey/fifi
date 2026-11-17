//! The banked EXL3 expert load's family-neutral half: the host store that holds every
//! routed expert's bytes as one contiguous record per member in the exact device-bank
//! layout, so a later miss copy moves a member into a slot as a single DMA; the LRU slot
//! table with the sticky hot set ([`LayerSlots`], served from the shared slot-policy module
//! and re-exported here); and the resolver that maps one call's picks to slots and
//! refills the misses. The store is page-locked after the load
//! ([`ExpertStore::pin_all`]) so refills DMA without staging; the pin is best effort, and
//! a host that refuses it serves staged but correct. The admission arithmetic that decides
//! banking stays with the family that owns its settings.

pub(crate) use crate::family::expert_bank::policy::LayerSlots;
// The sentinels serve only the pick-golden tests, so the re-export stays test-gated.
#[cfg(test)]
pub(in crate::family) use crate::family::expert_bank::policy::{DECAY_CALLS, EMPTY, NO_SLOT};

use crate::{
    EngError,
    family::project::cuda::exl3::{Exl3Bank, Exl3MemberStaging, exl3_member_layout},
    kernels::cuda::{
        DeviceBuffer, DeviceEvent, DeviceStream, device_synchronize, host_register, host_unregister,
    },
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

pub(in crate::family) fn per_member_bytes(
    routed: u64,
    layers: usize,
    n_expert: usize,
) -> Result<u64, EngError> {
    if layers == 0 || n_expert == 0 {
        return Err(refuse("the model has no layers or no experts".into()));
    }
    let per_layer = routed / layers as u64;
    if per_layer * layers as u64 != routed {
        return Err(refuse(format!(
            "the routed figure {} does not divide over {layers} layers",
            gib(routed),
        )));
    }
    let per_member = per_layer / n_expert as u64;
    if per_member * n_expert as u64 != per_layer {
        return Err(refuse(
            "the routed figure does not divide evenly over the experts".into(),
        ));
    }
    Ok(per_member)
}

/// Every routed expert of one projection of one layer, one record per member in the exact
/// device-bank layout: member `i`'s `[trellis, scales]` bytes at `i * record`, so a slot
/// fill or miss copy is a single contiguous write.
pub(in crate::family) struct PartStore {
    records: Vec<u8>,
    pub(in crate::family) record: usize,
    trellis: usize,
    scale_stride: usize,
    pinned: bool,
    /// Set when [`Self::pin`] runs: the load is over and refill DMAs may read
    /// `records` concurrently, so no write may touch it again (a pinned source
    /// must stay valid and unmodified until its copy completes).
    sealed: bool,
}

impl PartStore {
    pub(in crate::family) fn new(shape: [usize; 2], bits_twice: u8, members: usize) -> Self {
        let (trellis, scale_stride) = exl3_member_layout(shape, bits_twice);
        let record = trellis + scale_stride;
        Self {
            records: vec![0; record * members],
            record,
            trellis,
            scale_stride,
            pinned: false,
            sealed: false,
        }
    }

    /// Writes member `member`'s staged bytes as its record. The length checks are the
    /// store's own rate guard: a member whose staged bytes disagree with the first member's
    /// strides is refused, not mis-laid.
    pub(in crate::family) fn put(
        &mut self,
        member: usize,
        staged: &Exl3MemberStaging,
    ) -> Result<(), EngError> {
        if self.sealed {
            return Err(EngError::Checkpoint(format!(
                "EXL3 expert bank: member {member} put after the store was pinned; \
                 a sealed store is the source of in-flight refill copies and may \
                 not be written",
            )));
        }
        if staged.trellis.len() != self.trellis || staged.scales.len() != self.scale_stride {
            return Err(EngError::Checkpoint(format!(
                "EXL3 expert bank: member {member} stages {} trellis and {} scale bytes, \
                 the bank layout holds {} and {}",
                staged.trellis.len(),
                staged.scales.len(),
                self.trellis,
                self.scale_stride,
            )));
        }
        let at = member * self.record;
        self.records[at..at + self.trellis].copy_from_slice(&staged.trellis);
        self.records[at + self.trellis..at + self.record].copy_from_slice(&staged.scales);
        Ok(())
    }

    /// Member `member`'s record bytes, the source of a miss copy.
    pub(in crate::family) fn member(&self, member: usize) -> &[u8] {
        let at = member * self.record;
        &self.records[at..at + self.record]
    }

    pub(in crate::family) fn bytes(&self) -> usize {
        self.records.len()
    }

    /// Page-locks the record storage, best effort: a store that fails to pin
    /// stays pageable, and a pageable copy source is consumed at enqueue, so the
    /// refills still work, only staged. The vector never moves after the load, so
    /// the registered pointer stays the buffer's own. Returns whether the pin
    /// took. Also seals the store (see [`Self::put`]); a second call is harmless
    /// and reports the first call's outcome without re-registering.
    pub(in crate::family) fn pin(&mut self) -> bool {
        // The seal is set whether or not the registration takes: refills read
        // this store either way, so the load is over the moment pin runs. A
        // second call just reports the first outcome, never re-registering.
        if self.sealed {
            return self.pinned;
        }
        self.sealed = true;
        if host_register(&self.records).is_ok() {
            self.pinned = true;
        }
        self.pinned
    }
}

impl Drop for PartStore {
    fn drop(&mut self) {
        // The page lock must go before the Vec free; a failure leaves the lock to the
        // process exit, which the CUDA runtime cleans up. Unregistering while a refill
        // DMA still reads the range is undefined, so drain the device first; live drop
        // paths reach this with the stream already drained, and the cost is irrelevant
        // on a drop path.
        if self.pinned {
            let _ = device_synchronize();
            let _ = host_unregister(&self.records);
        }
    }
}

/// Every routed expert of every layer: `layers` entries of the `[gate, up, down]` parts.
pub(crate) struct ExpertStore {
    layers: Vec<[PartStore; 3]>,
}

impl ExpertStore {
    pub(crate) fn with_layers(layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(layers),
        }
    }

    /// Appends one layer's three part stores; the load fills them layer by layer.
    pub(in crate::family) fn push_layer(&mut self, parts: [PartStore; 3]) {
        self.layers.push(parts);
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> &[PartStore; 3] {
        &self.layers[layer]
    }

    /// Page-locks every member's storage (see [`PartStore::pin`]); true when every
    /// member pinned. Every part is pinned even when some registrations fail:
    /// `Iterator::all` would short-circuit past the first failure, and the parts
    /// it skipped would never seal, leaving them writable while refills read
    /// them.
    pub(crate) fn pin_all(&mut self) -> bool {
        self.layers
            .iter_mut()
            .flat_map(|parts| parts.iter_mut())
            .fold(true, |all, part| part.pin() & all)
    }

    /// The whole store's bytes, for the start line and the admission's echo.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|parts| parts.iter().map(PartStore::bytes).sum::<usize>())
            .sum::<usize>() as u64
    }
}

/// One routed bank's device upload target: the record buffer a slot's bytes land in, and
/// the record stride of [`exl3_member_layout`].
struct BankTarget<'w> {
    bank: &'w DeviceBuffer,
    pub(in crate::family) record: usize,
}

/// The upload target of one bank, by shared layout (a plain fn so the borrowed buffers keep
/// the resolver's lifetime).
fn target_of<'w>(bank: &'w Exl3Bank) -> BankTarget<'w> {
    BankTarget {
        bank: bank.bank_buffer(),
        record: bank.record_bytes(),
    }
}

/// The banked decode state: every layer's slot table with its three banks' upload targets,
/// and the host store the misses copy from. The banks the tables point into are the same
/// banks the block's views read, resolved once at the start.
pub(crate) struct Exl3SlotResolver<'w> {
    layers: Vec<(LayerSlots, [BankTarget<'w>; 3])>,
    store: ExpertStore,
    hits: u64,
    misses: u64,
    /// The overlap pair (`Q36_CUDA_BANK_OVERLAP=1`): a copy stream that carries miss refills
    /// beside the launch stream, and the event the launches reading the refilled slots wait.
    /// Only [`LayerResolver::resolve_split`] uses it; a plain [`LayerResolver::resolve`]
    /// never does, so a caller that cannot split its launches keeps today's ordering.
    overlap: Option<(DeviceStream, DeviceEvent)>,
}

impl<'w> Exl3SlotResolver<'w> {
    /// Builds the resolver from one resolved [`Exl3MoeWeights`](crate::family::exl3::moe::Exl3MoeWeights)
    /// per layer (banked: banks of `slots` members) and the matching store, so table, banks and
    /// store agree on the layer order and the part order (gate, up, down).
    pub(crate) fn new(
        store: ExpertStore,
        blocks: &[crate::family::exl3::moe::Exl3MoeWeights<'w>],
        experts: usize,
        sticky: usize,
    ) -> Result<Self, EngError> {
        if store.layers() != blocks.len() {
            return Err(EngError::MemoryPlan(format!(
                "EXL3 expert bank: the store holds {} layers, the step resolved {}",
                store.layers(),
                blocks.len()
            )));
        }
        let layers = blocks
            .iter()
            .map(|block| {
                let slots = block.gate.count();
                for bank in [block.gate, block.up, block.down] {
                    if bank.count() != slots {
                        return Err(EngError::MemoryPlan(format!(
                            "EXL3 expert bank: a layer's banks hold {} gate, {} up and {} down \
                             slots",
                            slots,
                            block.up.count(),
                            block.down.count()
                        )));
                    }
                }
                Ok((
                    LayerSlots::new(experts, slots, sticky),
                    [
                        target_of(block.gate),
                        target_of(block.up),
                        target_of(block.down),
                    ],
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            layers,
            store,
            hits: 0,
            misses: 0,
            overlap: None,
        })
    }

    /// Arms the overlap pair of [`LayerResolver::resolve_split`]: one copy stream for miss
    /// refills, one event to gate the launches that read them. The start of a banked session
    /// whose environment asked for it arms this once, before the first token; a later arm
    /// stays safe (every resolve_split sits at the readback quiescence anchor), but the
    /// start never does it and callers should not rely on mid-session arming.
    pub(crate) fn enable_overlap(
        &mut self,
        context: &crate::kernels::cuda::CudaContext,
    ) -> Result<(), EngError> {
        if self.overlap.is_some() {
            return Ok(());
        }
        self.overlap = Some((context.stream()?, context.event()?));
        Ok(())
    }

    /// Whether [`LayerResolver::resolve_split`] refills on the copy stream (the start armed
    /// the pair from `Q36_CUDA_BANK_OVERLAP=1`); the MoE blocks read this to split their
    /// launches around the refills.
    pub(crate) fn overlap_armed(&self) -> bool {
        self.overlap.is_some()
    }

    /// Drains the overlap copy stream (see [`LayerResolver::fence_copy_stream`]); a no-op
    /// without the pair armed.
    pub(crate) fn fence_copy_stream(&mut self) {
        if let Some((copy, _)) = &self.overlap
            && let Err(error) = copy.synchronize()
        {
            eprintln!("q36 cuda: overlap fence failed: {error:?}");
        }
    }

    /// One layer's resolver view, what the MoE block hooks take.
    pub(crate) fn layer(&mut self, layer: usize) -> LayerResolver<'_, 'w> {
        LayerResolver {
            resolver: self,
            layer,
        }
    }

    /// The slot count of every layer's bank (the prefill sub-block arithmetic).
    pub(crate) fn slots(&self) -> usize {
        self.layers
            .first()
            .map(|(slots, _)| slots.slots)
            .unwrap_or(0)
    }

    /// The hit/miss counts of the session, for the start-line echo and the report.
    pub(crate) fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }
}

/// One layer's slice of the resolver: the block hooks need the layer bound once, not threaded
/// through every launch builder.
pub(crate) struct LayerResolver<'r, 'w> {
    resolver: &'r mut Exl3SlotResolver<'w>,
    layer: usize,
}

/// The slot for every pick, the picks gated on a refill of their slot, and the refill's event
/// when one fired (absent when nothing refilled).
type ResolveOutcome<'a> = Result<(Vec<usize>, Vec<usize>, Option<&'a DeviceEvent>), EngError>;

impl LayerResolver<'_, '_> {
    /// Whether this resolver's calls refill on the copy stream (the arming of the resolver
    /// it borrows); the launch builders read this to split around the refills. Every reader
    /// lives in test or device-harness code today, so the passthrough itself is test-gated.
    #[cfg(test)]
    pub(crate) fn overlap_armed(&self) -> bool {
        self.resolver.overlap_armed()
    }

    /// Drains the overlap copy stream. The error path out of a split MoE block queues no
    /// wait, so its refills keep writing with no fence; a later token's hit on a refilling
    /// slot could read torn bytes, because only the launch stream is drained at the next
    /// readback. Every block that resolves with a split fences on its way out with an error.
    pub(crate) fn fence_copy_stream(&mut self) {
        self.resolver.fence_copy_stream();
    }

    /// Maps one call's picks (a decode token's, a prefill sub-block's, or a K-row window
    /// row's) to slots, refilling the missed slots' device bytes from the store. The copies
    /// are enqueued on `stream` (`write_at_enqueue`; the store is page-locked after the load):
    /// the readback that produced the picks has already waited for the stream, and the
    /// launches this return feeds queue behind the copies on the same stream.
    pub(crate) fn resolve(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
    ) -> Result<Vec<usize>, EngError> {
        Ok(self.resolve_mapped(stream, picks, false)?.0)
    }

    /// [`Self::resolve`] for a caller that splits its launches around the refills
    /// (`Q36_CUDA_BANK_OVERLAP=1`): the missed slots refill on the resolver's copy stream
    /// instead, the event lands in the return, and the caller queues the launches that read
    /// only already-resident slots before the wait (their compute hides the refill DMA), the
    /// launches that read a refilled slot after it. Without the pair armed this is exactly
    /// [`Self::resolve`] with an absent event.
    ///
    /// Safety rests on the readback that produced the picks: it waited the launch stream,
    /// which had waited every earlier copy event before the launches it read back, so both
    /// streams are quiescent here and a copy-stream refill cannot overtake a queued reader.
    pub(crate) fn resolve_split(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
        overlap: bool,
    ) -> ResolveOutcome<'_> {
        let (slots, refilled, event) = self.resolve_mapped(stream, picks, overlap)?;
        // A pick gates on the event when its SLOT refilled this call, not when the pick
        // itself reported the miss: a duplicate pick of a just-refilled expert reads the
        // same refilling bytes and must wait with the miss.
        let gated = (0..picks.len())
            .filter(|&p| refilled.contains(&slots[p]))
            .collect();
        Ok((slots, gated, event))
    }

    /// The shared body: `assign`, the refills, the stats. With `split`, the refills go on
    /// the overlap copy stream and the event is recorded on it (absent when nothing
    /// refilled — no wait orders anything); without, they ride `stream` as they always did.
    fn resolve_mapped(
        &mut self,
        stream: &DeviceStream,
        picks: &[usize],
        split: bool,
    ) -> ResolveOutcome<'_> {
        #[cfg(test)]
        let began = std::time::Instant::now();
        let (table, parts) = &mut self.resolver.layers[self.layer];
        let store = self.resolver.store.layer(self.layer);
        let (slots, misses) = table.assign(picks)?;
        let miss_count = misses.len();
        let copy = if split && miss_count > 0 {
            self.resolver.overlap.as_ref()
        } else {
            None
        };
        let refill = copy.map_or(stream, |(copy, _)| copy);
        for op in refill_plan(table, &misses, parts_strides(parts)) {
            let record = store[op.part].member(op.expert);
            // On the launch stream (no split): the copy races the launches queued
            // after it instead of a device-wide sync per miss, and the next layer's
            // readback (already a stream wait) covers its completion. On the copy
            // stream (split): the event below gates the launches that read the slot;
            // both streams were quiescent at the readback that produced the picks.
            // The store is pinned, so the record must stay untouched until the
            // streams drain; the store is immutable post-load and
            // process-long-lived, which is that invariant — and pin() seals the
            // store so a post-load write refuses instead of racing this DMA.
            if let Err(err) = parts[op.part]
                .bank
                .write_at_enqueue(refill, op.offset(), record)
            {
                // The write failed mid-plan: the enqueued copies of this call's misses
                // cannot be trusted to be complete, and the table still names their
                // incoming experts. Both directions of every missed slot read absent
                // again, so the next call re-misses them and refills over whatever
                // bytes landed.
                table.invalidate(&misses);
                return Err(err);
            }
        }
        let event = match (copy, miss_count > 0) {
            (Some((_, event)), true) => {
                if let Err(err) = refill.record(event) {
                    // The writes were enqueued, but the gate the launches wait on was
                    // not: treat the misses as torn, as above, so correctness does not
                    // depend on the caller's fence succeeding.
                    table.invalidate(&misses);
                    return Err(err);
                }
                Some(event)
            }
            _ => None,
        };
        self.resolver.hits += (picks.len() - miss_count) as u64;
        self.resolver.misses += miss_count as u64;
        #[cfg(test)]
        if trace::armed() {
            trace::record(self.layer, picks);
        }
        #[cfg(test)]
        probe::resolve(began.elapsed());
        Ok((slots, misses, event))
    }
}

/// One record copy of a layer refill: the slot that refills, the expert now in it, the
/// part's buffer index, and the record stride of that part; the write offset is the slot
/// times the stride ([`RefillOp::offset`]).
pub(in crate::family) struct RefillOp {
    slot: usize,
    pub(in crate::family) expert: usize,
    pub(in crate::family) part: usize,
    pub(in crate::family) record: usize,
}

impl RefillOp {
    pub(in crate::family) fn offset(&self) -> usize {
        self.slot * self.record
    }
}

/// The record strides of the three parts — the one place the plan's stride inputs are
/// derived. `resolve_mapped` passes exactly this to [`refill_plan`]; the pick-replay
/// harness mirrors the values (`PART_LEN`) against this derivation.
fn parts_strides(parts: &[BankTarget<'_>; 3]) -> [usize; 3] {
    [parts[0].record, parts[1].record, parts[2].record]
}

/// The refill plan of one layer's misses — in miss order, then part order — as the exact
/// copies [`LayerResolver::resolve_mapped`] issues, parameterized by the three parts'
/// record strides. Pure host data over the table: the pick-replay harness records this
/// instead of recomputing the refill semantics, so a change in the engine's refill order
/// fails the golden compare.
pub(in crate::family) fn refill_plan(
    table: &LayerSlots,
    misses: &[usize],
    strides: [usize; 3],
) -> Vec<RefillOp> {
    let mut ops = Vec::with_capacity(misses.len() * 3);
    for &slot in misses {
        let expert = table.expert_of(slot);
        for (part, &record) in strides.iter().enumerate() {
            ops.push(RefillOp {
                slot,
                expert,
                part,
                record,
            });
        }
    }
    ops
}

/// Test-only pick-trace capture: the admission-policy study replays real routing through
/// offline slot policies. `arm()` starts capture, `dump` writes it.
#[cfg(test)]
pub(crate) mod trace {
    use std::{
        cell::RefCell,
        sync::atomic::{AtomicBool, Ordering},
    };

    static ARMED: AtomicBool = AtomicBool::new(false);
    thread_local! {
        static CALLS: RefCell<Vec<(usize, Vec<u32>)>> = const { RefCell::new(Vec::new()) };
    }

    pub(crate) fn arm() {
        ARMED.store(true, Ordering::Relaxed);
        CALLS.with(|c| c.borrow_mut().clear());
    }

    pub(crate) fn armed() -> bool {
        ARMED.load(Ordering::Relaxed)
    }

    pub(crate) fn record(layer: usize, picks: &[usize]) {
        CALLS.with(|c| {
            c.borrow_mut()
                .push((layer, picks.iter().map(|&e| e as u32).collect()));
        });
    }

    /// One line per resolve call: `layer pick pick ...`.
    pub(crate) fn dump(path: &std::path::Path) -> std::io::Result<usize> {
        use std::io::Write;
        let mut lines = 0;
        CALLS.with(|c| {
            let calls = c.borrow();
            let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
            for (layer, picks) in calls.iter() {
                write!(out, "{layer}")?;
                for &e in picks {
                    write!(out, " {e}")?;
                }
                writeln!(out)?;
                lines += 1;
            }
            out.flush()
        })?;
        ARMED.store(false, Ordering::Relaxed);
        Ok(lines)
    }
}

#[cfg(test)]
#[cfg(test)]
pub(crate) mod probe {
    //! Test-only timing of the banked readback and resolve, so a decode probe can report
    //! what share of a token the host round trips take.
    use std::cell::Cell;

    thread_local! {
        static READBACK: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
        static RESOLVE: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    }

    pub(crate) fn readback(spent: std::time::Duration) {
        READBACK.with(|cell| {
            let (calls, nanos) = cell.get();
            cell.set((calls + 1, nanos + spent.as_nanos() as u64));
        });
    }

    pub(crate) fn resolve(spent: std::time::Duration) {
        RESOLVE.with(|cell| {
            let (calls, nanos) = cell.get();
            cell.set((calls + 1, nanos + spent.as_nanos() as u64));
        });
    }

    /// The accumulated `(calls, nanos)` of each phase since the last take.
    pub(crate) fn take() -> ((u64, u64), (u64, u64)) {
        (
            READBACK.with(|cell| cell.replace((0, 0))),
            RESOLVE.with(|cell| cell.replace((0, 0))),
        )
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_store_lays_members_as_contiguous_records() {
        let shape = [64usize, 32];
        let (stride, scale_stride) = exl3_member_layout(shape, 8);
        let mut store = PartStore::new(shape, 8, 4);
        let member = |trellis_byte: u8| Exl3MemberStaging {
            trellis: vec![trellis_byte; stride],
            scales: vec![trellis_byte + 1; scale_stride],
        };
        for i in 0..4u8 {
            store.put(i as usize, &member(i)).unwrap();
        }
        for i in 0..4u8 {
            let record = store.member(i as usize);
            // One contiguous record per member: its trellis then its scales, adjacent.
            assert_eq!(record.len(), stride + scale_stride);
            assert!(record[..stride].iter().all(|&byte| byte == i));
            assert!(record[stride..].iter().all(|&byte| byte == i + 1));
        }
        // A member staged at another rate is refused, not mis-laid.
        let error = format!(
            "{:?}",
            store
                .put(
                    2,
                    &Exl3MemberStaging {
                        trellis: vec![0; stride - 1],
                        scales: vec![0; scale_stride],
                    },
                )
                .unwrap_err()
        );
        assert!(error.contains("member 2"), "{error}");
    }

    #[test]
    fn pin_seals_the_store_against_writes() {
        let shape = [64usize, 32];
        let (stride, scale_stride) = exl3_member_layout(shape, 8);
        let staged = Exl3MemberStaging {
            trellis: vec![7u8; stride],
            scales: vec![8u8; scale_stride],
        };
        // An unsealed store accepts writes.
        let mut fresh = PartStore::new(shape, 8, 2);
        fresh
            .put(0, &staged)
            .expect("an unsealed store accepts writes");
        // pin() seals whether or not the registration takes (a machine with no
        // visible device refuses the register; the seal must hold regardless).
        let _pinned = fresh.pin();
        let refusal = format!(
            "{:?}",
            fresh
                .put(1, &staged)
                .expect_err("a pinned store refuses writes")
        );
        assert!(refusal.contains("member 1"), "{refusal}");
        assert!(refusal.contains("sealed"), "{refusal}");
        // The refused write left member 1 zeroed, and the record written before
        // the seal is untouched: its trellis half and scale half separately,
        // as member() returns one whole record.
        assert!(fresh.member(1).iter().all(|&byte| byte == 0));
        let record = fresh.member(0);
        assert!(record[..stride].iter().all(|&byte| byte == 7));
        assert!(record[stride..].iter().all(|&byte| byte == 8));
        // A second pin() is harmless: same answer, no re-registration.
        assert_eq!(fresh.pin(), fresh.pin());
    }

    #[test]
    fn pin_all_seals_every_part_even_when_registrations_fail() {
        let shape = [64usize, 32];
        let (stride, scale_stride) = exl3_member_layout(shape, 8);
        let staged = Exl3MemberStaging {
            trellis: vec![7u8; stride],
            scales: vec![8u8; scale_stride],
        };
        let part = || PartStore::new(shape, 8, 2);
        let mut store = ExpertStore::with_layers(1);
        store.push_layer([part(), part(), part()]);
        // On a host with no visible device every registration fails; on a GPU
        // host they all succeed. Either way, every part must seal: Iterator::all
        // short-circuits past the first failure and left the rest writable.
        let _ = store.pin_all();
        let last = 2;
        let refusal = format!(
            "{:?}",
            store.layers[0][last]
                .put(1, &staged)
                .expect_err("pin_all seals every part, the last included")
        );
        assert!(refusal.contains("member 1"), "{refusal}");
        assert!(refusal.contains("sealed"), "{refusal}");
    }
}
