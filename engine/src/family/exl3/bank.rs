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

use crate::{
    EngError,
    kernels::cuda::{DeviceBuffer, DeviceEvent, DeviceStream, device_synchronize, host_unregister},
};

/// Every routed expert of one projection of one layer, one record per member in the exact
/// device-bank layout: member `i`'s `[trellis, scales]` bytes at `i * record`, so a slot
/// fill or miss copy is a single contiguous write.
pub(in crate::family) struct PartStore {
    records: Vec<u8>,
    pub(in crate::family) record: usize,
    pinned: bool,
}

impl PartStore {
    /// Member `member`'s record bytes, the source of a miss copy.
    pub(in crate::family) fn member(&self, member: usize) -> &[u8] {
        let at = member * self.record;
        &self.records[at..at + self.record]
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
    pub(crate) fn layer(&self, layer: usize) -> &[PartStore; 3] {
        &self.layers[layer]
    }
}

/// One routed bank's device upload target: the record buffer a slot's bytes land in, and
/// the record stride of [`exl3_member_layout`].
struct BankTarget<'w> {
    bank: &'w DeviceBuffer,
    pub(in crate::family) record: usize,
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
    /// Drains the overlap copy stream (see [`LayerResolver::fence_copy_stream`]); a no-op
    /// without the pair armed.
    pub(crate) fn fence_copy_stream(&mut self) {
        if let Some((copy, _)) = &self.overlap
            && let Err(error) = copy.synchronize()
        {
            eprintln!("q36 cuda: overlap fence failed: {error:?}");
        }
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
            // process-long-lived, which is that invariant - and pin() seals the
            // store so a post-load write refuses instead of racing this DMA.
            // Rule 4 (the readback ordering) is a convention - see
            // doc/reference/cuda-conventions.md; the store-lifetime half is
            // enforced by pin()/Drop (rule 3 there).
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

    pub(crate) fn armed() -> bool {
        ARMED.load(Ordering::Relaxed)
    }

    pub(crate) fn record(layer: usize, picks: &[usize]) {
        CALLS.with(|c| {
            c.borrow_mut()
                .push((layer, picks.iter().map(|&e| e as u32).collect()));
        });
    }
}

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
}
