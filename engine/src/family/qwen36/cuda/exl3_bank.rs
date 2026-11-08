//! The banked EXL3 expert load: the admission arithmetic that decides banking and the slot
//! count (pure, mirrored on `project::cuda`'s demand rule), and the host store that holds
//! every routed expert's bytes as one contiguous record per member in the exact device-bank
//! layout, so a later miss copy moves a member into a slot as a single DMA. The store is
//! page-locked after the load ([`ExpertStore::pin_all`]) so refills DMA without staging;
//! the pin is best effort, and a host that refuses it serves staged but correct.

use crate::{
    EngError,
    family::{
        project::cuda::{Reserve, exl3::Exl3MemberStaging, exl3::exl3_member_layout},
        qwen36::settings::{BankMode, SlotRequest},
    },
    kernels::cuda::{DeviceBuffer, DeviceStream, host_register, host_unregister},
};

/// The refusal helper of this module: every admission error names its numbers.
fn refuse(detail: String) -> EngError {
    EngError::MemoryPlan(format!("EXL3 expert bank: {detail}"))
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1 << 30) as f64)
}

/// The resolved banked load: the slot count and the figures the start line reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BankAdmission {
    pub(super) slots: usize,
    /// Dense bytes plus the slot banks' bytes: what the device holds after the load.
    pub(super) device_bytes: u64,
    /// The host store's bytes: every routed expert of every layer, in the bank layout.
    pub(super) host_bytes: u64,
}

/// Decides the load shape from the pack figures and the free memory. `Ok(None)` means stay
/// resident: the whole pack fits and the mode is `auto`. `dense` is the walk's device bytes
/// without the routed experts; `routed` is the routed experts' device bytes alone (the same
/// figure the host store holds, because a member's store bytes are its bank bytes).
///
/// The slot arithmetic mirrors `project::cuda::admit`'s demand rule: a load of `tensors`
/// bytes plus the reserve's `kv_cache + work + headroom` must fit `free`. Banking replaces
/// the routed share with `per_member * slots * layers`.
// Every figure is one named admission input; a config struct would only rename them.
#[allow(clippy::too_many_arguments)]
pub(super) fn decide(
    request: SlotRequest,
    mode: BankMode,
    dense: u64,
    routed: u64,
    layers: usize,
    n_expert: usize,
    free: u64,
    reserve: Reserve,
    host: Option<(u64, u64)>,
) -> Result<Option<BankAdmission>, EngError> {
    let hold = reserve
        .kv_cache
        .saturating_add(reserve.work)
        .saturating_add(reserve.headroom);
    let per_member = per_member_bytes(routed, layers, n_expert)?;
    let resident = dense
        .checked_add(routed)
        .ok_or_else(|| refuse("the resident figure overflows u64".into()))?;
    if mode == BankMode::Auto && resident.saturating_add(hold) <= free {
        return Ok(None);
    }
    admit_host(routed, host)?;
    // The load also leaves the resident flow's post-load floor free: the CUDA context and
    // workspaces eat bytes the figures above do not count, and the per-tensor admits after
    // the banks re-subtract the full hold (the embed table's admit is the last one).
    let budget = free
        .saturating_sub(hold)
        .saturating_sub(dense)
        .saturating_sub(super::pinned::MIN_FREE_AFTER_LOAD as u64);
    let slots = match request {
        SlotRequest::Exactly(want) => {
            if want < 1 || want > n_expert {
                return Err(refuse(format!(
                    "{want} slots are outside 1 to {n_expert}, one banked expert to the experts \
                     of a layer",
                )));
            }
            let demand = per_member
                .checked_mul(want as u64)
                .and_then(|bytes| bytes.checked_mul(layers as u64))
                .ok_or_else(|| refuse("the slot demand overflows u64".into()))?;
            if demand > budget {
                return Err(refuse(format!(
                    "{} slots need {} of expert bytes, the budget after the dense weights and \
                     the reserve is {} (free {}, hold {}, dense {})",
                    want,
                    gib(demand),
                    gib(budget),
                    gib(free),
                    gib(hold),
                    gib(dense),
                )));
            }
            want
        }
        SlotRequest::Fit => {
            // One slot costs `per_member` per LAYER; the budget must cover every layer's bank.
            let slots = (budget / per_member.max(1) / layers as u64) as usize;
            if slots < 1 {
                return Err(refuse(format!(
                    "the budget after the dense weights and the reserve leaves {slots} slots, \
                     below one (budget {}, per member {})",
                    gib(budget),
                    gib(per_member),
                )));
            }
            slots.min(n_expert)
        }
    };
    let device_bytes = dense.saturating_add(
        per_member
            .saturating_mul(slots as u64)
            .saturating_mul(layers as u64),
    );
    Ok(Some(BankAdmission {
        slots,
        device_bytes,
        host_bytes: routed,
    }))
}

/// The routed-expert bytes of one expert of one layer (the three projections summed). The
/// walk guarantees one rate per projection, so the division is exact; a remainder means a
/// pack the walk did not see, and is refused rather than mis-banked.
fn per_member_bytes(routed: u64, layers: usize, n_expert: usize) -> Result<u64, EngError> {
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

/// The host store's admission: the routed bytes must fit the available RAM under an OS
/// margin of the larger of 2 GiB and 20% of the total. The figures come from
/// `/proc/meminfo` (`probe_host`); a host without them does not get a banked load.
fn admit_host(routed: u64, host: Option<(u64, u64)>) -> Result<(), EngError> {
    let Some((total, available)) = host else {
        return Err(refuse(
            "the host memory figures are unavailable, so the expert store cannot be admitted"
                .into(),
        ));
    };
    let margin = (2 << 30).max(total / 5);
    let budget = available.saturating_sub(margin);
    if routed > budget {
        return Err(refuse(format!(
            "the expert store needs {} of host RAM, the budget is {} (available {}, margin {})",
            gib(routed),
            gib(budget),
            gib(available),
            gib(margin),
        )));
    }
    Ok(())
}

/// Every routed expert of one projection of one layer, one record per member in the exact
/// device-bank layout: member `i`'s `[trellis, scales]` bytes at `i * record`, so a slot
/// fill or miss copy is a single contiguous write.
pub(super) struct PartStore {
    records: Vec<u8>,
    record: usize,
    trellis: usize,
    scale_stride: usize,
    pinned: bool,
}

impl PartStore {
    pub(super) fn new(shape: [usize; 2], bits_twice: u8, members: usize) -> Self {
        let (trellis, scale_stride) = exl3_member_layout(shape, bits_twice);
        let record = trellis + scale_stride;
        Self {
            records: vec![0; record * members],
            record,
            trellis,
            scale_stride,
            pinned: false,
        }
    }

    /// Writes member `member`'s staged bytes as its record. The length checks are the
    /// store's own rate guard: a member whose staged bytes disagree with the first member's
    /// strides is refused, not mis-laid.
    pub(super) fn put(
        &mut self,
        member: usize,
        staged: &Exl3MemberStaging,
    ) -> Result<(), EngError> {
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
    pub(super) fn member(&self, member: usize) -> &[u8] {
        let at = member * self.record;
        &self.records[at..at + self.record]
    }

    pub(super) fn bytes(&self) -> usize {
        self.records.len()
    }

    /// Page-locks the record storage, best effort: a store that fails to pin stays
    /// pageable (the refills still work, only staged). The vector never moves after
    /// the load, so the registered pointer stays the buffer's own.
    pub(super) fn pin(&mut self) {
        if host_register(&self.records).is_ok() {
            self.pinned = true;
        }
    }
}

impl Drop for PartStore {
    fn drop(&mut self) {
        // The page lock must go before the Vec free; a failure leaves the lock to the
        // process exit, which the CUDA runtime cleans up.
        if self.pinned {
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
    pub(super) fn push_layer(&mut self, parts: [PartStore; 3]) {
        self.layers.push(parts);
    }

    pub(crate) fn layers(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn layer(&self, layer: usize) -> &[PartStore; 3] {
        &self.layers[layer]
    }

    /// Page-locks every member's storage (see [`PartStore::pin`]).
    pub(crate) fn pin_all(&mut self) {
        for parts in &mut self.layers {
            for part in parts {
                part.pin();
            }
        }
    }

    /// The whole store's bytes, for the start line and the admission's echo.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|parts| parts.iter().map(PartStore::bytes).sum::<usize>())
            .sum::<usize>() as u64
    }
}

/// The least-recently-used slot table of one layer: which expert sits in which slot of the
/// layer's three banks. Pure — the device copies live one level up, so the policy is testable
/// without a GPU.
///
/// One slot index addresses member `i` of all three parts' device banks and of the store's
/// part stores alike, so the table is one array per layer, not per part.
struct LayerSlots {
    experts: usize,
    slots: usize,
    /// Expert -> slot, [`EMPTY`] when absent.
    slot_of: Vec<u16>,
    /// Slot -> expert, [`EMPTY`] when the slot never filled.
    expert_of: Vec<u16>,
    last_used: Vec<u64>,
    tick: u64,
}

/// The absent marker of both directions of the slot table.
const EMPTY: u16 = u16::MAX;

impl LayerSlots {
    fn new(experts: usize, slots: usize) -> Self {
        Self {
            experts,
            slots,
            slot_of: vec![EMPTY; experts],
            expert_of: vec![EMPTY; slots],
            last_used: vec![0; slots],
            tick: 0,
        }
    }

    /// Maps `picks` (this layer's routed experts of one token, distinct) to slot indices,
    /// evicting for every miss the least-recently-used slot that no pick of THIS call uses —
    /// all picks must be resident together, because the launches of one block read them all.
    /// Returns the slots in pick order and, for each miss, the slot whose device bytes the
    /// caller must refill from the store.
    fn assign(&mut self, picks: &[usize]) -> Result<(Vec<usize>, Vec<usize>), EngError> {
        let mut slots = Vec::with_capacity(picks.len());
        let mut misses = Vec::new();
        for &expert in picks {
            if expert >= self.experts {
                return Err(EngError::BadRequest(format!(
                    "EXL3 expert bank: pick {expert} exceeds the {} experts of the layer",
                    self.experts
                )));
            }
            let slot = match usize::from(self.slot_of[expert]) {
                s if s < self.slots => s,
                _ => self.victim(&slots)?,
            };
            let miss = self.slot_of[expert] == EMPTY;
            if miss {
                let evicted = self.expert_of[slot];
                if evicted != EMPTY {
                    self.slot_of[usize::from(evicted)] = EMPTY;
                }
                self.expert_of[slot] = u16::try_from(expert).map_err(|_| {
                    EngError::BadRequest(format!(
                        "EXL3 expert bank: expert {expert} does not fit the slot table"
                    ))
                })?;
                self.slot_of[expert] = u16::try_from(slot).expect("slots fit u16 (max 256)");
                misses.push(slot);
            }
            slots.push(slot);
        }
        self.tick += 1;
        for &slot in &slots {
            self.last_used[slot] = self.tick;
        }
        Ok((slots, misses))
    }

    /// The next slot for a miss: an empty one, else the least-recently-used slot no pick of
    /// this call took.
    fn victim(&self, taken: &[usize]) -> Result<usize, EngError> {
        if let Some(slot) = (0..self.slots).find(|&slot| self.expert_of[slot] == EMPTY) {
            return Ok(slot);
        }
        (0..self.slots)
            .filter(|slot| !taken.contains(slot))
            .min_by_key(|&slot| self.last_used[slot])
            .ok_or_else(|| {
                EngError::BadRequest(format!(
                    "EXL3 expert bank: {} slots cannot hold the {} picks of one call \
                     (a decode token, or a prefill sub-block's rows together)",
                    self.slots,
                    taken.len() + 1
                ))
            })
    }

    /// The expert a slot holds (for the store read of a miss copy).
    fn expert_of(&self, slot: usize) -> usize {
        usize::from(self.expert_of[slot])
    }
}

/// One routed bank's device upload target: the record buffer a slot's bytes land in, and
/// the record stride of [`exl3_member_layout`].
struct BankTarget<'w> {
    bank: &'w DeviceBuffer,
    record: usize,
}

/// The upload target of one bank, by shared layout (a plain fn so the borrowed buffers keep
/// the resolver's lifetime).
fn target_of<'w>(bank: &'w crate::family::project::cuda::exl3::Exl3Bank) -> BankTarget<'w> {
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
}

impl<'w> Exl3SlotResolver<'w> {
    /// Builds the resolver from one resolved [`Exl3MoeWeights`](super::exl3_moe::Exl3MoeWeights)
    /// per layer (banked: banks of `slots` members) and the matching store, so table, banks and
    /// store agree on the layer order and the part order (gate, up, down).
    pub(crate) fn new(
        store: ExpertStore,
        blocks: &[super::exl3_moe::Exl3MoeWeights<'w>],
        experts: usize,
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
                    LayerSlots::new(experts, slots),
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
        })
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

impl LayerResolver<'_, '_> {
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
        #[cfg(test)]
        let began = std::time::Instant::now();
        let (table, parts) = &mut self.resolver.layers[self.layer];
        let store = self.resolver.store.layer(self.layer);
        let (slots, misses) = table.assign(picks)?;
        let miss_count = misses.len();
        for slot in misses {
            let expert = table.expert_of(slot);
            for (part, target) in parts.iter().enumerate() {
                let record = store[part].member(expert);
                // Enqueued on the launch stream: the copy races the launches queued
                // after it instead of a device-wide sync per miss, and the next
                // layer's readback (already a stream wait) covers its completion.
                target
                    .bank
                    .write_at_enqueue(stream, slot * target.record, record)?;
            }
        }
        self.resolver.hits += (picks.len() - miss_count) as u64;
        self.resolver.misses += miss_count as u64;
        #[cfg(test)]
        probe::resolve(began.elapsed());
        Ok(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense ≈ 6 GiB, routed ≈ 10.1 GiB: the q36 EXL3 pack's shape, with the routed figure
    /// built from an exactly-divisible per-member byte count (the guard rejects a figure
    /// that does not divide over the layers and experts).
    fn figures() -> (u64, u64) {
        let per_member = 900_000u64; // ≈ 0.86 MiB per expert per layer (the three parts)
        (6 << 30, 47 * 256 * per_member)
    }

    /// A 16 GiB host with ~14 GiB available: the store fits under the 3.2 GiB margin.
    fn host() -> Option<(u64, u64)> {
        Some((16 << 30, 14 << 30))
    }

    fn reserve(hold: u64) -> Reserve {
        Reserve {
            kv_cache: hold,
            work: 0,
            headroom: 0,
        }
    }

    fn decide_auto(
        dense_routed: (u64, u64),
        free: u64,
        request: SlotRequest,
        host: Option<(u64, u64)>,
    ) -> Result<Option<BankAdmission>, EngError> {
        let (dense, routed) = dense_routed;
        decide(
            request,
            BankMode::Auto,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host,
        )
    }

    #[test]
    fn a_pack_that_fits_stays_resident_in_auto_mode() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30) + (1 << 20);
        assert_eq!(
            decide_auto((dense, routed), free, SlotRequest::Fit, host()).unwrap(),
            None
        );
    }

    #[test]
    fn a_pack_that_does_not_fit_banks_the_fitting_slot_count() {
        let (dense, routed) = figures();
        // Free for 1 GiB of dense + hold + the post-load floor, plus 64 slots per layer.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 64 * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 64);
        assert_eq!(got.host_bytes, routed);
        assert_eq!(got.device_bytes, dense + per_member * 64 * 47);
    }

    #[test]
    fn on_mode_banks_even_when_the_pack_fits() {
        let (dense, routed) = figures();
        let free = dense + routed + (1 << 30);
        let got = decide(
            SlotRequest::Exactly(32),
            BankMode::On,
            dense,
            routed,
            47,
            256,
            free,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 32);
        assert_eq!(
            got.device_bytes,
            dense + (routed / 47 / 256) * 32 * 47,
            "linear in the slot count"
        );
    }

    #[test]
    fn an_exact_count_over_budget_is_refused_with_numbers() {
        let (dense, routed) = figures();
        let free = dense + (1 << 30) + (routed / 47 / 256) * 16;
        let error = format!(
            "{:?}",
            decide_auto((dense, routed), free, SlotRequest::Exactly(200), host()).unwrap_err()
        );
        assert!(error.contains("200 slots"), "{error}");
        assert!(error.contains("budget"), "{error}");
    }

    #[test]
    fn a_host_without_room_is_refused_before_any_slot() {
        let (dense, routed) = figures();
        // On mode: banking is requested even though the pack would fit, and the host refuses.
        // Available 8 GiB under a 3.2 GiB margin (20% of 16): 4.8 GiB < ~10.1 GiB of experts.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                Some((16 << 30, 8 << 30)),
            )
            .unwrap_err()
        );
        assert!(error.contains("host RAM"), "{error}");
        // And a host without figures at all.
        let error = format!(
            "{:?}",
            decide(
                SlotRequest::Fit,
                BankMode::On,
                dense,
                routed,
                47,
                256,
                1 << 40,
                reserve(1 << 30),
                None,
            )
            .unwrap_err()
        );
        assert!(error.contains("unavailable"), "{error}");
    }

    #[test]
    fn fit_never_returns_zero_or_above_the_expert_count() {
        let (dense, routed) = figures();
        // Barely past the dense figure + hold + the post-load floor: one slot fits.
        let per_member = routed / 47 / 256;
        let free = dense + (1 << 30) + (1 << 30) + per_member * 47;
        let got = decide_auto((dense, routed), free, SlotRequest::Fit, host())
            .unwrap()
            .expect("banks");
        assert_eq!(got.slots, 1);
        // And a huge free figure clamps at the full expert count (On mode: Auto would stay
        // resident with room to spare).
        let got = decide(
            SlotRequest::Fit,
            BankMode::On,
            dense,
            routed,
            47,
            256,
            1 << 44,
            reserve(1 << 30),
            host(),
        )
        .unwrap()
        .expect("banks");
        assert_eq!(got.slots, 256);
    }

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

    /// The first picks fill the empty slots in order; a repeat of the same picks is all hits
    /// with the same slots and no misses.
    #[test]
    fn slot_table_fills_in_order_then_hits() {
        let mut table = LayerSlots::new(64, 8);
        let picks = [3usize, 41, 7, 19];
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert_eq!(misses, [0, 1, 2, 3]);
        let (slots, misses) = table.assign(&picks).unwrap();
        assert_eq!(slots, [0, 1, 2, 3]);
        assert!(misses.is_empty());
    }

    /// A miss evicts the least-recently-used slot no pick of the call uses: touching expert 3
    /// again makes expert 41's slot the victim, not 3's.
    #[test]
    fn slot_table_evicts_least_recently_used() {
        let mut table = LayerSlots::new(64, 8);
        table.assign(&[3, 41, 5, 6, 7, 8, 9, 10]).unwrap();
        table.assign(&[3]).unwrap(); // touch 3; 41 is now the least-recently-used
        let (slots, misses) = table.assign(&[50]).unwrap();
        assert_eq!(slots, [1], "the victim is 41's slot");
        assert_eq!(misses, [1]);
        // 41 must miss again; it lands in some other slot.
        let (slots, misses) = table.assign(&[41]).unwrap();
        assert_eq!(misses.len(), 1);
        assert_ne!(
            slots[0], 1,
            "41 cannot return to the slot it was evicted from"
        );
    }

    /// A token whose picks are all new may replace every resident: the victims hold old
    /// experts, which no new pick uses. A token with more picks than slots is refused.
    #[test]
    fn slot_table_replaces_everything_then_refuses_overload() {
        let mut table = LayerSlots::new(64, 4);
        table.assign(&[0, 1, 2, 3]).unwrap();
        let (slots, misses) = table.assign(&[10, 11, 12, 13]).unwrap();
        assert_eq!(misses.len(), 4);
        assert_eq!(slots.len(), 4);
        assert!((0..4).all(|slot| table.expert_of(slot) >= 10));
        let error = format!("{:?}", table.assign(&[20, 21, 22, 23, 24]).unwrap_err());
        assert!(error.contains("cannot hold"), "{error}");
    }

    /// A pick outside the layer's experts is refused by name.
    #[test]
    fn slot_table_refuses_out_of_range_picks() {
        let mut table = LayerSlots::new(64, 8);
        let error = format!("{:?}", table.assign(&[64]).unwrap_err());
        assert!(error.contains("pick 64"), "{error}");
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

    /// The accumulated `(calls, nanos)` of each phase since the last take.
    pub(crate) fn take() -> ((u64, u64), (u64, u64)) {
        (
            READBACK.with(|cell| cell.replace((0, 0))),
            RESOLVE.with(|cell| cell.replace((0, 0))),
        )
    }
}
