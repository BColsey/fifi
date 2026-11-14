//! The expert bank of the Qwen3.8 Flash-Next CUDA step: which routed experts sit in the device slots
//! of every layer, which pinned host place holds every other expert, and the copies a token's picks
//! cause. The policy, the host places and the budget are pure host code; the device owner at the end
//! of the file only allocates, fills and copies.
//!
//! WHAT, in the order of a decode layer:
//!
//! 1. The router's picks go through `tiered_classify` against the layer's slot row
//!    ([`ExpertBankState::slot_row`], the oracle is `SlotTable::classify` / `classify_reference`): a
//!    resident pick maps to its slot, a missed pick to slot 0 with weight +0.0 (the filler rule).
//! 2. The resident pass runs, and the host learns the picks (hits included) so that
//!    [`ExpertBankState::begin_pass`] can plan the copies: every expert of the pass is protected, a
//!    miss takes the least recently used unprotected slot, and the slot's old expert leaves the card.
//! 3. [`PassCopies`] says what to move. The copy is synchronous (the missed picks wait for it).
//! 4. The missed picks run on their new slots with their true weights — the missed picks ONLY,
//!    no filler ([`miss_pass_inputs`]); the two passes' weights add to the router's weights bit
//!    for bit.
//!
//! WHY this shape:
//!
//! - Policy. The T0 replay (`flash-next-routing-locality-t0-trace`) gives per-layer LRU 0.870 / 0.915
//!   hits at 143 / 184 slots against 0.683 / 0.763 for a static popularity core, so the popularity
//!   profile ([`parse_route_counts`], `TierPlan::hot_set`) only fills the slots before the first
//!   prompt. Q36's cap of 0.25 swaps per token came from SSD-backed misses; a Flash-Next miss is one
//!   72 us copy from pinned RAM, so every miss is admitted at once.
//! - Slot 0 is protected while a pass has a miss, because the filler reads it while the copies write
//!   other slots. A torn float there would make `0.0 * x` a NaN. So a bank needs `top_k + 1` slots.
//! - The pinned store is exclusive by default ([`HostMode::Exclusive`]): it holds only the experts
//!   that are not on the card, 25.6 to 31.1 GiB instead of the 43.07 GiB of an inclusive store. When
//!   an expert leaves a slot, its record goes back into the host place its successor just left
//!   (one spare place per layer makes the order work: demote first, then free the incoming place).
//!   The write-back (D2H) leg is not measured yet. [`HostMode::Inclusive`] keeps every expert in the
//!   store and drops the D2H leg; [`HostPinPlan`] prices both.
//! - One record per expert and one device buffer per layer: the record layout is `SlotBankLayout`'s
//!   (equal strides for the id-indexed launch), the policy and the placement are pure and tested on
//!   the host.
//!
//! The slot policy lives in the shared module (`family::expert_bank::policy`): the same
//! `ReplaceOnMissLru` behind the same [`SlotPolicy`] seam, moved there so a second family's bank
//! can serve it unchanged. The interface is `plan_pass(layer, picks) -> PassPlan`,
//! `admit(layer, experts_by_use) -> PassPlan`, `slot_row(layer)` and `expert_in_slot(layer, slot)`.
//! Host places, copies, budget and the device owner do not change. The policy must keep three rules:
//! the experts of a pass are never victims, slot 0 is not a victim in a pass with a miss, and the
//! slot row it publishes is the classifier's table.
//!
//! Nothing here is called by a model step yet; the decode step wires it in a later change.

// Staged: the Flash-Next CUDA step is the first caller.
#![cfg_attr(not(test), allow(dead_code))]

use crate::{
    EngError,
    family::{
        flash_next::moe::{slot_bank::SlotBankLayout, tier_plan::TierPlan},
        qwen36::tiered::classify::Classified,
    },
};

// The seam moved to `family/expert_bank` (c2): re-exported here so every existing path
// under `flash_next::cuda` and the golden harnesses keep compiling unchanged. The five
// marked names are consumed only by native-cuda-gated modules, so a plain host build sees
// the re-export as unused (knowledge: `cpu-clippy-red-on-main-delta-layer-seam`).
#[allow(unused_imports)]
pub(super) use crate::family::expert_bank::{
    BankGeometry, BankStats, CopyOp, Demote, ExpertBankState, HostMode, HostPlaces, NO_PLACE,
    PassCopies, PassPlan, ReplaceOnMissLru, SlotMove, SlotPolicy, SlotTable, bank_error,
    plan_error,
};

impl BankGeometry {
    /// The bank the tier plan settled, for records of `layout`. The record size must be the plan's
    /// expert size, which comes from the header walk. Flash-Next's construction of the shared
    /// type, kept here so the seam module does not depend on the Flash-Next moe modules.
    pub(super) fn from_plan(plan: &TierPlan, layout: &SlotBankLayout) -> Result<Self, EngError> {
        let shape = plan.shape();
        if layout.record_bytes() as u64 != shape.expert_bytes {
            return Err(plan_error(format!(
                "the record is {} bytes, the tier plan counts {} per expert",
                layout.record_bytes(),
                shape.expert_bytes
            )));
        }
        Self::new(
            shape.layers,
            shape.experts,
            shape.top_k,
            plan.slots_per_layer(),
            layout.record_bytes(),
        )
    }
}

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

// ---------------------------------------------------------------------------------------------
// Host memory budget.
// ---------------------------------------------------------------------------------------------

/// `MemTotal` and `MemAvailable` of the machine, in bytes. `available` is the caller's minimum of
/// the host figure and every cgroup ancestor's remaining bytes (`flint::exl3_resident::HostBudget`
/// reads that on the CUDA path; see [`HostPinPlan::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostMemory {
    pub(super) total: u64,
    pub(super) available: u64,
}

impl HostMemory {
    /// Parses the text of `/proc/meminfo`. Refuses a missing field, a unit other than kB and an
    /// available figure above the total.
    pub(super) fn from_meminfo(text: &str) -> Result<Self, EngError> {
        let field = |name: &str| -> Result<u64, EngError> {
            let invalid = || plan_error(format!("/proc/meminfo has no valid {name} in kB"));
            let line = text
                .lines()
                .find_map(|line| line.strip_prefix(name))
                .and_then(|rest| rest.strip_prefix(':'))
                .ok_or_else(invalid)?;
            let mut words = line.split_whitespace();
            let kib: u64 = words
                .next()
                .and_then(|value| value.parse().ok())
                .ok_or_else(invalid)?;
            if words.next() != Some("kB") || words.next().is_some() {
                return Err(invalid());
            }
            kib.checked_mul(1024).ok_or_else(invalid)
        };
        let (total, available) = (field("MemTotal")?, field("MemAvailable")?);
        if total == 0 || available > total {
            return Err(plan_error("/proc/meminfo totals are inconsistent".into()));
        }
        Ok(Self { total, available })
    }

    /// The OS floor: a tenth of the RAM, at least 8 GiB. The same rule as Flint's
    /// `HostBudget::from_meminfo`, which keeps it private.
    pub(super) fn os_floor(&self) -> u64 {
        self.total.div_ceil(10).max(8 * GIB)
    }
}

/// The host terms besides the pinned store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostTerms {
    /// Process memory outside the pins: the BF16 embedding (1.18 GiB), the CUDA driver and process
    /// (about 1.5 GiB), small host buffers (0.1 GiB).
    pub(super) engine_heap: u64,
    /// Kept free so the plan does not ride the line (exllamav3 keeps 2 GiB as well).
    pub(super) margin: u64,
    /// The least page cache the n-gram table's random reads must keep. A floor, not a target: all
    /// RAM the plan leaves free is page cache. The default is 0, from measured cold n-gram row
    /// reads on the real file: 52 us median per row (read-only bench), 160-200 us per 16-row decode
    /// token (about 1% of a token, and the reads can start before the first layer) and 242 ms per
    /// 4096-token chunk (production reader).
    /// A warm page is a bonus, not a requirement, so the pinned store may take the RAM. At 0 the
    /// floor check never refuses; a nonzero floor needs a measured low hit rate on real traffic.
    /// The n-gram feed's own reserve constant, also 0, replaces this literal when that feed is
    /// merged.
    pub(super) ngram_cache_min: u64,
    /// Rows of n-gram embedding in flight at once in host memory: the staging of the row reader.
    /// A 4096-token chunk reads 65,536 rows and a double-buffered chunk worker holds two blocks.
    pub(super) ngram_staging_rows: u64,
    /// Metadata allowance charged to each of the two stream-bank slots besides the payload.
    pub(super) stream_slot_metadata: u64,
}

/// Host bytes per n-gram row in flight: 160 decoded F32 values plus the row id and its sort key.
/// 656 bytes. The n-gram feed's staging function gives the same figure and replaces this constant
/// when that feed is merged.
pub(super) const NGRAM_STAGING_BYTES_PER_ROW: u64 = 160 * 4 + 8 + 8;

impl HostTerms {
    /// The defaults for the 62 GiB host: 3 GiB heap, 2 GiB margin, no n-gram page-cache floor
    /// (measured, see the field), two 4096-token chunks of n-gram staging (86 MB).
    pub(super) const DEFAULT: Self = Self {
        engine_heap: 3 * GIB,
        margin: 2 * GIB,
        ngram_cache_min: 0,
        ngram_staging_rows: 2 * 65_536,
        stream_slot_metadata: MIB,
    };

    /// Host bytes the n-gram row reader holds in flight.
    pub(super) fn ngram_staging_bytes(&self) -> Option<u64> {
        self.ngram_staging_rows
            .checked_mul(NGRAM_STAGING_BYTES_PER_ROW)
    }
}

/// What the host must hold for a bank, and what that leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct HostPinPlan {
    pub(super) mode: HostMode,
    pub(super) places_per_layer: usize,
    /// Pinned bytes of the expert places of all layers.
    pub(super) store_bytes: u64,
    /// The two-slot scratch of the stream bank (prefill belt), charged now so the belt does not
    /// re-plan the host: both slots, each a full record plus metadata
    /// (`stream-bank-scratch-is-two-slot-total`: the argument of `StreamBankPlan` is the total).
    pub(super) scratch_bytes: u64,
    /// The n-gram row reader's host staging (`HostTerms::ngram_staging_rows` rows).
    pub(super) ngram_staging_bytes: u64,
    pub(super) os_floor: u64,
    /// Store, scratch, n-gram staging, heap and margin: what this process takes.
    pub(super) process_bytes: u64,
    /// RAM left for the page cache: `MemAvailable` (other processes' holds already out) minus what
    /// this plan needs now (store, scratch, n-gram staging, heap, margin and the OS floor).
    pub(super) page_cache_left: u64,
}

impl HostPinPlan {
    /// Sizes the pinned store from the machine and refuses (it never shrinks the store, and never
    /// changes the context) when
    /// - the process and the OS floor exceed `available`, or
    /// - the page cache left (`available` minus the need) is below `ngram_cache_min`.
    ///
    /// On the CUDA path the caller also runs `HostBudget::probe()?.admit_for(process_bytes, ..)`,
    /// which applies the cgroup ancestors (and refuses a cgroup-namespace root, so run on the host).
    pub(super) fn new(
        geometry: &BankGeometry,
        mode: HostMode,
        memory: &HostMemory,
        terms: &HostTerms,
    ) -> Result<Self, EngError> {
        let overflow = || plan_error("the host byte counts overflow".into());
        let places_per_layer = mode.places_per_layer(geometry);
        let store_bytes = (geometry.layers as u64)
            .checked_mul(places_per_layer as u64)
            .and_then(|places| places.checked_mul(geometry.record_bytes as u64))
            .ok_or_else(overflow)?;
        let scratch_bytes = 2 * (geometry.record_bytes as u64 + terms.stream_slot_metadata);
        let os_floor = memory.os_floor();
        let ngram_staging_bytes = terms.ngram_staging_bytes().ok_or_else(overflow)?;
        let process_bytes = store_bytes
            .checked_add(scratch_bytes)
            .and_then(|sum| sum.checked_add(ngram_staging_bytes))
            .and_then(|sum| sum.checked_add(terms.engine_heap))
            .and_then(|sum| sum.checked_add(terms.margin))
            .ok_or_else(overflow)?;
        let needed_now = process_bytes.checked_add(os_floor).ok_or_else(overflow)?;
        if needed_now > memory.available {
            return Err(plan_error(format!(
                "{store_bytes} pinned store bytes, {scratch_bytes} stream scratch, \
                 {ngram_staging_bytes} n-gram staging, {} heap and {} margin plus the {os_floor} \
                 byte OS floor need {needed_now} bytes, {} are available",
                terms.engine_heap, terms.margin, memory.available
            )));
        }
        // Cannot underflow: `needed_now <= available` above.
        let page_cache_left = memory.available - needed_now;
        if page_cache_left < terms.ngram_cache_min {
            return Err(plan_error(format!(
                "{page_cache_left} bytes of page cache are left of {} available; the n-gram \
                 page-cache floor is {}",
                memory.available, terms.ngram_cache_min
            )));
        }
        Ok(Self {
            mode,
            places_per_layer,
            store_bytes,
            scratch_bytes,
            ngram_staging_bytes,
            os_floor,
            process_bytes,
            page_cache_left,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The popularity profile.
// ---------------------------------------------------------------------------------------------

/// Parses a route-count profile (`flash-next-route-counts/1`): `#` header lines (the schema, the
/// layer and expert counts and the source), then `layer<TAB>expert<TAB>count` for the nonzero
/// cells. Returns `counts[layer][expert]`, the input of `TierPlan::hot_set`. Refuses another
/// schema or shape, a cell outside the shape, a cell twice and a malformed line.
pub(super) fn parse_route_counts(
    text: &str,
    layers: usize,
    experts: usize,
) -> Result<Vec<Vec<u64>>, EngError> {
    let bad = |what: String| plan_error(format!("route counts: {what}"));
    let (mut schema, mut seen_layers, mut seen_experts) = (false, None, None);
    let mut counts = vec![vec![0u64; experts]; layers];
    let mut seen = vec![vec![false; experts]; layers];
    for (number, line) in text.lines().enumerate() {
        if let Some(header) = line.strip_prefix('#') {
            let header = header.trim();
            if header == "schema=flash-next-route-counts/1" {
                schema = true;
            } else if let Some(value) = header.strip_prefix("layers=") {
                seen_layers = value.parse::<usize>().ok();
            } else if let Some(value) = header.strip_prefix("experts=") {
                seen_experts = value.parse::<usize>().ok();
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        if !schema || seen_layers != Some(layers) || seen_experts != Some(experts) {
            return Err(bad(format!(
                "line {} comes before a header for schema flash-next-route-counts/1 with {layers} \
                 layers and {experts} experts",
                number + 1
            )));
        }
        let mut cell = line.split('\t');
        let mut next = |what: &str| -> Result<u64, EngError> {
            cell.next()
                .and_then(|word| word.parse().ok())
                .ok_or_else(|| bad(format!("line {}: no {what}", number + 1)))
        };
        let (layer, expert, count) = (next("layer")?, next("expert")?, next("count")?);
        if cell.next().is_some() {
            return Err(bad(format!(
                "line {} has more than three fields",
                number + 1
            )));
        }
        let (layer, expert) = (layer as usize, expert as usize);
        let place = seen
            .get_mut(layer)
            .and_then(|row| row.get_mut(expert))
            .ok_or_else(|| {
                bad(format!(
                    "line {}: cell {layer}/{expert} is outside the shape",
                    number + 1
                ))
            })?;
        if std::mem::replace(place, true) {
            return Err(bad(format!(
                "line {}: cell {layer}/{expert} twice",
                number + 1
            )));
        }
        counts[layer][expert] = count;
    }
    if !schema || seen_layers != Some(layers) || seen_experts != Some(experts) {
        return Err(bad(format!(
            "no header for schema flash-next-route-counts/1 with {layers} layers and {experts} experts"
        )));
    }
    Ok(counts)
}

impl ExpertBankState<ReplaceOnMissLru> {
    /// The bank of `plan`, filled from the popularity profile `counts[layer][expert]` (the static
    /// fill of `TierPlan::hot_set`), then managed by replace-on-miss LRU.
    pub(super) fn new(
        plan: &TierPlan,
        layout: &SlotBankLayout,
        mode: HostMode,
        counts: &[Vec<u64>],
    ) -> Result<Self, EngError> {
        let geometry = BankGeometry::from_plan(plan, layout)?;
        let hot = plan.hot_set(counts)?;
        if hot.table.slots() != geometry.slots {
            return Err(bank_error(
                "the hot set and the bank disagree on the slot count".into(),
            ));
        }
        let rows: Vec<&[i32]> = (0..geometry.layers)
            .map(|layer| hot.table.row(layer))
            .collect();
        let policy = ReplaceOnMissLru::from_fill(geometry, &rows, counts, None)?;
        Self::with_policy(geometry, mode, policy)
    }
}

// ---------------------------------------------------------------------------------------------
// The filler rule for the second pass.
// ---------------------------------------------------------------------------------------------

/// The slot ids and weights of the missed-pick launch, after the copies: every missed pick runs
/// on its new slot with the router's weight — the missed picks ONLY, no +0.0 filler, so the
/// launch carries [`Classified::misses`] entries instead of `top_k`. The resident pass
/// (`classified.slot_ids`, `classified.weights`) carries the rest: the two passes' weights add
/// to the router's weights bit for bit, so the combine sees each pick exactly once.
///
/// Refuses a row the classifier flagged invalid, a router fault, lists of different lengths and a
/// missed expert that still has no slot.
pub(super) fn miss_pass_inputs(
    ids: &[i32],
    weights: &[f32],
    classified: &Classified,
    slot_row_after: &[i32],
) -> Result<(Vec<i32>, Vec<f32>), EngError> {
    if ids.len() != weights.len() || ids.len() != classified.slot_ids.len() {
        return Err(bank_error("the pick lists differ in length".into()));
    }
    if classified.word != 0 {
        return Err(bank_error(
            "the classifier flagged the route invalid".into(),
        ));
    }
    let mut slots = Vec::with_capacity(classified.misses.len());
    let mut out = Vec::with_capacity(classified.misses.len());
    for (&id, &weight) in ids.iter().zip(weights) {
        if id < 0 {
            return Err(bank_error("a router fault has no second pass".into()));
        }
        if classified.misses.iter().any(|&(missed, _)| missed == id) {
            let slot = slot_row_after
                .get(id as usize)
                .copied()
                .filter(|&slot| slot >= 0)
                .ok_or_else(|| {
                    bank_error(format!("missed expert {id} has no slot after the copy"))
                })?;
            slots.push(slot);
            out.push(weight);
        }
    }
    Ok((slots, out))
}

// ---------------------------------------------------------------------------------------------
// The record source.
// ---------------------------------------------------------------------------------------------

/// Where the bytes of an expert record come from (the pack through `SlotBankLayout`, see
/// `bank_source::PackRecordSource`). The device owner reads every record once at build.
pub(in crate::family::flash_next) trait RecordSource {
    /// Writes the record of `expert` in `layer` into `out` (exactly one record).
    fn record(&self, layer: usize, expert: u16, out: &mut [u8]) -> Result<(), EngError>;
}

// ---------------------------------------------------------------------------------------------
// The device owner.
// ---------------------------------------------------------------------------------------------

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(super) mod device {
    //! One device buffer of `slots` records and one pinned buffer of host places per layer. The
    //! pinned buffers come from `CudaContext::pinned_buf` (`cudaHostAlloc`, zeroed, 0.4 s per GiB),
    //! the path Flint's store works on under the 8 MiB `RLIMIT_MEMLOCK`; nothing here calls `mlock`.
    //! They are allocated once and held: dropping one synchronizes the device.

    use super::{
        BankGeometry, BankStats, ExpertBankState, HostPinPlan, PassCopies, RecordSource,
        ReplaceOnMissLru,
    };
    use crate::{
        EngError,
        kernels::cuda::{CudaContext, DeviceBuffer, DeviceRange, DeviceStream, PinnedBuf},
    };

    pub(in crate::family::flash_next) struct DeviceBank {
        state: ExpertBankState<ReplaceOnMissLru>,
        slots: Vec<DeviceBuffer>,
        store: Vec<PinnedBuf>,
        stream: DeviceStream,
    }

    impl DeviceBank {
        /// Allocates the buffers, writes the host places and fills every slot from `source`.
        /// Returns when the device holds the fill. `plan` is the host budget the caller admitted
        /// (`HostPinPlan::new`, plus Flint's cgroup-aware `HostBudget::admit_for` on the CUDA path):
        /// it must be for this bank's mode and place count, so a store is never pinned unchecked.
        pub(in crate::family::flash_next) fn build(
            ctx: &CudaContext,
            state: ExpertBankState<ReplaceOnMissLru>,
            source: &dyn RecordSource,
            plan: &HostPinPlan,
        ) -> Result<Self, EngError> {
            let geometry: BankGeometry = *state.geometry();
            let places = state.mode().places_per_layer(&geometry);
            let store_bytes = (geometry.layers as u64)
                .checked_mul(places as u64)
                .and_then(|places| places.checked_mul(geometry.record_bytes as u64));
            if plan.mode != state.mode()
                || plan.places_per_layer != places
                || Some(plan.store_bytes) != store_bytes
            {
                return Err(EngError::BadRequest(format!(
                    "Flash-Next expert bank: the host plan is for {:?} with {} places per layer and \
                     {} store bytes, the bank is {:?} with {places} places and {store_bytes:?} bytes",
                    plan.mode,
                    plan.places_per_layer,
                    plan.store_bytes,
                    state.mode()
                )));
            }
            let stream = ctx.stream()?;
            let mut slots = Vec::with_capacity(geometry.layers);
            let mut store = Vec::with_capacity(geometry.layers);
            let mut record = vec![0u8; geometry.record_bytes];
            for layer in 0..geometry.layers {
                slots.push(ctx.allocate(geometry.slots * geometry.record_bytes)?);
                let mut pinned = ctx.pinned_buf(places * geometry.record_bytes)?;
                for (place, expert) in state.places().occupied(layer) {
                    source.record(layer, expert, &mut record)?;
                    pinned.write(geometry.offset(place as usize), &record)?;
                }
                store.push(pinned);
                for slot in 0..geometry.slots {
                    let expert = state.expert_in_slot(layer, slot).ok_or_else(|| {
                        EngError::BadRequest("a slot of the filled bank is empty".into())
                    })?;
                    source.record(layer, expert, &mut record)?;
                    slots[layer].write_at(&stream, geometry.offset(slot), &record)?;
                }
            }
            stream.synchronize()?;
            Ok(Self {
                state,
                slots,
                store,
                stream,
            })
        }

        /// The host model, for tests only: it still answers after a failed copy, when it is ahead
        /// of the device. Production code reads [`Self::slot_row`], [`Self::stats`] and
        /// [`Self::geometry`].
        #[cfg(test)]
        pub(in crate::family::flash_next) fn state(&self) -> &ExpertBankState<ReplaceOnMissLru> {
            &self.state
        }

        /// The counters of the passes so far (hits, misses, write-backs, miss histogram).
        pub(in crate::family::flash_next) fn stats(&self) -> &BankStats {
            self.state.stats()
        }

        /// The sizes of the bank: layers, experts, top-k, slots per layer and the record size.
        pub(in crate::family::flash_next) fn geometry(&self) -> &BankGeometry {
            self.state.geometry()
        }

        /// Whether a failed copy left the device out of step with the host model. Every later
        /// `step`, `admit`, `slot_row`, `slots` and `read_place` is refused.
        pub(in crate::family::flash_next) fn is_poisoned(&self) -> bool {
            self.state.is_poisoned()
        }

        fn check_usable(&self) -> Result<(), EngError> {
            if self.state.is_poisoned() {
                return Err(EngError::BadRequest(
                    "Flash-Next expert bank: a failed copy left the device out of step with the \
                     host model; drop the bank and build a new one"
                        .into(),
                ));
            }
            Ok(())
        }

        /// Waits every copy `stream` may still hold on layer `layer`'s pinned store and
        /// clears its pending record: the prefill ring's window hands the stores back to the
        /// decode path this way.
        pub(in crate::family::flash_next) fn synchronize_store(
            &mut self,
            layer: usize,
            stream: &DeviceStream,
        ) -> Result<(), EngError> {
            let store = self.store.get_mut(layer).ok_or_else(|| {
                EngError::BadRequest(format!("layer {layer} is outside the bank"))
            })?;
            store.synchronize(stream)
        }

        /// The classifier's slot row of `layer` (upload it after every pass that moved a slot).
        /// Refused once the bank is poisoned, so a launch cannot classify against a table the
        /// device does not hold.
        pub(in crate::family::flash_next) fn slot_row(
            &self,
            layer: usize,
        ) -> Result<&[i32], EngError> {
            self.check_usable()?;
            match self.state.slot_row(layer) {
                [] => Err(EngError::BadRequest(format!(
                    "layer {layer} is outside the bank"
                ))),
                row => Ok(row),
            }
        }

        /// The bank buffer of `layer`: slot `s` starts at `s * record_bytes`. Refused once the
        /// bank is poisoned.
        pub(in crate::family::flash_next) fn slots(
            &self,
            layer: usize,
        ) -> Result<&DeviceBuffer, EngError> {
            self.check_usable()?;
            self.slots
                .get(layer)
                .ok_or_else(|| EngError::BadRequest(format!("layer {layer} is outside the bank")))
        }

        /// Queues the bytes of one expert place from the pinned store into `dst` on `stream` (the
        /// prefill ring's copy stream). The host model is not changed.
        /// The pinned host place of an expert's record, for the prefill ring's plan (a
        /// non-resident expert without a place cannot stream and rides the miss pass).
        pub(in crate::family::flash_next) fn place_of(
            &self,
            layer: usize,
            expert: u16,
        ) -> Option<u32> {
            self.state.places().place_of(layer, expert)
        }

        pub(in crate::family::flash_next) fn stream_place(
            &mut self,
            ctx: &CudaContext,
            stream: &DeviceStream,
            layer: usize,
            place: u32,
            dst: DeviceRange<'_>,
        ) -> Result<(), EngError> {
            self.check_usable()?;
            let geometry = self.state.geometry();
            let store = self.store.get_mut(layer).ok_or_else(|| {
                EngError::BadRequest(format!("layer {layer} is outside the bank"))
            })?;
            ctx.copy_from_pinned(stream, store, geometry.offset(place as usize), dst)
        }

        pub(in crate::family::flash_next) fn stream(&self) -> &DeviceStream {
            &self.stream
        }

        /// One decode layer: plans the copies for the router's picks (all of them, hits included),
        /// runs them and returns when they have finished. The only way to move a slot, so the
        /// device cannot lag the host model; if a copy fails the state is poisoned and every later
        /// step is refused. A `ctx` of another adapter than the one that built the bank fails in
        /// the first copy, after the host model has moved, so it poisons the bank too: pass the
        /// context of the build.
        ///
        /// The copies run on the bank's own stream, unordered against the caller's stream, so a
        /// caller with reads of the banked bytes still pending on its own stream must drain it
        /// before every `step` (`SplitMoe::run_rows` does; `run` relies on its caller's drains).
        pub(in crate::family::flash_next) fn step(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            picks: &[i32],
        ) -> Result<PassCopies, EngError> {
            self.check_usable()?;
            let copies = self.state.begin_pass(layer, picks)?;
            self.finish(ctx, copies)
        }

        /// Admits the experts a prefill chunk used (oldest use first) and runs the copies.
        pub(in crate::family::flash_next) fn admit(
            &mut self,
            ctx: &CudaContext,
            layer: usize,
            experts_by_use: &[u16],
        ) -> Result<PassCopies, EngError> {
            self.check_usable()?;
            let copies = self.state.admit(layer, experts_by_use)?;
            self.finish(ctx, copies)
        }

        fn finish(
            &mut self,
            ctx: &CudaContext,
            copies: PassCopies,
        ) -> Result<PassCopies, EngError> {
            match self.run_copies(ctx, &copies) {
                Ok(()) => Ok(copies),
                Err(error) => {
                    self.state.poison();
                    Err(error)
                }
            }
        }

        /// Runs the copies of `copies` on the bank's stream, in order, and returns when they have
        /// finished: per op the write-back (D2H) into the host place, then the H2D into the slot.
        fn run_copies(&mut self, ctx: &CudaContext, copies: &PassCopies) -> Result<(), EngError> {
            if copies.ops.is_empty() {
                return Ok(());
            }
            let geometry: BankGeometry = *self.state.geometry();
            let layer = copies.layer;
            let (Some(bank), Some(store)) = (self.slots.get(layer), self.store.get_mut(layer))
            else {
                return Err(EngError::BadRequest(format!(
                    "layer {layer} is outside the bank"
                )));
            };
            let record = geometry.record_bytes;
            for op in &copies.ops {
                let slot = bank.range(geometry.offset(usize::from(op.slot)), record)?;
                if let Some(demote) = op.demote {
                    ctx.copy_to_pinned(
                        &self.stream,
                        bank.range(geometry.offset(usize::from(op.slot)), record)?,
                        store,
                        geometry.offset(demote.dst_place as usize),
                    )?;
                }
                ctx.copy_from_pinned(
                    &self.stream,
                    store,
                    geometry.offset(op.src_place as usize),
                    slot,
                )?;
            }
            store.synchronize(&self.stream)
        }

        /// Reads one host place back (tests): the pinned buffer must have no copy outstanding.
        pub(in crate::family::flash_next) fn read_place(
            &self,
            layer: usize,
            place: u32,
            out: &mut [u8],
        ) -> Result<(), EngError> {
            self.check_usable()?;
            let geometry = self.state.geometry();
            self.store
                .get(layer)
                .ok_or_else(|| EngError::BadRequest(format!("layer {layer} is outside the bank")))?
                .read(geometry.offset(place as usize), out)
        }
    }
}
