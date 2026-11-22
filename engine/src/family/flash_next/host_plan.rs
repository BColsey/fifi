//! The host side of the Flash-Next expert bank: what the machine has (`HostMemory`), what the
//! process needs beside the pinned store (`HostTerms`), and the tier and byte plan that follow
//! (`HostPinPlan`).
//!
//! The context choice (`TierPlan`) and the start's own host plan both answer from this one
//! module, over the same host facts and the same terms, so a context the choice admits is a host
//! plan the start admits. Pure host arithmetic: no device, no clock, no I/O.

// Staged like the tier plan: the plans are read by the CUDA start and the tests.
#![cfg_attr(
    not(any(test, all(feature = "native-cuda", target_os = "linux"))),
    allow(dead_code, reason = "read by the CUDA start and the tests")
)]

use crate::{
    EngError,
    family::expert_bank::{
        BankGeometry, HostMode,
        bank_tier::{TierBudget, TierChoice, choose_tier},
        plan_error,
    },
};

/// The labels the Flash-Next starts read the host and the card under.
pub(crate) const HOST_LABELS: (&str, &str) = ("Flash-Next expert bank", "Flash-Next");
const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;

pub(in crate::family::flash_next) use crate::family::mem::HostMemory;

/// The host side of one plan: the machine and the terms the process needs beside the store. The
/// context choice and the start's host plan are built from one frame, so they cannot read the
/// host differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct HostFrame {
    pub(in crate::family::flash_next) memory: HostMemory,
    pub(in crate::family::flash_next) terms: HostTerms,
}

impl HostFrame {
    pub(in crate::family::flash_next) fn new(memory: HostMemory, terms: HostTerms) -> Self {
        Self { memory, terms }
    }

    /// The frame of a machine with the default terms (no PLE bytes, victim cache or prefetch).
    pub(in crate::family::flash_next) fn with_default_terms(memory: HostMemory) -> Self {
        Self::new(memory, HostTerms::DEFAULT)
    }
}

/// The host terms besides the pinned store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct HostTerms {
    /// Process memory outside the pins: the BF16 embedding (1.18 GiB), the CUDA driver and process
    /// (about 1.5 GiB), small host buffers (0.1 GiB). The start folds the two PLE projection
    /// matrices' F32 bytes into this figure, so the plan counts every byte the guard admits.
    pub(in crate::family::flash_next) engine_heap: u64,
    /// Kept free beyond the OS floor so the plan does not ride the line: the shared host margin.
    pub(in crate::family::flash_next) margin: u64,
    /// The least page cache the n-gram table's random reads must keep. A floor, not a target: all
    /// RAM the plan leaves free is page cache. The default is 0, from measured cold n-gram row
    /// reads on the real file: 52 us median per row (read-only bench), 160-200 us per 16-row decode
    /// token (about 1% of a token, and the reads can start before the first layer) and 242 ms per
    /// 4096-token chunk (production reader).
    /// A warm page is a bonus, not a requirement, so the pinned store may take the RAM. At 0 the
    /// floor check never refuses; a nonzero floor needs a measured low hit rate on real traffic.
    /// The n-gram feed's own reserve constant, also 0, replaces this literal when that feed is
    /// merged.
    pub(in crate::family::flash_next) ngram_cache_min: u64,
    /// Rows of n-gram embedding in flight at once in host memory: the staging of the row reader.
    /// A 4096-token chunk reads 65,536 rows and a double-buffered chunk worker holds two blocks.
    pub(in crate::family::flash_next) ngram_staging_rows: u64,
    /// Metadata allowance charged to each of the two stream-bank slots besides the payload.
    pub(in crate::family::flash_next) stream_slot_metadata: u64,
}

/// Host bytes per n-gram row in flight: 160 decoded F32 values plus the row id and its sort key.
/// 656 bytes. The n-gram feed's staging function gives the same figure and replaces this constant
/// when that feed is merged.
pub(in crate::family::flash_next) const NGRAM_STAGING_BYTES_PER_ROW: u64 = 160 * 4 + 8 + 8;

impl HostTerms {
    /// The defaults for the 62 GiB host: 3 GiB heap, the shared host margin (2 GiB), no n-gram page-cache floor
    /// (measured, see the field), two 4096-token chunks of n-gram staging (86 MB).
    pub(in crate::family::flash_next) const DEFAULT: Self = Self {
        engine_heap: 3 * GIB,
        margin: crate::family::mem::HOST_MARGIN_BYTES,
        ngram_cache_min: 0,
        ngram_staging_rows: 2 * 65_536,
        stream_slot_metadata: MIB,
    };

    /// Host bytes the n-gram row reader holds in flight.
    pub(in crate::family::flash_next) fn ngram_staging_bytes(&self) -> Option<u64> {
        self.ngram_staging_rows
            .checked_mul(NGRAM_STAGING_BYTES_PER_ROW)
    }
}

/// What the host must hold for a bank, and what that leaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::family::flash_next) struct HostPinPlan {
    pub(in crate::family::flash_next) mode: HostMode,
    pub(in crate::family::flash_next) places_per_layer: usize,
    /// Pinned bytes of the expert places of all layers.
    pub(in crate::family::flash_next) store_bytes: u64,
    /// The two-slot scratch of the stream bank (prefill belt), charged now so the belt does not
    /// re-plan the host: both slots, each a full record plus metadata
    /// (`stream-bank-scratch-is-two-slot-total`: the argument of `StreamBankPlan` is the total).
    pub(in crate::family::flash_next) scratch_bytes: u64,
    /// The n-gram row reader's host staging (`HostTerms::ngram_staging_rows` rows).
    pub(in crate::family::flash_next) ngram_staging_bytes: u64,
    pub(in crate::family::flash_next) os_floor: u64,
    /// Store, scratch, n-gram staging, heap and margin: what this process takes.
    pub(in crate::family::flash_next) process_bytes: u64,
    /// RAM left for the page cache: `MemAvailable` (other processes' holds already out) minus what
    /// this plan needs now (store, scratch, n-gram staging, heap, margin and the OS floor).
    pub(in crate::family::flash_next) page_cache_left: u64,
}

impl HostPinPlan {
    /// Sizes the pinned store from the machine and refuses (it never shrinks the store, and never
    /// changes the context) when
    /// - the process and the OS floor exceed `available`, or
    /// - the page cache left (`available` minus the need) is below `ngram_cache_min`.
    ///
    /// On the CUDA path the plan and the guard read one admission snapshot: the start reads
    /// `HostFacts` once, builds `HostMemory` from it (the budget and what it keeps back) and
    /// guards the plan's `process_bytes` with `HostFacts::admit_process_bytes` on the same facts.
    /// A cgroup tree that cannot be verified reads as the bare `MemAvailable` figure.
    pub(in crate::family::flash_next) fn new(
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

    /// The disk rung's plan: no pinned store at all — `ring_bytes` (the staging ring's backing)
    /// replaces the store term while everything else charges exactly as `new` does, so the
    /// ladder's disk arm (`ring + overhead + cache_floor <= available`) and this admission stay
    /// one arithmetic: a choice the ladder admits is a plan this constructor admits. Below that
    /// floor the refusal carries this plan's own byte counts — the honest minimum tier.
    pub(in crate::family::flash_next) fn new_disk(
        geometry: &BankGeometry,
        memory: &HostMemory,
        terms: &HostTerms,
        ring_bytes: u64,
    ) -> Result<Self, EngError> {
        let overflow = || plan_error("the host byte counts overflow".into());
        let scratch_bytes = 2 * (geometry.record_bytes as u64 + terms.stream_slot_metadata);
        let os_floor = memory.os_floor();
        let ngram_staging_bytes = terms.ngram_staging_bytes().ok_or_else(overflow)?;
        let process_bytes = ring_bytes
            .checked_add(scratch_bytes)
            .and_then(|sum| sum.checked_add(ngram_staging_bytes))
            .and_then(|sum| sum.checked_add(terms.engine_heap))
            .and_then(|sum| sum.checked_add(terms.margin))
            .ok_or_else(overflow)?;
        let needed_now = process_bytes.checked_add(os_floor).ok_or_else(overflow)?;
        if needed_now > memory.available {
            return Err(plan_error(format!(
                "{ring_bytes} staging-ring bytes, {scratch_bytes} stream scratch, \
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
            mode: HostMode::Disk,
            places_per_layer: 0,
            store_bytes: 0,
            scratch_bytes,
            ngram_staging_bytes,
            os_floor,
            process_bytes,
            page_cache_left,
        })
    }

    /// Maps this plan's cost model onto the shared store-tier ladder and chooses the tier from the
    /// one host budget: inclusive while it fits, then the exclusive store at today's one spare
    /// place per layer, then the disk tier (`ring_bytes` is the staging ring's backing, the disk
    /// tier's own minimum beside the fixed overhead). `new` stays the cost model of one mode;
    /// this is the choice among modes, so an `auto` start never changes what today's `auto`
    /// carries (the ladder's exclusive rung is exactly today's exclusive store) and only adds the
    /// disk tier below today's refusal floor.
    pub(in crate::family::flash_next) fn choose_store_tier(
        geometry: &BankGeometry,
        memory: &HostMemory,
        terms: &HostTerms,
        ring_bytes: u64,
    ) -> Result<TierChoice, EngError> {
        let overflow = || plan_error("the host byte counts overflow".into());
        let record = geometry.record_bytes as u64;
        // Everything but the store: the same sum `new` charges beside the pins, so the ladder's
        // arithmetic and the plan's stay one arithmetic.
        let slot = record
            .checked_add(terms.stream_slot_metadata)
            .ok_or_else(overflow)?;
        let staging = terms.ngram_staging_bytes().ok_or_else(overflow)?;
        let overhead_bytes = slot
            .checked_mul(2)
            .and_then(|scratch| scratch.checked_add(staging))
            .and_then(|sum| sum.checked_add(terms.engine_heap))
            .and_then(|sum| sum.checked_add(terms.margin))
            .and_then(|sum| sum.checked_add(memory.os_floor()))
            .ok_or_else(overflow)?;
        choose_tier(&TierBudget {
            record_bytes: record,
            layers: geometry.layers,
            experts: geometry.experts,
            slots: geometry.slots,
            // Today's exclusive store: the cold experts plus one spare place per layer, and the
            // policy cannot go below one.
            max_spare_places: 1,
            min_spare_places: 1,
            overhead_bytes,
            cache_floor: terms.ngram_cache_min,
            ring_bytes,
            available: memory.available,
            card: None,
        })
        .map_err(|ladder| {
            // Below the minimum the ladder's own text used to replace the plan's counts; carry
            // them so the refusal says no less than main's pre-ladder one did.
            match Self::new(geometry, HostMode::Exclusive, memory, terms) {
                Err(refused) => EngError::MemoryPlan(format!(
                    "{}; the host budget cannot carry the exclusive store either: {}",
                    ladder.describe(),
                    refused.describe()
                )),
                // The exclusive store fitting means the ladder did not refuse on the budget
                // (overflow arithmetic); keep the ladder's own text.
                Ok(_) => ladder,
            }
        })
    }

    /// The start's tier choice from `FIFI_FLASH_NEXT_BANK_MODE` (`None` = unset): `auto` takes
    /// the ladder, `inclusive`, `exclusive` and `disk` force their tier, and any other value is
    /// refused, never a silent fallback. The start maps the chosen tier onto `HostPinPlan::new`
    /// (disk: `new_disk`), so a forced tier that does not fit still refuses with the plan's own
    /// byte counts.
    pub(in crate::family::flash_next) fn bank_mode_tier(
        mode: Option<&str>,
        geometry: &BankGeometry,
        memory: &HostMemory,
        terms: &HostTerms,
        ring_bytes: u64,
    ) -> Result<TierChoice, EngError> {
        match mode {
            None | Some("") | Some("auto") => {
                Self::choose_store_tier(geometry, memory, terms, ring_bytes)
            }
            Some("inclusive") => Ok(TierChoice::Inclusive),
            Some("exclusive") => Ok(TierChoice::Exclusive { spare_places: 1 }),
            Some("disk") => Ok(TierChoice::Disk),
            Some(other) => Err(EngError::BadRequest(format!(
                "FIFI_FLASH_NEXT_BANK_MODE={other}: only auto, inclusive, exclusive and disk are \
                 valid"
            ))),
        }
    }
}
