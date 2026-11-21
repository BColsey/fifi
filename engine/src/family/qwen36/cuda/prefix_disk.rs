//! The step's disk tier for saved prefixes. This is a child of step, so device owners stay private.
//!
//! A pinned save is copied to the disk cache behind the request: the engine thread reads the
//! held pinned slot in chunks while it has nothing else to do (between decode tokens, between
//! prefill pieces, and while idle), and a writer thread hashes, checksums and stages them. A
//! pinned-RAM miss looks the prompt up on disk first; a hit is read into a free pinned slot,
//! verified in full, stored as an ordinary entry, and restored through the existing pinned path,
//! so the disk tier adds no second restore code. Every failure here is a logged miss.
use super::super::{
    prefix::{CudaPrefillPath, CudaSavedPrefix, HOST_HEADROOM, entry_bytes},
    prefix_shape::entry_slab,
    state::{LAYOUT_TAG, pinned::PinnedSnapshot},
};
use super::QwenResidentStep;
use crate::{
    EngError,
    family::{
        host_admit::HostAdmission,
        kv_disk::{
            Capture, DiskTier, FamilyContract, Held, Hit, KvDisk, LoadIdentity, MIN_POSITIONS,
            PUMP_EMIT_BYTES, PUMP_IDLE_BYTES, PUMP_PREFILL_BYTES, Reason, Resolved, SlotBytes,
            Stored, drain_bound, host_memory_is_short, hurry_flag, restore_report,
            retry_while_removed, spill_candidates, spill_picks, start_line, stop_picks,
        },
        session::SaveKind,
    },
    kernels::cuda::PinnedBuf,
    sched::job::{PrefixSource, PrefixTier},
};
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

/// What the shared disk tier calls this engine in its log lines.
const TIER_LABEL: &str = "q36 cuda";

/// What a Qwen3.6 single-card CUDA load fixes about the entries it writes.
const CONTRACT: FamilyContract = FamilyContract {
    adapter: "qwen36.cuda",
    backend: "cuda.single-card",
    snapshot_schema: "qwen36.compact-state.v1",
    payload_abi: "delta-f32.kv-compact.v1",
    position_policy: "absolute-rope-from-zero",
    render_mode: "chat",
};

/// Lookups one restore may spend on entries that turn out damaged.
const PROMOTE_ATTEMPTS: usize = 3;
/// The least time between two logged background failures.
const BACKGROUND_ERROR_GAP: Duration = Duration::from_secs(30);
impl SlotBytes for RefCell<PinnedBuf> {
    fn read(&self, at: usize, out: &mut [u8]) -> Result<(), EngError> {
        self.borrow().read(at, out)
    }
}

/// A stored pinned entry as a spill: `None` for a pageable entry or one too short to repay a read.
fn held_of(
    layout: &super::super::state_layout::StateLayout,
    key: &[u32],
    kind: SaveKind,
    saved: &CudaSavedPrefix,
    reason: Reason,
) -> Option<Held> {
    let CudaSavedPrefix::Pinned { state, .. } = saved else {
        return None;
    };
    let positions = state.tokens();
    if positions < MIN_POSITIONS {
        return None;
    }
    let shape = entry_slab(layout, positions).ok()?;
    Some(Held {
        reason,
        slot: state.hold(),
        key: key.to_vec(),
        capture: capture_of(kind),
        positions,
        bytes: shape.object_bytes().iter().sum(),
    })
}

fn capture_of(kind: SaveKind) -> Capture {
    match kind {
        SaveKind::Tip => Capture::Tip,
        SaveKind::Root | SaveKind::Periodic | SaveKind::Block => Capture::Periodic,
    }
}

fn kind_of(capture: Capture) -> SaveKind {
    match capture {
        Capture::Tip => SaveKind::Tip,
        Capture::Periodic => SaveKind::Periodic,
    }
}

/// What a disk entry of this step was built under: the checkpoint, tokenizer and template digests
/// and the state layout. The KV form (fp16 or int8) is part of the layout and of the numeric path,
/// so an entry written under one form is never found by a load of the other.
fn load_identity_of(
    identity: &crate::family::session::PrefixIdentity,
    layout: &super::super::state_layout::StateLayout,
) -> LoadIdentity {
    let per_token: usize = layout
        .kv
        .prefix_ranges(0, 1)
        .map_or(0, |ranges| ranges.iter().map(|range| range.bytes).sum());
    LoadIdentity {
        weights: identity.weights.to_string(),
        tokenizer: identity.tokenizer.to_string(),
        template: identity.template.to_string(),
        numeric_path: format!("{:?}", CudaPrefillPath::Steps(layout.kv.mode)),
        family: CONTRACT,
        layout: format!(
            "{LAYOUT_TAG}; delta {} layers rec {} conv {}; kv {} layers {:?} {per_token} B/token/layer",
            layout.delta.layers,
            layout.delta.rec_values,
            layout.delta.conv_values,
            layout.kv.layers,
            layout.kv.mode,
        ),
    }
}

impl QwenResidentStep<'_> {
    fn load_identity(&self) -> LoadIdentity {
        load_identity_of(&self.prefix_identity, self.state.layout())
    }

    /// Opens the disk tier at `resolved`, or logs why the tier stays off. It runs after the pinned
    /// pool exists: the tier saves and restores through pinned slots. A directory that cannot
    /// be opened keeps the tier off and never fails the start.
    pub(crate) fn enable_disk_tier(&mut self, resolved: &Resolved, origin: &str) {
        if self.prefix_pins.is_none() {
            eprintln!("q36 cuda: kv disk tier off: the pinned prefix pool is off");
            return;
        }
        if !self.prefix_identity.is_persistable() {
            eprintln!(
                "q36 cuda: kv disk tier off: the load has no weights, tokenizer and template digest to name its entries"
            );
            return;
        }
        let retained = self
            .prefix_pins
            .as_ref()
            .map_or(1, |pool| pool.retained_entries());
        match KvDisk::open(&resolved.dir, resolved.quota_bytes, &self.load_identity()) {
            Ok((cache, report)) => {
                eprintln!(
                    "{}",
                    start_line(TIER_LABEL, resolved, origin, &report, retained)
                );
                self.disk = Some(DiskTier::new(TIER_LABEL, cache, retained));
            }
            Err(error) => {
                eprintln!(
                    "q36 cuda: kv disk tier off: {} cannot be used: {error}",
                    resolved.dir.display()
                );
            }
        }
    }

    pub(crate) fn disk_tier_enabled(&self) -> bool {
        self.disk.is_some()
    }

    /// Whether the host's memory is short: what the start plan read to size the pinned pool (the
    /// cgroup-clamped available budget) is under the headroom the plan keeps plus one more entry.
    fn host_memory_short(&self) -> bool {
        let entry = self
            .prefix_pins
            .as_ref()
            .map_or(0, |pool| pool.bytes() / (pool.retained_entries() + 1));
        let available = HostAdmission::probe("Qwen36 kv disk", "Qwen36")
            .ok()
            .map(|admission| admission.available_budget());
        host_memory_is_short(available, HOST_HEADROOM as u64, entry as u64)
    }

    /// After a save publishes: when the store holds as many entries as the watermark and the
    /// host's memory is short, queue the least recently used ones that the next saves would push
    /// out, so their eviction never waits for a write. With memory to spare, or below the
    /// watermark, the disk is not touched.
    pub(super) fn spill_over_watermark(&mut self) {
        if !self.prefix_identity.is_persistable() {
            return;
        }
        let stored = self.prefix.entries_len() + self.prefix.roots_len();
        let want = match self.disk.as_ref() {
            Some(tier) => spill_candidates(stored, tier.retained),
            None => return,
        };
        if want == 0 || !self.host_memory_short() {
            return;
        }
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        let layout = *self.state.layout();
        let entries: Vec<_> = self.prefix.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        let offers: Vec<Held> = spill_picks(&view, want)
            .into_iter()
            .filter_map(|at| {
                let (key, kind, saved) = entries[at];
                held_of(&layout, key, kind, saved, Reason::Watermark)
            })
            .collect();
        for offer in offers {
            tier.enqueue(offer);
        }
    }

    /// Before a save takes a slot: when none is free because spills hold them, drops the spills
    /// (counted) instead of making the save fail or wait.
    pub(super) fn free_slot_for_save(&mut self) {
        let Some(pool) = self.prefix_pins.as_ref() else {
            return;
        };
        let free = pool.free_slots();
        if let Some(tier) = self.disk.as_mut() {
            tier.drop_spills_for_slots(free, "a save");
        }
    }

    /// Logs a failure of the engine's background work, at most once in `BACKGROUND_ERROR_GAP`: the
    /// worker polls every few milliseconds, and a state that stays failed would otherwise fill
    /// the log.
    pub(crate) fn note_background_error(&mut self, when: &str, error: &EngError) {
        let now = Instant::now();
        if self
            .background_error_at
            .is_some_and(|at| now.duration_since(at) < BACKGROUND_ERROR_GAP)
        {
            return;
        }
        self.background_error_at = Some(now);
        eprintln!("q36 cuda: deferred prefix save failed {when}: {error:?}");
    }

    pub(crate) fn pump_disk_prefill(&mut self) {
        self.pump_disk(PUMP_PREFILL_BYTES);
    }

    pub(crate) fn pump_disk_emit(&mut self) {
        self.pump_disk(PUMP_EMIT_BYTES);
    }

    pub(crate) fn pump_disk_idle(&mut self) {
        self.pump_disk(PUMP_IDLE_BYTES);
    }

    /// Whether the disk tier has work the engine thread should advance while idle.
    pub(crate) fn disk_pending(&self) -> bool {
        self.disk.as_ref().is_some_and(DiskTier::pending)
    }

    /// Copies up to `budget` bytes of pending saves to the writer and logs what finished.
    fn pump_disk(&mut self, budget: usize) {
        let layout = *self.state.layout();
        if let Some(tier) = self.disk.as_mut() {
            tier.pump(&|positions| entry_slab(&layout, positions), budget);
        }
    }

    /// A graceful stop: writes the tips that exist only in memory, newest first, so a restart
    /// restores them. What is not a tip (a periodic or block entry waiting, or being copied) is
    /// left out. The wait is bounded by the bytes the tips still have to move and the measured
    /// write rate, and ends at once on the process's second stop signal; one line says the bound
    /// and one says how it ended.
    pub(crate) fn drain_disk(&mut self) {
        if self.disk.is_none() {
            return;
        }
        // The live tips in memory, newest first.
        let layout = *self.state.layout();
        let mut tips: Vec<Held> = Vec::new();
        if self.prefix_identity.is_persistable() {
            let entries: Vec<_> = self.prefix.entries_lru().collect();
            let view: Vec<Stored<'_>> = entries
                .iter()
                .map(|&(key, kind, _)| Stored {
                    key,
                    tip: kind == SaveKind::Tip,
                })
                .collect();
            tips = stop_picks(&view)
                .into_iter()
                .filter_map(|at| {
                    let (key, kind, saved) = entries[at];
                    held_of(&layout, key, kind, saved, Reason::Stop)
                })
                .collect();
        }
        let left_in_memory = self.prefix.entries_len().saturating_sub(tips.len());
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        tier.drain(
            tips,
            left_in_memory,
            &|positions| entry_slab(&layout, positions),
            hurry_flag(),
            &drain_bound,
        );
    }

    /// Before a restore: when the disk holds a longer stored prefix of `prompt` than pinned
    /// memory does, reads it (verified in full) into a free pinned slot and stores it as an
    /// ordinary entry, so the restore that follows finds it. A miss changes nothing.
    pub(crate) fn promote_from_disk(&mut self, prompt: &[u32]) {
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        tier.hit = None;
        // An entry that fails its integrity check is removed, so the next lookup names the
        // next longest stored prefix: a damaged tip falls back to its block cut, not to a
        // cold prefill.
        retry_while_removed(PROMOTE_ATTEMPTS, || self.promote_once(prompt));
    }

    /// One lookup and read. Returns true when the entry failed its integrity check and was
    /// removed, so another candidate may serve the prompt.
    fn promote_once(&mut self, prompt: &[u32]) -> bool {
        let Some(tier) = self.disk.as_mut() else {
            return false;
        };
        let path = (
            CudaPrefillPath::Steps(self.state.layout().kv.mode),
            self.prefix_identity.clone(),
        );
        let Some(candidate) = tier.cache.lookup(prompt) else {
            return false;
        };
        if self
            .prefix
            .longest_candidate_len(prompt, &path)
            .is_some_and(|len| len >= candidate.key_len)
        {
            return false;
        }
        let began = Instant::now();
        let positions = candidate.key_len - 1;
        let layout = *self.state.layout();
        if positions >= layout.kv.capacity {
            eprintln!(
                "q36 cuda: kv disk entry of {positions} positions does not fit the {} position context",
                layout.kv.capacity
            );
            return false;
        }
        let shape = match entry_slab(&layout, positions) {
            Ok(shape) => shape,
            Err(error) => {
                eprintln!("q36 cuda: kv disk restore skipped: {error:?}");
                return false;
            }
        };
        let Some(pool) = self.prefix_pins.as_mut() else {
            return false;
        };
        tier.drop_spills_for_slots(pool.free_slots(), "a restore");
        let landing = match pool.begin_landing(&self.stream, &self.state, positions) {
            Ok(landing) => landing,
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("q36 cuda: kv disk restore skipped: no free pinned slot ({error:?})");
                return false;
            }
        };
        let stats = match tier.read_into(&candidate, prompt, &shape, &mut |slab, part| {
            landing.write(slab, part)
        }) {
            Ok(stats) => stats,
            Err(removed) => return removed,
        };
        let snapshot: PinnedSnapshot = match pool.finish_landing(landing) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("q36 cuda: kv disk restore dropped: {error:?}");
                return false;
            }
        };
        let Some(&pending) = prompt[..candidate.key_len].last() else {
            return false;
        };
        let charge = entry_bytes(&layout, positions).unwrap_or(0);
        let stored = self.prefix.save_loaded(
            &prompt[..candidate.key_len],
            pending,
            path,
            kind_of(candidate.capture),
            CudaSavedPrefix::Pinned {
                state: snapshot,
                payload_bytes: charge,
            },
        );
        match stored {
            Ok(true) => {
                tier.counters.restored += 1;
                tier.hit = Some(Hit {
                    key_len: candidate.key_len,
                    bytes: stats.bytes,
                    ms: stats.ms,
                });
                eprintln!(
                    "q36 cuda: kv disk read {} positions, {} B, {:.1} ms (verified), {:.1} ms with the lookup",
                    positions,
                    stats.bytes,
                    stats.ms,
                    began.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(false) => {
                tier.counters.missed += 1;
                eprintln!(
                    "q36 cuda: kv disk entry read but the store refused it; the prompt is prefilled"
                );
            }
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("q36 cuda: kv disk entry could not be stored: {error:?}");
            }
        }
        false
    }

    /// The tier facts of the restore that just covered `covered` tokens. A restore served from
    /// pinned memory names its source; one that promoted a disk entry names the disk and what
    /// the read cost.
    pub(super) fn restore_tier(&mut self, covered: usize) -> PrefixTier {
        let source = self.restore_source.take();
        restore_report(self.disk.as_mut(), covered, source, PrefixSource::Pinned)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{
        dims::{KvMode, StepDims},
        start::step_options,
        state_layout::StateLayout,
    };
    use super::*;
    use crate::family::{
        QWEN3_6_35B_A3B, gguf_model::Inventory, qwen36::host::HostShape, session::PrefixIdentity,
    };

    fn identity() -> PrefixIdentity {
        PrefixIdentity {
            weights: "weights".into(),
            tokenizer: "tokenizer".into(),
            template: "template".into(),
        }
    }

    fn namespace_under(mode: KvMode, context: u32) -> [u8; 32] {
        let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B).unwrap();
        let shape = HostShape::from_spec(
            &QWEN3_6_35B_A3B,
            crate::family::qwen36::settings::ROTARY_DIM,
        )
        .unwrap();
        let dims = StepDims::new(&shape, &inventory).unwrap();
        let options = step_options(
            context,
            crate::family::project::cuda::WeightLayout::Native,
            mode,
        )
        .unwrap();
        let layout = StateLayout::new(&dims, &options).unwrap();
        let load = load_identity_of(&identity(), &layout);
        assert!(load.is_complete());
        load.namespace()
    }

    #[test]
    fn an_fp16_entry_is_never_found_by_an_int8_cache() {
        let fp16 = namespace_under(KvMode::Fp16, 32_768);
        let int8 = namespace_under(KvMode::Int8G32, 32_768);
        assert_ne!(fp16, int8, "the two cache forms share a namespace");
        // The same form at the same size is the same namespace (an entry survives a restart), and
        // the capacity does not enter it (an entry survives a different planned context).
        assert_eq!(fp16, namespace_under(KvMode::Fp16, 32_768));
        assert_eq!(int8, namespace_under(KvMode::Int8G32, 65_536));
    }
}
