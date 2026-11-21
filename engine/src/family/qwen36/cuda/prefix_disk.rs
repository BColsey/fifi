//! The step's disk tier for saved prefixes. This is a child of step, so device owners stay private.
//!
//! A pinned save is copied to the disk cache behind the request: the engine thread reads the
//! held pinned slot in chunks while it has nothing else to do (between decode tokens, between
//! prefill pieces, and while idle), and a writer thread hashes, checksums and stages them. A
//! pinned-RAM miss looks the prompt up on disk first; a hit is read into a free pinned slot,
//! verified in full, stored as an ordinary entry, and restored through the existing pinned path,
//! so the disk tier adds no second restore code. Every failure here is a logged miss.
use super::super::{
    prefix::{CudaPrefillPath, CudaSavedPrefix, entry_bytes},
    prefix_shape::entry_slab,
    state::{
        LAYOUT_TAG, ResidentState,
        pinned::{PinnedLanding, PrefixPinPool},
    },
    state_layout::StateLayout,
};
use super::QwenResidentStep;
use crate::{
    EngError,
    family::{
        kv_disk::{
            DiskTier, FamilyContract, HostFamily, HostStoreDisk, LoadIdentity, PUMP_EMIT_BYTES,
            PUMP_IDLE_BYTES, PUMP_PREFILL_BYTES, Resolved, SlabLayout, SlotBytes, SnapshotView,
            restore_tier,
        },
        session::PrefixIdentity,
    },
    kernels::cuda::{DeviceStream, PinnedBuf},
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

/// The least time between two logged background failures.
const BACKGROUND_ERROR_GAP: Duration = Duration::from_secs(30);
impl SlotBytes for RefCell<PinnedBuf> {
    fn read(&self, at: usize, out: &mut [u8]) -> Result<(), EngError> {
        self.borrow().read(at, out)
    }
}

/// The Qwen3.6 engine as the host-store adapter sees it: its entries are pinned slots of the
/// prefix pool, and a disk read lands in a free slot.
struct QwenFamily<'a> {
    layout: StateLayout,
    /// The pool a disk read takes a slot from; absent when only the saves are in play.
    pool: Option<&'a mut PrefixPinPool>,
    stream: &'a DeviceStream,
    state: &'a ResidentState,
}

impl HostFamily for QwenFamily<'_> {
    type Snapshot = CudaSavedPrefix;
    type Path = (CudaPrefillPath, PrefixIdentity);
    type Landing = PinnedLanding;

    /// A pinned entry offers its slot; a pageable one stays in memory only.
    fn view(&self, saved: &CudaSavedPrefix) -> Option<SnapshotView> {
        let CudaSavedPrefix::Pinned { state, .. } = saved else {
            return None;
        };
        let positions = state.tokens();
        let shape = entry_slab(&self.layout, positions).ok()?;
        Some(SnapshotView {
            positions,
            bytes: shape.object_bytes().iter().sum(),
            slot: state.hold(),
        })
    }

    fn layout(&self, positions: usize) -> Result<SlabLayout, EngError> {
        if positions >= self.layout.kv.capacity {
            return Err(EngError::ContextLimit(format!(
                "a kv disk entry of {positions} positions does not fit the {} position context",
                self.layout.kv.capacity
            )));
        }
        entry_slab(&self.layout, positions)
    }

    fn begin_landing(
        &mut self,
        tier: &mut DiskTier,
        positions: usize,
        _layout: &SlabLayout,
    ) -> Result<PinnedLanding, EngError> {
        let pool = self
            .pool
            .as_deref_mut()
            .ok_or_else(|| EngError::MemoryPlan("no pinned prefix pool".into()))?;
        // The spills that hold slots give them up to a restore.
        tier.drop_spills_for_slots(pool.free_slots(), "a restore");
        pool.begin_landing(self.stream, self.state, positions)
    }

    fn place(&self, landing: &mut PinnedLanding, slab: usize, part: &[u8]) -> Result<(), EngError> {
        landing.write(slab, part)
    }

    fn finish_landing(
        &mut self,
        landing: PinnedLanding,
        positions: usize,
    ) -> Result<CudaSavedPrefix, EngError> {
        let pool = self
            .pool
            .as_deref_mut()
            .ok_or_else(|| EngError::MemoryPlan("no pinned prefix pool".into()))?;
        let state = pool.finish_landing(landing)?;
        Ok(CudaSavedPrefix::Pinned {
            state,
            payload_bytes: entry_bytes(&self.layout, positions).unwrap_or(0),
        })
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
        self.disk = HostStoreDisk::open(
            TIER_LABEL,
            resolved,
            origin,
            &self.load_identity(),
            retained,
        );
    }

    pub(crate) fn disk_tier_enabled(&self) -> bool {
        self.disk.is_some()
    }

    /// After a save publishes: when the store holds as many entries as the watermark and the
    /// host's memory is short, queue the least recently used ones that the next saves would push
    /// out, so their eviction never waits for a write. With memory to spare, or below the
    /// watermark, the disk is not touched.
    pub(super) fn spill_over_watermark(&mut self) {
        if !self.prefix_identity.is_persistable() {
            return;
        }
        let layout = *self.state.layout();
        if let Some(disk) = self.disk.as_mut() {
            let family = QwenFamily {
                layout,
                pool: None,
                stream: &self.stream,
                state: &self.state,
            };
            disk.spill_after_save(&family, &self.prefix);
        }
    }

    /// Before a save takes a slot: when none is free because spills hold them, drops the spills
    /// (counted) instead of making the save fail or wait.
    pub(super) fn free_slot_for_save(&mut self) {
        let Some(pool) = self.prefix_pins.as_ref() else {
            return;
        };
        let free = pool.free_slots();
        if let Some(disk) = self.disk.as_mut() {
            disk.tier.drop_spills_for_slots(free, "a save");
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
        self.disk.as_ref().is_some_and(HostStoreDisk::pending)
    }

    /// Copies up to `budget` bytes of pending saves to the writer and logs what finished.
    fn pump_disk(&mut self, budget: usize) {
        let layout = *self.state.layout();
        if let Some(disk) = self.disk.as_mut() {
            let family = QwenFamily {
                layout,
                pool: None,
                stream: &self.stream,
                state: &self.state,
            };
            disk.pump_with(&family, budget);
        }
    }

    /// A graceful stop: writes the tips that exist only in memory, newest first, so a restart
    /// restores them, within the bound the tier derives from their bytes and the write rate; a
    /// second stop signal ends the wait at once.
    pub(crate) fn drain_disk(&mut self) {
        if !self.prefix_identity.is_persistable() {
            return;
        }
        let layout = *self.state.layout();
        if let Some(disk) = self.disk.as_mut() {
            let family = QwenFamily {
                layout,
                pool: None,
                stream: &self.stream,
                state: &self.state,
            };
            disk.drain(&family, &self.prefix);
        }
    }

    /// Before a restore: when the disk holds a longer stored prefix of `prompt` than pinned
    /// memory does, reads it (verified in full) into a free pinned slot and stores it as an
    /// ordinary entry, so the restore that follows finds it. A miss changes nothing.
    pub(crate) fn promote_from_disk(&mut self, prompt: &[u32]) {
        let path = (
            CudaPrefillPath::Steps(self.state.layout().kv.mode),
            self.prefix_identity.clone(),
        );
        let layout = *self.state.layout();
        if let Some(disk) = self.disk.as_mut() {
            let mut family = QwenFamily {
                layout,
                pool: self.prefix_pins.as_mut(),
                stream: &self.stream,
                state: &self.state,
            };
            disk.promote(&mut family, &mut self.prefix, &path, prompt);
        }
    }

    /// The tier facts of the restore that just covered `covered` tokens. A restore served from
    /// pinned memory names its source; one that promoted a disk entry names the disk and what
    /// the read cost.
    pub(super) fn restore_tier(&mut self, covered: usize) -> crate::sched::job::PrefixTier {
        let source = self.restore_source.take();
        restore_tier(self.disk.as_mut(), covered, source)
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
