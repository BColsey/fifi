//! The Qwen3.6 MLX engine's use of the shared on-disk prefix tier. A saved prefix is a
//! `ResidentSnapshot`: shared device arrays. The family tells the shared host-store adapter
//! (`kv_disk::HostFamily`) how many positions a snapshot covers, how an entry is cut into one
//! object per layer, how the device arrays are read back into a host copy when a write is queued,
//! and how a verified read becomes a snapshot again. The policy (when to write, how long a stop
//! may wait, how a damaged entry falls back) is the tier's.

use super::{PrefillPath, Q36Executor, Q36Session, Result};
use crate::{
    EngError,
    family::{
        kv_disk::{
            DiskTier, FamilyContract, HostFamily, HostStoreDisk, LoadIdentity, PUMP_IDLE_BYTES,
            SlabLayout, SnapshotView, object_per_part, place_in_vec, plan_tier, restore_tier,
            vec_landing,
        },
        qwen36::{
            resident::{
                state::{ResidentSnapshot, ResidentState, object_bytes},
                weights::ResidentWeights,
            },
            settings::{Q36Settings, tokenizer_pin_for_spec},
        },
        session::PrefixIdentity,
    },
    kernels::MlxKernels,
    sched::{SessionBudget, job::PrefixTier},
    weights::sha256_hex,
};
use std::{cell::RefCell, rc::Rc};

/// What the shared tier calls this engine in its log lines.
const TIER_LABEL: &str = "q36 mlx";

/// Names the layout of a snapshot's bytes: bump it when `ResidentSnapshot` is stored differently.
const LAYOUT_TAG: &str = "q36-mlx-snapshot-v1";

/// What a Qwen3.6 MLX load fixes about the entries it writes.
const CONTRACT: FamilyContract = FamilyContract {
    adapter: "qwen36.mlx",
    backend: "mlx.apple-silicon",
    snapshot_schema: "qwen36.resident-state.v1",
    payload_abi: "delta-f32.kv-by-layer.v1",
    position_policy: "absolute-rope-from-zero",
    render_mode: "chat",
};

/// The model as the host-store adapter sees it. A snapshot's host copy is read from the device
/// when a write is queued, and a disk read is turned back into device arrays.
pub(super) struct MlxFamily<'a> {
    pub(super) kernels: RefCell<&'a mut MlxKernels>,
    pub(super) weights: &'a ResidentWeights,
    /// What a disk read needs to become a snapshot; absent when only writes are in play.
    pub(super) restore: Option<Restore<'a>>,
    pub(super) context_limit: usize,
}

pub(super) struct Restore<'a> {
    pub(super) state: &'a ResidentState,
    pub(super) budget: &'a mut SessionBudget,
    /// The bytes the saved prefixes may hold beyond the live session, when a device budget sets
    /// them, and what the store holds now.
    pub(super) room: Option<u64>,
    pub(super) held: u64,
}

impl HostFamily for MlxFamily<'_> {
    type Snapshot = ResidentSnapshot;
    type Path = PrefillPath;
    type Landing = Vec<u8>;

    fn prepare(&self, snapshot: &ResidentSnapshot) -> Result<()> {
        snapshot.make_host_copy(&mut self.kernels.borrow_mut(), self.weights)
    }

    fn view(&self, snapshot: &ResidentSnapshot) -> Option<SnapshotView> {
        let positions = snapshot.pos() as usize;
        let bytes = object_bytes(self.weights, positions)
            .ok()?
            .iter()
            .map(|&bytes| bytes as u64)
            .sum();
        let slot: Rc<Vec<u8>> = snapshot
            .host_copy()
            .get()
            .unwrap_or_else(|| Rc::new(Vec::new()));
        Some(SnapshotView {
            positions,
            bytes,
            slot,
        })
    }

    fn layout(&self, positions: usize) -> Result<SlabLayout> {
        if positions >= self.context_limit {
            return Err(EngError::ContextLimit(format!(
                "a kv disk entry of {positions} positions does not fit the {} position context",
                self.context_limit
            )));
        }
        Ok(object_per_part(&object_bytes(self.weights, positions)?))
    }

    fn begin_landing(
        &mut self,
        _tier: &mut DiskTier,
        positions: usize,
        layout: &SlabLayout,
    ) -> Result<Vec<u8>> {
        let restore = self
            .restore
            .as_ref()
            .ok_or_else(|| EngError::MemoryPlan("no session to restore into".into()))?;
        if let Some(room) = restore.room {
            let need = restore.state.loaded_charge(self.weights, positions)?;
            if restore.held.saturating_add(need) > room {
                return Err(EngError::MemoryPlan(
                    "a stored prefix would take memory the prefill piece needs".into(),
                ));
            }
        }
        vec_landing(TIER_LABEL, layout)
    }

    fn place(&self, landing: &mut Vec<u8>, slab: usize, part: &[u8]) -> Result<()> {
        place_in_vec(landing, slab, part)
    }

    fn finish_landing(&mut self, landing: Vec<u8>, positions: usize) -> Result<ResidentSnapshot> {
        let restore = self
            .restore
            .as_mut()
            .ok_or_else(|| EngError::MemoryPlan("no session to restore into".into()))?;
        let mut at = 0;
        let mut parts: Vec<&[u8]> = Vec::new();
        for bytes in object_bytes(self.weights, positions)? {
            let part = landing
                .get(at..at + bytes)
                .ok_or_else(|| EngError::BadRequest("a stored snapshot is too short".into()))?;
            parts.push(part);
            at += bytes;
        }
        restore.state.snapshot_from_objects(
            self.kernels.get_mut(),
            self.weights,
            restore.budget,
            positions,
            &parts,
        )
    }
}

/// What names this load to an entry kept outside the process.
fn load_identity(
    identity: &PrefixIdentity,
    weights: &ResidentWeights,
    path: PrefillPath,
) -> LoadIdentity {
    let d = &weights.shape.delta;
    let a = &weights.shape.attn;
    LoadIdentity {
        weights: identity.weights.to_string(),
        tokenizer: identity.tokenizer.to_string(),
        template: identity.template.to_string(),
        numeric_path: format!(
            "mlx {path:?} act {:?} kv {:?}",
            weights.consts.act, weights.kv
        ),
        family: CONTRACT,
        layout: format!(
            "{LAYOUT_TAG}; {} layers; delta heads {} value {} key {} conv {} kernel {}; \
             attention kv heads {} head dim {}",
            weights.layers.len(),
            d.value_heads,
            d.value_dim,
            d.key_dim,
            d.conv_dim(),
            d.conv_kernel,
            a.heads_kv,
            a.head_dim,
        ),
    }
}

/// What names this load's weights, tokenizer and template to an entry kept outside the process.
fn prefix_identity(settings: &Q36Settings) -> Result<PrefixIdentity> {
    let pin = tokenizer_pin_for_spec(&settings.spec)?;
    Ok(PrefixIdentity {
        weights: sha256_hex(&crate::weights::pack_digest(&settings.artifact)?).into(),
        tokenizer: sha256_hex(&pin.tokenizer_json).into(),
        template: sha256_hex(&pin.chat_template_jinja).into(),
    })
}

impl Q36Executor {
    /// Starts the on-disk prefix tier when the config, the environment or the build default asks
    /// for it, and says in one line whether it runs. A malformed override stops the start; a
    /// directory that cannot be used leaves the tier off. It runs after the prefill plan: the
    /// numeric path of an entry names the chunk size.
    pub(crate) fn enable_disk_tier(&mut self, settings: &Q36Settings) -> Result<()> {
        let Some((resolved, origin)) = plan_tier(settings.kv_cache.as_ref(), "qwen36", TIER_LABEL)?
        else {
            return Ok(());
        };
        // The tier is optional: a load whose pack or tokenizer cannot be named leaves it off.
        let identity = match prefix_identity(settings) {
            Ok(identity) => identity,
            Err(error) => {
                eprintln!("{TIER_LABEL}: kv disk tier off: the load cannot be named: {error:?}");
                return Ok(());
            }
        };
        let limits = self.prefix.limits();
        let load = load_identity(&identity, &self.decoder.weights, self.prefill_path());
        self.disk = HostStoreDisk::open(
            TIER_LABEL,
            &resolved,
            origin,
            &load,
            limits.roots + limits.entries,
        );
        Ok(())
    }

    /// Runs `work` on the disk tier with the model as the family and the host store.
    fn with_disk(
        &mut self,
        restore: Option<(&ResidentState, u64)>,
        work: impl FnOnce(
            &mut HostStoreDisk,
            &mut MlxFamily<'_>,
            &mut super::PrefixSlot<ResidentSnapshot, PrefillPath>,
        ),
    ) {
        let Some(disk) = self.disk.as_mut() else {
            return;
        };
        let decoder = &mut self.decoder;
        let held = restore.map_or(0, |(_, charged)| {
            decoder.budget.used().saturating_sub(charged)
        });
        let mut family = MlxFamily {
            kernels: RefCell::new(&mut decoder.kernels),
            weights: &decoder.weights,
            restore: restore.map(|(state, _)| Restore {
                state,
                budget: &mut decoder.budget,
                room: self.prefix_room,
                held,
            }),
            context_limit: self.context_limit,
        };
        work(disk, &mut family, &mut self.prefix);
    }

    /// After a save: queues the entries the next saves would push out, when the host's memory is
    /// short. With memory to spare the disk is not touched.
    pub(super) fn spill_after_save(&mut self) {
        self.with_disk(None, |disk, family, prefix| {
            disk.spill_after_save(family, prefix)
        });
    }

    /// Before a restore: when the disk holds a longer stored prefix of `prompt` than the host
    /// store does, reads it (verified in full) back onto the device and stores it as an
    /// ordinary entry, so the restore that follows finds it. A miss changes nothing.
    pub(super) fn promote_from_disk(&mut self, st: &Q36Session, prompt: &[u32]) {
        let path = self.prefill_path();
        let charged = st.state.charged_bytes().unwrap_or(0);
        self.with_disk(Some((&st.state, charged)), |disk, family, prefix| {
            disk.promote(family, prefix, &path, prompt)
        });
    }

    /// The tier facts of the restore that just covered `covered` tokens.
    pub(super) fn restore_tier(&mut self, covered: usize) -> PrefixTier {
        restore_tier(self.disk.as_mut(), covered, None)
    }

    /// Whether the disk tier has saves the engine thread should advance while it waits.
    pub(super) fn disk_pending(&self) -> bool {
        self.disk.as_ref().is_some_and(HostStoreDisk::pending)
    }

    /// Copies a bounded part of the pending saves to the writer.
    pub(super) fn pump_disk(&mut self) {
        self.with_disk(None, |disk, family, _| {
            disk.pump_with(family, PUMP_IDLE_BYTES)
        });
    }

    /// A graceful stop: writes the tips that exist only in host memory, within the tier's bound.
    pub(super) fn drain_disk(&mut self) {
        self.with_disk(None, |disk, family, prefix| disk.drain(family, prefix));
    }
}
