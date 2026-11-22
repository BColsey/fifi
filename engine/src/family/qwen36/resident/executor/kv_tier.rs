//! The Qwen3.6 MLX engine's use of the shared on-disk prefix tier. A saved prefix is a
//! `ResidentSnapshot`: shared device arrays. The family tells the shared host-store adapter
//! (`kv_disk::HostFamily`) how many positions a snapshot covers, how an entry is cut into one
//! object per layer, how the device arrays are read back into a host copy when a write is queued,
//! and how a verified read becomes a snapshot again. The policy (when to write, how long a stop
//! may wait, how a damaged entry falls back) is the tier's.

use super::{PrefillPath, Q36Executor, Result};
use crate::{
    EngError,
    family::{
        kv_disk::{
            DiskOwner, DiskParts, FamilyContract, HostStoreDisk, LoadIdentity, SlotBytes,
            SnapshotView, VecFamily, plan_tier,
        },
        qwen36::{
            resident::{
                state::{LoadShape, ResidentSnapshot, object_bytes},
                weights::ResidentWeights,
            },
            settings::{Q36Settings, tokenizer_pin_for_spec},
        },
        session::PrefixIdentity,
    },
    kernels::MlxKernels,
    sched::SessionBudget,
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
pub(crate) struct MlxFamily<'a> {
    pub(super) kernels: RefCell<&'a mut MlxKernels>,
    pub(super) weights: &'a ResidentWeights,
    pub(super) budget: &'a mut SessionBudget,
    pub(super) context_limit: usize,
    /// What a disk read needs of the session it restores into; absent when only writes are in
    /// play.
    pub(super) restore: Option<Restore>,
    /// The bytes the saved prefixes may hold beyond the live session, when a device budget sets
    /// them, and what the store holds now.
    pub(super) room: Option<u64>,
    pub(super) held: u64,
    /// The device budget the engine runs under, when it has one: what a stored entry may take of
    /// the host is what that budget leaves beside the memory the engine holds.
    pub(super) cap: Option<u64>,
}

/// What a restore takes from the session it restores into.
#[derive(Clone, Copy)]
pub(crate) struct Restore {
    pub(super) shape: LoadShape,
    /// What the session's own state charges the budget; the rest is what the store holds.
    pub(super) charged: u64,
}

impl VecFamily for MlxFamily<'_> {
    type Snapshot = ResidentSnapshot;
    type Path = PrefillPath;
    const LABEL: &'static str = TIER_LABEL;

    fn longest_entry(&self) -> usize {
        self.context_limit.saturating_sub(1)
    }

    fn part_bytes(&self, positions: usize) -> Result<Vec<usize>> {
        object_bytes(self.weights, positions)
    }

    fn prepare_slot(&self, snapshot: &ResidentSnapshot) -> Result<Option<Rc<dyn SlotBytes>>> {
        let copy = snapshot.make_host_copy(&mut self.kernels.borrow_mut(), self.weights)?;
        Ok(Some(copy))
    }

    fn view_of(&self, snapshot: &ResidentSnapshot) -> Option<SnapshotView> {
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

    fn available_bytes(&self) -> Option<u64> {
        let system = crate::family::mem::tier_available_bytes(Self::LABEL);
        let Some(cap) = self.cap else {
            return system;
        };
        let active = self.kernels.borrow().memory_stats().ok()?.active;
        system.map(|bytes| bytes.min(cap.saturating_sub(active)))
    }

    fn room_for(&self, positions: usize) -> Result<()> {
        let restore = self
            .restore
            .ok_or_else(|| EngError::MemoryPlan("no session to restore into".into()))?;
        if let Some(room) = self.room {
            let need = restore.shape.loaded_charge(self.weights, positions)?;
            if self.held.saturating_add(need) > room {
                return Err(EngError::MemoryPlan(
                    "a stored prefix would take memory the prefill piece needs".into(),
                ));
            }
        }
        Ok(())
    }

    fn finish(&mut self, landing: Vec<u8>, positions: usize) -> Result<ResidentSnapshot> {
        let restore = self
            .restore
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
        restore.shape.snapshot_from_objects(
            self.kernels.get_mut(),
            self.weights,
            self.budget,
            positions,
            &parts,
        )
    }
}

/// What names this load to an entry kept outside the process.
pub(super) fn load_identity(
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
        weights: sha256_hex(&crate::weights::pack_digest(
            &settings.artifact,
            crate::weights::Hidden::Skipped,
        )?)
        .into(),
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
}

impl DiskOwner for Q36Executor {
    type Snapshot = ResidentSnapshot;
    type Path = PrefillPath;
    type Family<'a> = MlxFamily<'a>;
    type Restore = Restore;

    fn disk_ref(&self) -> Option<&HostStoreDisk> {
        self.disk.as_ref()
    }

    fn disk_mut(&mut self) -> Option<&mut HostStoreDisk> {
        self.disk.as_mut()
    }

    /// A restore carries the growth rules of the session it restores into, and the store's bytes
    /// are what the session budget holds beyond that session's own charge.
    fn disk_parts(&mut self, restore: Option<Restore>) -> Option<DiskParts<'_, MlxFamily<'_>>> {
        let disk = self.disk.as_mut()?;
        let decoder = &mut self.decoder;
        let held = restore.map_or(0, |restore| {
            decoder.budget.used().saturating_sub(restore.charged)
        });
        Some(DiskParts {
            disk,
            family: MlxFamily {
                kernels: RefCell::new(&mut decoder.kernels),
                weights: &decoder.weights,
                budget: &mut decoder.budget,
                context_limit: self.context_limit,
                restore,
                room: self.prefix_room,
                held,
                cap: self.tier_cap,
            },
            store: &mut self.prefix,
        })
    }

    fn restore_path(&self) -> PrefillPath {
        self.prefill_path()
    }
}
