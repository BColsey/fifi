//! The disk tier behind a family's host prefix store. A family whose saved prefixes are copies
//! in host memory (`PrefixStore<Snapshot, Path>`) describes them with a [`HostFamily`]: how many
//! positions a snapshot covers, the bytes a reader sees, the layout of an entry in objects, and
//! how to rebuild a snapshot from the bytes of a verified read. Everything else is here, once:
//! when to spill (only past the watermark and while the host's memory is short), the write-behind
//! in chunks, what a graceful stop writes, how a hit becomes an ordinary store entry, and how a
//! damaged entry falls back. The family keeps its own store and its own restore.

use super::{
    Capture, DiskTier, Held, Hit, KV_CACHE_DEFAULT_ON, KvCacheSettings, KvDisk, KvEnv,
    LoadIdentity, MIN_POSITIONS, ObjectRole, PUMP_EMIT_BYTES, Plan, Reason, Resolved, Run,
    SlabLayout, SlotBytes, Stored, drain_bound, free_bytes, host_memory_is_short, hurry_flag, plan,
    restore_report, retry_while_removed, spill_candidates, spill_picks, start_line, stop_picks,
};
use crate::{
    EngError,
    family::session::{PrefixStore, SaveKind},
    sched::job::{PrefixSource, PrefixTier},
};
use std::{rc::Rc, time::Instant};

/// Lookups one restore may spend on entries that turn out damaged.
const PROMOTE_ATTEMPTS: usize = 3;

/// What the disk tier sees of one stored snapshot: the positions it covers, the host bytes it
/// holds and a reader of them.
pub(crate) struct SnapshotView {
    pub(crate) positions: usize,
    pub(crate) bytes: u64,
    /// A second owner of the snapshot's bytes, to copy out of while the store still holds it.
    pub(crate) slot: Rc<dyn SlotBytes>,
}

/// What a family tells the tier about its host snapshots.
pub(crate) trait HostFamily {
    type Snapshot;
    type Path: PartialEq + Clone;
    /// Where the bytes of a disk read land before they become a snapshot.
    type Landing;

    /// What the disk may take of a snapshot, or `None` for one that cannot be written (an entry
    /// that is not kept in host memory).
    fn view(&self, snapshot: &Self::Snapshot) -> Option<SnapshotView>;
    /// The slot to read `snapshot` from, for a family whose host copy is made on demand (its
    /// state lives in device memory): the copy lives while the queued write holds the slot. Called
    /// before a snapshot is queued for a write; an error leaves it out of the queue. Families
    /// whose snapshot is host memory already keep the default, and the view's slot serves.
    fn prepare(&self, _snapshot: &Self::Snapshot) -> Result<Option<Rc<dyn SlotBytes>>, EngError> {
        Ok(None)
    }
    /// The host memory a stored entry may still take, read now; `None` when it cannot be read,
    /// which counts as short. The default is what the host has free; a family that runs under a
    /// memory budget of its own bounds it by what the budget has left.
    fn host_available(&self) -> Option<u64> {
        crate::family::mem::tier_available_bytes("kv disk")
    }
    /// The layout of an entry of `positions`, or why this load cannot hold one that long.
    fn layout(&self, positions: usize) -> Result<SlabLayout, EngError>;
    /// Makes room for a snapshot of `positions` laid out as `layout`: a free slot, or host memory.
    /// `tier` lets a family give up the spills that hold the slots it needs.
    fn begin_landing(
        &mut self,
        tier: &mut DiskTier,
        positions: usize,
        layout: &SlabLayout,
    ) -> Result<Self::Landing, EngError>;
    /// Puts `part`, which belongs at `slab` of the entry's slab, into the landing.
    fn place(&self, landing: &mut Self::Landing, slab: usize, part: &[u8]) -> Result<(), EngError>;
    /// The snapshot a filled landing becomes.
    fn finish_landing(
        &mut self,
        landing: Self::Landing,
        positions: usize,
    ) -> Result<Self::Snapshot, EngError>;
}

/// The layout of an entry that is one object per part, the parts back to back in one slab in the
/// order given: a family that cuts a snapshot per layer says how many bytes each layer takes.
pub(crate) fn object_per_part(part_bytes: &[usize]) -> SlabLayout {
    let mut at = 0;
    let objects = part_bytes
        .iter()
        .map(|&bytes| {
            let run = Run { slab: at, bytes };
            at += bytes;
            vec![run]
        })
        .collect();
    SlabLayout {
        objects,
        roles: vec![ObjectRole::Pages; part_bytes.len()],
    }
}

/// A landing in a vector of host memory, for a family whose snapshot is one: the budget is
/// checked against the host's available memory and the allocation is fallible.
pub(crate) fn vec_landing(label: &str, layout: &SlabLayout) -> Result<Vec<u8>, EngError> {
    let total = slab_bytes(layout);
    let available = crate::family::mem::tier_available_bytes("a kv disk restore");
    if available.is_some_and(|available| total as u64 > available) {
        return Err(EngError::MemoryPlan(format!(
            "{label}: the host budget cannot hold {total} B"
        )));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(total).map_err(|_| {
        EngError::MemoryPlan(format!("{label}: {total} B do not fit in host memory"))
    })?;
    data.resize(total, 0u8);
    Ok(data)
}

/// Copies `part` to `slab` of a vector landing.
pub(crate) fn place_in_vec(data: &mut [u8], slab: usize, part: &[u8]) -> Result<(), EngError> {
    let end = slab
        .checked_add(part.len())
        .filter(|end| *end <= data.len())
        .ok_or_else(|| EngError::BadRequest("kv disk: a part past the snapshot".into()))?;
    data[slab..end].copy_from_slice(part);
    Ok(())
}

pub(crate) type Store<F> = PrefixStore<<F as HostFamily>::Snapshot, <F as HostFamily>::Path>;

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

/// A stored snapshot as a spill: `None` when it is too short to repay a read or its key is not
/// the positions plus the pending token.
pub(crate) fn held_of<F: HostFamily>(
    family: &F,
    key: &[u32],
    kind: SaveKind,
    snapshot: &F::Snapshot,
    reason: Reason,
) -> Option<Held> {
    let prepared = match family.prepare(snapshot) {
        Ok(prepared) => prepared,
        Err(error) => {
            eprintln!("kv disk: a snapshot could not be prepared for a write: {error:?}");
            return None;
        }
    };
    let mut view = family.view(snapshot)?;
    if let Some(slot) = prepared {
        view.slot = slot;
    }
    if view.positions < MIN_POSITIONS || key.len() != view.positions + 1 {
        return None;
    }
    Some(Held {
        reason,
        slot: view.slot,
        key: key.to_vec(),
        capture: capture_of(kind),
        positions: view.positions,
        bytes: view.bytes,
    })
}

/// The bytes a slab of this layout spans.
fn slab_bytes(layout: &SlabLayout) -> usize {
    layout
        .flat()
        .iter()
        .map(|run| run.slab + run.bytes)
        .max()
        .unwrap_or(0)
}

/// Plans the tier of `family_key`'s engines from the config member, the environment and the build
/// default, and says in one line when it stays off. A malformed override stops the start; a
/// directory that cannot be used is found later, when the tier opens, and leaves it off.
pub(crate) fn plan_tier(
    settings: Option<&KvCacheSettings>,
    family_key: &'static str,
    label: &str,
) -> Result<Option<(Resolved, &'static str)>, EngError> {
    let planned = plan(
        settings,
        &KvEnv::from_process().for_family(family_key),
        KV_CACHE_DEFAULT_ON,
        free_bytes,
    )
    .map_err(|message| EngError::BadRequest(format!("engine config: {message}")))?;
    match planned {
        Plan::Off(reason) => {
            eprintln!("{label}: kv disk tier off: {reason}");
            Ok(None)
        }
        Plan::On { resolved, origin } => Ok(Some((resolved, origin))),
    }
}

/// The disk tier of one engine whose prefixes live in a host store.
pub(crate) struct HostStoreDisk {
    pub(crate) tier: DiskTier,
    label: &'static str,
}

impl HostStoreDisk {
    /// Opens the tier at `resolved`, or logs why it stays off. A directory that cannot be used
    /// never fails the start. `retained` is how many entries the host store keeps.
    pub(crate) fn open(
        label: &'static str,
        resolved: &Resolved,
        origin: &str,
        load: &LoadIdentity,
        retained: usize,
    ) -> Option<Self> {
        match KvDisk::open(&resolved.dir, resolved.quota_bytes, load) {
            Ok((cache, report)) => {
                eprintln!("{}", start_line(label, resolved, origin, &report, retained));
                Some(Self {
                    tier: DiskTier::new(label, cache, retained),
                    label,
                })
            }
            Err(error) => {
                eprintln!(
                    "{label}: kv disk tier off: {} cannot be used: {error}",
                    resolved.dir.display()
                );
                None
            }
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.tier.pending()
    }

    /// Copies a bounded part of the pending saves to the writer: a few MiB per call, which a
    /// batching worker makes between ticks and an idle one every 2 ms.
    pub(crate) fn pump<F: HostFamily>(&mut self, family: &F) {
        self.pump_with(family, PUMP_EMIT_BYTES);
    }

    /// [`Self::pump`] with the bytes to copy this call given by the caller: a decode token takes
    /// a few MiB, a prefill piece or an idle poll many.
    pub(crate) fn pump_with<F: HostFamily>(&mut self, family: &F, budget: usize) {
        self.tier
            .pump(&|positions| family.layout(positions), budget);
    }

    /// After a save: when the store holds as many entries as the watermark and `short` says the
    /// host's memory is short, queue the entries the next saves would push out. With memory to
    /// spare the disk is not touched.
    pub(crate) fn spill_over_watermark<F: HostFamily>(
        &mut self,
        family: &F,
        store: &Store<F>,
        short: bool,
    ) {
        let stored = store.entries_len() + store.roots_len();
        let want = spill_candidates(stored, self.tier.retained);
        if want == 0 || !short {
            return;
        }
        let entries: Vec<_> = store.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        for at in spill_picks(&view, want) {
            let (key, kind, snapshot) = entries[at];
            if let Some(held) = held_of(family, key, kind, snapshot, Reason::Watermark) {
                self.tier.enqueue(held);
            }
        }
    }

    /// After a save: [`Self::spill_over_watermark`] with the host's memory read now.
    pub(crate) fn spill_after_save<F: HostFamily>(&mut self, family: &F, store: &Store<F>) {
        let short = host_memory_is_short(
            family.host_available(),
            crate::family::mem::HOST_HEADROOM_BYTES,
            Self::largest(family, store),
        );
        self.spill_over_watermark(family, store, short);
    }

    /// The largest snapshot the store holds, for the host-memory check of a spill.
    pub(crate) fn largest<F: HostFamily>(family: &F, store: &Store<F>) -> u64 {
        store
            .snapshots()
            .filter_map(|snapshot| family.view(snapshot))
            .map(|view| view.bytes)
            .max()
            .unwrap_or(0)
    }

    /// A graceful stop: writes the live tips that exist only in memory, newest first, within the
    /// bound the tier derives from their bytes and the disk's write rate.
    pub(crate) fn drain<F: HostFamily>(&mut self, family: &F, store: &Store<F>) {
        self.drain_within(family, store, &drain_bound);
    }

    /// [`Self::drain`] with the bound of the wait given by the caller (a test on a loaded host
    /// does not want to be cut off by the disk's pace).
    pub(crate) fn drain_within<F: HostFamily>(
        &mut self,
        family: &F,
        store: &Store<F>,
        bound: &dyn Fn(u64, Option<f64>) -> std::time::Duration,
    ) {
        let entries: Vec<_> = store.entries_lru().collect();
        let view: Vec<Stored<'_>> = entries
            .iter()
            .map(|&(key, kind, _)| Stored {
                key,
                tip: kind == SaveKind::Tip,
            })
            .collect();
        let tips: Vec<Held> = stop_picks(&view)
            .into_iter()
            .filter_map(|at| {
                let (key, kind, snapshot) = entries[at];
                held_of(family, key, kind, snapshot, Reason::Stop)
            })
            .collect();
        let left_in_memory = entries.len().saturating_sub(tips.len());
        self.tier.drain(
            tips,
            left_in_memory,
            &|positions| family.layout(positions),
            hurry_flag(),
            bound,
        );
    }

    /// Before a restore: when the disk holds a longer stored prefix of `prompt` than the host
    /// store does, reads it (verified in full) into host memory and stores it as an ordinary
    /// entry, so the restore that follows finds it. A miss changes nothing.
    pub(crate) fn promote<F: HostFamily>(
        &mut self,
        family: &mut F,
        store: &mut Store<F>,
        path: &F::Path,
        prompt: &[u32],
    ) {
        self.tier.hit = None;
        retry_while_removed(PROMOTE_ATTEMPTS, || {
            self.promote_once(family, store, path, prompt)
        });
    }

    /// One lookup and read. Returns true when the entry failed its integrity check and was
    /// removed, so another candidate may serve the prompt.
    fn promote_once<F: HostFamily>(
        &mut self,
        family: &mut F,
        store: &mut Store<F>,
        path: &F::Path,
        prompt: &[u32],
    ) -> bool {
        let label = self.label;
        let tier = &mut self.tier;
        let Some(candidate) = tier.cache.lookup(prompt) else {
            return false;
        };
        if store
            .longest_candidate_len(prompt, path)
            .is_some_and(|len| len >= candidate.key_len)
        {
            return false;
        }
        let began = Instant::now();
        let positions = candidate.key_len - 1;
        let shape = match family.layout(positions) {
            Ok(shape) => shape,
            Err(error) => {
                eprintln!("{label}: kv disk restore skipped: {error:?}");
                return false;
            }
        };
        let mut landing = match family.begin_landing(tier, positions, &shape) {
            Ok(landing) => landing,
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("{label}: kv disk restore skipped: {error:?}");
                return false;
            }
        };
        let stats = match tier.read_into(&candidate, prompt, &shape, &mut |slab, part| {
            family.place(&mut landing, slab, part)
        }) {
            Ok(stats) => stats,
            Err(removed) => return removed,
        };
        let snapshot = match family.finish_landing(landing, positions) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("{label}: kv disk restore dropped: {error:?}");
                return false;
            }
        };
        let Some(&pending) = prompt[..candidate.key_len].last() else {
            return false;
        };
        let stored = store.save_loaded(
            &prompt[..candidate.key_len],
            pending,
            path.clone(),
            kind_of(candidate.capture),
            snapshot,
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
                    "{label}: kv disk read {positions} positions, {} B, {:.1} ms (verified), {:.1} ms with the lookup",
                    stats.bytes,
                    stats.ms,
                    began.elapsed().as_secs_f64() * 1000.0
                );
            }
            Ok(false) => {
                tier.counters.missed += 1;
                eprintln!(
                    "{label}: kv disk entry read but the store refused it; the prompt is prefilled"
                );
            }
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("{label}: kv disk entry could not be stored: {error:?}");
            }
        }
        false
    }
}

/// The tier facts of the restore that just covered `covered` tokens.
pub(crate) fn restore_tier(
    disk: Option<&mut HostStoreDisk>,
    covered: usize,
    source: Option<PrefixSource>,
) -> PrefixTier {
    restore_report(
        disk.map(|disk| &mut disk.tier),
        covered,
        source,
        PrefixSource::Pinned,
    )
}

/// A family whose snapshot is read back into one vector of host memory and cut into one object
/// per part: the layout (with the context check), the landing and the placing are provided; the
/// family says how many bytes each part takes, what a snapshot looks like to the writer and what a
/// filled vector becomes. Every `VecFamily` is a [`HostFamily`].
pub(crate) trait VecFamily {
    type Snapshot;
    type Path: PartialEq + Clone;

    /// What the shared tier calls this engine in its log lines.
    const LABEL: &'static str;

    /// The longest entry, in positions, a session of this load can hold; a longer one is refused.
    fn longest_entry(&self) -> usize;
    /// The bytes of each part (a layer) of an entry of `positions`, in order.
    fn part_bytes(&self, positions: usize) -> Result<Vec<usize>, EngError>;
    /// What the disk may take of a snapshot, as [`HostFamily::view`].
    fn view_of(&self, snapshot: &Self::Snapshot) -> Option<SnapshotView>;
    /// The snapshot a filled landing becomes.
    fn finish(&mut self, landing: Vec<u8>, positions: usize) -> Result<Self::Snapshot, EngError>;
    /// The host memory a stored entry may still take, as [`HostFamily::host_available`].
    fn available_bytes(&self) -> Option<u64> {
        crate::family::mem::tier_available_bytes(Self::LABEL)
    }
    /// Whether the host has room to hold an entry of `positions` beyond what is stored; a family
    /// that keeps a budget of its own for stored prefixes checks it here.
    fn room_for(&self, _positions: usize) -> Result<(), EngError> {
        Ok(())
    }
    /// The slot to read `snapshot` from, as [`HostFamily::prepare`].
    fn prepare_slot(
        &self,
        _snapshot: &Self::Snapshot,
    ) -> Result<Option<Rc<dyn SlotBytes>>, EngError> {
        Ok(None)
    }
}

impl<T: VecFamily> HostFamily for T {
    type Snapshot = <T as VecFamily>::Snapshot;
    type Path = <T as VecFamily>::Path;
    type Landing = Vec<u8>;

    fn prepare(&self, snapshot: &Self::Snapshot) -> Result<Option<Rc<dyn SlotBytes>>, EngError> {
        self.prepare_slot(snapshot)
    }

    fn view(&self, snapshot: &Self::Snapshot) -> Option<SnapshotView> {
        self.view_of(snapshot)
    }

    fn host_available(&self) -> Option<u64> {
        self.available_bytes()
    }

    fn layout(&self, positions: usize) -> Result<SlabLayout, EngError> {
        let longest = self.longest_entry();
        if positions > longest {
            return Err(EngError::ContextLimit(format!(
                "a kv disk entry of {positions} positions is longer than the {longest} this load can hold"
            )));
        }
        Ok(object_per_part(&self.part_bytes(positions)?))
    }

    fn begin_landing(
        &mut self,
        _tier: &mut DiskTier,
        positions: usize,
        layout: &SlabLayout,
    ) -> Result<Vec<u8>, EngError> {
        self.room_for(positions)?;
        vec_landing(T::LABEL, layout)
    }

    fn place(&self, landing: &mut Vec<u8>, slab: usize, part: &[u8]) -> Result<(), EngError> {
        place_in_vec(landing, slab, part)
    }

    fn finish_landing(
        &mut self,
        landing: Vec<u8>,
        positions: usize,
    ) -> Result<Self::Snapshot, EngError> {
        self.finish(landing, positions)
    }
}

/// The three parts of an engine's disk tier that a call works on at once: the tier, the family
/// view of the model, and the host store.
pub(crate) struct DiskParts<'a, F: HostFamily> {
    pub(crate) disk: &'a mut HostStoreDisk,
    pub(crate) family: F,
    pub(crate) store: &'a mut Store<F>,
}

/// An engine that keeps its saved prefixes in a host store with a [`HostStoreDisk`] behind it. It
/// says how to take its own parts apart; every call the engine makes on the tier is provided.
pub(crate) trait DiskOwner {
    type Snapshot;
    type Path: PartialEq + Clone;
    /// The family view of the model, borrowing from the engine for one call.
    type Family<'a>: HostFamily<Snapshot = Self::Snapshot, Path = Self::Path>
    where
        Self: 'a;
    /// What a restore needs from the session it restores into; nothing for a write.
    type Restore: Copy;

    fn disk_ref(&self) -> Option<&HostStoreDisk>;
    fn disk_mut(&mut self) -> Option<&mut HostStoreDisk>;
    /// The tier, the family and the store, or `None` when the tier is off. A restore passes what
    /// it needs from the session; a write passes `None`.
    fn disk_parts(
        &mut self,
        restore: Option<Self::Restore>,
    ) -> Option<DiskParts<'_, Self::Family<'_>>>;
    /// The store path a restore looks entries up under.
    fn restore_path(&self) -> Self::Path;

    /// After a save: queues the entries the next saves would push out when the host's memory is
    /// short. With memory to spare the disk is not touched.
    fn spill_after_save(&mut self) {
        if let Some(parts) = self.disk_parts(None) {
            parts.disk.spill_after_save(&parts.family, parts.store);
        }
    }

    /// Test support: queues the entries the next saves would push out as if the host were short.
    #[cfg(test)]
    fn spill_now(&mut self) {
        if let Some(parts) = self.disk_parts(None) {
            parts
                .disk
                .spill_over_watermark(&parts.family, parts.store, true);
        }
    }

    /// Before a restore: reads a longer stored prefix of `prompt` than the store holds, when the
    /// disk has one, into the store as an ordinary entry. A miss changes nothing.
    fn promote_from_disk(&mut self, restore: Self::Restore, prompt: &[u32]) {
        let path = self.restore_path();
        if let Some(mut parts) = self.disk_parts(Some(restore)) {
            parts
                .disk
                .promote(&mut parts.family, parts.store, &path, prompt);
        }
    }

    /// Whether the tier has saves the engine thread should advance while it waits.
    fn disk_pending(&self) -> bool {
        self.disk_ref().is_some_and(HostStoreDisk::pending)
    }

    /// Copies up to `budget` bytes of the pending saves to the writer.
    fn pump_disk(&mut self, budget: usize) {
        if let Some(parts) = self.disk_parts(None) {
            parts.disk.pump_with(&parts.family, budget);
        }
    }

    /// A graceful stop: writes the tips that exist only in memory, within the tier's bound.
    fn drain_disk(&mut self) {
        if let Some(parts) = self.disk_parts(None) {
            parts.disk.drain(&parts.family, parts.store);
        }
    }

    /// The tier facts of the restore that just covered `covered` tokens; `source` names where a
    /// restore served from memory came from.
    fn restore_tier_of(&mut self, covered: usize, source: Option<PrefixSource>) -> PrefixTier {
        restore_tier(self.disk_mut(), covered, source)
    }
}

#[cfg(test)]
#[path = "host_store_tests.rs"]
mod tests;
