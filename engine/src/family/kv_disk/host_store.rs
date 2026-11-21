//! The disk tier behind a family's host prefix store. A family whose saved prefixes are copies
//! in host memory (`PrefixStore<Snapshot, Path>`) describes them with a [`HostFamily`]: how many
//! positions a snapshot covers, the bytes a reader sees, the layout of an entry in objects, and
//! how to rebuild a snapshot from the bytes of a verified read. Everything else is here, once:
//! when to spill (only past the watermark and while the host's memory is short), the write-behind
//! in chunks, what a graceful stop writes, how a hit becomes an ordinary store entry, and how a
//! damaged entry falls back. The family keeps its own store and its own restore.

use super::{
    Capture, DiskTier, Held, Hit, KvDisk, LoadIdentity, MIN_POSITIONS, PUMP_EMIT_BYTES, Reason,
    Resolved, SlabLayout, SlotBytes, Stored, drain_bound, host_memory_is_short, hurry_flag,
    restore_report, retry_while_removed, spill_candidates, spill_picks, start_line, stop_picks,
};
use crate::{
    EngError,
    family::session::{PrefixStore, SaveKind},
    sched::job::{PrefixSource, PrefixTier},
};
use std::{rc::Rc, time::Instant};

/// Host memory the engine leaves to the system before it keeps a copy of its own: what the
/// start plans of the other stores keep.
const HOST_RESERVE: u64 = 1 << 30;

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

/// A landing in a vector of host memory, for a family whose snapshot is one: the budget is
/// checked against the host's available memory and the allocation is fallible.
pub(crate) fn vec_landing(label: &str, layout: &SlabLayout) -> Result<Vec<u8>, EngError> {
    let total = slab_bytes(layout);
    let available = crate::family::mem::HostFacts::read("a kv disk restore", "a kv disk restore")
        .ok()
        .map(|host| host.budget());
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
    let view = family.view(snapshot)?;
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

/// Whether the host's available memory leaves room for one more entry of `entry_bytes` above the
/// reserve. Unknown counts as short, so no entry is lost for lack of a look.
pub(crate) fn host_is_short(label: &'static str, entry_bytes: u64) -> bool {
    let available = crate::family::mem::HostFacts::read(label, label)
        .ok()
        .map(|host| host.budget());
    host_memory_is_short(available, HOST_RESERVE, entry_bytes)
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
        let short = host_is_short(self.label, Self::largest(family, store));
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

/// The tests open a cache, which needs direct I/O: Linux and Android only.
#[cfg(all(test, any(target_os = "linux", target_os = "android")))]
#[path = "host_store_tests.rs"]
mod tests;
