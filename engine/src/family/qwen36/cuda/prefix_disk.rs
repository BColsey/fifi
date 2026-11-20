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
    prefix_shape::{EntryShape, Run},
    state::{LAYOUT_TAG, pinned::PinnedSnapshot},
};
use super::QwenResidentStep;
use crate::{
    EngError,
    family::{
        expert_bank::disk_store::DiskMiss,
        qwen36::kv_cache::{
            ASSUMED_WRITE_RATE, CHUNK_BYTES, Capture, KvDisk, LoadIdentity, MIN_POSITIONS,
            OpenReport, Push, Resolved, WriteEvent, WriteJob, WriteSpec, drain_bound,
            spill_candidates, watermark,
        },
        session::SaveKind,
    },
    kernels::cuda::PinnedBuf,
    sched::job::{PrefixSource, PrefixTier},
};
use std::{cell::RefCell, rc::Rc, time::Instant};

/// Bytes the engine thread copies out of a held slot per prefill piece.
pub(super) const PUMP_PREFILL_BYTES: usize = 64 << 20;
/// Bytes copied per decode token.
pub(super) const PUMP_EMIT_BYTES: usize = 4 << 20;
/// Bytes copied per idle poll.
pub(super) const PUMP_IDLE_BYTES: usize = 64 << 20;
/// Saves that may wait behind the copy in progress. Each holds its pinned slot.
const QUEUE_DEPTH: usize = 2;
/// Lookups one restore may spend on entries that turn out damaged.
const PROMOTE_ATTEMPTS: usize = 3;

/// Why an entry goes to disk. The disk is used only when pinned memory is short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reason {
    /// The store passed its watermark: this is one of the entries the next saves push out.
    Watermark,
    /// The engine is stopping and this entry exists only in memory.
    Stop,
}

impl Reason {
    fn name(self) -> &'static str {
        match self {
            Reason::Watermark => "watermark",
            Reason::Stop => "stop",
        }
    }
}

/// A pinned entry waiting to be copied to disk, holding its slot.
struct Held {
    reason: Reason,
    slot: Rc<RefCell<PinnedBuf>>,
    key: Vec<u32>,
    capture: Capture,
    positions: usize,
    /// The bytes of the entry's objects.
    bytes: u64,
}

/// The copy in progress: which run, and how far into it.
struct Copying {
    job: WriteJob,
    held: Held,
    runs: Vec<Run>,
    at: usize,
    within: usize,
}

impl Copying {
    /// The bytes of the entry not yet handed to the writer.
    fn remaining(&self) -> u64 {
        let rest: usize = self.runs[self.at..].iter().map(|run| run.bytes).sum();
        (rest - self.within) as u64
    }
}

/// What the last disk restore cost, kept until the restore's report is built.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Hit {
    key_len: usize,
    bytes: u64,
    ms: f64,
}

/// Entries and bytes written for one reason.
#[derive(Default, Clone, Copy)]
struct Spilled {
    entries: u64,
    bytes: u64,
}

#[derive(Default)]
struct Counters {
    watermark: Spilled,
    stop: Spilled,
    /// Spills dropped before they were written: their slot was needed by a save.
    dropped: u64,
    saved: u64,
    skipped: u64,
    failed: u64,
    restored: u64,
    missed: u64,
}

pub(super) struct DiskTier {
    cache: KvDisk,
    /// Saves waiting for the copy in progress, oldest first, at most `QUEUE_DEPTH`.
    queued: Vec<Held>,
    copying: Option<Copying>,
    /// Jobs whose bytes are all handed to the writer and that have not reported yet.
    finishing: usize,
    /// The reason of every job that has begun and not yet reported.
    inflight: Vec<(u64, Reason)>,
    /// The retained slots of the pinned pool, which set the watermark.
    retained: usize,
    hit: Option<Hit>,
    counters: Counters,
}

impl DiskTier {
    fn pending(&self) -> bool {
        !self.queued.is_empty() || self.copying.is_some() || self.finishing > 0
    }

    /// Whether `key` is already on disk, waiting for the writer or being copied.
    fn has(&self, key: &[u32]) -> bool {
        self.queued.iter().any(|held| held.key == key)
            || self.copying.as_ref().is_some_and(|c| c.held.key == key)
            || self.cache.has_exact(key)
    }

    /// Adds a spill behind the copy in progress. A waiting tip that the new tip extends is dead;
    /// a full queue drops its least valuable entry (a periodic one before a tip, oldest first),
    /// or the offer itself when it is a periodic entry and every waiter is a tip.
    fn enqueue(&mut self, offer: Held) {
        if self.has(&offer.key) {
            return;
        }
        if offer.capture == Capture::Tip {
            let before = self.queued.len();
            self.queued.retain(|held| {
                !(held.capture == Capture::Tip
                    && held.key.len() < offer.key.len()
                    && offer.key.starts_with(&held.key))
            });
            self.counters.skipped += (before - self.queued.len()) as u64;
        }
        if self.queued.len() >= QUEUE_DEPTH {
            let victim = self
                .queued
                .iter()
                .position(|held| held.capture == Capture::Periodic)
                .or_else(|| (offer.capture == Capture::Tip).then_some(0));
            let Some(at) = victim else {
                self.counters.dropped += 1;
                return;
            };
            self.queued.remove(at);
            self.counters.dropped += 1;
        }
        self.queued.push(offer);
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
    let shape = EntryShape::of(layout, positions).ok()?;
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

impl QwenResidentStep<'_> {
    fn load_identity(&self) -> LoadIdentity {
        let layout = self.state.layout();
        let per_token: usize = layout
            .kv
            .prefix_ranges(0, 1)
            .map_or(0, |ranges| ranges.iter().map(|range| range.bytes).sum());
        LoadIdentity {
            weights: self.prefix_identity.weights.to_string(),
            tokenizer: self.prefix_identity.tokenizer.to_string(),
            template: self.prefix_identity.template.to_string(),
            numeric_path: format!("{:?}", CudaPrefillPath::Steps(layout.kv.mode)),
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
                eprintln!("{}", start_line(resolved, origin, &report, retained));
                self.disk = Some(DiskTier {
                    cache,
                    queued: Vec::new(),
                    copying: None,
                    finishing: 0,
                    inflight: Vec::new(),
                    retained,
                    hit: None,
                    counters: Counters::default(),
                });
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

    /// After a save publishes: when the store holds as many entries as the watermark, queue the
    /// least recently used ones that the next saves would push out, so their eviction never
    /// waits for a write. Below the watermark the disk is not touched.
    pub(super) fn spill_over_watermark(&mut self) {
        if !self.prefix_identity.is_persistable() {
            return;
        }
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        let stored = self.prefix.entries_len() + self.prefix.roots_len();
        let want = spill_candidates(stored, tier.retained);
        if want == 0 {
            return;
        }
        let layout = *self.state.layout();
        for (key, kind, saved) in self.prefix.entries_lru().take(want) {
            if let Some(offer) = held_of(&layout, key, kind, saved, Reason::Watermark) {
                tier.enqueue(offer);
            }
        }
    }

    /// Before a save takes a slot: when none is free because spills hold them, drops the spills
    /// (counted) instead of making the save fail or wait.
    pub(super) fn free_slot_for_save(&mut self) {
        let Some(pool) = self.prefix_pins.as_ref() else {
            return;
        };
        if pool.free_slots() > 0 {
            return;
        }
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        let mut dropped = tier.queued.drain(..).count();
        if let Some(copying) = tier.copying.take() {
            tier.cache.abort(&copying.job);
            tier.inflight.retain(|(id, _)| *id != copying.job.id());
            dropped += 1;
        }
        if dropped > 0 {
            tier.counters.dropped += dropped as u64;
            eprintln!(
                "q36 cuda: kv disk: {dropped} spills dropped, their pinned slots are needed by a save"
            );
        }
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
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        // The writer sends a job's report before it releases the job, so an idle cache means
        // every report is already waiting: read the flag first, then the reports.
        let idle = tier.cache.is_idle();
        for event in tier.cache.poll() {
            tier.finishing = tier.finishing.saturating_sub(1);
            match event {
                WriteEvent::Done {
                    id,
                    positions,
                    bytes,
                    ms,
                } => {
                    tier.counters.saved += 1;
                    let why = take_reason(&mut tier.inflight, id);
                    let spilled = match why {
                        Reason::Watermark => &mut tier.counters.watermark,
                        Reason::Stop => &mut tier.counters.stop,
                    };
                    spilled.entries += 1;
                    spilled.bytes += bytes;
                    eprintln!(
                        "q36 cuda: kv disk saved {positions} positions, {bytes} B ({}), {ms:.1} ms; to disk so far: {}",
                        why.name(),
                        tier.counters.summary()
                    );
                }
                WriteEvent::Failed { id, reason } => {
                    take_reason(&mut tier.inflight, id);
                    tier.counters.failed += 1;
                    eprintln!("q36 cuda: kv disk save failed: {reason}");
                }
            }
        }
        if idle && tier.copying.is_none() {
            tier.finishing = 0;
        }
        let mut left = budget;
        while left > 0 {
            if tier.copying.is_none() {
                if tier.queued.is_empty() {
                    return;
                }
                // Tips go before periodic and block entries, each oldest first.
                let next = tier
                    .queued
                    .iter()
                    .position(|held| held.capture == Capture::Tip)
                    .unwrap_or(0);
                let held = tier.queued.remove(next);
                let layout = *self.state.layout();
                let shape = match EntryShape::of(&layout, held.positions) {
                    Ok(shape) => shape,
                    Err(error) => {
                        eprintln!("q36 cuda: kv disk save skipped: {error:?}");
                        continue;
                    }
                };
                let spec = WriteSpec {
                    tokens: held.key.clone(),
                    capture: held.capture,
                    object_bytes: shape.object_bytes(),
                };
                let evicted_before = tier.cache.evicted();
                match tier.cache.begin_write(spec) {
                    Ok(job) => {
                        tier.inflight.push((job.id(), held.reason));
                        let evicted = tier.cache.evicted() - evicted_before;
                        if evicted > 0 {
                            eprintln!(
                                "q36 cuda: kv disk quota: {evicted} least recently used entries removed to make room for {} positions",
                                held.positions
                            );
                        }
                        tier.copying = Some(Copying {
                            job,
                            runs: shape.flat(),
                            held,
                            at: 0,
                            within: 0,
                        });
                    }
                    Err(skip) => {
                        tier.counters.skipped += 1;
                        eprintln!(
                            "q36 cuda: kv disk save skipped for {} positions: {skip}",
                            held.positions
                        );
                        continue;
                    }
                }
            }
            let Some(copying) = tier.copying.as_mut() else {
                return;
            };
            match copy_chunk(copying, left.min(CHUNK_BYTES)) {
                Ok((chunk, done)) => {
                    let sent = chunk.len();
                    match tier.cache.push(&copying.job, chunk) {
                        Push::Sent => left = left.saturating_sub(sent.max(1)),
                        Push::Full(back) => {
                            // Put the cursor back: the writer is behind, try again later.
                            rewind(copying, back.len());
                            return;
                        }
                        Push::Dead => {
                            tier.counters.failed += 1;
                            eprintln!("q36 cuda: kv disk save failed: the writer stopped");
                            tier.copying = None;
                            return;
                        }
                    }
                    if done {
                        tier.copying = None;
                        tier.finishing += 1;
                    }
                }
                Err(error) => {
                    tier.cache.abort(&copying.job);
                    tier.inflight.retain(|(id, _)| *id != copying.job.id());
                    tier.counters.failed += 1;
                    eprintln!("q36 cuda: kv disk save failed: {error:?}");
                    tier.copying = None;
                }
            }
        }
    }

    /// A graceful stop: writes the tips that exist only in memory, newest first, so a restart
    /// restores them. What is not a tip (a periodic or block entry waiting, or being copied) is
    /// left out. The wait is bounded by the bytes the tips still have to move and the measured
    /// write rate; one line says the bound and one says how it ended.
    pub(crate) fn drain_disk(&mut self) {
        if self.disk.is_none() {
            return;
        }
        // The tips in memory that the disk does not hold, newest first; a tip that a longer tip
        // of the same conversation extends is dead.
        let layout = *self.state.layout();
        let mut tips: Vec<Held> = Vec::new();
        if self.prefix_identity.is_persistable() {
            let mut stored: Vec<Held> = self
                .prefix
                .entries_lru()
                .filter(|(_, kind, _)| *kind == SaveKind::Tip)
                .filter_map(|(key, kind, saved)| held_of(&layout, key, kind, saved, Reason::Stop))
                .collect();
            stored.reverse();
            for held in stored {
                if !tips.iter().any(|newer| {
                    newer.key.len() > held.key.len() && newer.key.starts_with(&held.key)
                }) {
                    tips.push(held);
                }
            }
        }
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        tips.retain(|held| !tier.has(&held.key));
        let left_in_memory = self.prefix.entries_len().saturating_sub(tips.len());
        let mut dropped = 0usize;
        // A waiting spill that is a tip stays (behind the new ones); anything else goes.
        let waiting = std::mem::take(&mut tier.queued);
        for held in waiting {
            if held.capture == Capture::Tip {
                tips.push(Held {
                    reason: Reason::Stop,
                    ..held
                });
            } else {
                dropped += 1;
            }
        }
        if tier
            .copying
            .as_ref()
            .is_some_and(|copying| copying.held.capture != Capture::Tip)
            && let Some(copying) = tier.copying.take()
        {
            tier.cache.abort(&copying.job);
            tier.inflight.retain(|(id, _)| *id != copying.job.id());
            dropped += 1;
        }
        tier.queued = tips;
        if !tier.pending() {
            eprintln!(
                "q36 cuda: kv disk stop: no tip exists only in memory; to disk: {}",
                tier.counters.summary()
            );
            return;
        }
        let remaining: u64 = tier.queued.iter().map(|held| held.bytes).sum::<u64>()
            + tier.copying.as_ref().map_or(0, Copying::remaining);
        let rate = tier.cache.write_rate();
        let bound = drain_bound(remaining, rate);
        let count = tier.queued.len() + usize::from(tier.copying.is_some());
        eprintln!(
            "q36 cuda: kv disk stop: {count} tips only in memory ({remaining} B to move), {dropped} waiting spills dropped, {left_in_memory} other entries stay in memory only; waiting at most {:.0} s at {}",
            bound.as_secs_f64(),
            rate.map_or_else(
                || format!("an assumed {:.0} MB/s", ASSUMED_WRITE_RATE / 1e6),
                |rate| format!("the measured {:.0} MB/s", rate / 1e6)
            )
        );
        let began = Instant::now();
        while self.disk_pending() && began.elapsed() < bound {
            self.pump_disk(PUMP_IDLE_BYTES);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let Some(tier) = self.disk.as_mut() else {
            return;
        };
        if tier.pending() {
            if let Some(copying) = tier.copying.take() {
                tier.cache.abort(&copying.job);
            }
            eprintln!(
                "q36 cuda: kv disk stop: gave up after {:.1} s with {} saves unwritten; to disk: {}",
                began.elapsed().as_secs_f64(),
                tier.queued.len() + tier.finishing,
                tier.counters.summary()
            );
        } else {
            eprintln!(
                "q36 cuda: kv disk stop: tips written in {:.1} s; to disk: {}",
                began.elapsed().as_secs_f64(),
                tier.counters.summary()
            );
        }
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
        for _ in 0..PROMOTE_ATTEMPTS {
            if !self.promote_once(prompt) {
                break;
            }
        }
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
        let shape = match EntryShape::of(&layout, positions) {
            Ok(shape) => shape,
            Err(error) => {
                eprintln!("q36 cuda: kv disk restore skipped: {error:?}");
                return false;
            }
        };
        let Some(pool) = self.prefix_pins.as_mut() else {
            return false;
        };
        let landing = match pool.begin_landing(&self.stream, &self.state, positions) {
            Ok(landing) => landing,
            Err(error) => {
                tier.counters.missed += 1;
                eprintln!("q36 cuda: kv disk restore skipped: no free pinned slot ({error:?})");
                return false;
            }
        };
        let read = tier.cache.read(
            &candidate,
            prompt,
            &shape.object_bytes(),
            &mut |object, offset, bytes| {
                shape
                    .scatter(object, offset, bytes, &mut |slab, part| {
                        landing.write(slab, part)
                    })
                    .map_err(|error| DiskMiss::Refused(format!("{error:?}")))
            },
        );
        let stats = match read {
            Ok(stats) => stats,
            Err(miss) => {
                tier.counters.missed += 1;
                let removed = matches!(miss, DiskMiss::Corrupt(_) | DiskMiss::Absent);
                eprintln!(
                    "q36 cuda: kv disk miss for {} tokens ({miss}){}",
                    candidate.key_len,
                    if removed {
                        "; the entry is removed"
                    } else {
                        "; the prompt is prefilled"
                    }
                );
                return removed;
            }
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
        let mut tier = PrefixTier::default();
        if let Some(disk) = self.disk.as_ref() {
            let c = &disk.counters;
            tier.disk_written_entries =
                u32::try_from(c.watermark.entries + c.stop.entries).unwrap_or(u32::MAX);
            tier.disk_written_bytes = c.watermark.bytes + c.stop.bytes;
            tier.disk_dropped = u32::try_from(c.dropped).unwrap_or(u32::MAX);
        }
        if covered == 0 {
            return tier;
        }
        if let Some(hit) = self
            .disk
            .as_mut()
            .and_then(|tier| tier.hit.take())
            .filter(|hit| hit.key_len == covered)
        {
            tier.source = PrefixSource::Disk;
            tier.disk_bytes = hit.bytes;
            tier.disk_ms = hit.ms.round().min(f64::from(u32::MAX)) as u32;
            return tier;
        }
        tier.source = source.unwrap_or(PrefixSource::Pinned);
        tier
    }
}

/// The first line of a started tier: where it lives, how much it may hold and what it found.
fn start_line(resolved: &Resolved, origin: &str, report: &OpenReport, retained: usize) -> String {
    format!(
        "q36 cuda: kv disk tier on ({origin}), dir {}, quota {} B, {} entries of this load found ({} of other loads, {} removed, {} stray objects removed), {} B used; written only when pinned memory is short (from {} of {retained} entries) and at a graceful stop; purge with --purge-kv-cache",
        resolved.dir.display(),
        resolved.quota_bytes,
        report.entries,
        report.foreign,
        report.dropped,
        report.orphans,
        report.used_bytes,
        watermark(retained),
    )
}

/// The reason a finished job was started for (a stopped one reads as the watermark).
fn take_reason(inflight: &mut Vec<(u64, Reason)>, id: u64) -> Reason {
    inflight
        .iter()
        .position(|(job, _)| *job == id)
        .map_or(Reason::Watermark, |at| inflight.remove(at).1)
}

impl Counters {
    /// The totals by reason, for a log line.
    fn summary(&self) -> String {
        format!(
            "{} entries {} B at the watermark, {} entries {} B at a stop, {} dropped",
            self.watermark.entries,
            self.watermark.bytes,
            self.stop.entries,
            self.stop.bytes,
            self.dropped
        )
    }
}

/// Reads the next chunk of at most `max` bytes from the held slot. The second value is true once
/// the last byte of the entry has been read.
fn copy_chunk(copying: &mut Copying, max: usize) -> Result<(Vec<u8>, bool), EngError> {
    let mut out = Vec::new();
    let slot = copying.held.slot.borrow();
    while out.len() < max && copying.at < copying.runs.len() {
        let run = copying.runs[copying.at];
        let take = (run.bytes - copying.within).min(max - out.len());
        let from = out.len();
        out.resize(from + take, 0);
        slot.read(run.slab + copying.within, &mut out[from..])?;
        copying.within += take;
        if copying.within == run.bytes {
            copying.at += 1;
            copying.within = 0;
        }
    }
    Ok((out, copying.at == copying.runs.len()))
}

/// Moves the cursor back by `bytes`, undoing a chunk the writer did not take.
fn rewind(copying: &mut Copying, mut bytes: usize) {
    while bytes > 0 {
        if copying.within == 0 {
            if copying.at == 0 {
                return;
            }
            copying.at -= 1;
            copying.within = copying.runs[copying.at].bytes;
        }
        let back = copying.within.min(bytes);
        copying.within -= back;
        bytes -= back;
    }
}
