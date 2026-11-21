//! The tier that sits between a family's host prefix store and the disk cache: when to spill
//! (only past a watermark of the store), how a held slot is copied out in chunks between the
//! engine's other work, what a graceful stop writes and how long it may wait, when writes back
//! off, and how a verified read lands in a slot. The family supplies the slot bytes
//! (`SlotBytes`), the layout of an entry (`SlabLayout`) and its own prefix store; the tier owns
//! the queue, the counters and the policy, so every family that wires it behaves the same.

use super::{
    ASSUMED_WRITE_RATE, CHUNK_BYTES, Candidate, Capture, KvDisk, OpenReport, Push, Put, ReadStats,
    Resolved, Run, SlabLayout, WriteEvent, WriteJob, WriteSpec, drain_bound, watermark,
};
use crate::{EngError, family::expert_bank::disk_store::DiskMiss};
use std::{
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// Bytes the engine thread copies out of a held slot per prefill piece.
pub(crate) const PUMP_PREFILL_BYTES: usize = 64 << 20;
/// Bytes copied per decode token.
pub(crate) const PUMP_EMIT_BYTES: usize = 4 << 20;
/// Bytes copied per idle poll.
pub(crate) const PUMP_IDLE_BYTES: usize = 64 << 20;
/// Saves that may wait behind the copy in progress. Each holds its pinned slot.
const QUEUE_DEPTH: usize = 2;
/// Consecutive failed writes (a full or read-only disk) after which the tier stops writing.
const WRITE_FAILURES_BEFORE_PAUSE: u32 = 3;
/// How long the tier stays away from the disk after those failures.
const WRITE_PAUSE: Duration = Duration::from_secs(600);

/// Why an entry goes to disk. The disk is used only when pinned memory is short.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reason {
    /// The store passed its watermark: this is one of the entries the next saves push out.
    Watermark,
    /// The engine is stopping and this entry exists only in memory.
    Stop,
}

impl Reason {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Reason::Watermark => "watermark",
            Reason::Stop => "stop",
        }
    }
}

/// The bytes of a held slot, read without moving them. The pinned buffer is the only production
/// source; a test supplies memory.
pub(crate) trait SlotBytes {
    fn read(&self, at: usize, out: &mut [u8]) -> Result<(), EngError>;
}

/// A pinned entry waiting to be copied to disk, holding its slot.
pub(crate) struct Held {
    pub(crate) reason: Reason,
    pub(crate) slot: Rc<dyn SlotBytes>,
    pub(crate) key: Vec<u32>,
    pub(crate) capture: Capture,
    pub(crate) positions: usize,
    /// The bytes of the entry's objects.
    pub(crate) bytes: u64,
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
pub(crate) struct Hit {
    pub(crate) key_len: usize,
    pub(crate) bytes: u64,
    pub(crate) ms: f64,
}

/// Entries and bytes written for one reason.
#[derive(Default, Clone, Copy)]
pub(crate) struct Spilled {
    pub(crate) entries: u64,
    pub(crate) bytes: u64,
}

#[derive(Default)]
pub(crate) struct Counters {
    pub(crate) watermark: Spilled,
    pub(crate) stop: Spilled,
    /// Spills dropped before they were written: their slot was needed by a save.
    pub(crate) dropped: u64,
    pub(crate) saved: u64,
    pub(crate) skipped: u64,
    pub(crate) failed: u64,
    pub(crate) restored: u64,
    pub(crate) missed: u64,
}

pub(crate) struct DiskTier {
    /// What the family says its engine is called in the log lines (`q36 cuda`).
    label: &'static str,
    pub(crate) cache: KvDisk,
    /// Saves waiting for the copy in progress, oldest first, at most `QUEUE_DEPTH`.
    queued: Vec<Held>,
    copying: Option<Copying>,
    /// Jobs whose bytes are all handed to the writer and that have not reported yet.
    finishing: usize,
    /// The reason of every job that has begun and not yet reported.
    inflight: Vec<(u64, Reason)>,
    /// The retained slots of the pinned pool, which set the watermark.
    pub(crate) retained: usize,
    pub(crate) hit: Option<Hit>,
    pub(crate) counters: Counters,
    /// Writes that failed one after another since the last one that worked.
    failures: u32,
    /// While set and in the future, nothing is queued: the disk refused several writes in a row.
    paused_until: Option<Instant>,
}

impl DiskTier {
    pub(crate) fn new(label: &'static str, cache: KvDisk, retained: usize) -> Self {
        Self {
            label,
            cache,
            queued: Vec::new(),
            copying: None,
            finishing: 0,
            inflight: Vec::new(),
            retained,
            hit: None,
            counters: Counters::default(),
            failures: 0,
            paused_until: None,
        }
    }

    /// Whether the tier is staying away from the disk at `now`. The pause ends by itself.
    pub(crate) fn paused(&mut self, now: Instant) -> bool {
        let label = self.label;
        match self.paused_until {
            Some(until) if now < until => true,
            Some(_) => {
                self.paused_until = None;
                self.failures = 0;
                eprintln!("{label}: kv disk writes resume after the pause");
                false
            }
            None => false,
        }
    }

    /// Records one failed write; after `WRITE_FAILURES_BEFORE_PAUSE` in a row drops what waits and
    /// pauses, with one line instead of one per save.
    pub(crate) fn note_failure(&mut self, now: Instant) {
        let label = self.label;
        self.counters.failed += 1;
        self.failures += 1;
        if self.failures >= WRITE_FAILURES_BEFORE_PAUSE && self.paused_until.is_none() {
            self.paused_until = Some(now + WRITE_PAUSE);
            let waiting = self.queued.drain(..).count();
            self.counters.dropped += waiting as u64;
            eprintln!(
                "{label}: kv disk: {} writes failed in a row; no writes for {} s ({waiting} waiting spills dropped)",
                self.failures,
                WRITE_PAUSE.as_secs()
            );
        }
    }

    /// When no pinned slot is free and spills hold them, drops the spills (counted) so a save or
    /// a restore can take a slot instead of failing or waiting. Returns how many were dropped.
    pub(crate) fn drop_spills_for_slots(&mut self, free_slots: usize, needed_by: &str) -> usize {
        let label = self.label;
        if free_slots > 0 {
            return 0;
        }
        let mut dropped = self.queued.drain(..).count();
        if let Some(copying) = self.copying.take() {
            self.cache.abort(&copying.job);
            self.inflight.retain(|(id, _)| *id != copying.job.id());
            dropped += 1;
        }
        if dropped > 0 {
            self.counters.dropped += dropped as u64;
            eprintln!(
                "{label}: kv disk: {dropped} spills dropped, their pinned slots are needed by {needed_by}"
            );
        }
        dropped
    }

    /// Copies up to `budget` bytes of pending saves to the writer and logs what finished.
    /// `shape_of` names the objects of an entry of a given number of positions.
    pub(crate) fn pump(
        &mut self,
        shape_of: &dyn Fn(usize) -> Result<SlabLayout, EngError>,
        budget: usize,
    ) {
        let label = self.label;
        // The writer sends a job's report before it releases the job, so an idle cache means
        // every report is already waiting: read the flag first, then the reports.
        let idle = self.cache.is_idle();
        for event in self.cache.poll() {
            self.finishing = self.finishing.saturating_sub(1);
            match event {
                WriteEvent::Done {
                    id,
                    positions,
                    bytes,
                    ms,
                } => {
                    self.counters.saved += 1;
                    self.failures = 0;
                    let why = take_reason(&mut self.inflight, id);
                    let spilled = match why {
                        Reason::Watermark => &mut self.counters.watermark,
                        Reason::Stop => &mut self.counters.stop,
                    };
                    spilled.entries += 1;
                    spilled.bytes += bytes;
                    eprintln!(
                        "{label}: kv disk saved {positions} positions, {bytes} B ({}), {ms:.1} ms; to disk so far: {}",
                        why.name(),
                        self.counters.summary()
                    );
                }
                WriteEvent::Failed { id, reason } => {
                    take_reason(&mut self.inflight, id);
                    self.note_failure(Instant::now());
                    eprintln!("{label}: kv disk save failed: {reason}");
                }
            }
        }
        if idle && self.copying.is_none() {
            self.finishing = 0;
        }
        let mut left = budget;
        while left > 0 {
            if self.copying.is_none() {
                if self.queued.is_empty() {
                    return;
                }
                // Tips go before periodic and block entries, each oldest first.
                let next = self
                    .queued
                    .iter()
                    .position(|held| held.capture == Capture::Tip)
                    .unwrap_or(0);
                let held = self.queued.remove(next);
                let shape = match shape_of(held.positions) {
                    Ok(shape) => shape,
                    Err(error) => {
                        eprintln!("{label}: kv disk save skipped: {error:?}");
                        continue;
                    }
                };
                let spec = WriteSpec {
                    tokens: held.key.clone(),
                    capture: held.capture,
                    object_bytes: shape.object_bytes(),
                };
                let evicted_before = self.cache.evicted();
                match self.cache.begin_write(spec) {
                    Ok(job) => {
                        self.inflight.push((job.id(), held.reason));
                        let evicted = self.cache.evicted() - evicted_before;
                        if evicted > 0 {
                            eprintln!(
                                "{label}: kv disk quota: {evicted} least recently used entries removed to make room for {} positions",
                                held.positions
                            );
                        }
                        self.copying = Some(Copying {
                            job,
                            runs: shape.flat(),
                            held,
                            at: 0,
                            within: 0,
                        });
                    }
                    Err(skip) => {
                        self.counters.skipped += 1;
                        eprintln!(
                            "{label}: kv disk save skipped for {} positions: {skip}",
                            held.positions
                        );
                        continue;
                    }
                }
            }
            let Some(copying) = self.copying.as_mut() else {
                return;
            };
            match copy_chunk(copying, left.min(CHUNK_BYTES)) {
                Ok((chunk, done)) => {
                    let sent = chunk.len();
                    match self.cache.push(&copying.job, chunk) {
                        Push::Sent => left = left.saturating_sub(sent.max(1)),
                        Push::Full(back) => {
                            // Put the cursor back: the writer is behind, try again later.
                            rewind(copying, back.len());
                            return;
                        }
                        Push::Dead => {
                            self.note_failure(Instant::now());
                            eprintln!("{label}: kv disk save failed: the writer stopped");
                            self.copying = None;
                            return;
                        }
                    }
                    if done {
                        self.copying = None;
                        self.finishing += 1;
                    }
                }
                Err(error) => {
                    self.cache.abort(&copying.job);
                    self.inflight.retain(|(id, _)| *id != copying.job.id());
                    self.counters.failed += 1;
                    eprintln!("{label}: kv disk save failed: {error:?}");
                    self.copying = None;
                }
            }
        }
    }

    /// The disk half of a graceful stop: `tips` are the entries only in memory, newest first.
    /// Waiting spills that are tips stay (behind them), anything else is dropped. Pumps the
    /// writer until it is done, the bound from `bound_of(remaining bytes, recent write rate)` passes
    /// or `hurry` is set.
    pub(crate) fn drain(
        &mut self,
        mut tips: Vec<Held>,
        left_in_memory: usize,
        shape_of: &dyn Fn(usize) -> Result<SlabLayout, EngError>,
        hurry: &AtomicBool,
        bound_of: &dyn Fn(u64, Option<f64>) -> Duration,
    ) {
        let label = self.label;
        if self.paused(Instant::now()) {
            eprintln!(
                "{label}: kv disk stop: writes are paused after repeated failures; {} unsaved entries stay in memory only",
                tips.len() + self.queued.len()
            );
            return;
        }
        tips.retain(|held| !self.has(&held.key));
        let mut dropped = 0usize;
        // A waiting spill that is a tip stays (behind the new ones); anything else goes.
        for held in std::mem::take(&mut self.queued) {
            if held.capture == Capture::Tip {
                tips.push(Held {
                    reason: Reason::Stop,
                    ..held
                });
            } else {
                dropped += 1;
            }
        }
        if self
            .copying
            .as_ref()
            .is_some_and(|copying| copying.held.capture != Capture::Tip)
            && let Some(copying) = self.copying.take()
        {
            self.cache.abort(&copying.job);
            self.inflight.retain(|(id, _)| *id != copying.job.id());
            dropped += 1;
        }
        self.queued = tips;
        if !self.pending() {
            eprintln!(
                "{label}: kv disk stop: no tip exists only in memory; to disk: {}",
                self.counters.summary()
            );
            return;
        }
        let remaining: u64 = self.queued.iter().map(|held| held.bytes).sum::<u64>()
            + self.copying.as_ref().map_or(0, Copying::remaining);
        let rate = self.cache.write_rate();
        let bound = bound_of(remaining, rate);
        let count = self.queued.len() + usize::from(self.copying.is_some());
        eprintln!(
            "{label}: kv disk stop: {count} tips only in memory ({remaining} B to move), {dropped} waiting spills dropped, {left_in_memory} other entries stay in memory only; waiting at most {:.0} s at {}; a second stop signal ends the wait",
            bound.as_secs_f64(),
            rate.map_or_else(
                || format!("an assumed {:.0} MB/s", ASSUMED_WRITE_RATE / 1e6),
                |rate| format!("{:.1} MB/s, the rate of its last saves", rate / 1e6)
            )
        );
        let began = Instant::now();
        let mut hurried = false;
        while self.pending() && began.elapsed() < bound {
            if hurry.load(Ordering::SeqCst) {
                hurried = true;
                break;
            }
            self.pump(shape_of, PUMP_IDLE_BYTES);
            std::thread::sleep(Duration::from_millis(2));
        }
        if self.pending() {
            if let Some(copying) = self.copying.take() {
                self.cache.abort(&copying.job);
            }
            eprintln!(
                "{label}: kv disk stop: gave up after {:.1} s ({}) with {} saves unwritten; to disk: {}",
                began.elapsed().as_secs_f64(),
                if hurried {
                    "second stop signal"
                } else {
                    "the bound passed"
                },
                self.queued.len() + self.finishing,
                self.counters.summary()
            );
        } else {
            eprintln!(
                "{label}: kv disk stop: tips written in {:.1} s; to disk: {}",
                began.elapsed().as_secs_f64(),
                self.counters.summary()
            );
        }
    }

    /// Reads `candidate` from disk, handing every verified byte to `place` (slab offset, bytes).
    /// A miss is logged and counted. The error says whether the entry was damaged or gone and
    /// has been removed, so another lookup may name a shorter stored prefix; any other miss
    /// leaves the prompt to a cold prefill.
    pub(crate) fn read_into(
        &mut self,
        candidate: &Candidate,
        prompt: &[u32],
        shape: &SlabLayout,
        place: &mut Put<'_>,
    ) -> Result<ReadStats, bool> {
        let label = self.label;
        let read = self.cache.read(
            candidate,
            prompt,
            &shape.object_bytes(),
            &mut |object, offset, bytes| {
                shape
                    .scatter(object, offset, bytes, place)
                    .map_err(|error| DiskMiss::Refused(format!("{error:?}")))
            },
        );
        read.map_err(|miss| {
            self.counters.missed += 1;
            let removed = matches!(miss, DiskMiss::Corrupt(_) | DiskMiss::Absent);
            eprintln!(
                "{label}: kv disk miss for {} tokens ({miss}){}",
                candidate.key_len,
                if removed {
                    "; the entry is removed"
                } else {
                    "; the prompt is prefilled"
                }
            );
            removed
        })
    }

    pub(crate) fn pending(&self) -> bool {
        !self.queued.is_empty() || self.copying.is_some() || self.finishing > 0
    }

    /// Whether `key` is already on disk, waiting for the writer or being copied.
    pub(crate) fn has(&self, key: &[u32]) -> bool {
        self.queued.iter().any(|held| held.key == key)
            || self.copying.as_ref().is_some_and(|c| c.held.key == key)
            || self.cache.has_exact(key)
    }

    /// Adds a spill behind the copy in progress. A waiting tip that the new tip extends is dead;
    /// a full queue drops its least valuable entry (a periodic one before a tip, oldest first),
    /// or the offer itself when it is a periodic entry and every waiter is a tip.
    pub(crate) fn enqueue(&mut self, offer: Held) {
        if self.has(&offer.key) {
            return;
        }
        if self.paused(Instant::now()) {
            self.counters.skipped += 1;
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

/// The first line of a started tier: where it lives, how much it may hold, what it found, what it
/// keeps (the conversations' exact tokens), how to turn it off, and how long a stop may wait.
pub(crate) fn start_line(
    label: &str,
    resolved: &Resolved,
    origin: &str,
    report: &OpenReport,
    retained: usize,
) -> String {
    format!(
        "{label}: kv disk tier on ({origin}), dir {dir}, quota {quota} B, {} entries of this load found ({} of other loads, {} removed, {} stray objects removed), {} B used; written only when pinned memory is short (from {} of {retained} entries) and at a graceful stop, which may wait up to {} s for unsaved prefixes (a second stop signal ends the wait); conversations (their exact token lists and model state) persist on disk at {dir} up to {quota} B; turn it off with kv_cache.enabled false or FIFI_KV_CACHE=0; purge with --purge-kv-cache",
        report.entries,
        report.foreign,
        report.dropped,
        report.orphans,
        report.used_bytes,
        watermark(retained),
        drain_bound(u64::MAX, None).as_secs(),
        dir = resolved.dir.display(),
        quota = resolved.quota_bytes,
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
    let slot = &copying.held.slot;
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

#[cfg(test)]
#[path = "tier_tests.rs"]
mod tests;
