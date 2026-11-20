#![allow(unsafe_code)]
//! The pooled backend over the expert bank's O_DIRECT ring: the production
//! ring, wrapped unchanged.
//!
//! Unsafe scope: the lease's `Send` impl and its one `Deref` block — both
//! carry the same SAFETY argument, pinned slot memory nobody writes until
//! the lease drops. The source-scan test below pins that scope.
//!
//! Every record lands in one ring slot: the batch acquires slots, stages
//! each record's aligned span, places the exact bytes at the slot front,
//! and arms the slot's copy phase with the host fence. The lease IS the
//! slot: it dereferences to the record's bytes in the pinned slot memory
//! and releases the slot back to the ring on drop. A batch larger than the
//! slot count is refused before any I/O; a batch that cannot gather enough
//! free slots within the store's wait bound fails instead of blocking
//! forever — a caller holding this store's leases must not expect a new
//! batch to be satisfied by those same slots. Alignment is the ring's:
//! sector-aligned, staged, and placed here, never exposed to the caller.
//! The file handle is the caller's: open it through
//! `expert_bank::host_tier::open_direct` for the production O_DIRECT path
//! (a plain page-cache handle also works and simply bypasses O_DIRECT;
//! the store does not check).

use std::fs::File;
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::family::expert_bank::host_tier::{
    CopyFence, ODirectRing, SECTOR, Slot as RingSlot, align_up_sector, aligned_span, open_direct,
};

use super::{
    ExpertRecordStore, RecordSite, buffered::pread_at_most, index_arithmetic_overflow,
    record_size_not_addressable, site_outside_store,
};
use crate::EngError;

/// The fence for a record whose bytes are final when `read_records`
/// returns: the synchronous host read staged them and nothing writes the
/// slot again before release. It is the fence ONLY because this backend
/// ends at host bytes — a caller that enqueues an async device upload
/// FROM a lease must hold that lease until its own completion event
/// fires, and a device-copy backend records the real event fence instead;
/// the always-complete answer is honest only on this host path.
struct HostBytesFinal;

impl CopyFence for HostBytesFinal {
    fn is_complete(&self) -> bool {
        true
    }
}

/// A lease over one record's slot: dereferences to the record's exact
/// bytes in the ring's pinned memory and releases the slot on drop.
pub(crate) struct SlotLease {
    ring: Arc<Mutex<ODirectRing>>,
    slot: RingSlot,
    ptr: *const u8,
    len: usize,
}

// SAFETY: the pointer aims into the ring's process-lifetime pinned memory.
// The slot stays in the copy phase — nobody writes it — until this lease
// drops, and release (the only path back to the free list) happens under
// the ring's mutex after the drop. Moving the lease to another thread
// moves only these facts; they hold wherever the lease is read.
unsafe impl Send for SlotLease {}

impl Deref for SlotLease {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: see the `Send` impl; the window this pointer came from
        // is valid until the slot is released, which only this lease's
        // drop does.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        // The host fence is complete by construction, so release cannot
        // refuse; if it ever does, the slot leaks rather than panicking
        // inside drop.
        if let Ok(mut ring) = self.ring.lock() {
            let _ = ring.release(self.slot);
        }
    }
}

/// One extent of a span-addressed record: the payload's window in
/// `shard`, and where the payload lands in the slot's record area. The
/// store's sites carry these merged where extents abut, so one read
/// serves what the pack wrote contiguously.
#[derive(Clone)]
pub(crate) struct SpanRead {
    pub(crate) shard: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) len: u64,
    pub(crate) record_offset: u64,
}

/// How the span worker opens a shard. Decided once at start by the
/// caller's probe of every shard — O_DIRECT where every one answers it,
/// buffered otherwise — and carried in every job, so no read ever
/// re-decides and a direct-mode shard that stops answering fails loudly
/// instead of silently switching regimes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ShardOpen {
    Direct,
    Buffered,
}

/// One pooled read. The flat grid reads one aligned window into the
/// slot's span at offset 0 (the payload is `place`d afterwards); a
/// span-addressed record stages each extent's aligned window in the
/// WORKER's own scratch and copies the payload straight to its record
/// offset, so the slot's span is touched only through this job — no
/// ring-lock `place` on the span path.
enum Job {
    Flat {
        file: File,
        slot: RingSlot,
        span: &'static mut [u8],
        start: u64,
        total: usize,
    },
    Spans {
        slot: RingSlot,
        span: &'static mut [u8],
        reads: Vec<SpanRead>,
        /// The largest aligned window any extent of any site needs; the
        /// worker's scratch grows to this once.
        max_window: usize,
        mode: ShardOpen,
    },
}

impl Job {
    fn slot(&self) -> RingSlot {
        match self {
            Job::Flat { slot, .. } | Job::Spans { slot, .. } => *slot,
        }
    }

    fn into_span(self) -> &'static mut [u8] {
        match self {
            Job::Flat { span, .. } | Job::Spans { span, .. } => span,
        }
    }
}

/// A finished pooled read: the span back, and either the payload bytes
/// read or a fully composed message ("short: ..." for a file that ends
/// inside a record, "io: ..." for a failed read). Slot ids ride along so
/// the reaper can match outcomes to records — workers finish out of
/// order.
struct Outcome {
    slot: RingSlot,
    span: &'static mut [u8],
    got: Result<usize, String>,
}

/// One outstanding positioned read per thread. The threads park on the
/// job channel between batches and are joined when the store drops — a
/// start that fails after building the store leaves none behind. A start
/// that keeps the store for the process lifetime (the disk rung leaks
/// its reader) keeps the workers for that same lifetime.
struct ReadPool {
    workers: Vec<JoinHandle<()>>,
    job_tx: Option<Sender<Job>>,
    outcome_rx: Mutex<Receiver<Outcome>>,
}

impl ReadPool {
    fn spawn(depth: usize) -> Self {
        let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
        let (outcome_tx, outcome_rx) = std::sync::mpsc::channel::<Outcome>();
        let job_rx = Arc::new(Mutex::new(job_rx));
        let workers = (0..depth)
            .map(|_| {
                let job_rx = Arc::clone(&job_rx);
                let outcome_tx = outcome_tx.clone();
                std::thread::Builder::new()
                    .name("expert-record-read".into())
                    .spawn(move || {
                        // Per-worker, for the worker's life: the staging
                        // scratch of the span path (sector-aligned, grown
                        // once to the largest window any job asks for) and
                        // the cached shard handles — positioned reads never
                        // move a shared offset.
                        let mut scratch = AlignedScratch::default();
                        let mut shards: std::collections::HashMap<PathBuf, File> =
                            std::collections::HashMap::new();
                        loop {
                            let job = {
                                let guard = job_rx.lock().expect("job channel");
                                guard.recv()
                            };
                            let job = match job {
                                Ok(job) => job,
                                // The store is gone: nothing more to serve.
                                Err(_) => break,
                            };
                            match job {
                                Job::Flat {
                                    file,
                                    slot,
                                    span,
                                    start,
                                    total,
                                } => {
                                    let got = pread_at_most(&file, &mut span[..total], start)
                                        .map_err(|e| {
                                            format!("io: read at byte {start} failed: {e}")
                                        });
                                    if finish(&outcome_tx, slot, span, got) {
                                        break;
                                    }
                                }
                                Job::Spans {
                                    slot,
                                    span,
                                    reads,
                                    max_window,
                                    mode,
                                } => {
                                    let got = read_spans_into(
                                        &mut scratch,
                                        &mut shards,
                                        span,
                                        &reads,
                                        max_window,
                                        mode,
                                    );
                                    if finish(&outcome_tx, slot, span, got) {
                                        break;
                                    }
                                }
                            }
                        }
                    })
                    .expect("spawn an expert record read worker")
            })
            .collect();
        // The pool holds no outcome sender: the channel closes when the
        // last worker exits, which is the drain signal a dropping store
        // relies on.
        drop(outcome_tx);
        Self {
            workers,
            job_tx: Some(job_tx),
            outcome_rx: Mutex::new(outcome_rx),
        }
    }

    fn send(&self, job: Job) -> Result<(), std::sync::mpsc::SendError<Job>> {
        self.job_tx
            .as_ref()
            .expect("the pool outlives every fill_batch call")
            .send(job)
    }

    fn recv(&self) -> Result<Outcome, std::sync::mpsc::RecvError> {
        self.outcome_rx.lock().expect("outcome channel").recv()
    }

    /// Drops the job channel and joins every worker. Called by the
    /// store's Drop; never leaves a thread behind.
    fn shutdown(&mut self) {
        self.job_tx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Sends one outcome back; `true` when the store side is gone (stop).
fn finish(
    outcome_tx: &Sender<Outcome>,
    slot: RingSlot,
    span: &'static mut [u8],
    got: Result<usize, String>,
) -> bool {
    outcome_tx.send(Outcome { slot, span, got }).is_err()
}

/// A worker's staging scratch: a `Vec` grown with a page of slop so the
/// window it hands out is sector-aligned — O_DIRECT refuses a
/// misaligned destination, and a plain `Vec`'s own pointer is only
/// 16-byte aligned. The interior is recomputed per call; a resize may
/// move the buffer.
#[derive(Default)]
struct AlignedScratch {
    raw: Vec<u8>,
}

impl AlignedScratch {
    /// A sector-aligned `bytes`-long window, zero-filled on growth.
    fn window(&mut self, bytes: usize) -> &mut [u8] {
        const ALIGN: usize = 4096;
        if self.raw.len() < bytes + ALIGN {
            self.raw.resize(bytes + ALIGN - 1, 0);
        }
        let address = self.raw.as_ptr() as usize;
        let head = (ALIGN - address % ALIGN) % ALIGN;
        &mut self.raw[head..head + bytes]
    }
}

/// Reads one span-addressed record into its loaned slot span: each
/// extent's aligned window is staged in `scratch` (O_DIRECT needs the
/// sector-aligned destination; a record's interior is not) and its
/// payload copied to the extent's record offset. Short files and failed
/// reads return fully composed messages ("short: ..." / "io: ..."); the
/// payload never lands partially. Every shard opens in the store's
/// start-decided `mode` — a direct-mode shard that stops answering
/// O_DIRECT is a filesystem that changed under the running store, and
/// the read fails loudly rather than silently switching regimes.
fn read_spans_into(
    scratch: &mut AlignedScratch,
    shards: &mut std::collections::HashMap<PathBuf, File>,
    span: &mut [u8],
    reads: &[SpanRead],
    max_window: usize,
    mode: ShardOpen,
) -> Result<usize, String> {
    let mut payload = 0usize;
    for read in reads {
        let file = match shards.get(&read.shard) {
            Some(file) => file,
            None => {
                let opened = match mode {
                    ShardOpen::Direct => open_direct(&read.shard).map_err(|e| {
                        format!(
                            "io: {} no longer serves O_DIRECT (it did at start): {e}",
                            read.shard.display()
                        )
                    })?,
                    ShardOpen::Buffered => std::fs::File::open(&read.shard).map_err(|e| {
                        format!("io: {} cannot open buffered: {e}", read.shard.display())
                    })?,
                };
                shards.insert(read.shard.clone(), opened);
                shards.get(&read.shard).expect("just inserted")
            }
        };
        let (start, total) = aligned_span(read.file_offset, read.len);
        let head = (read.file_offset - start) as usize;
        let staged = scratch.window(max_window);
        let got = pread_at_most(file, &mut staged[..total as usize], start).map_err(|e| {
            format!(
                "io: read at byte {start} of {} failed: {e}",
                read.shard.display()
            )
        })?;
        if got < head + read.len as usize {
            return Err(format!(
                "short: {} ends inside the record at byte {}",
                read.shard.display(),
                read.file_offset + (got as u64).saturating_sub(head as u64),
            ));
        }
        span[read.record_offset as usize..][..read.len as usize]
            .copy_from_slice(&staged[head..head + read.len as usize]);
        payload += read.len as usize;
    }
    Ok(payload)
}

/// How sites become disk reads. The flat grid keeps the buffered
/// backend's arithmetic (offset = index x record_bytes in one file); the
/// span grid carries per-site extent lists — a pack's records live as
/// component tensors at arbitrary offsets across shard files.
enum Addressing {
    Flat {
        file: File,
    },
    Spans {
        sites: Vec<Vec<SpanRead>>,
        max_window: usize,
    },
}

/// An expert record store backed by the expert bank's O_DIRECT ring: one
/// pinned slot per record, staged and placed by the ring. The flat grid is
/// the same as the buffered backend's.
pub(crate) struct RingRecords {
    addressing: Arc<Addressing>,
    ring: Arc<Mutex<ODirectRing>>,
    layers: usize,
    experts_per_layer: usize,
    record_bytes: u64,
    slots: usize,
    /// How span jobs open shards — decided once by the caller's probe,
    /// carried in every job.
    shard_open: ShardOpen,
    max_wait: Duration,
    /// One read worker per slot (the pool's depth IS the slot count: a
    /// batch cannot have more reads in flight than it has slots to land
    /// them in). Threads park between batches; a host that never misses
    /// pays only their parked stacks.
    pool: ReadPool,
    /// Serializes batches internally so the shared outcome channel needs
    /// no batch tags: one batch's reads are reaped before the next
    /// dispatches. Reads within a batch still fly at full pool depth.
    fill_lock: Mutex<()>,
}

impl RingRecords {
    /// `layers` x `experts_per_layer` records of exactly `record_bytes`
    /// each, read from `file` through the ring built over `mem` (the
    /// caller's pinned, sector-aligned memory of `slots` slots). A batch
    /// that cannot gather its slots within `max_wait` fails. A zero record
    /// size, an empty pool, or a grid whose extent does not fit its
    /// integer types is an index defect refused here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        file: File,
        layers: usize,
        experts_per_layer: usize,
        record_bytes: u64,
        slots: usize,
        mem: &'static mut [u8],
        max_wait: Duration,
    ) -> Result<Self, EngError> {
        if record_bytes == 0 {
            return Err(EngError::Checkpoint(
                "expert record size must be nonzero".into(),
            ));
        }
        let _ =
            usize::try_from(record_bytes).map_err(|_| record_size_not_addressable(record_bytes))?;
        if record_bytes > isize::MAX as u64 {
            // An allocation larger than isize::MAX is invalid by
            // definition; catching it here beats an aborting vec! at read
            // time.
            return Err(record_size_not_addressable(record_bytes));
        }
        let records = layers
            .checked_mul(experts_per_layer)
            .and_then(|r| u64::try_from(r).ok())
            .ok_or_else(|| index_arithmetic_overflow("record count"))?;
        let _ = records
            .checked_mul(record_bytes)
            .ok_or_else(|| index_arithmetic_overflow("grid byte extent"))?;
        let slots = NonZeroUsize::new(slots)
            .ok_or_else(|| EngError::Checkpoint("a slot pool needs at least one slot".into()))?;
        // One record's aligned span plus a sector covers the worst head
        // and tail padding at any file offset.
        let slot_span = NonZeroU64::new(align_up_sector(record_bytes) + SECTOR)
            .ok_or_else(|| index_arithmetic_overflow("slot span"))?;
        let ring = ODirectRing::new(slots, slot_span, mem).map_err(EngError::Checkpoint)?;
        Ok(Self {
            addressing: Arc::new(Addressing::Flat { file }),
            ring: Arc::new(Mutex::new(ring)),
            layers,
            experts_per_layer,
            record_bytes,
            slots: slots.get(),
            // The flat grid reads through the caller's handle; the span
            // open mode never applies to it.
            shard_open: ShardOpen::Direct,
            max_wait,
            pool: ReadPool::spawn(slots.get()),
            fill_lock: Mutex::new(()),
        })
    }

    /// `layers` x `experts_per_layer` span-addressed records: `sites` in
    /// the trait's site order (layer-major), each entry that record's
    /// extents, already merged where they abut in one file. Every site
    /// must tile from byte 0 without gap or overlap and stay within
    /// `record_bytes` (a site may be shorter — its lease is). The grid
    /// checks mirror the flat constructor's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_span_addressed(
        sites: Vec<Vec<SpanRead>>,
        layers: usize,
        experts_per_layer: usize,
        record_bytes: u64,
        slots: usize,
        mode: ShardOpen,
        mem: &'static mut [u8],
        max_wait: Duration,
    ) -> Result<Self, EngError> {
        if record_bytes == 0 {
            return Err(EngError::Checkpoint(
                "expert record size must be nonzero".into(),
            ));
        }
        if record_bytes > isize::MAX as u64 {
            return Err(record_size_not_addressable(record_bytes));
        }
        let expected = layers
            .checked_mul(experts_per_layer)
            .ok_or_else(|| index_arithmetic_overflow("record count"))?;
        if sites.len() != expected {
            return Err(EngError::Checkpoint(format!(
                "the store covers {expected} sites, {} were given",
                sites.len()
            )));
        }
        let mut max_window = 0usize;
        for (index, site) in sites.iter().enumerate() {
            let mut sorted: Vec<&SpanRead> = site.iter().collect();
            sorted.sort_by_key(|read| read.record_offset);
            let mut cursor = 0u64;
            for read in sorted {
                if read.record_offset != cursor {
                    return Err(EngError::Checkpoint(format!(
                        "site {index}'s extents do not tile: one starts at {} where the record ends at {cursor}",
                        read.record_offset
                    )));
                }
                if read
                    .record_offset
                    .checked_add(read.len)
                    .map(|end| end > record_bytes)
                    .unwrap_or(true)
                {
                    return Err(EngError::Checkpoint(format!(
                        "site {index}'s extents reach past the record's {record_bytes} bytes"
                    )));
                }
                cursor += read.len;
                let (_, total) = aligned_span(read.file_offset, read.len);
                max_window = max_window.max(total as usize);
            }
        }
        let slots = NonZeroUsize::new(slots)
            .ok_or_else(|| EngError::Checkpoint("a slot pool needs at least one slot".into()))?;
        let slot_span = NonZeroU64::new(align_up_sector(record_bytes) + SECTOR)
            .ok_or_else(|| index_arithmetic_overflow("slot span"))?;
        let ring = ODirectRing::new(slots, slot_span, mem).map_err(EngError::Checkpoint)?;
        Ok(Self {
            addressing: Arc::new(Addressing::Spans { sites, max_window }),
            ring: Arc::new(Mutex::new(ring)),
            layers,
            experts_per_layer,
            record_bytes,
            slots: slots.get(),
            shard_open: mode,
            max_wait,
            pool: ReadPool::spawn(slots.get()),
            fill_lock: Mutex::new(()),
        })
    }

    fn linear_index(&self, site: RecordSite) -> Result<u64, EngError> {
        if site.layer >= self.layers || usize::from(site.expert) >= self.experts_per_layer {
            return Err(site_outside_store(site));
        }
        let index = site
            .layer
            .checked_mul(self.experts_per_layer)
            .and_then(|l| l.checked_add(usize::from(site.expert)))
            .ok_or_else(|| index_arithmetic_overflow("linear index"))?;
        u64::try_from(index).map_err(|_| index_arithmetic_overflow("linear index"))
    }

    /// Acquires exactly `wanted` slots, or fails after the wait bound. The
    /// lock is dropped between attempts so other threads' lease drops can
    /// return slots while this batch waits.
    fn checkout(&self, wanted: usize) -> Result<Vec<RingSlot>, EngError> {
        let deadline = Instant::now() + self.max_wait;
        loop {
            let gathered = {
                let mut ring = self.ring.lock().expect("ring lock");
                let mut got: Vec<RingSlot> = Vec::with_capacity(wanted);
                while got.len() < wanted {
                    match ring.acquire() {
                        Some(slot) => got.push(slot),
                        None => {
                            for slot in got.drain(..) {
                                let _ = ring.abandon(slot);
                            }
                            break;
                        }
                    }
                }
                got
            };
            if gathered.len() == wanted {
                return Ok(gathered);
            }
            if Instant::now() >= deadline {
                return Err(EngError::Backend {
                    operation: "expert record read",
                    message: format!(
                        "could not gather {wanted} of {} slots within the wait bound; drop outstanding leases and retry",
                        self.slots
                    ),
                });
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for RingRecords {
    /// Joins the read pool: the store's last act is to leave no thread
    /// behind (a failed start that already built the reader takes this
    /// path through drop too).
    fn drop(&mut self) {
        self.pool.shutdown();
    }
}

impl ExpertRecordStore for RingRecords {
    type Lease = SlotLease;

    fn read_records(&self, sites: &[RecordSite]) -> Result<Vec<SlotLease>, EngError> {
        if sites.len() > self.slots {
            return Err(EngError::BadRequest(format!(
                "batch of {} records exceeds the pool's {} slots",
                sites.len(),
                self.slots
            )));
        }
        // Every site is validated before any slot is taken: a refusal
        // leaves the pool untouched.
        let mut offsets: Vec<u64> = Vec::with_capacity(sites.len());
        for &s in sites {
            let index = self.linear_index(s)?;
            offsets.push(
                index
                    .checked_mul(self.record_bytes)
                    .ok_or_else(|| index_arithmetic_overflow("record byte offset"))?,
            );
        }
        let slots = self.checkout(sites.len())?;
        let mut leases: Vec<SlotLease> = Vec::with_capacity(slots.len());
        if let Err(e) = self.fill_batch(&slots, &offsets, &mut leases) {
            // The batch fails as a unit. Return the slot that failed, in
            // whatever state it stopped (release covers an armed slot,
            // abandon a merely-reading one), and every slot the loop never
            // reached. The already-leased slots go back through their own
            // drops — AFTER this guard ends, because a lease's drop takes
            // the same mutex.
            {
                let mut ring = self.ring.lock().expect("ring lock");
                for &s in &slots[leases.len()..] {
                    if ring.release(s).is_err() {
                        let _ = ring.abandon(s);
                    }
                }
            }
            drop(leases);
            return Err(e);
        }
        Ok(leases)
    }

    fn record_bytes(&self) -> u64 {
        self.record_bytes
    }

    fn max_batch(&self) -> usize {
        self.slots
    }
}

impl RingRecords {
    /// A single-slot lane over the same index with its own checkout pool and fill lock.
    /// Demand reads can proceed even while this lane has a read in flight.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(crate) fn one_slot_reader(&self, mem: &'static mut [u8]) -> Result<Self, EngError> {
        let span = NonZeroU64::new(align_up_sector(self.record_bytes) + SECTOR)
            .ok_or_else(|| index_arithmetic_overflow("slot span"))?;
        let slots = NonZeroUsize::MIN;
        let ring = ODirectRing::new(slots, span, mem).map_err(EngError::Checkpoint)?;
        Ok(Self {
            addressing: Arc::clone(&self.addressing),
            ring: Arc::new(Mutex::new(ring)),
            layers: self.layers,
            experts_per_layer: self.experts_per_layer,
            record_bytes: self.record_bytes,
            slots: 1,
            shard_open: self.shard_open,
            max_wait: self.max_wait,
            pool: ReadPool::spawn(1),
            fill_lock: Mutex::new(()),
        })
    }

    /// Fills a batch through the read pool — dispatch, reap, lease — the
    /// shape both addressings share; the per-mode work lives in
    /// [`Self::fill_flat`] and [`Self::fill_spans`]. On error the batch's
    /// cleanup belongs to the caller: every dispatched read is reaped
    /// and its span returned before this returns, so no slot's memory
    /// stays on loan, and no lease is dropped here.
    fn fill_batch(
        &self,
        slots: &[RingSlot],
        offsets: &[u64],
        leases: &mut Vec<SlotLease>,
    ) -> Result<(), EngError> {
        debug_assert_eq!(slots.len(), offsets.len());
        let _fill = self.fill_lock.lock().expect("fill lock");
        match self.addressing.as_ref() {
            Addressing::Flat { file } => self.fill_flat(file, slots, offsets, leases),
            Addressing::Spans { sites, max_window } => {
                self.fill_spans(sites, *max_window, slots, offsets, leases)
            }
        }
    }

    /// The flat grid: one aligned positioned read per record, the payload
    /// `place`d to offset 0 when the file offset was not aligned, the
    /// lease covering exactly `record_bytes`.
    fn fill_flat(
        &self,
        file: &File,
        slots: &[RingSlot],
        offsets: &[u64],
        leases: &mut Vec<SlotLease>,
    ) -> Result<(), EngError> {
        let batch: Vec<(RingSlot, u64, u64, u64)> = slots
            .iter()
            .zip(offsets.iter())
            .map(|(&slot, &offset)| {
                let (start, total) = aligned_span(offset, self.record_bytes);
                (slot, offset, start, total)
            })
            .collect();
        let mut sent = 0usize;
        let dispatch = (|| -> Result<(), EngError> {
            for &(slot, _offset, start, total) in &batch {
                let span = self.loan_for_pool(slot)?;
                // One handle per read, cloned from the store's: pread never
                // moves the shared file's offset. A failure between the
                // loan and the send returns the loan — the job never
                // reached a worker, so nothing else ever will.
                let file = match file.try_clone() {
                    Ok(file) => file,
                    Err(e) => {
                        self.return_loan(slot, span);
                        return Err(EngError::Backend {
                            operation: "expert record read",
                            message: format!("the handle cannot serve a pooled read: {e}"),
                        });
                    }
                };
                self.send_job(Job::Flat {
                    file,
                    slot,
                    span,
                    start,
                    total: total as usize,
                })?;
                sent += 1;
            }
            Ok(())
        })();
        if let Err(error) = dispatch {
            // A failed dispatch drains what it started: every dispatched
            // job must land and return its span before the error leaves,
            // or the loaned slots sit abandoned and the next batch
            // double-loans them. A drain failure here cannot outrank the
            // dispatch error above (a dead pool is what that error
            // reports), so the drain's own error is dropped after it
            // returns every span it did land.
            let _ = self.reap(sent);
            return Err(error);
        }
        let mut landed = self.reap(batch.len())?;
        for &(slot, offset, start, _total) in &batch {
            let got = landed
                .remove(&slot)
                .expect("every dispatched read landed")?;
            let want_start = (offset - start) as usize;
            if got < want_start + self.record_bytes as usize {
                return Err(EngError::Checkpoint(format!(
                    "the record file is shorter than the index claims at byte {offset}"
                )));
            }
            let (ptr, len) = {
                let mut ring = self.ring.lock().expect("ring lock");
                if want_start != 0
                    && let Err(e) = ring.place(slot, want_start as u64, 0, self.record_bytes)
                {
                    return Err(EngError::Backend {
                        operation: "expert record read",
                        message: e,
                    });
                }
                let window = ring
                    .begin_copy(slot, Arc::new(HostBytesFinal))
                    .and_then(|()| ring.slot_window(slot, 0, self.record_bytes))
                    .map_err(|e| EngError::Backend {
                        operation: "expert record read",
                        message: e,
                    })?;
                (window.as_ptr(), window.len())
            };
            leases.push(SlotLease {
                ring: Arc::clone(&self.ring),
                slot,
                ptr,
                len,
            });
        }
        Ok(())
    }

    /// The span grid: one job per record carrying its site's extent
    /// list; the worker assembles the record (payload already at its
    /// record offsets), so there is nothing to place and the lease
    /// covers exactly the site's raw bytes.
    fn fill_spans(
        &self,
        sites: &[Vec<SpanRead>],
        max_window: usize,
        slots: &[RingSlot],
        offsets: &[u64],
        leases: &mut Vec<SlotLease>,
    ) -> Result<(), EngError> {
        let batch: Vec<(RingSlot, usize)> = slots
            .iter()
            .zip(offsets.iter())
            .map(|(&slot, &offset)| {
                let index = usize::try_from(offset / self.record_bytes)
                    .expect("read_records built the offset from a site index");
                (slot, index)
            })
            .collect();
        let mut sent = 0usize;
        let dispatch = (|| -> Result<(), EngError> {
            for &(slot, index) in &batch {
                let span = self.loan_for_pool(slot)?;
                let reads = sites[index].clone();
                self.send_job(Job::Spans {
                    slot,
                    span,
                    reads,
                    max_window,
                    mode: self.shard_open,
                })?;
                sent += 1;
            }
            Ok(())
        })();
        if let Err(error) = dispatch {
            // A failed dispatch drains what it started, exactly as the
            // flat grid does: every dispatched job must land and return
            // its span before the error leaves, or the loaned slots sit
            // abandoned and the next batch double-loans them. The drain's
            // own error cannot outrank the dispatch error above.
            let _ = self.reap(sent);
            return Err(error);
        }
        let mut landed = self.reap(batch.len())?;
        for &(slot, index) in &batch {
            landed
                .remove(&slot)
                .expect("every dispatched read landed")?;
            let raw: u64 = sites[index].iter().map(|read| read.len).sum();
            let (ptr, len) = {
                let ring = self.ring.lock().expect("ring lock");
                let window = ring
                    .begin_copy(slot, Arc::new(HostBytesFinal))
                    .and_then(|()| ring.slot_window(slot, 0, raw))
                    .map_err(|e| EngError::Backend {
                        operation: "expert record read",
                        message: e,
                    })?;
                (window.as_ptr(), window.len())
            };
            leases.push(SlotLease {
                ring: Arc::clone(&self.ring),
                slot,
                ptr,
                len,
            });
        }
        Ok(())
    }

    /// Loans one slot's span out for a pooled read.
    fn loan_for_pool(&self, slot: RingSlot) -> Result<&'static mut [u8], EngError> {
        let mut ring = self.ring.lock().expect("ring lock");
        ring.loan_span(slot).map_err(|e| EngError::Backend {
            operation: "expert record read",
            message: e,
        })
    }

    fn send_job(&self, job: Job) -> Result<(), EngError> {
        self.pool.send(job).map_err(|failed| {
            // The loan never reached a worker; return it before the error
            // leaves, or the slot sits abandoned for the next batch to
            // double-loan.
            let error = EngError::Backend {
                operation: "expert record read",
                message: "the read pool is gone".into(),
            };
            self.return_loan(failed.0.slot(), failed.0.into_span());
            error
        })
    }

    /// Returns a loan that never reached the pool (the send or the
    /// handle clone failed after the loan).
    fn return_loan(&self, slot: RingSlot, span: &'static mut [u8]) {
        self.ring
            .lock()
            .expect("ring lock")
            .return_span(slot, span)
            .expect("an unsent loan returns")
    }

    /// Reaps `count` outcomes: spans back under the lock as they land,
    /// reads matched by slot id (workers finish out of order), a
    /// worker-composed message mapped to its class ("short: ..." is a
    /// pack defect, anything else a backend failure). Every dispatched
    /// read is reaped whatever its outcome — a failed record still leaves
    /// its slot's memory available for the caller's cleanup.
    fn reap(
        &self,
        count: usize,
    ) -> Result<std::collections::HashMap<RingSlot, Result<usize, EngError>>, EngError> {
        let mut landed = std::collections::HashMap::with_capacity(count);
        for _ in 0..count {
            let outcome = self.pool.recv().map_err(|_| EngError::Backend {
                operation: "expert record read",
                message: "a pooled read never answered".into(),
            })?;
            let got = outcome.got.map_err(|message| {
                if let Some(rest) = message.strip_prefix("short: ") {
                    EngError::Checkpoint(rest.to_string())
                } else {
                    EngError::Backend {
                        operation: "expert record read",
                        message: message.strip_prefix("io: ").unwrap_or(&message).to_string(),
                    }
                }
            });
            let slot = outcome.slot;
            {
                let mut ring = self.ring.lock().expect("ring lock");
                ring.return_span(slot, outcome.span)
                    .map_err(|e| EngError::Backend {
                        operation: "expert record read",
                        message: e,
                    })?;
            }
            landed.insert(slot, got);
        }
        Ok(landed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::expert_store::buffered::tests_support::ErrOf;
    use std::io::Write;
    use tempfile::TempDir;

    fn pinned(len: usize) -> &'static mut [u8] {
        // A Vec's pointer is only 16-byte aligned; production ring memory
        // (cudaHostAlloc/mmap) is page-aligned. Carve a sector-aligned
        // sub-slice out of an oversized leaked buffer instead — no unsafe
        // in tests.
        let slop = SECTOR as usize - 1;
        let box_ = Box::leak(vec![0u8; len + slop].into_boxed_slice());
        let base = box_.as_ptr() as usize;
        let off = (SECTOR as usize - (base % SECTOR as usize)) % SECTOR as usize;
        &mut box_[off..off + len]
    }

    const LAYERS: usize = 4;
    const EXPERTS: usize = 8;
    const RECORD: u64 = 64;

    /// The shard-open mode this test's temp filesystem answers: O_DIRECT
    /// where it is granted, buffered where it refuses (tmpfs, overlayfs).
    /// No default-run test may depend on the host's temp filesystem, so
    /// each span test builds its store in the mode granted here.
    fn granted_shard_open(dir: &TempDir) -> ShardOpen {
        let probe = dir.path().join("o-direct-probe");
        std::fs::write(&probe, vec![0u8; SECTOR as usize]).expect("write the probe");
        let granted = crate::family::expert_bank::host_tier::open_direct(&probe).is_ok();
        std::fs::remove_file(&probe).expect("remove the probe");
        if granted {
            ShardOpen::Direct
        } else {
            ShardOpen::Buffered
        }
    }

    fn grid_file(dir: &TempDir, truncate: usize) -> (Vec<u8>, File) {
        let total = (LAYERS * EXPERTS) as u64 * RECORD;
        let bytes: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
        let path = dir.path().join("grid.bin");
        let mut file = File::create(&path).expect("create");
        file.write_all(&bytes[..bytes.len() - truncate])
            .expect("write");
        drop(file);
        (bytes, File::open(&path).expect("open"))
    }

    fn store(file: File, slots: usize) -> RingRecords {
        let slot_bytes = (align_up_sector(RECORD) + SECTOR) as usize * slots;
        RingRecords::new(
            file,
            LAYERS,
            EXPERTS,
            RECORD,
            slots,
            pinned(slot_bytes),
            Duration::from_millis(250),
        )
        .expect("the store builds")
    }

    fn site(layer: usize, expert: u16) -> RecordSite {
        RecordSite { layer, expert }
    }

    /// The span grid: extents across two shards at non-sector-aligned
    /// file offsets, assembled to their record offsets by the pool's
    /// workers; a shard that ends inside a record; and a site whose
    /// extents leave a gap, refused at construction.
    #[test]
    fn span_addressed_reads_assemble_records() {
        let dir = tempfile::tempdir().expect("dir");
        let shard = 8192usize;
        let a: Vec<u8> = (0..shard).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..shard).map(|i| ((i + 7) % 251) as u8).collect();
        std::fs::write(dir.path().join("a.bin"), &a).expect("write a");
        std::fs::write(dir.path().join("b.bin"), &b).expect("write b");
        let extent = |shard: &str, file_offset: u64, len: u64, record_offset: u64| SpanRead {
            shard: dir.path().join(shard),
            file_offset,
            len,
            record_offset,
        };
        // Every site reads the same three extents, shifted by its index
        // so the bytes differ: 20 from a at +37, 30 from b at +111, 10
        // from a at +5000, tiling a 60-byte record.
        let sites: Vec<Vec<SpanRead>> = (0..4)
            .map(|i| {
                let salt = (i as u64) * 1000;
                vec![
                    extent("a.bin", 37 + salt, 20, 0),
                    extent("b.bin", 111 + salt, 30, 20),
                    extent("a.bin", 5000 + salt, 10, 50),
                ]
            })
            .collect();
        let expected = |site_index: usize| -> Vec<u8> {
            let salt = site_index * 1000;
            let mut out = vec![0u8; 60];
            out[..20].copy_from_slice(&a[37 + salt..57 + salt]);
            out[20..50].copy_from_slice(&b[111 + salt..141 + salt]);
            out[50..60].copy_from_slice(&a[5000 + salt..5010 + salt]);
            out
        };
        let build = |sites: Vec<Vec<SpanRead>>| {
            let slot_bytes = (align_up_sector(60) + SECTOR) as usize * 2;
            RingRecords::new_span_addressed(
                sites,
                2,
                2,
                60,
                2,
                granted_shard_open(&dir),
                pinned(slot_bytes),
                Duration::from_millis(250),
            )
        };
        let store = build(sites.clone()).expect("the span store builds");
        let leases = store
            .read_records(&[site(0, 0), site(1, 1)])
            .expect("the batch reads");
        assert_eq!(leases.len(), 2);
        assert_eq!(&leases[0][..], expected(0).as_slice());
        assert_eq!(&leases[1][..], expected(3).as_slice());
        drop(leases);

        // A shard that ends inside a record is a pack defect: Checkpoint.
        let short = {
            let mut bytes = b.clone();
            bytes.truncate(120);
            bytes
        };
        std::fs::write(dir.path().join("b.bin"), &short).expect("truncate b");
        let store = build(sites).expect("the span store builds");
        let error = match store.read_records(&[site(0, 0)]) {
            Ok(_) => panic!("the read is short"),
            Err(error) => error,
        };
        assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");

        // A site whose extents leave a gap never becomes a store.
        let gapped = vec![vec![
            extent("a.bin", 37, 20, 0),
            extent("a.bin", 63, 10, 30),
        ]];
        let error = match build(gapped) {
            Ok(_) => panic!("the gap refuses the build"),
            Err(error) => error,
        };
        assert!(matches!(error, EngError::Checkpoint(_)), "{error:?}");
    }

    /// Span grid: a failure in the dispatch phase
    /// (not the read phase) drains what it started. Slot 0 — the second
    /// slot the free stack hands a two-site batch — is left free with
    /// its span on loan, so the second dispatch fails after one job was
    /// sent; the in-flight job must land and return its span before the
    /// error leaves, and a clean batch must then read both sites.
    #[test]
    fn a_send_failure_after_the_loan_returns_the_span() {
        // A send that fails AFTER the loan must return
        // the span, or the slot sits abandoned and the next batch
        // double-loans it. A pool with no workers is the injection: the
        // job channel's receiver is gone, so the FIRST dispatch loans and
        // then fails the send.
        let dir = tempfile::tempdir().expect("dir");
        let shard = 8192usize;
        let a: Vec<u8> = (0..shard).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..shard).map(|i| ((i + 7) % 251) as u8).collect();
        std::fs::write(dir.path().join("a.bin"), &a).expect("write a");
        std::fs::write(dir.path().join("b.bin"), &b).expect("write b");
        let extent = |shard: &str, file_offset: u64, len: u64, record_offset: u64| SpanRead {
            shard: dir.path().join(shard),
            file_offset,
            len,
            record_offset,
        };
        let sites: Vec<Vec<SpanRead>> = (0..2)
            .map(|i| {
                let salt = (i as u64) * 1000;
                vec![
                    extent("a.bin", 37 + salt, 20, 0),
                    extent("b.bin", 111 + salt, 30, 20),
                    extent("a.bin", 5000 + salt, 10, 50),
                ]
            })
            .collect();
        let slot_bytes = (align_up_sector(60) + SECTOR) as usize * 2;
        let mut store = RingRecords::new_span_addressed(
            sites,
            1,
            2,
            60,
            2,
            granted_shard_open(&dir),
            pinned(slot_bytes),
            Duration::from_millis(250),
        )
        .expect("the span store builds");
        store.pool = ReadPool::spawn(0);
        let err = store
            .read_records(&[site(0, 0), site(0, 1)])
            .err_of("the send fails after the loan");
        assert!(
            matches!(&err, EngError::Backend { message, .. } if message.contains("the read pool is gone")),
            "the failure is the dead pool: {err:?}"
        );
        let expected = |site_index: usize| -> Vec<u8> {
            let salt = site_index * 1000;
            let a = std::fs::read(dir.path().join("a.bin")).expect("shard a");
            let b = std::fs::read(dir.path().join("b.bin")).expect("shard b");
            let mut out = vec![0u8; 60];
            out[..20].copy_from_slice(&a[37 + salt..57 + salt]);
            out[20..50].copy_from_slice(&b[111 + salt..141 + salt]);
            out[50..60].copy_from_slice(&a[5000 + salt..5010 + salt]);
            out
        };
        // A live pool again: the returned loan serves — an abandoned
        // span would double-loan on this batch.
        store.pool = ReadPool::spawn(2);
        let clean = store
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("the returned span serves the next batch");
        assert_eq!(&clean[0][..], expected(0).as_slice());
        assert_eq!(&clean[1][..], expected(1).as_slice());
    }

    #[test]
    fn a_flat_send_failure_after_the_loan_returns_the_span() {
        // The flat grid's loan-after-send-failure path, same injection:
        // the first dispatch loans, clones the handle, and the send
        // fails; the span must come back or the next batch double-loans.
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 5);
        let mut records = store(file, 5);
        records.pool = ReadPool::spawn(0);
        let err = records
            .read_records(&[site(0, 0), site(1, 0)])
            .err_of("the send fails after the loan");
        assert!(
            matches!(&err, EngError::Backend { message, .. } if message.contains("the read pool is gone")),
            "the failure is the dead pool: {err:?}"
        );
        records.pool = ReadPool::spawn(5);
        let clean = records
            .read_records(&[site(0, 0), site(1, 0)])
            .expect("the returned span serves the next batch");
        assert_eq!(&*clean[0], expected(&grid, site(0, 0)));
        assert_eq!(&*clean[1], expected(&grid, site(1, 0)));
    }

    #[test]
    fn a_failed_span_dispatch_drains_what_it_started() {
        let dir = tempfile::tempdir().expect("dir");
        let shard = 8192usize;
        let a: Vec<u8> = (0..shard).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..shard).map(|i| ((i + 7) % 251) as u8).collect();
        std::fs::write(dir.path().join("a.bin"), &a).expect("write a");
        std::fs::write(dir.path().join("b.bin"), &b).expect("write b");
        let extent = |shard: &str, file_offset: u64, len: u64, record_offset: u64| SpanRead {
            shard: dir.path().join(shard),
            file_offset,
            len,
            record_offset,
        };
        let sites: Vec<Vec<SpanRead>> = (0..2)
            .map(|i| {
                let salt = (i as u64) * 1000;
                vec![
                    extent("a.bin", 37 + salt, 20, 0),
                    extent("b.bin", 111 + salt, 30, 20),
                    extent("a.bin", 5000 + salt, 10, 50),
                ]
            })
            .collect();
        let expected = |site_index: usize| -> Vec<u8> {
            let salt = site_index * 1000;
            let mut out = vec![0u8; 60];
            out[..20].copy_from_slice(&a[37 + salt..57 + salt]);
            out[20..50].copy_from_slice(&b[111 + salt..141 + salt]);
            out[50..60].copy_from_slice(&a[5000 + salt..5010 + salt]);
            out
        };
        let slot_bytes = (align_up_sector(60) + SECTOR) as usize * 2;
        let store = RingRecords::new_span_addressed(
            sites,
            1,
            2,
            60,
            2,
            granted_shard_open(&dir),
            pinned(slot_bytes),
            Duration::from_millis(250),
        )
        .expect("the span store builds");
        // The free stack hands the batch its slots in a fixed but
        // unspecified order, so the injection targets the SECOND-acquired
        // slot: whatever it is, the batch's second dispatch finds its span
        // already on loan while the first job is in flight.
        let (held_slot, held) = {
            let mut ring = store.ring.lock().expect("ring lock");
            let first = ring.acquire().expect("a first slot");
            let second = ring.acquire().expect("a second slot");
            let span = ring
                .loan_span(second)
                .expect("the second slot's span loans");
            ring.abandon(second).expect("the second slot back to free");
            ring.abandon(first).expect("the first slot back to free");
            (second, span)
        };
        let err = store
            .read_records(&[site(0, 0), site(0, 1)])
            .err_of("the second dispatch fails");
        assert!(
            matches!(&err, EngError::Backend { message, .. } if message.contains("already on loan")),
            "the failure is the injected double loan: {err:?}"
        );
        {
            let mut ring = store.ring.lock().expect("ring lock");
            ring.return_span(held_slot, held)
                .expect("the held span returns");
        }
        let clean = store
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("the drained slot serves the next batch");
        assert_eq!(&clean[0][..], expected(0).as_slice());
        assert_eq!(&clean[1][..], expected(1).as_slice());
    }

    fn expected(grid: &[u8], s: RecordSite) -> &[u8] {
        let index = (s.layer * EXPERTS + usize::from(s.expert)) as u64;
        let base = (index * RECORD) as usize;
        &grid[base..base + RECORD as usize]
    }

    #[test]
    fn a_buffered_span_store_reads_the_same_bytes() {
        // The start-decided buffered mode (a filesystem that refuses
        // O_DIRECT): the same span records, byte-identical, through plain
        // buffered opens — the mode changes the open, never the data.
        let dir = tempfile::tempdir().expect("dir");
        let shard = 8192usize;
        let a: Vec<u8> = (0..shard).map(|i| (i % 251) as u8).collect();
        let b: Vec<u8> = (0..shard).map(|i| ((i + 7) % 251) as u8).collect();
        std::fs::write(dir.path().join("a.bin"), &a).expect("write a");
        std::fs::write(dir.path().join("b.bin"), &b).expect("write b");
        let extent = |shard: &str, file_offset: u64, len: u64, record_offset: u64| SpanRead {
            shard: dir.path().join(shard),
            file_offset,
            len,
            record_offset,
        };
        let sites: Vec<Vec<SpanRead>> = (0..2)
            .map(|i| {
                let salt = (i as u64) * 1000;
                vec![
                    extent("a.bin", 37 + salt, 20, 0),
                    extent("b.bin", 111 + salt, 30, 20),
                    extent("a.bin", 5000 + salt, 10, 50),
                ]
            })
            .collect();
        let expected = |site_index: usize| -> Vec<u8> {
            let salt = site_index * 1000;
            let mut out = vec![0u8; 60];
            out[..20].copy_from_slice(&a[37 + salt..57 + salt]);
            out[20..50].copy_from_slice(&b[111 + salt..141 + salt]);
            out[50..60].copy_from_slice(&a[5000 + salt..5010 + salt]);
            out
        };
        let slot_bytes = (align_up_sector(60) + SECTOR) as usize * 2;
        let store = RingRecords::new_span_addressed(
            sites,
            1,
            2,
            60,
            2,
            ShardOpen::Buffered,
            pinned(slot_bytes),
            Duration::from_millis(250),
        )
        .expect("the buffered span store builds");
        let got = store
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("the buffered batch reads");
        assert_eq!(&got[0][..], expected(0).as_slice());
        assert_eq!(&got[1][..], expected(1).as_slice());
    }

    #[test]
    fn a_batch_returns_each_sites_bytes_in_order() {
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 0);
        let records = store(file, 4);
        let sites = [site(0, 1), site(2, 0), site(3, 7)];
        let got = records.read_records(&sites).expect("the batch reads");
        for (lease, &s) in got.iter().zip(sites.iter()) {
            assert_eq!(&**lease, expected(&grid, s), "site {s:?}");
        }
    }

    #[test]
    fn duplicate_sites_each_read_their_own_slot() {
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 0);
        let records = store(file, 2);
        let sites = [site(2, 2), site(2, 2)];
        let got = records.read_records(&sites).expect("the batch reads");
        assert_eq!(got.len(), 2);
        assert_eq!(&*got[0], expected(&grid, sites[0]));
        assert_eq!(&*got[1], expected(&grid, sites[1]));
    }

    #[test]
    fn leases_return_their_slots_on_drop() {
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 0);
        let records = store(file, 2);
        {
            let got = records
                .read_records(&[site(0, 0), site(0, 1)])
                .expect("both slots check out");
            assert_eq!(got.len(), 2);
        }
        // The slots are back: a full-size batch works again immediately.
        let got = records
            .read_records(&[site(1, 0), site(1, 1)])
            .expect("slots returned after drop");
        assert_eq!(&*got[0], expected(&grid, site(1, 0)));
        assert_eq!(&*got[1], expected(&grid, site(1, 1)));
    }

    #[test]
    fn overlapping_batches_never_lose_a_slot() {
        // Two interleaved half batches: the second finds the remaining
        // slots while the first batch's leases are alive, and every slot
        // must be back afterwards.
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 0);
        let records = std::sync::Arc::new(store(file, 4));
        let first = records
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("first half batch");
        let second = records
            .read_records(&[site(1, 0), site(1, 1)])
            .expect("second half batch, disjoint slots");
        assert_eq!(&*first[0], expected(&grid, site(0, 0)));
        assert_eq!(&*second[1], expected(&grid, site(1, 1)));
        drop(first);
        drop(second);
        let after = records
            .read_records(&[site(2, 0), site(2, 1), site(2, 2), site(2, 3)])
            .expect("all four slots usable again");
        assert_eq!(after.len(), 4);
    }

    #[test]
    fn a_batch_larger_than_the_pool_is_refused_before_any_read() {
        let dir = TempDir::new().expect("tempdir");
        let (_grid, file) = grid_file(&dir, 0);
        let records = store(file, 2);
        let err = records
            .read_records(&[site(0, 0), site(0, 1), site(0, 2)])
            .err_of("three records, two slots");
        assert!(matches!(err, EngError::BadRequest(_)), "{err:?}");
        // The refusal left every slot free.
        records
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("the pool is untouched");
    }

    #[test]
    fn a_batch_that_cannot_gather_slots_fails_within_the_wait_bound() {
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 0);
        let slot_bytes = (align_up_sector(RECORD) + SECTOR) as usize * 2;
        let records = RingRecords::new(
            file,
            LAYERS,
            EXPERTS,
            RECORD,
            2,
            pinned(slot_bytes),
            Duration::from_millis(40),
        )
        .expect("the store builds");
        let held = records
            .read_records(&[site(0, 0), site(0, 1)])
            .expect("both slots held");
        let started = Instant::now();
        let err = records
            .read_records(&[site(1, 0), site(1, 1)])
            .err_of("the same caller holds every slot");
        assert!(matches!(err, EngError::Backend { .. }), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the wait is bounded, not forever"
        );
        drop(held);
        let after = records
            .read_records(&[site(1, 0), site(1, 1)])
            .expect("readable again after the leases drop");
        assert_eq!(&*after[0], expected(&grid, site(1, 0)));
    }

    #[test]
    fn a_short_file_is_a_checkpoint_error_and_the_slots_recover() {
        let dir = TempDir::new().expect("tempdir");
        let (_grid, file) = grid_file(&dir, 5);
        let records = store(file, 3);
        let err = records
            .read_records(&[site(3, 7)])
            .err_of("the file ends inside the record");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
        // The failed read's slot went back: a full-size batch still works
        // on the same store.
        let dir2 = TempDir::new().expect("tempdir");
        let (grid2, file2) = grid_file(&dir2, 0);
        let _ = grid2;
        let _ = file2;
        let full = records
            .read_records(&[site(0, 0), site(0, 1), site(0, 2)])
            .expect("the truncated store's pool recovered");
        assert_eq!(full.len(), 3);
    }

    #[test]
    fn a_failure_at_a_later_site_returns_the_batch_without_hanging() {
        // The earlier site's lease is already built when the later one
        // fails; before the fix, that error path dropped the lease while
        // holding the ring mutex the lease's own drop takes, and the
        // caller hung forever. Completing is the assertion.
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 5);
        let records = store(file, 4);
        let started = Instant::now();
        let err = records
            .read_records(&[site(0, 0), site(3, 7)])
            .err_of("the second record crosses the file end");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the failed batch returns, it does not hang"
        );
        let _ = grid;
    }

    #[test]
    fn a_failed_dispatch_drains_what_it_started() {
        // A failure in the dispatch phase (not the read
        // phase) must not abandon in-flight jobs. The third-acquired slot
        // is left free with its span on loan, so the third loan of a
        // five-slot batch fails after two jobs were sent; those two must
        // land and return their spans before the error leaves, or the
        // next batch double-loans. The free stack's exact order is an
        // implementation detail; the stack is restored by abandoning in
        // reverse, so the loaned slot stays the batch's third dispatch.
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 5);
        let records = store(file, 5);
        let (held_slot, held) = {
            let mut ring = records.ring.lock().expect("ring lock");
            let first = ring.acquire().expect("a first slot");
            let second = ring.acquire().expect("a second slot");
            let third = ring.acquire().expect("a third slot");
            let span = ring.loan_span(third).expect("the third slot's span loans");
            ring.abandon(third).expect("the third slot back to free");
            ring.abandon(second).expect("the second slot back to free");
            ring.abandon(first).expect("the first slot back to free");
            (third, span)
        };
        let err = records
            .read_records(&[site(0, 0), site(1, 0), site(2, 0), site(3, 0), site(0, 1)])
            .err_of("the third dispatch fails");
        assert!(
            matches!(&err, EngError::Backend { message, .. } if message.contains("already on loan")),
            "the failure is the injected double loan: {err:?}"
        );
        {
            let mut ring = records.ring.lock().expect("ring lock");
            ring.return_span(held_slot, held)
                .expect("the held span returns");
        }
        // The clean batch over the same sites proves the two dispatched
        // jobs were drained: their slots serve again without a double
        // loan.
        let full = records
            .read_records(&[site(0, 0), site(1, 0), site(2, 0), site(3, 0), site(0, 1)])
            .expect("the drained slots serve the next batch");
        assert_eq!(full.len(), 5);
        assert_eq!(&*full[0], expected(&grid, site(0, 0)));
        assert_eq!(&*full[4], expected(&grid, site(0, 1)));
    }

    #[test]
    fn a_failed_batch_returns_every_slot_to_the_pool() {
        // The failed batch held a leased slot and a failed one; all four
        // must be back for a full-size batch to read at once, and failing
        // again must not shrink the pool.
        let dir = TempDir::new().expect("tempdir");
        let (grid, file) = grid_file(&dir, 5);
        let records = store(file, 4);
        for _ in 0..2 {
            let err = records
                .read_records(&[site(0, 0), site(3, 7)])
                .err_of("the later site crosses the file end");
            assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
            let full = records
                .read_records(&[site(0, 0), site(1, 0), site(2, 0), site(3, 0)])
                .expect("every slot the failed batch held came back");
            assert_eq!(full.len(), 4);
            assert_eq!(&*full[0], expected(&grid, site(0, 0)));
            assert_eq!(&*full[3], expected(&grid, site(3, 0)));
        }
    }

    #[test]
    fn a_site_outside_the_store_is_a_bad_request() {
        let dir = TempDir::new().expect("tempdir");
        let (_grid, file) = grid_file(&dir, 0);
        let records = store(file, 2);
        for bad in [site(4, 0), site(0, 8)] {
            let err = records.read_records(&[bad]).err_of("rejected");
            assert!(matches!(err, EngError::BadRequest(_)), "{bad:?}: {err:?}");
        }
    }

    #[test]
    fn a_zero_record_size_or_empty_pool_is_refused_at_construction() {
        let dir = TempDir::new().expect("tempdir");
        let (_grid, file) = grid_file(&dir, 0);
        assert!(matches!(
            RingRecords::new(
                file,
                LAYERS,
                EXPERTS,
                0,
                4,
                pinned(4096),
                Duration::from_secs(1)
            )
            .err_of("zero record size"),
            EngError::Checkpoint(_)
        ));
        let dir2 = TempDir::new().expect("tempdir");
        let (_g2, file2) = grid_file(&dir2, 0);
        assert!(matches!(
            RingRecords::new(
                file2,
                LAYERS,
                EXPERTS,
                RECORD,
                0,
                pinned(0),
                Duration::from_secs(1)
            )
            .err_of("empty pool"),
            EngError::Checkpoint(_)
        ));
    }

    #[test]
    fn the_store_is_sync() {
        // The lease moves between threads (Send) but needs no Sync: two
        // threads never share one lease, they hold separate leases.
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Send + Sync>() {}
        assert_send::<SlotLease>();
        assert_sync::<RingRecords>();
    }

    #[test]
    fn the_unsafe_scope_stays_pinned() {
        // Mirrors page_cache.rs: the module declares its unsafe scope in
        // the header, and this scan keeps it from growing unnoticed. The
        // needles are assembled at runtime so this test's own literals
        // never match themselves.
        let block = ["unsafe", " {"].concat();
        let send_impl = ["unsafe", " impl"].concat();
        let source = include_str!("ring.rs");
        assert_eq!(
            source.matches(&block).count(),
            1,
            "one Deref block, no more"
        );
        assert_eq!(
            source.matches(&send_impl).count(),
            1,
            "one Send impl, no more"
        );
    }
}
