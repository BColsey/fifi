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
use std::io::ErrorKind;
use std::num::{NonZeroU64, NonZeroUsize};
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::family::expert_bank::host_tier::{
    CopyFence, ODirectRing, SECTOR, Slot as RingSlot, align_up_sector,
};

use super::{
    ExpertRecordStore, RecordSite, index_arithmetic_overflow, record_size_not_addressable,
    site_outside_store,
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

/// An expert record store backed by the expert bank's O_DIRECT ring: one
/// pinned slot per record, staged and placed by the ring. The flat grid is
/// the same as the buffered backend's.
pub(crate) struct RingRecords {
    file: File,
    ring: Arc<Mutex<ODirectRing>>,
    layers: usize,
    experts_per_layer: usize,
    record_bytes: u64,
    slots: usize,
    max_wait: Duration,
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
            file,
            ring: Arc::new(Mutex::new(ring)),
            layers,
            experts_per_layer,
            record_bytes,
            slots: slots.get(),
            max_wait,
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
    /// Stages, places, and arms each record in order, pushing one lease
    /// per finished record. On error the batch's cleanup belongs to the
    /// caller: nothing here drops a lease or returns a slot, so a failure
    /// can never take the ring lock this function still holds.
    fn fill_batch(
        &self,
        slots: &[RingSlot],
        offsets: &[u64],
        leases: &mut Vec<SlotLease>,
    ) -> Result<(), EngError> {
        debug_assert_eq!(slots.len(), offsets.len());
        for (&slot, &offset) in slots.iter().zip(offsets.iter()) {
            // One handle per stage, cloned from the store's: the ring
            // reads through shared references and threads share the file.
            let file = self.file.try_clone().map_err(|e| EngError::Backend {
                operation: "expert record read",
                message: format!("the handle cannot serve a staged read: {e}"),
            })?;
            let inner = {
                let mut ring = self.ring.lock().expect("ring lock");
                match ring.stage_span(&file, offset, self.record_bytes, slot, 0) {
                    Ok(inner) => inner,
                    Err(e) => {
                        return Err(if e.kind() == ErrorKind::UnexpectedEof {
                            EngError::Checkpoint(format!(
                                "the record file is shorter than the index claims at byte {offset}"
                            ))
                        } else {
                            EngError::Backend {
                                operation: "expert record read",
                                message: format!("staged read at byte {offset} failed: {e}"),
                            }
                        });
                    }
                }
            };
            let (ptr, len) = {
                let mut ring = self.ring.lock().expect("ring lock");
                if inner != 0
                    && let Err(e) = ring.place(slot, inner, 0, self.record_bytes)
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

    fn expected(grid: &[u8], s: RecordSite) -> &[u8] {
        let index = (s.layer * EXPERTS + usize::from(s.expert)) as u64;
        let base = (index * RECORD) as usize;
        &grid[base..base + RECORD as usize]
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
