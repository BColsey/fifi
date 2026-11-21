//! The portable buffered backend: positioned reads over a shared handle.
//!
//! `read_at_offset` is plain pread semantics: it fills a caller buffer at a
//! file offset without touching any cursor, so one shared handle serves
//! several threads without a mutex. Records form one flat grid: record
//! `(layer, expert)` sits at `((layer * experts_per_layer) + expert) *
//! record_bytes` — no header, no per-record table, no second file. A real
//! pack index replaces the arithmetic inside this module when one exists.

use std::fs::File;
use std::io;
use std::ops::{Deref, Range};
use std::sync::Arc;

use super::{
    ExpertRecordStore, RecordSite, index_arithmetic_overflow, record_size_not_addressable,
    site_outside_store,
};
use crate::EngError;

/// Positioned read from a shared handle: fills `dst` starting at `offset`
/// without moving any handle's cursor. Implementations follow pread
/// semantics: they may return short, and `Ok(0)` means end of file.
pub(crate) trait PositionedReader: Send + Sync {
    fn read_at_offset(&self, dst: &mut [u8], offset: u64) -> io::Result<usize>;
}

impl PositionedReader for File {
    fn read_at_offset(&self, dst: &mut [u8], offset: u64) -> io::Result<usize> {
        // Spelled fully: a bare read_at here would resolve to this trait
        // and recurse.
        std::os::unix::fs::FileExt::read_at(self, dst, offset)
    }
}

/// Reads at most `dst.len()` bytes positioned at `offset`, retrying
/// `Interrupted` (a signal must not fail a read that was making progress).
/// Returns how many bytes were read; fewer than asked means end of file.
/// This is the one copy of that loop in the tree: the expert-bank ring
/// reads through it too.
pub(crate) fn pread_at_most<R: PositionedReader + ?Sized>(
    reader: &R,
    dst: &mut [u8],
    offset: u64,
) -> io::Result<usize> {
    let mut done = 0;
    while done < dst.len() {
        let at = offset
            .checked_add(done as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset overflow"))?;
        match reader.read_at_offset(&mut dst[done..], at) {
            Ok(n) => {
                if n == 0 {
                    break;
                }
                done += n;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// The largest merged span one positioned read fills. A longer run of
/// adjacent records splits at this cap, bounding the peak allocation of a
/// single call no matter how large the batch is.
pub(crate) const MERGE_CAP_BYTES: u64 = 8 * 1024 * 1024;

/// A lease over one record's bytes: either the record's own buffer (a
/// single-record run) or a slice of the shared buffer a coalesced run was
/// read into. Valid until dropped; nothing expires on the next call.
pub(crate) struct RunLease {
    run: Arc<Box<[u8]>>,
    range: Range<usize>,
}

impl Deref for RunLease {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.run[self.range.clone()]
    }
}

/// Expert records as one flat grid behind positioned reads.
pub(crate) struct BufferedRecords<R: PositionedReader> {
    handle: R,
    layers: usize,
    experts_per_layer: usize,
    record_bytes: u64,
}

impl<R: PositionedReader> BufferedRecords<R> {
    /// `layers` x `experts_per_layer` records of exactly `record_bytes`
    /// each, readable through one shared positioned handle. A zero record
    /// size, or a grid whose byte extent does not fit its integer types, is
    /// an index defect and refused here.
    pub(crate) fn new(
        handle: R,
        layers: usize,
        experts_per_layer: usize,
        record_bytes: u64,
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
        Ok(Self {
            handle,
            layers,
            experts_per_layer,
            record_bytes,
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

    fn byte_offset(&self, index: u64) -> Result<u64, EngError> {
        index
            .checked_mul(self.record_bytes)
            .ok_or_else(|| index_arithmetic_overflow("record byte offset"))
    }

    /// Fills `dst` completely from `offset`. An I/O failure is
    /// `Backend`; a file that ends before the index's extent is `Checkpoint`.
    fn read_exact(&self, dst: &mut [u8], offset: u64) -> Result<(), EngError> {
        let mut done = 0;
        while done < dst.len() {
            let at = offset
                .checked_add(done as u64)
                .ok_or_else(|| index_arithmetic_overflow("read offset"))?;
            let n = pread_at_most(&self.handle, &mut dst[done..], at).map_err(|error| {
                EngError::Backend {
                    operation: "expert record read",
                    message: format!("read at byte {at} failed: {error}"),
                }
            })?;
            if n == 0 {
                return Err(EngError::Checkpoint(format!(
                    "the record file is shorter than the index claims: {done} of {} bytes at byte {at}",
                    dst.len()
                )));
            }
            done += n;
        }
        Ok(())
    }
}

impl<R: PositionedReader> ExpertRecordStore for BufferedRecords<R> {
    type Lease = RunLease;

    fn read_records(&self, sites: &[RecordSite]) -> Result<Vec<RunLease>, EngError> {
        let size = usize::try_from(self.record_bytes)
            .map_err(|_| record_size_not_addressable(self.record_bytes))?;
        let indexes: Vec<u64> = sites
            .iter()
            .map(|&s| self.linear_index(s))
            .collect::<Result<_, _>>()?;
        let mut out: Vec<RunLease> = Vec::with_capacity(sites.len());
        let mut run_start = 0;
        while run_start < sites.len() {
            // A maximal run of adjacent linear indexes, split at the merge
            // cap so no single read allocates without bound.
            let cap_records = (MERGE_CAP_BYTES / self.record_bytes).max(1);
            let mut run_end = run_start + 1;
            while run_end < sites.len() {
                let taken = (run_end - run_start) as u64;
                if taken >= cap_records {
                    break;
                }
                if indexes[run_end] != indexes[run_end - 1] + 1 {
                    break;
                }
                run_end += 1;
            }
            let first = self.byte_offset(indexes[run_start])?;
            if run_end - run_start == 1 {
                // Single record: read straight into its final buffer.
                let mut bytes = vec![0u8; size];
                self.read_exact(&mut bytes, first)?;
                out.push(RunLease {
                    run: Arc::new(bytes.into_boxed_slice()),
                    range: 0..size,
                });
            } else {
                // Coalesced run: one shared buffer, one read, one copy; each
                // lease slices the range it owns.
                let span_records = (run_end - run_start) as u64;
                let span = span_records
                    .checked_mul(self.record_bytes)
                    .and_then(|s| usize::try_from(s).ok())
                    .ok_or_else(|| index_arithmetic_overflow("run span"))?;
                let mut bytes = vec![0u8; span];
                self.read_exact(&mut bytes, first)?;
                let run = Arc::new(bytes.into_boxed_slice());
                for chunk in (0..span).step_by(size) {
                    out.push(RunLease {
                        run: Arc::clone(&run),
                        range: chunk..chunk + size,
                    });
                }
            }
            run_start = run_end;
        }
        Ok(out)
    }

    fn record_bytes(&self) -> u64 {
        self.record_bytes
    }

    fn max_batch(&self) -> usize {
        usize::MAX
    }
}

/// Shared fixtures for the expert-store backends: an in-memory grid behind
/// the positioned-read seam that counts read calls, so the coalescing and
/// pool contracts stay observable through the count.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::PositionedReader;
    /// `expect_err` for types without `Debug`: panics with `msg` on `Ok`.
    pub(crate) trait ErrOf<E> {
        fn err_of(self, msg: &str) -> E;
    }

    impl<T, E> ErrOf<E> for Result<T, E> {
        fn err_of(self, msg: &str) -> E {
            match self {
                Ok(_) => panic!("{msg}"),
                Err(e) => e,
            }
        }
    }

    use std::io;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub(crate) struct CountingGrid {
        pub(crate) bytes: Vec<u8>,
        pub(crate) reads: AtomicU64,
        /// When set, the next read first fails once with Interrupted.
        pub(crate) eintr_once: AtomicBool,
        /// When set, reads hand back only part of the buffer, exercising
        /// the fill loop's short-read progress.
        pub(crate) partial_give: AtomicBool,
    }

    pub(crate) fn counting_grid(
        layers: usize,
        experts: usize,
        record_bytes: usize,
    ) -> CountingGrid {
        let total = layers * experts * record_bytes;
        CountingGrid {
            bytes: (0..total as u16).map(|b| b as u8).collect(),
            reads: AtomicU64::new(0),
            eintr_once: AtomicBool::new(false),
            partial_give: AtomicBool::new(false),
        }
    }

    impl CountingGrid {
        pub(crate) fn reads(&self) -> u64 {
            self.reads.load(Ordering::SeqCst)
        }
    }

    impl PositionedReader for &CountingGrid {
        fn read_at_offset(&self, dst: &mut [u8], offset: u64) -> io::Result<usize> {
            if self.eintr_once.swap(false, Ordering::SeqCst) {
                return Err(io::Error::new(io::ErrorKind::Interrupted, "retry me"));
            }
            self.reads.fetch_add(1, Ordering::SeqCst);
            let start = offset as usize;
            let end = start
                .checked_add(dst.len())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "span overflow"))?;
            if end > self.bytes.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "past end"));
            }
            let give = if self.partial_give.load(Ordering::SeqCst) {
                (dst.len() / 2).max(1).min(end - start)
            } else {
                end - start
            };
            dst[..give].copy_from_slice(&self.bytes[start..start + give]);
            Ok(give)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::{CountingGrid, ErrOf, counting_grid};
    use super::*;
    use std::sync::atomic::Ordering;
    use tempfile::TempDir;

    fn store(grid: &CountingGrid) -> Result<BufferedRecords<&CountingGrid>, EngError> {
        BufferedRecords::new(grid, 4, 8, 16)
    }

    fn site(layer: usize, expert: u16) -> RecordSite {
        RecordSite { layer, expert }
    }

    #[test]
    fn every_layer_has_the_uniform_record_size() {
        let grid = counting_grid(4, 8, 16);
        let store = store(&grid).expect("the store builds");
        for layer in 0..4 {
            assert_eq!(store.record_bytes_at(layer).unwrap(), store.record_bytes());
        }
    }

    fn expected(grid: &CountingGrid, s: RecordSite) -> &[u8] {
        let index = (s.layer * 8 + usize::from(s.expert)) * 16;
        &grid.bytes[index..index + 16]
    }

    #[test]
    fn a_batch_returns_each_sites_bytes_in_order() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let sites = [site(0, 1), site(2, 0), site(3, 7)];
        let got = records.read_records(&sites).expect("the batch reads");
        assert_eq!(got.len(), sites.len());
        for (lease, &s) in got.iter().zip(sites.iter()) {
            assert_eq!(&**lease, expected(&grid, s), "site {s:?}");
        }
    }

    #[test]
    fn coalesced_runs_return_the_right_bytes() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let sites = [site(1, 2), site(1, 3), site(1, 4)];
        let got = records.read_records(&sites).expect("the batch reads");
        assert_eq!(got.len(), 3);
        for (lease, &s) in got.iter().zip(sites.iter()) {
            assert_eq!(&**lease, expected(&grid, s), "site {s:?}");
        }
        assert_eq!(grid.reads(), 1, "one coalesced positioned read");
    }

    #[test]
    fn a_gap_ends_a_run_and_the_next_run_reads_again() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let sites = [site(0, 0), site(0, 1), site(0, 3), site(0, 4)];
        let got = records.read_records(&sites).expect("the batch reads");
        for (lease, &s) in got.iter().zip(sites.iter()) {
            assert_eq!(&**lease, expected(&grid, s), "site {s:?}");
        }
        assert_eq!(grid.reads(), 2, "two runs, two positioned reads");
    }

    #[test]
    fn duplicate_sites_each_read_their_own_copy() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let sites = [site(2, 2), site(2, 2)];
        let got = records.read_records(&sites).expect("the batch reads");
        assert_eq!(got.len(), 2);
        assert_eq!(&*got[0], expected(&grid, sites[0]));
        assert_eq!(&*got[1], expected(&grid, sites[1]));
        assert_eq!(grid.reads(), 2, "a duplicate is not merged");
    }

    #[test]
    fn a_run_longer_than_the_merge_cap_splits() {
        // Virtual grid: bytes are derived from the offset, so a 4 MiB
        // record needs no fixture memory. The cap (8 MiB) is two records,
        // so three adjacent sites split into runs of two and one.
        struct VirtualGrid {
            total: usize,
            reads: std::sync::atomic::AtomicU64,
        }
        impl PositionedReader for &VirtualGrid {
            fn read_at_offset(&self, dst: &mut [u8], offset: u64) -> std::io::Result<usize> {
                use std::sync::atomic::Ordering;
                self.reads.fetch_add(1, Ordering::SeqCst);
                let start = offset as usize;
                let end = start + dst.len();
                assert!(end <= self.total, "read past the virtual grid");
                for (i, b) in dst.iter_mut().enumerate() {
                    *b = ((start + i) % 251) as u8;
                }
                Ok(dst.len())
            }
        }
        let record: u64 = 4 * 1024 * 1024;
        let grid = VirtualGrid {
            total: (record * 3) as usize,
            reads: std::sync::atomic::AtomicU64::new(0),
        };
        let records = BufferedRecords::new(&grid, 1, 3, record).expect("the store builds");
        let sites = [site(0, 0), site(0, 1), site(0, 2)];
        let got = records.read_records(&sites).expect("the batch reads");
        assert_eq!(got.len(), 3);
        use std::sync::atomic::Ordering;
        assert_eq!(
            grid.reads.load(Ordering::SeqCst),
            2,
            "the capped run splits: two records then one"
        );
        for (i, lease) in got.iter().enumerate() {
            assert_eq!(lease.len(), record as usize);
            let base = (record * i as u64) as usize;
            assert_eq!(lease[0], (base % 251) as u8);
            assert_eq!(
                lease[lease.len() - 1],
                ((base + record as usize - 1) % 251) as u8
            );
        }
    }

    #[test]
    fn a_site_outside_the_store_is_a_bad_request_before_any_read() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        for bad in [site(4, 0), site(0, 8), site(usize::MAX, 0)] {
            let err = records.read_records(&[bad]).err_of("rejected");
            assert!(matches!(err, EngError::BadRequest(_)), "{bad:?}: {err:?}");
        }
        assert_eq!(grid.reads(), 0, "no I/O for a rejected batch");
    }

    #[test]
    fn an_empty_batch_reads_nothing() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let got = records.read_records(&[]).expect("empty batch is empty");
        assert!(got.is_empty());
        assert_eq!(grid.reads(), 0);
    }

    #[test]
    fn an_interrupted_read_is_retried_not_failed() {
        let grid = counting_grid(4, 8, 16);
        grid.eintr_once
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let records = store(&grid).expect("the store builds");
        let got = records
            .read_records(&[site(0, 0)])
            .expect("the signal does not fail the read");
        assert_eq!(&*got[0], expected(&grid, site(0, 0)));
    }

    #[test]
    fn short_partial_reads_still_fill_the_whole_record() {
        let grid = counting_grid(4, 8, 16);
        grid.partial_give.store(true, Ordering::SeqCst);
        let records = store(&grid).expect("the store builds");
        let got = records
            .read_records(&[site(1, 3), site(2, 6)])
            .expect("partial reads loop to a full fill");
        assert_eq!(&*got[0], expected(&grid, site(1, 3)));
        assert_eq!(&*got[1], expected(&grid, site(2, 6)));
    }

    #[test]
    fn the_last_record_of_the_grid_reads_whole() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let got = records
            .read_records(&[site(3, 7)])
            .expect("the final record reads");
        assert_eq!(&*got[0], expected(&grid, site(3, 7)));
    }

    #[test]
    fn a_zero_record_size_is_refused_at_construction() {
        let grid = counting_grid(4, 8, 16);
        let err = BufferedRecords::new(&grid, 4, 8, 0).err_of("zero is an index defect");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
    }

    #[test]
    fn a_grid_whose_extent_overflows_is_refused_at_construction() {
        let grid = counting_grid(1, 1, 16);
        let err = BufferedRecords::new(&grid, usize::MAX, 2, 16).err_of("overflow refused");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
        let err = BufferedRecords::new(&grid, 1, 1, u64::MAX).err_of("huge record refused");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
    }

    #[test]
    fn a_real_file_backs_the_store_and_a_short_file_is_a_checkpoint_error() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("grid.bin");
        let mut bytes = vec![0u8; 4 * 8 * 16];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        std::fs::write(&path, &bytes[..bytes.len() - 5]).expect("write truncated grid");
        let file = File::open(&path).expect("open");
        let records = BufferedRecords::new(file, 4, 8, 16).expect("the store builds");
        // The final record crosses the truncation: end of file, not I/O.
        let err = records
            .read_records(&[site(3, 7)])
            .err_of("the file is short");
        assert!(matches!(err, EngError::Checkpoint(_)), "{err:?}");
        // An intact record still reads its exact bytes from the real file.
        let got = records
            .read_records(&[site(0, 0), site(1, 4)])
            .expect("intact records read");
        assert_eq!(&*got[0], &bytes[0..16]);
        let second = (8 + 4) * 16;
        assert_eq!(&*got[1], &bytes[second..second + 16]);
    }

    #[test]
    fn a_read_failure_is_a_backend_error() {
        struct Failing;
        impl PositionedReader for &Failing {
            fn read_at_offset(&self, _dst: &mut [u8], _offset: u64) -> io::Result<usize> {
                Err(io::Error::other("disk on fire"))
            }
        }
        let failing = Failing;
        let records = BufferedRecords::new(&failing, 4, 8, 16).expect("the store builds");
        let err = records.read_records(&[site(0, 0)]).err_of("I/O failed");
        match err {
            EngError::Backend { operation, .. } => assert_eq!(operation, "expert record read"),
            other => panic!("wrong class: {other:?}"),
        }
    }

    #[test]
    fn the_store_is_object_safe_and_usable_through_dyn() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        let dyn_store: &dyn ExpertRecordStore<Lease = RunLease> = &records;
        let got = dyn_store
            .read_records(&[site(2, 5)])
            .expect("dyn call reads");
        assert_eq!(&*got[0], expected(&grid, site(2, 5)));
        assert_eq!(dyn_store.record_bytes(), 16);
        assert_eq!(dyn_store.max_batch(), usize::MAX);
    }

    #[test]
    fn two_threads_read_disjoint_sites_concurrently() {
        let grid = counting_grid(4, 8, 16);
        let records = store(&grid).expect("the store builds");
        std::thread::scope(|s| {
            let a = &records;
            let ga = &grid;
            s.spawn(move || {
                let got = a.read_records(&[site(0, 0), site(1, 1)]).expect("a reads");
                assert_eq!(&*got[0], expected(ga, site(0, 0)));
                assert_eq!(&*got[1], expected(ga, site(1, 1)));
            });
            let b = &records;
            let gb = &grid;
            s.spawn(move || {
                let got = b.read_records(&[site(2, 2), site(3, 3)]).expect("b reads");
                assert_eq!(&*got[0], expected(gb, site(2, 2)));
                assert_eq!(&*got[1], expected(gb, site(3, 3)));
            });
        });
    }

    #[test]
    fn the_record_size_is_reported_for_tier_arithmetic() {
        let grid = counting_grid(4, 8, 16);
        assert_eq!(store(&grid).expect("builds").record_bytes(), 16);
    }

    #[test]
    fn the_store_is_sync_so_two_models_hold_two_stores() {
        fn assert_sync<T: Send + Sync>() {}
        assert_sync::<BufferedRecords<&CountingGrid>>();
    }
}
