//! Expert record stores: reading one expert's bytes from a backing store
//! into host memory.
//!
//! I/O lives here: this module reads expert records from a backing store
//! into host memory and knows nothing about placement. Policy stays in
//! `family/expert_bank` (`bank_tier` chooses the rung, `host_tier` owns the
//! Linux O_DIRECT ring; the pooled backend in this module stages through
//! that ring itself, and only `pread_at_most` is shared with it).

mod buffered;
pub(crate) mod ring;

// `pread_at_most` is the tree's one positioned-read retry loop; the
// expert-bank host tier reads through it. The store types are re-exported
// here when their first callers land.
pub(crate) use buffered::pread_at_most;

use std::ops::Deref;

use crate::EngError;

/// One expert record site. The whole family-neutral vocabulary of a store:
/// callers say which layer and which expert, never which family or file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordSite {
    pub layer: usize,
    pub expert: u16,
}

/// Reads expert records from a backing store into host memory, in batches.
///
/// # Contract
///
/// - One lease per `sites` entry, in the same position: buffer `i` belongs
///   to `sites[i]`. Sites need not be sorted; a repeated site gets one
///   lease per entry. An empty batch returns an empty vector.
/// - A lease owns its bytes until dropped: `Deref` gives exactly that
///   record's bytes, and a pooled backend keeps the record's slot checked
///   out for as long as the lease lives. Nothing expires "on the next
///   call". A caller that enqueues an async device upload FROM a lease
///   must hold that lease until its own completion event fires; no store
///   checks that for it.
/// - A batch larger than [`max_batch`](Self::max_batch) is refused with
///   `EngError::BadRequest` before any I/O happens.
/// - A site outside the store's index is `EngError::BadRequest`, also
///   refused before any I/O.
/// - On `Err`, no lease is returned and no site's bytes are defined; the
///   store releases everything the batch had acquired internally.
/// - Bytes are readable the moment `read_records` returns. Calls may come
///   from several threads on one store; a pooled backend may serialize
///   readers internally, and when its pool cannot satisfy a batch it may
///   fail with `Backend` after a bounded wait rather than block forever.
/// - Read order is not observable and not promised; only the buffer-to-site
///   mapping above is.
pub(crate) trait ExpertRecordStore: Send + Sync {
    /// The guard this store hands out for one record: dereferences to the
    /// record's bytes and releases them (and any pool slot behind them)
    /// on drop.
    type Lease: Deref<Target = [u8]> + Send;

    /// Reads one lease per site, in order.
    fn read_records(&self, sites: &[RecordSite]) -> Result<Vec<Self::Lease>, EngError>;

    /// Bytes of one expert record. Every record has the same size; this
    /// drives tier arithmetic and batching bounds.
    fn record_bytes(&self) -> u64;

    /// The largest batch this store accepts in one call; a bigger batch is
    /// a `BadRequest` refusal before any I/O. Stores without a bound
    /// return `usize::MAX`.
    fn max_batch(&self) -> usize;
}

/// An out-of-index site: the caller asked for a layer or expert the store's
/// index never had.
pub(crate) fn site_outside_store(site: RecordSite) -> EngError {
    EngError::BadRequest(format!("expert record site {site:?} is outside the store"))
}

/// Index arithmetic that cannot be represented: a grid whose byte extent
/// overflows its integer type is an index defect, not a bad request.
pub(crate) fn index_arithmetic_overflow(what: &str) -> EngError {
    EngError::Checkpoint(format!("expert record index arithmetic overflows: {what}"))
}

/// A record size that cannot be addressed on this platform is an index
/// defect.
pub(crate) fn record_size_not_addressable(record_bytes: u64) -> EngError {
    EngError::Checkpoint(format!(
        "expert record size {record_bytes} does not fit this platform's addressable memory"
    ))
}
