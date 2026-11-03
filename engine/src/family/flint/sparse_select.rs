//! Logical-to-physical mapping and basis validation for the sparse KV path.
//!
//! The selection kernels speak logical positions (`cell = position`, the sketch and
//! top-K planes), while [`CudaContext::sparse_decode`] consumes physical cells of the
//! resident Q8 ring (`slot = position % cache_rows`, see `cuda_kv::KvAppend::slot`).
//! A physically valid slot is not a logically correct token: after a ring wrap the
//! slot holds a newer token, and after a prefix restore the committed prefix rewinds.
//! This module binds a selection to the cache basis it was derived from and refuses
//! — never silently omits — a token that is no longer resident: the caller takes the
//! dense Q8 fallback, which stays authoritative while the sparse path is staged.

use crate::EngError;

fn bad(msg: &str) -> EngError {
    EngError::BadRequest(format!("sparse selection: {msg}"))
}

/// The cache state a selection was derived against: the committed absolute
/// prefix and the per-layer ring capacity. The pair is the selection's
/// generation binding — `committed_prefix` moves on every append commit and on
/// every prefix restore, so any reuse or rewind is visible at validation time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectionBasis {
    committed_prefix: usize,
    cache_rows: usize,
}

impl SelectionBasis {
    pub(crate) fn new(committed_prefix: usize, cache_rows: usize) -> Result<Self, EngError> {
        if cache_rows == 0 {
            return Err(bad("cache_rows must not be zero"));
        }
        Ok(Self {
            committed_prefix,
            cache_rows,
        })
    }

    /// First position still resident: the ring holds positions
    /// `[resident_floor, committed_prefix)` — older cells were reused by wraps.
    fn resident_floor(&self) -> usize {
        self.committed_prefix.saturating_sub(self.cache_rows)
    }

    /// A position is resident iff it is committed and its cell has not been
    /// reused by a newer position (wrap) — `position >= resident_floor`.
    fn contains(&self, position: u32) -> bool {
        let p = position as usize;
        (self.resident_floor()..self.committed_prefix).contains(&p)
    }
}

/// A top-K selection bound to its cache basis. `positions` are logical token
/// positions in selection order (top-K first, then sinks and the recent
/// window, ascending), deduplicated by logical identity.
#[derive(Clone, Debug)]
pub(crate) struct SparseSelection {
    basis: SelectionBasis,
    positions: Vec<u32>,
}

impl SparseSelection {
    /// Binds a selection (the top-K output row for one KV head) to the basis it
    /// was scored against. Duplicate logical identities collapse to the first
    /// occurrence — the same token attended twice would double its weight, which
    /// is not the selected set. A position outside the basis residency is a
    /// broken selection and is refused here, before any mapping.
    pub(crate) fn bind(positions: &[u32], basis: SelectionBasis) -> Result<Self, EngError> {
        let mut deduped: Vec<u32> = Vec::with_capacity(positions.len());
        for &position in positions {
            if !basis.contains(position) {
                return Err(bad(&format!(
                    "position {position} is not resident in the selection basis \
                     (committed {}, cache_rows {})",
                    basis.committed_prefix, basis.cache_rows
                )));
            }
            if !deduped.contains(&position) {
                deduped.push(position);
            }
        }
        Ok(Self {
            basis,
            positions: deduped,
        })
    }

    pub(crate) fn positions(&self) -> &[u32] {
        &self.positions
    }

    pub(crate) fn len(&self) -> usize {
        self.positions.len()
    }

    /// Revalidates every selected token against the cache state at decode time.
    /// A ring wrap that reused a selected token's cell, a prefix restore that
    /// rewound the committed prefix past a selected token, or a reallocation
    /// that changed the ring capacity is an explicit refusal: no token is
    /// silently dropped, because that would change the effective attention set
    /// without a quality decision. The caller takes the dense fallback.
    pub(crate) fn validate_against(&self, now: SelectionBasis) -> Result<(), EngError> {
        if now.cache_rows != self.basis.cache_rows {
            return Err(bad(&format!(
                "cache basis changed: selection has cache_rows {}, decode state has {}",
                self.basis.cache_rows, now.cache_rows
            )));
        }
        for &position in &self.positions {
            if !now.contains(position) {
                return Err(bad(&format!(
                    "selected token {position} is no longer resident \
                     (committed {}, cache_rows {})",
                    now.committed_prefix, now.cache_rows
                )));
            }
        }
        Ok(())
    }

    /// Physical ring cells for [`CudaContext::sparse_decode`], in selection
    /// order. Only meaningful after [`Self::validate_against`] accepted the
    /// decode-time basis: without that check an evicted token's cell would map
    /// to whichever newer token reused it.
    pub(crate) fn cells(&self) -> Vec<u32> {
        self.positions
            .iter()
            .map(|&p| p % self.basis.cache_rows as u32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind(positions: &[u32], committed: usize, rows: usize) -> SparseSelection {
        SparseSelection::bind(positions, SelectionBasis::new(committed, rows).unwrap()).unwrap()
    }

    #[test]
    fn bind_dedups_by_logical_identity_keeping_selection_order() {
        // picks, then sinks, then the recent window — the same logical token can
        // appear twice; it must count once.
        let s = bind(&[7, 12, 7, 0, 1, 62, 63], 64, 128);
        assert_eq!(s.positions(), &[7, 12, 0, 1, 62, 63]);
    }

    #[test]
    fn bind_refuses_a_position_outside_the_basis() {
        let err = SparseSelection::bind(&[64], SelectionBasis::new(64, 128).unwrap());
        assert!(
            err.is_err(),
            "a position at the committed prefix is not a token yet"
        );
        assert!(SparseSelection::bind(&[0], SelectionBasis::new(0, 128).unwrap()).is_err());
    }

    #[test]
    fn validate_accepts_growth_while_tokens_stay_resident() {
        let s = bind(&[0, 31, 63], 64, 128);
        s.validate_against(SelectionBasis::new(100, 128).unwrap())
            .expect("no wrap happened; every token is still resident");
    }

    #[test]
    fn validate_refuses_a_wrap_evicted_token_instead_of_omitting_it() {
        let s = bind(&[0, 63], 64, 128);
        // Committing position 128 reuses slot 0 (position 0 + 128): token 0 is gone.
        let msg = match s.validate_against(SelectionBasis::new(129, 128).unwrap()) {
            Err(EngError::BadRequest(msg)) => msg,
            other => panic!("expected BadRequest, got {other:?}"),
        };
        assert!(msg.contains("no longer resident"), "{msg}");
        assert!(
            msg.contains("token 0"),
            "the refusal must name the token: {msg}"
        );
    }

    #[test]
    fn validate_refuses_a_prefix_restore_that_rewinds_past_a_token() {
        let s = bind(&[90, 95], 100, 128);
        // A prefix restore rewinds the committed prefix to 80: tokens 90 and 95
        // are now beyond the committed state they were selected against.
        assert!(
            matches!(
                s.validate_against(SelectionBasis::new(80, 128).unwrap()),
                Err(EngError::BadRequest(_))
            ),
            "the restore rewound past the selected tokens"
        );
    }

    #[test]
    fn validate_refuses_a_changed_cache_capacity() {
        let s = bind(&[10], 64, 128);
        assert!(
            s.validate_against(SelectionBasis::new(64, 256).unwrap())
                .is_err(),
            "a reallocated ring is a different basis even at the same prefix"
        );
    }

    #[test]
    fn residency_boundaries_are_exact() {
        let s = bind(&[0, 31], 32, 32);
        s.validate_against(SelectionBasis::new(32, 32).unwrap())
            .expect("the resident floor is inclusive: token 0 still holds slot 0");
        s.validate_against(SelectionBasis::new(33, 32).unwrap())
            .expect_err("position 32 reused slot 0, so token 0 is gone");
        let later = bind(&[1, 32], 33, 32);
        later
            .validate_against(SelectionBasis::new(33, 32).unwrap())
            .expect("token 1 is the floor now and stays resident");
    }

    #[test]
    fn cells_maps_positions_to_ring_slots_and_never_aliases() {
        // Distinct resident positions always take distinct slots: the residency
        // window spans exactly cache_rows, so two positions sharing a slot would
        // have to be the same position. Aliasing is therefore only reachable
        // through an invalid basis - which bind and validate refuse.
        let s = bind(&[6, 130, 133], 134, 128);
        assert_eq!(s.cells(), vec![6, 2, 5]);
        let cells = s.cells();
        let mut sorted = cells.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            cells.len(),
            "cells of a bound selection alias"
        );
    }
}
