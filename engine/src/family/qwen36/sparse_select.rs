//! Selection binding for the qwen36 resident sparse path.
//!
//! The selection kernels speak logical positions (`cell = position`, the sketch and
//! top-K planes) and the resident Q8 decode consumes the same positions directly:
//! the qwen36 resident cache is linear (`attn_decode_q8` runs `ring = 0`, so
//! `kv_len <= cells` and cell equals position - no wrap, no eviction within
//! capacity). What can still invalidate a selection is TIME: the visible prefix
//! moves on every append and a prefix restore rewinds it. This module binds a
//! selection to the visible-key count it was derived from and refuses - never
//! silently omits - a token that is no longer visible: the caller takes the dense
//! Q8 fallback, which stays authoritative while the sparse path is staged.
//!
//! The flint ring variant (`pos % cache_rows` with a residency window) lives in
//! history at commit 84473b986; if a future cache changes the linear invariant,
//! the mapping must be revised, not reused.

use crate::EngError;

fn bad(msg: &str) -> EngError {
    EngError::BadRequest(format!("sparse selection: {msg}"))
}

/// The cache state a selection was derived against: the visible-key count and
/// the plane capacity. The pair is the selection's generation binding - it
/// moves on every append commit and on every prefix restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SelectionBasis {
    visible_keys: usize,
    capacity: usize,
}

impl SelectionBasis {
    pub(crate) fn new(visible_keys: usize, capacity: usize) -> Result<Self, EngError> {
        if capacity == 0 {
            return Err(bad("capacity must not be zero"));
        }
        if visible_keys > capacity {
            return Err(bad("visible keys exceed the plane capacity"));
        }
        Ok(Self {
            visible_keys,
            capacity,
        })
    }

    fn contains(&self, position: u32) -> bool {
        (position as usize) < self.visible_keys
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
    /// Binds a selection (one top-K output row) to the basis it was scored
    /// against. Duplicate logical identities collapse to the first occurrence -
    /// the same token attended twice would double its weight, which is not the
    /// selected set. A position outside the visible prefix is refused here,
    /// before any mapping.
    pub(crate) fn bind(positions: &[u32], basis: SelectionBasis) -> Result<Self, EngError> {
        let mut deduped: Vec<u32> = Vec::with_capacity(positions.len());
        for &position in positions {
            if !basis.contains(position) {
                return Err(bad(&format!(
                    "position {position} is not visible in the selection basis \
                     (visible {}, capacity {})",
                    basis.visible_keys, basis.capacity
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
    /// Growth is fine while the basis capacity holds; a prefix restore that
    /// rewound the visible count past a selected token, or a reallocated plane,
    /// is an explicit refusal: no token is silently dropped, because that would
    /// change the effective attention set without a quality decision. The
    /// caller takes the dense fallback.
    pub(crate) fn validate_against(&self, now: SelectionBasis) -> Result<(), EngError> {
        if now.capacity != self.basis.capacity {
            return Err(bad(&format!(
                "cache basis changed: selection has capacity {}, decode state has {}",
                self.basis.capacity, now.capacity
            )));
        }
        for &position in &self.positions {
            if !now.contains(position) {
                return Err(bad(&format!(
                    "selected token {position} is no longer visible (visible {})",
                    now.visible_keys
                )));
            }
        }
        Ok(())
    }

    /// The resident plane cells for the sparse decode, in selection order.
    /// The linear cache maps position to cell identically; only meaningful
    /// after [`Self::validate_against`] accepted the decode-time basis.
    pub(crate) fn cells(&self) -> Vec<u32> {
        self.positions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bind(positions: &[u32], visible: usize, capacity: usize) -> SparseSelection {
        SparseSelection::bind(positions, SelectionBasis::new(visible, capacity).unwrap()).unwrap()
    }

    #[test]
    fn bind_dedups_by_logical_identity_keeping_selection_order() {
        let s = bind(&[7, 12, 7, 0, 1, 62, 63], 64, 128);
        assert_eq!(s.positions(), &[7, 12, 0, 1, 62, 63]);
    }

    #[test]
    fn bind_refuses_positions_outside_the_visible_prefix() {
        assert!(
            SparseSelection::bind(&[64], SelectionBasis::new(64, 128).unwrap()).is_err(),
            "a position at the visible count is not a token yet"
        );
        assert!(
            bind(&[63], 64, 128).len() == 1,
            "the last visible token binds"
        );
    }

    #[test]
    fn validate_accepts_growth_within_capacity() {
        let s = bind(&[0, 31, 63], 64, 128);
        s.validate_against(SelectionBasis::new(100, 128).unwrap())
            .expect("growth keeps earlier tokens visible");
    }

    #[test]
    fn validate_refuses_a_restore_rewind_instead_of_omitting() {
        let s = bind(&[90, 95], 100, 128);
        let msg = match s.validate_against(SelectionBasis::new(80, 128).unwrap()) {
            Err(EngError::BadRequest(msg)) => msg,
            other => panic!("expected BadRequest, got {other:?}"),
        };
        assert!(msg.contains("no longer visible"), "{msg}");
        assert!(
            msg.contains("token 90"),
            "the refusal must name the token: {msg}"
        );
    }

    #[test]
    fn validate_refuses_a_changed_capacity() {
        let s = bind(&[10], 64, 128);
        assert!(
            s.validate_against(SelectionBasis::new(64, 256).unwrap())
                .is_err(),
            "a reallocated plane is a different basis even at the same visible count"
        );
    }

    #[test]
    fn cells_are_positions_under_the_linear_contract() {
        let s = bind(&[5, 130, 133], 134, 256);
        assert_eq!(s.cells(), vec![5, 130, 133]);
    }

    #[test]
    fn visible_count_may_not_exceed_capacity() {
        assert!(SelectionBasis::new(129, 128).is_err());
    }
}
