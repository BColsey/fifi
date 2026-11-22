//! The cell geometry of a sliding-window layer's KV cache. A layer that attends to the last
//! `window` positions needs `window + rows - 1` cells, not the whole context: position `p` lives
//! in cell `p % ring`. A layer without a window keeps `cell = position`. The kernels take the same
//! `ring` number (`kv8_append`, `attn_decode_gemma`), and the host side (plane sizes, the cells a
//! snapshot copies) reads it from here, so a family states its window once.

use std::ops::Range;

use super::spec::{AttentionKind, FamilySpec};

/// The cache geometry of one layer kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvRing {
    /// Positions a query can reach; 0 for a layer that reaches the whole context.
    window: usize,
    /// Cells a sliding layer keeps: `window + rows - 1`; 0 when `window` is 0.
    ring: usize,
}

/// A contiguous piece of one plane head's cells.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
    )
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CellRun {
    /// First cell of the run.
    pub(crate) cell: usize,
    /// Cells in the run.
    pub(crate) rows: usize,
}

impl KvRing {
    /// A layer that attends to every position.
    pub(crate) const LINEAR: Self = Self { window: 0, ring: 0 };

    /// A layer that attends to the last `window` positions and is fed at most `rows` tokens at a
    /// time (the widest decode step or prefill chunk). `window` 0 is the linear layer.
    pub(crate) fn sliding(window: usize, rows: usize) -> Self {
        if window == 0 {
            return Self::LINEAR;
        }
        Self {
            window,
            ring: window + rows.max(1) - 1,
        }
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn window(self) -> usize {
        self.window
    }

    /// Whether a cache for `context` positions is smaller as a ring than as a linear plane.
    pub(crate) fn is_ring(self, context: usize) -> bool {
        self.window > 0 && self.ring < context
    }

    /// The `ring` argument of the kernels: the cell count of a ring, 0 for a linear plane.
    #[cfg_attr(not(feature = "native-cuda"), allow(dead_code))]
    pub(crate) fn kernel_ring(self, context: usize) -> usize {
        if self.is_ring(context) { self.ring } else { 0 }
    }

    /// The cells of one head's plane for `context` positions.
    pub(crate) fn cells(self, context: usize) -> usize {
        if self.is_ring(context) {
            self.ring
        } else {
            context
        }
    }

    /// The positions a snapshot of a sequence fed `fed` tokens must keep: the last `window` (all
    /// of them for a linear layer). A ring layer can only be restored at exactly `fed`, because
    /// older positions are overwritten.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn kept(self, fed: usize) -> Range<usize> {
        if self.window == 0 {
            0..fed
        } else {
            fed.saturating_sub(self.window)..fed
        }
    }

    /// The cell runs that hold the kept positions of one head, in position order: one run for a
    /// linear plane or a ring that has not wrapped over them, two when they wrap.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn runs(self, context: usize, fed: usize) -> Vec<CellRun> {
        let kept = self.kept(fed);
        if kept.is_empty() {
            return Vec::new();
        }
        if !self.is_ring(context) {
            return vec![CellRun {
                cell: kept.start,
                rows: kept.len(),
            }];
        }
        let first = kept.start % self.ring;
        let head = kept.len().min(self.ring - first);
        let mut runs = vec![CellRun {
            cell: first,
            rows: head,
        }];
        if head < kept.len() {
            runs.push(CellRun {
                cell: 0,
                rows: kept.len() - head,
            });
        }
        runs
    }

    /// The `(byte offset, bytes)` ranges of every kv head's kept cells in a plane laid out
    /// `[kv head][cell][unit]`, head by head, each head's runs in position order.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn plane_ranges(
        self,
        context: usize,
        fed: usize,
        kv_heads: usize,
        unit: usize,
    ) -> Vec<(usize, usize)> {
        let cells = self.cells(context);
        let runs = self.runs(context, fed);
        (0..kv_heads)
            .flat_map(|head| {
                runs.iter()
                    .map(move |run| ((head * cells + run.cell) * unit, run.rows * unit))
            })
            .collect()
    }

    /// The bytes a plane of `kv_heads` heads holds for the kept positions of a snapshot.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn kept_bytes(self, fed: usize, kv_heads: usize, unit: usize) -> usize {
        self.kept(fed).len() * kv_heads * unit
    }
}

impl FamilySpec {
    /// The cache geometry of a layer of `kind` that is fed at most `rows` tokens at a time. A
    /// sliding layer reaches `sliding_window` positions; every other kind reaches the whole
    /// context.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the Bolt snapshot reads the ring cells when bolt.rs takes the ring"
        )
    )]
    pub(crate) fn kv_ring(&self, kind: AttentionKind, rows: usize) -> KvRing {
        match kind {
            AttentionKind::Sliding => KvRing::sliding(self.sliding_window as usize, rows),
            _ => KvRing::LINEAR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::spec::NEMOTRON_BOLT_E4B;

    #[test]
    fn a_ring_keeps_the_window_plus_the_widest_step() {
        let ring = KvRing::sliding(512, 16);
        assert_eq!(ring.cells(30_464), 527);
        assert_eq!(ring.kernel_ring(30_464), 527);
        // A context no larger than the ring is a linear plane.
        assert_eq!(ring.cells(500), 500);
        assert_eq!(ring.kernel_ring(500), 0);
        assert_eq!(KvRing::LINEAR.cells(30_464), 30_464);
        assert_eq!(KvRing::LINEAR.kernel_ring(30_464), 0);
        assert_eq!(KvRing::sliding(0, 16), KvRing::LINEAR);
    }

    #[test]
    fn the_kept_positions_are_the_last_window() {
        let ring = KvRing::sliding(512, 1);
        assert_eq!(ring.kept(100), 0..100);
        assert_eq!(ring.kept(512), 0..512);
        assert_eq!(ring.kept(5_000), 4_488..5_000);
        assert_eq!(KvRing::LINEAR.kept(5_000), 0..5_000);
    }

    #[test]
    fn runs_split_where_the_ring_wraps_and_cover_the_kept_positions_in_order() {
        let ring = KvRing::sliding(8, 3); // 10 cells
        assert_eq!(ring.cells(100), 10);
        // Unwrapped: positions 0..5 sit in cells 0..5.
        assert_eq!(ring.runs(100, 5), vec![CellRun { cell: 0, rows: 5 }]);
        // fed 12: kept 4..12, cells 4..10 then 0..2.
        assert_eq!(
            ring.runs(100, 12),
            vec![CellRun { cell: 4, rows: 6 }, CellRun { cell: 0, rows: 2 }]
        );
        // A restore of every kept position reads each cell exactly once.
        for fed in 0..200 {
            let mut seen = [0u32; 10];
            let mut position = ring.kept(fed).start;
            for run in ring.runs(100, fed) {
                for row in 0..run.rows {
                    assert_eq!(run.cell + row, position % 10, "fed {fed}");
                    seen[run.cell + row] += 1;
                    position += 1;
                }
            }
            assert_eq!(position, fed.max(ring.kept(fed).start), "fed {fed}");
            assert!(seen.iter().all(|&count| count <= 1), "fed {fed}");
        }
    }

    #[test]
    fn a_linear_plane_keeps_the_whole_prefix_and_the_ranges_match_the_old_layout() {
        // [kv head][cell][unit] with 2 heads, 100 cells, 4 bytes a cell, 30 positions.
        let ranges = KvRing::LINEAR.plane_ranges(100, 30, 2, 4);
        assert_eq!(ranges, vec![(0, 120), (400, 120)]);
        assert_eq!(KvRing::LINEAR.kept_bytes(30, 2, 4), 240);
    }

    #[test]
    fn ring_ranges_stay_inside_the_ring_plane_and_total_the_kept_bytes() {
        let ring = KvRing::sliding(8, 3);
        let ranges = ring.plane_ranges(100, 12, 2, 4);
        // Head 1 starts 10 cells * 4 bytes in; each head has two runs.
        assert_eq!(ranges, vec![(16, 24), (0, 8), (56, 24), (40, 8)]);
        let plane_bytes = 2 * ring.cells(100) * 4;
        assert!(ranges.iter().all(|&(at, len)| at + len <= plane_bytes));
        assert_eq!(
            ranges.iter().map(|range| range.1).sum::<usize>(),
            ring.kept_bytes(12, 2, 4)
        );
    }

    #[test]
    fn the_spec_declares_the_window_and_only_sliding_layers_ring() {
        let spec = NEMOTRON_BOLT_E4B;
        let sliding = spec.kv_ring(AttentionKind::Sliding, 1);
        assert_eq!(sliding.window(), 512);
        assert_eq!(sliding.cells(30_464), 512);
        assert_eq!(spec.kv_ring(AttentionKind::Full, 1), KvRing::LINEAR);
    }
}
