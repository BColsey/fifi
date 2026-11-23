//! Compact selected experts without changing the expert-major activation/output layout.
use std::ops::Range;

pub(crate) struct GroupedDqRows {
    // Keep the table at the full output extent for the grouped launch's range checks.
    // The bounds end at `selected`; ids after that point are never dereferenced.
    ids: Vec<i32>,
    bounds: Vec<i32>,
    selected: usize,
}

impl GroupedDqRows {
    /// Groups must cover `0..rows` once, in increasing expert order. Empty experts
    /// are represented by gaps in the expert indices, not by zero-length groups.
    pub(crate) fn new(
        experts: usize,
        rows: usize,
        minimum: usize,
        groups: impl IntoIterator<Item = (usize, Range<usize>)>,
    ) -> Option<Self> {
        if experts == 0 || experts > 65_535 || rows == 0 || rows > 65_536 || minimum == 0 {
            return None;
        }
        let mut ids = Vec::with_capacity(rows);
        let mut bounds = vec![0; experts + 1];
        let (mut next_expert, mut next_row) = (0, 0);
        for (expert, span) in groups {
            if expert < next_expert
                || expert >= experts
                || span.start != next_row
                || span.is_empty()
                || span.end > rows
            {
                return None;
            }
            bounds[next_expert..=expert].fill(ids.len() as i32);
            next_row = span.end;
            next_expert = expert + 1;
            if span.len() >= minimum {
                ids.extend(span.map(|slot| slot as i32));
            }
        }
        if next_row != rows || ids.is_empty() {
            return None;
        }
        let selected = ids.len();
        bounds[next_expert..].fill(selected as i32);
        ids.resize(rows, 0);
        Some(Self {
            ids,
            bounds,
            selected,
        })
    }

    pub(crate) fn ids(&self) -> &[i32] {
        &self.ids
    }

    pub(crate) fn bounds(&self) -> &[i32] {
        &self.bounds
    }

    pub(crate) fn selected(&self) -> usize {
        self.selected
    }

    pub(crate) fn includes(&self, expert: usize) -> bool {
        expert < self.bounds.len() - 1 && self.bounds[expert] < self.bounds[expert + 1]
    }
}

#[cfg(test)]
mod tests {
    use super::GroupedDqRows;

    #[test]
    fn sparse_tile_plan_preserves_expert_major_slots() {
        let p = GroupedDqRows::new(8, 10, 3, [(1, 0..3), (2, 3..4), (4, 4..8), (6, 8..10)])
            .expect("two qualifying groups");
        assert_eq!(p.ids(), [0, 1, 2, 4, 5, 6, 7, 0, 0, 0]);
        assert_eq!(p.bounds(), [0, 0, 3, 3, 3, 7, 7, 7, 7]);
        assert_eq!(p.selected(), 7);
        assert_eq!(
            (0..10).filter(|&id| p.includes(id)).collect::<Vec<_>>(),
            [1, 4]
        );
    }

    #[test]
    fn tile_plan_refuses_overlapping_missing_or_invalid_groups() {
        for groups in [
            [(0, 0..2), (1, 1..4)], // overlapping rows
            [(0, 0..2), (1, 3..4)], // missing row
            [(0, 0..2), (1, 2..3)], // missing tail
            [(0, 0..2), (1, 2..5)], // out of range
            [(0, 0..2), (3, 2..4)], // invalid expert
            [(1, 0..2), (0, 2..4)], // out of order
            [(1, 0..2), (1, 2..4)], // repeated expert
        ] {
            assert!(GroupedDqRows::new(3, 4, 1, groups).is_none());
        }
        assert!(GroupedDqRows::new(3, 4, 0, [(1, 0..4)]).is_none());
        assert!(GroupedDqRows::new(3, 4, 5, [(1, 0..4)]).is_none());
    }
}
