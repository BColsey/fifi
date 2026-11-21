//! RAM residency of records just evicted from the CUDA disk-tier bank.

use crate::EngError;
use std::collections::VecDeque;

pub(super) struct VictimLayout {
    records: usize,
    reserve: usize,
    record_bytes: usize,
}

impl VictimLayout {
    pub(super) fn new(
        budget: usize,
        reserve: usize,
        record_bytes: usize,
    ) -> Result<Self, EngError> {
        let records = budget.checked_div(record_bytes).unwrap_or(0);
        if records == 0
            || records
                .checked_add(reserve)
                .and_then(|n| n.checked_mul(record_bytes))
                .is_none()
        {
            return Err(EngError::MemoryPlan(
                "Flash-Next victim cache needs a representable budget of at least one record"
                    .into(),
            ));
        }
        Ok(Self {
            records,
            reserve,
            record_bytes,
        })
    }

    pub(super) fn resident_records(&self) -> usize {
        self.records
    }

    pub(super) fn bytes(&self) -> usize {
        (self.records + self.reserve) * self.record_bytes
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) struct Key {
    pub(super) layer: usize,
    pub(super) expert: u16,
}

/// The two most recent distinct decode positions for each (layer, expert).
pub(super) struct RecentPicks {
    positions: Vec<[u64; 2]>,
    experts: usize,
}

impl RecentPicks {
    const NEVER: [u64; 2] = [u64::MAX; 2];
    const WINDOW: u64 = 16;

    pub(super) fn new(layers: usize, experts: usize) -> Self {
        Self {
            positions: vec![Self::NEVER; layers * experts],
            experts,
        }
    }

    pub(super) fn bytes(layers: usize, experts: usize) -> u64 {
        (layers * experts * size_of::<[u64; 2]>()) as u64
    }

    pub(super) fn observe(&mut self, layer: usize, picks: &[i32], position: u64) {
        for expert in picks.iter().filter_map(|&id| usize::try_from(id).ok()) {
            let pair = &mut self.positions[layer * self.experts + expert];
            if pair[0] != position {
                *pair = [position, pair[0]];
            }
        }
    }

    pub(super) fn keep(&self, key: Key, position: u64) -> bool {
        let [last, previous] = self.positions[key.layer * self.experts + usize::from(key.expert)];
        last < position
            && position
                .checked_sub(previous)
                .is_some_and(|age| age <= Self::WINDOW)
    }

    pub(super) fn clear(&mut self) {
        self.positions.fill(Self::NEVER);
    }
}

/// Least recent at the front. A taken slot remains reserved until all the
/// uploads that use it have been queued. Spare slots cover those loans.
pub(super) struct VictimIndex {
    entries: VecDeque<(Key, usize)>,
    free: Vec<usize>,
    capacity: usize,
    slots: usize,
}

impl VictimIndex {
    pub(super) fn new(layout: &VictimLayout) -> Self {
        let slots = layout.records + layout.reserve;
        Self {
            entries: VecDeque::with_capacity(layout.records),
            free: (0..slots).rev().collect(),
            capacity: layout.records,
            slots,
        }
    }

    pub(super) fn take(&mut self, key: Key) -> Option<usize> {
        let index = self.entries.iter().position(|&(entry, _)| entry == key)?;
        Some(self.entries.remove(index)?.1)
    }

    /// The slot a victim record is saved in. Fails when every spare slot is loaned to a hit
    /// upload, which only a step with more distinct misses than the layout's reserve can do.
    pub(super) fn insert(&mut self, key: Key) -> Result<usize, EngError> {
        let slot = match self.take(key) {
            Some(slot) => slot,
            None if self.entries.len() == self.capacity => {
                self.entries.pop_front().expect("nonempty victim cache").1
            }
            None => self.free.pop().ok_or_else(|| {
                EngError::BadRequest(
                    "the Flash-Next victim cache has no spare slot: a decode step carried more \
                     distinct misses than the cache reserves for hit uploads"
                        .into(),
                )
            })?,
        };
        self.entries.push_back((key, slot));
        Ok(slot)
    }

    pub(super) fn release(&mut self, slot: usize) {
        self.free.push(slot);
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.free.clear();
        self.free.extend((0..self.slots).rev());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(layer: usize, expert: u16) -> Key {
        Key { layer, expert }
    }

    #[test]
    fn admission_counts_distinct_recent_positions_in_the_same_layer() {
        let mut recent = RecentPicks::new(2, 4);
        assert_eq!(RecentPicks::bytes(2, 4), 128);
        recent.observe(0, &[3, -1, 3], 8);
        recent.observe(1, &[3], 9);
        assert!(!recent.keep(key(0, 3), 10));
        recent.observe(0, &[3], 12);
        assert!(!recent.keep(key(0, 3), 12));
        assert!(recent.keep(key(0, 3), 24));
        assert!(!recent.keep(key(0, 3), 25));
        assert!(!recent.keep(key(1, 3), 24));
        recent.clear();
        assert!(!recent.keep(key(0, 3), 24));
    }

    #[test]
    fn hit_slots_stay_reserved_while_victims_are_saved() {
        let layout = VictimLayout::new(2048, 2, 1024).unwrap();
        assert_eq!(layout.bytes(), 4096);
        assert_eq!(layout.resident_records(), 2);
        let mut cache = VictimIndex::new(&layout);
        let first = cache.insert(key(0, 4)).unwrap();
        let second = cache.insert(key(1, 4)).unwrap();
        assert_eq!(cache.take(key(0, 4)), Some(first));
        assert_eq!(cache.take(key(1, 4)), Some(second));
        let third = cache.insert(key(0, 8)).unwrap();
        let fourth = cache.insert(key(1, 8)).unwrap();
        assert!(![first, second].contains(&third));
        assert!(![first, second, third].contains(&fourth));
        cache.release(first);
        cache.release(second);
        assert_eq!(cache.take(key(0, 4)), None);
        assert_eq!(
            cache.insert(key(2, 4)).unwrap(),
            third,
            "evict the oldest victim"
        );
        assert_eq!(cache.take(key(1, 8)), Some(fourth));
        cache.clear();
        assert_eq!(cache.take(key(2, 4)), None);
        assert_eq!(cache.insert(key(0, 4)).unwrap(), first);
    }

    #[test]
    fn more_hit_loans_than_the_reserve_is_an_error_not_a_panic() {
        let layout = VictimLayout::new(4096, 1, 1024).unwrap();
        let mut cache = VictimIndex::new(&layout);
        for expert in 0..3 {
            cache.insert(key(0, expert)).unwrap();
        }
        // Two hits are loaned out and the reserve is one slot: entries 1 and 2 of 4 slots.
        assert!(cache.take(key(0, 0)).is_some());
        assert!(cache.take(key(0, 1)).is_some());
        cache.insert(key(1, 0)).unwrap();
        cache.insert(key(1, 1)).unwrap();
        let error = cache.insert(key(1, 2)).unwrap_err().describe();
        assert!(error.contains("no spare slot"), "{error}");
    }
}
