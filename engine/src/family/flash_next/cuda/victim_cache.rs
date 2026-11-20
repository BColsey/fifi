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

    pub(super) fn insert(&mut self, key: Key) -> usize {
        let slot = self.take(key).unwrap_or_else(|| {
            if self.entries.len() == self.capacity {
                self.entries.pop_front().expect("nonempty victim cache").1
            } else {
                self.free.pop().expect("spare slots cover hit uploads")
            }
        });
        self.entries.push_back((key, slot));
        slot
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
    fn hit_slots_stay_reserved_while_victims_are_saved() {
        let layout = VictimLayout::new(2048, 2, 1024).unwrap();
        assert_eq!(layout.bytes(), 4096);
        assert_eq!(layout.resident_records(), 2);
        let mut cache = VictimIndex::new(&layout);
        let first = cache.insert(key(0, 4));
        let second = cache.insert(key(1, 4));
        assert_eq!(cache.take(key(0, 4)), Some(first));
        assert_eq!(cache.take(key(1, 4)), Some(second));
        let third = cache.insert(key(0, 8));
        let fourth = cache.insert(key(1, 8));
        assert!(![first, second].contains(&third));
        assert!(![first, second, third].contains(&fourth));
        cache.release(first);
        cache.release(second);
        assert_eq!(cache.take(key(0, 4)), None);
        assert_eq!(cache.insert(key(2, 4)), third, "evict the oldest victim");
        assert_eq!(cache.take(key(1, 8)), Some(fourth));
        cache.clear();
        assert_eq!(cache.take(key(2, 4)), None);
        assert_eq!(cache.insert(key(0, 4)), first);
    }
}
