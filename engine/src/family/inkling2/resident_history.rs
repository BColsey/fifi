//! Rows needed in a device attention ring before its next append.

use std::ops::Range;

/// The host position is the next committed row to append. A known device
/// cursor describes a contiguous prefix of the current lineage, which may
/// include speculative rows the host discarded. An unknown cursor, or one
/// behind the host, has no useful history guarantee.
pub(super) fn upload_rows(
    position: usize,
    device_cursor: Option<usize>,
    capacity: usize,
) -> Range<usize> {
    match device_cursor {
        Some(cursor) if cursor == position => position..position,
        Some(cursor) if cursor > position => {
            // The new row replaces its own slot before it can be read, so
            // only the preceding capacity - 1 rows can need repair. A
            // retained row q was overwritten iff the discarded range
            // reached q + capacity.
            let first = position.saturating_sub(capacity - 1);
            let end = position.min(cursor.saturating_sub(capacity));
            first..end.max(first)
        }
        _ => position.saturating_sub(capacity)..position,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_lineage_or_advancing_the_host_reseeds_its_visible_ring() {
        assert_eq!(upload_rows(0, None, 4), 0..0);
        assert_eq!(upload_rows(2, None, 4), 0..2);
        assert_eq!(upload_rows(8, None, 4), 4..8);
        assert_eq!(upload_rows(8, Some(6), 4), 4..8);
        assert_eq!(upload_rows(8, Some(8), 4), 8..8);
    }

    #[test]
    fn discarded_rows_repair_only_overwritten_keys_visible_to_the_next_append() {
        for capacity in 1..=19 {
            for position in 0_usize..=3 * capacity {
                for discarded in 0..=2 * capacity {
                    let cursor = position + discarded;
                    let mut physical = vec![None; capacity];
                    for committed in position.saturating_sub(capacity)..position {
                        physical[committed % capacity] = Some(committed);
                    }
                    for rejected in position..cursor {
                        physical[rejected % capacity] = Some(rejected);
                    }

                    let visible = position.saturating_sub(capacity - 1)..position;
                    let overwritten: Vec<_> = visible
                        .clone()
                        .filter(|&row| physical[row % capacity] != Some(row))
                        .collect();
                    let repair = upload_rows(position, Some(cursor), capacity);
                    assert_eq!(
                        repair.clone().collect::<Vec<_>>(),
                        overwritten,
                        "capacity {capacity}, position {position}, discarded {discarded}",
                    );
                    for row in repair {
                        physical[row % capacity] = Some(row);
                    }
                    for row in visible {
                        assert_eq!(physical[row % capacity], Some(row));
                    }
                    physical[position % capacity] = Some(position);
                    for row in (position + 1).saturating_sub(capacity)..=position {
                        assert_eq!(physical[row % capacity], Some(row));
                    }
                }
            }
        }
    }
}
