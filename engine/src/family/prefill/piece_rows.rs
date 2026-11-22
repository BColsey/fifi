//! The rows of one prefill piece at a depth, from a latency target and the device.
//!
//! A deep prompt re-reads the prefix for every query tile, so the time of one row grows with the
//! depth it attends over. A piece of rows is cancellable only between pieces, so its time is
//! bounded: the rows of a piece shrink as the depth grows, until the piece takes about the same
//! time at every depth. The rule is a model of the time of one row, linear in depth and inverse
//! in the multiprocessor count, calibrated by two measured anchors: the depth at which a piece
//! of `near.rows` rows reaches the target on the card that was measured, and the depth at which a
//! piece of `far.rows` rows does. The rows at a depth are the largest power of two the model
//! allows, so the pieces of one depth take the same shape on every run.

/// The multiprocessor count of the card the anchors were measured on (an RTX 3090).
pub(crate) const ANCHOR_SM_COUNT: usize = 82;

/// The fewest rows of a piece the rule returns.
const MIN_PIECE_ROWS: usize = 128;

/// One measured point: a piece of `rows` rows ending at `depth` positions takes the target time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Anchor {
    pub(crate) depth: usize,
    pub(crate) rows: usize,
}

impl Anchor {
    pub(crate) const fn at(depth: usize, rows: usize) -> Self {
        Self { depth, rows }
    }
}

/// The rows a piece may have when it ends at `depth` positions on a device of `sm_count`
/// multiprocessors (the anchors' card when it is zero): the largest power of two of the model's
/// rows, never under [`MIN_PIECE_ROWS`]. The depths between and beyond the anchors follow the
/// line through them; a shallower depth allows more rows than either anchor.
pub(crate) fn allowed_rows(near: Anchor, far: Anchor, sm_count: usize, depth: usize) -> usize {
    let (near_depth, far_depth) = (near.depth as i128, far.depth as i128);
    let (near_rows, far_rows) = (near.rows as i128, far.rows as i128);
    // Time per row is 1/near.rows at the near depth and 1/far.rows at the far one, and linear
    // between: its reciprocal is the rows the target allows.
    let span = far_depth - near_depth;
    let denominator = far_rows * span + (depth as i128 - near_depth) * (near_rows - far_rows);
    if span <= 0 || denominator <= 0 {
        return MIN_PIECE_ROWS;
    }
    let sm = if sm_count == 0 {
        ANCHOR_SM_COUNT
    } else {
        sm_count
    } as i128;
    let raw = near_rows * far_rows * span * sm / (denominator * ANCHOR_SM_COUNT as i128);
    let raw = usize::try_from(raw).unwrap_or(usize::MAX);
    match raw {
        0..=MIN_PIECE_ROWS => MIN_PIECE_ROWS,
        _ => 1usize << raw.ilog2(),
    }
}

/// `rows` limited to what a piece ending `rows` after `p0` may have.
pub(crate) fn piece_rows(
    (near, far): (Anchor, Anchor),
    sm_count: usize,
    p0: usize,
    rows: usize,
) -> usize {
    rows.min(allowed_rows(near, far, sm_count, p0.saturating_add(rows)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEAR: Anchor = Anchor {
        depth: 73_728,
        rows: 1024,
    };
    const FAR: Anchor = Anchor {
        depth: 155_648,
        rows: 512,
    };

    #[test]
    fn the_anchors_are_where_the_rows_halve_on_the_card_they_were_measured_on() {
        let at = |depth| allowed_rows(NEAR, FAR, ANCHOR_SM_COUNT, depth);
        assert_eq!(at(73_728), 1024);
        assert_eq!(at(73_729), 512);
        assert_eq!(at(155_648), 512);
        assert_eq!(at(155_649), 256);
        assert_eq!(at(262_144), 256);
        // Shallower depths allow more than the anchors; nothing is under the floor.
        assert!(at(0) >= 4096);
        assert_eq!(at(usize::MAX / 2), MIN_PIECE_ROWS);
    }

    #[test]
    fn the_rows_never_grow_with_depth() {
        let mut last = usize::MAX;
        for depth in (0..400_000).step_by(997) {
            let rows = allowed_rows(NEAR, FAR, ANCHOR_SM_COUNT, depth);
            assert!(rows <= last && rows.is_power_of_two(), "{depth}: {rows}");
            last = rows;
        }
    }

    #[test]
    fn a_smaller_card_halves_its_rows_earlier_and_a_larger_one_later() {
        let at = |sm, depth| allowed_rows(NEAR, FAR, sm, depth);
        assert_eq!(at(0, 100_000), at(ANCHOR_SM_COUNT, 100_000));
        for depth in [10_000, 70_000, 100_000, 200_000] {
            assert!(at(20, depth) <= at(ANCHOR_SM_COUNT, depth), "{depth}");
            assert!(at(170, depth) >= at(ANCHOR_SM_COUNT, depth), "{depth}");
        }
        // A fourth of the multiprocessors is a fourth of the rows, by the model's own scale.
        assert_eq!(at(41, 73_728), 512);
        assert!(at(20, 73_728) <= 256);
    }

    #[test]
    fn a_piece_is_the_rows_asked_for_up_to_what_its_end_depth_allows() {
        let piece = |p0, rows| piece_rows((NEAR, FAR), ANCHOR_SM_COUNT, p0, rows);
        assert_eq!(piece(0, 2048), 2048);
        assert_eq!(piece(72_704, 1024), 1024);
        assert_eq!(piece(72_705, 1024), 512);
        assert_eq!(piece(155_136, 512), 512);
        assert_eq!(piece(155_137, 512), 256);
        assert_eq!(piece(196_608, 16), 16);
        assert_eq!(piece(196_608, 128), 128);
    }
}
