//! The pure chunk walk. Each family supplies its step list and chooses how a one-row tail runs.

use crate::EngError;

/// One chunk at `p0`, or one row at `p0` through the family's feed path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefillStep {
    Chunk { p0: usize, rows: usize },
    Feed { p0: usize },
}

/// Checks the continuation position before the chunk changes device state.
pub(crate) fn check_chunk_start(p0: usize, committed: usize) -> Result<(), EngError> {
    if p0 != committed {
        return Err(EngError::BadRequest(format!(
            "prefill chunk: p0 {p0} is not the committed rows {committed}"
        )));
    }
    Ok(())
}

/// Drives `steps` over the prompt rows of one feed_rows call: each step gets its own rows, a feed
/// step exactly one. The list is the family's own; this walk only slices `tokens` at each step
/// and runs it. The list that ran is returned.
pub(crate) fn drive_steps(
    start: usize,
    tokens: &[u32],
    steps: Vec<PrefillStep>,
    mut run: impl FnMut(PrefillStep, &[u32]) -> Result<(), EngError>,
) -> Result<Vec<PrefillStep>, EngError> {
    for &step in &steps {
        let (at, rows) = match step {
            PrefillStep::Chunk { p0, rows } => (p0, rows),
            PrefillStep::Feed { p0 } => (p0, 1),
        };
        let offset = at.checked_sub(start).ok_or_else(|| {
            EngError::BadRequest(format!(
                "prefill chunk: step at {at} precedes the start row {start}"
            ))
        })?;
        let end = offset.checked_add(rows).ok_or_else(|| {
            EngError::BadRequest("prefill chunk: the row window overflows".to_owned())
        })?;
        let slice = tokens.get(offset..end).ok_or_else(|| {
            EngError::BadRequest(format!(
                "prefill chunk: the plan names rows {offset}..{end} of {} tokens",
                tokens.len()
            ))
        })?;
        run(step, slice)?;
    }
    Ok(steps)
}
