//! The one chunk loop: how a prompt becomes pieces, and what one piece runs, for every family
//! that prefills in chunks and for the resident and the tiered arm alike.
//!
//! WHAT. [`plan_pieces`] turns the rows of a feed into the list of [`PrefillStep`]s. A piece has
//! the most rows the memory plan admits (`max_rows`, the family's arena), cut by the piece-time
//! model ([`piece_rows`]) at the depth it ends at. A one-row piece is a feed step. [`drive_prompt`] walks the list through
//! [`drive_steps`]. [`drive_chunk`] is what a chunk runs: open the window, stage the inputs, clear
//! the status word, queue the layers, read the status, commit; the family supplies the surface
//! ([`ChunkFamily`]) and three hooks for the parts only it has (the head of a last row, a report).
//!
//! WHY one loop. The resident and the tiered arm of Qwen3.6 and the chunked prefill of Flash-Next
//! each carried a list builder and the same six calls in their own order. A streamed model wants
//! the widest pieces the arena holds (each piece copies every streamed layer once), a resident
//! one the same; the arena is the one input that differs.

use super::{
    piece_rows::{Anchor, piece_rows},
    steps::{PrefillStep, drive_steps},
};
use crate::EngError;

/// What bounds the pieces of a prompt: the arena's rows and the piece-time model.
pub(crate) struct PiecePlan {
    /// The most rows of one piece: the arena the memory plan admitted.
    pub(crate) max_rows: usize,
    /// The model's two anchors and the device's multiprocessor count (0 for the card the
    /// anchors were measured on).
    pub(crate) model: ((Anchor, Anchor), usize),
}

/// The pieces that feed `rows` rows from position `start`. A piece never has zero rows and a
/// one-row piece is a feed step.
pub(crate) fn plan_pieces(
    start: usize,
    rows: usize,
    plan: &PiecePlan,
) -> Result<Vec<PrefillStep>, EngError> {
    if plan.max_rows == 0 {
        return Err(EngError::BadRequest(
            "prefill pieces: the arena holds no rows".to_owned(),
        ));
    }
    let end = start.checked_add(rows).ok_or_else(|| {
        EngError::BadRequest("prefill pieces: the row window overflows".to_owned())
    })?;
    let mut steps = Vec::new();
    let mut at = start;
    while at < end {
        let (anchors, sm_count) = plan.model;
        let take = piece_rows(anchors, sm_count, at, plan.max_rows.min(end - at));
        steps.push(match take {
            1 => PrefillStep::Feed { p0: at },
            _ => PrefillStep::Chunk { p0: at, rows: take },
        });
        at += take;
    }
    Ok(steps)
}

/// Plans the pieces of `tokens` (which start at position `start`) and runs each through `run`.
/// The list that ran is returned.
pub(crate) fn drive_prompt(
    start: usize,
    tokens: &[u32],
    plan: &PiecePlan,
    run: impl FnMut(PrefillStep, &[u32]) -> Result<(), EngError>,
) -> Result<Vec<PrefillStep>, EngError> {
    let steps = plan_pieces(start, tokens.len(), plan)?;
    drive_steps(start, tokens, steps, run)
}

/// Whether a family clears its status word before or after it stages the inputs of a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StatusClear {
    BeforeInputs,
    AfterInputs,
}

/// The order a family's chunk runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ChunkOrder {
    /// Open the family's window before this chunk.
    pub(crate) open_window: bool,
    pub(crate) clear_status: StatusClear,
}

/// Runs one chunk of `m` rows at `p0` through the family's surface and returns the status word it
/// committed. `before_inputs` runs after the window opens and before the inputs are staged,
/// `after_layers` after the layers are queued and before the status is read, and `after_health`
/// receives the status once it is read and before the commit.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn drive_chunk<F: super::chunk_family::ChunkFamily>(
    family: &mut F,
    ctx: &mut F::Ctx,
    (p0, m): (usize, usize),
    order: ChunkOrder,
    before_inputs: impl FnOnce(&mut F, &mut F::Ctx) -> Result<(), EngError>,
    after_layers: impl FnOnce(&mut F, &mut F::Ctx) -> Result<(), EngError>,
    after_health: impl FnOnce(i32) -> Result<(), EngError>,
) -> Result<i32, EngError> {
    if order.open_window {
        family.begin_window(ctx)?;
    }
    before_inputs(family, ctx)?;
    if order.clear_status == StatusClear::BeforeInputs {
        family.clear_chunk_status(ctx)?;
    }
    family.write_chunk_inputs(ctx, p0, m)?;
    if order.clear_status == StatusClear::AfterInputs {
        family.clear_chunk_status(ctx)?;
    }
    family.run_layers(ctx, p0, m)?;
    after_layers(family, ctx)?;
    let status = family.read_health(ctx)?;
    after_health(status)?;
    family.commit_chunk(ctx, p0, m, status)?;
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::family::prefill::piece_rows::MEASURED_ANCHORS;

    fn plan(max_rows: usize) -> PiecePlan {
        PiecePlan {
            max_rows,
            model: (MEASURED_ANCHORS, 0),
        }
    }

    fn covers(steps: &[PrefillStep], start: usize, rows: usize) {
        let mut at = start;
        for step in steps {
            let (p0, take) = match *step {
                PrefillStep::Chunk { p0, rows } => (p0, rows),
                PrefillStep::Feed { p0 } => (p0, 1),
            };
            assert_eq!(p0, at, "{steps:?}");
            assert!(take >= 1);
            at += take;
        }
        assert_eq!(at, start + rows, "{steps:?}");
    }

    #[test]
    fn pieces_cover_the_rows_once_in_order_and_never_exceed_the_arena() {
        for (start, rows, max) in [
            (0, 1, 256),
            (0, 4096, 4096),
            (7, 5000, 1024),
            (300, 257, 128),
        ] {
            let steps = plan_pieces(start, rows, &plan(max)).unwrap();
            covers(&steps, start, rows);
            for step in &steps {
                if let PrefillStep::Chunk { rows, .. } = step {
                    assert!((2..=max).contains(rows), "{steps:?}");
                }
            }
        }
        assert!(plan_pieces(0, 5, &plan(0)).is_err());
        assert!(plan_pieces(usize::MAX, 5, &plan(8)).is_err());
        assert!(plan_pieces(10, 0, &plan(8)).unwrap().is_empty());
    }

    #[test]
    fn a_one_row_tail_is_a_feed_step() {
        let steps = plan_pieces(0, 129, &plan(128)).unwrap();
        assert_eq!(
            steps,
            vec![
                PrefillStep::Chunk { p0: 0, rows: 128 },
                PrefillStep::Feed { p0: 128 }
            ]
        );
    }

    #[test]
    fn the_piece_model_narrows_deep_pieces_and_leaves_shallow_ones_at_the_arena() {
        let plan = plan(4096);
        let shallow = plan_pieces(0, 8192, &plan).unwrap();
        assert_eq!(shallow[0], PrefillStep::Chunk { p0: 0, rows: 4096 });
        let deep = plan_pieces(155_648, 4096, &plan).unwrap();
        covers(&deep, 155_648, 4096);
        assert!(deep.len() > 1, "{deep:?}");
        for step in &deep {
            if let PrefillStep::Chunk { rows, .. } = step {
                assert!(*rows <= 256, "{step:?}");
            }
        }
    }

    #[test]
    fn a_wider_arena_means_fewer_pieces_and_so_fewer_crossings_of_a_streamed_layer() {
        let narrow = plan_pieces(0, 32_768, &plan(256)).unwrap().len();
        let wide = plan_pieces(0, 32_768, &plan(4096)).unwrap().len();
        assert_eq!(narrow, 128);
        // The model narrows a 4,096-row arena to 2,048 rows from about 10K of depth.
        assert_eq!(wide, 13);
    }

    #[test]
    fn drive_prompt_hands_each_piece_its_own_rows() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut seen = Vec::new();
        let ran = drive_prompt(100, &tokens, &plan(4), |step, rows| {
            seen.push((step, rows.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(ran.len(), seen.len());
        assert_eq!(seen[0].1, vec![0, 1, 2, 3]);
        assert_eq!(
            seen[2],
            (PrefillStep::Chunk { p0: 108, rows: 2 }, vec![8, 9])
        );
    }

    struct Log(Vec<&'static str>);

    impl super::super::chunk_family::ChunkFamily for Log {
        type Ctx = ();
        fn begin_window(&mut self, _: &mut ()) -> Result<(), EngError> {
            self.0.push("window");
            Ok(())
        }
        fn clear_chunk_status(&mut self, _: &mut ()) -> Result<(), EngError> {
            self.0.push("clear");
            Ok(())
        }
        fn write_chunk_inputs(&mut self, _: &mut (), _: usize, _: usize) -> Result<(), EngError> {
            self.0.push("inputs");
            Ok(())
        }
        fn run_layers(&mut self, _: &mut (), _: usize, _: usize) -> Result<(), EngError> {
            self.0.push("layers");
            Ok(())
        }
        fn read_health(&mut self, _: &mut ()) -> Result<i32, EngError> {
            self.0.push("health");
            Ok(0)
        }
        fn commit_chunk(&mut self, _: &mut (), _: usize, _: usize, _: i32) -> Result<(), EngError> {
            self.0.push("commit");
            Ok(())
        }
    }

    #[test]
    fn a_chunk_runs_its_calls_in_the_family_s_order() {
        let mut log = Log(Vec::new());
        let order = ChunkOrder {
            open_window: true,
            clear_status: StatusClear::BeforeInputs,
        };
        drive_chunk(
            &mut log,
            &mut (),
            (0, 8),
            order,
            |f, _| {
                f.0.push("before");
                Ok(())
            },
            |f, _| {
                f.0.push("after-layers");
                Ok(())
            },
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(
            log.0,
            [
                "window",
                "before",
                "clear",
                "inputs",
                "layers",
                "after-layers",
                "health",
                "commit"
            ]
        );
        let mut log = Log(Vec::new());
        let order = ChunkOrder {
            open_window: false,
            clear_status: StatusClear::AfterInputs,
        };
        drive_chunk(
            &mut log,
            &mut (),
            (8, 8),
            order,
            |_, _| Ok(()),
            |_, _| Ok(()),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(log.0, ["inputs", "clear", "layers", "health", "commit"]);
    }

    #[test]
    fn a_failing_step_stops_the_chunk_before_the_commit() {
        let mut log = Log(Vec::new());
        let order = ChunkOrder {
            open_window: true,
            clear_status: StatusClear::BeforeInputs,
        };
        let result = drive_chunk(
            &mut log,
            &mut (),
            (0, 2),
            order,
            |_, _| Ok(()),
            |_, _| Err(EngError::BadRequest("stop".into())),
            |_| Ok(()),
        );
        assert!(result.is_err());
        assert!(!log.0.contains(&"commit") && !log.0.contains(&"health"));
    }
}
