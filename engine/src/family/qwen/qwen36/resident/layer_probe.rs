//! A test probe: the host values of the hidden rows after each mixer and each feed-forward block,
//! for the one-row step and the window step, so a test can name the first layer where a window
//! row differs from its solo result. Off unless a test starts it; the hooks cost one thread-local
//! read.

use super::ops::{Ctx, values};
use crate::{
    EngError,
    tensor::{Dtype, ExecutionTensor},
};
use std::cell::RefCell;

/// One recorded block output: where it is and its values, row-major.
pub(super) struct Probed {
    pub(super) layer: usize,
    pub(super) after: &'static str,
    pub(super) rows: usize,
    pub(super) values: Vec<f32>,
}

thread_local! {
    static PROBE: RefCell<Option<Vec<Probed>>> = const { RefCell::new(None) };
    static LAYER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Names the layer the next sub-operation records belong to.
pub(super) fn enter(layer: usize) {
    LAYER.with(|at| at.set(layer));
}

/// Records `x` under the layer last entered.
pub(super) fn record_here(
    ctx: &mut Ctx<'_>,
    after: &'static str,
    x: &ExecutionTensor,
) -> Result<(), EngError> {
    record(ctx, LAYER.with(std::cell::Cell::get), after, x)
}

/// Starts recording on this thread.
pub(super) fn start() {
    PROBE.with(|probe| *probe.borrow_mut() = Some(Vec::new()));
}

/// Stops recording and returns what was recorded, in order.
pub(super) fn take() -> Vec<Probed> {
    PROBE.with(|probe| probe.borrow_mut().take().unwrap_or_default())
}

/// Records `x` (rank two, `[rows, width]`) as the output of `after` in `layer`, when recording.
pub(super) fn record(
    ctx: &mut Ctx<'_>,
    layer: usize,
    after: &'static str,
    x: &ExecutionTensor,
) -> Result<(), EngError> {
    if PROBE.with(|probe| probe.borrow().is_none()) {
        return Ok(());
    }
    let rows = x.shape().first().copied().unwrap_or(1) as usize;
    let wide = if x.dtype() == Dtype::F32 {
        x.share()
    } else {
        ctx.kernels.cast(x, Dtype::F32)?
    };
    let flat = values(ctx.kernels, &wide)?;
    PROBE.with(|probe| {
        if let Some(rows_so_far) = probe.borrow_mut().as_mut() {
            rows_so_far.push(Probed {
                layer,
                after,
                rows,
                values: flat,
            });
        }
    });
    Ok(())
}
