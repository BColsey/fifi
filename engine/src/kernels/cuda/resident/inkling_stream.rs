//! The resident Inkling depth stream, launch-only: the weighted mix over the sealed blocks,
//! the seal of the current row and the add into it.
//!
//! A stream state packs the current row and then `blocks` sealed rows, `width` floats each
//! (`native/resident/inkling_stream.h`). `health` is one float per launch: left alone, or quiet
//! NaN when an input or an output of that launch was not finite.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the resident decode step")
)]

#[cfg(test)]
mod tests;

use std::mem::{align_of, size_of};

use crate::EngError;
use crate::kernels::cuda::{
    CudaContext, DeviceRange, DeviceStream,
    context::{Claim, require_apart, require_fit},
};

pub(crate) use super::super::native::resident::inkling_stream::MAX_BLOCKS;
use super::super::native::resident::inkling_stream::{self as native};

const MAX_WIDTH: usize = 8192;

/// The ranges of a mix launch. `state` holds the current row and then `shape.blocks` sealed rows.
pub(crate) struct MixCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) query: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The sealed rows `first_block..blocks` join the mix, newest first; the current row comes last.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MixShape {
    pub(crate) first_block: usize,
    pub(crate) blocks: usize,
    pub(crate) width: usize,
    pub(crate) eps: f32,
}

/// The shape of a seal or add launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StreamShape {
    pub(crate) blocks: usize,
    pub(crate) width: usize,
    pub(crate) eps: f32,
}

/// The ranges of a seal launch. `state` must hold the new sealed row: `blocks + 2` rows.
pub(crate) struct SealCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) next: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

/// The ranges of an add launch.
pub(crate) struct AddCall<'a> {
    pub(crate) state: DeviceRange<'a>,
    pub(crate) row: DeviceRange<'a>,
    pub(crate) health: DeviceRange<'a>,
}

fn check_shape(
    what: &str,
    first_block: usize,
    blocks: usize,
    width: usize,
    eps: f32,
) -> Result<(), EngError> {
    if first_block <= blocks
        && blocks <= MAX_BLOCKS
        && (1..=MAX_WIDTH).contains(&width)
        && eps.is_finite()
        && eps > 0.0
    {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA Inkling stream {what} takes first_block <= blocks <= {MAX_BLOCKS}, 1 to \
         {MAX_WIDTH} columns and a finite positive eps, got first_block {first_block}, blocks \
         {blocks}, width {width}, eps {eps}"
    )))
}

fn check_row(name: &'static str, range: &DeviceRange<'_>, bytes: usize) -> Result<(), EngError> {
    require_fit(name, range, bytes, align_of::<f32>())?;
    if range.bytes() == bytes {
        return Ok(());
    }
    Err(EngError::BadRequest(format!(
        "the CUDA {name} range holds {} bytes but the call needs exactly {bytes}",
        range.bytes()
    )))
}

/// `out` and `health` share no byte with each other or with a range in `reads`.
fn writes_apart<'a>(
    writes: [Claim<'a>; 2],
    reads: impl IntoIterator<Item = Claim<'a>>,
) -> Result<(), EngError> {
    let [out, health] = writes;
    require_apart(&out, &health)?;
    for read in reads {
        require_apart(&out, &read)?;
        require_apart(&health, &read)?;
    }
    Ok(())
}

impl CudaContext {
    /// `Stream::mix(first_block, query)` of the host: the softmax-weighted mean of the sealed
    /// rows `first_block..blocks` (newest first) and the normalized current row, each scored by
    /// `dot(row, query) / sqrt(width)`. Agrees with the host within floating-point error.
    pub(crate) fn inkling_stream_mix(
        &self,
        stream: &DeviceStream,
        call: MixCall<'_>,
        shape: MixShape,
    ) -> Result<(), EngError> {
        check_shape(
            "mix",
            shape.first_block,
            shape.blocks,
            shape.width,
            shape.eps,
        )?;
        let row = shape.width * size_of::<f32>();
        let state = (shape.blocks + 1) * row;
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream query", &call.query, row)?;
        check_row("Inkling stream mix output", &call.out, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream mix output", &call.out, row),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream query", &call.query, row),
            ],
        )?;
        self.ensure_usable()?;
        let (state, query, out, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.query)?,
            self.raw_range(&call.out)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            native::mix(
                raw,
                state,
                query,
                out,
                health,
                [shape.first_block, shape.blocks, shape.width],
                shape.eps,
            )
        })
    }

    /// `Stream::seal(next)` of the host: the current row is RMS-normalized into the new sealed
    /// row, then replaced by `next`. The sealed row agrees with the host within rounding.
    pub(crate) fn inkling_stream_seal(
        &self,
        stream: &DeviceStream,
        call: SealCall<'_>,
        shape: StreamShape,
    ) -> Result<(), EngError> {
        check_shape("seal", 0, shape.blocks, shape.width, shape.eps)?;
        let row = shape.width * size_of::<f32>();
        let state = (shape.blocks + 2) * row;
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, state)?;
        check_row("Inkling stream next row", &call.next, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, state),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream next row", &call.next, row)],
        )?;
        self.ensure_usable()?;
        let (state, next, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.next)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| {
            native::seal(
                raw,
                state,
                next,
                health,
                [shape.blocks, shape.width],
                shape.eps,
            )
        })
    }

    /// `Stream::add_current(row)` of the host: `current = current + row`, bit-identical.
    pub(crate) fn inkling_stream_add(
        &self,
        stream: &DeviceStream,
        call: AddCall<'_>,
        width: usize,
    ) -> Result<(), EngError> {
        check_shape("add", 0, 0, width, 1.0)?;
        let row = width * size_of::<f32>();
        let health = size_of::<f32>();
        check_row("Inkling stream state", &call.state, row)?;
        check_row("Inkling stream addend row", &call.row, row)?;
        check_row("Inkling stream health", &call.health, health)?;
        writes_apart(
            [
                Claim::of("Inkling stream state", &call.state, row),
                Claim::of("Inkling stream health", &call.health, health),
            ],
            [Claim::of("Inkling stream addend row", &call.row, row)],
        )?;
        self.ensure_usable()?;
        let (state, row, health) = (
            self.raw_range(&call.state)?,
            self.raw_range(&call.row)?,
            self.raw_range(&call.health)?,
        );
        self.launch_on(stream, |raw| native::add(raw, state, row, health, width))
    }
}
