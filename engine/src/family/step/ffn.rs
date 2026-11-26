//! The gated feed-forward block of a chunk: `down(act(gate(x), up(x)))`.
//!
//! The gate and up projections read the same rows, the activation joins them, and the down
//! projection reads the result. The activation is the model's; the projections are the shared
//! ones, so the row count picks their kernels.

use crate::EngError;
use crate::family::step::rows::{Out, Rows, StepIo, q8};
use crate::kernels::cuda::prefill::swiglu_rows::{SwigluRowsCall, SwigluRowsShape};
use crate::kernels::cuda::resident::block_ops::{BlockOp, RowOutputs};
use crate::kernels::cuda::resident::quantize_q8_1::QuantizeCall;
use crate::kernels::cuda::{DeviceBuffer, NativeWeight};

/// How the gate and the up rows join.
#[derive(Clone, Copy)]
pub(crate) enum Join {
    /// `silu(gate) * up`. The gate and up rows share one buffer, the up values at their own
    /// column offset of each row.
    Swiglu,
    /// The SiTU activation with its two slopes.
    Situ { beta: f32, linear_beta: f32 },
}

#[derive(Clone, Copy)]
pub(crate) struct Weights<'w> {
    pub(crate) gate: &'w NativeWeight,
    pub(crate) up: &'w NativeWeight,
    pub(crate) down: &'w NativeWeight,
}

impl<'w> Weights<'w> {
    /// The three tensors of a block by name.
    pub(crate) fn named(
        io: &StepIo<'w>,
        gate: &str,
        up: &str,
        down: &str,
    ) -> Result<Self, EngError> {
        Ok(Self {
            gate: io.native(gate)?,
            up: io.native(up)?,
            down: io.native(down)?,
        })
    }
}

/// The scratch of one block: where the gate and up rows land, and the joined rows the down
/// projection reads (q8_1 always, f32 when the batched product may read them).
#[derive(Clone, Copy)]
pub(crate) struct Scratch<'a> {
    pub(crate) gate: Out<'a>,
    pub(crate) up: Out<'a>,
    pub(crate) act: Rows<'a>,
}

impl<'a> Scratch<'a> {
    /// The gate and up rows of `n` rows of `width` values back to back in their own buffers.
    pub(crate) fn packed(
        gate: &'a DeviceBuffer,
        up: &'a DeviceBuffer,
        (n, width): (usize, usize),
        act: Rows<'a>,
    ) -> Result<Self, EngError> {
        Ok(Self {
            gate: Out::packed(gate, n, width)?,
            up: Out::packed(up, n, width)?,
            act,
        })
    }
}

/// Runs the block over `n` rows of `x` into `out`, rows back to back.
pub(crate) fn gated(
    io: &StepIo<'_>,
    weights: Weights<'_>,
    join: Join,
    x: Rows<'_>,
    n: usize,
    scratch: Scratch<'_>,
    out: &DeviceBuffer,
) -> Result<(), EngError> {
    let width = weights.gate.n_rows();
    io.project_into(weights.gate, x, n, scratch.gate)?;
    io.project_into(weights.up, x, n, scratch.up)?;
    let reads_q8 = io.reads_q8(weights.down, scratch.act, n);
    joined(io, join, (n, width), scratch, reads_q8)?;
    io.project(weights.down, scratch.act, out, n)
}

/// Joins `n` gate and up rows of `width` values into the joined rows of `scratch`. `q8` says
/// whether a reader needs the q8_1 rows of a swiglu (the SiTU kernel always writes them).
pub(crate) fn joined(
    io: &StepIo<'_>,
    join: Join,
    (n, width): (usize, usize),
    scratch: Scratch<'_>,
    q8_rows: bool,
) -> Result<(), EngError> {
    let act = scratch.act;
    match join {
        Join::Swiglu => {
            let act_f32 = act.f32.ok_or_else(|| {
                EngError::BadRequest(format!("{}: the swiglu rows need f32 scratch", io.family))
            })?;
            let (gate, up) = (scratch.gate, scratch.up);
            if gate.ld != up.ld || gate.col_off != 0 || gate.dst.offset() != up.dst.offset() {
                return Err(EngError::BadRequest(format!(
                    "{}: the swiglu gate and up rows do not share one buffer",
                    io.family
                )));
            }
            io.ctx.swiglu_rows(
                io.stream,
                SwigluRowsCall {
                    shape: SwigluRowsShape {
                        rows: n,
                        n_ff: width,
                        up_off: up.col_off,
                        ld_in: gate.ld,
                        ld_out: width,
                    },
                    gu: gate.dst,
                    act: act_f32.range(0, n * width * 4)?,
                },
            )?;
            if q8_rows {
                io.ctx.quantize_q8_1(
                    io.stream,
                    QuantizeCall {
                        x: act_f32.range(0, n * width * 4)?,
                        y: act.q8.range(0, q8(width, n))?,
                        k: width,
                        ncols: n,
                    },
                )?;
            }
        }
        Join::Situ { beta, linear_beta } => {
            let span = |o: &Out<'_>| ((n - 1) * o.ld + width) * 4;
            let (gate, up) = (scratch.gate, scratch.up);
            io.run(BlockOp::situ_quant(
                n,
                width,
                gate.ld,
                up.ld,
                (beta, linear_beta),
                gate.dst
                    .buffer()
                    .range(gate.dst.offset() + gate.col_off * 4, span(&gate))?,
                up.dst
                    .buffer()
                    .range(up.dst.offset() + up.col_off * 4, span(&up))?,
                RowOutputs {
                    q8: Some(act.q8.range(0, q8(width, n))?),
                    f32: act.f32.map(|b| b.range(0, n * width * 4)).transpose()?,
                },
            )?)?;
        }
    }
    Ok(())
}
