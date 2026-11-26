//! The residual sites of a visit: how a branch reads its input rows and how its output returns
//! to the residual.
//!
//! A plain site adds the branch to the residual rows ([`add`]). A routed site keeps two streams
//! per row, `[a || b]`: a router reads both and writes the branch input and a route cell
//! ([`Routed::route`]), the branch output merges back into both streams by that cell
//! ([`Routed::merge`]), and the residual row is their mean ([`Routed::mean`]). A mixed site keeps
//! four streams per row as the residual itself: the read mixes them into the branch input
//! ([`Mixed::read`]) and the apply mixes the streams and adds the branch output to each
//! ([`Mixed::apply`]). Every call takes the row count of the chunk; one row is a decode step.

use super::rows::{Norm, StepIo, q8};
use crate::EngError;
use crate::kernels::cuda::resident::block_ops::BlockOp;
use crate::kernels::cuda::resident::hyper::{HyperMixApplyCall, HyperShape};
use crate::kernels::cuda::resident::mhc_route::{MhcAverageCall, MhcMergeCall, MhcRouteCall};
use crate::kernels::cuda::resident::pack_rows::{PackRowsCall, PackSegment};
use crate::kernels::cuda::resident::router::{RouterLogitsCall, RouterShape};
use crate::kernels::cuda::resident::row_add::RowAddCall;
use crate::kernels::cuda::{DeviceBuffer, DeviceRange};

const F32: usize = size_of::<f32>();

/// Rows of one `pack_rows` launch.
const PACK_ROWS: usize = 16;

/// Logits of one router row.
const ROUTER_LOGITS: usize = 8;

/// Floats of one route cell.
const ROUTE_CELL: usize = 6;

/// Streams of one mixed-site row.
const MIXED: usize = 4;

fn floats(buffer: &DeviceBuffer, at: usize, len: usize) -> Result<DeviceRange<'_>, EngError> {
    buffer.range(at * F32, len * F32)
}

/// `out = x + y` over `len` floats: the plain residual add. `out` is not `x`.
pub(crate) fn add(
    io: &StepIo<'_>,
    out: DeviceRange<'_>,
    x: DeviceRange<'_>,
    y: DeviceRange<'_>,
    len: usize,
) -> Result<(), EngError> {
    io.ctx.row_add(
        io.stream,
        RowAddCall {
            out,
            x,
            y,
            n: len,
            flag: None,
            fused: false,
        },
    )
}

/// The two-stream routed site of a model and the buffers it works in.
pub(crate) struct Routed<'a> {
    pub(crate) hidden: usize,
    pub(crate) eps: f32,
    pub(crate) pre_scale: f32,
    pub(crate) iterations: usize,
    pub(crate) fixed_identity: bool,
    /// Unit norm weights, two `hidden` wide.
    pub(crate) ones: &'a DeviceBuffer,
    /// The `[a || b]` rows.
    pub(crate) streams: &'a DeviceBuffer,
    /// Scratch: the unit-RMS `[a || b]` rows and their q8_1 rows.
    pub(crate) normed: &'a DeviceBuffer,
    pub(crate) normed_q8: &'a DeviceBuffer,
    pub(crate) logits: &'a DeviceBuffer,
    /// The branch input rows a route writes. It may be `normed`: the router has read those
    /// rows by the time the input lands.
    pub(crate) input: &'a DeviceBuffer,
    pub(crate) cell: &'a DeviceBuffer,
}

impl Routed<'_> {
    /// The a and b halves of `n` stream rows.
    fn halves(&self, n: usize) -> Result<(DeviceRange<'_>, DeviceRange<'_>), EngError> {
        let h = self.hidden;
        let tail = (n - 1) * 2 * h + h;
        Ok((
            floats(self.streams, 0, tail)?,
            floats(self.streams, h, tail)?,
        ))
    }

    /// Both streams of `n` rows start as the residual rows `cur`.
    pub(crate) fn fork(
        &self,
        io: &StepIo<'_>,
        cur: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        let h = self.hidden;
        for at in (0..n).step_by(PACK_ROWS) {
            let rows = (n - at).min(PACK_ROWS);
            let src = floats(cur, at * h, rows * h)?;
            let segment = |dst_off| PackSegment { src, n: h, dst_off };
            io.ctx.pack_rows(
                io.stream,
                PackRowsCall {
                    m: rows,
                    ld: 2 * h,
                    dst: floats(self.streams, at * 2 * h, rows * 2 * h)?,
                    segments: vec![segment(0), segment(h)],
                },
            )?;
        }
        Ok(())
    }

    /// One router pass over `n` rows: the unit-RMS norm of every `[a || b]` row, the router
    /// rows of the raw f32 matrix `weight`, and the routing kernel (`bias` folds into the
    /// logits there). The branch input lands in `input`.
    pub(crate) fn route(
        &self,
        io: &StepIo<'_>,
        weight: &DeviceBuffer,
        bias: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        let h = self.hidden;
        io.norm(
            n,
            Norm {
                d: 2 * h,
                eps: self.eps,
                x: floats(self.streams, 0, n * 2 * h)?,
                delta: None,
                weight: floats(self.ones, 0, 2 * h)?,
            },
            floats(self.normed, 0, n * 2 * h)?,
            self.normed_q8.range(0, q8(2 * h, n))?,
        )?;
        io.ctx.router_logits_rows(
            io.stream,
            RouterLogitsCall {
                shape: RouterShape {
                    m: n,
                    hidden: 2 * h,
                    n_expert: ROUTER_LOGITS,
                    top_k: 1,
                },
                x: floats(self.normed, 0, n * 2 * h)?,
                w: floats(weight, 0, ROUTER_LOGITS * 2 * h)?,
                logits: floats(self.logits, 0, n * ROUTER_LOGITS)?,
            },
        )?;
        let (a, b) = self.halves(n)?;
        io.ctx.mhc_route(
            io.stream,
            MhcRouteCall {
                hidden: h,
                rows: n,
                ld_ab: 2 * h,
                pre_scale: self.pre_scale,
                iterations: self.iterations,
                fixed_identity: self.fixed_identity,
                logits: floats(self.logits, 0, n * ROUTER_LOGITS)?,
                bias: Some(floats(bias, 0, ROUTER_LOGITS)?),
                a,
                b,
                input: floats(self.input, 0, n * h)?,
                route: floats(self.cell, 0, n * ROUTE_CELL)?,
            },
        )
    }

    /// The `n` branch output rows of `branch` merge into both streams by the last route.
    pub(crate) fn merge(
        &self,
        io: &StepIo<'_>,
        branch: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        let (a, b) = self.halves(n)?;
        io.ctx.mhc_merge(
            io.stream,
            MhcMergeCall {
                hidden: self.hidden,
                rows: n,
                ld_ab: 2 * self.hidden,
                a,
                b,
                branch: floats(branch, 0, n * self.hidden)?,
                route: floats(self.cell, 0, n * ROUTE_CELL)?,
            },
        )
    }

    /// The residual rows `out` become the mean of both streams.
    pub(crate) fn mean(
        &self,
        io: &StepIo<'_>,
        out: DeviceRange<'_>,
        n: usize,
    ) -> Result<(), EngError> {
        let (a, b) = self.halves(n)?;
        io.ctx.mhc_average(
            io.stream,
            MhcAverageCall {
                hidden: self.hidden,
                rows: n,
                ld_ab: 2 * self.hidden,
                a,
                b,
                out,
            },
        )
    }
}

/// The mixed site of a model: its constants and the rows a read leaves for the branch and for
/// the matching apply.
pub(crate) struct Mixed<'a> {
    pub(crate) hidden: usize,
    pub(crate) eps: f32,
    pub(crate) magnitude: f32,
    /// The branch input rows a read writes.
    pub(crate) input: &'a DeviceBuffer,
    /// The post weights and the stream-mixing matrix of every row.
    pub(crate) post: &'a DeviceBuffer,
    pub(crate) comb: &'a DeviceBuffer,
}

/// The raw tensors of one mixed site.
pub(crate) struct MixWeights<'a> {
    pub(crate) mix: DeviceRange<'a>,
    pub(crate) base: DeviceRange<'a>,
    pub(crate) scale: DeviceRange<'a>,
}

impl Mixed<'_> {
    /// The read of `n` rows of the four-stream residual `resid`.
    pub(crate) fn read(
        &self,
        io: &StepIo<'_>,
        weights: MixWeights<'_>,
        resid: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        io.run(BlockOp::hc_read(
            n,
            self.hidden,
            self.eps,
            self.magnitude,
            floats(resid, 0, n * MIXED * self.hidden)?,
            weights.mix,
            weights.base,
            weights.scale,
            floats(self.input, 0, n * self.hidden)?,
            floats(self.post, 0, n * MIXED)?,
            floats(self.comb, 0, n * MIXED * MIXED)?,
        )?)
    }

    /// The stream update after the branch: `out = comb * resid + post * branch`.
    pub(crate) fn apply(
        &self,
        io: &StepIo<'_>,
        resid: &DeviceBuffer,
        out: &DeviceBuffer,
        branch: &DeviceBuffer,
        n: usize,
    ) -> Result<(), EngError> {
        io.ctx.hyper_mix_apply(
            io.stream,
            HyperMixApplyCall {
                shape: HyperShape {
                    m: n,
                    branches: MIXED,
                    hidden: self.hidden,
                    rank: 1,
                },
                x: floats(resid, 0, n * MIXED * self.hidden)?,
                out: floats(out, 0, n * MIXED * self.hidden)?,
                mix: floats(self.comb, 0, n * MIXED * MIXED)?,
                gate: floats(self.post, 0, n * MIXED)?,
                block: floats(branch, 0, n * self.hidden)?,
            },
        )
    }
}
