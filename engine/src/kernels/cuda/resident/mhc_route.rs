//! Safe launch wrappers of the mHC stream routing kernels of the SenseNova Loop family:
//! [`CudaContext::mhc_route`], [`CudaContext::mhc_merge`] and [`CudaContext::mhc_average`]
//! each queue ONE kernel on a [`DeviceStream`] and return. The calls allocate nothing, copy
//! nothing and never synchronize; a fault in the queued work surfaces at the next synchronize
//! of the stream.
//!
//! The math is the CPU reference's (`family/qwen/sensenova/step.rs`), element for element;
//! the header `native/resident/mhc_route.h` states it in full.
//!
//! Every refusal is a `BadRequest` and happens before any native call, and the local rules
//! mirror `valid()` of `mhc_route.cu`, so the native entry cannot return `INVALID_ARGUMENT`
//! (which would latch the backend). In order: the backend is usable; `hidden` is a positive
//! multiple of 256 at most 2^20 (the grid math); `pre_scale` is finite and `iterations` is at
//! most 32 (route); `rows` is 1..=4096 and `ld_ab` at least two `hidden`; every range holds
//! the bytes of all its rows at 4-byte alignment; the written ranges
//! overlap no other range of the call. A refused call changes nothing.
//!
//! The caller of one visit sequences route -> (branch) -> merge -> ... -> average on ONE
//! stream, so every kernel reads what the previous one wrote; `a` and `b` must hold distinct
//! elements (merge writes both in place after one read of each, per thread): two ranges for
//! one row, or the halves of `ld_ab`-wide `[a || b]` rows for a chunk; and the 6-float route
//! cell written by route is the cell merge reads.
//!
//! One call serves one decode step (`rows: 1`) or one prefill chunk (`rows` rows): every range
//! then holds its rows back to back.

use crate::EngError;
use crate::kernels::cuda::context::{CudaContext, DeviceRange, DeviceStream};
use crate::kernels::cuda::native::resident::mhc_route::{
    MhcAverageRaw, MhcMergeRaw, MhcRouteRaw, mhc_average_launch, mhc_merge_launch, mhc_route_launch,
};

const FLOAT: usize = size_of::<f32>();
const UNIT: usize = 256;
const MAX_HIDDEN: usize = 1 << 20;
const LOGIT_FLOATS: usize = 8;
const ROUTE_FLOATS: usize = 6;
/// Rows of one chunk: the mmq chunk cap.
const MAX_ROWS: usize = 4096;

/// Checks the row count and the stream stride, and returns the bytes one stream half covers:
/// `rows` rows whose first is a full row and whose rest step `ld_ab`. A chunk's halves
/// interleave inside their `[a || b]` rows; `ld_ab >= 2 * hidden` keeps every element apart.
fn stream_bytes(rows: usize, ld_ab: usize, hidden: usize) -> Result<usize, EngError> {
    if rows == 0 || rows > MAX_ROWS || ld_ab < 2 * hidden {
        return Err(EngError::BadRequest(format!(
            "mhc_route: rows is {rows} and ld_ab is {ld_ab}, not 1..=4096 rows of ld_ab >= 2 * hidden"
        )));
    }
    let floats = rows
        .checked_sub(1)
        .and_then(|lead| lead.checked_mul(ld_ab))
        .and_then(|lead| lead.checked_add(hidden))
        .ok_or_else(|| EngError::BadRequest("mhc_route: the stream rows overflow".to_owned()))?;
    bytes("stream", floats)
}

fn check_hidden(hidden: usize) -> Result<(), EngError> {
    if hidden % UNIT != 0 || hidden > MAX_HIDDEN {
        return Err(EngError::BadRequest(format!(
            "mhc_route: hidden is {hidden}, not a multiple of 256 up to 2^20"
        )));
    }
    Ok(())
}

fn bytes(what: &str, floats: usize) -> Result<usize, EngError> {
    floats
        .checked_mul(FLOAT)
        .ok_or_else(|| EngError::BadRequest(format!("mhc_route: the {what} size overflows")))
}

fn fit(what: &str, range: &DeviceRange<'_>, need: usize) -> Result<(), EngError> {
    if range.bytes() < need || range.offset() % 4 != 0 {
        return Err(EngError::BadRequest(format!(
            "mhc_route: {what} holds {} bytes at offset {}, not {need} bytes at a 4-byte alignment",
            range.bytes(),
            range.offset()
        )));
    }
    Ok(())
}

/// One routing call. `logits` is the router matvec of the unit-RMS'd `[a || b]` row plus its
/// bias (8 f32); `a`, `b` and `input` are `[hidden]` f32; `route` is 6 f32 (post then mix)
/// written by the call and read by the matching merge.
pub(crate) struct MhcRouteCall<'b> {
    pub(crate) hidden: usize,
    /// Rows of the call: 1 for a decode step, a chunk's row count for prefill.
    pub(crate) rows: usize,
    /// The stride between the `[a || b]` rows of a chunk, at least two `hidden`.
    pub(crate) ld_ab: usize,
    pub(crate) pre_scale: f32,
    pub(crate) iterations: usize,
    pub(crate) fixed_identity: bool,
    pub(crate) logits: DeviceRange<'b>,
    /// The router bias (8 f32), added to the logits before every use.
    pub(crate) bias: Option<DeviceRange<'b>>,
    pub(crate) a: DeviceRange<'b>,
    pub(crate) b: DeviceRange<'b>,
    pub(crate) input: DeviceRange<'b>,
    pub(crate) route: DeviceRange<'b>,
}

/// One merge call: `a` and `b` in place with the branch output, using the route cell.
pub(crate) struct MhcMergeCall<'b> {
    pub(crate) hidden: usize,
    pub(crate) rows: usize,
    pub(crate) ld_ab: usize,
    pub(crate) a: DeviceRange<'b>,
    pub(crate) b: DeviceRange<'b>,
    pub(crate) branch: DeviceRange<'b>,
    pub(crate) route: DeviceRange<'b>,
}

/// One average call: `out[j] = 0.5 * (a[j] + b[j])`.
pub(crate) struct MhcAverageCall<'b> {
    pub(crate) hidden: usize,
    pub(crate) rows: usize,
    pub(crate) ld_ab: usize,
    pub(crate) a: DeviceRange<'b>,
    pub(crate) b: DeviceRange<'b>,
    pub(crate) out: DeviceRange<'b>,
}

impl CudaContext {
    /// Queues the routing kernel of `call` on `stream`.
    pub(crate) fn mhc_route(
        &self,
        stream: &DeviceStream,
        call: MhcRouteCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_hidden(call.hidden)?;
        if !call.pre_scale.is_finite() || call.iterations > 32 {
            return Err(EngError::BadRequest(
                "mhc_route: pre_scale is not finite or iterations is above 32".to_owned(),
            ));
        }
        let half = stream_bytes(call.rows, call.ld_ab, call.hidden)?;
        let logit_row = bytes("logits", LOGIT_FLOATS)?;
        let logit_bytes = bytes("logits", LOGIT_FLOATS * call.rows)?;
        let route_bytes = bytes("route cells", ROUTE_FLOATS * call.rows)?;
        let input = bytes("input", call.hidden * call.rows)?;
        fit("logits", &call.logits, logit_bytes)?;
        if let Some(bias) = &call.bias {
            fit("bias", bias, logit_row)?;
        }
        fit("a", &call.a, half)?;
        fit("b", &call.b, half)?;
        fit("input", &call.input, input)?;
        fit("route", &call.route, route_bytes)?;
        let logits = self.raw_range(&call.logits)?;
        let bias = call.bias.as_ref().map(|r| self.raw_range(r)).transpose()?;
        let a = self.raw_range(&call.a)?;
        let b = self.raw_range(&call.b)?;
        let input = self.raw_range(&call.input)?;
        let route = self.raw_range(&call.route)?;
        self.launch_on(stream, move |raw| {
            mhc_route_launch(
                raw,
                MhcRouteRaw {
                    logits,
                    bias,
                    a,
                    b,
                    input,
                    route,
                    hidden: call.hidden,
                    pre_scale: call.pre_scale,
                    iterations: call.iterations,
                    fixed_identity: call.fixed_identity,
                    rows: call.rows,
                    ld_ab: call.ld_ab,
                },
            )
        })
    }

    /// Queues the merge kernel of `call` on `stream`.
    pub(crate) fn mhc_merge(
        &self,
        stream: &DeviceStream,
        call: MhcMergeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_hidden(call.hidden)?;
        let half = stream_bytes(call.rows, call.ld_ab, call.hidden)?;
        let branch = bytes("branch", call.hidden * call.rows)?;
        let route_bytes = bytes("route cells", ROUTE_FLOATS * call.rows)?;
        fit("a", &call.a, half)?;
        fit("b", &call.b, half)?;
        fit("branch", &call.branch, branch)?;
        fit("route", &call.route, route_bytes)?;
        let a = self.raw_range(&call.a)?;
        let b = self.raw_range(&call.b)?;
        let branch = self.raw_range(&call.branch)?;
        let route = self.raw_range(&call.route)?;
        self.launch_on(stream, move |raw| {
            mhc_merge_launch(
                raw,
                MhcMergeRaw {
                    a,
                    b,
                    branch,
                    route,
                    hidden: call.hidden,
                    rows: call.rows,
                    ld_ab: call.ld_ab,
                },
            )
        })
    }

    /// Queues the average kernel of `call` on `stream`.
    pub(crate) fn mhc_average(
        &self,
        stream: &DeviceStream,
        call: MhcAverageCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        check_hidden(call.hidden)?;
        let half = stream_bytes(call.rows, call.ld_ab, call.hidden)?;
        let out = bytes("out", call.hidden * call.rows)?;
        fit("a", &call.a, half)?;
        fit("b", &call.b, half)?;
        fit("out", &call.out, out)?;
        let a = self.raw_range(&call.a)?;
        let b = self.raw_range(&call.b)?;
        let out = self.raw_range(&call.out)?;
        self.launch_on(stream, move |raw| {
            mhc_average_launch(
                raw,
                MhcAverageRaw {
                    a,
                    b,
                    out,
                    hidden: call.hidden,
                    rows: call.rows,
                    ld_ab: call.ld_ab,
                },
            )
        })
    }
}
