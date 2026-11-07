//! Router launches of a K-row verify window: the 4-row chunking that keeps the decode kernel, and
//! the logits launch with a row stride.
//!
//! The router logits kernel is one warp per dot product for `m < 8` and a tiled kernel (a warp
//! tile of 4 rows of the matrix by 8 tokens) for `m >= 8`; the two give the same logit bits on the
//! real routers (`router.cu`, with one documented sign-of-zero exception at `hidden / 4` not a
//! multiple of 512), but the K-row design does not rely on that: a window of 8 rows runs two 4-row
//! launches, so the decode kernel runs for every row at every `K`. [`CudaContext::router_rows`] is
//! [`CudaContext::router`] with that chunking; [`CudaContext::router_logits_rows_strided`] is the
//! plain F32 projection (the GDN beta and alpha gates) writing row `t`'s values at
//! `logits[t * stride]` of a wider row, so beta and alpha land in the `[K][ld]` projection row
//! with no copy, and chunks the same way.
//!
//! A window of at most 7 rows is one launch; 8 rows (`router.cu`, `kTiledMinRows`) are two
//! launches of 4. The measured costs of the two forms are in the K-row design (section 3.1).
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the K-row verify body")
)]

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::router_rows::{
    STRIDED_MAX_ROWS, StridedLogitsRaw, router_logits_strided_launch,
};
use crate::kernels::cuda::resident::router::{RouterCall, RouterOut, RouterShape, RouterWeights};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// Rows of one launch when a window is chunked.
pub(crate) const CHUNK_ROWS: usize = 4;
/// The first row count that would run the tiled kernel in one launch.
pub(crate) const TILED_MIN_ROWS: usize = 8;

/// The launches of `m` rows as `(first row, rows)`: one launch below the tiled threshold, else
/// chunks of [`CHUNK_ROWS`] (the last one possibly shorter).
pub(crate) fn router_chunks(m: usize) -> Vec<(usize, usize)> {
    if m < TILED_MIN_ROWS {
        return vec![(0, m)];
    }
    (0..m)
        .step_by(CHUNK_ROWS)
        .map(|first| (first, CHUNK_ROWS.min(m - first)))
        .collect()
}

/// `rows` rows' worth of `range`, which holds rows of `row_bytes` each, from row `first`.
fn rows_of<'b>(
    range: &DeviceRange<'b>,
    row_bytes: usize,
    (first, rows): (usize, usize),
) -> Result<DeviceRange<'b>, EngError> {
    let overflow = || EngError::BadRequest("the CUDA router chunk offsets overflow".to_owned());
    let offset = first.checked_mul(row_bytes).ok_or_else(overflow)?;
    let bytes = rows.checked_mul(row_bytes).ok_or_else(overflow)?;
    let end = offset.checked_add(bytes).ok_or_else(overflow)?;
    if end > range.bytes() {
        return Err(EngError::BadRequest(format!(
            "the CUDA router range of {} bytes does not hold rows {first}..{}",
            range.bytes(),
            first + rows
        )));
    }
    range.buffer().range(
        range.offset().checked_add(offset).ok_or_else(overflow)?,
        bytes,
    )
}

/// The plain F32 projection of a K-row window: `shape.m` rows of `x` against the `n_expert` rows
/// of `w`, row `t`'s `n_expert` values stored at float `t * logit_stride` of `logits`.
pub(crate) struct RouterLogitsStridedCall<'a> {
    /// `top_k` is not used (only the logits kernel runs); give 1.
    pub(crate) shape: RouterShape,
    /// f32 `[m][hidden]`, 16-byte aligned.
    pub(crate) x: DeviceRange<'a>,
    /// f32 `[n_expert][hidden]`, 16-byte aligned.
    pub(crate) w: DeviceRange<'a>,
    /// `(m - 1) * logit_stride + n_expert` floats, 4-byte aligned. Only the `n_expert` values of
    /// each row are written.
    pub(crate) logits: DeviceRange<'a>,
    /// Floats between two output rows, `>= n_expert`.
    pub(crate) logit_stride: usize,
}

impl CudaContext {
    /// [`CudaContext::router`] for a window of `call.shape.m` rows with the decode kernel at every
    /// row count: below 8 rows one launch (exactly `router`), from 8 rows chunks of 4. Row `t`
    /// of every output is the row `t` of the single launch of `router` for the same row, bit for
    /// bit.
    pub(crate) fn router_rows(
        &self,
        stream: &DeviceStream,
        call: RouterCall<'_>,
    ) -> Result<(), EngError> {
        let RouterShape {
            m,
            hidden,
            n_expert,
            top_k,
        } = call.shape;
        let chunks = router_chunks(m);
        if let [only] = chunks.as_slice() {
            debug_assert_eq!(only.1, m);
            return self.router(stream, call);
        }
        let f32s = size_of::<f32>();
        for &(first, rows) in &chunks {
            let part = RouterCall {
                shape: RouterShape {
                    m: rows,
                    hidden,
                    n_expert,
                    top_k,
                },
                x: rows_of(&call.x, hidden * f32s, (first, rows))?,
                weights: RouterWeights {
                    w: call.weights.w,
                    w_shared: call.weights.w_shared,
                },
                logits: rows_of(&call.logits, (n_expert + 1) * f32s, (first, rows))?,
                out: RouterOut {
                    ids: rows_of(&call.out.ids, top_k * f32s, (first, rows))?,
                    weights: rows_of(&call.out.weights, top_k * f32s, (first, rows))?,
                    shared_gate: call
                        .out
                        .shared_gate
                        .as_ref()
                        .map(|gate| rows_of(gate, f32s, (first, rows)))
                        .transpose()?,
                    status: call.out.status,
                },
            };
            self.router(stream, part)?;
        }
        Ok(())
    }

    /// Queues the strided logits launches of `call`: one launch below 8 rows, chunks of 4 from 8
    /// rows (the decode kernel at every row count). Refused as `BadRequest` before any native
    /// call: a stream or range of another context, a shape outside the router's limits, a stride
    /// below `n_expert`, a range that is short or misaligned, and a `logits` range that overlaps
    /// `x` or `w`.
    pub(crate) fn router_logits_rows_strided(
        &self,
        stream: &DeviceStream,
        call: RouterLogitsStridedCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let RouterShape {
            m,
            hidden,
            n_expert,
            ..
        } = call.shape;
        let overflow = || EngError::BadRequest("the CUDA strided router sizes overflow".to_owned());
        if m == 0 || hidden == 0 || !hidden.is_multiple_of(4) || n_expert == 0 {
            return Err(EngError::BadRequest(format!(
                "the CUDA strided router takes m, n_expert >= 1 and hidden a multiple of 4; got m {m}, hidden {hidden}, n_expert {n_expert}"
            )));
        }
        if call.logit_stride < n_expert {
            return Err(EngError::BadRequest(format!(
                "the CUDA strided router stride {} is below n_expert {n_expert}",
                call.logit_stride
            )));
        }
        let f32s = size_of::<f32>();
        let x_bytes = m
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(f32s))
            .ok_or_else(overflow)?;
        let w_bytes = n_expert
            .checked_mul(hidden)
            .and_then(|v| v.checked_mul(f32s))
            .ok_or_else(overflow)?;
        let logits_bytes = (m - 1)
            .checked_mul(call.logit_stride)
            .and_then(|v| v.checked_add(n_expert))
            .and_then(|v| v.checked_mul(f32s))
            .ok_or_else(overflow)?;
        require_fit("x", &call.x, x_bytes, 16)?;
        require_fit("w", &call.w, w_bytes, 16)?;
        require_fit("logits", &call.logits, logits_bytes, 4)?;
        let logits = Claim::of("logits", &call.logits, logits_bytes);
        require_apart(&logits, &Claim::of("x", &call.x, x_bytes))?;
        require_apart(&logits, &Claim::of("w", &call.w, w_bytes))?;
        for (first, rows) in router_chunks(m) {
            debug_assert!(rows <= STRIDED_MAX_ROWS);
            let x = rows_of(&call.x, hidden * f32s, (first, rows))?;
            // The chunk's output starts at its first row; it ends at its last row's n_expert.
            let offset = first * call.logit_stride * f32s;
            let bytes = ((rows - 1) * call.logit_stride + n_expert) * f32s;
            let out = call
                .logits
                .buffer()
                .range(call.logits.offset() + offset, bytes)?;
            let raw = StridedLogitsRaw {
                x: self.raw_range(&x)?,
                w: self.raw_range(&call.w)?,
                logits: self.raw_range(&out)?,
                m: rows,
                hidden,
                n_expert,
                logit_stride: call.logit_stride,
            };
            #[cfg(test)]
            self.note_row_kernel(crate::kernels::row_launches::RowKernel::RouterStrided);
            self.launch_on(stream, move |stream_raw| {
                router_logits_strided_launch(stream_raw, raw)
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
