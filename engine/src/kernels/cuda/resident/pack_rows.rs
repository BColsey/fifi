//! Safe wrapper of the pack kernel of a K-row verify window: [`CudaContext::pack_rows`].
//!
//! The rows kernel of the EXL3 projections writes each role's `m` output rows tight, `[m][n_i]`
//! f32. The kernels that read a layer's projections read ONE row of `ld` floats with the roles at
//! fixed column offsets: the delta rule reads z, beta and alpha at `z_off`, `beta_off` and
//! `alpha_off` of one `proj` row; `attn_prep` and `attn_decode` read q, k and v at `0`, `k_off`
//! and `v_off`. One launch per layer copies the tight sources into that row layout: z of a
//! DeltaNet layer (beta and alpha are written straight into the row by the strided logits launch of
//! `router_logits_rows_strided`, and the conv kernel reads the qkv output in place through its own
//! `ld`), or q, k and v of an attention layer.
//!
//! The copy is bit-exact: every element moves as one 32-bit word that no float instruction
//! touches, so a NaN payload, a negative zero or a denormal arrives as it left, and a pack is
//! equal to the single-row projections writing their outputs at the same columns. Columns of the
//! destination that no segment covers are not written (the beta/alpha columns are another
//! launch's).
//!
//! Refused as `BadRequest` before any native call: a stream or range of another context, `m`
//! outside 1..=16, 1..=3 segments, a segment of 0 or more than 65536 columns, a column range that
//! ends past `ld` or overlaps another segment's, a range that is short or not 4-byte aligned, and a
//! source that overlaps the destination.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the K-row verify body")
)]

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::pack_rows::{
    PACK_MAX_LD, PACK_MAX_N, PACK_MAX_ROWS, PACK_MAX_SEGMENTS, PackRowsRaw, PackSegmentRange,
    pack_rows_launch,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

const WORD: usize = size_of::<f32>();

/// One source of a pack: f32 `[m][n]` tight, copied to column `dst_off` of every destination row.
pub(crate) struct PackSegment<'b> {
    pub(crate) src: DeviceRange<'b>,
    pub(crate) n: usize,
    pub(crate) dst_off: usize,
}

/// The call: `segments` into the f32 `[m][ld]` row layout `dst`.
pub(crate) struct PackRowsCall<'b> {
    pub(crate) m: usize,
    pub(crate) ld: usize,
    pub(crate) dst: DeviceRange<'b>,
    pub(crate) segments: Vec<PackSegment<'b>>,
}

fn refuse(detail: String) -> EngError {
    EngError::BadRequest(format!("pack_rows: {detail}"))
}

/// The columns of one destination row that the segments span.
fn check_segments(call: &PackRowsCall<'_>) -> Result<usize, EngError> {
    if !(1..=PACK_MAX_ROWS).contains(&call.m) {
        return Err(refuse(format!(
            "takes 1 to {PACK_MAX_ROWS} rows, got {}",
            call.m
        )));
    }
    if !(1..=PACK_MAX_LD).contains(&call.ld) {
        return Err(refuse(format!(
            "takes ld in 1..={PACK_MAX_LD}, got {}",
            call.ld
        )));
    }
    if !(1..=PACK_MAX_SEGMENTS).contains(&call.segments.len()) {
        return Err(refuse(format!(
            "takes 1 to {PACK_MAX_SEGMENTS} segments, got {}",
            call.segments.len()
        )));
    }
    let mut reach = 0usize;
    for (i, seg) in call.segments.iter().enumerate() {
        if !(1..=PACK_MAX_N).contains(&seg.n) {
            return Err(refuse(format!(
                "segment {i} takes 1 to {PACK_MAX_N} columns, got {}",
                seg.n
            )));
        }
        let end = seg
            .dst_off
            .checked_add(seg.n)
            .filter(|end| *end <= call.ld)
            .ok_or_else(|| {
                refuse(format!(
                    "segment {i} (columns {}..+{}) ends past ld {}",
                    seg.dst_off, seg.n, call.ld
                ))
            })?;
        for (j, other) in call.segments.iter().enumerate().skip(i + 1) {
            if !(end <= other.dst_off || other.dst_off + other.n <= seg.dst_off) {
                return Err(refuse(format!("segments {i} and {j} overlap in columns")));
            }
        }
        reach = reach.max(end);
    }
    Ok(reach)
}

impl CudaContext {
    /// Queues the pack of `call` on `stream` (one kernel). See the module doc.
    pub(crate) fn pack_rows(
        &self,
        stream: &DeviceStream,
        call: PackRowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let reach = check_segments(&call)?;
        let overflow = || refuse("the sizes overflow".to_owned());
        let dst_bytes = (call.m - 1)
            .checked_mul(call.ld)
            .and_then(|words| words.checked_add(reach))
            .and_then(|words| words.checked_mul(WORD))
            .ok_or_else(overflow)?;
        require_fit("dst", &call.dst, dst_bytes, WORD)?;
        let dst_claim = Claim::of("dst", &call.dst, dst_bytes);
        let mut segments = Vec::with_capacity(call.segments.len());
        for seg in &call.segments {
            let src_bytes = call
                .m
                .checked_mul(seg.n)
                .and_then(|words| words.checked_mul(WORD))
                .ok_or_else(overflow)?;
            require_fit("src", &seg.src, src_bytes, WORD)?;
            require_apart(&Claim::of("src", &seg.src, src_bytes), &dst_claim)?;
            segments.push(PackSegmentRange {
                src: self.raw_range(&seg.src)?,
                n: seg.n,
                dst_off: seg.dst_off,
            });
        }
        let raw = PackRowsRaw {
            dst: self.raw_range(&call.dst)?,
            segments,
            m: call.m,
            ld: call.ld,
        };
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::PackRows);
        self.launch_on(stream, move |stream_raw| pack_rows_launch(stream_raw, raw))
    }
}

#[cfg(test)]
mod tests;
