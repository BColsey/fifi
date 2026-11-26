//! Stable device top-20 selection without a full logits readback. Launch-only; no allocation,
//! copy or synchronization. The result is a compact candidate list,not a device RNG/draw.
use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::shortlist::{
    self as raw, MassRaw, ShortlistRaw, ShortlistRowsRaw,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};
pub(crate) const K: usize = raw::K;
pub(crate) const RESULT_BYTES: usize = raw::RESULT_BYTES;
pub(crate) const RESULT_OFFSET: usize = raw::RESULT_OFFSET;
#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "sized for the multi-row shortlist, which no step launches now; tests do"
    )
)]
pub(crate) const SCRATCH_BYTES: usize = raw::SCRATCH_BYTES;
/// Where the [`MASS_PARTS`] f64 partials of [`CudaContext::shortlist_mass`] sit in the scratch.
pub(crate) const MASS_OFFSET: usize = raw::MASS_OFFSET;
/// The f64 partials of one mass launch.
pub(crate) const MASS_PARTS: usize = raw::MASS_PARTS;
/// Scratch bytes a mass launch needs.
pub(crate) const MASS_SCRATCH_BYTES: usize = raw::MASS_SCRATCH_BYTES;
/// Bytes of the partials of one row of the narrow K-row launch (the final record is not part of
/// them).
#[cfg(test)]
pub(crate) const PARTS_BYTES: usize = raw::PARTS_BYTES;
/// The wide width (Bolt's `top_k` 64); [`K`] is the narrow one.
pub(crate) const WIDE_K: usize = raw::WIDE_K;
/// The byte layout of a launch at one ranked width.
pub(crate) type Shape = raw::Shape;

/// The layout at width `k`, or a refusal for a width the kernels do not instantiate.
pub(crate) fn shape(k: usize) -> Result<Shape, EngError> {
    Shape::of(k).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the device shortlist ranks {K} or {WIDE_K} candidates, not {k}"
        ))
    })
}
/// The arguments of one K-row launch: the `m` rows of `logits` (`ld` floats apart) each get the record
/// a single-row launch would give them, stored at `out + r * RESULT_BYTES`.
pub(crate) struct ShortlistRowsCall<'a> {
    /// Rows of the window.
    pub(crate) m: usize,
    /// Values per row.
    pub(crate) n: usize,
    /// Floats between the starts of two logits rows (`>= n`).
    pub(crate) ld: usize,
    /// `(m - 1) * ld + n` floats, 4-byte aligned. Read only.
    pub(crate) logits: DeviceRange<'a>,
    /// Scratch of `m * scratch_stride` bytes, 16-byte aligned; needs no initialization.
    pub(crate) scratch: DeviceRange<'a>,
    /// Bytes between the partials of two rows: a multiple of 16, at least the partials of a row at this width.
    pub(crate) scratch_stride: usize,
    /// `m * shape(k).part_bytes` bytes, 16-byte aligned: record `r` is row `r`'s.
    pub(crate) out: DeviceRange<'a>,
    /// The ranked width of every record.
    pub(crate) k: usize,
}

/// The arguments of the softmax-mass launch that follows a single-row shortlist on the same
/// scratch and stream.
pub(crate) struct ShortlistMassCall<'a> {
    pub(crate) logits: DeviceRange<'a>,
    /// The shortlist scratch extended to [`MASS_SCRATCH_BYTES`].
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) n: usize,
    /// One over the sampling temperature: finite and positive.
    pub(crate) inv_temperature: f32,
    /// The width of the record the shortlist launch wrote.
    pub(crate) k: usize,
}

pub(crate) struct ShortlistCall<'a> {
    pub(crate) logits: DeviceRange<'a>,
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) n: usize,
    /// The ranked width: [`K`] or [`WIDE_K`].
    pub(crate) k: usize,
}
impl CudaContext {
    /// The caller orders the finite-input certificate/result read after this stream work.
    pub(crate) fn shortlist(
        &self,
        stream: &DeviceStream,
        call: ShortlistCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let logits = self.raw_range(&call.logits)?;
        let scratch = self.raw_range(&call.scratch)?;
        if !(1..=raw::MAX_LOGITS).contains(&call.n) {
            return Err(EngError::BadRequest(
                "device shortlist vocabulary must be in 1..=2^30".to_owned(),
            ));
        }
        let bytes = call
            .n
            .checked_mul(4)
            .ok_or_else(|| EngError::BadRequest("device shortlist size overflows".to_owned()))?;
        require_fit("shortlist logits", &call.logits, bytes, 4)?;
        let layout = shape(call.k)?;
        require_fit("shortlist scratch", &call.scratch, layout.scratch_bytes, 16)?;
        require_apart(
            &Claim::of("shortlist scratch", &call.scratch, layout.scratch_bytes),
            &Claim::of("shortlist logits", &call.logits, bytes),
        )?;
        self.launch_on(stream, move |s| {
            raw::launch(
                s,
                ShortlistRaw {
                    logits,
                    scratch,
                    n: call.n,
                    k: call.k,
                },
            )
        })
    }
}
impl CudaContext {
    /// The softmax mass of the row the shortlist just ranked: sum of
    /// `expf((logit - top) * inv_temperature)` over the whole row, as [`MASS_PARTS`] f64 partials
    /// at [`MASS_OFFSET`] of the scratch. One kernel on `stream`; no allocation, copy or
    /// synchronization. The caller orders the read of the partials after this stream work.
    pub(crate) fn shortlist_mass(
        &self,
        stream: &DeviceStream,
        call: ShortlistMassCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let logits = self.raw_range(&call.logits)?;
        let scratch = self.raw_range(&call.scratch)?;
        if !(1..=raw::MAX_LOGITS).contains(&call.n) {
            return Err(EngError::BadRequest(
                "device shortlist mass vocabulary must be in 1..=2^30".to_owned(),
            ));
        }
        if !call.inv_temperature.is_finite() || call.inv_temperature <= 0.0 {
            return Err(EngError::BadRequest(
                "device shortlist mass needs a finite positive inverse temperature".to_owned(),
            ));
        }
        let bytes = call.n.checked_mul(4).ok_or_else(|| {
            EngError::BadRequest("device shortlist mass size overflows".to_owned())
        })?;
        require_fit("shortlist mass logits", &call.logits, bytes, 4)?;
        let layout = shape(call.k)?;
        require_fit(
            "shortlist mass scratch",
            &call.scratch,
            layout.mass_scratch_bytes,
            16,
        )?;
        require_apart(
            &Claim::of(
                "shortlist mass scratch",
                &call.scratch,
                layout.mass_scratch_bytes,
            ),
            &Claim::of("shortlist mass logits", &call.logits, bytes),
        )?;
        self.launch_on(stream, move |s| {
            raw::launch_mass(
                s,
                MassRaw {
                    logits,
                    scratch,
                    n: call.n,
                    inv_temperature: call.inv_temperature,
                    k: call.k,
                },
            )
        })
    }
}
impl CudaContext {
    /// The K-row shortlist: two kernels on `stream`, one record per row, no allocation, copy or
    /// synchronization. The caller orders the read of `out` after this stream work.
    pub(crate) fn shortlist_rows(
        &self,
        stream: &DeviceStream,
        call: ShortlistRowsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        if !(1..=raw::MAX_ROWS).contains(&call.m) {
            return Err(EngError::BadRequest(format!(
                "device shortlist rows takes 1 to {} rows, got {}",
                raw::MAX_ROWS,
                call.m
            )));
        }
        if !(1..=raw::MAX_LOGITS).contains(&call.n) || call.ld < call.n {
            return Err(EngError::BadRequest(
                "device shortlist rows: the vocabulary must be in 1..=2^30 and fit the row stride"
                    .to_owned(),
            ));
        }
        let layout = shape(call.k)?;
        if call.scratch_stride < layout.result_offset || !call.scratch_stride.is_multiple_of(16) {
            return Err(EngError::BadRequest(
                "device shortlist rows: the scratch stride must be a multiple of 16 of at least the \
                 partials of a row"
                    .to_owned(),
            ));
        }
        let overflow = || EngError::BadRequest("device shortlist rows size overflows".to_owned());
        let logits_bytes = (call.m - 1)
            .checked_mul(call.ld)
            .and_then(|floats| floats.checked_add(call.n))
            .and_then(|floats| floats.checked_mul(4))
            .ok_or_else(overflow)?;
        let scratch_bytes = call
            .m
            .checked_mul(call.scratch_stride)
            .ok_or_else(overflow)?;
        let out_bytes = call.m * layout.part_bytes;
        let logits = self.raw_range(&call.logits)?;
        let scratch = self.raw_range(&call.scratch)?;
        let out = self.raw_range(&call.out)?;
        require_fit("shortlist rows logits", &call.logits, logits_bytes, 4)?;
        require_fit("shortlist rows scratch", &call.scratch, scratch_bytes, 16)?;
        require_fit("shortlist rows out", &call.out, out_bytes, 16)?;
        let claims = [
            Claim::of("shortlist rows logits", &call.logits, logits_bytes),
            Claim::of("shortlist rows scratch", &call.scratch, scratch_bytes),
            Claim::of("shortlist rows out", &call.out, out_bytes),
        ];
        for (i, a) in claims.iter().enumerate() {
            for b in &claims[i + 1..] {
                require_apart(a, b)?;
            }
        }
        #[cfg(test)]
        self.note_row_kernel(crate::kernels::row_launches::RowKernel::ShortlistRows);
        let (m, n, ld, scratch_stride, k) = (call.m, call.n, call.ld, call.scratch_stride, call.k);
        self.launch_on(stream, move |s| {
            raw::launch_rows(
                s,
                ShortlistRowsRaw {
                    logits,
                    scratch,
                    out,
                    n,
                    m,
                    ld,
                    scratch_stride,
                    k,
                },
            )
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rows_tests;
