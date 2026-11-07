//! Stable device top-20 selection without a full logits readback. Launch-only; no allocation,
//! copy or synchronization. The result is a compact candidate list,not a device RNG/draw.
use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
use crate::kernels::cuda::native::resident::shortlist::{
    self as raw, ShortlistRaw, ShortlistRowsRaw,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};
pub(crate) const K: usize = raw::K;
pub(crate) const RESULT_BYTES: usize = raw::RESULT_BYTES;
pub(crate) const RESULT_OFFSET: usize = raw::RESULT_OFFSET;
pub(crate) const SCRATCH_BYTES: usize = raw::SCRATCH_BYTES;
/// Bytes of the partials of one row of a K-row launch (the final record is not part of them).
pub(crate) const PARTS_BYTES: usize = raw::PARTS_BYTES;
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
    /// Bytes between the partials of two rows: a multiple of 16, at least [`PARTS_BYTES`].
    pub(crate) scratch_stride: usize,
    /// `m * RESULT_BYTES` bytes, 16-byte aligned: record `r` is row `r`'s.
    pub(crate) out: DeviceRange<'a>,
}

pub(crate) struct ShortlistCall<'a> {
    pub(crate) logits: DeviceRange<'a>,
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) n: usize,
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
        require_fit("shortlist scratch", &call.scratch, SCRATCH_BYTES, 16)?;
        require_apart(
            &Claim::of("shortlist scratch", &call.scratch, SCRATCH_BYTES),
            &Claim::of("shortlist logits", &call.logits, bytes),
        )?;
        self.launch_on(stream, move |s| {
            raw::launch(
                s,
                ShortlistRaw {
                    logits,
                    scratch,
                    n: call.n,
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
        if call.scratch_stride < PARTS_BYTES || !call.scratch_stride.is_multiple_of(16) {
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
        let out_bytes = call.m * RESULT_BYTES;
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
        let (m, n, ld, scratch_stride) = (call.m, call.n, call.ld, call.scratch_stride);
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
                },
            )
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod rows_tests;
