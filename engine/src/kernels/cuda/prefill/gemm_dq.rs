//! The in-house dequantizing prefill GEMM over native quant blocks: [`CudaContext::gemm_dq_q5k`]
//! queues one launch of `gemm_dq.cu`'s Q5_K instance and returns when it is queued. The kernel
//! decodes the file's own 176-byte Q5_K blocks into shared memory tile by tile and multiplies
//! against an f16 activation with f32 accumulation: no repack, no f16 weight copy, no vendored
//! code. The per-type decoder lives in `native/prefill/gemm_dq_decoders.cuh`; a new quant type
//! adds a decoder and one launch entry there, not a kernel.
//!
//! Every refusal is a `BadRequest` and happens before any native call: `k` must be a positive
//! multiple of 256 (the Q5_K block), `rows` and `ncols` positive, `w`/`x`/`y` exactly the byte
//! counts of the call and aligned (w to 16, the uint4 staging loads; y to 4; x to 2), and `y` overlaps neither `w` nor `x`.
//! A refused call changes nothing.

use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::prefill::gemm_dq::{
    GemmDqCodebookRaw, GemmDqIq4xsRaw, GemmDqLayout, GemmDqQ3kRaw, GemmDqQ5kRaw, GemmDqQ40Raw,
    gemm_dq_codebook_launch, gemm_dq_iq4xs_launch, gemm_dq_q3k_launch, gemm_dq_q5k_launch,
    gemm_dq_q40_launch,
};
use crate::kernels::cuda::resident::matvec_q::QuantType;

/// The family-neutral call for [`CudaContext::gemm_dq`]: the same fields as every
/// per-type call, because the per-type entries differ only in which block decoder the
/// launch picks.
#[derive(Clone)]
pub(crate) struct GemmDqTypedCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
    /// 0 keeps x row-major by k; otherwise x holds token rows `[ncols][x_col_stride]`.
    pub(crate) x_col_stride: usize,
    /// 0 keeps y row-major by rows; otherwise y holds token rows `[ncols][y_col_stride]`.
    pub(crate) y_col_stride: usize,
    /// x holds f32 values instead of f16 (the prefill norm-pack buffers).
    pub(crate) x_f32: bool,
}

/// One Q5_K dequantizing GEMM: `y[r][n] = sum_k dequant(w)[r][k] * x[k][n]`, with `w` the
/// native blocks `[rows][k / 256]`, `x` f16 row-major by `k`, `y` f32 row-major by `rows`.
#[derive(Clone)]
pub(crate) struct GemmDqQ5kCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
    /// 0 keeps x row-major by k; otherwise x holds token rows `[ncols][x_col_stride]`.
    pub(crate) x_col_stride: usize,
    /// 0 keeps y row-major by rows; otherwise y holds token rows `[ncols][y_col_stride]`.
    pub(crate) y_col_stride: usize,
    /// x holds f32 values instead of f16 (the prefill norm-pack buffers).
    pub(crate) x_f32: bool,
}

/// The same GEMM over Q3_K blocks: `w` is `[rows][k / 256]` blocks of 110 bytes.
#[derive(Clone)]
pub(crate) struct GemmDqQ3kCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
    /// 0 keeps x row-major by k; otherwise x holds token rows `[ncols][x_col_stride]`.
    pub(crate) x_col_stride: usize,
    /// 0 keeps y row-major by rows; otherwise y holds token rows `[ncols][y_col_stride]`.
    pub(crate) y_col_stride: usize,
    /// x holds f32 values instead of f16 (the prefill norm-pack buffers).
    pub(crate) x_f32: bool,
}

/// The same GEMM over Q4_0 blocks: `w` is `[rows][k / 32]` blocks of 18 bytes.
#[derive(Clone)]
pub(crate) struct GemmDqQ40Call<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
    /// 0 keeps x row-major by k; otherwise x holds token rows `[ncols][x_col_stride]`.
    pub(crate) x_col_stride: usize,
    /// 0 keeps y row-major by rows; otherwise y holds token rows `[ncols][y_col_stride]`.
    pub(crate) y_col_stride: usize,
    /// x holds f32 values instead of f16 (the prefill norm-pack buffers).
    pub(crate) x_f32: bool,
}

/// The activation and output spans for either layout: row-major packs the full logical
/// rectangle, token rows need the last column's full stride head plus its tail.
fn gemm_dq_layout_spans(
    k: usize,
    rows: usize,
    ncols: usize,
    x_col_stride: usize,
    y_col_stride: usize,
    x_f32: bool,
) -> Option<(usize, usize)> {
    let cap = i32::MAX as usize;
    let x_unit = if x_f32 { 4 } else { 2 };
    let need_x = if x_col_stride == 0 {
        k.checked_mul(ncols)?.checked_mul(x_unit)?
    } else if x_col_stride < k || ncols > cap / x_col_stride {
        return None;
    } else {
        (ncols - 1)
            .checked_mul(x_col_stride)?
            .checked_add(k)?
            .checked_mul(x_unit)?
    };
    let need_y = if y_col_stride == 0 {
        rows.checked_mul(ncols)?.checked_mul(4)?
    } else if y_col_stride < rows || ncols > cap / y_col_stride {
        return None;
    } else {
        (ncols - 1)
            .checked_mul(y_col_stride)?
            .checked_add(rows)?
            .checked_mul(4)?
    };
    Some((need_x, need_y))
}

/// The same GEMM over IQ4_XS super-blocks: `w` is `[rows][k / 256]` blocks of 136 bytes.
#[derive(Clone)]
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(crate) struct GemmDqIq4xsCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
    pub(crate) x_col_stride: usize,
    pub(crate) y_col_stride: usize,
    pub(crate) x_f32: bool,
}

impl CudaContext {
    /// The one family-neutral dispatch for the in-house dequantizing GEMM: `quant` picks
    /// the block decoder, the field checks are the per-type entries' own, and a family
    /// plan wires this call instead of copying a per-type match. A type without an
    /// in-house arm is a `BadRequest` before any native call.
    pub(crate) fn gemm_dq(
        &self,
        stream: &DeviceStream,
        quant: QuantType,
        call: GemmDqTypedCall<'_>,
    ) -> Result<(), EngError> {
        let GemmDqTypedCall {
            w,
            x,
            y,
            k,
            rows,
            ncols,
            x_col_stride,
            y_col_stride,
            x_f32,
        } = call;
        match quant {
            QuantType::Q5K => self.gemm_dq_q5k(
                stream,
                GemmDqQ5kCall {
                    w,
                    x,
                    y,
                    k,
                    rows,
                    ncols,
                    x_col_stride,
                    y_col_stride,
                    x_f32,
                },
            ),
            QuantType::Q40 => self.gemm_dq_q40(
                stream,
                GemmDqQ40Call {
                    w,
                    x,
                    y,
                    k,
                    rows,
                    ncols,
                    x_col_stride,
                    y_col_stride,
                    x_f32,
                },
            ),
            QuantType::Q3K => self.gemm_dq_q3k(
                stream,
                GemmDqQ3kCall {
                    w,
                    x,
                    y,
                    k,
                    rows,
                    ncols,
                    x_col_stride,
                    y_col_stride,
                    x_f32,
                },
            ),
            QuantType::Iq4Xs => self.gemm_dq_iq4xs(
                stream,
                GemmDqIq4xsCall {
                    w,
                    x,
                    y,
                    k,
                    rows,
                    ncols,
                    x_col_stride,
                    y_col_stride,
                    x_f32,
                },
            ),
            QuantType::Q80 | QuantType::Q50 | QuantType::Q4K | QuantType::Q6K => {
                Err(EngError::BadRequest(format!(
                    "{quant:?} rides the vendored MMQ pair, not the in-house dequantizing GEMM"
                )))
            }
            ty if ty.is_codebook_dense() => self.gemm_dq_codebook(
                stream,
                ty,
                GemmDqTypedCall {
                    w,
                    x,
                    y,
                    k,
                    rows,
                    ncols,
                    x_col_stride,
                    y_col_stride,
                    x_f32,
                },
            ),
            QuantType::Iq4Nl
            | QuantType::Iq2S
            | QuantType::Iq3Xxs
            | QuantType::Iq2Xxs
            | QuantType::Q20
            | QuantType::Iq2Xs
            | QuantType::Iq3S
            | QuantType::Q2K
            | QuantType::Iq1S
            | QuantType::Iq1M => unreachable!("covered by is_codebook_dense"),
        }
    }

    /// The dequantizing GEMM over the codebook types: the decode matvec's window decoders
    /// behind the same tile kernel. Checks mirror the per-type entries: `k` a positive
    /// multiple of the block size and of 64, rows and columns above zero, exact byte counts,
    /// 2-byte `w`/`x` and 4-byte `y` alignment, `y` apart from `w` and `x`.
    pub(crate) fn gemm_dq_codebook(
        &self,
        stream: &DeviceStream,
        ty: QuantType,
        call: GemmDqTypedCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        let unit = ty.block_size().max(64);
        if !ty.is_codebook_dense() || call.k == 0 || !call.k.is_multiple_of(unit) {
            return Err(EngError::BadRequest(format!(
                "the {ty:?} dequantizing GEMM needs k as a positive multiple of {unit}, got {}",
                call.k
            )));
        }
        if call.rows == 0 || call.ncols == 0 {
            return Err(EngError::BadRequest(format!(
                "the {ty:?} dequantizing GEMM needs rows and columns above zero"
            )));
        }
        let need_w = ty.weight_bytes(call.rows, call.k).ok_or_else(|| {
            EngError::BadRequest(format!("the {ty:?} GEMM weight size overflows"))
        })?;
        let (need_x, need_y) = gemm_dq_layout_spans(
            call.k,
            call.rows,
            call.ncols,
            call.x_col_stride,
            call.y_col_stride,
            call.x_f32,
        )
        .ok_or_else(|| {
            EngError::BadRequest(format!(
                "the {ty:?} GEMM size overflows for the layout strides"
            ))
        })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, if call.x_f32 { 4 } else { 2 })?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &call.w, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        self.launch_on(stream, |stream_raw| {
            gemm_dq_codebook_launch(
                stream_raw,
                &GemmDqCodebookRaw {
                    ty,
                    w: w_raw,
                    x: x_raw,
                    y: y_raw,
                    k: call.k,
                    rows: call.rows,
                    ncols: call.ncols,
                    layout: GemmDqLayout {
                        x_col_stride: call.x_col_stride,
                        y_col_stride: call.y_col_stride,
                        x_f32: call.x_f32,
                    },
                },
            )
        })
    }

    pub(crate) fn gemm_dq_q5k(
        &self,
        stream: &DeviceStream,
        call: GemmDqQ5kCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gemm_dq_q5k(&call)?;
        self.launch_on(stream, |stream_raw| gemm_dq_q5k_launch(stream_raw, &raw))
    }

    pub(crate) fn gemm_dq_q3k(
        &self,
        stream: &DeviceStream,
        call: GemmDqQ3kCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        if call.k == 0 || !call.k.is_multiple_of(256) {
            return Err(EngError::BadRequest(format!(
                "the Q3_K dequantizing GEMM needs k as a positive multiple of 256, got {}",
                call.k
            )));
        }
        if call.rows == 0 || call.ncols == 0 {
            return Err(EngError::BadRequest(
                "the Q3_K dequantizing GEMM needs rows and columns above zero".to_owned(),
            ));
        }
        let blocks = call.rows.checked_mul(call.k / 256).ok_or_else(|| {
            EngError::BadRequest("the Q3_K GEMM weight size overflows".to_owned())
        })?;
        let need_w = blocks.checked_mul(110).ok_or_else(|| {
            EngError::BadRequest("the Q3_K GEMM weight size overflows".to_owned())
        })?;
        let (need_x, need_y) = gemm_dq_layout_spans(
            call.k,
            call.rows,
            call.ncols,
            call.x_col_stride,
            call.y_col_stride,
            call.x_f32,
        )
        .ok_or_else(|| {
            EngError::BadRequest("the Q3_K GEMM size overflows for the layout strides".to_owned())
        })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, if call.x_f32 { 4 } else { 2 })?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &call.w, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        self.launch_on(stream, |stream_raw| {
            gemm_dq_q3k_launch(
                stream_raw,
                &GemmDqQ3kRaw {
                    w: w_raw,
                    x: x_raw,
                    y: y_raw,
                    k: call.k,
                    rows: call.rows,
                    ncols: call.ncols,
                    layout: GemmDqLayout {
                        x_col_stride: call.x_col_stride,
                        y_col_stride: call.y_col_stride,
                        x_f32: call.x_f32,
                    },
                },
            )
        })
    }

    #[allow(dead_code)] // Staged until a model caller is qualified.
    pub(crate) fn gemm_dq_iq4xs(
        &self,
        stream: &DeviceStream,
        call: GemmDqIq4xsCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        if call.k == 0 || !call.k.is_multiple_of(256) {
            return Err(EngError::BadRequest(format!(
                "the IQ4_XS dequantizing GEMM needs k as a positive multiple of 256, got {}",
                call.k
            )));
        }
        if call.rows == 0 || call.ncols == 0 {
            return Err(EngError::BadRequest(
                "the IQ4_XS dequantizing GEMM needs rows and columns above zero".to_owned(),
            ));
        }
        let blocks = call.rows.checked_mul(call.k / 256).ok_or_else(|| {
            EngError::BadRequest("the IQ4_XS GEMM weight size overflows".to_owned())
        })?;
        let need_w = blocks.checked_mul(136).ok_or_else(|| {
            EngError::BadRequest("the IQ4_XS GEMM weight size overflows".to_owned())
        })?;
        let (need_x, need_y) = gemm_dq_layout_spans(
            call.k,
            call.rows,
            call.ncols,
            call.x_col_stride,
            call.y_col_stride,
            call.x_f32,
        )
        .ok_or_else(|| {
            EngError::BadRequest("the IQ4_XS GEMM size overflows for the layout strides".to_owned())
        })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, if call.x_f32 { 4 } else { 2 })?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &call.w, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        self.launch_on(stream, |stream_raw| {
            gemm_dq_iq4xs_launch(
                stream_raw,
                &GemmDqIq4xsRaw {
                    w: w_raw,
                    x: x_raw,
                    y: y_raw,
                    k: call.k,
                    rows: call.rows,
                    ncols: call.ncols,
                    layout: GemmDqLayout {
                        x_col_stride: call.x_col_stride,
                        y_col_stride: call.y_col_stride,
                        x_f32: call.x_f32,
                    },
                },
            )
        })
    }

    pub(crate) fn gemm_dq_q40(
        &self,
        stream: &DeviceStream,
        call: GemmDqQ40Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        if call.k == 0 || !call.k.is_multiple_of(64) {
            return Err(EngError::BadRequest(format!(
                "the Q4_0 dequantizing GEMM needs k as a positive multiple of 64, got {}",
                call.k
            )));
        }
        if call.rows == 0 || call.ncols == 0 {
            return Err(EngError::BadRequest(
                "the Q4_0 dequantizing GEMM needs rows and columns above zero".to_owned(),
            ));
        }
        let blocks = call.rows.checked_mul(call.k / 32).ok_or_else(|| {
            EngError::BadRequest("the Q4_0 GEMM weight size overflows".to_owned())
        })?;
        let need_w = blocks.checked_mul(18).ok_or_else(|| {
            EngError::BadRequest("the Q4_0 GEMM weight size overflows".to_owned())
        })?;
        let (need_x, need_y) = gemm_dq_layout_spans(
            call.k,
            call.rows,
            call.ncols,
            call.x_col_stride,
            call.y_col_stride,
            call.x_f32,
        )
        .ok_or_else(|| {
            EngError::BadRequest("the Q4_0 GEMM size overflows for the layout strides".to_owned())
        })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, if call.x_f32 { 4 } else { 2 })?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &call.w, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        self.launch_on(stream, |stream_raw| {
            gemm_dq_q40_launch(
                stream_raw,
                &GemmDqQ40Raw {
                    w: w_raw,
                    x: x_raw,
                    y: y_raw,
                    k: call.k,
                    rows: call.rows,
                    ncols: call.ncols,
                    layout: GemmDqLayout {
                        x_col_stride: call.x_col_stride,
                        y_col_stride: call.y_col_stride,
                        x_f32: call.x_f32,
                    },
                },
            )
        })
    }

    fn plan_gemm_dq_q5k<'b>(&self, call: &GemmDqQ5kCall<'b>) -> Result<GemmDqQ5kRaw<'b>, EngError> {
        let (w_raw, x_raw, y_raw) = (
            self.raw_range(&call.w)?,
            self.raw_range(&call.x)?,
            self.raw_range(&call.y)?,
        );
        if call.k == 0 || !call.k.is_multiple_of(256) {
            return Err(EngError::BadRequest(format!(
                "the Q5_K dequantizing GEMM needs k as a positive multiple of 256, got {}",
                call.k
            )));
        }
        if call.rows == 0 || call.ncols == 0 {
            return Err(EngError::BadRequest(
                "the Q5_K dequantizing GEMM needs rows and columns above zero".to_owned(),
            ));
        }
        let blocks = call.rows.checked_mul(call.k / 256).ok_or_else(|| {
            EngError::BadRequest("the Q5_K GEMM weight size overflows".to_owned())
        })?;
        let need_w = blocks.checked_mul(176).ok_or_else(|| {
            EngError::BadRequest("the Q5_K GEMM weight size overflows".to_owned())
        })?;
        let (need_x, need_y) = gemm_dq_layout_spans(
            call.k,
            call.rows,
            call.ncols,
            call.x_col_stride,
            call.y_col_stride,
            call.x_f32,
        )
        .ok_or_else(|| {
            EngError::BadRequest("the Q5_K GEMM size overflows for the layout strides".to_owned())
        })?;
        require_fit("w", &call.w, need_w, 4)?;
        require_fit("x", &call.x, need_x, if call.x_f32 { 4 } else { 2 })?;
        require_fit("y", &call.y, need_y, 4)?;
        let written = Claim::of("y", &call.y, need_y);
        require_apart(&written, &Claim::of("w", &call.w, need_w))?;
        require_apart(&written, &Claim::of("x", &call.x, need_x))?;
        Ok(GemmDqQ5kRaw {
            w: w_raw,
            x: x_raw,
            y: y_raw,
            k: call.k,
            rows: call.rows,
            ncols: call.ncols,
            layout: GemmDqLayout {
                x_col_stride: call.x_col_stride,
                y_col_stride: call.y_col_stride,
                x_f32: call.x_f32,
            },
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
