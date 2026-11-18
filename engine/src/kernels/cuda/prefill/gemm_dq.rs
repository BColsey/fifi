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
    GemmDqIq4xsRaw, GemmDqQ3kRaw, GemmDqQ5kRaw, GemmDqQ40Raw, gemm_dq_iq4xs_launch,
    gemm_dq_q3k_launch, gemm_dq_q5k_launch, gemm_dq_q40_launch,
};

/// One Q5_K dequantizing GEMM: `y[r][n] = sum_k dequant(w)[r][k] * x[k][n]`, with `w` the
/// native blocks `[rows][k / 256]`, `x` f16 row-major by `k`, `y` f32 row-major by `rows`.
#[derive(Clone)]
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(crate) struct GemmDqQ5kCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
}

/// The same GEMM over Q3_K blocks: `w` is `[rows][k / 256]` blocks of 110 bytes.
#[derive(Clone)]
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(crate) struct GemmDqQ3kCall<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
}

/// The same GEMM over Q4_0 blocks: `w` is `[rows][k / 32]` blocks of 18 bytes.
#[derive(Clone)]
#[allow(dead_code)] // Staged until a model caller is qualified.
pub(crate) struct GemmDqQ40Call<'b> {
    pub(crate) w: DeviceRange<'b>,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) y: DeviceRange<'b>,
    pub(crate) k: usize,
    pub(crate) rows: usize,
    pub(crate) ncols: usize,
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
}

impl CudaContext {
    #[allow(dead_code)] // Staged until a model caller is qualified.
    pub(crate) fn gemm_dq_q5k(
        &self,
        stream: &DeviceStream,
        call: GemmDqQ5kCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gemm_dq_q5k(&call)?;
        self.launch_on(stream, |stream_raw| gemm_dq_q5k_launch(stream_raw, &raw))
    }

    #[allow(dead_code)] // Staged until a model caller is qualified.
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
        let need_x = call
            .k
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                EngError::BadRequest("the Q3_K GEMM activation size overflows".to_owned())
            })?;
        let need_y = call
            .rows
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                EngError::BadRequest("the Q3_K GEMM output size overflows".to_owned())
            })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, 2)?;
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
        let need_x = call
            .k
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                EngError::BadRequest("the IQ4_XS GEMM activation size overflows".to_owned())
            })?;
        let need_y = call
            .rows
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                EngError::BadRequest("the IQ4_XS GEMM output size overflows".to_owned())
            })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, 2)?;
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
                },
            )
        })
    }

    #[allow(dead_code)] // Staged until a model caller is qualified.
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
        let need_x = call
            .k
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                EngError::BadRequest("the Q4_0 GEMM activation size overflows".to_owned())
            })?;
        let need_y = call
            .rows
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                EngError::BadRequest("the Q4_0 GEMM output size overflows".to_owned())
            })?;
        require_fit("w", &call.w, need_w, 2)?;
        require_fit("x", &call.x, need_x, 2)?;
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
                },
            )
        })
    }

    #[allow(dead_code)] // Staged until a model caller is qualified.
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
        let need_x = call
            .k
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                EngError::BadRequest("the Q5_K GEMM activation size overflows".to_owned())
            })?;
        let need_y = call
            .rows
            .checked_mul(call.ncols)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| {
                EngError::BadRequest("the Q5_K GEMM output size overflows".to_owned())
            })?;
        require_fit("w", &call.w, need_w, 4)?;
        require_fit("x", &call.x, need_x, 2)?;
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
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
