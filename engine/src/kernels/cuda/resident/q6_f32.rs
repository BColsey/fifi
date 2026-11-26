//! Test-only checked Q6_K/F32 projection for the Flash-Next accuracy experiment.
//! It queues one launch and does not allocate, copy, or synchronize.
use crate::EngError;
use crate::kernels::cuda::context::{
    Claim, CudaContext, DeviceRange, DeviceStream, require_apart, require_fit,
};
use crate::kernels::cuda::native::resident::q6_f32::{Q6F32Raw, q6_f32_launch, sizes};

pub(crate) struct Q6F32Call<'a> {
    pub(crate) w: DeviceRange<'a>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) y: DeviceRange<'a>,
    pub(crate) rows: usize,
    pub(crate) k: usize,
}

impl CudaContext {
    pub(crate) fn q6_f32(
        &self,
        stream: &DeviceStream,
        call: Q6F32Call<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (w_bytes, x_bytes, y_bytes) = sizes(call.rows, call.k).ok_or_else(|| {
            EngError::BadRequest("q6_f32: invalid or overflowing geometry".into())
        })?;
        require_fit("w", &call.w, w_bytes, 2)?;
        require_fit("x", &call.x, x_bytes, 4)?;
        require_fit("y", &call.y, y_bytes, 4)?;
        require_apart(
            &Claim::of("y", &call.y, y_bytes),
            &Claim::of("w", &call.w, w_bytes),
        )?;
        require_apart(
            &Claim::of("y", &call.y, y_bytes),
            &Claim::of("x", &call.x, x_bytes),
        )?;
        let raw = Q6F32Raw {
            w: self.raw_range(&call.w)?,
            x: self.raw_range(&call.x)?,
            y: self.raw_range(&call.y)?,
            rows: call.rows,
            k: call.k,
        };
        self.launch_on(stream, |s| q6_f32_launch(s, &raw))
    }
}

#[cfg(test)]
mod tests;
