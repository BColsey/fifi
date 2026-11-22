//! Launch-only FFI for the packed Inkling controller workspace.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

#[derive(Clone, Copy)]
#[repr(i32)]
pub(in crate::kernels::cuda) enum Arithmetic {
    Generic = 0,
    CpuParity = 1,
}

unsafe extern "C" {
    fn inference_cuda_inkling_controller_norm(
        workspace: *mut c_void,
        health: *mut c_void,
        width: c_int,
        roles: c_int,
        eps: f32,
        cpu_parity: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn launch(
    stream: &RawStream,
    workspace: RawRange<'_>,
    health: RawRange<'_>,
    [width, roles]: [usize; 2],
    eps: f32,
    arithmetic: Arithmetic,
) -> Result<(), DeviceFailure> {
    if !(1..=4096).contains(&width)
        || !(1..=1536).contains(&roles)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(invalid_argument());
    }
    let bytes = 6 * roles * width * size_of::<f32>();
    let health_bytes = roles * size_of::<f32>();
    if workspace.bytes() < bytes
        || health.bytes() < health_bytes
        || !workspace.offset.is_multiple_of(4)
        || !health.offset.is_multiple_of(4)
        || !Span::of(&workspace, bytes).apart(&Span::of(&health, health_bytes))
    {
        return Err(invalid_argument());
    }
    let mut stage = STAGE_NONE;
    // SAFETY: both live ranges hold the accessed words and are aligned and disjoint.
    // The device frees allocations only after queued work completes.
    let status = unsafe {
        inference_cuda_inkling_controller_norm(
            workspace.address(),
            health.address(),
            width as c_int,
            roles as c_int,
            eps,
            arithmetic as c_int,
            stream.0.as_ptr(),
            &mut stage,
        )
    };
    checked(status, stage)
}
