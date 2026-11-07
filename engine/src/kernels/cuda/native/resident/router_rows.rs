//! Raw FFI to `inference_cuda_router_logits_strided_launch` of `native/resident/router.cu`
//! (declared in `native/resident/router.h`): the decode logits kernel with the row stride of the
//! output as a parameter, for the plain F32 projections (GDN beta and alpha) of a K-row verify
//! window. Child module of `native::resident`, so it inherits the `allow(unsafe_code)` of
//! `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The entry is called only through `CudaContext::launch_on`, on the owning thread and under the
//!   dispatch guard. It only enqueues and returns the launch status.
//! - Every range is a `RawRange`, proved inside a live allocation by `RawRange::new`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, before the call, on the
//!   extents the kernel touches; the C entry repeats the checks on the addresses and sizes.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The C argument struct is a live local that the entry reads only during the call. Its `repr(C)`
//!   mirror is private here and pinned by `const` asserts to the `static_assert` of `router.h`.
//!   No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::{offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// The most rows of one strided launch: the decode kernel only (the tiled kernel starts at 8).
pub(in crate::kernels::cuda) const STRIDED_MAX_ROWS: usize = 7;
const MAX_HIDDEN: usize = 65536;
const MAX_EXPERTS: usize = 512;
const MAX_STRIDE: usize = 1 << 20;

/// Mirror of `inference_cuda_router_logits_strided_args`.
#[repr(C)]
struct StridedArgsRaw {
    x: *const f32,
    x_bytes: usize,
    w: *const f32,
    w_bytes: usize,
    logits: *mut f32,
    logits_bytes: usize,
    m: c_int,
    hidden: c_int,
    n_expert: c_int,
    logit_stride: c_int,
}

const _: () = assert!(
    size_of::<StridedArgsRaw>() == 64
        && offset_of!(StridedArgsRaw, m) == 48
        && offset_of!(StridedArgsRaw, n_expert) == 56
        && offset_of!(StridedArgsRaw, logit_stride) == 60
);

unsafe extern "C" {
    #[link_name = "inference_cuda_router_logits_strided_launch"]
    fn router_logits_strided_raw(
        args: *const StridedArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) struct StridedLogitsRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) hidden: usize,
    pub(in crate::kernels::cuda) n_expert: usize,
    /// Floats between the starts of two output rows (`>= n_expert`).
    pub(in crate::kernels::cuda) logit_stride: usize,
}

fn fits(range: &RawRange<'_>, need: usize, align: usize) -> bool {
    range.bytes >= need && range.offset.is_multiple_of(align)
}

/// Queues the strided logits launch of `io`.
pub(in crate::kernels::cuda) fn router_logits_strided_launch(
    stream: &RawStream,
    io: StridedLogitsRaw<'_>,
) -> Result<(), DeviceFailure> {
    let shape_ok = (1..=STRIDED_MAX_ROWS).contains(&io.m)
        && io.hidden.is_multiple_of(4)
        && (4..=MAX_HIDDEN).contains(&io.hidden)
        && (1..=MAX_EXPERTS).contains(&io.n_expert)
        && (io.n_expert..=MAX_STRIDE).contains(&io.logit_stride);
    if !shape_ok {
        return Err(invalid_argument());
    }
    let product = |factors: &[usize]| {
        factors
            .iter()
            .try_fold(1usize, |acc, factor| acc.checked_mul(*factor))
            .ok_or_else(invalid_argument)
    };
    let x_bytes = product(&[io.m, io.hidden, 4])?;
    let w_bytes = product(&[io.n_expert, io.hidden, 4])?;
    let logits_bytes = (io.m - 1)
        .checked_mul(io.logit_stride)
        .and_then(|floats| floats.checked_add(io.n_expert))
        .and_then(|floats| floats.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let logits = Span::of(&io.logits, logits_bytes);
    if !(fits(&io.x, x_bytes, 16)
        && fits(&io.w, w_bytes, 16)
        && fits(&io.logits, logits_bytes, 4)
        && logits.apart(&Span::of(&io.x, x_bytes))
        && logits.apart(&Span::of(&io.w, w_bytes)))
    {
        return Err(invalid_argument());
    }
    let int = |value: usize| c_int::try_from(value).map_err(|_| invalid_argument());
    let args = StridedArgsRaw {
        x: io.x.address().cast::<f32>().cast_const(),
        x_bytes: io.x.bytes,
        w: io.w.address().cast::<f32>().cast_const(),
        w_bytes: io.w.bytes,
        logits: io.logits.address().cast(),
        logits_bytes: io.logits.bytes,
        m: int(io.m)?,
        hidden: int(io.hidden)?,
        n_expert: int(io.n_expert)?,
        logit_stride: int(io.logit_stride)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the tests above proved `x`, `w` and `logits` inside their live allocations, large
    // enough for the call (`logits` for `(m - 1) * logit_stride + n_expert` floats), aligned (16, 16
    // and 4), `m` below the tiled threshold, and `logits`, the only range the kernel writes, apart
    // from `x` and `w`; the entry repeats the checks on the addresses and sizes. The allocations
    // outlive the queued work (`inference_cuda_free` synchronizes first). `args` is a live local
    // that the entry reads only during the call. `stream` is a live handle for the call, and
    // `stage` is a live, writable c_int.
    let status = unsafe { router_logits_strided_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
