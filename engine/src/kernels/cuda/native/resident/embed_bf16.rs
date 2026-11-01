//! Raw FFI to inference_cuda_embed_bf16_launch of native/resident/embed_bf16.cu.
//! Child of native::resident.
//!
//! Safety invariants: RawRange::new proved each range in bounds; this module re-proves sizes,
//! alignment and that x and status are disjoint from every other range; the C entry repeats the
//! checks. Launch-only; the dispatch guard holds (decisions 0023 and 0024).

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_embed_bf16_launch"]
    fn embed_bf16_launch_raw(
        args: *const EmbedBf16ArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_embed_bf16_args, field for field.
#[repr(C)]
struct EmbedBf16ArgsRepr {
    w: *const c_void,
    ids: *const c_void,
    x: *mut c_void,
    status: *mut c_void,
    m: c_int,
    n_rows: c_int,
    k: c_int,
}

const _: () = assert!(
    size_of::<EmbedBf16ArgsRepr>() == 48
        && offset_of!(EmbedBf16ArgsRepr, w) == 0
        && offset_of!(EmbedBf16ArgsRepr, ids) == 8
        && offset_of!(EmbedBf16ArgsRepr, x) == 16
        && offset_of!(EmbedBf16ArgsRepr, status) == 24
        && offset_of!(EmbedBf16ArgsRepr, m) == 32
        && offset_of!(EmbedBf16ArgsRepr, n_rows) == 36
        && offset_of!(EmbedBf16ArgsRepr, k) == 40
);

pub(in crate::kernels::cuda) struct EmbedBf16Raw<'a> {
    pub(in crate::kernels::cuda) w: RawRange<'a>,
    pub(in crate::kernels::cuda) ids: RawRange<'a>,
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) m: usize,
    pub(in crate::kernels::cuda) n_rows: usize,
    pub(in crate::kernels::cuda) k: usize,
}

fn check_embed_bf16(call: &EmbedBf16Raw<'_>) -> Result<(), DeviceFailure> {
    let x_bytes = call
        .m
        .checked_mul(call.k)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(invalid_argument)?;
    let w_bytes = call
        .n_rows
        .checked_mul(call.k)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(invalid_argument)?;
    let ids_bytes = call.m.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.x.bytes >= x_bytes
        && call.w.bytes >= w_bytes
        && call.ids.bytes >= ids_bytes
        && call.status.bytes >= 4;
    let aligned = call.x.offset.is_multiple_of(16)
        && call.w.offset.is_multiple_of(4)
        && call.ids.offset.is_multiple_of(4)
        && call.status.offset.is_multiple_of(4);
    let (x, status) = (Span::of(&call.x, x_bytes), Span::of(&call.status, 4));
    let (w, ids) = (Span::of(&call.w, w_bytes), Span::of(&call.ids, ids_bytes));
    let apart =
        x.apart(&w) && x.apart(&ids) && x.apart(&status) && status.apart(&w) && status.apart(&ids);
    let dims = (1..=4096).contains(&call.m)
        && (1..=(1 << 28)).contains(&call.n_rows)
        && (8..=32_768).contains(&call.k)
        && call.k.is_multiple_of(8);
    if !(sized && aligned && apart && dims) {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn embed_bf16_launch(
    stream: &RawStream,
    call: &EmbedBf16Raw<'_>,
) -> Result<(), DeviceFailure> {
    check_embed_bf16(call)?;
    let args = EmbedBf16ArgsRepr {
        w: call.w.address().cast_const(),
        ids: call.ids.address().cast_const(),
        x: call.x.address(),
        status: call.status.address(),
        m: c_int::try_from(call.m).map_err(|_| invalid_argument())?,
        n_rows: c_int::try_from(call.n_rows).map_err(|_| invalid_argument())?,
        k: c_int::try_from(call.k).map_err(|_| invalid_argument())?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_embed_bf16 proved sizes, alignment and the write/read separation; the adapter
    // repeats them on the addresses.
    let status = unsafe { embed_bf16_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
