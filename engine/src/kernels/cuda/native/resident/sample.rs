//! Raw FFI to inference_cuda_sample_launch of native/resident/sample.cu.
//!
//! Safety invariants: RawRange::new proved each range in bounds; this module
//! re-proves sizes, alignment, parameter ranges and the separation of the three
//! outputs and the scratch row from the logits; the C entry repeats them.
//! Launch-only, under the dispatch guard (decisions 0023 and 0024).

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_sample_shortlist_launch"]
    fn sample_shortlist_launch_raw(
        args: *const SampleShortlistArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_sample_launch"]
    fn sample_launch_raw(
        args: *const SampleArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_sample_args, field for field.
#[repr(C)]
struct SampleArgsRepr {
    token: *mut c_void,
    logprob: *mut c_void,
    status: *mut c_void,
    probs: *mut c_void,
    logits: *const c_void,
    n: c_int,
    temperature: f32,
    top_k: c_int,
    top_p: f32,
    min_p: f32,
    seed_lo: u32,
    seed_hi: u32,
    pos_lo: u32,
    pos_hi: u32,
}

const _: () = assert!(
    size_of::<SampleArgsRepr>() == 80
        && offset_of!(SampleArgsRepr, token) == 0
        && offset_of!(SampleArgsRepr, logprob) == 8
        && offset_of!(SampleArgsRepr, status) == 16
        && offset_of!(SampleArgsRepr, probs) == 24
        && offset_of!(SampleArgsRepr, logits) == 32
        && offset_of!(SampleArgsRepr, n) == 40
        && offset_of!(SampleArgsRepr, temperature) == 44
        && offset_of!(SampleArgsRepr, top_k) == 48
        && offset_of!(SampleArgsRepr, top_p) == 52
        && offset_of!(SampleArgsRepr, min_p) == 56
        && offset_of!(SampleArgsRepr, seed_lo) == 60
        && offset_of!(SampleArgsRepr, seed_hi) == 64
        && offset_of!(SampleArgsRepr, pos_lo) == 68
        && offset_of!(SampleArgsRepr, pos_hi) == 72
);

pub(in crate::kernels::cuda) struct SampleRaw<'a> {
    pub(in crate::kernels::cuda) token: RawRange<'a>,
    pub(in crate::kernels::cuda) logprob: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) probs: RawRange<'a>,
    pub(in crate::kernels::cuda) logits: RawRange<'a>,
    pub(in crate::kernels::cuda) n: usize,
    pub(in crate::kernels::cuda) temperature: f32,
    pub(in crate::kernels::cuda) top_k: usize,
    pub(in crate::kernels::cuda) top_p: f32,
    pub(in crate::kernels::cuda) min_p: f32,
    pub(in crate::kernels::cuda) seed: u64,
    pub(in crate::kernels::cuda) position: u64,
}

fn check(call: &SampleRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.n.checked_mul(4).ok_or_else(invalid_argument)?;
    let sized = call.probs.bytes >= bytes && call.logits.bytes >= bytes;
    let small = |r: &RawRange<'_>| r.bytes >= 4 && r.offset.is_multiple_of(4);
    let aligned = call.probs.offset.is_multiple_of(4) && call.logits.offset.is_multiple_of(4);
    let logits = Span::of(&call.logits, bytes);
    let probs = Span::of(&call.probs, bytes);
    let outputs = [&call.token, &call.logprob, &call.status];
    let mut apart = probs.apart(&logits);
    for (index, range) in outputs.iter().enumerate() {
        let out = Span::of(range, 4);
        apart = apart && out.apart(&logits) && out.apart(&probs);
        for other in outputs.iter().skip(index + 1) {
            apart = apart && out.apart(&Span::of(other, 4));
        }
    }
    let params = (1..=4_194_304).contains(&call.n)
        && call.temperature.is_finite()
        && call.top_k <= call.n
        && call.top_p > 0.0
        && call.top_p <= 1.0
        && call.min_p >= 0.0
        && call.min_p < 1.0;
    if !(sized
        && aligned
        && small(&call.token)
        && small(&call.logprob)
        && small(&call.status)
        && apart
        && params)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn sample_launch(
    stream: &RawStream,
    call: &SampleRaw<'_>,
) -> Result<(), DeviceFailure> {
    check(call)?;
    let args = SampleArgsRepr {
        token: call.token.address(),
        logprob: call.logprob.address(),
        status: call.status.address(),
        probs: call.probs.address(),
        logits: call.logits.address().cast_const(),
        n: c_int::try_from(call.n).map_err(|_| invalid_argument())?,
        temperature: call.temperature,
        top_k: c_int::try_from(call.top_k).map_err(|_| invalid_argument())?,
        top_p: call.top_p,
        min_p: call.min_p,
        seed_lo: call.seed as u32,
        seed_hi: (call.seed >> 32) as u32,
        pos_lo: call.position as u32,
        pos_hi: (call.position >> 32) as u32,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check proved the extents and the parameter ranges; the adapter repeats them.
    let status = unsafe { sample_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Mirror of inference_cuda_sample_shortlist_args, field for field.
#[repr(C)]
struct SampleShortlistArgsRepr {
    token: *mut c_void,
    logprob: *mut c_void,
    status: *mut c_void,
    entries: *const c_void,
    flag: *const c_void,
    count: c_int,
    temperature: f32,
    top_k: c_int,
    top_p: f32,
    seed_lo: u32,
    seed_hi: u32,
    pos_lo: u32,
    pos_hi: u32,
}

const _: () = assert!(
    size_of::<SampleShortlistArgsRepr>() == 72
        && offset_of!(SampleShortlistArgsRepr, entries) == 24
        && offset_of!(SampleShortlistArgsRepr, flag) == 32
        && offset_of!(SampleShortlistArgsRepr, count) == 40
        && offset_of!(SampleShortlistArgsRepr, pos_hi) == 68
);

/// The most entries of a shortlist the draw reads.
pub(in crate::kernels::cuda) const SHORTLIST_MAX: usize = 64;

pub(in crate::kernels::cuda) struct SampleShortlistRaw<'a> {
    pub(in crate::kernels::cuda) token: RawRange<'a>,
    pub(in crate::kernels::cuda) logprob: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) entries: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) count: usize,
    pub(in crate::kernels::cuda) temperature: f32,
    pub(in crate::kernels::cuda) top_k: usize,
    pub(in crate::kernels::cuda) top_p: f32,
    pub(in crate::kernels::cuda) seed: u64,
    pub(in crate::kernels::cuda) position: u64,
}

fn check_shortlist(call: &SampleShortlistRaw<'_>) -> Result<(), DeviceFailure> {
    let bytes = call.count.checked_mul(8).ok_or_else(invalid_argument)?;
    let entries = Span::of(&call.entries, bytes);
    let flag = Span::of(&call.flag, 4);
    let outs = [&call.token, &call.logprob, &call.status];
    let mut apart = entries.apart(&flag);
    for (index, range) in outs.iter().enumerate() {
        let out = Span::of(range, 4);
        apart = apart && out.apart(&entries) && out.apart(&flag);
        for other in outs.iter().skip(index + 1) {
            apart = apart && out.apart(&Span::of(other, 4));
        }
    }
    let small = |r: &RawRange<'_>| r.bytes >= 4 && r.offset.is_multiple_of(4);
    let params = (1..=SHORTLIST_MAX).contains(&call.count)
        && call.temperature.is_finite()
        && call.top_k <= call.count
        && call.top_p > 0.0
        && call.top_p <= 1.0;
    if !(call.entries.bytes >= bytes
        && call.entries.offset.is_multiple_of(4)
        && small(&call.flag)
        && small(&call.token)
        && small(&call.logprob)
        && small(&call.status)
        && apart
        && params)
    {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn sample_shortlist_launch(
    stream: &RawStream,
    call: &SampleShortlistRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_shortlist(call)?;
    let args = SampleShortlistArgsRepr {
        token: call.token.address(),
        logprob: call.logprob.address(),
        status: call.status.address(),
        entries: call.entries.address().cast_const(),
        flag: call.flag.address().cast_const(),
        count: c_int::try_from(call.count).map_err(|_| invalid_argument())?,
        temperature: call.temperature,
        top_k: c_int::try_from(call.top_k).map_err(|_| invalid_argument())?,
        top_p: call.top_p,
        seed_lo: call.seed as u32,
        seed_hi: (call.seed >> 32) as u32,
        pos_lo: call.position as u32,
        pos_hi: (call.position >> 32) as u32,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: check_shortlist proved the extents and the parameter ranges; the adapter repeats them.
    let status = unsafe { sample_shortlist_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
