//! Raw FFI to the split-KV decode attention entry of `native/resident/attn_decode_loop.cu`
//! (declared in `native/resident/attn_decode_loop.h`), the SenseNova Loop geometry: 4 query
//! heads per kv head, head_dim 256, linear fp16 cache. Child module of `native/resident.rs`,
//! so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on (as `attn_decode.rs`):
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard.
//! - Every range is a `RawRange`, proved inside its live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here before the call. The C entry checks
//!   pointer/scalar/alignment conditions, not allocation extents or overlap.
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_attn_decode_loop_args` in `attn_decode_loop.h`, field for field.
#[repr(C)]
struct AttnDecodeLoopArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    kv_len: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    ld: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    cache_rows: c_int,
    max_splits: c_int,
    min_keys: c_int,
    scale: f32,
}

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_decode_loop_part_floats"]
    fn attn_decode_loop_part_floats_raw(m: c_int, n_kv_head: c_int, max_splits: c_int) -> usize;
    #[link_name = "inference_cuda_attn_decode_loop_launch"]
    fn attn_decode_loop_launch_raw(
        args: *const AttnDecodeLoopArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure. `ld` is 0 without `proj`.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnLoopInts {
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) cache_rows: c_int,
    pub(in crate::kernels::cuda) max_splits: c_int,
    pub(in crate::kernels::cuda) min_keys: c_int,
}

pub(in crate::kernels::cuda) struct AttnDecodeLoopRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) kv_len: RawRange<'a>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: AttnLoopInts,
    pub(in crate::kernels::cuda) scale: f32,
}

/// Part scratch of one launch, in floats: the C rule pinned from the Rust side.
pub(in crate::kernels::cuda) fn attn_loop_part_floats(
    ints: &AttnLoopInts,
) -> Result<usize, DeviceFailure> {
    // SAFETY: a pure function of its three integers; it reads no memory and needs no device.
    Ok(unsafe { attn_decode_loop_part_floats_raw(ints.m, ints.n_kv_head, ints.max_splits) })
}

fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

fn product(factors: &[usize]) -> Result<usize, DeviceFailure> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(invalid_argument)
}

const DIM: usize = 256;
const GROUP: usize = 4;

fn needs(io: &AttnDecodeLoopRaw<'_>) -> Result<(usize, usize, usize, usize), DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (ld, rows) = (size(ints.ld)?, size(ints.cache_rows)?);
    let part_floats = attn_loop_part_floats(ints)?;
    let row_floats = product(&[heads, DIM])?;
    let proj = if io.proj.is_some() {
        let last_row = product(&[m.saturating_sub(1), ld, 4])?;
        last_row
            .checked_add(product(&[row_floats, 2, 4])?)
            .ok_or_else(invalid_argument)?
    } else {
        0
    };
    Ok((
        product(&[m, row_floats, 4])?,
        proj,
        product(&[rows, kv_heads, DIM, 2])?,
        product(&[part_floats, 4])?,
    ))
}

/// Queues the two attention kernels (partials, then the fixed-order combine) on `stream`.
pub(in crate::kernels::cuda) fn attn_decode_loop_launch(
    stream: &RawStream,
    io: AttnDecodeLoopRaw<'_>,
) -> Result<(), DeviceFailure> {
    let ints = io.ints;
    if ints.n_head
        != ints
            .n_kv_head
            .checked_mul(GROUP as c_int)
            .ok_or_else(invalid_argument)?
    {
        return Err(invalid_argument());
    }
    let (q, proj_need, cache, part) = needs(&io)?;
    let out_bytes = product(&[size(ints.m)?, size(ints.n_head)?, DIM, 4])?;
    let sized = io.q.bytes >= q
        && io.k_cache.bytes >= cache
        && io.v_cache.bytes >= cache
        && io.kv_len.bytes >= 4
        && io.part.bytes >= part
        && io.out.bytes >= out_bytes
        && io.proj.as_ref().is_none_or(|proj| proj.bytes >= proj_need);
    let aligned = io.q.offset.is_multiple_of(16)
        && io.k_cache.offset.is_multiple_of(16)
        && io.v_cache.offset.is_multiple_of(16)
        && io.kv_len.offset.is_multiple_of(4)
        && io.part.offset.is_multiple_of(16)
        && io.out.offset.is_multiple_of(4)
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.offset.is_multiple_of(4));
    let out = Span::of(&io.out, out_bytes);
    let part_span = Span::of(&io.part, part);
    let mut reads = vec![
        Span::of(&io.q, q),
        Span::of(&io.k_cache, cache),
        Span::of(&io.v_cache, cache),
        Span::of(&io.kv_len, 4),
    ];
    if let Some(proj) = &io.proj {
        reads.push(Span::of(proj, proj_need));
    }
    let apart = out.apart(&part_span)
        && reads
            .iter()
            .all(|read| out.apart(read) && part_span.apart(read));
    if !(sized && aligned && apart) {
        return Err(invalid_argument());
    }
    let args = AttnDecodeLoopArgsRaw {
        q: io.q.address().cast_const(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        k_cache: io.k_cache.address().cast_const(),
        v_cache: io.v_cache.address().cast_const(),
        kv_len: io.kv_len.address().cast_const(),
        part: io.part.address(),
        out: io.out.address(),
        ld: ints.ld,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        cache_rows: ints.cache_rows,
        max_splits: ints.max_splits,
        min_keys: ints.min_keys,
        scale: io.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the claims above proved every range inside its live allocation, large enough for
    // the call, aligned, and `out` and `part` disjoint from every other range. The C entry checks
    // pointer/scalar/alignment conditions, not allocation extents or overlap. Allocations stay
    // valid until work finishes because `inference_cuda_free` synchronizes the device first.
    // `args` lives across the call and the entry copies it before returning. `stream` is live
    // for the call.
    let status =
        unsafe { attn_decode_loop_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
