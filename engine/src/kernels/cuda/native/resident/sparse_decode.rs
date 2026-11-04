//! Raw FFI to the sparse-attention decode entry of `native/resident/sparse_decode.cu`
//! (declared in `native/resident/sparse_decode.h`): attention over the SELECTED cells only,
//! with the exact Q8 arithmetic of `attn_decode_q8.cu`. Child module of `native/resident.rs`,
//! so it inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. It only enqueues.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness (`super::Span`) are proved again here, on the extents the kernel touches,
//!   before the call; the C entry repeats the alignment and shape checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes the
//!   device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.
use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_sparse_decode_args` in `sparse_decode.h`, field for field.
#[repr(C)]
struct SparseDecodeArgsRaw {
    q: *const c_void,
    k_codes: *const c_void,
    k_scales: *const c_void,
    v_codes: *const c_void,
    v_scales: *const c_void,
    indices: *const c_void,
    counts: *const c_void,
    out: *mut c_void,
    proj: *const c_void,
    status: *mut c_void,
    n_kv_head: c_int,
    n_head: c_int,
    head_dim: c_int,
    cells: c_int,
    row: c_int,
    scale: f32,
}

const _: () = assert!(
    size_of::<SparseDecodeArgsRaw>() == 104
        && align_of::<SparseDecodeArgsRaw>() == 8
        && offset_of!(SparseDecodeArgsRaw, q) == 0
        && offset_of!(SparseDecodeArgsRaw, k_codes) == 8
        && offset_of!(SparseDecodeArgsRaw, k_scales) == 16
        && offset_of!(SparseDecodeArgsRaw, v_codes) == 24
        && offset_of!(SparseDecodeArgsRaw, v_scales) == 32
        && offset_of!(SparseDecodeArgsRaw, indices) == 40
        && offset_of!(SparseDecodeArgsRaw, counts) == 48
        && offset_of!(SparseDecodeArgsRaw, out) == 56
        && offset_of!(SparseDecodeArgsRaw, proj) == 64
        && offset_of!(SparseDecodeArgsRaw, status) == 72
        && offset_of!(SparseDecodeArgsRaw, n_kv_head) == 80
);

unsafe extern "C" {
    #[link_name = "inference_cuda_sparse_decode_launch"]
    fn sparse_decode_launch_raw(
        args: *const SparseDecodeArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct SparseDecodeInts {
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
    pub(in crate::kernels::cuda) row: c_int,
    pub(in crate::kernels::cuda) scale: f32,
}

pub(in crate::kernels::cuda) struct SparseDecodeDataRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) k_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) k_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) v_codes: RawRange<'a>,
    pub(in crate::kernels::cuda) v_scales: RawRange<'a>,
    pub(in crate::kernels::cuda) indices: RawRange<'a>,
    pub(in crate::kernels::cuda) counts: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    /// Optional sigmoid-gate range `None` (raw NULL) or `[n_head][2 * head_dim]` f32.
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: SparseDecodeInts,
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

/// The byte extent the kernel touches per buffer on the extents declared by the wrapper.
pub(in crate::kernels::cuda) struct SparseDecodeExtents {
    pub(in crate::kernels::cuda) codes: usize,
    pub(in crate::kernels::cuda) scales: usize,
    pub(in crate::kernels::cuda) indices: usize,
    pub(in crate::kernels::cuda) q_out: usize,
    /// The optional gate span: n_head * 2 * head_dim f32.
    pub(in crate::kernels::cuda) gate: usize,
}

pub(in crate::kernels::cuda) fn sparse_decode_extents(
    ints: &SparseDecodeInts,
) -> Result<SparseDecodeExtents, DeviceFailure> {
    let (kv_heads, dim, cells, row, heads) = (
        size(ints.n_kv_head)?,
        size(ints.head_dim)?,
        size(ints.cells)?,
        size(ints.row)?,
        size(ints.n_head)?,
    );
    Ok(SparseDecodeExtents {
        codes: product(&[kv_heads, cells, dim])?,
        scales: product(&[kv_heads, cells, dim / 32, 2])?,
        indices: product(&[kv_heads, row, 4])?,
        q_out: product(&[heads, dim, 4])?,
        gate: product(&[heads, 2, dim, 4])?,
    })
}

/// Alignment per buffer: q and out carry float4 reads/writes (16), the int8 code planes pack
/// int words (4), the fp16-bit scale planes read u16 (2), indices/counts/status are u32/i32.
pub(in crate::kernels::cuda) fn sparse_decode_launch(
    stream: &RawStream,
    io: SparseDecodeDataRaw<'_>,
) -> Result<(), DeviceFailure> {
    let e = sparse_decode_extents(&io.ints)?;
    let fits = |range: &RawRange<'_>, need: usize, align: usize| {
        range.bytes >= need && range.offset.is_multiple_of(align)
    };
    if !(fits(&io.q, e.q_out, 16)
        && fits(&io.k_codes, e.codes, 4)
        && fits(&io.k_scales, e.scales, 2)
        && fits(&io.v_codes, e.codes, 4)
        && fits(&io.v_scales, e.scales, 2)
        && fits(&io.indices, e.indices, 4)
        && fits(&io.counts, size(io.ints.n_kv_head)? * 4, 4)
        && fits(&io.out, e.q_out, 16)
        && io.proj.as_ref().is_none_or(|proj| fits(proj, e.gate, 4))
        && fits(&io.status, 4, 4))
    {
        return Err(invalid_argument());
    }
    // `out` and `status` are the written ranges; each must be apart from every other range of
    // the call. The reads (q, the KV planes, indices, counts) may overlap each other.
    let written = [Span::of(&io.out, e.q_out), Span::of(&io.status, 4)];
    let mut read = vec![
        Span::of(&io.q, e.q_out),
        Span::of(&io.k_codes, e.codes),
        Span::of(&io.k_scales, e.scales),
        Span::of(&io.v_codes, e.codes),
        Span::of(&io.v_scales, e.scales),
        Span::of(&io.indices, e.indices),
        Span::of(&io.counts, size(io.ints.n_kv_head)? * 4),
    ];
    if let Some(proj) = io.proj.as_ref() {
        read.push(Span::of(proj, e.gate));
    }
    for (i, a) in written.iter().enumerate() {
        let apart = written[i + 1..]
            .iter()
            .chain(read.iter())
            .all(|b| a.apart(b));
        if !apart {
            return Err(invalid_argument());
        }
    }
    let ints = io.ints;
    let args = SparseDecodeArgsRaw {
        q: io.q.address().cast_const(),
        k_codes: io.k_codes.address().cast_const(),
        k_scales: io.k_scales.address().cast_const(),
        v_codes: io.v_codes.address().cast_const(),
        v_scales: io.v_scales.address().cast_const(),
        indices: io.indices.address().cast_const(),
        counts: io.counts.address().cast_const(),
        out: io.out.address(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |p| p.address().cast_const()),
        status: io.status.address(),
        n_kv_head: ints.n_kv_head,
        n_head: ints.n_head,
        head_dim: ints.head_dim,
        cells: ints.cells,
        row: ints.row,
        scale: ints.scale,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: (a) the fit and apart checks above proved every range inside its live allocation,
    // large enough, correctly aligned, and the written `out` extent and `status` word disjoint
    // from every other range, so each pointer in `args` addresses bytes the kernel may touch;
    // the C entry repeats the checks on the addresses. (b) The allocations outlive the queued
    // work because `inference_cuda_free` synchronizes the device before `cudaFree`. (c) `stream`
    // is borrowed from a live `RawStream`. (d) `args` is a live local that the entry reads only
    // during the call. (e) The dispatch guard is held by `launch_on`, the only caller.
    let status =
        unsafe { sparse_decode_launch_raw(&raw const args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::SparseDecodeArgsRaw;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn sparse_decode_argument_layout_pins_the_c_struct() {
        assert_eq!(size_of::<SparseDecodeArgsRaw>(), 104);
        assert_eq!(offset_of!(SparseDecodeArgsRaw, q), 0);
        assert_eq!(offset_of!(SparseDecodeArgsRaw, indices), 40);
        assert_eq!(offset_of!(SparseDecodeArgsRaw, proj), 64);
        assert_eq!(offset_of!(SparseDecodeArgsRaw, status), 72);
        assert_eq!(offset_of!(SparseDecodeArgsRaw, n_kv_head), 80);
        assert_eq!(align_of::<SparseDecodeArgsRaw>(), 8);
    }
}
