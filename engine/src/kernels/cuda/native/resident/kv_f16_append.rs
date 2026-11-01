//! Raw FFI to the standalone fp16 KV append entry of `native/resident/kv_f16_append.cu`
//! (declared in `native/resident/kv_f16_append.h`). Child module of `native/resident.rs`, so it
//! inherits the `allow(unsafe_code)` of `native.rs`.
//!
//! The kernel is the store arm of the vendored attn_prep on its own launch: the gathered,
//! post-rope k rows and post-norm v rows of one step, f32 `[m][n_kv_head][head_dim]` each,
//! converted with `__float2half` into the fused kernel's fp16 plane layout
//! `[cells][n_kv_head][head_dim]`, row `*pos + t`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that owns
//!   the context and under the dispatch guard. It only enqueues.
//! - Every range is a `RawRange`, proved inside a live allocation. Sizes, alignment and
//!   disjointness are proved again here, on the extents the kernel touches, before the call;
//!   the C entry repeats the alignment and shape checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declarations, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

use std::ffi::{c_int, c_void};
use std::mem::{align_of, offset_of, size_of};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

/// Mirror of `inference_cuda_kv_f16_append_args` in `kv_f16_append.h`, field for field.
#[repr(C)]
struct KvF16AppendArgsRaw {
    k: *const c_void,
    v: *const c_void,
    k_cache: *mut c_void,
    v_cache: *mut c_void,
    pos: *const c_void,
    status: *mut c_void,
    m: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cells: c_int,
}

const _: () = assert!(
    size_of::<KvF16AppendArgsRaw>() == 64
        && align_of::<KvF16AppendArgsRaw>() == 8
        && offset_of!(KvF16AppendArgsRaw, k_cache) == 16
        && offset_of!(KvF16AppendArgsRaw, pos) == 32
        && offset_of!(KvF16AppendArgsRaw, status) == 40
        && offset_of!(KvF16AppendArgsRaw, m) == 48
        && offset_of!(KvF16AppendArgsRaw, cells) == 60
);

unsafe extern "C" {
    #[link_name = "inference_cuda_kv_f16_append_launch"]
    fn kv_f16_append_launch_raw(
        args: *const KvF16AppendArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Every integer argument of the launch, already bounded and converted by the wrapper before
/// `launch_on`, so no conversion can fail inside the launch closure.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct KvF16Ints {
    pub(in crate::kernels::cuda) rows: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) cells: c_int,
}

/// The ranges of the entry: the f32 k and v rows in, the two fp16 caches out, and the device
/// words (the position of token 0, read; the status word, mask 4 OR-ed in on a refused launch).
pub(in crate::kernels::cuda) struct KvF16DataRaw<'a> {
    pub(in crate::kernels::cuda) k: RawRange<'a>,
    pub(in crate::kernels::cuda) v: RawRange<'a>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) pos: RawRange<'a>,
    pub(in crate::kernels::cuda) status: RawRange<'a>,
    pub(in crate::kernels::cuda) ints: KvF16Ints,
}

/// `n` as `usize`, or an invalid-argument failure for a negative value.
fn size(n: c_int) -> Result<usize, DeviceFailure> {
    usize::try_from(n).map_err(|_| invalid_argument())
}

/// Product of the factors, or an invalid-argument failure on overflow.
fn product(factors: &[usize]) -> Result<usize, DeviceFailure> {
    factors
        .iter()
        .try_fold(1usize, |acc, &factor| acc.checked_mul(factor))
        .ok_or_else(invalid_argument)
}

/// The bytes each data range must hold.
struct Needs {
    rows: usize,
    cache: usize,
}

fn needs(ints: &KvF16Ints) -> Result<Needs, DeviceFailure> {
    let (rows, kv_heads, dim) = (
        size(ints.rows)?,
        size(ints.n_kv_head)?,
        size(ints.head_dim)?,
    );
    let cells = size(ints.cells)?;
    let values = product(&[rows, kv_heads, dim])?;
    Ok(Needs {
        rows: product(&[values, 4])?,
        cache: product(&[cells, kv_heads, dim, 2])?,
    })
}

/// Proves every range of the entry: sizes (the f32 rows, the fp16 caches, the two 4-byte words),
/// alignments (4 for the f32 reads and the words, 16 for the caches, as attn_prep requires of
/// them), and that the three written ranges (both caches, the status word) share no byte with
/// each other or with the three read ranges. The two read rows may overlap each other, and the
/// two words may alias the rows, exactly as the other append entries allow.
fn check(io: &KvF16DataRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(&io.ints)?;
    let fits4 =
        |range: &RawRange<'_>, bytes: usize| range.bytes >= bytes && range.offset.is_multiple_of(4);
    if !(fits4(&io.k, need.rows)
        && fits4(&io.v, need.rows)
        && fits4(&io.pos, 4)
        && fits4(&io.status, 4))
    {
        return Err(invalid_argument());
    }
    for cache in [&io.k_cache, &io.v_cache] {
        if cache.bytes < need.cache || !cache.offset.is_multiple_of(16) {
            return Err(invalid_argument());
        }
    }
    let written = [
        Span::of(&io.k_cache, need.cache),
        Span::of(&io.v_cache, need.cache),
        Span::of(&io.status, 4),
    ];
    let read = [Span::of(&io.k, need.rows), Span::of(&io.v, need.rows)];
    for (i, a) in written.iter().enumerate() {
        let apart_from_written = written[i + 1..].iter().all(|b| a.apart(b));
        if !(apart_from_written && read.iter().all(|b| a.apart(b))) {
            return Err(invalid_argument());
        }
    }
    Ok(())
}

/// Queues the fp16 append (m rows at the device position word) on `stream`. Returns when it is
/// queued.
pub(in crate::kernels::cuda) fn kv_f16_append_launch(
    stream: &RawStream,
    io: KvF16DataRaw<'_>,
) -> Result<(), DeviceFailure> {
    check(&io)?;
    let ints = io.ints;
    let args = KvF16AppendArgsRaw {
        k: io.k.address().cast_const(),
        v: io.v.address().cast_const(),
        k_cache: io.k_cache.address(),
        v_cache: io.v_cache.address(),
        pos: io.pos.address().cast_const(),
        status: io.status.address(),
        m: ints.rows,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cells: ints.cells,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: `check` proved every extent, alignment and disjointness the C entry re-checks on
    // the addresses, and `RawRange` proved the extents live inside their allocations. The
    // allocations outlive the queued work (`inference_cuda_free` synchronizes first). `stream`
    // is live for the call and the launch only enqueues.
    let status = unsafe { kv_f16_append_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}
