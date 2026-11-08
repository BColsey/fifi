//! Raw FFI to the chunked-prefill attention entry of `native/prefill/attn_prefill.cu` (declared in
//! `native/prefill/attn_prefill.h`). Child module of `native/prefill.rs`, so it inherits the
//! `allow(unsafe_code)` of `native.rs`.
//!
//! Safety invariants this file relies on:
//! - The launch entry is called only through `CudaContext::launch_on`, on the thread that
//!   owns the context and under the dispatch guard. It only enqueues the selected phase(s)
//!   and returns; the stream is live for the call.
//! - Every range is a `RawRange`, proved inside a live allocation by `raw_range`. Sizes,
//!   alignment and disjointness (`super::Span`) are proved again here, on the extents the
//!   kernels touch, before the call; the C entry repeats the pointer, alignment and shape
//!   checks on the addresses.
//! - The C argument struct is built in one `unsafe` block that cites those checks.
//! - A device allocation outlives the queued work because `inference_cuda_free` synchronizes
//!   the device before `cudaFree`.
//! - The `extern` declaration, the `repr(C)` mirror and every raw pointer are private to this
//!   file. Nothing is re-exported.

#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the chunked-prefill driver")
)]

use core::mem::{align_of, offset_of, size_of};
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};
use crate::kernels::cuda::prefill::attn_prefill::{AttnKvLayout, AttnKvStorage, AttnScalarLayout};
use crate::tensor::Dtype;

/// The fp16 KV mode of `attn_prefill.h` (the KV8 loader arrives with the KV8 cache path).
pub(in crate::kernels::cuda) const KV_FP16: c_int = 0;

/// Shared tagged-plane ABI; only Q8 affine is submitted through this carrier.
#[repr(C)]
struct AttnKvViewRaw {
    payload: *const c_void,
    scales: *const c_void,
    bias: *const c_void,
    payload_time_stride_bytes: usize,
    payload_head_stride_bytes: usize,
    scale_time_stride_bytes: usize,
    scale_head_stride_bytes: usize,
    scale_group_stride_bytes: usize,
    bias_time_stride_bytes: usize,
    bias_head_stride_bytes: usize,
    bias_group_stride_bytes: usize,
    storage: c_int,
    group_size: c_int,
    scale_dtype: c_int,
    bias_dtype: c_int,
    q8_signed: c_int,
    q4_low_even: c_int,
}

#[repr(C)]
struct AttnKvPairRaw {
    k: AttnKvViewRaw,
    v: AttnKvViewRaw,
}

const _: () = assert!(
    size_of::<AttnKvViewRaw>() == 112
        && align_of::<AttnKvViewRaw>() == 8
        && offset_of!(AttnKvViewRaw, payload) == 0
        && offset_of!(AttnKvViewRaw, scales) == 8
        && offset_of!(AttnKvViewRaw, bias) == 16
        && offset_of!(AttnKvViewRaw, payload_time_stride_bytes) == 24
        && offset_of!(AttnKvViewRaw, payload_head_stride_bytes) == 32
        && offset_of!(AttnKvViewRaw, scale_time_stride_bytes) == 40
        && offset_of!(AttnKvViewRaw, scale_head_stride_bytes) == 48
        && offset_of!(AttnKvViewRaw, scale_group_stride_bytes) == 56
        && offset_of!(AttnKvViewRaw, bias_time_stride_bytes) == 64
        && offset_of!(AttnKvViewRaw, bias_head_stride_bytes) == 72
        && offset_of!(AttnKvViewRaw, bias_group_stride_bytes) == 80
        && offset_of!(AttnKvViewRaw, storage) == 88
        && offset_of!(AttnKvViewRaw, group_size) == 92
        && offset_of!(AttnKvViewRaw, scale_dtype) == 96
        && offset_of!(AttnKvViewRaw, bias_dtype) == 100
        && offset_of!(AttnKvViewRaw, q8_signed) == 104
        && offset_of!(AttnKvViewRaw, q4_low_even) == 108
        && size_of::<AttnKvPairRaw>() == 224
        && align_of::<AttnKvPairRaw>() == 8
        && offset_of!(AttnKvPairRaw, k) == 0
        && offset_of!(AttnKvPairRaw, v) == 112
);

/// Mirror of `inference_cuda_attn_prefill_args` in `attn_prefill.h`, field for field.
#[repr(C)]
struct AttnPrefillArgsRaw {
    q: *const c_void,
    proj: *const c_void,
    k_cache: *const c_void,
    v_cache: *const c_void,
    kv8_k_codes: *const c_void,
    kv8_k_scales: *const c_void,
    kv8_v_codes: *const c_void,
    kv8_v_scales: *const c_void,
    part: *mut c_void,
    out: *mut c_void,
    status: *mut c_void,
    ld: c_int,
    p0: c_int,
    m: c_int,
    n_head: c_int,
    n_kv_head: c_int,
    head_dim: c_int,
    cache_rows: c_int,
    max_splits: c_int,
    min_keys: c_int,
    window: c_int,
    ring: c_int,
    kv_mode: c_int,
    kv8_capacity: c_int,
    scale: f32,
    sinks: *const c_void,
    kv_views: *const AttnKvPairRaw,
}

const _: () = assert!(
    size_of::<AttnPrefillArgsRaw>() == 160
        && align_of::<AttnPrefillArgsRaw>() == 8
        && offset_of!(AttnPrefillArgsRaw, q) == 0
        && offset_of!(AttnPrefillArgsRaw, proj) == 8
        && offset_of!(AttnPrefillArgsRaw, k_cache) == 16
        && offset_of!(AttnPrefillArgsRaw, v_cache) == 24
        && offset_of!(AttnPrefillArgsRaw, kv8_k_codes) == 32
        && offset_of!(AttnPrefillArgsRaw, kv8_k_scales) == 40
        && offset_of!(AttnPrefillArgsRaw, kv8_v_codes) == 48
        && offset_of!(AttnPrefillArgsRaw, kv8_v_scales) == 56
        && offset_of!(AttnPrefillArgsRaw, part) == 64
        && offset_of!(AttnPrefillArgsRaw, out) == 72
        && offset_of!(AttnPrefillArgsRaw, status) == 80
        && offset_of!(AttnPrefillArgsRaw, ld) == 88
        && offset_of!(AttnPrefillArgsRaw, p0) == 92
        && offset_of!(AttnPrefillArgsRaw, m) == 96
        && offset_of!(AttnPrefillArgsRaw, n_head) == 100
        && offset_of!(AttnPrefillArgsRaw, n_kv_head) == 104
        && offset_of!(AttnPrefillArgsRaw, head_dim) == 108
        && offset_of!(AttnPrefillArgsRaw, cache_rows) == 112
        && offset_of!(AttnPrefillArgsRaw, max_splits) == 116
        && offset_of!(AttnPrefillArgsRaw, min_keys) == 120
        && offset_of!(AttnPrefillArgsRaw, window) == 124
        && offset_of!(AttnPrefillArgsRaw, ring) == 128
        && offset_of!(AttnPrefillArgsRaw, kv_mode) == 132
        && offset_of!(AttnPrefillArgsRaw, kv8_capacity) == 136
        && offset_of!(AttnPrefillArgsRaw, scale) == 140
        && offset_of!(AttnPrefillArgsRaw, sinks) == 144
        && offset_of!(AttnPrefillArgsRaw, kv_views) == 152
);

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_part_floats"]
    fn attn_prefill_part_floats_raw(
        m: c_int,
        n_kv_head: c_int,
        head_dim: c_int,
        max_splits: c_int,
        group: c_int,
    ) -> usize;
    #[link_name = "inference_cuda_attn_prefill_launch"]
    fn attn_prefill_launch_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_prefill_part_launch"]
    fn attn_prefill_part_launch_raw(
        args: *const AttnPrefillArgsRaw,
        key_origin: c_int,
        key_rows: c_int,
        part_index: c_int,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_attn_prefill_combine_launch"]
    fn attn_prefill_combine_launch_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) enum AttnPrefillLaunch {
    Full,
    Part {
        key_origin: c_int,
        key_rows: c_int,
        part_index: c_int,
    },
    #[cfg(all(test, feature = "cuda-q8-reference"))]
    /// Same-ELF fixture path: identical admission and args, but the
    /// native entry launches the q8 part kernel with the pre-hoist per-element math so
    /// tests can compare it with the hoisted kernel byte for byte. Fixture builds only
    /// (feature `cuda-q8-reference`).
    Q8Reference,
    #[cfg(all(test, feature = "cuda-q8-reference"))]
    /// Same-ELF fixture path: identical admission and args, but the native entry passes
    /// the tiled route — one of the two ways tests reach the tiled kernel (the
    /// default-ON production route entry is the other). Fixture builds only
    /// (feature `cuda-q8-reference`).
    Q8Tiled,
    /// Production route: the entry the Rust driver calls
    /// when FIFI_FLINT_CUDA_Q8_TILED is armed; `use_tiled = false` is the same-entry
    /// old-path form the route test exercises. Predicate rejects fall back inside the
    /// entry and count into [`Q8_TILED_FALLBACKS`].
    Q8TiledRoute {
        use_tiled: bool,
    },
    Combine,
}

#[cfg(all(test, feature = "cuda-fault-qualification"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_fault_bad_bounds_launch"]
    fn attn_prefill_fault_bad_bounds_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

#[cfg(all(test, feature = "cuda-q8-reference"))]
unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_q8_reference_launch"]
    fn attn_prefill_q8_reference_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;

    #[link_name = "inference_cuda_attn_prefill_q8_tiled_launch"]
    fn attn_prefill_q8_tiled_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

unsafe extern "C" {
    #[link_name = "inference_cuda_attn_prefill_q8_tiled_route"]
    fn attn_prefill_q8_tiled_route_raw(
        args: *const AttnPrefillArgsRaw,
        stream: *mut c_void,
        out_stage: *mut c_int,
        use_tiled: c_int,
        out_route: *mut c_int,
    ) -> c_int;
}

/// Fallbacks counted by the production route entry: tiled requested but the predicate
/// rejected the shape, so the default old kernel ran. Test-visible for the
/// mutation-checked route test; production code only increments, and the route logs
/// the first fallback per process to stderr so silent fallbacks are visible
/// without a fixture build.
pub(in crate::kernels::cuda) static Q8_TILED_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Route-acceptance count: successful tile-kernel launches through the route entry.
/// Test-visible; kept live in production builds as telemetry on the same pattern as
/// the fallback counter above: one relaxed add per accepted call on the tiled
/// route (default ON, rollback via a rejected knob value).
pub(in crate::kernels::cuda) static Q8_TILED_ROUTES: AtomicU64 = AtomicU64::new(0);

/// `proj`.
#[derive(Debug, Clone, Copy)]
pub(in crate::kernels::cuda) struct AttnPrefillInts {
    pub(in crate::kernels::cuda) ld: c_int,
    pub(in crate::kernels::cuda) p0: c_int,
    pub(in crate::kernels::cuda) m: c_int,
    pub(in crate::kernels::cuda) n_head: c_int,
    pub(in crate::kernels::cuda) n_kv_head: c_int,
    pub(in crate::kernels::cuda) head_dim: c_int,
    pub(in crate::kernels::cuda) cache_rows: c_int,
    pub(in crate::kernels::cuda) max_splits: c_int,
    pub(in crate::kernels::cuda) min_keys: c_int,
    pub(in crate::kernels::cuda) window: c_int,
    pub(in crate::kernels::cuda) ring: c_int,
}

pub(in crate::kernels::cuda) struct AttnKvRaw<'a> {
    pub(in crate::kernels::cuda) layout: AttnKvLayout,
    pub(in crate::kernels::cuda) payload: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) bias: Option<RawRange<'a>>,
}

/// `proj` is the output gate.
pub(in crate::kernels::cuda) struct AttnPrefillRaw<'a> {
    pub(in crate::kernels::cuda) q: RawRange<'a>,
    pub(in crate::kernels::cuda) proj: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) k_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) v_cache: RawRange<'a>,
    pub(in crate::kernels::cuda) kv_views: Option<[AttnKvRaw<'a>; 2]>,
    pub(in crate::kernels::cuda) part: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) status: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) sinks: Option<RawRange<'a>>,
    pub(in crate::kernels::cuda) ints: AttnPrefillInts,
    pub(in crate::kernels::cuda) scale: f32,
}

struct Needs {
    q: usize,
    proj: usize,
    sinks: usize,
    cache: [usize; 2],
    part: usize,
    out: usize,
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

fn needs(io: &AttnPrefillRaw<'_>) -> Result<Needs, DeviceFailure> {
    let ints = &io.ints;
    let (m, heads, kv_heads) = (size(ints.m)?, size(ints.n_head)?, size(ints.n_kv_head)?);
    let (dim, ld, rows) = (size(ints.head_dim)?, size(ints.ld)?, size(ints.cache_rows)?);
    let group = heads
        .checked_div(kv_heads)
        .filter(|_| heads.is_multiple_of(kv_heads))
        .ok_or_else(invalid_argument)?;
    let group = c_int::try_from(group).map_err(|_| invalid_argument())?;
    // SAFETY: a pure function of its five integers; it reads no memory and needs no device.
    let part_floats = unsafe {
        attn_prefill_part_floats_raw(
            ints.m,
            ints.n_kv_head,
            ints.head_dim,
            ints.max_splits,
            group,
        )
    };
    if part_floats == 0 {
        return Err(invalid_argument());
    }
    let row_floats = product(&[heads, dim])?;
    let proj = if io.proj.is_some() {
        let last_row = product(&[m.saturating_sub(1), ld, 4])?;
        last_row
            .checked_add(product(&[row_floats, 2, 4])?)
            .ok_or_else(invalid_argument)?
    } else {
        0
    };
    let cache = match &io.kv_views {
        Some(views) => {
            let extent = |v: &AttnKvRaw<'_>| {
                v.layout
                    .extents(rows, kv_heads, dim)
                    .map(|e| e.payload_bytes)
                    .map_err(|_| invalid_argument())
            };
            [extent(&views[0])?, extent(&views[1])?]
        }
        None => [product(&[rows, kv_heads, dim, 2])?; 2],
    };
    Ok(Needs {
        q: product(&[m, row_floats, 4])?,
        proj,
        sinks: product(&[heads, 4])?,
        cache,
        part: product(&[part_floats, 4])?,
        out: product(&[m, row_floats, 4])?,
    })
}

fn check_views(io: &AttnPrefillRaw<'_>, need: &Needs) -> Result<(), DeviceFailure> {
    let Some(views) = &io.kv_views else {
        return Ok(());
    };
    /* Uniform q8: both planes Q8Affine. Mixed ("q8k"): K stays Q8Affine, V is
     * the F16 row plane; the C entry's valid() admits exactly this pair. */
    if !matches!(views[0].layout.storage, AttnKvStorage::Q8Affine { .. })
        || !matches!(
            views[1].layout.storage,
            AttnKvStorage::Q8Affine { .. } | AttnKvStorage::F16
        )
    {
        return Err(invalid_argument());
    }
    let rows = size(io.ints.cache_rows)?;
    let heads = size(io.ints.n_kv_head)?;
    let dim = size(io.ints.head_dim)?;
    let out = Span::of(&io.out, need.out);
    let part = Span::of(&io.part, need.part);
    let status = io.status.as_ref().map(|r| Span::of(r, 4));
    if status.as_ref().is_some_and(|status| {
        !status.apart(&out)
            || !status.apart(&part)
            || !status.apart(&Span::of(&io.q, need.q))
            || io
                .proj
                .as_ref()
                .is_some_and(|r| !status.apart(&Span::of(r, need.proj)))
    }) {
        return Err(invalid_argument());
    }
    for (view, cache) in views.iter().zip([&io.k_cache, &io.v_cache]) {
        if view.payload.address() != cache.address() || view.payload.bytes() != cache.bytes() {
            return Err(invalid_argument());
        }
        let extents = view
            .layout
            .extents(rows, heads, dim)
            .map_err(|_| invalid_argument())?;
        let groups = match view.layout.storage {
            AttnKvStorage::Q8Affine { group_size, .. } => dim / group_size,
            /* The mixed V plane carries no scale or bias planes. */
            AttnKvStorage::F16 => 0,
            _ => return Err(invalid_argument()),
        };
        let apart = |range: &RawRange<'_>, bytes| {
            let read = Span::of(range, bytes);
            out.apart(&read) && part.apart(&read) && status.as_ref().is_none_or(|s| s.apart(&read))
        };
        if view.payload.bytes() < extents.payload_bytes
            || !apart(&view.payload, extents.payload_bytes)
        {
            return Err(invalid_argument());
        }
        for (layout, range, bytes) in [
            (
                view.layout.scales,
                view.scales.as_ref(),
                extents.scale_bytes,
            ),
            (view.layout.bias, view.bias.as_ref(), extents.bias_bytes),
        ] {
            match (layout, range, bytes) {
                (None, None, None) => {}
                (Some(layout), Some(range), Some(bytes)) => {
                    let align = usize::try_from(layout.dtype.byte_width())
                        .map_err(|_| invalid_argument())?;
                    if range.bytes() < bytes
                        || !range.offset().is_multiple_of(align)
                        || (rows > 1 && !layout.strides.time_stride_bytes.is_multiple_of(align))
                        || (heads > 1 && !layout.strides.head_stride_bytes.is_multiple_of(align))
                        || (groups > 1 && !layout.group_stride_bytes.is_multiple_of(align))
                        || !apart(range, bytes)
                    {
                        return Err(invalid_argument());
                    }
                }
                _ => return Err(invalid_argument()),
            }
        }
    }
    Ok(())
}

fn scalar_dtype(layout: Option<AttnScalarLayout>) -> Result<c_int, DeviceFailure> {
    match layout.map(|layout| layout.dtype) {
        None | Some(Dtype::F16) => Ok(0),
        Some(Dtype::Bf16) => Ok(1),
        Some(Dtype::F32) => Ok(2),
        _ => Err(invalid_argument()),
    }
}

fn view_args(view: &AttnKvRaw<'_>) -> Result<AttnKvViewRaw, DeviceFailure> {
    /* Uniform q8 planes and the mixed ("q8k") V plane both pass through here;
     * the raw struct is self-describing, so the C validators pick per view. */
    let (storage, group_size, signed) = match view.layout.storage {
        AttnKvStorage::Q8Affine { group_size, signed } => (
            3,
            c_int::try_from(group_size).map_err(|_| invalid_argument())?,
            signed,
        ),
        AttnKvStorage::F16 => (0, 0, false),
        _ => return Err(invalid_argument()),
    };
    let (scale, bias) = (view.layout.scales, view.layout.bias);
    Ok(AttnKvViewRaw {
        payload: view.payload.address().cast_const(),
        scales: view
            .scales
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast_const()),
        bias: view
            .bias
            .as_ref()
            .map_or(std::ptr::null(), |r| r.address().cast_const()),
        payload_time_stride_bytes: view.layout.payload_strides.time_stride_bytes,
        payload_head_stride_bytes: view.layout.payload_strides.head_stride_bytes,
        scale_time_stride_bytes: scale.map_or(0, |s| s.strides.time_stride_bytes),
        scale_head_stride_bytes: scale.map_or(0, |s| s.strides.head_stride_bytes),
        scale_group_stride_bytes: scale.map_or(0, |s| s.group_stride_bytes),
        bias_time_stride_bytes: bias.map_or(0, |s| s.strides.time_stride_bytes),
        bias_head_stride_bytes: bias.map_or(0, |s| s.strides.head_stride_bytes),
        bias_group_stride_bytes: bias.map_or(0, |s| s.group_stride_bytes),
        storage,
        group_size,
        scale_dtype: scalar_dtype(scale)?,
        bias_dtype: scalar_dtype(bias)?,
        q8_signed: c_int::from(signed),
        q4_low_even: 0,
    })
}

// Keep this host pair alive across the C call; enqueue copies it into kernel parameters.
fn c_views(io: &AttnPrefillRaw<'_>) -> Result<Option<AttnKvPairRaw>, DeviceFailure> {
    match &io.kv_views {
        Some([k, v]) => Ok(Some(AttnKvPairRaw {
            k: view_args(k)?,
            v: view_args(v)?,
        })),
        None => Ok(None),
    }
}

fn check_attn(io: &AttnPrefillRaw<'_>) -> Result<(), DeviceFailure> {
    let need = needs(io)?;
    let sized = io.q.bytes() >= need.q
        && io.k_cache.bytes() >= need.cache[0]
        && io.v_cache.bytes() >= need.cache[1]
        && io.part.bytes() >= need.part
        && io.out.bytes() >= need.out
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.bytes() >= need.proj)
        && io.status.as_ref().is_none_or(|s| s.bytes() >= 4)
        && io.sinks.as_ref().is_none_or(|s| s.bytes() >= need.sinks);
    let cache_align = if io.kv_views.is_some() { 1 } else { 16 };
    let aligned = io.q.offset().is_multiple_of(16)
        && io.k_cache.offset().is_multiple_of(cache_align)
        && io.v_cache.offset().is_multiple_of(cache_align)
        && io.part.offset().is_multiple_of(16)
        && io.out.offset().is_multiple_of(16)
        && io
            .proj
            .as_ref()
            .is_none_or(|proj| proj.offset().is_multiple_of(4))
        && io
            .status
            .as_ref()
            .is_none_or(|s| s.offset().is_multiple_of(4))
        && io
            .sinks
            .as_ref()
            .is_none_or(|s| s.offset().is_multiple_of(4));
    let out = Span::of(&io.out, need.out);
    let part = Span::of(&io.part, need.part);
    let reads = [
        Some(Span::of(&io.q, need.q)),
        Some(Span::of(&io.k_cache, need.cache[0])),
        Some(Span::of(&io.v_cache, need.cache[1])),
        io.proj.as_ref().map(|proj| Span::of(proj, need.proj)),
        io.sinks.as_ref().map(|sinks| Span::of(sinks, need.sinks)),
    ];
    let apart = out.apart(&part)
        && reads
            .iter()
            .flatten()
            .all(|read| out.apart(read) && part.apart(read));
    let sink_status_apart = match (&io.sinks, &io.status) {
        (Some(sinks), Some(status)) => Span::of(sinks, need.sinks).apart(&Span::of(status, 4)),
        _ => true,
    };
    if !(sized && aligned && apart && sink_status_apart) {
        return Err(invalid_argument());
    }
    check_views(io, &need)
}

/// Tagged metadata is copied synchronously by the C entry.
fn c_args(io: &AttnPrefillRaw<'_>, views: Option<&AttnKvPairRaw>) -> AttnPrefillArgsRaw {
    let ints = &io.ints;
    AttnPrefillArgsRaw {
        q: io.q.address().cast_const(),
        proj: io
            .proj
            .as_ref()
            .map_or(std::ptr::null(), |proj| proj.address().cast_const()),
        k_cache: io.k_cache.address().cast_const(),
        v_cache: io.v_cache.address().cast_const(),
        kv8_k_codes: std::ptr::null(),
        kv8_k_scales: std::ptr::null(),
        kv8_v_codes: std::ptr::null(),
        kv8_v_scales: std::ptr::null(),
        part: io.part.address(),
        out: io.out.address(),
        status: io
            .status
            .as_ref()
            .map_or(std::ptr::null_mut(), |s| s.address()),
        ld: ints.ld,
        p0: ints.p0,
        m: ints.m,
        n_head: ints.n_head,
        n_kv_head: ints.n_kv_head,
        head_dim: ints.head_dim,
        cache_rows: ints.cache_rows,
        max_splits: ints.max_splits,
        min_keys: ints.min_keys,
        window: ints.window,
        ring: ints.ring,
        kv_mode: if views.is_some() { 2 } else { KV_FP16 },
        kv8_capacity: 0,
        scale: io.scale,
        sinks: io
            .sinks
            .as_ref()
            .map_or(std::ptr::null(), |s| s.address().cast_const()),
        kv_views: views.map_or(std::ptr::null(), std::ptr::from_ref),
    }
}

pub(in crate::kernels::cuda) fn attn_prefill_launch(
    stream: &RawStream,
    io: AttnPrefillRaw<'_>,
    launch: AttnPrefillLaunch,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    let views = c_views(&io)?;
    let args = c_args(&io, views.as_ref());
    let mut stage = STAGE_NONE;
    // SAFETY: `check_attn` proved every range inside its live allocation, large enough for
    // the call, aligned, and that `out` and `part` are disjoint from every other range, so
    // each pointer in `args` addresses bytes the kernels may touch; the C entry repeats these
    // checks on the addresses. The allocations stay valid until the queued work finishes
    // because `inference_cuda_free` synchronizes the device first. `args` lives across the
    // call and the entry copies it before returning. `stream` is live for the call.
    let status = unsafe {
        match launch {
            AttnPrefillLaunch::Full => attn_prefill_launch_raw(&args, stream.as_ptr(), &mut stage),
            AttnPrefillLaunch::Part {
                key_origin,
                key_rows,
                part_index,
            } => attn_prefill_part_launch_raw(
                &args,
                key_origin,
                key_rows,
                part_index,
                stream.as_ptr(),
                &mut stage,
            ),
            AttnPrefillLaunch::Combine => {
                attn_prefill_combine_launch_raw(&args, stream.as_ptr(), &mut stage)
            }
            #[cfg(all(test, feature = "cuda-q8-reference"))]
            AttnPrefillLaunch::Q8Reference => {
                attn_prefill_q8_reference_raw(&args, stream.as_ptr(), &mut stage)
            }
            #[cfg(all(test, feature = "cuda-q8-reference"))]
            AttnPrefillLaunch::Q8Tiled => {
                attn_prefill_q8_tiled_raw(&args, stream.as_ptr(), &mut stage)
            }
            AttnPrefillLaunch::Q8TiledRoute { use_tiled } => {
                let mut route: c_int = 0;
                let status = attn_prefill_q8_tiled_route_raw(
                    &args,
                    stream.as_ptr(),
                    &mut stage,
                    c_int::from(use_tiled),
                    &mut route,
                );
                if route == 1 && status == 0 {
                    Q8_TILED_ROUTES.fetch_add(1, Ordering::Relaxed);
                }
                if route == 2 {
                    Q8_TILED_FALLBACKS.fetch_add(1, Ordering::Relaxed);
                }
                status
            }
        }
    };
    checked(status, stage)
}

/// Queues the same two kernels with p0 = -1, invalid for both linear and ring positions. The device
/// re-checks (no partial, NaN out, mask 8) are under test; every other rule of the call holds.
/// Qualified fault path only.
#[cfg(all(test, feature = "cuda-fault-qualification"))]
pub(in crate::kernels::cuda) fn attn_prefill_fault_bad_bounds(
    stream: &RawStream,
    io: AttnPrefillRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_attn(&io)?;
    let views = c_views(&io)?;
    let mut args = c_args(&io, views.as_ref());
    args.p0 = -1;
    let mut stage = STAGE_NONE;
    // SAFETY: as `attn_prefill_launch`; the part kernel refuses before any partial write and
    // the combine's NaN write stays inside the proved `out` range.
    let status = unsafe { attn_prefill_fault_bad_bounds_raw(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::{AttnKvPairRaw, AttnKvViewRaw, AttnPrefillArgsRaw, attn_prefill_part_floats_raw};

    #[test]
    fn cpf_attn_row_block_sink_abi_reserves_one_pointer() {
        assert_eq!(core::mem::size_of::<AttnPrefillArgsRaw>(), 160);
        assert_eq!(core::mem::offset_of!(AttnPrefillArgsRaw, sinks), 144);
    }

    #[test]
    fn cpf_attn_tagged_kv_transport_has_exact_abi_offsets() {
        use core::mem::{align_of, offset_of, size_of};

        assert_eq!(size_of::<AttnKvViewRaw>(), 112);
        assert_eq!(align_of::<AttnKvViewRaw>(), 8);
        assert_eq!(offset_of!(AttnKvViewRaw, payload), 0);
        assert_eq!(offset_of!(AttnKvViewRaw, scales), 8);
        assert_eq!(offset_of!(AttnKvViewRaw, bias), 16);
        assert_eq!(offset_of!(AttnKvViewRaw, payload_time_stride_bytes), 24);
        assert_eq!(offset_of!(AttnKvViewRaw, payload_head_stride_bytes), 32);
        assert_eq!(offset_of!(AttnKvViewRaw, scale_time_stride_bytes), 40);
        assert_eq!(offset_of!(AttnKvViewRaw, scale_head_stride_bytes), 48);
        assert_eq!(offset_of!(AttnKvViewRaw, scale_group_stride_bytes), 56);
        assert_eq!(offset_of!(AttnKvViewRaw, bias_time_stride_bytes), 64);
        assert_eq!(offset_of!(AttnKvViewRaw, bias_head_stride_bytes), 72);
        assert_eq!(offset_of!(AttnKvViewRaw, bias_group_stride_bytes), 80);
        assert_eq!(offset_of!(AttnKvViewRaw, storage), 88);
        assert_eq!(offset_of!(AttnKvViewRaw, group_size), 92);
        assert_eq!(offset_of!(AttnKvViewRaw, scale_dtype), 96);
        assert_eq!(offset_of!(AttnKvViewRaw, bias_dtype), 100);
        assert_eq!(offset_of!(AttnKvViewRaw, q8_signed), 104);
        assert_eq!(offset_of!(AttnKvViewRaw, q4_low_even), 108);
        assert_eq!(size_of::<AttnKvPairRaw>(), 224);
        assert_eq!(align_of::<AttnKvPairRaw>(), 8);
        assert_eq!(offset_of!(AttnKvPairRaw, k), 0);
        assert_eq!(offset_of!(AttnKvPairRaw, v), 112);
        assert_eq!(size_of::<AttnPrefillArgsRaw>(), 160);
        assert_eq!(offset_of!(AttnPrefillArgsRaw, kv_views), 152);
    }

    fn empty_view() -> AttnKvViewRaw {
        AttnKvViewRaw {
            payload: core::ptr::null(),
            scales: core::ptr::null(),
            bias: core::ptr::null(),
            payload_time_stride_bytes: 0,
            payload_head_stride_bytes: 0,
            scale_time_stride_bytes: 0,
            scale_head_stride_bytes: 0,
            scale_group_stride_bytes: 0,
            bias_time_stride_bytes: 0,
            bias_head_stride_bytes: 0,
            bias_group_stride_bytes: 0,
            storage: 0,
            group_size: 0,
            scale_dtype: 0,
            bias_dtype: 0,
            q8_signed: 0,
            q4_low_even: 0,
        }
    }

    fn reserved_args(kv_mode: super::c_int, kv_views: *const AttnKvPairRaw) -> AttnPrefillArgsRaw {
        let data = core::ptr::without_provenance::<super::c_void>(16);
        AttnPrefillArgsRaw {
            q: data,
            proj: core::ptr::null(),
            k_cache: data,
            v_cache: data,
            kv8_k_codes: core::ptr::null(),
            kv8_k_scales: core::ptr::null(),
            kv8_v_codes: core::ptr::null(),
            kv8_v_scales: core::ptr::null(),
            part: data.cast_mut(),
            out: data.cast_mut(),
            status: core::ptr::null_mut(),
            ld: 0,
            p0: 0,
            m: 2,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 128,
            cache_rows: 2,
            max_splits: 1,
            min_keys: 32,
            window: 0,
            ring: 0,
            kv_mode,
            kv8_capacity: 0,
            scale: 1.0,
            sinks: core::ptr::null(),
            kv_views,
        }
    }

    #[test]
    fn cpf_attn_unselected_or_missing_tagged_transport_is_refused_before_any_cuda_work() {
        type Entry = unsafe extern "C" fn(
            *const AttnPrefillArgsRaw,
            *mut super::c_void,
            *mut super::c_int,
        ) -> super::c_int;
        let entries: &[(&str, Entry)] = &[
            ("full", super::attn_prefill_launch_raw),
            ("combine", super::attn_prefill_combine_launch_raw),
            #[cfg(feature = "cuda-fault-qualification")]
            ("fault", super::attn_prefill_fault_bad_bounds_raw),
        ];
        let pair = AttnKvPairRaw {
            k: empty_view(),
            v: empty_view(),
        };
        let expected = super::invalid_argument();
        let stream = core::ptr::without_provenance_mut::<super::c_void>(16);
        for (mode, views) in [
            (super::KV_FP16, &pair as *const AttnKvPairRaw),
            (super::KV_FP16, core::ptr::dangling()),
            (1, core::ptr::null()),
            (2, core::ptr::null()),
            (-1, core::ptr::null()),
        ] {
            let args = reserved_args(mode, views);
            for (name, entry) in entries {
                let mut stage = -1;
                // SAFETY: unsupported metadata must be refused before any data/stream access.
                let status = unsafe { entry(&args, stream, &mut stage) };
                assert_eq!(
                    (status, stage),
                    (expected.status, expected.stage),
                    "{name}: mode {mode}"
                );
            }
            let mut stage = -1;
            // SAFETY: same refusal contract; non-NULL stream avoids a vacuous stream rejection.
            let status =
                unsafe { super::attn_prefill_part_launch_raw(&args, 0, 2, 0, stream, &mut stage) };
            assert_eq!(
                (status, stage),
                (expected.status, expected.stage),
                "part: mode {mode}"
            );
        }
    }

    #[test]
    fn cpf_attn_tagged_sizing_covers_all_dims_and_positive_gqa() {
        for dim in [64, 128, 256] {
            for group in [1, 2, 3, 4, 5, 8, 16, 32] {
                for kv_heads in [1, 3] {
                    // SAFETY: this entry is pure checked integer sizing, with no device access.
                    let floats =
                        unsafe { attn_prefill_part_floats_raw(17, kv_heads, dim, 128, group) };
                    let expected =
                        24usize * kv_heads as usize * 128 * group as usize * (2 + dim as usize);
                    assert_eq!(floats, expected, "D{dim}/GQA{group}/KV{kv_heads}");
                }
            }
        }
        // SAFETY: an overflowing byte product must be refused without touching a device.
        assert_eq!(
            unsafe { attn_prefill_part_floats_raw(65535, 64, 256, 128, i32::MAX) },
            0
        );
    }

    #[test]
    fn cpf_attn_row_block_native_sizing_covers_each_query_head_cell() {
        for dim in [128, 256] {
            for group in [1, 8] {
                // SAFETY: this pure integer sizing entry touches no memory or device.
                let floats = unsafe { attn_prefill_part_floats_raw(9, 2, dim, 3, group) };
                assert_eq!(floats, (16 * 2 * 3 * group * (2 + dim)) as usize);
            }
        }
        for (m, heads, dim, splits, group) in [
            (0, 2, 128, 3, 1),
            (9, 0, 128, 3, 1),
            (9, 2, 96, 3, 1),
            (9, 2, 128, 0, 1),
            (9, 2, 128, 3, 0),
        ] {
            // SAFETY: invalid integers are refused before the pure byte product.
            assert_eq!(
                unsafe { attn_prefill_part_floats_raw(m, heads, dim, splits, group) },
                0
            );
        }
    }
}
