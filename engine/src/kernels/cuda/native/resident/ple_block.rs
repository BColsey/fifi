//! Raw FFI to the three launch entries of native/resident/ple_block.cu (declared in
//! native/resident/ple_block.h). Child of native::resident, so it inherits the
//! allow(unsafe_code) of native.rs.
//!
//! Safety invariants this module relies on:
//! - A device range is proved in bounds by RawRange::new before it gets here. This module
//!   proves again, on the sizes of the call, that every range holds the bytes the kernel
//!   touches and is aligned; the safe wrapper additionally proves that written ranges share
//!   no byte with any other range of the call, and the C entry repeats the null and
//!   alignment checks on the addresses.
//! - A launch entry only queues work on the stream it is given and returns the launch status;
//!   a fault in the kernel surfaces at the next synchronize.
//! - The allocations stay valid until the queued launch finishes because
//!   inference_cuda_free synchronizes the whole device before cudaFree.
//! - Every call runs under the process dispatch guard and on the thread that owns the context.
//! - No raw pointer and no unsafe function leaves this module.

use std::ffi::{c_int, c_void};
use std::mem::offset_of;

use super::{DeviceFailure, RawRange, RawStream, STAGE_NONE, Span, checked, invalid_argument};

unsafe extern "C" {
    #[link_name = "inference_cuda_ple_rows_launch"]
    fn ple_rows_launch_raw(
        args: *const PleRowsArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_ple_gate_launch"]
    fn ple_gate_launch_raw(
        args: *const PleGateArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    #[link_name = "inference_cuda_ple_conv_launch"]
    fn ple_conv_launch_raw(
        args: *const PleConvArgsRepr,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

/// Mirror of inference_cuda_ple_rows_args (ple_block.h), field for field.
#[repr(C)]
struct PleRowsArgsRepr {
    out: *mut c_void,
    raw: *const c_void,
    scale: f32,
    rows: c_int,
    head_dim: c_int,
}

/// Mirror of inference_cuda_ple_gate_args.
#[repr(C)]
struct PleGateArgsRepr {
    keys: *const c_void,
    hyper: *const c_void,
    value: *const c_void,
    norm_key: *const c_void,
    norm_query: *const c_void,
    norm_conv: *const c_void,
    gated: *mut c_void,
    normed: *mut c_void,
    eps: f32,
    branches: c_int,
    hidden: c_int,
}

/// Mirror of inference_cuda_ple_conv_args.
#[repr(C)]
struct PleConvArgsRepr {
    gated: *const c_void,
    normed: *const c_void,
    conv: *const c_void,
    ring: *mut c_void,
    out: *mut c_void,
    width: c_int,
    conv_kernel: c_int,
    ngram: c_int,
}

const _: () = assert!(
    size_of::<PleRowsArgsRepr>() == 32
        && offset_of!(PleRowsArgsRepr, scale) == 16
        && offset_of!(PleRowsArgsRepr, rows) == 20
        && offset_of!(PleRowsArgsRepr, head_dim) == 24
        && size_of::<PleGateArgsRepr>() == 80
        && offset_of!(PleGateArgsRepr, eps) == 64
        && offset_of!(PleGateArgsRepr, branches) == 68
        && offset_of!(PleGateArgsRepr, hidden) == 72
);
const _: () = assert!(
    size_of::<PleConvArgsRepr>() == 56
        && offset_of!(PleConvArgsRepr, width) == 40
        && offset_of!(PleConvArgsRepr, conv_kernel) == 44
        && offset_of!(PleConvArgsRepr, ngram) == 48
);

pub(in crate::kernels::cuda) struct PleRowsRaw<'a> {
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) raw: RawRange<'a>,
    pub(in crate::kernels::cuda) scale: f32,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) head_dim: usize,
}

pub(in crate::kernels::cuda) struct PleGateRaw<'a> {
    pub(in crate::kernels::cuda) keys: RawRange<'a>,
    pub(in crate::kernels::cuda) hyper: RawRange<'a>,
    pub(in crate::kernels::cuda) value: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_key: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_query: RawRange<'a>,
    pub(in crate::kernels::cuda) norm_conv: RawRange<'a>,
    pub(in crate::kernels::cuda) gated: RawRange<'a>,
    pub(in crate::kernels::cuda) normed: RawRange<'a>,
    pub(in crate::kernels::cuda) eps: f32,
    pub(in crate::kernels::cuda) branches: usize,
    pub(in crate::kernels::cuda) hidden: usize,
}

pub(in crate::kernels::cuda) struct PleConvRaw<'a> {
    pub(in crate::kernels::cuda) gated: RawRange<'a>,
    pub(in crate::kernels::cuda) normed: RawRange<'a>,
    pub(in crate::kernels::cuda) conv: RawRange<'a>,
    pub(in crate::kernels::cuda) ring: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) width: usize,
    pub(in crate::kernels::cuda) conv_kernel: usize,
    pub(in crate::kernels::cuda) ngram: usize,
}

fn holds(range: &RawRange<'_>, bytes: usize, align: usize) -> bool {
    range.bytes >= bytes && range.offset.is_multiple_of(align)
}

fn int(value: usize) -> Result<c_int, DeviceFailure> {
    c_int::try_from(value).map_err(|_| invalid_argument())
}

/// Proves the shape, the sizes and the alignment, and that `out` shares no byte with `raw`.
fn check_rows(call: &PleRowsRaw<'_>) -> Result<(), DeviceFailure> {
    if !call.head_dim.is_multiple_of(32)
        || !(32..=8192).contains(&call.head_dim)
        || !(1..=65_536).contains(&call.rows)
        || !call.scale.is_finite()
    {
        return Err(invalid_argument());
    }
    // The shape is bounded above, so the products cannot overflow.
    let out_bytes = call.rows * call.head_dim * 4;
    let raw_bytes = call.rows * (call.head_dim / 32 * 18);
    let out = Span::of(&call.out, out_bytes);
    let raw = Span::of(&call.raw, raw_bytes);
    if !(holds(&call.out, out_bytes, 4) && holds(&call.raw, raw_bytes, 2) && out.apart(&raw)) {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn ple_rows_launch(
    stream: &RawStream,
    call: &PleRowsRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_rows(call)?;
    let args = PleRowsArgsRepr {
        out: call.out.address(),
        raw: call.raw.address().cast_const(),
        scale: call.scale,
        rows: int(call.rows)?,
        head_dim: int(call.head_dim)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges were proved inside their live allocations, large enough, aligned and
    // apart above; the adapter repeats the null and alignment checks on the addresses. The
    // allocations stay valid until the queued launch finishes because inference_cuda_free
    // synchronizes the device first. stream is live for the call and args is a plain value.
    let status = unsafe { ple_rows_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// True when every written extent shares no byte with another written extent or with a read.
fn written_apart(written: &[Span<'_>], read: &[Span<'_>]) -> bool {
    written.iter().enumerate().all(|(i, span)| {
        written[i + 1..].iter().all(|other| span.apart(other))
            && read.iter().all(|other| span.apart(other))
    })
}

/// Proves the shape, the sizes and the alignment, and that each written range shares no byte
/// with any other range; the read ranges may overlap.
fn check_gate(call: &PleGateRaw<'_>) -> Result<(), DeviceFailure> {
    if !(1..=64).contains(&call.branches)
        || !(1..=(1 << 20)).contains(&call.hidden)
        || !call.eps.is_finite()
        || call.eps < 0.0
    {
        return Err(invalid_argument());
    }
    // The shape is bounded above, so the products cannot overflow.
    let width = call.branches * call.hidden * 4;
    let value = call.hidden * 4;
    let sized = [
        &call.keys,
        &call.hyper,
        &call.norm_key,
        &call.norm_query,
        &call.norm_conv,
        &call.gated,
        &call.normed,
    ]
    .iter()
    .all(|range| holds(range, width, 4))
        && holds(&call.value, value, 4);
    let written = [Span::of(&call.gated, width), Span::of(&call.normed, width)];
    let read = [
        Span::of(&call.keys, width),
        Span::of(&call.hyper, width),
        Span::of(&call.value, value),
        Span::of(&call.norm_key, width),
        Span::of(&call.norm_query, width),
        Span::of(&call.norm_conv, width),
    ];
    if !(sized && written_apart(&written, &read)) {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn ple_gate_launch(
    stream: &RawStream,
    call: &PleGateRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_gate(call)?;
    let args = PleGateArgsRepr {
        keys: call.keys.address().cast_const(),
        hyper: call.hyper.address().cast_const(),
        value: call.value.address().cast_const(),
        norm_key: call.norm_key.address().cast_const(),
        norm_query: call.norm_query.address().cast_const(),
        norm_conv: call.norm_conv.address().cast_const(),
        gated: call.gated.address(),
        normed: call.normed.address(),
        eps: call.eps,
        branches: int(call.branches)?,
        hidden: int(call.hidden)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as for ple_rows_launch; every range was proved large enough and aligned above,
    // and each written range apart from every other range.
    let status = unsafe { ple_gate_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

/// Proves the shape, the sizes and the alignment, and that `out` and `ring` share no byte
/// with each other or with a read range. The ring is read and written, so it counts as written.
fn check_conv(call: &PleConvRaw<'_>) -> Result<(), DeviceFailure> {
    if !(1..=(1 << 22)).contains(&call.width)
        || !(2..=16).contains(&call.conv_kernel)
        || !(1..=16).contains(&call.ngram)
    {
        return Err(invalid_argument());
    }
    // The shape is bounded above, so the products cannot overflow.
    let width = call.width * 4;
    let conv = call.width * call.conv_kernel * 4;
    let ring = call.width * (call.conv_kernel - 1) * call.ngram * 4;
    let sized = holds(&call.gated, width, 4)
        && holds(&call.normed, width, 4)
        && holds(&call.out, width, 4)
        && holds(&call.conv, conv, 4)
        && holds(&call.ring, ring, 4);
    let written = [Span::of(&call.out, width), Span::of(&call.ring, ring)];
    let read = [
        Span::of(&call.gated, width),
        Span::of(&call.normed, width),
        Span::of(&call.conv, conv),
    ];
    if !(sized && written_apart(&written, &read)) {
        return Err(invalid_argument());
    }
    Ok(())
}

pub(in crate::kernels::cuda) fn ple_conv_launch(
    stream: &RawStream,
    call: &PleConvRaw<'_>,
) -> Result<(), DeviceFailure> {
    check_conv(call)?;
    let args = PleConvArgsRepr {
        gated: call.gated.address().cast_const(),
        normed: call.normed.address().cast_const(),
        conv: call.conv.address().cast_const(),
        ring: call.ring.address(),
        out: call.out.address(),
        width: int(call.width)?,
        conv_kernel: int(call.conv_kernel)?,
        ngram: int(call.ngram)?,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: as for ple_rows_launch; every range was proved large enough and aligned above,
    // and each written range apart from every other range.
    let status = unsafe { ple_conv_launch_raw(&args, stream.0.as_ptr(), &mut stage) };
    checked(status, stage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::cuda::native::{RawDevice, poison_test_raw};

    /// A range of one made-up allocation. The checks read offsets and sizes only, so no
    /// pointer is dereferenced and nothing is launched.
    fn range(device: &RawDevice, offset: usize, bytes: usize) -> RawRange<'_> {
        RawRange::new(device, offset, bytes).expect("the range lies inside the allocation")
    }

    /// The rows call of five rows of 128 values: `out` is 2560 bytes and `raw` 360.
    fn rows(device: &RawDevice, out: usize, raw: usize) -> PleRowsRaw<'_> {
        PleRowsRaw {
            out: range(device, out, 2560),
            raw: range(device, raw, 360),
            scale: 0.5,
            rows: 5,
            head_dim: 128,
        }
    }

    #[test]
    fn the_rows_launch_refuses_an_output_over_its_input() {
        let device = poison_test_raw(1 << 15);
        let other = poison_test_raw(1 << 15);
        assert!(check_rows(&rows(&device, 0, 4096)).is_ok(), "apart");
        assert!(
            check_rows(&rows(&device, 3584, 4096)).is_err(),
            "out over raw"
        );
        assert!(check_rows(&rows(&device, 0, 0)).is_err(), "raw inside out");
        let apart = PleRowsRaw {
            raw: range(&other, 0, 360),
            ..rows(&device, 0, 0)
        };
        assert!(check_rows(&apart).is_ok(), "two allocations share no byte");
    }

    /// The offsets of the gate's eight ranges: keys, hyper, value, norm_key, norm_query,
    /// norm_conv, gated, normed. Four branches of 256 values make a wide range 4096 bytes and
    /// `value` 1024.
    const GATE_APART: [usize; 8] = [0, 4096, 8192, 12288, 16384, 20480, 24576, 28672];

    fn gate(device: &RawDevice, at: [usize; 8]) -> PleGateRaw<'_> {
        let [
            keys,
            hyper,
            value,
            norm_key,
            norm_query,
            norm_conv,
            gated,
            normed,
        ] = at;
        let wide = |offset| range(device, offset, 4096);
        PleGateRaw {
            keys: wide(keys),
            hyper: wide(hyper),
            value: range(device, value, 1024),
            norm_key: wide(norm_key),
            norm_query: wide(norm_query),
            norm_conv: wide(norm_conv),
            gated: wide(gated),
            normed: wide(normed),
            eps: 1e-6,
            branches: 4,
            hidden: 256,
        }
    }

    #[test]
    fn the_gate_launch_refuses_a_written_range_over_any_other() {
        let device = poison_test_raw(1 << 15);
        assert!(check_gate(&gate(&device, GATE_APART)).is_ok(), "apart");
        // keys, hyper, value, norm_key, norm_query, norm_conv, gated, normed
        assert!(
            check_gate(&gate(
                &device,
                [0, 4096, 8192, 12288, 16384, 20480, 2048, 28672]
            ))
            .is_err(),
            "gated over keys"
        );
        assert!(
            check_gate(&gate(
                &device,
                [0, 4096, 8192, 12288, 16384, 20480, 24576, 25088]
            ))
            .is_err(),
            "normed over gated"
        );
        assert!(
            check_gate(&gate(
                &device,
                [0, 4096, 8192, 12288, 16384, 20480, 24576, 8448]
            ))
            .is_err(),
            "normed over value"
        );
        assert!(
            check_gate(&gate(
                &device,
                [0, 0, 8192, 12288, 16384, 20480, 24576, 28672]
            ))
            .is_ok(),
            "two reads may overlap"
        );
    }

    /// The offsets of the convolution's five ranges: gated, normed, conv, ring, out, for 96
    /// channels, a kernel of 4 and an n-gram of 3 (384, 384, 1536, 3456 and 384 bytes).
    const CONV_APART: [usize; 5] = [0, 1024, 2048, 4096, 8192];

    fn conv(device: &RawDevice, at: [usize; 5]) -> PleConvRaw<'_> {
        let [gated, normed, conv, ring, out] = at;
        PleConvRaw {
            gated: range(device, gated, 384),
            normed: range(device, normed, 384),
            conv: range(device, conv, 1536),
            ring: range(device, ring, 3456),
            out: range(device, out, 384),
            width: 96,
            conv_kernel: 4,
            ngram: 3,
        }
    }

    #[test]
    fn the_convolution_launch_refuses_its_written_ranges_over_any_other() {
        let device = poison_test_raw(1 << 14);
        assert!(check_conv(&conv(&device, CONV_APART)).is_ok(), "apart");
        assert!(
            check_conv(&conv(&device, [0, 1024, 2048, 4096, 192])).is_err(),
            "out over gated"
        );
        assert!(
            check_conv(&conv(&device, [0, 1024, 2048, 3072, 8192])).is_err(),
            "ring over conv"
        );
        assert!(
            check_conv(&conv(&device, [0, 1024, 2048, 4096, 5120])).is_err(),
            "out inside ring"
        );
        assert!(
            check_conv(&conv(&device, [0, 1024, 2048, 4096, 7552])).is_ok(),
            "out starts where the ring ends"
        );
    }
}
