use super::*;

pub(in crate::kernels::cuda) fn clipped_expert_bytes(
    rows: usize,
    middle: usize,
) -> Option<(usize, usize, usize)> {
    let (out, scales) = row_pow2_geometry(rows, middle)?;
    let input = row_pow2_geometry(rows, middle.checked_mul(2)?)?.0;
    Some((input, scales, out))
}

pub(in crate::kernels::cuda) struct ClippedExpertRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) middle: usize,
    pub(in crate::kernels::cuda) caps: [f32; 2],
}

pub(in crate::kernels::cuda) struct IndexedRankSumRaw<'a> {
    pub(in crate::kernels::cuda) x: RawRange<'a>,
    pub(in crate::kernels::cuda) inverse_ids: RawRange<'a>,
    pub(in crate::kernels::cuda) scales: RawRange<'a>,
    pub(in crate::kernels::cuda) weights: RawRange<'a>,
    pub(in crate::kernels::cuda) out: RawRange<'a>,
    pub(in crate::kernels::cuda) flag: RawRange<'a>,
    pub(in crate::kernels::cuda) rows: usize,
    pub(in crate::kernels::cuda) columns: usize,
    pub(in crate::kernels::cuda) rank: usize,
}

#[repr(C)]
struct ClippedExpertArgs {
    x: *const f32,
    x_bytes: usize,
    scales: *const f32,
    scales_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    middle: usize,
    gate_cap: f32,
    up_cap: f32,
}

#[repr(C)]
struct IndexedRankSumArgs {
    x: *const f32,
    x_bytes: usize,
    inverse_ids: *const i32,
    inverse_ids_bytes: usize,
    scales: *const f32,
    scales_bytes: usize,
    weights: *const f32,
    weights_bytes: usize,
    out: *mut f32,
    out_bytes: usize,
    flag: *mut u32,
    flag_bytes: usize,
    rows: usize,
    columns: usize,
    rank: usize,
}

const _: () = assert!(
    size_of::<ClippedExpertArgs>() == 88
        && offset_of!(ClippedExpertArgs, gate_cap) == 80
        && size_of::<IndexedRankSumArgs>() == 120
        && offset_of!(IndexedRankSumArgs, rank) == 112
);

unsafe extern "C" {
    fn inference_cuda_row_clipped_expert_launch(
        args: *const ClippedExpertArgs,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
    fn inference_cuda_row_indexed_rank_sum_launch(
        args: *const IndexedRankSumArgs,
        stream: *mut c_void,
        out_stage: *mut c_int,
    ) -> c_int;
}

pub(in crate::kernels::cuda) fn launch_clipped_expert(
    stream: &RawStream,
    call: &ClippedExpertRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (input, scales, out) =
        clipped_expert_bytes(call.rows, call.middle).ok_or_else(invalid_argument)?;
    if !call.caps.iter().all(|cap| cap.is_finite() && *cap > 0.0) {
        return Err(invalid_argument());
    }
    check_regions(&[
        (&call.x, input, false),
        (&call.scales, scales, false),
        (&call.out, out, true),
        (&call.flag, 4, true),
    ])?;
    let args = ClippedExpertArgs {
        x: call.x.address().cast_const().cast(),
        x_bytes: call.x.bytes(),
        scales: call.scales.address().cast_const().cast(),
        scales_bytes: call.scales.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        middle: call.middle,
        gate_cap: call.caps[0],
        up_cap: call.caps[1],
    };
    let mut stage = STAGE_NONE;
    // SAFETY: ranges are live, sized and disjoint for writes. C rechecks all ranges.
    let status =
        unsafe { inference_cuda_row_clipped_expert_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}

pub(in crate::kernels::cuda) fn launch_indexed_rank_sum(
    stream: &RawStream,
    call: &IndexedRankSumRaw<'_>,
) -> Result<(), DeviceFailure> {
    let (input, out) =
        rank_sum_bytes(call.rows, call.columns, call.rank).ok_or_else(invalid_argument)?;
    let metadata = call.rows * call.rank * 4;
    check_regions(&[
        (&call.x, input, false),
        (&call.inverse_ids, metadata, false),
        (&call.scales, metadata, false),
        (&call.weights, metadata, false),
        (&call.out, out, true),
        (&call.flag, 4, true),
    ])?;
    let args = IndexedRankSumArgs {
        x: call.x.address().cast_const().cast(),
        x_bytes: call.x.bytes(),
        inverse_ids: call.inverse_ids.address().cast_const().cast(),
        inverse_ids_bytes: call.inverse_ids.bytes(),
        scales: call.scales.address().cast_const().cast(),
        scales_bytes: call.scales.bytes(),
        weights: call.weights.address().cast_const().cast(),
        weights_bytes: call.weights.bytes(),
        out: call.out.address().cast(),
        out_bytes: call.out.bytes(),
        flag: call.flag.address().cast(),
        flag_bytes: call.flag.bytes(),
        rows: call.rows,
        columns: call.columns,
        rank: call.rank,
    };
    let mut stage = STAGE_NONE;
    // SAFETY: the ranges and stream are live. The kernel bounds every inverse ID.
    let status =
        unsafe { inference_cuda_row_indexed_rank_sum_launch(&args, stream.as_ptr(), &mut stage) };
    checked(status, stage)
}
