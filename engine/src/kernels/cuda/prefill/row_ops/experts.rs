use super::*;

pub(crate) struct ClippedExpertCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) scales: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
    pub(crate) caps: [f32; 2],
}

pub(crate) struct IndexedRankSumCall<'a> {
    pub(crate) x: DeviceRange<'a>,
    pub(crate) inverse_ids: DeviceRange<'a>,
    pub(crate) scales: DeviceRange<'a>,
    pub(crate) weights: DeviceRange<'a>,
    pub(crate) out: DeviceRange<'a>,
    pub(crate) flag: DeviceRange<'a>,
    pub(crate) shape: RowShape,
    pub(crate) rank: usize,
}

impl CudaContext {
    /// F32 expert activation from concatenated gate and up rows and a scale per row.
    pub(crate) fn row_clipped_expert(
        &self,
        stream: &DeviceStream,
        call: ClippedExpertCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (input, scales, out) = ffi::clipped_expert_bytes(call.shape.rows, call.shape.columns)
            .ok_or_else(geometry_error)?;
        if !call.caps.iter().all(|cap| cap.is_finite() && *cap > 0.0) {
            return Err(geometry_error());
        }
        check_regions(
            self,
            &[
                ("x", &call.x, input, false),
                ("scales", &call.scales, scales, false),
                ("out", &call.out, out, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = ffi::ClippedExpertRaw {
            x: self.raw_range(&call.x)?,
            scales: self.raw_range(&call.scales)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            middle: call.shape.columns,
            caps: call.caps,
        };
        self.launch_on(stream, |raw_stream| {
            ffi::launch_clipped_expert(raw_stream, &raw)
        })
    }

    /// Sum expert-grouped rows in original rank order with separate scale and weight rounding.
    pub(crate) fn row_indexed_rank_sum(
        &self,
        stream: &DeviceStream,
        call: IndexedRankSumCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let (input, out) = ffi::rank_sum_bytes(call.shape.rows, call.shape.columns, call.rank)
            .ok_or_else(geometry_error)?;
        let metadata = call.shape.rows * call.rank * 4;
        check_regions(
            self,
            &[
                ("x", &call.x, input, false),
                ("inverse_ids", &call.inverse_ids, metadata, false),
                ("scales", &call.scales, metadata, false),
                ("weights", &call.weights, metadata, false),
                ("out", &call.out, out, true),
                ("flag", &call.flag, 4, true),
            ],
        )?;
        let raw = ffi::IndexedRankSumRaw {
            x: self.raw_range(&call.x)?,
            inverse_ids: self.raw_range(&call.inverse_ids)?,
            scales: self.raw_range(&call.scales)?,
            weights: self.raw_range(&call.weights)?,
            out: self.raw_range(&call.out)?,
            flag: self.raw_range(&call.flag)?,
            rows: call.shape.rows,
            columns: call.shape.columns,
            rank: call.rank,
        };
        self.launch_on(stream, |raw_stream| {
            ffi::launch_indexed_rank_sum(raw_stream, &raw)
        })
    }
}
