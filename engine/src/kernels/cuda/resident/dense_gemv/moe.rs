//! One stream sequence for an ordered set of GGUF experts.

use super::DenseDtype;
use crate::{
    EngError,
    kernels::cuda::{
        CudaContext, DeviceRange, DeviceStream,
        context::{Claim, require_apart, require_fit},
        native::resident::dense_gemv::moe::{GgufMoeRaw, GgufMoeRawWeights, launch_gguf_moe},
    },
};

pub(crate) use crate::kernels::cuda::native::resident::dense_gemv::moe::{
    GgufMoePick, GgufMoeShape,
};

pub(crate) enum GgufMoeWeights<'a> {
    Stacks([DeviceRange<'a>; 2]),
    /// Matrix ranges in the same order as the picks.
    Separate(&'a [[DeviceRange<'a>; 2]]),
}

pub(crate) struct GgufMoeCall<'a> {
    pub(crate) shape: GgufMoeShape,
    pub(crate) dtypes: [DenseDtype; 2],
    pub(crate) caps: [f32; 2],
    pub(crate) picks: &'a [GgufMoePick],
    pub(crate) weights: GgufMoeWeights<'a>,
    pub(crate) x: DeviceRange<'a>,
    pub(crate) scratch: DeviceRange<'a>,
    pub(crate) y: DeviceRange<'a>,
}

impl CudaContext {
    pub(crate) fn gguf_moe(
        &self,
        stream: &DeviceStream,
        call: GgufMoeCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_gguf_moe(&call)?;
        self.launch_on(stream, |stream_raw| launch_gguf_moe(stream_raw, raw))
    }

    fn plan_gguf_moe<'a>(&self, c: &GgufMoeCall<'a>) -> Result<GgufMoeRaw<'a>, EngError> {
        let scratch = c.shape.scratch_bytes(c.picks.len()).ok_or_else(bad_group)?;
        let matrix_bytes = c.shape.matrix_bytes(c.dtypes).ok_or_else(bad_group)?;
        let positive = |v: f32| v.is_finite() && v > 0.0;
        if !c.caps.into_iter().all(positive)
            || !c.picks.iter().all(|p| {
                (p.expert as usize) < c.shape.experts
                    && positive(p.gate_scale)
                    && positive(p.down_scale)
                    && p.weight.is_finite()
            })
        {
            return Err(bad_group());
        }
        let row = c.shape.hidden * size_of::<f32>();
        require_fit("x", &c.x, row, 4)?;
        require_fit("scratch", &c.scratch, scratch, 4)?;
        require_fit("y", &c.y, row, 4)?;

        let temporary = Claim::of("scratch", &c.scratch, scratch);
        let output = Claim::of("y", &c.y, row);
        require_apart(&temporary, &output)?;
        let input = Claim::of("x", &c.x, row);
        require_apart(&temporary, &input)?;
        require_apart(&output, &input)?;
        let weights = self.moe_weights(c, matrix_bytes, [&temporary, &output])?;
        Ok(GgufMoeRaw {
            shape: c.shape,
            dtypes: c.dtypes,
            caps: c.caps,
            picks: c.picks,
            weights,
            x: self.raw_range(&c.x)?,
            scratch: self.raw_range(&c.scratch)?,
            y: self.raw_range(&c.y)?,
        })
    }

    fn moe_weights<'a>(
        &self,
        call: &GgufMoeCall<'a>,
        sizes: [usize; 2],
        writes: [&Claim<'_>; 2],
    ) -> Result<GgufMoeRawWeights<'a>, EngError> {
        match call.weights {
            GgufMoeWeights::Stacks(stacks) => {
                for (stack, size) in stacks.iter().zip(sizes) {
                    let bytes = size.checked_mul(call.shape.experts).ok_or_else(bad_group)?;
                    validate_weight(stack, bytes, "expert stack", writes)?;
                }
                Ok(GgufMoeRawWeights::Stacks([
                    self.raw_range(&stacks[0])?,
                    self.raw_range(&stacks[1])?,
                ]))
            }
            GgufMoeWeights::Separate(parts) => {
                if parts.len() != call.picks.len() {
                    return Err(bad_group());
                }
                let mut matrices = Vec::with_capacity(parts.len());
                for ranges in parts {
                    for (range, size) in ranges.iter().zip(sizes) {
                        validate_weight(range, size, "expert matrix", writes)?;
                    }
                    matrices.push([self.raw_range(&ranges[0])?, self.raw_range(&ranges[1])?]);
                }
                Ok(GgufMoeRawWeights::Separate(matrices))
            }
        }
    }
}

fn bad_group() -> EngError {
    EngError::BadRequest("invalid CUDA GGUF expert group".into())
}

fn validate_weight(
    range: &DeviceRange<'_>,
    bytes: usize,
    name: &'static str,
    writes: [&Claim<'_>; 2],
) -> Result<(), EngError> {
    require_fit(name, range, bytes, 2)?;
    for write in writes {
        require_apart(write, &Claim::of(name, range, bytes))?;
    }
    Ok(())
}
