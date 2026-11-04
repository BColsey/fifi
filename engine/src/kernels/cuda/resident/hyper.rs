//! The hyper-connection (four-branch gated residual) read and apply of Qwen3.8 Flash-Next on the
//! device. The formulas and layouts are in `native/resident/hyper.h`; they follow the gated
//! residual of the Qwen3.8-Next architecture report (arXiv 2608.30320, Eq. 30-34) and the host
//! code of `family::flash_next::hyper`.
//!
//! [`CudaContext::hyper_read`] queues four kernels on a [`DeviceStream`] and returns: the
//! per-branch RMS norm with `(1 + w)` weights, the low-rank down projection with SiLU and the
//! inject projection with `2 * sigmoid`, the up projection, and the sigmoid-gated branch mean.
//! [`CudaContext::hyper_apply`] queues the write `x[t][b][d] += gate[t][b] * block[t][d]`. The
//! calls allocate nothing, copy nothing and never synchronize; a fault in the queued work
//! surfaces at the next synchronize of the stream.
//!
//! The residual stream and every intermediate are f32. The weights stay in their stored dtype
//! (binary16 for the EXL3 packs, bfloat16 for the official checkpoint) and are widened exactly to
//! f32 inside the kernels; the caller names the dtype, which is never guessed.
//!
//! Every refusal is a `BadRequest` and happens before any native call, because a status from the
//! native layer other than an allocation failure latches the backend terminal. A refused call
//! changes nothing.
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "staged for the Flash-Next CUDA step")
)]

use crate::EngError;
use crate::kernels::cuda::context::{Claim, require_apart, require_fit};
pub(crate) use crate::kernels::cuda::native::resident::hyper::HyperDtype;
use crate::kernels::cuda::native::resident::hyper::{
    HYPER_MAX_BRANCHES, HYPER_MAX_HIDDEN, HYPER_MAX_M, HYPER_MAX_RANK, HYPER_MAX_WIDTH,
    HyperApplyRaw, HyperDims, HyperReadBytes, HyperReadRaw, dims_width, hyper_apply_launch,
    hyper_read_launch, read_need,
};
use crate::kernels::cuda::{CudaContext, DeviceRange, DeviceStream};

/// `m` tokens of a stream with `branches` branches of `hidden` values, and the low rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HyperShape {
    pub(crate) m: usize,
    pub(crate) branches: usize,
    pub(crate) hidden: usize,
    pub(crate) rank: usize,
}

impl HyperShape {
    fn dims(self) -> HyperDims {
        HyperDims {
            m: self.m,
            branches: self.branches,
            hidden: self.hidden,
        }
    }
}

/// The dense weights of one hyper-connection site, all of `dtype`: `norm` `[width]`, `down`
/// `[rank][width]`, `up` `[width][rank]` and `inject` `[branches][width]` (absent for the final
/// mixer). Each 2-byte aligned.
pub(crate) struct HyperWeights<'b> {
    pub(crate) dtype: HyperDtype,
    pub(crate) norm: DeviceRange<'b>,
    pub(crate) down: DeviceRange<'b>,
    pub(crate) up: DeviceRange<'b>,
    pub(crate) inject: Option<DeviceRange<'b>>,
}

/// The f32 scratch of one read: `normed` `[m][width]`, `low` `[m][rank]`, `mix` `[m][width]`.
pub(crate) struct HyperScratch<'b> {
    pub(crate) normed: DeviceRange<'b>,
    pub(crate) low: DeviceRange<'b>,
    pub(crate) mix: DeviceRange<'b>,
}

/// One read: `x` f32 `[m][width]` is read; `mixed` f32 `[m][hidden]` and, with `inject`, `gate`
/// f32 `[m][branches]` are written.
pub(crate) struct HyperReadCall<'b> {
    pub(crate) shape: HyperShape,
    pub(crate) eps: f32,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) weights: HyperWeights<'b>,
    pub(crate) scratch: HyperScratch<'b>,
    pub(crate) mixed: DeviceRange<'b>,
    pub(crate) gate: Option<DeviceRange<'b>>,
}

/// One apply: `x` f32 `[m][width]` is updated in place from `gate` f32 `[m][branches]` and
/// `block` f32 `[m][hidden]`.
pub(crate) struct HyperApplyCall<'b> {
    pub(crate) shape: HyperShape,
    pub(crate) x: DeviceRange<'b>,
    pub(crate) gate: DeviceRange<'b>,
    pub(crate) block: DeviceRange<'b>,
}

/// The bytes each range of a read of `shape` must hold.
pub(crate) fn hyper_read_bytes(shape: HyperShape) -> Result<HyperReadBytes, EngError> {
    check_dims(shape)?;
    if !(1..=HYPER_MAX_RANK).contains(&shape.rank) {
        return Err(EngError::BadRequest(format!(
            "the CUDA hyper-connection rank must be 1 to {HYPER_MAX_RANK}, got {}",
            shape.rank
        )));
    }
    read_need(shape.dims(), shape.rank).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA hyper-connection shape {shape:?} overflows"
        ))
    })
}

fn check_dims(shape: HyperShape) -> Result<usize, EngError> {
    dims_width(shape.dims()).ok_or_else(|| {
        EngError::BadRequest(format!(
            "the CUDA hyper-connection takes 1 to {HYPER_MAX_M} tokens, 1 to {HYPER_MAX_BRANCHES} \
             branches, 1 to {HYPER_MAX_HIDDEN} values per branch and a width up to \
             {HYPER_MAX_WIDTH}, got {shape:?}"
        ))
    })
}

impl CudaContext {
    /// Queues the hyper-connection read of `call.shape.m` tokens on `stream`.
    ///
    /// Returns `BadRequest` for a stream or buffer of another adapter, for a shape outside the
    /// limits, for a negative or non-finite `eps`, for an `inject` without a `gate` or the
    /// reverse, for a short or misaligned range, and for a written range that overlaps another
    /// range.
    pub(crate) fn hyper_read(
        &self,
        stream: &DeviceStream,
        call: HyperReadCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_hyper_read(&call)?;
        self.launch_on(stream, move |stream_raw| hyper_read_launch(stream_raw, raw))
    }

    /// Queues `x += gate * block` per branch on `stream`.
    pub(crate) fn hyper_apply(
        &self,
        stream: &DeviceStream,
        call: HyperApplyCall<'_>,
    ) -> Result<(), EngError> {
        self.ensure_usable()?;
        let raw = self.plan_hyper_apply(&call)?;
        self.launch_on(stream, move |stream_raw| {
            hyper_apply_launch(stream_raw, raw)
        })
    }

    fn plan_hyper_read<'b>(&self, call: &HyperReadCall<'b>) -> Result<HyperReadRaw<'b>, EngError> {
        let need = hyper_read_bytes(call.shape)?;
        if !(call.eps.is_finite() && call.eps >= 0.0) {
            return Err(EngError::BadRequest(format!(
                "the CUDA hyper-connection eps must be finite and nonnegative, got {}",
                call.eps
            )));
        }
        let weights = &call.weights;
        if weights.inject.is_some() != call.gate.is_some() {
            return Err(EngError::BadRequest(
                "the CUDA hyper-connection takes inject and gate together or neither".into(),
            ));
        }
        require_fit("x", &call.x, need.x, 4)?;
        require_fit("norm", &weights.norm, need.norm_w, 2)?;
        require_fit("down", &weights.down, need.down, 2)?;
        require_fit("up", &weights.up, need.up, 2)?;
        if let Some(inject) = &weights.inject {
            require_fit("inject", inject, need.inject, 2)?;
        }
        let scratch = &call.scratch;
        require_fit("normed", &scratch.normed, need.normed, 4)?;
        require_fit("low", &scratch.low, need.low, 4)?;
        require_fit("mix", &scratch.mix, need.mix, 4)?;
        require_fit("mixed", &call.mixed, need.mixed, 4)?;
        if let Some(gate) = &call.gate {
            require_fit("gate", gate, need.gate, 4)?;
        }
        let mut written = vec![
            Claim::of("normed", &scratch.normed, need.normed),
            Claim::of("low", &scratch.low, need.low),
            Claim::of("mix", &scratch.mix, need.mix),
            Claim::of("mixed", &call.mixed, need.mixed),
        ];
        if let Some(gate) = &call.gate {
            written.push(Claim::of("gate", gate, need.gate));
        }
        let mut read = vec![
            Claim::of("x", &call.x, need.x),
            Claim::of("norm", &weights.norm, need.norm_w),
            Claim::of("down", &weights.down, need.down),
            Claim::of("up", &weights.up, need.up),
        ];
        if let Some(inject) = &weights.inject {
            read.push(Claim::of("inject", inject, need.inject));
        }
        for (index, out) in written.iter().enumerate() {
            for input in &read {
                require_apart(out, input)?;
            }
            for other in &written[index + 1..] {
                require_apart(out, other)?;
            }
        }
        let inject = weights
            .inject
            .as_ref()
            .map(|range| self.raw_range(range))
            .transpose()?;
        let gate = call
            .gate
            .as_ref()
            .map(|range| self.raw_range(range))
            .transpose()?;
        Ok(HyperReadRaw {
            dims: call.shape.dims(),
            rank: call.shape.rank,
            dtype: weights.dtype,
            eps: call.eps,
            x: self.raw_range(&call.x)?,
            norm_w: self.raw_range(&weights.norm)?,
            down: self.raw_range(&weights.down)?,
            up: self.raw_range(&weights.up)?,
            inject,
            normed: self.raw_range(&scratch.normed)?,
            low: self.raw_range(&scratch.low)?,
            mix: self.raw_range(&scratch.mix)?,
            mixed: self.raw_range(&call.mixed)?,
            gate,
        })
    }

    fn plan_hyper_apply<'b>(
        &self,
        call: &HyperApplyCall<'b>,
    ) -> Result<HyperApplyRaw<'b>, EngError> {
        let width = check_dims(call.shape)?;
        let shape = call.shape;
        let overflow = || {
            EngError::BadRequest(format!(
                "the CUDA hyper-connection shape {shape:?} overflows"
            ))
        };
        let floats = |count: usize, per_row: usize| {
            count
                .checked_mul(per_row)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(overflow)
        };
        let (x_bytes, gate_bytes, block_bytes) = (
            floats(shape.m, width)?,
            floats(shape.m, shape.branches)?,
            floats(shape.m, shape.hidden)?,
        );
        require_fit("x", &call.x, x_bytes, 4)?;
        require_fit("gate", &call.gate, gate_bytes, 4)?;
        require_fit("block", &call.block, block_bytes, 4)?;
        let x = Claim::of("x", &call.x, x_bytes);
        require_apart(&x, &Claim::of("gate", &call.gate, gate_bytes))?;
        require_apart(&x, &Claim::of("block", &call.block, block_bytes))?;
        Ok(HyperApplyRaw {
            dims: shape.dims(),
            x: self.raw_range(&call.x)?,
            gate: self.raw_range(&call.gate)?,
            block: self.raw_range(&call.block)?,
        })
    }
}

#[cfg(all(test, feature = "native-cuda"))]
mod tests;
