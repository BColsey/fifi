//! Cross-pass controllers and their attention projections share a staging transaction.

use super::{
    ControllerWeights, DenseGemvCall, DenseStep, ResidentProjection, Slot, bad, split_projections,
};
use crate::{EngError, kernels::cuda::CudaContext};

#[cfg(test)]
mod tests;

pub(in crate::family::inkling2) struct ControlledAttentionInput<'a> {
    pub gate: &'a [f32],
    pub input: &'a [f32],
    pub message: &'a [f32],
    pub gamma: &'a [f32],
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention<const N: usize>(
        &self,
        context: &CudaContext,
        inputs: [ControlledAttentionInput<'_>; 3],
        controllers: [ControllerWeights<'_>; 3],
        weights: [Option<ResidentProjection<'_>>; N],
        eps: f32,
    ) -> Result<Option<[Vec<f32>; N]>, EngError> {
        let hidden = inputs[0].input.len();
        if !(1..=4096).contains(&hidden) {
            return Ok(None);
        }
        // The idle logits slot packs 6 gate rows, 9 raw input/message/gamma
        // rows, 6 controller outputs, 3 normalized inputs, then the readback.
        let stage = 15 * hidden;
        let health_at = 24 * hidden;
        let out_at = health_at + 3;
        let rows: usize = weights.iter().flatten().map(|p| p.weight.shape[0]).sum();
        let Ok(readback) = self.range(Slot::Output, health_at, 3 + rows) else {
            return Ok(None);
        };
        if inputs.iter().any(|row| {
            row.gate.len() != 2 * hidden
                || [row.input, row.message, row.gamma].map(<[f32]>::len) != [hidden; 3]
        }) {
            return Err(bad("invalid CUDA controlled attention input shape"));
        }
        let upload = self.range(Slot::Output, 0, stage)?;
        let workspace = self.range(Slot::Output, 6 * hidden, 18 * hidden)?;
        let health = self.range(Slot::Output, health_at, 3)?;
        let control = self.controlled_attention_calls(hidden, controllers)?;
        let (projections, _) =
            self.group_calls([21, 22, 23].map(|i| i * hidden), out_at, weights)?;
        let staged: Vec<_> = inputs
            .iter()
            .map(|row| row.gate)
            .chain(
                inputs
                    .iter()
                    .flat_map(|row| [row.input, row.message, row.gamma]),
            )
            .collect();
        let values = self.run_rows(context, &self.stream, &staged, [upload, readback], || {
            context.dense_gemv_group(&self.stream, &control)?;
            context.inkling_controller_norm(&self.stream, workspace, health, (hidden, eps))?;
            context.dense_gemv_group(&self.stream, &projections)
        })?;
        Ok(Some(split_projections(values.into_iter().skip(3), weights)))
    }

    fn controlled_attention_calls<'a>(
        &'a self,
        hidden: usize,
        controllers: [ControllerWeights<'a>; 3],
    ) -> Result<Vec<DenseGemvCall<'a>>, EngError> {
        let reduced = self.range(Slot::Reduced, 0, controllers[0][0].0.shape[0])?;
        let mut calls = Vec::with_capacity(9);
        for (role, weights) in controllers.into_iter().enumerate() {
            let gate = self.range(Slot::Output, role * 2 * hidden, 2 * hidden)?;
            let gain = self.range(Slot::Output, (15 + 2 * role) * hidden, hidden)?;
            let forget = self.range(Slot::Output, (16 + 2 * role) * hidden, hidden)?;
            calls.extend(self.controller_calls(weights, [gate, reduced, gain, forget])?);
        }
        Ok(calls)
    }
}
