//! Cross-pass controllers and their attention projections share a staging transaction.

use super::{
    ControllerWeights, DenseGemvCall, DenseGemvShape, DenseStep, ResidentProjection, Slot, bad,
    dense_gemv_bytes, split_projections,
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
    // Staged transaction: the grouped prefill caller is wired separately after quality checks.
    #[allow(dead_code)]
    pub(in crate::family::inkling2::cuda_projection) fn controlled_attention_batch<
        const N: usize,
    >(
        &self,
        context: &CudaContext,
        inputs: &[[ControlledAttentionInput<'_>; 3]],
        controllers: [ControllerWeights<'_>; 3],
        weights: [Option<ResidentProjection<'_>>; N],
        eps: f32,
    ) -> Result<Option<Vec<[Vec<f32>; N]>>, EngError> {
        let Some(first) = inputs.first() else {
            return Ok(None);
        };
        let hidden = first[0].input.len();
        let rank = controllers[0][0].0.shape[0];
        let m = inputs.len();
        if !(1..=4096).contains(&hidden)
            || rank == 0
            || rank > 4096
            || m > DenseGemvShape::MAX_ACTIVATION_ROWS
            || m > 512
        {
            return Ok(None);
        }
        if controllers.iter().any(|w| {
            w.map(|(x, _)| x.shape) != [[rank, 2 * hidden], [hidden, rank], [hidden, rank]]
        }) || inputs.iter().flatten().any(|r| {
            r.gate.len() != 2 * hidden
                || [r.input.len(), r.message.len(), r.gamma.len()] != [hidden; 3]
        }) || weights
            .iter()
            .flatten()
            .any(|p| p.input >= 3 || p.weight.shape[1] != hidden)
        {
            return Err(bad("invalid batched CUDA controlled attention shape"));
        }
        let widths = weights.map(|p| p.map_or(0, |p| p.weight.shape[0]));
        let output_width = widths
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .ok_or_else(|| bad("batched CUDA attention output size overflows"))?;
        let per_row = (24 * hidden + 3 * rank + 3)
            .checked_add(output_width)
            .ok_or_else(|| bad("batched CUDA attention workspace overflows"))?;
        let capacity =
            self.arena.tensor(Slot::Output as usize)?.bytes() / size_of::<f32>() / per_row;
        if m > capacity {
            return Ok(None);
        }
        let gates = 6 * m * hidden;
        let raw = 9 * m * hidden;
        let norm = gates;
        let reduced = 24 * m * hidden;
        let health_at = reduced + 3 * m * rank;
        let out_at = health_at + 3 * m;
        let range = |offset, len| self.range(Slot::Output, offset, len);
        let upload = range(0, gates + raw)?;
        let workspace = range(norm, 18 * m * hidden)?;
        let health = range(health_at, 3 * m)?;
        let readback = range(health_at, 3 * m + m * output_width)?;
        let mut control = Vec::with_capacity(9);
        for (role, w) in controllers.into_iter().enumerate() {
            let gate = role * m * 2 * hidden;
            let mid = reduced + role * m * rank;
            let gain = gates + raw + role * m * 2 * hidden;
            control.push(self.batch_call(
                w[0],
                DenseGemvShape::packed(m, rank, 2 * hidden),
                [gate, mid],
            )?);
            let mut shape = DenseGemvShape::packed(m, hidden, rank);
            shape.ld_y = 2 * hidden;
            control.push(self.batch_call(w[1], shape, [mid, gain])?);
            control.push(self.batch_call(w[2], shape, [mid, gain + hidden])?);
        }
        let mut projections = Vec::with_capacity(N);
        let mut output_offset = 0;
        for p in weights.into_iter().flatten() {
            let mut shape = DenseGemvShape::packed(m, p.weight.shape[0], hidden);
            shape.ld_y = output_width;
            let x = gates + 15 * m * hidden + p.input * m * hidden;
            projections.push(self.batch_call(
                (p.weight, p.dtype),
                shape,
                [x, out_at + output_offset],
            )?);
            output_offset += p.weight.shape[0];
        }
        let mut staged = Vec::with_capacity(12 * m);
        for role in 0..3 {
            for row in inputs {
                staged.push(row[role].gate);
            }
        }
        for role in 0..3 {
            for row in inputs {
                staged.extend([row[role].input, row[role].message, row[role].gamma]);
            }
        }
        let values = self.run_rows(context, &self.stream, &staged, [upload, readback], || {
            context.dense_gemv_group(&self.stream, &control)?;
            context.inkling_controller_norm_roles(
                &self.stream,
                workspace,
                health,
                ([hidden, 3 * m], eps),
            )?;
            context.dense_gemv_group(&self.stream, &projections)
        })?;
        let mut values = values.into_iter().skip(3 * m);
        Ok(Some(
            (0..m)
                .map(|_| split_projections(&mut values, weights))
                .collect(),
        ))
    }

    fn batch_call<'a>(
        &'a self,
        (weight, dtype): (&'a super::Resident, super::DenseDtype),
        shape: DenseGemvShape,
        [x, y]: [usize; 2],
    ) -> Result<DenseGemvCall<'a>, EngError> {
        let need = dense_gemv_bytes(shape, dtype)?;
        Ok(DenseGemvCall {
            shape,
            dtype,
            w: weight.codes.range(0, need.w)?,
            x: self.range(Slot::Output, x, need.x / size_of::<f32>())?,
            y: self.range(Slot::Output, y, need.y / size_of::<f32>())?,
        })
    }

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

    pub(super) fn controlled_attention_calls<'a>(
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
