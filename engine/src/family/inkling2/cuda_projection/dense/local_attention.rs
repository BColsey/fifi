//! Queue the same one-row GEMVs as the host path, targeting resident attention.

use super::{
    ControlledAttentionInput, ControllerWeights, DenseStep, ResidentProjection, Slot, bad,
};
use crate::{
    EngError,
    kernels::cuda::{CudaContext, DeviceRange, DeviceStream, PinnedBuf},
};

#[allow(clippy::large_enum_variant)]
pub(in crate::family::inkling2::cuda_projection) enum LocalInputs<'a> {
    Plain([&'a [f32]; 3]),
    Controlled(
        [ControlledAttentionInput<'a>; 3],
        [ControllerWeights<'a>; 3],
        f32,
    ),
}

pub(in crate::family::inkling2::cuda_projection) struct LocalProjection<'a> {
    pub dense: &'a DenseStep,
    pub inputs: LocalInputs<'a>,
    pub weights: [ResidentProjection<'a>; 6],
}

pub(in crate::family::inkling2::cuda_projection) struct LocalTargets<'a> {
    pub projections: [DeviceRange<'a>; 6],
    pub controller_health: Option<DeviceRange<'a>>,
    pub stream: &'a DeviceStream,
    pub transfer: &'a mut PinnedBuf,
    pub upload_at: usize,
}

impl DenseStep {
    pub(in crate::family::inkling2::cuda_projection) fn local_inputs_fit(
        &self,
        hidden: usize,
        controlled: bool,
    ) -> bool {
        (1..=4096).contains(&hidden)
            && self
                .range(Slot::Output, 0, if controlled { 24 } else { 3 } * hidden)
                .is_ok()
    }

    pub(in crate::family::inkling2::cuda_projection) fn enqueue_local(
        &self,
        context: &CudaContext,
        inputs: LocalInputs<'_>,
        weights: [ResidentProjection<'_>; 6],
        targets: LocalTargets<'_>,
    ) -> Result<(), EngError> {
        let (rows, hidden, control) = match inputs {
            LocalInputs::Plain(rows) => (rows.to_vec(), rows[0].len(), None),
            LocalInputs::Controlled(inputs, weights, eps) => {
                let hidden = inputs[0].input.len();
                if inputs.iter().any(|row| {
                    row.gate.len() != 2 * hidden
                        || [row.input, row.message, row.gamma].map(<[f32]>::len) != [hidden; 3]
                }) {
                    return Err(bad("invalid CUDA local controller input shape"));
                }
                let staged = inputs
                    .iter()
                    .map(|row| row.gate)
                    .chain(
                        inputs
                            .iter()
                            .flat_map(|row| [row.input, row.message, row.gamma]),
                    )
                    .collect();
                (
                    staged,
                    hidden,
                    Some((self.controlled_attention_calls(hidden, weights)?, eps)),
                )
            }
        };
        let (start, upload) = if control.is_some() { (21, 15) } else { (0, 3) };
        if !rows
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite())
        {
            return Err(bad("CUDA local attention requires finite input"));
        }
        let packed: Vec<_> = rows
            .into_iter()
            .flatten()
            .map(|v| v.to_ne_bytes())
            .collect();
        targets
            .transfer
            .write(targets.upload_at, packed.as_flattened())?;
        context.copy_from_pinned(
            targets.stream,
            targets.transfer,
            targets.upload_at,
            self.range(Slot::Output, 0, upload * hidden)?,
        )?;
        if let Some((calls, eps)) = control {
            context.dense_gemv_group(targets.stream, &calls)?;
            context.inkling_controller_norm(
                targets.stream,
                self.range(Slot::Output, 6 * hidden, 18 * hidden)?,
                targets
                    .controller_health
                    .ok_or_else(|| bad("missing local controller health"))?,
                (hidden, eps),
            )?;
        }
        let calls = weights
            .into_iter()
            .zip(targets.projections)
            .map(|(p, y)| {
                let x = self.range(Slot::Output, (start + p.input) * hidden, hidden)?;
                self.call((&p.weight.codes, p.weight.shape, p.dtype), x, y)
            })
            .collect::<Result<Vec<_>, _>>()?;
        context.dense_gemv_group(targets.stream, &calls)
    }
}
