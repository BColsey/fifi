//! Keep a banked prefill MLP's normalized rows on device until expert dispatch.
//! Attention, message mixing, route selection and final residuals stay on the host.

use crate::family::inkling2::switches::Switch;
use std::{borrow::Cow, sync::Once};

use super::{CpuForward, GgufTextWeights, PromptRow, Role, bad, control_names};
use crate::{
    EngError,
    family::inkling2::{
        cpu_forward::MIX_EPS,
        cuda_projection::gguf::{ExpertPrefillRows, ResidentController, ResidentMlpController},
        phase_profile::Work,
    },
};

pub(super) fn enabled() -> bool {
    Switch::ResidentPrefillMlp.on()
}

#[allow(clippy::large_enum_variant)]
#[derive(Default)]
pub(super) enum Workspace {
    #[default]
    Unallocated,
    Ready(ResidentController),
    Declined,
}

pub(super) struct Visit<'a> {
    physical: u32,
    weights: &'a GgufTextWeights,
    controller: ResidentMlpController<'a>,
    scratch: &'a mut ResidentController,
    query: Cow<'a, [f32]>,
    gamma: Cow<'a, [f32]>,
    router: String,
    bias: Cow<'a, [f32]>,
    experts: [String; 2],
    caps: [f32; 2],
}

impl Workspace {
    pub(super) fn admit<'a>(
        &'a mut self,
        forward: &'a CpuForward,
        weights: &'a GgufTextWeights,
        (physical, layer, prefix): (usize, &str, &str),
        rows: usize,
    ) -> Result<Option<Visit<'a>>, EngError> {
        if matches!(self, Self::Declined) {
            return Ok(None);
        }
        let Some(cuda) = weights.cuda() else {
            return Ok(None);
        };
        let width = forward.hidden();
        let rank = forward.spec.core.gate_rank as usize;
        let n_experts = forward.spec.n_routed_experts as usize;
        let top_k = forward.spec.num_experts_per_tok as usize;
        let shape = [rows, width, rank];
        let router = format!("{prefix}.gate.weight");
        if !forward.math.rms_norm_eps.is_finite()
            || forward.math.rms_norm_eps <= 0.0
            || weights.resident_prefill_router_shape(physical as u32, &router, rows)
                != Some([n_experts, width])
        {
            return Ok(None);
        }
        let control = control_names(layer, Role::Mlp);
        let Some(controller) =
            cuda.resident_mlp_controller(control.each_ref().map(String::as_str), shape)?
        else {
            return Ok(None);
        };
        let experts = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let caps = [forward.math.mlp_gate_softcap, forward.math.mlp_up_softcap];
        if !weights.admit_resident_prefill_experts(
            experts.each_ref().map(String::as_str),
            physical as u32,
            [rows, top_k, width, n_experts],
            caps,
        )? {
            return Ok(None);
        }
        if matches!(self, Self::Unallocated) {
            *self = match ResidentController::allocate(cuda.context(), shape)? {
                Some(workspace) => Self::Ready(workspace),
                None => Self::Declined,
            };
        }
        let Self::Ready(scratch) = self else {
            return Ok(None);
        };
        if !scratch.fits(shape) {
            return Ok(None);
        }
        Ok(Some(Visit {
            physical: physical as u32,
            weights,
            controller,
            scratch,
            query: forward.control_query(physical, Role::Mlp)?,
            gamma: forward
                .weights
                .vec(&format!("{layer}.mlp_norm.weight"), width)?,
            router,
            bias: forward
                .weights
                .vec(&format!("{prefix}.gate.bias"), n_experts)?,
            experts,
            caps,
        }))
    }
}

impl Visit<'_> {
    pub(super) fn execute(
        self,
        forward: &CpuForward,
        rows: &[PromptRow<'_>],
        inputs: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let control = forward.phase_span(Work::Controller);
        let messages = {
            let _span = forward.phase_span(Work::MlpControlPrepare);
            rows.iter()
                .map(|row| {
                    let bank = row
                        .bank
                        .as_ref()
                        .ok_or_else(|| bad("Inkling resident MLP lost its message bank"))?;
                    Ok(bank.stream.mix(bank.first_block, &self.query))
                })
                .collect::<Result<Vec<_>, EngError>>()?
        };
        let tile = self.scratch.tile_rows();
        let activations = {
            let _span = forward.phase_span(Work::MlpControlProject);
            self.controller.run(
                self.scratch,
                (inputs, &messages, &self.gamma),
                (MIX_EPS, forward.math.rms_norm_eps),
            )?
        };
        drop(control);
        let routes = {
            let _span = forward.phase_span(Work::Router);
            let logits = self
                .weights
                .resident_prefill_router_logits(&self.router, activations)?;
            forward.route_prefill_logits(self.physical as usize, &logits, &self.bias, rows.len())?
        };
        let ids: Vec<_> = routes
            .iter()
            .map(|route| route.selected.as_slice())
            .collect();
        let outputs = {
            let _span = forward.phase_span(Work::Experts);
            self.weights
                .prefill_device_experts(
                    self.experts.each_ref().map(String::as_str),
                    self.physical,
                    (
                        ExpertPrefillRows::Resident {
                            ids: &ids,
                            hidden: forward.hidden(),
                            activations,
                        },
                        None,
                    ),
                    self.caps,
                    |row, id| routes[row].weight(id, forward.math.route_scale),
                )?
                .ok_or_else(|| bad("previously admitted Inkling resident MLP experts declined"))?
        };
        if outputs.len() != rows.len() {
            return Err(bad("incorrect Inkling resident MLP output rows"));
        }
        static USED: Once = Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA used resident prefill MLP"));
        if forward.phase_profile.is_some() {
            eprintln!(
                "Inkling CUDA resident prefill MLP {}",
                serde_json::json!({
                    "layer": self.physical, "rows": rows.len(), "tile_rows": tile,
                    "tiles": rows.len().div_ceil(tile),
                    "coefficient_d2h_bytes_avoided": 2 * activations.bytes(),
                    "router_input_h2d_bytes_avoided": activations.bytes(),
                    "input_h2d_bytes": 2 * activations.bytes() + self.gamma.len() * size_of::<f32>(),
                    "activation_d2d_bytes": activations.bytes(),
                    "health_d2h_bytes": 3 * rows.len() * size_of::<f32>(),
                })
            );
        }
        Ok(outputs)
    }
}
