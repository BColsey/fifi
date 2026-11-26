//! Core pass visits that read the message bank, run on the device prompt streams. Each role's
//! input and bank message are mixed into controller pairs on the device; the controllers, the
//! attention, the residual, the MLP controller and the experts read them there.

use super::super::super::{
    ATTENTION_ROLES, control_names,
    controller_batch::{controller_batch_enabled, multirow_controllers_enabled, name_refs},
    prompt_streams::MixTarget,
};
use super::super::resident_mlp;
use super::{
    Attention, Controls, CpuForward, DevicePrompt, EngError, GgufTextWeights, LayerVisit,
    VisitState, WORD, bad, constant, prefill,
};
use crate::family::inkling2::{
    cpu_forward::MIX_EPS, cuda_projection::gguf::ControlledRows, phase_profile::Work,
    switches::Switch,
};
use crate::kernels::cuda::CudaContext;

/// Whether banked visits may run on the device: their switch and the switches of the host's
/// chained controller attention, which they replace.
pub(in super::super) fn banked_enabled() -> bool {
    Switch::PrefillDeviceBanked.on()
        && Switch::GroupedChainAttention.on()
        && controller_batch_enabled()
        && multirow_controllers_enabled()
        && prefill::resident_query_enabled()
        && std::env::var("FIFI_INKLING_DEBUG_CHAIN_COMPARE").as_deref() != Ok("1")
}

impl DevicePrompt {
    /// The banked visits' controller pairs, allocated once; none when they do not fit.
    fn controls(&mut self, context: &CudaContext, width: usize) -> Result<bool, EngError> {
        if self.controls.is_some() {
            return Ok(true);
        }
        let rows = self.streams.rows();
        let allocate = || {
            Ok::<_, EngError>(Controls {
                pairs: context.allocate(3 * rows * 2 * width * WORD)?,
                health: context.allocate(3 * rows * 2 * WORD)?,
            })
        };
        match allocate() {
            Ok(controls) => {
                self.controls = Some(controls);
                Ok(true)
            }
            Err(EngError::MemoryPlan(_)) => Ok(false),
            Err(error) => Err(error),
        }
    }
}

impl CpuForward {
    /// Runs a visit that reads the message bank on the device prompt streams. Returns false,
    /// with nothing advanced, when the visit cannot run there.
    pub(in super::super) fn device_banked_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        prompt: &mut DevicePrompt,
        weights: &GgufTextWeights,
        (layer, prefix, mlp): (&str, &str, &mut resident_mlp::Workspace),
    ) -> Result<bool, EngError> {
        let rows = prompt.streams.rows();
        let width = self.hidden();
        let physical = visit.weight_layer;
        let seal = physical.is_multiple_of(self.spec.n_layers_per_attn_res_block);
        if prompt.bank.is_none() || (seal && !prompt.streams.has_room()) || !banked_enabled() {
            return Ok(false);
        }
        let Some((context, attention)) = weights.resident_attention() else {
            return Ok(false);
        };
        let shape = [self.spec.core.gate_rank as usize, width];
        let names = ATTENTION_ROLES.map(|(role, _)| control_names(layer, role));
        let Some(controllers) = weights.attention_controllers(name_refs(&names), shape, rows)?
        else {
            return Ok(false);
        };
        let Some(bridge) = self.prefill_bridge(layer, visit, true)? else {
            return Ok(false);
        };
        if !prompt.controls(context, width)? {
            return Ok(false);
        }
        let Some(mut conv) = bridge.admit_conv(self, state, rows)? else {
            return Ok(false);
        };
        let Some(mut mlp_conv) = self.admit_prompt_mlp_conv(state, layer, rows)? else {
            return Ok(false);
        };
        let Some(mlp) = mlp.admit(self, weights, (physical, layer, prefix), rows)? else {
            return Ok(false);
        };
        self.upload_visit_constants(prompt, visit, layer, true)?;
        let local = visit.attention == Attention::Local;
        let (Some(bank), Some(controls)) = (&prompt.bank, &prompt.controls) else {
            return Err(bad("the device prompt lost its bank"));
        };
        let control_span = self.phase_span(Work::Controller);
        let prepare_span = self.phase_span(Work::AttentionControlPrepare);
        let pair_words = 2 * width;
        let plane = rows * pair_words;
        let mixed = |role: usize, side: usize| {
            let at = (role * plane + side * width) * WORD;
            controls
                .pairs
                .range(at, ((rows - 1) * pair_words + width) * WORD)
        };
        // Each role's [input, message] pair: the input as `prepare_attention_rows` mixes it
        // (local roles share the first role's), the message as `control_input_from` does.
        for role in 0..3 {
            let source = if local { 0 } else { role };
            prompt.streams.mix(
                context,
                &prompt.stream,
                MixTarget {
                    query: (physical > 0)
                        .then(|| constant(&prompt.constants, source, width))
                        .transpose()?,
                    norm: None,
                    first_block: 0,
                    out: mixed(role, 0)?,
                    stride: pair_words,
                },
            )?;
            bank.streams.mix(
                context,
                &prompt.stream,
                MixTarget {
                    query: Some(constant(&prompt.constants, 8 + role, width)?),
                    norm: None,
                    first_block: bank.first_block,
                    out: mixed(role, 1)?,
                    stride: pair_words,
                },
            )?;
        }
        prompt.stream.synchronize()?;
        drop(prepare_span);
        let gamma =
            |role: usize| constant(&prompt.constants, 3 + if local { 0 } else { role }, width);
        let control = ControlledRows {
            pairs: controls.pairs.range(0, 3 * plane * WORD)?,
            gammas: [gamma(0)?, gamma(1)?, gamma(2)?],
            health: controls.health.range(0, 3 * rows * 2 * WORD)?,
            rows,
            hidden: width,
            gate_eps: MIX_EPS,
        };
        let eps = self.math.rms_norm_eps;
        let keep = bridge.keep();
        let admitted = self.attention_requests(layer, visit.attention, |calls| {
            controllers.grouped_chained_device_tiles(&control, calls, (eps, keep), |host, range| {
                let range = range.ok_or_else(|| bad("missing resident Inkling query tile"))?;
                if !bridge
                    .consume(self, state, host, range, Some((&mut conv, false)))?
                    .is_empty()
                {
                    return Err(bad("unexpected device banked attention readback"));
                }
                Ok(())
            })
        })?;
        drop(control_span);
        if !admitted {
            // Declined before its first launch.
            return Ok(false);
        }
        // From here the visit advanced state; a failure is an error, not a decline.
        let attended = conv.convolved_plane()?;
        if seal {
            prompt.streams.seal(context, &prompt.stream, attended)?;
        } else {
            prompt.streams.add(context, &prompt.stream, attended)?;
        }
        // The attention plane is free once the residual has read it.
        prompt.stream.synchronize()?;
        bridge.publish_conv_tail(state, conv)?;
        let destination = mlp_conv.expert_destination()?;
        mlp.execute_device(
            self,
            rows,
            |[pairs, gamma]| {
                let half = |side: usize| {
                    pairs.slice(
                        side * width * WORD,
                        ((rows - 1) * pair_words + width) * WORD,
                    )
                };
                prompt.streams.mix(
                    context,
                    &prompt.stream,
                    MixTarget {
                        query: Some(constant(&prompt.constants, 6, width)?),
                        norm: None,
                        first_block: 0,
                        out: half(0)?,
                        stride: pair_words,
                    },
                )?;
                bank.streams.mix(
                    context,
                    &prompt.stream,
                    MixTarget {
                        query: Some(constant(&prompt.constants, 11, width)?),
                        norm: None,
                        first_block: bank.first_block,
                        out: half(1)?,
                        stride: pair_words,
                    },
                )?;
                context.copy_device(
                    &prompt.stream,
                    constant(&prompt.constants, 7, width)?,
                    gamma,
                )?;
                prompt.stream.synchronize()
            },
            destination,
        )?;
        attention.finish_prefill_expert_conv(context, &mut mlp_conv)?;
        prompt
            .streams
            .add(context, &prompt.stream, mlp_conv.convolved_plane()?)?;
        prompt.stream.synchronize()?;
        prompt.streams.check(&prompt.stream)?;
        bank.streams.check(&prompt.stream)?;
        check_health(controls, &prompt.stream)?;
        self.publish_prompt_mlp_tail(state, mlp_conv)?;
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA kept banked prompt visits on the device"));
        Ok(true)
    }
}

/// Fails when a gate the controllers read was not finite.
fn check_health(
    controls: &Controls,
    stream: &crate::kernels::cuda::DeviceStream,
) -> Result<(), EngError> {
    let mut bytes = vec![0; controls.health.bytes()];
    controls.health.read_at(stream, 0, &mut bytes)?;
    if bytes
        .as_chunks::<WORD>()
        .0
        .iter()
        .any(|w| f32::from_ne_bytes(*w) != 0.0)
    {
        return Err(bad("non-finite Inkling device controller gate"));
    }
    Ok(())
}
