//! Prompt visits without a message bank, run on the device prompt streams. The depth mixes,
//! norms, attention, residual, MLP input, experts and both output convolutions stay on the
//! device; the host reads the router logits to route, and the K/V rows it mirrors.

use super::super::prompt_streams::{MixTarget, PromptStreams};
use super::{
    Attention, CpuForward, EngError, ExpertPrefillRows, GgufTextWeights, LayerVisit, PromptRow,
    VisitState, attention_norm_name, bad,
};
use crate::family::inkling2::{cuda_projection::gguf::attention::prefill, switches::Switch};
use crate::kernels::cuda::{
    DeviceBuffer, DeviceRange, DeviceStream, resident::inkling_stream::MAX_BLOCKS,
};

const WORD: usize = size_of::<f32>();
/// Constant rows per visit: three attention queries and norms, the MLP query and norm.
const CONSTANTS: usize = 8;

/// A prompt's device streams and the planes its visits reuse.
pub(super) struct DevicePrompt {
    streams: PromptStreams,
    stream: DeviceStream,
    /// Each row's three attention inputs, packed.
    inputs: DeviceBuffer,
    /// Each row's MLP input.
    mlp: DeviceBuffer,
    constants: DeviceBuffer,
}

impl DevicePrompt {
    /// Uploads the rows' streams, or declines when the device path is off or does not fit.
    /// `seals` is how many more blocks the coming visits may seal.
    pub(super) fn open(
        forward: &CpuForward,
        weights: &GgufTextWeights,
        rows: &[PromptRow<'_>],
        seals: usize,
    ) -> Result<Option<Self>, EngError> {
        let Some((context, _)) = weights.resident_attention() else {
            return Ok(None);
        };
        let Some(first) = rows.first() else {
            return Ok(None);
        };
        let capacity = (first.stream.block_count() + seals).min(MAX_BLOCKS);
        if !Switch::PrefillDevicePrompt.on()
            || !prefill::enabled()
            || forward.draft_window_enabled()
            || rows.len() < 2
            || !forward.taps.is_empty()
            || rows
                .iter()
                .any(|row| row.bank.is_some() || row.journal.is_some())
        {
            return Ok(None);
        }
        let width = forward.hidden();
        let opened = (|| {
            let stream = context.stream()?;
            let streams: Vec<_> = rows.iter().map(|row| &row.stream).collect();
            Ok::<_, EngError>(Self {
                streams: PromptStreams::upload(context, &stream, &streams, capacity)?,
                inputs: context.allocate(rows.len() * 3 * width * WORD)?,
                mlp: context.allocate(rows.len() * width * WORD)?,
                constants: context.allocate(CONSTANTS * width * WORD)?,
                stream,
            })
        })();
        match opened {
            Ok(prompt) => Ok(Some(prompt)),
            Err(EngError::MemoryPlan(_)) => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Writes the device streams back into the rows.
    pub(super) fn materialize(&self, rows: &mut [PromptRow<'_>]) -> Result<(), EngError> {
        let streams = self.streams.download(&self.stream)?;
        if streams.len() != rows.len() {
            return Err(bad("device prompt streams lost a row"));
        }
        for (row, stream) in rows.iter_mut().zip(streams) {
            row.stream = stream;
        }
        Ok(())
    }

    fn constant(&self, index: usize, width: usize) -> Result<DeviceRange<'_>, EngError> {
        self.constants.range(index * width * WORD, width * WORD)
    }
}

impl CpuForward {
    /// Runs an unbanked prompt visit on the device prompt streams. Returns false, with nothing
    /// advanced, when the visit cannot run there.
    pub(super) fn device_prompt_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        prompt: &mut DevicePrompt,
        weights: &GgufTextWeights,
        (layer, prefix): (&str, &str),
    ) -> Result<bool, EngError> {
        let rows = prompt.streams.rows();
        let width = self.hidden();
        let physical = visit.weight_layer;
        let experts = self.spec.n_routed_experts as usize;
        let router = format!("{prefix}.gate.weight");
        let names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let names = names.each_ref().map(String::as_str);
        let caps = [self.math.mlp_gate_softcap, self.math.mlp_up_softcap];
        let seal = physical.is_multiple_of(self.spec.n_layers_per_attn_res_block);
        if (seal && !prompt.streams.has_room())
            || weights.resident_prefill_router_shape(physical as u32, &router, rows)
                != Some([experts, width])
        {
            return Ok(false);
        }
        let Some((context, attention)) = weights.resident_attention() else {
            return Ok(false);
        };
        let Some(bridge) = self.prefill_bridge(layer, visit, prefill::resident_query_enabled())?
        else {
            return Ok(false);
        };
        let keep = bridge.keep();
        let widths = [width; 3];
        let Some(projector) = self.attention_requests(layer, visit.attention, |calls| {
            weights.prefill_projection(widths, calls)
        })?
        else {
            return Ok(false);
        };
        let capacity = bridge
            .tile_capacity()
            .min(projector.capacity_for(widths, keep)?);
        if capacity < 2 {
            return Ok(false);
        }
        let Some(mut conv) = bridge.admit_conv(self, state, rows)? else {
            return Ok(false);
        };
        let Some(mut mlp_conv) = self.admit_prompt_mlp_conv(state, layer, rows)? else {
            return Ok(false);
        };
        let dims = [rows, self.spec.num_experts_per_tok as usize, width, experts];
        if !weights.admit_resident_prefill_experts(names, physical as u32, dims, caps)? {
            return Ok(false);
        }
        self.upload_visit_constants(prompt, visit, layer)?;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        let eps = self.math.rms_norm_eps;
        // From here the visit advances state; a failure is an error, not a decline.
        let local = visit.attention == Attention::Local;
        let row_words = 3 * width;
        for role in 0..3 {
            // Local roles share the first role's input, as `attention_inputs_scalar` does.
            let source = if local { 0 } else { role };
            prompt.streams.mix(
                context,
                &prompt.stream,
                MixTarget {
                    query: (visit.weight_layer > 0)
                        .then(|| prompt.constant(source, width))
                        .transpose()?,
                    norm: Some((prompt.constant(3 + source, width)?, eps)),
                    first_block: 0,
                    out: prompt
                        .inputs
                        .range(role * width * WORD, ((rows - 1) * row_words + width) * WORD)?,
                    stride: row_words,
                },
            )?;
        }
        prompt.stream.synchronize()?;
        for first in (0..rows).step_by(capacity) {
            let count = (rows - first).min(capacity);
            let x = prompt
                .inputs
                .range(first * row_words * WORD, count * row_words * WORD)?;
            let (host, side) = projector.tile_device(x, (count, widths), keep)?;
            if !bridge
                .consume(self, state, host, side, Some((&mut conv, false)))?
                .is_empty()
            {
                return Err(bad("unexpected device prompt attention readback"));
            }
        }
        let attended = conv.convolved_plane()?;
        if seal {
            prompt.streams.seal(context, &prompt.stream, attended)?;
        } else {
            prompt.streams.add(context, &prompt.stream, attended)?;
        }
        let activations = prompt.mlp.range(0, rows * width * WORD)?;
        prompt.streams.mix(
            context,
            &prompt.stream,
            MixTarget {
                query: Some(prompt.constant(6, width)?),
                norm: Some((prompt.constant(7, width)?, eps)),
                first_block: 0,
                out: activations,
                stride: width,
            },
        )?;
        // The attention plane is free once the residual has read it.
        prompt.stream.synchronize()?;
        bridge.publish_conv_tail(state, conv)?;
        let routes = {
            let _span = self.phase_span(super::super::Work::Router);
            let logits = weights.resident_prefill_router_logits(&router, activations)?;
            self.route_prefill_logits(physical, &logits, &bias, rows)?
        };
        let ids: Vec<_> = routes
            .iter()
            .map(|route| route.selected.as_slice())
            .collect();
        {
            let _span = self.phase_span(super::super::Work::Experts);
            weights
                .prefill_device_experts(
                    names,
                    physical as u32,
                    (
                        ExpertPrefillRows::Resident {
                            ids: &ids,
                            hidden: width,
                            activations,
                        },
                        Some(mlp_conv.expert_destination()?),
                    ),
                    caps,
                    |row, id| routes[row].weight(id, self.math.route_scale),
                )?
                .ok_or_else(|| bad("admitted Inkling device experts declined a prompt visit"))?;
        }
        attention.finish_prefill_expert_conv(context, &mut mlp_conv)?;
        prompt
            .streams
            .add(context, &prompt.stream, mlp_conv.convolved_plane()?)?;
        prompt.stream.synchronize()?;
        prompt.streams.check(&prompt.stream)?;
        self.publish_prompt_mlp_tail(state, mlp_conv)?;
        static USED: std::sync::Once = std::sync::Once::new();
        USED.call_once(|| eprintln!("Inkling CUDA kept prompt depth streams on the device"));
        Ok(true)
    }

    /// The visit's attention queries and norms (one role on local visits) and its MLP query and
    /// norm, in the order [`DevicePrompt::constant`] reads them.
    fn upload_visit_constants(
        &self,
        prompt: &DevicePrompt,
        visit: LayerVisit,
        layer: &str,
    ) -> Result<(), EngError> {
        let width = self.hidden();
        let local = visit.attention == Attention::Local;
        let roles = if local { 1 } else { 3 };
        let physical = visit.weight_layer;
        let mut words = vec![0.0; CONSTANTS * width];
        let mut put = |index: usize, values: &[f32]| {
            words[index * width..][..width].copy_from_slice(values);
        };
        if let Some(query) = physical.checked_sub(1) {
            for role in 0..roles {
                let name = format!("model.llm.attn_res.attn_queries.{query}");
                put(role, &self.weights.row(&name, role, width)?);
            }
        }
        for role in 0..roles {
            put(
                3 + role,
                &self
                    .weights
                    .vec(&attention_norm_name(layer, local, role), width)?,
            );
        }
        put(
            6,
            &self
                .weights
                .vec(&format!("model.llm.attn_res.mlp_queries.{physical}"), width)?,
        );
        put(
            7,
            &self
                .weights
                .vec(&format!("{layer}.mlp_norm.weight"), width)?,
        );
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
        prompt.constants.write_at(&prompt.stream, 0, &bytes)
    }
}
