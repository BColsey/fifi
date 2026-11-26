//! Batch independent Q/K/V cross-pass controllers before host attention.

use crate::family::inkling2::switches::Switch;
use std::sync::Once;

use super::{
    ATTENTION_ROLES, Attention, ControlInput, CpuForward, LayerVisit, MessageBank,
    PROMPT_ROW_SPLIT, Stream, VisitState, Weights, Work, apply_control, attention_norm_name, bad,
    control_input_from, control_names,
};
use crate::EngError;
use crate::family::host_rows::map_rows;
use crate::family::inkling2::{
    cpu_attention::{AttentionProjection, note_resident_queries},
    cuda_projection::gguf::{
        AttentionControllers, ControlledAttentionInput, LocalDecode, attention::prefill,
        project_controller_roles,
    },
};

type AttentionInputs = [Vec<f32>; 3];

pub(super) enum ChainedAttention {
    Host(Vec<AttentionProjection>),
    Resident(Vec<Vec<f32>>),
}

struct PreparedInput {
    input: Vec<f32>,
    control: ControlInput,
    gate_scales: [f32; 2],
}

impl CpuForward {
    pub(super) fn chained_attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        stream: &Stream,
        bank: &MessageBank,
    ) -> Result<Option<AttentionProjection>, EngError> {
        let projected = self.with_chained_attention_inputs(
            visit,
            layer,
            stream,
            bank,
            |controllers, inputs| {
                self.attention_requests(layer, visit.attention, |calls| {
                    controllers.chained_projections(inputs, calls, self.math.rms_norm_eps)
                })
            },
        )?;
        if projected.is_some() {
            static USED: Once = Once::new();
            USED.call_once(|| eprintln!("Inkling CUDA used chained controller attention input"));
        }
        Ok(projected)
    }

    pub(super) fn chained_local_decode(
        &self,
        visit: LayerVisit,
        layer: &str,
        state: &mut VisitState,
        stream: &Stream,
        bank: Option<&MessageBank>,
    ) -> Result<Option<Vec<f32>>, EngError> {
        if visit.attention != Attention::Local || self.draft_window_enabled() {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let Some(cuda) = weights.cuda() else {
            return Ok(None);
        };
        let cursor = (visit.state_slot, state.position());
        if !cuda
            .resident_attention()
            .is_some_and(|(_, attention)| attention.can_chain_local(cursor.0, cursor.1))
        {
            return Ok(None);
        }
        let read_weights = |[q, relative, k, v]: [usize; 4]| {
            let read = |name, len| self.weights.vec(&format!("{layer}.attn.{name}"), len);
            Ok([
                read("q_l2_clip.scale", q)?,
                read("rel_l2_clip.scale", relative)?,
                read("k_sconv.weight", k)?,
                read("v_sconv.weight", v)?,
            ])
        };
        let decode = LocalDecode {
            cursor,
            prep: self.local_decode_prep(state, &read_weights),
            output: weights.relative_attention_projection(
                &format!("{layer}.attn.wo_ud.weight"),
                [self.hidden(), self.local_query_channels()],
            )?,
        };
        let result = {
            let _span = self.phase_span(Work::DeviceAttention);
            if let Some(bank) = bank {
                self.with_chained_attention_inputs(
                    visit,
                    layer,
                    stream,
                    bank,
                    |controllers, inputs| {
                        self.attention_requests(layer, visit.attention, |calls| {
                            controllers.local_decode(inputs, calls, decode)
                        })
                    },
                )?
            } else {
                let inputs = self.attention_inputs(visit, layer, stream, None)?;
                self.attention_requests(layer, visit.attention, |calls| {
                    cuda.plain_local_decode(inputs.each_ref().map(Vec::as_slice), calls, decode)
                })?
            }
        };
        result
            .map(|result| self.finish_local_decode(layer, state, result))
            .transpose()
    }

    fn with_chained_attention_inputs<R>(
        &self,
        visit: LayerVisit,
        layer: &str,
        stream: &Stream,
        bank: &MessageBank,
        run: impl FnOnce(
            &AttentionControllers<'_>,
            [ControlledAttentionInput<'_>; 3],
        ) -> Result<Option<R>, EngError>,
    ) -> Result<Option<R>, EngError> {
        if !controller_batch_enabled() || !Switch::ChainAttentionInput.on() {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let names = ATTENTION_ROLES.map(|(role, _)| control_names(layer, role));
        let shape = [self.spec.core.gate_rank as usize, self.hidden()];
        let Some(controllers) = weights.attention_controllers(name_refs(&names), shape, 1)? else {
            return Ok(None);
        };
        let local = visit.attention == Attention::Local;
        let (prepared, gamma) = {
            let _span = self.phase_span(Work::Controller);
            let prepared = self.prepare_attention_controllers(visit, stream, bank)?;
            let gamma = |index| {
                self.weights
                    .vec(&attention_norm_name(layer, local, index), shape[1])
            };
            let gamma = if local {
                let norm = gamma(0)?;
                [Some(norm), None, None]
            } else {
                [Some(gamma(0)?), Some(gamma(1)?), Some(gamma(2)?)]
            };
            (prepared, gamma)
        };
        let inputs = std::array::from_fn(|i| ControlledAttentionInput {
            gate: &prepared[i].control.gate,
            input: &prepared[i].input,
            message: &prepared[i].control.message,
            gamma: gamma[i]
                .as_deref()
                .unwrap_or_else(|| gamma[0].as_deref().expect("query gamma")),
        });
        run(&controllers, inputs)
    }

    pub(super) fn grouped_chained_attention<'a>(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: impl ExactSizeIterator<Item = (&'a Stream, Option<&'a MessageBank>)>,
        state: &mut VisitState,
    ) -> Result<Option<ChainedAttention>, EngError> {
        if !Switch::GroupedChainAttention.on()
            || !controller_batch_enabled()
            || !multirow_controllers_enabled()
            || rows.len() < 2
        {
            return Ok(None);
        }
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        let shape = [self.spec.core.gate_rank as usize, self.hidden()];
        let names = ATTENTION_ROLES.map(|(role, _)| control_names(layer, role));
        let Some(controllers) =
            weights.attention_controllers(name_refs(&names), shape, rows.len())?
        else {
            return Ok(None);
        };
        let rows: Vec<_> = rows.collect();
        if rows.iter().any(|(_, bank)| bank.is_none()) {
            return Ok(None);
        }
        let _span = self.phase_span(Work::Controller);
        let prepared = {
            let _subspan = self.phase_span(Work::AttentionControlPrepare);
            let rows: Vec<_> = rows
                .iter()
                .map(|&(stream, bank)| (stream, bank.expect("banks checked")))
                .collect();
            self.prepare_attention_rows(visit, &rows)?
        };
        let local = visit.attention == Attention::Local;
        let gamma = |i| {
            self.weights
                .vec(&attention_norm_name(layer, local, i), shape[1])
        };
        let gamma = if local {
            [Some(gamma(0)?), None, None]
        } else {
            [Some(gamma(0)?), Some(gamma(1)?), Some(gamma(2)?)]
        };
        let inputs: Vec<_> = prepared
            .iter()
            .map(|row| {
                std::array::from_fn(|i| ControlledAttentionInput {
                    gate: &row[i].control.gate,
                    input: &row[i].input,
                    message: &row[i].control.message,
                    gamma: gamma[i]
                        .as_deref()
                        .unwrap_or_else(|| gamma[0].as_deref().expect("query gamma")),
                })
            })
            .collect();
        // The projection comparison requires all raw projections on the host.
        let keep_query = prefill::resident_query_enabled()
            && std::env::var("FIFI_INKLING_DEBUG_CHAIN_COMPARE").as_deref() != Ok("1");
        let bridge = keep_query
            .then(|| self.prefill_bridge(layer, visit, true))
            .transpose()?
            .flatten();
        let result =
            self.attention_requests(layer, visit.attention, |calls| -> Result<_, EngError> {
                if let Some(bridge) = &bridge {
                    let rebuild = Switch::PrefillRebuildGates.on();
                    let scales: Option<Vec<_>> = rebuild.then(|| {
                        prepared
                            .iter()
                            .map(|row| row.each_ref().map(|role| role.gate_scales))
                            .collect()
                    });
                    let mut output = Vec::with_capacity(inputs.len());
                    let mut conv = bridge.admit_conv(self, state, inputs.len())?;
                    let admitted = controllers.grouped_chained_tiles(
                        &inputs,
                        calls,
                        (self.math.rms_norm_eps, bridge.keep(), scales.as_deref()),
                        |host, range| {
                            let range =
                                range.ok_or_else(|| bad("missing resident Inkling query tile"))?;
                            output.extend(bridge.consume(
                                self,
                                state,
                                host,
                                range,
                                conv.as_mut().map(|conv| (conv, true)),
                            )?);
                            Ok(())
                        },
                    )?;
                    if admitted {
                        if let Some(conv) = conv {
                            bridge.publish_conv_tail(state, conv)?;
                        }
                        note_resident_queries(true);
                        return Ok(Some(ChainedAttention::Resident(output)));
                    }
                }
                Ok(controllers
                    .grouped_chained_projections(&inputs, calls, self.math.rms_norm_eps)?
                    .map(ChainedAttention::Host))
            })?;
        if result.is_some() {
            static USED: Once = Once::new();
            USED.call_once(|| eprintln!("Inkling CUDA used grouped chained controller attention"));
        }
        Ok(result)
    }

    pub(super) fn grouped_attention_inputs<'a>(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: impl ExactSizeIterator<Item = (&'a Stream, Option<&'a MessageBank>)>,
    ) -> Result<Option<Vec<AttentionInputs>>, EngError> {
        let mut rows = rows.peekable();
        let Weights::Gguf(weights) = &self.weights else {
            return Ok(None);
        };
        if !matches!(rows.peek(), Some((_, Some(_)))) || !controller_batch_enabled() {
            return Ok(None);
        }
        let shape = [self.spec.core.gate_rank as usize, self.hidden()];
        let names = ATTENTION_ROLES.map(|(role, _)| control_names(layer, role));
        let names = name_refs(&names);
        let multirow = multirow_controllers_enabled();
        let batched = if multirow
            && rows.len() > 1
            && !self.draft_window_enabled()
            && weights.stages_prefill_controllers()
        {
            let batch =
                |index: usize| weights.batch_controller(names[index], shape, rows.len(), true);
            [batch(0)?, batch(1)?, batch(2)?]
        } else {
            [None, None, None]
        };
        let staged_controller = batched.iter().any(Option::is_some);
        let resident = if staged_controller {
            None
        } else if let Some(controllers) = weights.attention_controllers(names, shape, rows.len())? {
            Some(controllers)
        } else {
            return Ok(None);
        };

        let _span = self.phase_span(Work::Controller);
        let prepared = {
            let _subspan = self.phase_span(Work::AttentionControlPrepare);
            let rows = rows
                .map(|(stream, bank)| {
                    bank.map(|bank| (stream, bank))
                        .ok_or_else(|| bad("Inkling grouped attention lost a message bank"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.prepare_attention_rows(visit, &rows)?
        };
        let inputs: Vec<_> = prepared
            .iter()
            .map(|row| row.each_ref().map(|item| item.control.gate.as_slice()))
            .collect();
        let projections = {
            let _subspan = self.phase_span(Work::AttentionControlProject);
            match resident {
                Some(controllers) => controllers.project(&inputs, multirow)?,
                None => project_controller_roles(batched, &inputs)?,
            }
        };
        let _subspan = self.phase_span(Work::AttentionControlApply);
        let local = visit.attention == Attention::Local;
        let capture = self.capture_host_controller_inputs(
            layer,
            local,
            &prepared,
            &projections,
            staged_controller,
        )?;
        let mut result = Vec::with_capacity(prepared.len());
        for (row, projections) in prepared.into_iter().zip(projections) {
            let mut inputs = AttentionInputs::default();
            for (index, (prepared, projection)) in row.into_iter().zip(projections).enumerate() {
                let PreparedInput {
                    mut input, control, ..
                } = prepared;
                let ControlInput { gate, message } = control;
                let projection = match projection {
                    Some(projection) => projection,
                    None => self.weights.controller(names[index], &gate, shape)?,
                };
                apply_control(&mut input, message, projection);
                self.norm_attention_input(&mut input, layer, local, index)?;
                inputs[index] = input;
            }
            result.push(inputs);
        }
        if let Some((dir, count, mut words)) = capture {
            for role in 0..3 {
                for row in result.iter().take(count) {
                    words.extend_from_slice(&row[role]);
                }
            }
            let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_ne_bytes()).collect();
            std::fs::write(dir.join("cpu.f32"), bytes)
                .map_err(|e| bad(format!("capture CPU controller: {e}")))?;
        }
        Ok(Some(result))
    }

    fn capture_host_controller_inputs(
        &self,
        layer: &str,
        local: bool,
        prepared: &[[PreparedInput; 3]],
        projections: &[[Option<[Vec<f32>; 2]>; 3]],
        staged: bool,
    ) -> Result<Option<(std::path::PathBuf, usize, Vec<f32>)>, EngError> {
        let Some(dir) = std::env::var_os("FIFI_INKLING_DEBUG_CONTROLLER_CAPTURE") else {
            return Ok(None);
        };
        let dir = std::path::PathBuf::from(dir);
        if !dir.join("meta.json").exists() || dir.join("cpu.f32").exists() {
            return Ok(None);
        }
        let metadata = std::fs::read(dir.join("meta.json"))
            .map_err(|e| bad(format!("read capture metadata: {e}")))?;
        let meta: serde_json::Value = serde_json::from_slice(&metadata)
            .map_err(|e| bad(format!("parse capture metadata: {e}")))?;
        let count = meta["rows"]
            .as_u64()
            .ok_or_else(|| bad("missing capture rows"))? as usize;
        if count == 0
            || count > prepared.len()
            || count > projections.len()
            || meta["hidden"].as_u64() != Some(self.hidden() as u64)
        {
            return Err(bad("invalid capture dimensions"));
        }
        let mut words = Vec::with_capacity(24 * count * self.hidden());
        for role in 0..3 {
            for row in prepared.iter().take(count) {
                words.extend_from_slice(&row[role].control.gate);
            }
        }
        for role in 0..3 {
            let gamma = self
                .weights
                .vec(&attention_norm_name(layer, local, role), self.hidden())?;
            for row in prepared.iter().take(count) {
                words.extend_from_slice(&row[role].input);
                words.extend_from_slice(&row[role].control.message);
                words.extend_from_slice(&gamma);
            }
        }
        for role in 0..3 {
            for row in projections.iter().take(count) {
                let [gain, forget] = row[role]
                    .as_ref()
                    .ok_or_else(|| bad("capture requires completed controller projections"))?;
                words.extend_from_slice(gain);
                words.extend_from_slice(forget);
            }
        }
        eprintln!(
            "Inkling controller input capture {}",
            serde_json::json!({"layer":layer,"rows":count,"staged":staged})
        );
        Ok(Some((dir, count, words)))
    }

    fn prepare_attention_controllers(
        &self,
        visit: LayerVisit,
        stream: &Stream,
        bank: &MessageBank,
    ) -> Result<[PreparedInput; 3], EngError> {
        let [row] = self
            .prepare_attention_rows(visit, &[(stream, bank)])?
            .try_into()
            .map_err(|_| bad("Inkling controller preparation lost its row"))?;
        Ok(row)
    }

    /// The Q/K/V controller inputs of many rows. The query weights are read once; each row's
    /// mixes and norms read only its own stream and bank, so the rows are split over host threads.
    fn prepare_attention_rows(
        &self,
        visit: LayerVisit,
        rows: &[(&Stream, &MessageBank)],
    ) -> Result<Vec<[PreparedInput; 3]>, EngError> {
        // Local roles share the residual before their separate cross-pass gates.
        let local = visit.attention == Attention::Local;
        let hidden = self.hidden();
        // No queries: the first visit's input is the stream's current row.
        let attention = match visit.weight_layer.checked_sub(1) {
            None => Vec::new(),
            Some(query) => (0..if local { 1 } else { 3 })
                .map(|row| {
                    self.weights.row(
                        &format!("model.llm.attn_res.attn_queries.{query}"),
                        row,
                        hidden,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        let [q, k, v] =
            ATTENTION_ROLES.map(|(role, _)| self.control_query(visit.weight_layer, role));
        let queries = [q?, k?, v?];
        let prepare = |&(stream, bank): &(&Stream, &MessageBank)| -> [PreparedInput; 3] {
            let mixed = |index: usize| match attention.get(index) {
                Some(query) => stream.mix(0, query),
                None => stream.current().to_vec(),
            };
            let common = local.then(|| mixed(0));
            std::array::from_fn(|index| {
                let input = common.clone().unwrap_or_else(|| mixed(index));
                let (control, gate_scales) = control_input_from(&input, bank, &queries[index]);
                PreparedInput {
                    input,
                    control,
                    gate_scales,
                }
            })
        };
        Ok(map_rows(&mut rows.to_vec(), PROMPT_ROW_SPLIT, |row| {
            prepare(row)
        }))
    }
}

fn name_refs(names: &[[String; 3]; 3]) -> [[&str; 3]; 3] {
    names
        .each_ref()
        .map(|names| names.each_ref().map(String::as_str))
}

pub(super) fn multirow_controllers_enabled() -> bool {
    Switch::MultirowControllers.on()
}

pub(super) fn controller_batch_enabled() -> bool {
    Switch::BatchControllers.on()
}
