//! Batch independent Q/K/V cross-pass controllers before host attention.

use std::sync::{Once, OnceLock};

use super::{
    ATTENTION_ROLES, Attention, ControlInput, CpuForward, LayerVisit, MessageBank, Stream,
    VisitState, Weights, Work, apply_control, attention_norm_name, bad, control_names,
};
use crate::EngError;
use crate::family::inkling2::{
    cpu_attention::AttentionProjection,
    cuda_projection::gguf::{
        AttentionControllers, ControlledAttentionInput, LocalDecode, project_controller_roles,
    },
};

type AttentionInputs = [Vec<f32>; 3];

struct PreparedInput {
    input: Vec<f32>,
    control: ControlInput,
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
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !controller_batch_enabled()
            || !*ENABLED.get_or_init(|| {
                std::env::var("FIFI_INKLING_CUDA_CHAIN_ATTENTION_INPUT").as_deref() == Ok("1")
            })
        {
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
        let resident = if batched.iter().any(Option::is_some) {
            None
        } else if let Some(controllers) = weights.attention_controllers(names, shape, rows.len())? {
            Some(controllers)
        } else {
            return Ok(None);
        };

        let _span = self.phase_span(Work::Controller);
        let prepared = {
            let _subspan = self.phase_span(Work::AttentionControlPrepare);
            rows.map(|(stream, bank)| {
                let bank =
                    bank.ok_or_else(|| bad("Inkling grouped attention lost a message bank"))?;
                self.prepare_attention_controllers(visit, stream, bank)
            })
            .collect::<Result<Vec<_>, _>>()?
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
        let mut result = Vec::with_capacity(prepared.len());
        for (row, projections) in prepared.into_iter().zip(projections) {
            let mut inputs = AttentionInputs::default();
            for (index, (prepared, projection)) in row.into_iter().zip(projections).enumerate() {
                let PreparedInput { mut input, control } = prepared;
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
        Ok(Some(result))
    }

    fn prepare_attention_controllers(
        &self,
        visit: LayerVisit,
        stream: &Stream,
        bank: &MessageBank,
    ) -> Result<[PreparedInput; 3], EngError> {
        // Local roles share the residual before their separate cross-pass gates.
        let common = (visit.attention == Attention::Local)
            .then(|| self.attention_input(visit, stream, 0))
            .transpose()?;
        let [q, k, v] = std::array::from_fn(|index| {
            let input = match &common {
                Some(input) => input.clone(),
                None => self.attention_input(visit, stream, index)?,
            };
            let control =
                self.control_input(&input, bank, visit.weight_layer, ATTENTION_ROLES[index].0)?;
            Ok::<_, EngError>(PreparedInput { input, control })
        });
        Ok([q?, k?, v?])
    }
}

fn name_refs(names: &[[String; 3]; 3]) -> [[&str; 3]; 3] {
    names
        .each_ref()
        .map(|names| names.each_ref().map(String::as_str))
}

pub(super) fn multirow_controllers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_MULTIROW_CONTROLLERS").as_deref() == Ok("1")
    })
}

pub(super) fn controller_batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("FIFI_INKLING_CUDA_BATCH_CONTROLLERS").as_deref() == Ok("1"))
}
