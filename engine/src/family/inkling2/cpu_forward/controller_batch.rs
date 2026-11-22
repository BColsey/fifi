//! Batch independent Q/K/V cross-pass controllers before host attention.

use super::{
    ATTENTION_ROLES, Attention, ControlInput, CpuForward, LayerVisit, MessageBank, Stream, Weights,
    Work, apply_control, bad, control_names,
};
use crate::EngError;

type AttentionInputs = [Vec<f32>; 3];

struct PreparedInput {
    input: Vec<f32>,
    control: ControlInput,
}

impl CpuForward {
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
        let names = names
            .each_ref()
            .map(|names| names.each_ref().map(String::as_str));
        let Some(controllers) = weights.attention_controllers(names, shape, rows.len())? else {
            return Ok(None);
        };

        let _span = self.phase_span(Work::Controller);
        let prepared = rows
            .map(|(stream, bank)| {
                let bank =
                    bank.ok_or_else(|| bad("Inkling grouped attention lost a message bank"))?;
                self.prepare_attention_controllers(visit, stream, bank)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let inputs: Vec<_> = prepared
            .iter()
            .map(|row| row.each_ref().map(|item| item.control.gate.as_slice()))
            .collect();
        let projections = controllers.project(&inputs)?;
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

pub(super) fn controller_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("FIFI_INKLING_CUDA_BATCH_CONTROLLERS").as_deref() == Ok("1"))
}
