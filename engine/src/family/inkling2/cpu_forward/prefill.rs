//! Layer-major prompt prefill. Each visit advances its own causal state in token
//! order, then computes the routed expert outputs using the shared CUDA banks.

use super::{CpuForward, LayerVisit, MessageBank, MoeRoute, Stream, VisitState, Weights, add, bad};
use crate::{
    EngError,
    family::inkling2::{
        cuda_projection::gguf::streaming::PrefillRoute, gguf_weights::GgufTextWeights,
    },
};

struct PromptRow {
    stream: Stream,
    prelude: Option<Stream>,
    bank: Option<MessageBank>,
}

impl PromptRow {
    fn new(stream: Stream) -> Self {
        Self {
            stream,
            prelude: None,
            bank: None,
        }
    }

    fn begin_pass(&mut self, pass: usize) -> Result<(), EngError> {
        if pass == 0 {
            self.prelude = Some(self.stream.clone());
        } else {
            let prelude = self
                .prelude
                .as_ref()
                .ok_or_else(|| bad("Inkling prefill lost its core prelude"))?;
            self.bank = Some(MessageBank {
                first_block: prelude.blocks.len() + 1,
                stream: std::mem::replace(&mut self.stream, prelude.clone()),
            });
        }
        Ok(())
    }
}

struct MlpRow {
    route: MoeRoute,
    input: Vec<f32>,
}

impl CpuForward {
    pub(in crate::family::inkling2) fn supports_grouped_prefill(&self) -> bool {
        matches!(&self.weights, Weights::Gguf(weights) if weights.supports_grouped_prefill())
    }

    pub(in crate::family::inkling2) fn prefill_workspace_bytes_per_row(
        &self,
    ) -> Result<u64, EngError> {
        let hidden = u128::from(self.spec.hidden_size);
        let middle = u128::from(self.spec.intermediate_size);
        let experts = u128::from(self.spec.n_routed_experts);
        let top_k = u128::from(self.spec.num_experts_per_tok);
        let blocks = self
            .visits
            .iter()
            .filter(|visit| {
                visit
                    .weight_layer
                    .is_multiple_of(self.spec.n_layers_per_attn_res_block)
            })
            .count() as u128;

        // Current stream, prelude, prior pass bank, and a transient stream clone.
        let stream_values = 4 * (blocks + 1) * hidden;
        // Routed contributions, MLP input, accumulated output, and staging
        // copies of the input, output, and gate/up and down intermediates.
        let mut projection_values = (top_k + 6) * hidden + 6 * middle;
        let mut projection_metadata = 0;
        if prefill_projection_batch_enabled() {
            let query = u128::from(
                self.spec
                    .num_attention_heads
                    .max(self.spec.swa_num_attention_heads),
            );
            let kv = u128::from(
                self.spec
                    .num_key_value_heads
                    .max(self.spec.swa_num_key_value_heads),
            );
            let width = u128::from(self.spec.head_dim.max(self.spec.swa_head_dim));
            projection_values += 3 * hidden
                + 2 * (query + kv) * width
                + query * (self.math.d_rel as u128 + 1)
                + 2 * kv * self.math.sconv_kernel_size as u128;
            projection_metadata = (size_of::<[Vec<f32>; 3]>()
                + size_of::<super::super::cpu_attention::AttentionProjection>()
                + size_of::<[&[f32]; 3]>()) as u128;
        }
        // Batched router logits coexist with the selected rows' scores.
        let numeric_bytes =
            (stream_values + projection_values + 2 * experts) * size_of::<f32>() as u128;
        // Include row handles and reserve room for geometrically grown lane lists.
        let row_metadata = (size_of::<PromptRow>()
            + size_of::<MlpRow>()
            + size_of::<PrefillRoute<'_>>()
            + size_of::<Vec<f32>>()) as u128;
        let stream_metadata = 8 * blocks * size_of::<Vec<f32>>() as u128;
        let lanes = size_of::<(usize, usize)>();
        let resident_slots = size_of::<(crate::expert_tier::ExpertId, (usize, usize))>();
        let route_metadata = top_k
            * (2 * (lanes + resident_slots) + size_of::<usize>() + size_of::<Vec<f32>>()) as u128;
        u64::try_from(
            numeric_bytes + row_metadata + stream_metadata + route_metadata + projection_metadata,
        )
        .map_err(|_| EngError::MemoryPlan("Inkling prefill workspace size overflows".into()))
    }

    pub(in crate::family::inkling2) fn prefill_grouped(
        &mut self,
        tokens: &[u32],
    ) -> Result<(), EngError> {
        let Weights::Gguf(weights) = &self.weights else {
            return Err(bad("grouped prefill needs CUDA GGUF weights"));
        };
        if self.failed {
            return Err(bad(
                "a previous Inkling forward invalidated the sequence state",
            ));
        }
        if tokens.len() < 2
            || tokens.len() > self.context.saturating_sub(self.position)
            || tokens
                .iter()
                .any(|&id| u64::from(id) >= self.spec.unpadded_vocab_size)
        {
            return Err(bad(
                "prefill rows or tokens are outside the Inkling text limits",
            ));
        }
        self.failed = true;
        let mut state = std::mem::take(&mut self.state);
        let mut rows = tokens
            .iter()
            .map(|&id| self.embed(id).map(PromptRow::new))
            .collect::<Result<Vec<_>, _>>()?;
        let core = &self.spec.core;
        let width = core.core_end - core.core_start;
        let end = core.core_start + width * core.num_passes;
        for &visit in &self.visits {
            let slot = visit.state_slot;
            if (core.core_start..end).contains(&slot)
                && (slot - core.core_start).is_multiple_of(width)
            {
                for row in &mut rows {
                    row.begin_pass((slot - core.core_start) / width)?;
                }
            } else if slot == end {
                for row in &mut rows {
                    row.bank = None;
                    row.prelude = None;
                }
            }
            self.prompt_visit(visit, &mut state[slot], &mut rows, weights)?;
        }
        self.state = state;
        self.position += tokens.len();
        self.failed = false;
        Ok(())
    }

    fn prompt_visit(
        &self,
        visit: LayerVisit,
        state: &mut VisitState,
        rows: &mut [PromptRow],
        weights: &GgufTextWeights,
    ) -> Result<(), EngError> {
        let physical = visit.weight_layer;
        let layer = format!("model.llm.layers.{physical}");
        let prefix = format!("{layer}.mlp");
        let mut projected = self
            .prompt_attention_inputs(visit, &layer, rows, weights)?
            .into_iter()
            .flatten();
        let inputs = rows.iter_mut().map(|row| {
            self.mlp_input(
                visit,
                &layer,
                state,
                &mut row.stream,
                row.bank.as_ref(),
                projected.next(),
            )
        });
        let router = format!("{prefix}.gate.weight");
        let mlp = if weights.batch_prefill_router(physical as u32, &router) {
            self.prompt_routes(
                weights,
                physical,
                &prefix,
                inputs.collect::<Result<_, _>>()?,
            )?
        } else {
            inputs
                .map(|input| {
                    let input = input?;
                    let route = self.moe_route(physical, &prefix, &input)?;
                    Ok(MlpRow { route, input })
                })
                .collect::<Result<_, EngError>>()?
        };
        let outputs = self.prompt_experts(weights, physical, &prefix, &mlp)?;
        for (row, output) in rows.iter_mut().zip(outputs) {
            self.finish_layer(&layer, state, &mut row.stream, output)?;
        }
        Ok(())
    }

    fn prompt_routes(
        &self,
        weights: &GgufTextWeights,
        layer: usize,
        prefix: &str,
        inputs: Vec<Vec<f32>>,
    ) -> Result<Vec<MlpRow>, EngError> {
        let input_rows: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
        let logits =
            weights.prefill_router_logits(&format!("{prefix}.gate.weight"), &input_rows)?;
        let experts = self.spec.n_routed_experts as usize;
        let bias = self.weights.vec(&format!("{prefix}.gate.bias"), experts)?;
        inputs
            .into_iter()
            .zip(logits.chunks_exact(experts))
            .map(|(input, scores)| {
                let route = self.moe_route_from_logits(layer, scores.to_vec(), &bias)?;
                Ok(MlpRow { route, input })
            })
            .collect()
    }

    fn prompt_attention_inputs(
        &self,
        visit: LayerVisit,
        layer: &str,
        rows: &[PromptRow],
        weights: &GgufTextWeights,
    ) -> Result<Option<Vec<super::super::cpu_attention::AttentionProjection>>, EngError> {
        if !prefill_projection_batch_enabled() {
            return Ok(None);
        }
        let inputs = rows
            .iter()
            .map(|row| self.attention_inputs(visit, layer, &row.stream, row.bank.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        self.prefill_attention_projections(weights, layer, visit.attention, &inputs)
    }

    fn prompt_experts(
        &self,
        weights: &GgufTextWeights,
        layer: usize,
        prefix: &str,
        mlp: &[MlpRow],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let routes: Vec<_> = mlp
            .iter()
            .map(|row| PrefillRoute {
                ids: &row.route.selected,
                input: &row.input,
            })
            .collect();
        let names = [
            format!("{prefix}.experts.w13_weight"),
            format!("{prefix}.experts.w2_weight"),
        ];
        let names = names.each_ref().map(String::as_str);
        if let Some(output) = weights.prefill_device_experts(
            names,
            layer as u32,
            &routes,
            [self.math.mlp_gate_softcap, self.math.mlp_up_softcap],
            |row, id| mlp[row].route.weight(id, self.math.route_scale),
        )? {
            return Ok(output);
        }
        let batch = weights.prefill_experts(names, layer as u32, &routes, |gate, up| {
            self.math.expert_activation(gate, up)
        })?;
        let Some(batch) = batch else {
            return mlp
                .iter()
                .map(|row| self.moe_with_route(layer, prefix, &row.input, &row.route))
                .collect();
        };
        Ok(mlp
            .iter()
            .zip(batch)
            .map(|(input, results)| {
                let mut output = vec![0.; self.hidden()];
                for (&id, contribution) in input.route.selected.iter().zip(results) {
                    add(
                        &mut output,
                        &contribution,
                        input.route.weight(id, self.math.route_scale),
                    );
                }
                output
            })
            .collect())
    }
}

fn prefill_projection_batch_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("FIFI_INKLING_CUDA_BATCH_ATTENTION_PROJECTIONS").as_deref() == Ok("1")
    })
}

#[test]
#[ignore = "requires a leased CUDA device, the Inkling GGUF and serial reference logits"]
fn grouped_prefill_matches_serial_logits_and_restored_state() {
    use std::{env, fs, path::Path};

    let directory = env::var_os("FIFI_INKLING_MODEL_DIR").expect("set the Inkling GGUF directory");
    let reference =
        env::var_os("FIFI_INKLING_REFERENCE_LOGITS").expect("set the serial logits path");
    let tokens: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("set comma-separated token ids")
        .split(',')
        .map(|value| value.parse().unwrap())
        .collect();
    let expected = fs::read(reference).unwrap();
    let mut model = CpuForward::open(Path::new(&directory), tokens.len()).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(model.supports_grouped_prefill());
    let vocab = model.vocab_size() as usize;
    let row_bytes = vocab * size_of::<f32>();
    assert_eq!(expected.len(), tokens.len() * row_bytes);

    for width in [2, 8, 16] {
        model.reset();
        // The same held prefix is used for two independent continuations.
        for &token in &tokens[..3] {
            model.token(token, false).unwrap();
        }
        let prefix = model.snapshot();
        for _ in 0..2 {
            model.restore(&prefix);
            for chunk in tokens[3..].chunks(width + 1) {
                let (&last, head) = chunk.split_last().unwrap();
                if head.len() > 1 {
                    model.prefill_grouped(head).unwrap();
                } else if let Some(&token) = head.first() {
                    model.token(token, false).unwrap();
                }
                let position = model.position();
                let logits = model.token(last, true).unwrap().unwrap();
                let reference = &expected[position * row_bytes..][..row_bytes];
                assert_eq!(logits.len(), vocab);
                assert!(
                    reference
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .zip(logits)
                        .all(|(&bytes, logit)| bytes == logit.to_le_bytes()),
                    "logit mismatch at position {position}, chunk {width}"
                );
            }
        }
    }
}
