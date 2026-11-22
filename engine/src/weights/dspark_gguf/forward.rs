//! Portable reference for target-conditioned dense draft blocks.

use std::sync::Arc;

use half::bf16;

use super::{DraftLayer, DsparkBlock, DsparkConfig, DsparkGguf, DsparkMatrix};
use crate::EngError;

impl DsparkGguf {
    /// Runs the trained block at `anchor_position` from the target's already
    /// normalized anchor embedding and one concatenated tap row per preceding
    /// token. Context positions begin at zero and exclude the anchor.
    pub(crate) fn forward(
        &self,
        anchor_embed: &[f32],
        target_features: &[Arc<[f32]>],
        anchor_position: usize,
    ) -> Result<DsparkBlock, EngError> {
        self.forward_with(
            anchor_embed,
            target_features,
            anchor_position,
            &mut HostMath,
        )
    }

    pub(super) fn forward_with(
        &self,
        anchor_embed: &[f32],
        target_features: &[Arc<[f32]>],
        anchor_position: usize,
        math: &mut impl DraftMath,
    ) -> Result<DsparkBlock, EngError> {
        let config = self.config();
        let Some(last) = target_features.last() else {
            return Err(EngError::BadRequest(
                "DSpark needs features from at least one committed token".into(),
            ));
        };
        if anchor_embed.len() != config.width || target_features.len() != anchor_position {
            return Err(EngError::BadRequest(
                "DSpark anchor embedding or committed context length does not match the draft block"
                    .into(),
            ));
        }
        let feature_width = config.width * config.target_layers.len();
        if target_features.iter().any(|row| row.len() != feature_width) {
            return Err(EngError::BadRequest(format!(
                "DSpark requires {feature_width} target features per token"
            )));
        }
        if anchor_position > config.context_length - config.block_size {
            return Err(EngError::BadRequest(
                "DSpark block exceeds the checkpoint context length".into(),
            ));
        }

        let mut block = vec![self.mask_embedding.clone(); config.block_size];
        block[0].copy_from_slice(anchor_embed);
        block.iter_mut().for_each(|row| round_row(row));
        for (index, layer) in self.layers.iter().enumerate() {
            let context_start = if config.local_layers[index] {
                anchor_position.saturating_sub(config.sliding_window)
            } else {
                0
            };
            let context = DraftContext {
                taps: &target_features[context_start..],
                start: context_start,
                alpha: &self.fusion[index],
            };
            layer.apply(&mut block, &context, config, &mut LayerMath { index, math })?;
        }
        let hidden = block
            .into_iter()
            .skip(1)
            .map(|row| rms(&row, &self.output_norm, config.epsilon))
            .collect();
        let recurrent_context = mix(
            last,
            &self.fusion[self.layers.len()],
            &self.recurrent_norm,
            config.epsilon,
        );
        Ok(DsparkBlock {
            hidden,
            recurrent_context,
        })
    }
}

struct DraftContext<'a> {
    taps: &'a [Arc<[f32]>],
    start: usize,
    alpha: &'a [f32],
}

#[derive(Clone, Copy)]
pub(super) enum Matrix {
    Query,
    Key,
    Value,
    AttentionOutput,
    ContextKey,
    ContextValue,
    FfnGate,
    FfnUp,
    FfnDown,
}

impl Matrix {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) const ALL: [Self; 9] = [
        Self::Query,
        Self::Key,
        Self::Value,
        Self::AttentionOutput,
        Self::ContextKey,
        Self::ContextValue,
        Self::FfnGate,
        Self::FfnUp,
        Self::FfnDown,
    ];

    pub(super) fn weight(self, layer: &DraftLayer) -> &DsparkMatrix {
        match self {
            Self::Query => &layer.query,
            Self::Key => &layer.key,
            Self::Value => &layer.value,
            Self::AttentionOutput => &layer.attention_output,
            Self::ContextKey => &layer.context_key,
            Self::ContextValue => &layer.context_value,
            Self::FfnGate => &layer.ffn_gate,
            Self::FfnUp => &layer.ffn_up,
            Self::FfnDown => &layer.ffn_down,
        }
    }
}

pub(super) struct KvRows {
    pub(super) keys: Vec<Vec<f32>>,
    pub(super) values: Vec<Vec<f32>>,
}

pub(super) struct DraftAttention<'a> {
    pub(super) query: &'a [Vec<f32>],
    pub(super) draft: KvRows,
    pub(super) committed: KvRows,
    pub(super) start: usize,
    pub(super) new_start: usize,
    pub(super) end: usize,
}

pub(super) trait DraftMath {
    fn project(
        &mut self,
        layer: usize,
        matrix: Matrix,
        weights: &DsparkMatrix,
        rows: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, EngError>;

    fn cached_until(&self, layer: usize, start: usize, end: usize) -> usize;

    fn attention(
        &mut self,
        layer: usize,
        input: DraftAttention<'_>,
        config: &DsparkConfig,
    ) -> Result<Vec<Vec<f32>>, EngError>;
}

struct HostMath;

impl DraftMath for HostMath {
    fn project(
        &mut self,
        _: usize,
        _: Matrix,
        weights: &DsparkMatrix,
        rows: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        Ok(weights.project_batch(rows))
    }

    fn cached_until(&self, _: usize, start: usize, _: usize) -> usize {
        start
    }

    fn attention(
        &mut self,
        _: usize,
        input: DraftAttention<'_>,
        config: &DsparkConfig,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let DraftAttention {
            query,
            draft,
            mut committed,
            start,
            new_start,
            end,
        } = input;
        debug_assert_eq!(start, new_start);
        debug_assert_eq!(committed.keys.len(), end - start);
        committed.keys.extend(draft.keys);
        committed.values.extend(draft.values);
        Ok(query
            .iter()
            .map(|query| attention_row(query, &committed.keys, &committed.values, config))
            .collect())
    }
}

struct LayerMath<'a, M> {
    index: usize,
    math: &'a mut M,
}

impl<M: DraftMath> LayerMath<'_, M> {
    fn project(
        &mut self,
        layer: &DraftLayer,
        matrix: Matrix,
        rows: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, EngError> {
        self.math
            .project(self.index, matrix, matrix.weight(layer), rows)
    }
}

impl DraftLayer {
    fn apply(
        &self,
        block: &mut [Vec<f32>],
        context: &DraftContext<'_>,
        config: &DsparkConfig,
        math: &mut LayerMath<'_, impl DraftMath>,
    ) -> Result<(), EngError> {
        let input = normalize_block(block, &self.attention_norm, config.epsilon);
        // Normalize the unrounded sum from the previous sublayer, then keep
        // the BF16 residual that the next addition will read.
        block.iter_mut().for_each(|row| round_row(row));
        let attended = self.attend(&input, context, config, math)?;
        let mut ffn_input = Vec::with_capacity(block.len());
        for (row, output) in block.iter_mut().zip(attended) {
            add_residual(row, &output);
            ffn_input.push(rms(row, &self.ffn_norm, config.epsilon));
            round_row(row);
        }
        let mut gate = math.project(self, Matrix::FfnGate, &ffn_input)?;
        let up = math.project(self, Matrix::FfnUp, &ffn_input)?;
        for (row, up) in gate.iter_mut().zip(up) {
            for (gate, up) in row.iter_mut().zip(up) {
                let x = round(*gate);
                *gate = round((x / (1.0 + (-x).exp())) * round(up));
            }
        }
        let down = math.project(self, Matrix::FfnDown, &gate)?;
        for (row, mut output) in block.iter_mut().zip(down) {
            round_row(&mut output);
            add_residual(row, &output);
        }
        Ok(())
    }

    fn attend(
        &self,
        input: &[Vec<f32>],
        context: &DraftContext<'_>,
        config: &DsparkConfig,
        math: &mut LayerMath<'_, impl DraftMath>,
    ) -> Result<Vec<Vec<f32>>, EngError> {
        let end = context.start + context.taps.len();
        let mut query = math.project(self, Matrix::Query, input)?;
        let mut keys = math.project(self, Matrix::Key, input)?;
        let mut values = math.project(self, Matrix::Value, input)?;
        norm_rotate_rows(&mut query, end, &self.query_norm, config);
        norm_rotate_rows(&mut keys, end, &self.key_norm, config);
        values.iter_mut().for_each(|row| round_row(row));
        let draft = KvRows { keys, values };

        let new_start = math.math.cached_until(math.index, context.start, end);
        let committed = self.project_context(context, new_start, config, math)?;
        let heads = math.math.attention(
            math.index,
            DraftAttention {
                query: &query,
                draft,
                committed,
                start: context.start,
                new_start,
                end,
            },
            config,
        )?;
        let mut output = math.project(self, Matrix::AttentionOutput, &heads)?;
        output.iter_mut().for_each(|row| round_row(row));
        Ok(output)
    }

    fn project_context(
        &self,
        context: &DraftContext<'_>,
        start: usize,
        config: &DsparkConfig,
        math: &mut LayerMath<'_, impl DraftMath>,
    ) -> Result<KvRows, EngError> {
        let input: Vec<_> = context.taps[start - context.start..]
            .iter()
            .map(|row| mix(row, context.alpha, &self.context_norm, config.epsilon))
            .collect();
        let mut keys = math.project(self, Matrix::ContextKey, &input)?;
        let mut values = math.project(self, Matrix::ContextValue, &input)?;
        norm_rotate_rows(&mut keys, start, &self.context_key_norm, config);
        values.iter_mut().for_each(|row| round_row(row));
        Ok(KvRows { keys, values })
    }
}

fn attention_row(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    config: &DsparkConfig,
) -> Vec<f32> {
    let width = config.head_width;
    let group = config.query_heads / config.kv_heads;
    let scale = (width as f32).sqrt().recip();
    let mut result = Vec::with_capacity(query.len());
    for (index, head) in query.chunks_exact(width).enumerate() {
        let start = index / group * width;
        let range = start..start + width;
        let scores = keys
            .iter()
            .map(|key| dot(head, &key[range.clone()]) * scale)
            .collect::<Vec<_>>();
        let mut output = vec![0.0; width];
        for (weight, value) in softmax(&scores).into_iter().zip(values) {
            for (out, &component) in output.iter_mut().zip(&value[range.clone()]) {
                *out += weight * component;
            }
        }
        round_row(&mut output);
        result.extend(output);
    }
    result
}

fn norm_rotate_rows(rows: &mut [Vec<f32>], start: usize, norm: &[f32], config: &DsparkConfig) {
    let width = config.head_width;
    let half = width / 2;
    for (offset, row) in rows.iter_mut().enumerate() {
        let position = (start + offset) as f32;
        let angles: Vec<_> = (0..half)
            .map(|i| {
                let exponent = (2 * i) as f32 / width as f32;
                (position / config.rope_theta.powf(exponent)).sin_cos()
            })
            .collect();
        round_row(row);
        for head in row.chunks_exact_mut(width) {
            let normalized = rms(head, norm, config.epsilon);
            let (left, right) = head.split_at_mut(half);
            for (i, &(sin, cos)) in angles.iter().enumerate() {
                let a = normalized[i];
                let b = normalized[i + half];
                left[i] = round(a * cos - b * sin);
                right[i] = round(b * cos + a * sin);
            }
        }
    }
}

fn normalize_block(block: &[Vec<f32>], weights: &[f32], epsilon: f32) -> Vec<Vec<f32>> {
    block.iter().map(|row| rms(row, weights, epsilon)).collect()
}

fn add_residual(row: &mut [f32], update: &[f32]) {
    for (base, &delta) in row.iter_mut().zip(update) {
        *base += delta;
    }
}

fn mix(features: &[f32], alpha: &[f32], norm: &[f32], epsilon: f32) -> Vec<f32> {
    let mut mixed = vec![0.0; norm.len()];
    for (tap, &coefficient) in features.chunks_exact(norm.len()).zip(alpha) {
        for (out, &value) in mixed.iter_mut().zip(tap) {
            *out += round(value) * coefficient;
        }
    }
    rms(&mixed, norm, epsilon)
}

fn rms(input: &[f32], weights: &[f32], epsilon: f32) -> Vec<f32> {
    let scale = (dot(input, input) / input.len() as f32 + epsilon)
        .sqrt()
        .recip();
    input
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| round(value * scale * weight))
        .collect()
}

pub(super) fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut output: Vec<_> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum = output.iter().sum::<f32>();
    output.iter_mut().for_each(|x| *x /= sum);
    output
}

pub(super) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn round_row(row: &mut [f32]) {
    row.iter_mut().for_each(|value| *value = round(*value));
}
