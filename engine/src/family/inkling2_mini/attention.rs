//! Single-token grouped-query attention for Inkling 2 Mini.

use std::collections::VecDeque;

use half::bf16;

use super::{
    CONV_TAPS, HEAD_DIM, HIDDEN, KV_HEADS, MAX_CONTEXT, QUERY_HEADS, VISITS, WINDOW,
    dense::DenseWeights,
    is_local,
    sconv::{InklingConvState, InklingConvStream, InklingConvTaps},
    visits,
};
use crate::EngError;

const HEAD: usize = HEAD_DIM as usize;
const Q_WIDTH: usize = QUERY_HEADS * HEAD;
const KV_WIDTH: usize = KV_HEADS as usize * HEAD;
const REL_DIM: usize = 16;
const REL_EXTENT: usize = WINDOW as usize;
const TAPS: usize = CONV_TAPS as usize;
const EPS: f32 = 1e-6;

struct KvRow {
    key: [bf16; KV_WIDTH],
    value: [bf16; KV_WIDTH],
}

#[derive(Default)]
struct VisitCache {
    position: usize,
    rows: VecDeque<Box<KvRow>>,
}

impl VisitCache {
    fn previous(&self, local: bool) -> impl Iterator<Item = &KvRow> {
        self.rows
            .iter()
            .skip(usize::from(local && self.rows.len() == REL_EXTENT))
            .map(Box::as_ref)
    }

    fn append(&mut self, local: bool, row: KvRow) {
        if local && self.rows.len() == REL_EXTENT {
            if let Some(mut oldest) = self.rows.pop_front() {
                *oldest = row;
                self.rows.push_back(oldest);
            }
        } else {
            self.rows.push_back(Box::new(row));
        }
        self.position += 1;
    }
}

/// Visit-private BF16 K/V history for one sequence.
pub struct InklingAttentionState {
    cache: [VisitCache; VISITS],
}

impl Default for InklingAttentionState {
    fn default() -> Self {
        Self {
            cache: std::array::from_fn(|_| VisitCache::default()),
        }
    }
}

impl InklingAttentionState {
    /// Project query, key and value input rows, attend to this visit's cached tokens,
    /// and return the BF16 output projection before the attention-output convolution.
    /// The caller supplies the full-attention log-position scale as `tau`.
    pub fn step(
        &mut self,
        weights: &DenseWeights,
        conv: &mut InklingConvState,
        visit: usize,
        (query, key, value): (&[bf16], &[bf16], &[bf16]),
        tau: f32,
    ) -> Result<Vec<bf16>, EngError> {
        let schedule = visits();
        let layer = schedule
            .get(visit)
            .ok_or_else(|| {
                EngError::BadRequest(format!("Inkling attention visit {visit} is out of range"))
            })?
            .weight_layer;
        let local = is_local(layer);
        let cache = &mut self.cache[visit];
        if cache.position >= MAX_CONTEXT as usize
            || [query, key, value].iter().any(|x| x.len() != HIDDEN)
            || (!local && (!tau.is_finite() || tau <= 0.0))
        {
            return Err(EngError::BadRequest(format!(
                "Inkling attention has invalid input at visit {visit}"
            )));
        }
        let tau = if local { 1.0 } else { tau };
        let prefix = format!("model.llm.layers.{layer}.attn.");
        let project = |name: &str, input: &[f32], rows: usize| {
            let name = format!("{prefix}{name}");
            check_shape(weights, &name, &[rows, input.len()])?;
            weights.matvec(&name, input).map(|row| narrow(&row))
        };
        let weight = |name: &str, shape: &[usize]| {
            let name = format!("{prefix}{name}");
            check_shape(weights, &name, shape)?;
            weights.bf16(&name)
        };

        let q_input = widen(query);
        let k_input = widen(key);
        let v_input = widen(value);
        let mut q = project("wq_du.weight", &q_input, Q_WIDTH)?;
        let mut rel = project("wr_du.weight", &q_input, QUERY_HEADS * REL_DIM)?;
        let gate = project("gate_attn.weight", &q_input, Q_WIDTH)?;
        let sink = project("attn_sink_proj.weight", &q_input, QUERY_HEADS)?;
        let k = project("wk_dv.weight", &k_input, KV_WIDTH)?;
        let v = project("wv_dv.weight", &v_input, KV_WIDTH)?;
        let q_scale = weight("q_l2_clip.scale", &[QUERY_HEADS, HEAD])?;
        let rel_scale = weight("rel_l2_clip.scale", &[QUERY_HEADS, REL_DIM])?;
        let rel_projection = weight("rel_logits_proj.proj", &[REL_DIM, REL_EXTENT])?;

        let mut row = KvRow {
            key: [bf16::ZERO; KV_WIDTH],
            value: [bf16::ZERO; KV_WIDTH],
        };
        let tap_weights = |name: &str| {
            if local {
                weight(&format!("{name}.weight"), &[KV_WIDTH, 1, TAPS])
            } else {
                weight(&format!("{name}.static_w"), &[TAPS, KV_WIDTH])
            }
        };
        let k_weights = tap_weights("k_sconv")?;
        let v_weights = tap_weights("v_sconv")?;
        let k_deltas;
        let v_deltas;
        let (key_taps, value_taps) = if local {
            (
                InklingConvTaps::Static(&k_weights),
                InklingConvTaps::Static(&v_weights),
            )
        } else {
            k_deltas = project("k_sconv.weight_proj", &k_input, KV_HEADS as usize * TAPS)?;
            v_deltas = project("v_sconv.weight_proj", &v_input, KV_HEADS as usize * TAPS)?;
            (
                InklingConvTaps::Dynamic {
                    weights: &k_weights,
                    deltas: &k_deltas,
                },
                InklingConvTaps::Dynamic {
                    weights: &v_weights,
                    deltas: &v_deltas,
                },
            )
        };
        check_shape(
            weights,
            &format!("{prefix}wo_ud.weight"),
            &[HIDDEN, Q_WIDTH],
        )?;
        conv.step(visit, InklingConvStream::Key, &k, key_taps, &mut row.key)?;
        conv.step(
            visit,
            InklingConvStream::Value,
            &v,
            value_taps,
            &mut row.value,
        )?;

        let threshold = 2.0 * (HEAD as f32).sqrt();
        clip_heads(&mut q, Some(&q_scale), HEAD, threshold, tau);
        clip_heads(&mut rel, Some(&rel_scale), REL_DIM, 8.0, tau);
        clip_heads(&mut row.key, None, HEAD, threshold, 1.0);
        for head in row.value.as_chunks_mut::<HEAD>().0 {
            let scale = (sum_squares(head.iter().map(|x| x.to_f32())) / HEAD as f32 + EPS)
                .sqrt()
                .recip();
            for x in head {
                *x = bf16::from_f32(x.to_f32() * scale);
            }
        }

        let attended = attend(
            cache.previous(local).chain(std::iter::once(&row)),
            &q,
            &rel,
            &rel_projection,
            &gate,
            &sink,
        );
        let output = project("wo_ud.weight", &attended, HIDDEN)?;
        cache.append(local, row);
        Ok(output)
    }
}

fn check_shape(weights: &DenseWeights, name: &str, expected: &[usize]) -> Result<(), EngError> {
    let shape = weights.shape(name)?;
    if shape != expected {
        return Err(EngError::Checkpoint(format!(
            "Inkling attention tensor {name} has shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn sum_squares(values: impl Iterator<Item = f32>) -> f32 {
    values.map(|x| x * x).sum()
}

fn narrow(input: &[f32]) -> Vec<bf16> {
    input.iter().copied().map(bf16::from_f32).collect()
}

fn widen(input: &[bf16]) -> Vec<f32> {
    input.iter().map(|value| value.to_f32()).collect()
}

fn clip_heads(rows: &mut [bf16], gamma: Option<&[bf16]>, width: usize, threshold: f32, tau: f32) {
    for (head, row) in rows.chunks_exact_mut(width).enumerate() {
        let mut scaled = [0.0; HEAD];
        for (channel, (dst, src)) in scaled.iter_mut().zip(row.iter()).enumerate() {
            let gamma = gamma.map_or(1.0, |g| g[head * width + channel].to_f32());
            *dst = src.to_f32() * gamma;
        }
        let norm = sum_squares(scaled[..width].iter().copied()).sqrt();
        let shrink = (threshold / (norm + EPS)).min(1.0);
        for (dst, value) in row.iter_mut().zip(scaled) {
            *dst = bf16::from_f32(bf16::from_f32(value * shrink).to_f32() * tau);
        }
    }
}

// Natural-log attention scores use QK divided by sqrt(head size), plus relative bias.
// The denominator uses the FP32 probabilities. The value dot product uses BF16
// probabilities, followed by the elementwise gate and the per-head sink correction.
// https://docs.vllm.ai/en/latest/api/vllm/models/inkling2/nvidia/ops/attention/
fn attend<'a>(
    rows: impl Iterator<Item = &'a KvRow>,
    query: &[bf16],
    relative: &[bf16],
    rel_projection: &[bf16],
    gate: &[bf16],
    sink: &[bf16],
) -> Vec<f32> {
    let rows: Vec<_> = rows.collect();
    let mut logits = vec![0.; rows.len()];
    let mut output = vec![0.; Q_WIDTH];
    let scale = (HEAD as f32).sqrt().recip();
    for (head, (out, q)) in output
        .as_chunks_mut::<HEAD>()
        .0
        .iter_mut()
        .zip(query.as_chunks::<HEAD>().0)
        .enumerate()
    {
        let kv_head = head / (QUERY_HEADS / KV_HEADS as usize);
        let kv_channels = kv_head * HEAD..(kv_head + 1) * HEAD;
        let rel = &relative[head * REL_DIM..(head + 1) * REL_DIM];
        let sink = sink[head].to_f32();
        let mut max = f32::NEG_INFINITY;
        for (position, (logit, row)) in logits.iter_mut().zip(&rows).enumerate() {
            let distance = rows.len() - 1 - position;
            let bias = if distance < REL_EXTENT {
                let raw = rel
                    .iter()
                    .enumerate()
                    .map(|(d, r)| r.to_f32() * rel_projection[d * REL_EXTENT + distance].to_f32())
                    .sum();
                bf16::from_f32(raw).to_f32()
            } else {
                0.0
            };
            let dot: f32 = q
                .iter()
                .zip(&row.key[kv_channels.clone()])
                .map(|(a, b)| a.to_f32() * b.to_f32())
                .sum();
            *logit = dot * scale + bias;
            max = max.max(*logit);
        }
        let mut denominator = 0.0;
        for (logit, row) in logits.iter().zip(&rows) {
            let p = (logit - max).exp();
            denominator += p;
            let p = bf16::from_f32(p).to_f32();
            for (dst, src) in out.iter_mut().zip(&row.value[kv_channels.clone()]) {
                *dst += p * src.to_f32();
            }
        }
        let lse = max + denominator.ln();
        let sink_scale = (1.0 + (sink - lse).exp()).recip();
        for (i, value) in out.iter_mut().enumerate() {
            let gate = gate[head * HEAD + i].to_f32();
            let sigmoid = (1.0 + (-gate).exp()).recip();
            *value = bf16::from_f32(*value / denominator * sigmoid * sink_scale).to_f32();
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_combines_relative_bias_grouped_heads_and_sink() {
        let make_row = |initial: f32, group: f32| {
            let mut row = KvRow {
                key: [bf16::ZERO; KV_WIDTH],
                value: [bf16::from_f32(initial); KV_WIDTH],
            };
            row.value[HEAD..2 * HEAD].fill(bf16::from_f32(group));
            row
        };
        let old = make_row(1.0, 4.0);
        let current = make_row(6.0, 8.0);
        let q = [bf16::ZERO; Q_WIDTH];
        let mut relative = [bf16::ZERO; QUERY_HEADS * REL_DIM];
        relative[0] = bf16::ONE;
        let mut projection = [bf16::ZERO; REL_DIM * REL_EXTENT];
        projection[1] = bf16::ONE;
        let gate = [bf16::ZERO; Q_WIDTH];
        let sink = [bf16::ZERO; QUERY_HEADS];
        let output = attend(
            [&old, &current].into_iter(),
            &q,
            &relative,
            &projection,
            &gate,
            &sink,
        );
        let p = (-1.0f32).exp();
        let numerator = 1.0 + 6.0 * bf16::from_f32(p).to_f32();
        let expected = numerator / (1.0 + 2.0 * p) * 0.5;
        assert_eq!(output[0], bf16::from_f32(expected).to_f32());
        assert_eq!(output[HEAD], bf16::from_f32(7.0 / 6.0).to_f32());
        assert_eq!(output[3 * HEAD], 2.0);
    }

    #[test]
    fn full_attention_keeps_old_rows_with_zero_bias_outside_the_relative_extent() {
        let mut old = KvRow {
            key: [bf16::ZERO; KV_WIDTH],
            value: [bf16::ZERO; KV_WIDTH],
        };
        old.value[0] = bf16::from_f32(1024.0);
        let recent = KvRow {
            key: [bf16::ZERO; KV_WIDTH],
            value: [bf16::ZERO; KV_WIDTH],
        };
        let q = [bf16::ZERO; Q_WIDTH];
        let relative = [bf16::ONE; QUERY_HEADS * REL_DIM];
        let mut projection = [bf16::ZERO; REL_DIM * REL_EXTENT];
        projection[REL_EXTENT - 1] = bf16::ONE;
        let gate = [bf16::ZERO; Q_WIDTH];
        let sink = [bf16::ZERO; QUERY_HEADS];
        let output = attend(
            std::iter::once(&old).chain(std::iter::repeat_n(&recent, REL_EXTENT)),
            &q,
            &relative,
            &projection,
            &gate,
            &sink,
        );
        let p = (-1.0f32).exp();
        let denominator = 1.0 + (REL_EXTENT + 1) as f32 * p;
        let expected = 512.0 * bf16::from_f32(p).to_f32() / denominator;
        assert_eq!(output[0], bf16::from_f32(expected).to_f32());
    }

    #[test]
    fn log_position_scale_follows_bf16_clip_rounding() {
        let mut input = [bf16::from_f32(1.5)];
        let gamma = [bf16::from_f32(1.0390625)];
        clip_heads(&mut input, Some(&gamma), 1, 8.0, 1.1);
        assert_eq!(input[0].to_f32(), 1.71875);
    }

    #[test]
    fn local_window_excludes_oldest_row_without_truncating_full_attention() {
        let mut cache = VisitCache::default();
        for position in 0..=REL_EXTENT {
            cache.append(
                true,
                KvRow {
                    key: [bf16::ZERO; KV_WIDTH],
                    value: [bf16::from_f32(position as f32); KV_WIDTH],
                },
            );
        }
        assert_eq!(cache.position, REL_EXTENT + 1);
        assert_eq!(cache.previous(true).count(), REL_EXTENT - 1);
        assert_eq!(cache.previous(false).count(), REL_EXTENT);
        assert_eq!(cache.previous(false).next().unwrap().value[0], bf16::ONE);
        assert_eq!(
            cache.previous(true).next().unwrap().value[0],
            bf16::from_f32(2.0)
        );
    }
}
