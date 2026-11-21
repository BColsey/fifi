//! Single-sequence native text decoding for the released fixed-loop checkpoint.
use super::{
    CORE_START, EXPERTS, HIDDEN, InklingExpertSource, InklingRoutes, LayerVisit, MAX_CONTEXT,
    attention::InklingAttentionState,
    dense::DenseWeights,
    is_local,
    mlx::{MlxExpertCache, MlxExpertCacheStats},
    sconv::{InklingConvState, InklingConvStream, InklingConvTaps},
    visits,
};
use crate::{
    EngError,
    kernels::MlxKernels,
    tensor::{Dtype, Tensor},
};
use half::bf16;
use std::{ffi::OsStr, path::Path, sync::Arc};

const VOCAB: usize = 200_058;
const MAX_PREFILL_ROWS: usize = 16;

fn requested_prefill_rows(setting: Option<&OsStr>, context: u32) -> usize {
    setting
        .and_then(OsStr::to_str)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, MAX_PREFILL_ROWS.min(context as usize))
}
type Row = Vec<f32>;

fn invalid() -> EngError {
    EngError::Checkpoint("Inkling text tensor geometry differs from the released checkpoint".into())
}

fn inv_rms(row: &[f32]) -> f32 {
    (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32 + 1e-6)
        .sqrt()
        .recip()
}

fn rounded(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

fn normalize(row: &[f32], weight: &[f32]) -> Result<Row, EngError> {
    if row.len() != HIDDEN || weight.len() != HIDDEN {
        return Err(invalid());
    }
    let scale = inv_rms(row);
    Ok(row
        .iter()
        .zip(weight)
        .map(|(v, w)| rounded(v * scale * w))
        .collect())
}

/// AttnRes scores and mixes normalized block values, separately for each learned query.
fn mix(blocks: &[Row], query: &[f32]) -> Result<Row, EngError> {
    if blocks.is_empty() || query.len() != HIDDEN || blocks.iter().any(|b| b.len() != HIDDEN) {
        return Err(invalid());
    }
    let scales: Vec<_> = blocks.iter().map(|b| inv_rms(b)).collect();
    let scores: Vec<_> = blocks
        .iter()
        .zip(&scales)
        .map(|(b, r)| {
            b.iter().zip(query).map(|(x, q)| x * q).sum::<f32>() * r / (HIDDEN as f32).sqrt()
        })
        .collect();
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = weights.iter().sum();
    let mut out = vec![0.; HIDDEN];
    for ((block, scale), weight) in blocks.iter().zip(scales).zip(weights) {
        let coefficient = weight / sum * scale;
        for (dst, x) in out.iter_mut().zip(block) {
            *dst = coefficient.mul_add(*x, *dst);
        }
    }
    Ok(out)
}

fn blend(
    dense: &mut DenseWeights,
    prefix: &str,
    dest: &[f32],
    msg: &[f32],
) -> Result<Row, EngError> {
    let mut pair = Vec::with_capacity(2 * HIDDEN);
    for row in [dest, msg] {
        let inv = inv_rms(row);
        pair.extend(row.iter().map(|v| rounded(v * inv)));
    }
    let mut hidden = dense.matvec(&format!("{prefix}.down_weight"), &pair)?;
    hidden.iter_mut().for_each(|x| *x = rounded(*x));
    let gain = dense.matvec(&format!("{prefix}.gain_weight"), &hidden)?;
    let forget = dense.matvec(&format!("{prefix}.forget_weight"), &hidden)?;
    if gain.len() != HIDDEN || forget.len() != HIDDEN {
        return Err(invalid());
    }
    Ok((0..HIDDEN)
        .map(|i| (1. - forget[i].tanh()) * dest[i] + gain[i].tanh() * msg[i])
        .collect())
}

fn conv(
    dense: &mut DenseWeights,
    state: &mut InklingConvState,
    prefix: &str,
    visit: usize,
    stream: InklingConvStream,
    row: &[bf16],
) -> Result<Row, EngError> {
    let taps = dense.bf16(&format!("{prefix}.weight"))?;
    let mut out = vec![bf16::ZERO; HIDDEN];
    state.step(visit, stream, row, InklingConvTaps::Static(&taps), &mut out)?;
    Ok(out.into_iter().map(bf16::to_f32).collect())
}

struct PrefillRow {
    blocks: Vec<Row>,
    prelude: Vec<Row>,
    bank: Vec<Row>,
}

pub struct Decoder {
    dense: DenseWeights,
    metal: MlxKernels,
    experts: MlxExpertCache,
    conv: InklingConvState,
    attention: InklingAttentionState,
    position: u32,
    context: u32,
    failed: bool,
    prefill_rows: usize,
}

impl Decoder {
    pub fn open(dir: &Path, context: u32) -> Result<Self, EngError> {
        let requested = std::env::var_os("FIFI_INKLING_PREFILL_ROWS");
        Self::open_with_prefill_rows(dir, context, requested.as_deref())
    }

    fn open_with_prefill_rows(
        dir: &Path,
        context: u32,
        requested: Option<&OsStr>,
    ) -> Result<Self, EngError> {
        if !(1..=MAX_CONTEXT).contains(&context) {
            return Err(EngError::BadRequest("Invalid Inkling context limit".into()));
        }
        let bytes = super::cache_budget(dir, context)?;
        let source = Arc::new(InklingExpertSource::open(dir)?);
        let metal = MlxKernels::try_new()?;
        let mut experts = MlxExpertCache::new(&metal, source, bytes);
        let requested = requested_prefill_rows(requested, context);
        let prefill_rows = experts.configure_prefill(requested)?;
        if requested > 1 {
            eprintln!(
                "Inkling 2 Mini: prefill_rows={prefill_rows} requested_prefill_rows={requested}"
            );
        }
        Ok(Self {
            dense: DenseWeights::open(dir)?,
            metal,
            experts,
            conv: Default::default(),
            attention: Default::default(),
            position: 0,
            context,
            failed: false,
            prefill_rows,
        })
    }
    pub fn expert_cache_stats(&self) -> MlxExpertCacheStats {
        self.experts.stats()
    }

    pub fn reset(&mut self) {
        self.conv = Default::default();
        self.attention = Default::default();
        self.position = 0;
        self.failed = false;
    }
    pub fn advance(&mut self, token: u32) -> Result<(), EngError> {
        self.consume(token, false).map(|_| ())
    }
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, EngError> {
        self.consume(token, true)
    }
    pub(super) fn prefill_rows(&self) -> usize {
        self.prefill_rows
    }

    pub fn position(&self) -> usize {
        self.position as usize
    }

    pub fn advance_rows(&mut self, tokens: &[u32]) -> Result<(), EngError> {
        self.check_tokens(tokens)?;
        if self.prefill_rows == 1 {
            return tokens.iter().try_for_each(|&token| self.advance(token));
        }
        for chunk in tokens.chunks(self.prefill_rows) {
            self.consume_rows(chunk, false)?;
        }
        Ok(())
    }

    fn check_tokens(&self, tokens: &[u32]) -> Result<(), EngError> {
        if self.failed
            || tokens.len() > (self.context - self.position) as usize
            || tokens.iter().any(|&token| token as usize >= VOCAB)
        {
            return Err(EngError::BadRequest(
                "Inkling sequence requires reset or token exceeds its limits".into(),
            ));
        }
        Ok(())
    }

    fn consume(&mut self, token: u32, logits: bool) -> Result<Row, EngError> {
        self.check_tokens(&[token])?;
        self.consume_rows(&[token], logits)
    }

    fn consume_rows(&mut self, tokens: &[u32], logits: bool) -> Result<Row, EngError> {
        // A failed visit may have appended to only a subset of the KV or convolution histories.
        self.failed = true;
        let result = self.feed_window(tokens, logits)?;
        self.failed = false;
        self.position += tokens.len() as u32;
        Ok(result)
    }

    fn moe(&mut self, layer: usize, prefix: &str, inputs: &[Row]) -> Result<Vec<bf16>, EngError> {
        let gate = format!("{prefix}.mlp.gate.weight");
        let mut logits = Vec::with_capacity(inputs.len());
        for input in inputs {
            let row: [f32; EXPERTS] = self
                .dense
                .matvec(&gate, input)?
                .try_into()
                .map_err(|_| invalid())?;
            logits.push(row);
        }
        let bias = self.dense.vector(&format!("{prefix}.mlp.gate.bias"))?;
        let routes = InklingRoutes::from_logits(
            &logits,
            bias.as_slice().try_into().map_err(|_| invalid())?,
        )?;
        let shape = vec![inputs.len() as u32, HIDDEN as u32];
        let host = Tensor::try_from_host(
            Dtype::F32,
            shape.clone(),
            inputs
                .iter()
                .flatten()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        )?;
        let host =
            self.experts
                .forward_quantized_readback(&mut self.metal, layer, &routes, &host)?;
        if host.dtype() != Dtype::F32 || host.shape() != shape.as_slice() {
            return Err(invalid());
        }
        Ok(host
            .host_bytes()
            .ok_or_else(invalid)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| bf16::from_f32(f32::from_le_bytes(*w)))
            .collect())
    }

    fn feed_window(&mut self, tokens: &[u32], logits: bool) -> Result<Row, EngError> {
        let embed_norm = self.dense.vector("model.llm.embed_norm.weight")?;
        let mut rows = Vec::with_capacity(tokens.len());
        for &token in tokens {
            let embedded = self.dense.row("model.llm.embed.weight", token as usize)?;
            rows.push(PrefillRow {
                blocks: vec![normalize(&embedded, &embed_norm)?],
                prelude: Vec::new(),
                bank: Vec::new(),
            });
        }
        let mut inputs = Vec::with_capacity(rows.len());
        for visit in visits() {
            let layer = visit.weight_layer;
            let prefix = format!("model.llm.layers.{layer}");
            // Every attention and convolution history is visit-private. Run rows in
            // causal order within each visit and keep the loop residual state per row.
            for (offset, row) in rows.iter_mut().enumerate() {
                inputs.push(self.prepare_mlp(
                    visit,
                    &prefix,
                    self.position + offset as u32,
                    row,
                )?);
            }
            let outputs = self.moe(layer, &prefix, &inputs)?;
            inputs.clear();
            let sconv = format!("{prefix}.mlp_sconv");
            for (row, moe) in rows.iter_mut().zip(outputs.chunks_exact(HIDDEN)) {
                let m = conv(
                    &mut self.dense,
                    &mut self.conv,
                    &sconv,
                    visit.state_slot,
                    InklingConvStream::MlpOutput,
                    moe,
                )?;
                for (x, v) in row.blocks.last_mut().unwrap().iter_mut().zip(m) {
                    *x = rounded(*x + v);
                }
            }
        }
        if logits {
            self.output_logits(&rows.last().ok_or_else(invalid)?.blocks)
        } else {
            Ok(Vec::new())
        }
    }

    fn prepare_mlp(
        &mut self,
        visit: LayerVisit,
        prefix: &str,
        position: u32,
        row: &mut PrefillRow,
    ) -> Result<Row, EngError> {
        let i = visit.weight_layer;
        let looped = visit.pass.unwrap_or(0) > 0;
        let PrefillRow {
            blocks,
            prelude,
            bank,
        } = row;
        if i == CORE_START {
            if looped {
                *bank = blocks.split_off(prelude.len());
                blocks.clone_from(prelude);
            } else {
                prelude.clone_from(blocks);
            }
        }
        let query = if i > 0 {
            self.dense
                .vector(&format!("model.llm.attn_res.attn_queries.{}", i - 1))?
        } else {
            vec![]
        };
        let mut inputs = Vec::with_capacity(3);
        for (index, stream) in ["attn", "k", "v"].into_iter().enumerate() {
            if is_local(i) && !looped && index > 0 {
                break;
            }
            let dest = if i == 0 {
                blocks[0].clone()
            } else {
                let offset = if is_local(i) { 0 } else { index * HIDDEN };
                mix(
                    blocks,
                    query.get(offset..offset + HIDDEN).ok_or_else(invalid)?,
                )?
            };
            let blended = self.maybe_blend(prefix, i, looped, bank, stream, &dest)?;
            let norm = if is_local(i) {
                "attn_norm"
            } else {
                ["q_input_norm", "k_input_norm", "v_input_norm"][index]
            };
            inputs.push(
                normalize(
                    &blended,
                    &self.dense.vector(&format!("{prefix}.{norm}.weight"))?,
                )?
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
            );
        }
        let tau = 1. + 0.1 * (((position + 1) as f32 / 8192.).max(1.)).ln();
        let attn = self.attention.step(
            &self.dense,
            &mut self.conv,
            visit.state_slot,
            (
                &inputs[0],
                inputs.get(1).unwrap_or(&inputs[0]),
                inputs.get(2).unwrap_or(&inputs[0]),
            ),
            tau,
        )?;
        let a = conv(
            &mut self.dense,
            &mut self.conv,
            &format!("{prefix}.attn_sconv"),
            visit.state_slot,
            InklingConvStream::AttentionOutput,
            &attn,
        )?;
        if i % 6 == 0 {
            blocks.push(a);
        } else {
            for (x, v) in blocks.last_mut().unwrap().iter_mut().zip(a) {
                *x += v;
            }
        }
        let mixed = mix(
            blocks,
            &self
                .dense
                .vector(&format!("model.llm.attn_res.mlp_queries.{i}"))?,
        )?;
        let blended = self.maybe_blend(prefix, i, looped, bank, "mlp", &mixed)?;
        normalize(
            &blended,
            &self.dense.vector(&format!("{prefix}.mlp_norm.weight"))?,
        )
    }

    fn output_logits(&mut self, blocks: &[Row]) -> Result<Row, EngError> {
        let mixed = mix(
            blocks,
            &self.dense.vector("model.llm.attn_res.output_query")?,
        )?;
        let result = normalize(&mixed, &self.dense.vector("model.llm.norm.weight")?)?;
        let mut scores = self
            .dense
            .matvec_rows("model.llm.unembed.weight", &result, 0..VOCAB)?;
        scores.iter_mut().for_each(|v| *v = 30. * (*v / 30.).tanh());
        if scores.len() != VOCAB || !scores.iter().all(|v| v.is_finite()) {
            return Err(invalid());
        }
        Ok(scores)
    }
    fn maybe_blend(
        &mut self,
        prefix: &str,
        layer: usize,
        looped: bool,
        bank: &[Row],
        stream: &str,
        dest: &[f32],
    ) -> Result<Row, EngError> {
        if !looped {
            return Ok(dest.to_vec());
        }
        let (suffix, offset) = match stream {
            "attn" => ("", 0),
            "k" => ("_kv", 0),
            "v" => ("_kv", 1),
            "mlp" => ("", 1),
            _ => return Err(invalid()),
        };
        let q = self.dense.row(
            &format!("model.llm.middle_cycle_cross_pass{suffix}_queries"),
            2 * (layer - CORE_START) + offset,
        )?;
        blend(
            &mut self.dense,
            &format!("{prefix}.controllers.{stream}"),
            dest,
            &mix(bank, &q)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_window_is_explicit_and_bounded() {
        let rows =
            |setting: &str, context| requested_prefill_rows(Some(OsStr::new(setting)), context);
        assert_eq!(requested_prefill_rows(None, 4096), 1);
        assert_eq!(rows("0", 4096), 1);
        assert_eq!(rows("invalid", 4096), 1);
        assert_eq!(rows("4", 4096), 4);
        assert_eq!(rows("4", 2), 2);
        assert_eq!(rows("100", 4096), MAX_PREFILL_ROWS);
    }

    #[test]
    #[ignore = "requires the full released Inkling 2 Mini checkpoint and native Metal"]
    fn checkpoint_prefill_matches_serial_logits_and_followup() {
        let path = std::env::var_os("FIFI_TEST_INKLING_CHECKPOINT")
            .expect("set FIFI_TEST_INKLING_CHECKPOINT to the native checkpoint directory");
        let _lock = crate::native_test_lock();
        let mut decoder =
            Decoder::open_with_prefill_rows(Path::new(&path), 32, Some(OsStr::new("2"))).unwrap();
        assert_eq!(decoder.prefill_rows(), 2, "test needs two admitted rows");
        let tokens = [42, 1147, 2034, 11, 150, 2582, 27];
        // The first six tokens form the prompt. The last token checks the state carried
        // into the next decode. Every prefix ends in a solo logits step, as serving does.
        let before = decoder.expert_cache_stats().visits;
        let reference: Vec<_> = tokens
            .iter()
            .map(|&token| decoder.step(token).unwrap())
            .collect();
        let visits_per_window = super::super::VISITS as u64;
        assert_eq!(
            decoder.expert_cache_stats().visits - before,
            tokens.len() as u64 * visits_per_window,
        );

        for (index, expected) in reference.iter().enumerate() {
            decoder.reset();
            let before = decoder.expert_cache_stats().visits;
            // Odd prefix lengths exercise a short final window after full windows.
            decoder.advance_rows(&tokens[..index]).unwrap();
            assert_eq!(decoder.position(), index);
            let actual = decoder.step(tokens[index]).unwrap();
            assert_eq!(actual.len(), VOCAB);
            let mismatch = actual
                .iter()
                .zip(expected)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert_eq!(
                mismatch, None,
                "logit differed at sequence position {index}"
            );
            assert_eq!(decoder.position(), index + 1);
            assert_eq!(
                decoder.expert_cache_stats().visits - before,
                (index.div_ceil(decoder.prefill_rows()) + 1) as u64 * visits_per_window,
            );
        }
    }

    #[test]
    fn block_attention_scores_normalized_values_and_scales_the_checkpoint_query() {
        let mut query = vec![0.; HIDDEN];
        query[0] = (HIDDEN as f32).sqrt();
        let first = vec![1.; HIDDEN];
        let second = vec![-2.; HIDDEN];
        let actual = mix(&[first, second], &query).unwrap();
        let expected = 1.0_f32.tanh();
        assert!(actual.iter().all(|x| (x - expected).abs() < 2e-6));
        assert!(mix(&[vec![1.]], &query).is_err());
    }
}
