//! The host forward pass of Flash-Next run over a GGUF: the existing host step
//! (`flash_next::step`) with weights read from the file and decoded on the CPU.
//!
//! The step expects the layout of the Hugging Face checkpoint, so the loader converts what
//! the GGUF stores differently, on read:
//! - norm gammas stored as `1 + w` are returned to `w` (the hyper-connection, indexer, PLE and
//!   attention query and key norms; the step applies the 1 itself);
//! - `ssm_a` holds `-exp(A_log)`, so `A_log` is `ln(-a)`;
//! - the DeltaNet value heads are stored tiled (key head `h % key_heads`) and the step reads
//!   them grouped (key head `h / (value_heads / key_heads)`): the rows of the qkv, z, beta and
//!   alpha projections, the value channels of the convolution, `A_log` and `dt_bias` are
//!   reordered on read, and the output projection's input is reordered before its matvec.
//!
//! Projections stream the tensor from the file in chunks and decode each chunk to F32, so no
//! matrix is ever held whole (the output head has 636 million weights). Routed experts are read
//! as one slice per expert. The n-gram rows come from [`GgufNgramRows`].
//!
//! This is a reference for correctness checks. It decodes at CPU speed.

use std::collections::BTreeMap;

use super::{
    GgufModel,
    ngram::GgufNgramRows,
    tensors::{Convention, Mapped, Role, Site},
};
use crate::{
    EngError,
    family::{
        project::Projector,
        qwen::flash_next::{
            deltanet::DeltaWeights,
            hyper::HyperWeights,
            moe::MoeWeights,
            ple::PleWeights,
            qsa::QsaWeights,
            step::{AttnWeights, FlashLayerWeights, ModelShape, ModelWeights},
        },
    },
    weights::{GgufType, GgufWeights, dequantize, dequantize_into},
};

/// Chunk of a tensor decoded at a time, in stored bytes.
const CHUNK_BYTES: usize = 24 << 20;

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF host forward: {}", detail.as_ref()))
}

/// How a projection's rows or input are reordered from the file's value-head order to the
/// step's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Remap {
    None,
    /// The outputs from `skip` on are value heads of `hd` values each, tiled in the file.
    ValueOut {
        skip: usize,
        hd: usize,
        key_heads: usize,
        value_heads: usize,
    },
    /// The input is value heads of `hd` values each, grouped; the file wants them tiled.
    ValueIn {
        hd: usize,
        key_heads: usize,
        value_heads: usize,
    },
}

/// Position in the file's tiled order of the value head the step calls `h`.
fn tiled_position(h: usize, key_heads: usize, value_heads: usize) -> usize {
    let per_key = value_heads / key_heads;
    let (group, rep) = (h / per_key, h % per_key);
    rep * key_heads + group
}

/// Reorders `values[skip..]`, heads of `unit` values, from tiled to grouped order.
fn untile(
    values: &[f32],
    skip: usize,
    unit: usize,
    key_heads: usize,
    value_heads: usize,
) -> Vec<f32> {
    let mut out = values.to_vec();
    for h in 0..value_heads {
        let from = skip + tiled_position(h, key_heads, value_heads) * unit;
        out[skip + h * unit..skip + (h + 1) * unit].copy_from_slice(&values[from..from + unit]);
    }
    out
}

/// Reorders `values`, heads of `unit` values, from grouped to tiled order.
fn tile(values: &[f32], unit: usize, key_heads: usize, value_heads: usize) -> Vec<f32> {
    let mut out = values.to_vec();
    for h in 0..value_heads {
        let to = tiled_position(h, key_heads, value_heads) * unit;
        out[to..to + unit].copy_from_slice(&values[h * unit..(h + 1) * unit]);
    }
    out
}

#[derive(Debug, Clone)]
struct Part {
    name: String,
    ty: GgufType,
    input: usize,
    out: usize,
}

/// A weight of the host step: tensors read from the file on use.
#[derive(Debug, Clone)]
pub(super) struct HostWeight {
    parts: Vec<Part>,
    remap: Remap,
    /// Whether the tensor is a stack of experts (the last dimension indexes them).
    experts: usize,
    ngram: bool,
}

impl HostWeight {
    fn new(entries: &[&Mapped], remap: Remap) -> Self {
        let parts = entries
            .iter()
            .map(|entry| Part {
                name: entry.name.clone(),
                ty: entry.ty,
                input: entry.dims[0] as usize,
                out: entry.dims.get(1).copied().unwrap_or(1) as usize,
            })
            .collect();
        let experts = entries[0].dims.get(2).copied().unwrap_or(0) as usize;
        Self {
            parts,
            remap,
            experts,
            ngram: entries[0].role == Role::NgramTable,
        }
    }
}

/// The projector of the host step over a GGUF.
pub(super) struct HostProjector<'a> {
    weights: &'a GgufWeights,
    ngram: &'a GgufNgramRows,
    threads: usize,
}

impl<'a> HostProjector<'a> {
    pub(super) fn new(weights: &'a GgufWeights, ngram: &'a GgufNgramRows, threads: usize) -> Self {
        Self {
            weights,
            ngram,
            threads: threads.max(1),
        }
    }

    /// `rows` rows of `part` from row `first` of expert `expert`, dotted with `x`.
    fn matvec_range(
        &self,
        part: &Part,
        base: u64,
        first: usize,
        count: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        let block = part.ty.block_elems() as usize;
        let row_bytes = part.input / block * part.ty.block_bytes() as usize;
        let per_chunk = (CHUNK_BYTES / row_bytes).max(1);
        let mut out = Vec::with_capacity(count);
        let mut bytes = Vec::new();
        let mut decoded = Vec::new();
        let mut done = 0;
        while done < count {
            let rows = per_chunk.min(count - done);
            bytes.resize(rows * row_bytes, 0);
            decoded.resize(rows * part.input, 0f32);
            self.weights.read_range(
                &part.name,
                base + ((first + done) * row_bytes) as u64,
                &mut bytes,
            )?;
            dequantize_into(part.ty, &bytes, &mut decoded)?;
            out.extend(
                decoded
                    .chunks_exact(part.input)
                    .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>()),
            );
            done += rows;
        }
        Ok(out)
    }

    fn matvec(&self, part: &Part, expert: Option<u32>, x: &[f32]) -> Result<Vec<f32>, EngError> {
        if x.len() != part.input {
            return Err(refuse(format!(
                "{} takes {} inputs, got {}",
                part.name,
                part.input,
                x.len()
            )));
        }
        let block = part.ty.block_elems() as usize;
        let row_bytes = (part.input / block * part.ty.block_bytes() as usize) as u64;
        let base = expert.map_or(0, |e| u64::from(e) * part.out as u64 * row_bytes);
        let threads = self.threads.min(part.out.div_ceil(64)).max(1);
        let share = part.out.div_ceil(threads);
        let pieces: Vec<Result<Vec<f32>, EngError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let first = (t * share).min(part.out);
                    let count = share.min(part.out - first);
                    scope.spawn(move || self.matvec_range(part, base, first, count, x))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("matvec thread"))
                .collect()
        });
        let mut out = Vec::with_capacity(part.out);
        for piece in pieces {
            out.extend(piece?);
        }
        Ok(out)
    }
}

impl Projector for HostProjector<'_> {
    type Weight = HostWeight;

    fn linear(&mut self, weight: &HostWeight, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let tiled_input;
        let x = match weight.remap {
            Remap::ValueIn {
                hd,
                key_heads,
                value_heads,
            } => {
                tiled_input = tile(x, hd, key_heads, value_heads);
                &tiled_input[..]
            }
            _ => x,
        };
        let mut out = Vec::new();
        for part in &weight.parts {
            out.extend(self.matvec(part, None, x)?);
        }
        if let Remap::ValueOut {
            skip,
            hd,
            key_heads,
            value_heads,
        } = weight.remap
        {
            out = untile(&out, skip, hd, key_heads, value_heads);
        }
        Ok(out)
    }

    fn expert_linear(
        &mut self,
        stack: &HostWeight,
        expert: u32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        if expert as usize >= stack.experts {
            return Err(refuse(format!("expert {expert} of {}", stack.experts)));
        }
        self.matvec(&stack.parts[0], Some(expert), x)
    }

    fn rows(&mut self, table: &HostWeight, rows: &[u32]) -> Result<Vec<f32>, EngError> {
        let part = &table.parts[0];
        let mut out = Vec::with_capacity(rows.len() * part.input);
        for &row in rows {
            if table.ngram {
                out.extend(self.ngram.row(u64::from(row))?);
            } else {
                let block = part.ty.block_elems() as usize;
                let row_bytes = part.input / block * part.ty.block_bytes() as usize;
                let mut bytes = vec![0u8; row_bytes];
                self.weights.read_range(
                    &part.name,
                    u64::from(row) * row_bytes as u64,
                    &mut bytes,
                )?;
                out.extend(dequantize(part.ty, &bytes)?);
            }
        }
        Ok(out)
    }
}

struct HHyper {
    norm: Vec<f32>,
    down: HostWeight,
    up: HostWeight,
    inject: Option<HostWeight>,
}

struct HDelta {
    qkv: HostWeight,
    z: HostWeight,
    b: HostWeight,
    a: HostWeight,
    out: HostWeight,
    conv: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
}

struct HQsa {
    q: HostWeight,
    k: HostWeight,
    v: HostWeight,
    o: HostWeight,
    idx: HostWeight,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    idx_q_norm: Vec<f32>,
    idx_k_norm: Vec<f32>,
}

enum HAttn {
    Delta(HDelta),
    Qsa(HQsa),
}

struct HMoe {
    router: Vec<f32>,
    gate: HostWeight,
    up: HostWeight,
    down: HostWeight,
    shared_gate: HostWeight,
    shared_up: HostWeight,
    shared_down: HostWeight,
    shared_scalar: HostWeight,
}

struct HLayer {
    attn_hyper: HHyper,
    attn: HAttn,
    mlp_hyper: HHyper,
    moe: HMoe,
}

/// The weights of the first layers of a Flash-Next GGUF in the form the host step reads.
pub(super) struct HostModel {
    pub(super) shape: ModelShape,
    embed: HostWeight,
    lm_head: HostWeight,
    mixer: HHyper,
    layers: Vec<HLayer>,
    ple_table: HostWeight,
    ple_rows: [u64; 1],
    ple_key: HostWeight,
    ple_value: HostWeight,
    ple_norms: [Vec<f32>; 3],
    ple_conv: Vec<f32>,
    multipliers: Vec<i64>,
    vocab: Vec<i64>,
    offsets: Vec<i64>,
}

fn read_floats(weights: &GgufWeights, entry: &Mapped) -> Result<Vec<f32>, EngError> {
    let (ty, bytes) = weights.read_tensor(&entry.name)?;
    let mut values = dequantize(ty, &bytes)?;
    if entry.convention == Convention::GammaOnePlusW {
        for value in &mut values {
            *value -= 1.0;
        }
    }
    Ok(values)
}

impl HostModel {
    /// Reads the first `layers` layers (the PLE layer must be among them), the embedding, the
    /// final mixer and the output head. The result runs the step on a truncated model.
    pub(super) fn load(model: &GgufModel, layers: usize) -> Result<Self, EngError> {
        let config = model.config();
        let dims = &config.dims;
        if layers <= dims.ple_layer || layers > dims.kinds.len() {
            return Err(refuse(format!(
                "{layers} layers must include the PLE layer {} and be at most {}",
                dims.ple_layer,
                dims.kinds.len()
            )));
        }
        let (kh, vh) = (dims.key_heads as usize, dims.value_heads as usize);
        let vd = dims.value_dim as usize;
        let map = model.map();
        let weights = &model.weights;
        let get = |role: Role, layer: Option<usize>| -> Result<&Mapped, EngError> {
            map.get(role, layer)
                .ok_or_else(|| refuse(format!("{role:?} of layer {layer:?} is not mapped")))
        };
        let plain = |role, layer| -> Result<HostWeight, EngError> {
            Ok(HostWeight::new(&[get(role, layer)?], Remap::None))
        };
        let floats =
            |role, layer| -> Result<Vec<f32>, EngError> { read_floats(weights, get(role, layer)?) };
        let hyper = |site: Site, layer: Option<usize>| -> Result<HHyper, EngError> {
            Ok(HHyper {
                norm: floats(Role::HyperNorm(site), layer)?,
                down: plain(Role::HyperDown(site), layer)?,
                up: plain(Role::HyperUp(site), layer)?,
                inject: if site == Site::Mixer {
                    None
                } else {
                    Some(plain(Role::HyperInject(site), layer)?)
                },
            })
        };
        let mut out_layers = Vec::new();
        for (index, &attention) in dims.kinds.iter().enumerate().take(layers) {
            let at = Some(index);
            let attn = if attention {
                HAttn::Qsa(HQsa {
                    q: plain(Role::QsaQ, at)?,
                    k: plain(Role::QsaK, at)?,
                    v: plain(Role::QsaV, at)?,
                    o: plain(Role::QsaO, at)?,
                    idx: HostWeight::new(
                        &[get(Role::QsaIndexQ, at)?, get(Role::QsaIndexK, at)?],
                        Remap::None,
                    ),
                    q_norm: floats(Role::QsaQNorm, at)?,
                    k_norm: floats(Role::QsaKNorm, at)?,
                    idx_q_norm: floats(Role::QsaIndexQNorm, at)?,
                    idx_k_norm: floats(Role::QsaIndexKNorm, at)?,
                })
            } else {
                let key_width = (dims.key_heads * dims.key_dim) as usize;
                let ck = dims.conv_kernel as usize;
                let tiled_rows = |role, skip| {
                    Ok::<_, EngError>(HostWeight::new(
                        &[get(role, at)?],
                        Remap::ValueOut {
                            skip,
                            hd: if role == Role::DeltaBeta || role == Role::DeltaAlpha {
                                1
                            } else {
                                vd
                            },
                            key_heads: kh,
                            value_heads: vh,
                        },
                    ))
                };
                let conv = untile(
                    &floats(Role::DeltaConv, at)?,
                    2 * key_width * ck,
                    vd * ck,
                    kh,
                    vh,
                );
                let a_log: Vec<f32> = floats(Role::DeltaDecay, at)?
                    .iter()
                    .map(|decay| (-decay).ln())
                    .collect();
                HAttn::Delta(HDelta {
                    qkv: tiled_rows(Role::DeltaQkv, 2 * key_width)?,
                    z: tiled_rows(Role::DeltaZ, 0)?,
                    b: tiled_rows(Role::DeltaBeta, 0)?,
                    a: tiled_rows(Role::DeltaAlpha, 0)?,
                    out: HostWeight::new(
                        &[get(Role::DeltaOut, at)?],
                        Remap::ValueIn {
                            hd: vd,
                            key_heads: kh,
                            value_heads: vh,
                        },
                    ),
                    conv,
                    a_log: untile(&a_log, 0, 1, kh, vh),
                    dt_bias: untile(&floats(Role::DeltaDtBias, at)?, 0, 1, kh, vh),
                    norm: floats(Role::DeltaNorm, at)?,
                })
            };
            out_layers.push(HLayer {
                attn_hyper: hyper(Site::Attn, at)?,
                attn,
                mlp_hyper: hyper(Site::Mlp, at)?,
                moe: HMoe {
                    router: floats(Role::Router, at)?,
                    gate: plain(Role::ExpertGate, at)?,
                    up: plain(Role::ExpertUp, at)?,
                    down: plain(Role::ExpertDown, at)?,
                    shared_gate: plain(Role::SharedGate, at)?,
                    shared_up: plain(Role::SharedUp, at)?,
                    shared_down: plain(Role::SharedDown, at)?,
                    shared_scalar: plain(Role::SharedScalar, at)?,
                },
            });
        }
        let ple = Some(dims.ple_layer);
        let table = get(Role::NgramTable, None)?;
        let hash = &config.hash;
        Ok(Self {
            shape: config.model_shape()?,
            embed: plain(Role::Embedding, None)?,
            lm_head: plain(Role::Head, None)?,
            mixer: hyper(Site::Mixer, None)?,
            layers: out_layers,
            ple_table: HostWeight::new(&[table], Remap::None),
            ple_rows: [table.dims[1]],
            ple_key: plain(Role::PleKey, ple)?,
            ple_value: plain(Role::PleValue, ple)?,
            ple_norms: [
                floats(Role::PleNormKey, ple)?,
                floats(Role::PleNormQuery, ple)?,
                floats(Role::PleNormConv, ple)?,
            ],
            ple_conv: floats(Role::PleConv, ple)?,
            multipliers: hash.multipliers.clone(),
            vocab: hash.vocab.clone(),
            offsets: hash.offsets.clone(),
        })
    }

    pub(super) fn view(&self) -> ModelWeights<'_, HostWeight> {
        fn hyper(h: &HHyper) -> HyperWeights<'_, HostWeight> {
            HyperWeights {
                norm: &h.norm,
                down: &h.down,
                up: &h.up,
                inject: h.inject.as_ref(),
            }
        }
        ModelWeights {
            embed: &self.embed,
            layers: self
                .layers
                .iter()
                .map(|layer| FlashLayerWeights {
                    attn_hyper: hyper(&layer.attn_hyper),
                    attn: match &layer.attn {
                        HAttn::Delta(d) => AttnWeights::Delta(DeltaWeights {
                            qkv: &d.qkv,
                            z: &d.z,
                            b: &d.b,
                            a: &d.a,
                            out: &d.out,
                            conv: &d.conv,
                            a_log: &d.a_log,
                            dt_bias: &d.dt_bias,
                            norm: &d.norm,
                        }),
                        HAttn::Qsa(q) => AttnWeights::Qsa(QsaWeights {
                            q: &q.q,
                            k: &q.k,
                            v: &q.v,
                            o: &q.o,
                            idx: &q.idx,
                            q_norm: &q.q_norm,
                            k_norm: &q.k_norm,
                            idx_q_norm: &q.idx_q_norm,
                            idx_k_norm: &q.idx_k_norm,
                        }),
                    },
                    mlp_hyper: hyper(&layer.mlp_hyper),
                    moe: MoeWeights {
                        router: &layer.moe.router,
                        gate: &layer.moe.gate,
                        up: &layer.moe.up,
                        down: &layer.moe.down,
                        shared_gate: &layer.moe.shared_gate,
                        shared_up: &layer.moe.shared_up,
                        shared_down: &layer.moe.shared_down,
                        shared_scalar: &layer.moe.shared_scalar,
                    },
                })
                .collect(),
            ple: PleWeights {
                multipliers: &self.multipliers,
                vocab: &self.vocab,
                offsets: &self.offsets,
                shards: std::slice::from_ref(&self.ple_table),
                shard_rows: &self.ple_rows,
                weight_scale: 1.0,
                key: &self.ple_key,
                value: &self.ple_value,
                norm_key: &self.ple_norms[0],
                norm_query: &self.ple_norms[1],
                norm_conv: &self.ple_norms[2],
                conv: &self.ple_conv,
            },
            mixer: hyper(&self.mixer),
            lm_head: &self.lm_head,
        }
    }
}

/// Summary of one forward over a token sequence.
pub(super) struct HostRun {
    /// Post-MoE residual of every layer after each token.
    pub(super) layer_out: Vec<Vec<Vec<f32>>>,
    pub(super) logits: Vec<Vec<f32>>,
}

/// Runs the host step over `tokens`, recording every layer's output.
pub(super) fn run(
    model: &GgufModel,
    host: &HostModel,
    tokens: &[u32],
    threads: usize,
) -> Result<HostRun, EngError> {
    let ngram = model.ngram()?;
    let mut projector = HostProjector::new(&model.weights, &ngram, threads);
    let weights = host.view();
    let mut state = crate::family::qwen::flash_next::step::ModelState::new(&host.shape, &weights);
    let mut run = HostRun {
        layer_out: Vec::new(),
        logits: Vec::new(),
    };
    for &token in tokens {
        let output = crate::family::qwen::flash_next::step::step(
            &mut projector,
            &host.shape,
            &weights,
            &mut state,
            token,
            true,
        )?;
        run.layer_out.push(output.layer_out.unwrap_or_default());
        run.logits.push(output.logits);
    }
    Ok(run)
}

/// A summary line per layer: the RMS of the residual after the layer.
pub(super) fn rms_by_layer(run: &HostRun) -> BTreeMap<usize, Vec<f32>> {
    let mut out: BTreeMap<usize, Vec<f32>> = BTreeMap::new();
    for token in &run.layer_out {
        for (layer, values) in token.iter().enumerate() {
            let rms = (values.iter().map(|v| v * v).sum::<f32>() / values.len() as f32).sqrt();
            out.entry(layer).or_default().push(rms);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiled_and_grouped_orders_are_inverse_permutations() {
        // 16 key heads, 48 value heads: three value heads per key head.
        let (kh, vh) = (16, 48);
        let grouped: Vec<f32> = (0..vh * 2).map(|i| i as f32).collect();
        let tiled = tile(&grouped, 2, kh, vh);
        assert_ne!(tiled, grouped);
        assert_eq!(untile(&tiled, 0, 2, kh, vh), grouped);
        // Grouped head 1 (key head 0, second value head) sits at tiled position 16.
        assert_eq!(tiled[16 * 2], grouped[2]);
        // A skipped prefix is left alone.
        let mut with_prefix = vec![-1.0, -2.0, -3.0];
        with_prefix.extend(&tiled);
        let back = untile(&with_prefix, 3, 2, kh, vh);
        assert_eq!(&back[..3], &[-1.0, -2.0, -3.0]);
        assert_eq!(&back[3..], &grouped[..]);
    }

    #[test]
    fn a_tiled_projection_row_comes_back_in_the_steps_order() {
        // Tiled order puts the value heads of the same repetition together: heads
        // (group g, repetition r) at r * key_heads + g. With 2 key heads and 4 value heads the
        // file's rows read [g0r0, g1r0, g0r1, g1r1]; the step wants [g0r0, g0r1, g1r0, g1r1].
        let file = [10.0, 20.0, 11.0, 21.0];
        assert_eq!(untile(&file, 0, 1, 2, 4), [10.0, 11.0, 20.0, 21.0]);
        assert_eq!(tile(&[10.0, 11.0, 20.0, 21.0], 1, 2, 4), file);
    }
}
