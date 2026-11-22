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
    tensors::{Convention, Mapped, Slot},
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
        recipe::{Global, Role as Wanted, Site, SiteTensor},
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
            ngram: entries[0].role == Global::NgramTable.into(),
        }
    }
}

/// The projector of the host step over a GGUF.
pub(super) struct HostProjector<'a> {
    weights: &'a GgufWeights,
    ngram: &'a GgufNgramRows,
    threads: usize,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    cuda: Option<super::cuda_probe::Projection<'a>>,
}

impl<'a> HostProjector<'a> {
    pub(super) fn new(weights: &'a GgufWeights, ngram: &'a GgufNgramRows, threads: usize) -> Self {
        Self {
            weights,
            ngram,
            threads: threads.max(1),
            #[cfg(all(feature = "native-cuda", target_os = "linux"))]
            cuda: None,
        }
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn with_cuda(mut self, cuda: super::cuda_probe::Projection<'a>) -> Self {
        self.cuda = Some(cuda);
        self
    }

    /// `rows` rows of `part` from row `first` of expert `expert`, dotted with `x`.
    fn matvec_range(
        weights: &GgufWeights,
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
            weights.read_range(
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

    fn matvec(
        &mut self,
        part: &Part,
        expert: Option<u32>,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
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
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = self.cuda.as_mut()
            && cuda.supports(part.ty)
        {
            let mut out = Vec::with_capacity(part.out);
            let mut bytes = Vec::new();
            let chunk_rows = (CHUNK_BYTES as u64 / row_bytes).max(1) as usize;
            for first in (0..part.out).step_by(chunk_rows) {
                let count = chunk_rows.min(part.out - first);
                bytes.resize(count * row_bytes as usize, 0);
                self.weights
                    .read_range(&part.name, base + first as u64 * row_bytes, &mut bytes)?;
                out.extend(cuda.run(part.ty, &bytes, x)?);
            }
            return Ok(out);
        }
        let threads = self.threads.min(part.out.div_ceil(64)).max(1);
        let share = part.out.div_ceil(threads);
        let weights = self.weights;
        let pieces: Vec<Result<Vec<f32>, EngError>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let first = (t * share).min(part.out);
                    let count = share.min(part.out - first);
                    scope.spawn(move || Self::matvec_range(weights, part, base, first, count, x))
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

impl HDelta {
    fn view(&self) -> DeltaWeights<'_, HostWeight> {
        DeltaWeights {
            qkv: &self.qkv,
            z: &self.z,
            b: &self.b,
            a: &self.a,
            out: &self.out,
            conv: &self.conv,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
            norm: &self.norm,
        }
    }
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

const DELTA_ROLES: [Wanted; 9] = [
    Wanted::DeltaQkv,
    Wanted::DeltaGate,
    Wanted::DeltaBeta,
    Wanted::DeltaAlpha,
    Wanted::DeltaOut,
    Wanted::DeltaConv,
    Wanted::DeltaDecay,
    Wanted::DeltaDtBias,
    Wanted::DeltaNorm,
];

fn delta_entries(model: &GgufModel, index: usize) -> Result<[&Mapped; 9], EngError> {
    let mut entries = Vec::with_capacity(DELTA_ROLES.len());
    for role in DELTA_ROLES {
        entries.push(
            model.map().get(role, Some(index)).ok_or_else(|| {
                refuse(format!("{role:?} of DeltaNet layer {index} is not mapped"))
            })?,
        );
    }
    Ok(entries.try_into().expect("one entry per DeltaNet role"))
}

fn load_delta(model: &GgufModel, index: usize) -> Result<HDelta, EngError> {
    let [qkv, z, b, a, out, conv, decay, dt_bias, norm] = delta_entries(model, index)?;
    let dims = &model.config().dims;
    let (kh, vh) = (dims.key_heads as usize, dims.value_heads as usize);
    let vd = dims.value_dim as usize;
    let twice_key = 2 * (dims.key_heads * dims.key_dim) as usize;
    let ck = dims.conv_kernel as usize;
    let floats = |entry| read_floats(&model.weights, entry);
    let tiled = |entry, skip, hd| {
        HostWeight::new(
            &[entry],
            Remap::ValueOut {
                skip,
                hd,
                key_heads: kh,
                value_heads: vh,
            },
        )
    };
    let a_log: Vec<f32> = floats(decay)?
        .into_iter()
        .map(|value| (-value).ln())
        .collect();
    Ok(HDelta {
        qkv: tiled(qkv, twice_key, vd),
        z: tiled(z, 0, vd),
        b: tiled(b, 0, 1),
        a: tiled(a, 0, 1),
        out: HostWeight::new(
            &[out],
            Remap::ValueIn {
                hd: vd,
                key_heads: kh,
                value_heads: vh,
            },
        ),
        conv: untile(&floats(conv)?, twice_key * ck, vd * ck, kh, vh),
        a_log: untile(&a_log, 0, 1, kh, vh),
        dt_bias: untile(&floats(dt_bias)?, 0, 1, kh, vh),
        norm: floats(norm)?,
    })
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
        let map = model.map();
        let weights = &model.weights;
        let get = |role: Slot, layer: Option<usize>| -> Result<&Mapped, EngError> {
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
                norm: floats(Slot::hyper(site, SiteTensor::Norm), layer)?,
                down: plain(Slot::hyper(site, SiteTensor::Down), layer)?,
                up: plain(Slot::hyper(site, SiteTensor::Up), layer)?,
                inject: if site == Site::Head {
                    None
                } else {
                    Some(plain(Slot::hyper(site, SiteTensor::Inject), layer)?)
                },
            })
        };
        let mut out_layers = Vec::new();
        for (index, &attention) in dims.kinds.iter().enumerate().take(layers) {
            let at = Some(index);
            let attn = if attention {
                HAttn::Qsa(HQsa {
                    q: plain(Wanted::QProj.into(), at)?,
                    k: plain(Wanted::KProj.into(), at)?,
                    v: plain(Wanted::VProj.into(), at)?,
                    o: plain(Wanted::OProj.into(), at)?,
                    idx: HostWeight::new(
                        &[
                            get(Wanted::IndexProj.into(), at)?,
                            get(Wanted::IndexKey.into(), at)?,
                        ],
                        Remap::None,
                    ),
                    q_norm: floats(Wanted::QNorm.into(), at)?,
                    k_norm: floats(Wanted::KNorm.into(), at)?,
                    idx_q_norm: floats(Wanted::IndexQNorm.into(), at)?,
                    idx_k_norm: floats(Wanted::IndexKNorm.into(), at)?,
                })
            } else {
                HAttn::Delta(load_delta(model, index)?)
            };
            out_layers.push(HLayer {
                attn_hyper: hyper(Site::Mixer, at)?,
                attn,
                mlp_hyper: hyper(Site::Ffn, at)?,
                moe: HMoe {
                    router: floats(Wanted::Router.into(), at)?,
                    gate: plain(Wanted::ExpertGate.into(), at)?,
                    up: plain(Wanted::ExpertUp.into(), at)?,
                    down: plain(Wanted::ExpertDown.into(), at)?,
                    shared_gate: plain(Wanted::SharedGate.into(), at)?,
                    shared_up: plain(Wanted::SharedUp.into(), at)?,
                    shared_down: plain(Wanted::SharedDown.into(), at)?,
                    shared_scalar: plain(Wanted::SharedRouterGate.into(), at)?,
                },
            });
        }
        let ple = Some(dims.ple_layer);
        let table = get(Global::NgramTable.into(), None)?;
        let hash = &config.hash;
        Ok(Self {
            shape: config.model_shape()?,
            embed: plain(Global::Embedding.into(), None)?,
            lm_head: plain(Global::Head.into(), None)?,
            mixer: hyper(Site::Head, None)?,
            layers: out_layers,
            ple_table: HostWeight::new(&[table], Remap::None),
            ple_rows: [table.dims[1]],
            ple_key: plain(Wanted::PleKey.into(), ple)?,
            ple_value: plain(Wanted::PleValue.into(), ple)?,
            ple_norms: [
                floats(Wanted::PleNormKey.into(), ple)?,
                floats(Wanted::PleNormQuery.into(), ple)?,
                floats(Wanted::PleNormConv.into(), ple)?,
            ],
            ple_conv: floats(Wanted::PleConv.into(), ple)?,
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
                        HAttn::Delta(d) => AttnWeights::Delta(d.view()),
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

/// Full logit rows for successive greedy decisions on one uninterrupted host state.
/// The first row predicts the token after the prompt; each later row feeds the prior argmax.
pub(super) fn greedy_rows(
    model: &GgufModel,
    host: &HostModel,
    prompt: &[u32],
    decisions: usize,
    threads: usize,
) -> Result<Vec<Vec<f32>>, EngError> {
    let ngram = model.ngram()?;
    let projector = HostProjector::new(&model.weights, &ngram, threads);
    greedy_rows_with(projector, host, prompt, decisions)
}

pub(super) fn greedy_rows_with(
    mut projector: HostProjector<'_>,
    host: &HostModel,
    prompt: &[u32],
    decisions: usize,
) -> Result<Vec<Vec<f32>>, EngError> {
    use crate::family::qwen::flash_next::step::{ModelState, step};

    let Some((&last, prefix)) = prompt.split_last() else {
        return Err(refuse("the greedy oracle needs a nonempty prompt"));
    };
    let weights = host.view();
    let mut state = ModelState::new(&host.shape, &weights);
    for &token in prefix {
        step(
            &mut projector,
            &host.shape,
            &weights,
            &mut state,
            token,
            false,
        )?;
    }
    let mut token = last;
    let mut rows = Vec::with_capacity(decisions);
    for _ in 0..decisions {
        let row = step(
            &mut projector,
            &host.shape,
            &weights,
            &mut state,
            token,
            false,
        )?
        .logits;
        token = greedy_token(&row)?;
        rows.push(row);
    }
    Ok(rows)
}

pub(super) fn greedy_token(row: &[f32]) -> Result<u32, EngError> {
    if row.is_empty() || row.iter().any(|value| !value.is_finite()) {
        return Err(refuse(
            "the greedy oracle needs a nonempty finite logit row",
        ));
    }
    let best = row.iter().enumerate().fold(
        0,
        |best, (id, value)| {
            if *value > row[best] { id } else { best }
        },
    );
    u32::try_from(best).map_err(|_| refuse("greedy token ID does not fit u32"))
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
    #[ignore = "emits tensor byte ranges from a header-only fixture"]
    fn delta_tensor_sample_plan() {
        let path = std::env::var_os("FIFI_FLASH_GGUF_PROJECTION_FIXTURE").unwrap();
        let model = GgufModel::open(std::path::Path::new(&path))
            .unwrap()
            .unwrap();
        let entries = delta_entries(&model, 0).unwrap();
        let items: Vec<_> = entries
            .iter()
            .map(|entry| {
                let tensor = model.weights.header().tensor(&entry.name).unwrap();
                serde_json::json!({"name": entry.name, "shard": tensor.shard(),
                "start": tensor.abs_offset(), "bytes": entry.bytes})
            })
            .collect();
        let output = std::env::var_os("FIFI_FLASH_DELTA_PLAN").unwrap();
        std::fs::write(output, serde_json::to_vec_pretty(&items).unwrap()).unwrap();
    }

    // SHA-256 of byte ranges retrieved from the pinned publisher revision. These detect
    // fixture corruption or sparse holes; they are not publisher full-shard verification.
    const DELTA_DIGESTS: [&str; 9] = [
        "c7f5666e403b1959924995e4a69b421efc2ea4d7c5684c8dde31ef8d715d0428",
        "42dc982fd458f06d8ebfdb1af3faea8fa1621c9c75e33f3dd4e4eaf1a9e17032",
        "9d8237ffbd8a955f0c77cebca4250a18027e1292257f72b9444716371d2756ff",
        "3177daafc8955dbf7cd57c31bc376cd8e3af530bae8a339e870cb990cc888bdb",
        "89140f4a519d9848dcfe5c590d28b924f249622ef81e15636303a8aa8efde502",
        "86971b2b293c0f8d662169eac001156cb7d5d81bbb28fe75d78e358f1c39b476",
        "6e38a1018f33d34678019e53974ad84738381bd8ac092f65596d4cf800ded59d",
        "61f670e1bfb6524a4ea2aa4bb8f9dbaeaba94c752f7485380ca626b336b8f0c6",
        "d27dd7b8d5dae1f3cc9880dd0367825008d8d7873ea0030aa92175634ae08e17",
    ];

    fn matches_digest(bytes: &[u8], expected: &str) -> bool {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(bytes)) == expected
    }

    fn verify_delta_fixture(model: &GgufModel) -> Result<(), EngError> {
        for (entry, expected) in delta_entries(model, 0)?.iter().zip(DELTA_DIGESTS) {
            let (_, data) = model.weights.read_tensor(&entry.name)?;
            if !matches_digest(&data, expected) {
                return Err(refuse(format!(
                    "pinned DeltaNet digest mismatch: {}",
                    entry.name
                )));
            }
        }
        Ok(())
    }

    #[test]
    fn delta_digest_rejects_changed_and_missing_bytes() {
        let hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(matches_digest(b"abc", hash));
        assert!(!matches_digest(b"abd", hash));
        assert!(!matches_digest(b"", hash));
    }

    #[test]
    #[ignore = "requires DeltaNet fixture and projection-only sparse negative fixture"]
    fn delta_fixture_integrity() {
        let open = |key| {
            let path = std::env::var_os(key).unwrap();
            GgufModel::open(std::path::Path::new(&path))
                .unwrap()
                .unwrap()
        };
        verify_delta_fixture(&open("FIFI_FLASH_DELTA_FIXTURE")).unwrap();
        assert!(verify_delta_fixture(&open("FIFI_FLASH_GGUF_PROJECTION_FIXTURE")).is_err());
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "needs staged real DeltaNet tensors and an admitted CUDA device"]
    fn real_delta_block_cpu_cuda_parity() {
        use super::super::cuda_probe::Projection;
        use crate::{
            family::qwen::flash_next::deltanet::{self, DeltaState},
            kernels::cuda::CudaContext,
        };
        let path = std::env::var_os("FIFI_FLASH_DELTA_FIXTURE").expect("set DeltaNet-only fixture");
        let model = GgufModel::open(std::path::Path::new(&path))
            .unwrap()
            .unwrap();
        verify_delta_fixture(&model).expect("complete pinned DeltaNet tensor payloads");
        let weights = load_delta(&model, 0).unwrap();
        let shape = model.config().model_shape().unwrap();
        let ngram = model.ngram().unwrap();
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let launches = std::cell::Cell::new(0);
        let mut cpu = HostProjector::new(&model.weights, &ngram, 8);
        let mut gpu = HostProjector::new(&model.weights, &ngram, 8).with_cuda(Projection {
            ctx: &ctx,
            stream: &stream,
            launches: &launches,
        });
        let mut cpu_state = DeltaState::new(&shape.delta);
        let mut gpu_state = DeltaState::new(&shape.delta);
        let hidden = weights.qkv.parts[0].input;
        for token in 0..3 {
            let x: Vec<f32> = (0..hidden)
                .map(|i| (((i * 37 + token * 23) % 251) as f32 - 125.0) / 125.0)
                .collect();
            let want = deltanet::step(&mut cpu, &shape.delta, &weights.view(), &mut cpu_state, &x)
                .unwrap();
            let got = deltanet::step(&mut gpu, &shape.delta, &weights.view(), &mut gpu_state, &x)
                .unwrap();
            assert_eq!(want.len(), got.len());
            let norm = want
                .iter()
                .map(|v| f64::from(*v).powi(2))
                .sum::<f64>()
                .sqrt();
            let error = want
                .iter()
                .zip(&got)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(norm.is_finite() && norm > 0.0 && error.is_finite());
            eprintln!("step {token}: relative L2 {}", error / norm);
            assert!(error / norm < 0.03);
        }
        assert!(launches.get() > 0);
        eprintln!("CUDA projection launches: {}", launches.get());
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    #[ignore = "needs an admitted CUDA device; synthetic split GGUF"]
    fn mapped_host_weight_dispatches_to_cuda() {
        use super::super::{
            cuda_probe::Projection,
            testkit::{Spec, write_flash},
        };
        use crate::kernels::cuda::CudaContext;

        let pack = write_flash(&Spec {
            shards: 4,
            ..Spec::default()
        });
        let model = GgufModel::open(&pack.first).unwrap().unwrap();
        let tensor = model
            .map
            .entries()
            .iter()
            .find(|t| t.ty == GgufType::Q80 && t.dims.len() == 2)
            .unwrap();
        let mut weight = HostWeight::new(&[tensor], Remap::None);
        let k = weight.parts[0].input;
        weight.parts[0].out = 3;
        let mut bytes = Vec::new();
        for code in [1_i8, -2, 3] {
            for _ in 0..k / 32 {
                bytes.extend(0x3800_u16.to_le_bytes());
                bytes.extend([code as u8; 32]);
            }
        }
        pack.patch(&tensor.name, 0, &bytes);
        let ngram = model.ngram().unwrap();
        let x = vec![1.0; k];
        let expected = HostProjector::new(&model.weights, &ngram, 1)
            .linear(&weight, &x)
            .unwrap();
        let ctx = CudaContext::new().unwrap();
        let stream = ctx.stream().unwrap();
        let launches = std::cell::Cell::new(0);
        let got = HostProjector::new(&model.weights, &ngram, 1)
            .with_cuda(Projection {
                ctx: &ctx,
                stream: &stream,
                launches: &launches,
            })
            .linear(&weight, &x)
            .unwrap();
        assert_eq!(launches.get(), 1);
        assert_eq!(got.len(), expected.len());
        for (a, b) in got.into_iter().zip(expected) {
            assert!((a - b).abs() <= b.abs() * 0.005, "native {a}, host {b}");
        }
    }

    #[test]
    fn greedy_token_rejects_invalid_rows_and_breaks_ties_by_lowest_id() {
        assert_eq!(greedy_token(&[-3.0, 5.0, 5.0, 0.0]).unwrap(), 1);
        for row in [&[][..], &[f32::NAN], &[1.0, f32::INFINITY]] {
            assert!(greedy_token(row).is_err());
        }
    }

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
