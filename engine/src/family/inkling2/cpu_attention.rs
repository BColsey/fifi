//! Per-visit state and streaming attention for the native Inkling CPU check.

use std::{collections::VecDeque, sync::Arc};

use super::cpu_forward::{CpuForward, dot, rms_norm, sigmoid};
use super::cpu_weights::Projection;
use super::{Attention, LayerVisit, phase_profile::Work};
use crate::EngError;

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::{cpu_weights::Weights, cuda_projection::gguf::attention};

pub(super) type AttentionProjection = [Vec<f32>; 8];

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct CausalConv(VecDeque<Vec<f32>>);

impl CausalConv {
    fn apply(
        &mut self,
        input: Vec<f32>,
        kernel: &[f32],
        taps: usize,
        dynamic: Option<DynamicTaps>,
    ) -> Vec<f32> {
        let kernels = if dynamic.is_some() { 2 } else { 1 };
        let norm = ((kernels * taps + 1) as f32).sqrt().recip();
        let width = input.len();
        let mut output = vec![0.0; width];
        for (lag, row) in std::iter::once(&input).chain(&self.0).enumerate() {
            for (channel, (result, &x)) in output.iter_mut().zip(row).enumerate() {
                let w = if let Some(dynamic) = &dynamic {
                    kernel[lag * width + channel]
                        + dynamic.coefficients[(channel / dynamic.head_width) * taps + lag]
                } else {
                    kernel[channel * taps + taps - 1 - lag]
                };
                *result += w * x;
            }
        }
        for (result, x) in output.iter_mut().zip(&input) {
            *result = (*result + x) * norm;
        }
        self.0.push_front(input);
        self.0.truncate(taps - 1);
        output
    }
}

struct DynamicTaps {
    coefficients: Vec<f32>,
    head_width: usize,
}

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
struct VisitConvs {
    k: CausalConv,
    v: CausalConv,
    attn: CausalConv,
    mlp: CausalConv,
}

#[derive(Clone, Default)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct VisitState {
    // KV rows are immutable after append, so prefix snapshots share their storage.
    keys: VecDeque<Arc<Vec<f32>>>,
    values: VecDeque<Arc<Vec<f32>>>,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    position: usize,
    conv: VisitConvs,
}

/// The parts of one visit overwritten by one forward row.
pub(super) struct VisitUndo {
    attention: AttentionUndo,
    mlp: CausalConv,
}

pub(super) struct AttentionUndo {
    evicted: Option<[Arc<Vec<f32>>; 2]>,
    k: CausalConv,
    v: CausalConv,
    attn: CausalConv,
}

impl AttentionUndo {
    pub(super) fn with_mlp(self, state: &VisitState) -> VisitUndo {
        VisitUndo {
            attention: self,
            mlp: state.conv.mlp.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct Heads {
    query: usize,
    kv: usize,
    width: usize,
}

impl Heads {
    fn q_channels(self) -> usize {
        self.query * self.width
    }
    fn kv_channels(self) -> usize {
        self.kv * self.width
    }
}

impl CpuForward {
    pub(super) fn attention(
        &self,
        layer: &str,
        visit: LayerVisit,
        state: &mut VisitState,
        projected: AttentionProjection,
    ) -> Result<Vec<f32>, EngError> {
        let heads = self.heads(visit.attention);
        let attn = format!("{layer}.attn");
        let [mut q, k, v, mut relative, sinks, gates, k_taps, v_taps] = projected;
        self.scale_channels(&mut q, &format!("{attn}.q_l2_clip.scale"))?;
        clip_rows(&mut q, heads.width, self.math.qk_l2_clip_threshold);

        let mut k = self.attention_plane(&attn, "k", [k, k_taps], heads, &mut state.conv.k)?;
        let mut v = self.attention_plane(&attn, "v", [v, v_taps], heads, &mut state.conv.v)?;
        clip_rows(&mut k, heads.width, self.math.qk_l2_clip_threshold);
        for head in v.chunks_exact_mut(heads.width) {
            rms_norm(head, self.math.rms_norm_eps);
        }

        self.scale_channels(&mut relative, &format!("{attn}.rel_l2_clip.scale"))?;
        clip_rows(
            &mut relative,
            self.math.d_rel,
            self.math.rel_l2_clip_threshold,
        );
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        let device_output = if let Weights::Gguf(weights) = &self.weights
            && let Some((context, attention)) = weights.resident_attention()
        {
            let _span = self.phase_span(Work::DeviceAttention);
            Some(attention.step(
                context,
                visit.state_slot,
                attention::Row {
                    query: &q,
                    key: &k,
                    value: &v,
                    relative: &relative,
                    sinks: &sinks,
                    gates: &gates,
                },
                attention::History {
                    position: state.position,
                    keys: &state.keys,
                    values: &state.values,
                },
            )?)
        } else {
            None
        };
        #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
        let device_output = None;
        state.push(k, v, self.kv_capacity(visit.attention));

        let out = if let Some(out) = device_output {
            out
        } else {
            let _span = self.phase_span(Work::CpuAttention);
            let table = self.weights.vec(
                &format!("{attn}.rel_logits_proj.proj"),
                self.math.d_rel * self.math.rel_extent,
            )?;
            let mut out = self.attend(state, heads, &q, &relative, &table, &sinks);
            for (out, gate) in out.iter_mut().zip(gates) {
                *out *= sigmoid(gate);
            }
            out
        };
        let raw = {
            let _span = self.phase_span(Work::Projection);
            self.weights
                .project(&format!("{attn}.wo_ud.weight"), &out, self.hidden())?
        };
        self.output_conv(
            raw,
            &format!("{layer}.attn_sconv.weight"),
            &mut state.conv.attn,
        )
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn invalidate_device_attention(&self) {
        if let Weights::Gguf(weights) = &self.weights
            && let Some((_, attention)) = weights.resident_attention()
        {
            attention.invalidate();
        }
    }

    pub(super) fn kv_capacity(&self, attention: Attention) -> usize {
        match attention {
            Attention::Local => self.spec.sliding_window_size as usize,
            Attention::Global => self.context,
        }
    }

    pub(super) fn project_attention(
        &self,
        layer: &str,
        attention: Attention,
        inputs: &[Vec<f32>; 3],
    ) -> Result<AttentionProjection, EngError> {
        self.attention_requests(
            &format!("{layer}.attn"),
            self.heads(attention),
            attention,
            |calls| {
                self.weights
                    .attention_projections(inputs.each_ref().map(Vec::as_slice), calls)
            },
        )
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn prefill_attention_projections(
        &self,
        weights: &super::gguf_weights::GgufTextWeights,
        layer: &str,
        attention: Attention,
        inputs: &[[Vec<f32>; 3]],
    ) -> Result<Option<Vec<AttentionProjection>>, EngError> {
        let refs: Vec<_> = inputs
            .iter()
            .map(|input| input.each_ref().map(Vec::as_slice))
            .collect();
        self.attention_requests(
            &format!("{layer}.attn"),
            self.heads(attention),
            attention,
            |calls| weights.attention_projections_rows(&refs, calls),
        )
    }

    fn attention_requests<R>(
        &self,
        attn: &str,
        heads: Heads,
        attention: Attention,
        run: impl FnOnce([Option<Projection<'_>>; 8]) -> R,
    ) -> R {
        let dynamic = attention == Attention::Global;
        let taps = self.math.sconv_kernel_size * heads.kv;
        let requests = [
            ("wq_du.weight", 0, heads.q_channels()),
            ("wk_dv.weight", 1, heads.kv_channels()),
            ("wv_dv.weight", 2, heads.kv_channels()),
            ("wr_du.weight", 0, heads.query * self.math.d_rel),
            ("attn_sink_proj.weight", 0, heads.query),
            ("gate_attn.weight", 0, heads.q_channels()),
            ("k_sconv.weight_proj", 1, taps),
            ("v_sconv.weight_proj", 2, taps),
        ];
        let names = requests.map(|(suffix, _, _)| format!("{attn}.{suffix}"));
        let calls = std::array::from_fn(|i| {
            let (_, input, rows) = requests[i];
            (i < 6 || dynamic).then_some(Projection {
                name: &names[i],
                input,
                rows,
            })
        });
        let _span = self.phase_span(Work::Projection);
        run(calls)
    }

    fn heads(&self, attention: Attention) -> Heads {
        let spec = &self.spec;
        let (query, kv, width) = match attention {
            Attention::Local => (
                spec.swa_num_attention_heads,
                spec.swa_num_key_value_heads,
                spec.swa_head_dim,
            ),
            Attention::Global => (
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
            ),
        };
        Heads {
            query: query as usize,
            kv: kv as usize,
            width: width as usize,
        }
    }

    fn scale_channels(&self, x: &mut [f32], tensor: &str) -> Result<(), EngError> {
        let weights = self.weights.vec(tensor, x.len())?;
        for (x, weight) in x.iter_mut().zip(weights) {
            *x *= weight;
        }
        Ok(())
    }

    fn attention_plane(
        &self,
        attn: &str,
        plane: &str,
        [raw, coefficients]: [Vec<f32>; 2],
        heads: Heads,
        conv: &mut CausalConv,
    ) -> Result<Vec<f32>, EngError> {
        let taps = self.math.sconv_kernel_size;
        let _span = self.phase_span(Work::Convolution);
        let channels = heads.kv_channels();
        let (name, dynamic) = if coefficients.is_empty() {
            ("weight", None)
        } else {
            (
                "static_w",
                Some(DynamicTaps {
                    coefficients,
                    head_width: heads.width,
                }),
            )
        };
        let kernel = self
            .weights
            .vec(&format!("{attn}.{plane}_sconv.{name}"), taps * channels)?;
        Ok(conv.apply(raw, &kernel, taps, dynamic))
    }

    fn attend(
        &self,
        state: &VisitState,
        heads: Heads,
        q: &[f32],
        relative: &[f32],
        table: &[f32],
        sinks: &[f32],
    ) -> Vec<f32> {
        let mut output = vec![0.0; heads.q_channels()];
        let mut scores = vec![0.0; state.keys.len()];
        let scale = (heads.width as f32).sqrt().recip();
        let dims = self.math.d_rel;
        let extent = self.math.rel_extent;
        for (head, dst) in output.chunks_exact_mut(heads.width).enumerate() {
            let query = &q[head * heads.width..][..heads.width];
            let rel = &relative[head * dims..][..dims];
            let kv_offset = head / (heads.query / heads.kv) * heads.width;
            for (distance, (k, score)) in state.keys.iter().zip(&mut scores).enumerate() {
                let bias = if distance < extent {
                    table
                        .chunks_exact(extent)
                        .zip(rel)
                        .map(|(table, coefficient)| table[distance] * coefficient)
                        .sum()
                } else {
                    0.0
                };
                *score = scale * dot(query, &k[kv_offset..][..heads.width]) + bias;
            }
            let peak = scores.iter().copied().fold(sinks[head], f32::max);
            let mut sum = (sinks[head] - peak).exp();
            for (value, score) in state.values.iter().zip(&scores) {
                let mass = (score - peak).exp();
                sum += mass;
                for (out, v) in dst.iter_mut().zip(&value[kv_offset..][..heads.width]) {
                    *out += mass * v;
                }
            }
            for out in dst {
                *out /= sum;
            }
        }
        output
    }

    fn output_conv(
        &self,
        input: Vec<f32>,
        name: &str,
        state: &mut CausalConv,
    ) -> Result<Vec<f32>, EngError> {
        let taps = self.math.sconv_kernel_size;
        let _span = self.phase_span(Work::Convolution);
        let kernel = self.weights.vec(name, input.len() * taps)?;
        Ok(state.apply(input, &kernel, taps, None))
    }

    pub(super) fn mlp_conv(
        &self,
        input: Vec<f32>,
        layer: &str,
        state: &mut VisitState,
    ) -> Result<Vec<f32>, EngError> {
        self.output_conv(
            input,
            &format!("{layer}.mlp_sconv.weight"),
            &mut state.conv.mlp,
        )
    }
}

impl VisitState {
    pub(super) fn checkpoint(&self, capacity: usize) -> VisitUndo {
        self.checkpoint_attention(capacity).with_mlp(self)
    }

    pub(super) fn checkpoint_attention(&self, capacity: usize) -> AttentionUndo {
        let evicted = if self.keys.len() == capacity {
            self.keys
                .back()
                .zip(self.values.back())
                .map(|(key, value)| [Arc::clone(key), Arc::clone(value)])
        } else {
            None
        };
        AttentionUndo {
            evicted,
            k: self.conv.k.clone(),
            v: self.conv.v.clone(),
            attn: self.conv.attn.clone(),
        }
    }

    pub(super) fn rewind(&mut self, undo: VisitUndo) {
        let AttentionUndo {
            evicted,
            k,
            v,
            attn,
        } = undo.attention;
        self.keys.pop_front();
        self.values.pop_front();
        if let Some([key, value]) = evicted {
            self.keys.push_back(key);
            self.values.push_back(value);
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            self.position -= 1;
        }
        self.conv = VisitConvs {
            k,
            v,
            attn,
            mlp: undo.mlp,
        };
    }

    fn push(&mut self, key: Vec<f32>, value: Vec<f32>, length: usize) {
        self.keys.push_front(Arc::new(key));
        self.values.push_front(Arc::new(value));
        self.keys.truncate(length);
        self.values.truncate(length);
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        {
            self.position += 1;
        }
    }
}

fn clip_rows(rows: &mut [f32], width: usize, limit: f32) {
    for row in rows.chunks_exact_mut(width) {
        let scale = (limit / (dot(row, row).sqrt() + 1e-6)).min(1.0);
        for value in row {
            *value *= scale;
        }
    }
}

#[test]
fn causal_conv_uses_per_visit_history_and_native_tap_order() {
    let mut first = CausalConv::default();
    let mut second = CausalConv::default();
    let weights = [2.0, 3.0];
    let root = 3.0_f32.sqrt();
    assert!((first.apply(vec![4.0], &weights, 2, None)[0] - 16.0 / root).abs() < 1e-6);
    assert!((second.apply(vec![5.0], &weights, 2, None)[0] - 20.0 / root).abs() < 1e-6);
    assert!((first.apply(vec![5.0], &weights, 2, None)[0] - 28.0 / root).abs() < 1e-6);
    let dynamic = |coefficients| {
        Some(DynamicTaps {
            coefficients,
            head_width: 1,
        })
    };
    let mut full = CausalConv::default();
    full.apply(vec![4.0], &[3.0, 2.0], 2, dynamic(vec![1.0, 2.0]));
    let actual = full.apply(vec![5.0], &[3.0, 2.0], 2, dynamic(vec![1.0, 2.0]));
    assert!((actual[0] - 41.0 / 5.0_f32.sqrt()).abs() < 2e-6);
}

#[test]
fn rewind_restores_both_evicted_kv_planes_and_every_convolution() {
    fn advance(visit: &mut VisitState, value: f32, limit: usize) {
        let convs = &mut visit.conv;
        for (phase, conv) in [&mut convs.k, &mut convs.v, &mut convs.attn, &mut convs.mlp]
            .into_iter()
            .enumerate()
        {
            conv.apply(vec![value + phase as f32], &[0.1; 3], 3, None);
        }
        visit.push(vec![value], vec![-value], limit);
    }

    let mut visits = [VisitState::default(), VisitState::default()];
    let limits = [2, 5];
    for (index, (visit, limit)) in visits.iter_mut().zip(limits).enumerate() {
        for row in 0..3 {
            advance(visit, (index * 10 + row) as f32, limit);
        }
    }
    let committed = visits.clone();
    let mut journal = Vec::new();
    for row in 3..6 {
        let mut undo = Vec::new();
        for (index, (visit, limit)) in visits.iter_mut().zip(limits).enumerate() {
            undo.push(visit.checkpoint(limit));
            advance(visit, (index * 10 + row) as f32, limit);
        }
        journal.push(undo);
    }
    for undo in journal.into_iter().rev() {
        for (visit, undo) in visits.iter_mut().zip(undo) {
            visit.rewind(undo);
        }
    }
    assert_eq!(visits, committed);
}
