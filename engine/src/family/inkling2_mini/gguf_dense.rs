//! Adapt GGUF text tensors to the shared Inkling decoder's weight names and strides.

use std::{collections::BTreeMap, ops::Range, sync::Arc};

use super::{
    CONV_TAPS, CORE_END, CORE_START, HIDDEN, LAYERS, checkpoint_error, gguf::Weights,
    gguf_metadata::Layout, is_local,
};
use crate::EngError;

pub(super) struct Dense {
    source: Arc<Weights>,
    entries: BTreeMap<String, Entry>,
}

struct Entry {
    name: String,
    shape: Vec<usize>,
    transform: Transform,
}

#[derive(Clone, Copy, Default)]
enum Transform {
    #[default]
    None,
    Transpose,
    ReverseTaps,
}

impl Dense {
    pub(super) fn new(source: Arc<Weights>) -> Result<Self, EngError> {
        let mut entries = BTreeMap::new();
        let mut add = |canonical: String, name: String, shape: Option<Vec<usize>>, transform| {
            let tensor = source
                .header()
                .tensor(&name)
                .ok_or_else(|| checkpoint_error(format!("missing GGUF tensor {name}")))?;
            let shape =
                shape.unwrap_or_else(|| tensor.dims().iter().rev().map(|v| *v as usize).collect());
            entries.insert(
                canonical,
                Entry {
                    name,
                    shape,
                    transform,
                },
            );
            Ok::<_, EngError>(())
        };
        for (canonical, gguf) in [
            ("embed.weight", "token_embd.weight"),
            ("embed_norm.weight", "token_embd_norm.weight"),
            ("norm.weight", "output_norm.weight"),
            ("unembed.weight", "output.weight"),
            ("attn_res.output_query", "output_res_query.weight"),
        ] {
            add(
                format!("model.llm.{canonical}"),
                gguf.into(),
                None,
                Transform::None,
            )?;
        }
        for i in 0..LAYERS {
            let mut layer = |canonical: &str, gguf: &str, shape, transform| {
                add(
                    format!("model.llm.layers.{i}.{canonical}"),
                    format!("blk.{i}.{gguf}"),
                    shape,
                    transform,
                )
            };
            for (canonical, gguf) in [
                ("attn.wq_du.weight", "attn_q.weight"),
                ("attn.wk_dv.weight", "attn_k.weight"),
                ("attn.wv_dv.weight", "attn_v.weight"),
                ("attn.wo_ud.weight", "attn_output.weight"),
                ("attn.wr_du.weight", "attn_rel.weight"),
                ("attn.gate_attn.weight", "attn_gate.weight"),
                ("attn.attn_sink_proj.weight", "attn_sink_proj.weight"),
                ("attn.q_l2_clip.scale", "attn_q_clip.weight"),
                ("attn.rel_l2_clip.scale", "attn_rel_clip.weight"),
                ("mlp.gate.weight", "ffn_gate_inp.weight"),
                ("mlp.gate.bias", "exp_probs_b.bias"),
                ("mlp_norm.weight", "ffn_norm.weight"),
                ("attn_sconv.weight", "attn_post_conv.weight"),
                ("mlp_sconv.weight", "ffn_post_conv.weight"),
            ] {
                layer(canonical, gguf, None, Transform::None)?;
            }
            layer(
                "attn.rel_logits_proj.proj",
                "attn_rel_bias.weight",
                Some(vec![16, 1024]),
                Transform::Transpose,
            )?;
            if is_local(i) {
                layer(
                    "attn_norm.weight",
                    "attn_norm.weight",
                    None,
                    Transform::None,
                )?;
            } else {
                for stream in ["q", "k", "v"] {
                    layer(
                        &format!("{stream}_input_norm.weight"),
                        &format!("attn_{stream}_inp_norm.weight"),
                        None,
                        Transform::None,
                    )?;
                }
            }
            for stream in ["k", "v"] {
                use super::{HEAD_DIM, KV_HEADS};
                let width = (HEAD_DIM * KV_HEADS) as usize;
                let taps = CONV_TAPS as usize;
                if is_local(i) {
                    layer(
                        &format!("attn.{stream}_sconv.weight"),
                        &format!("attn_{stream}_conv.weight"),
                        Some(vec![width, 1, taps]),
                        Transform::None,
                    )?;
                } else {
                    layer(
                        &format!("attn.{stream}_sconv.static_w"),
                        &format!("attn_{stream}_conv.weight"),
                        Some(vec![taps, width]),
                        Transform::ReverseTaps,
                    )?;
                    layer(
                        &format!("attn.{stream}_sconv.weight_proj"),
                        &format!("attn_{stream}_conv_proj.weight"),
                        None,
                        Transform::None,
                    )?;
                }
            }
            if (CORE_START..CORE_END).contains(&i) {
                for (canonical, mainline) in [
                    ("attn", "q_ctrl"),
                    ("k", "k_ctrl"),
                    ("v", "v_ctrl"),
                    ("mlp", "ffn_ctrl"),
                ] {
                    let stream = match source.layout() {
                        Layout::Mainline => mainline,
                        Layout::Fork => canonical,
                    };
                    for kind in ["down", "gain", "forget"] {
                        layer(
                            &format!("controllers.{canonical}.{kind}_weight"),
                            &format!("loop_{stream}_{kind}.weight"),
                            None,
                            Transform::None,
                        )?;
                    }
                }
            }
            add(
                format!("model.llm.attn_res.mlp_queries.{i}"),
                format!("blk.{i}.ffn_res_query.weight"),
                None,
                Transform::None,
            )?;
            if i > 0 {
                add(
                    format!("model.llm.attn_res.attn_queries.{}", i - 1),
                    format!("blk.{i}.attn_res_query.weight"),
                    None,
                    Transform::None,
                )?;
            }
        }
        Ok(Self { source, entries })
    }

    pub(super) fn shape(&self, name: &str) -> Result<&[usize], EngError> {
        Ok(&self.entry(name)?.shape)
    }

    pub(super) fn vector(&self, name: &str) -> Result<Vec<f32>, EngError> {
        let entry = self.entry(name)?;
        let tensor = self
            .source
            .header()
            .tensor(&entry.name)
            .expect("validated tensor");
        let values = self.source.values(tensor)?;
        match entry.transform {
            Transform::None => Ok(values),
            transform => {
                let [rows, cols] = entry.shape[..] else {
                    unreachable!("validated transpose")
                };
                let mut out = vec![0.; values.len()];
                for row in 0..rows {
                    let src = match transform {
                        Transform::ReverseTaps => rows - 1 - row,
                        _ => row,
                    };
                    for col in 0..cols {
                        out[row * cols + col] = values[col * rows + src];
                    }
                }
                Ok(out)
            }
        }
    }

    pub(super) fn row(&self, name: &str, row: usize) -> Result<Vec<f32>, EngError> {
        self.source.row(&self.entry(name)?.name, row)
    }

    pub(super) fn matvec_rows(
        &self,
        name: &str,
        input: &[f32],
        rows: Range<usize>,
    ) -> Result<Vec<f32>, EngError> {
        self.source.project(&self.entry(name)?.name, 0, rows, input)
    }

    pub(super) fn loop_query(&self, layer: usize, stream: usize) -> Result<Vec<f32>, EngError> {
        let result = match self.source.layout() {
            Layout::Mainline => self
                .source
                .row(&format!("blk.{layer}.loop_msg_query.weight"), stream),
            Layout::Fork => {
                let (name, offset) = match stream {
                    0 => ("loop_res_query.weight", 0),
                    1 => ("loop_kv_res_query.weight", 0),
                    2 => ("loop_kv_res_query.weight", 1),
                    3 => ("loop_res_query.weight", 1),
                    _ => return Err(checkpoint_error("invalid Inkling loop query stream")),
                };
                self.source.row(name, 2 * (layer - CORE_START) + offset)
            }
        }?;
        debug_assert_eq!(result.len(), HIDDEN);
        Ok(result)
    }

    fn entry(&self, name: &str) -> Result<&Entry, EngError> {
        self.entries
            .get(name)
            .ok_or_else(|| checkpoint_error(format!("missing GGUF text weight {name}")))
    }
}
