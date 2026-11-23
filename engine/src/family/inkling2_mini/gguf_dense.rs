//! Adapt GGUF text tensors to the shared Inkling decoder's weight names and strides.

use std::{collections::BTreeMap, ops::Range, sync::Arc};

use super::{
    CONV_TAPS, CORE_END, CORE_START, HIDDEN, LAYERS, checkpoint_error, gguf::Weights,
    gguf_metadata::Layout, is_local,
};
use crate::{EngError, family::mapped_file::MappedFile};

pub(super) struct Dense {
    source: Arc<Weights>,
    entries: BTreeMap<String, Entry>,
    mapping: Option<MappedFile>,
    attention_workers: usize,
}

struct Entry {
    name: String,
    shape: Vec<usize>,
    transform: Transform,
    mapped_len: u64,
}

#[derive(Clone, Copy, Default)]
enum Transform {
    #[default]
    None,
    Transpose,
    ReverseTaps,
}

impl Dense {
    pub(super) fn new(
        source: Arc<Weights>,
        mapping_budget: u64,
        attention_workers: usize,
    ) -> Result<(Self, u64), EngError> {
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
                    mapped_len: 0,
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
        let mapped_bytes = if mapping_budget == 0 {
            0
        } else {
            reserve_projections(&source, &mut entries, mapping_budget)?
        };
        let mapping = (mapped_bytes > 0)
            .then(|| source.map_projection_source())
            .transpose()?;
        Ok((
            Self {
                source,
                entries,
                mapping,
                attention_workers,
            },
            mapped_bytes,
        ))
    }

    #[cfg(all(
        test,
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    pub(super) fn has_mapping(&self) -> bool {
        self.mapping.is_some()
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn io_snapshot(&self) -> super::timing::DenseIoStats {
        self.source.dense_io_snapshot()
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
        let values = self.source.values(tensor, true)?;
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
        let entry = self.entry(name)?;
        let workers = attention_workers_for(&entry.name, self.attention_workers);
        if let Some(mapping) = self.mapping.as_ref().filter(|_| entry.mapped_len > 0) {
            return self.source.project_mapped(
                &entry.name,
                rows,
                input,
                mapping,
                entry.mapped_len,
                workers,
            );
        }
        self.source
            .project_with_workers(&entry.name, 0, rows, input, true, workers)
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

/// Charge each selected packed tensor's complete page span. The accounting
/// intentionally permits overlap between adjacent tensor spans.
fn reserve_projections(
    source: &Weights,
    entries: &mut BTreeMap<String, Entry>,
    budget: u64,
) -> Result<u64, EngError> {
    let page_size = super::mac_stage_page_size() as u64;
    if page_size == 0 {
        return Err(checkpoint_error("GGUF dense mapping page size unavailable"));
    }
    let mut candidates: Vec<_> = entries
        .values_mut()
        .filter_map(|entry| projection_uses(&entry.name).map(|uses| (uses, entry)))
        .collect();
    candidates.sort_unstable_by(|(a, left), (b, right)| b.cmp(a).then(left.name.cmp(&right.name)));
    let mut remaining = budget;
    for (_, entry) in candidates {
        let tensor = source
            .header()
            .tensor(&entry.name)
            .expect("validated tensor");
        let packed_bytes = if entry.name == "output.weight" {
            let padded_rows = tensor.dims()[1];
            tensor.byte_len() / padded_rows * super::UNPADDED_VOCAB as u64
        } else {
            tensor.byte_len()
        };
        let span = page_span(tensor.abs_offset(), packed_bytes, page_size)?;
        if span <= remaining {
            remaining -= span;
            entry.mapped_len = packed_bytes;
        }
    }
    Ok(budget - remaining)
}

fn page_span(offset: u64, bytes: u64, page: u64) -> Result<u64, EngError> {
    (offset % page)
        .checked_add(bytes)
        .and_then(|len| len.checked_next_multiple_of(page))
        .ok_or_else(super::staging_overflow)
}

fn attention_workers_for(name: &str, configured: usize) -> usize {
    if configured > 1
        && name
            .strip_prefix("blk.")
            .and_then(|name| name.split_once('.'))
            .is_some_and(|(_, projection)| {
                matches!(
                    projection,
                    "attn_q.weight" | "attn_k.weight" | "attn_v.weight" | "attn_output.weight"
                )
            })
    {
        configured
    } else {
        1
    }
}

fn projection_uses(name: &str) -> Option<usize> {
    if name == "output.weight" {
        return Some(1);
    }
    let (layer, suffix) = name.strip_prefix("blk.")?.split_once('.')?;
    let layer = layer.parse::<usize>().ok()?;
    let projection = matches!(
        suffix,
        "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_output.weight"
            | "attn_rel.weight"
            | "attn_gate.weight"
            | "attn_sink_proj.weight"
            | "attn_k_conv_proj.weight"
            | "attn_v_conv_proj.weight"
            | "ffn_gate_inp.weight"
    );
    let looped = (CORE_START..CORE_END).contains(&layer);
    if projection {
        return Some(if looped { 3 } else { 1 });
    }
    (looped && suffix.starts_with("loop_") && suffix.ends_with(".weight")).then_some(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_gguf_attention_matrices_use_the_requested_workers() {
        for name in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ] {
            let name = format!("blk.6.{name}");
            assert_eq!(attention_workers_for(&name, 4), 4);
            assert_eq!(attention_workers_for(&name, 1), 1);
        }
        for name in [
            "output.weight",
            "blk.6.attn_gate.weight",
            "blk.6.ffn_gate_inp.weight",
            "blk.6.loop_attn_down.weight",
        ] {
            assert_eq!(attention_workers_for(name, 4), 1);
        }
    }

    #[test]
    fn dense_mapping_selects_only_recurring_projections_and_charges_page_spans() {
        assert_eq!(projection_uses("blk.6.attn_q.weight"), Some(3));
        assert_eq!(projection_uses("blk.6.loop_attn_down.weight"), Some(2));
        assert_eq!(projection_uses("blk.20.ffn_gate_inp.weight"), Some(1));
        assert_eq!(projection_uses("output.weight"), Some(1));
        assert_eq!(projection_uses("token_embd.weight"), None);
        assert_eq!(projection_uses("blk.6.ffn_gate_up_exps.weight"), None);
        assert_eq!(projection_uses("blk.6.attn_rel_bias.weight"), None);
        assert_eq!(page_span(16380, 8, 16384).unwrap(), 32768);
        assert_eq!(page_span(16384, 16384, 16384).unwrap(), 16384);
    }
}
