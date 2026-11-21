//! Bind the Unsloth fixed-loop GGUF layout to the Inkling text graph.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::BufReader,
    ops::Range,
    path::{Path, PathBuf},
};

use super::{LoopConfig, TextSpec, bad};
use crate::{
    EngError,
    family::mapped_file::MappedFile,
    weights::{GgufHeader, GgufType, MetaValue, dequantize_into},
};

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
use super::cuda_projection::{
    DenseUpload,
    gguf::{ExpertView, GgufCuda},
};

const HEADER_LIMIT: u64 = 32 * 1024 * 1024;

pub(crate) fn gguf_path(path: &Path) -> Result<Option<PathBuf>, EngError> {
    if path.extension().is_some_and(|ext| ext == "gguf") && !path.is_dir() {
        return Ok(Some(path.to_path_buf()));
    }
    if path.join("model.safetensors.index.json").is_file() {
        return Ok(None);
    }
    let entries =
        fs::read_dir(path).map_err(|error| bad(format!("scan model directory: {error}")))?;
    let mut text = None;
    for entry in entries {
        let entry = entry.map_err(|error| bad(format!("read model directory: {error}")))?;
        let file = entry.path();
        if !file.extension().is_some_and(|ext| ext == "gguf") {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("mmproj") || name.starts_with("dspark-") {
            continue;
        }
        if text.replace(file).is_some() {
            return Err(bad(
                "multiple Inkling GGUFs in model_dir; name the GGUF file explicitly",
            ));
        }
    }
    Ok(text)
}

pub(super) struct Metadata<'a>(&'a GgufHeader);

impl<'a> Metadata<'a> {
    pub(super) fn new(header: &'a GgufHeader) -> Self {
        Self(header)
    }

    pub(super) fn integer(&self, key: &str) -> Result<u64, EngError> {
        match self.0.meta(key) {
            Some(MetaValue::U32(value)) => Ok(u64::from(*value)),
            Some(MetaValue::U64(value)) => Ok(*value),
            _ => Err(bad(format!("GGUF is missing integer {key}"))),
        }
    }

    pub(super) fn float(&self, key: &str) -> Result<f32, EngError> {
        match self.0.meta(key) {
            Some(MetaValue::F32(value)) if value.is_finite() => Ok(*value),
            _ => Err(bad(format!("GGUF is missing finite {key}"))),
        }
    }
}

fn text_spec(header: &GgufHeader) -> Result<TextSpec, EngError> {
    if !matches!(header.meta("general.architecture"), Some(MetaValue::String(a)) if a == "inkling2")
        || header.split().is_some_and(|split| split.count > 1)
    {
        return Err(bad(
            "expected a single-file Unsloth Inkling 2 Mini loop GGUF",
        ));
    }
    let meta = Metadata::new(header);
    let int = |key| meta.integer(&format!("inkling2.{key}"));
    let core_start = int("loop.core_start")? as usize;
    let core_end = int("loop.core_end")? as usize;
    let num_passes = int("num_loops")? as usize;
    let blocks = int("block_count")? as usize;
    let repeated = core_end
        .checked_sub(core_start)
        .and_then(|width| width.checked_mul(num_passes.saturating_sub(1)))
        .and_then(|added| blocks.checked_sub(added))
        .ok_or_else(|| bad("invalid GGUF loop geometry"))?;
    if repeated != 24 || blocks != 48 {
        return Err(bad(
            "expected 24 stored layers and 48 visits in the Mini GGUF",
        ));
    }
    let u32_int =
        |key| u32::try_from(int(key)?).map_err(|_| bad(format!("GGUF {key} overflows u32")));
    let spec = TextSpec {
        model_type: "inkling2_loop_text".into(),
        model_max_length: u32_int("context_length")?,
        hidden_size: int("embedding_length")?,
        num_hidden_layers: repeated,
        vocab_size: int("vocab_size")?,
        unpadded_vocab_size: int("vocab_size_unpadded")?,
        num_attention_heads: 24,
        num_key_value_heads: 8,
        head_dim: int("attention.key_length")?,
        swa_num_attention_heads: 24,
        swa_num_key_value_heads: 8,
        swa_head_dim: int("attention.key_length_swa")?,
        sliding_window_size: u32_int("attention.sliding_window")?,
        local_layer_ids: (0..repeated)
            .filter(|layer| {
                header
                    .tensor(&format!("blk.{layer}.attn_norm.weight"))
                    .is_some()
            })
            .collect(),
        n_layers_per_attn_res_block: int("attn_res.block_size")? as usize,
        n_routed_experts: int("expert_count")?,
        num_experts_per_tok: int("expert_used_count")?,
        n_shared_experts: 0,
        intermediate_size: int("expert_feed_forward_length")?,
        core: LoopConfig {
            core_start,
            core_end,
            num_passes,
            gate_rank: int("loop.gate_rank")?,
            is_varloop: false,
        },
    };
    spec.validate()?;
    if int("rope.dimension_count")? != 0
        || int("attention.value_length")? != spec.head_dim
        || int("attention.value_length_swa")? != spec.swa_head_dim
    {
        return Err(bad("unsupported GGUF attention dimensions"));
    }
    Ok(spec)
}

#[derive(Clone, Copy)]
enum VectorOrder {
    Stored,
    Transpose,
    ConvTaps,
}

struct Tensor {
    kind: GgufType,
    bytes: Range<usize>,
    columns: usize,
    rows: usize,
    planes: usize,
    vector_order: VectorOrder,
}

type Tensors = BTreeMap<String, Tensor>;

struct Bindings<'a> {
    header: &'a GgufHeader,
    tensors: Tensors,
}

impl Bindings<'_> {
    fn bind(&mut self, native: String, gguf: &str, dims: &[u64]) -> Result<(), EngError> {
        let tensor = self
            .header
            .tensor(gguf)
            .ok_or_else(|| bad(format!("GGUF is missing {gguf}")))?;
        if tensor.dims() != dims {
            return Err(bad(format!(
                "GGUF tensor {gguf} has shape {:?}, expected {dims:?}",
                tensor.dims()
            )));
        }
        // An empty decode checks whether the backend supports this block type.
        dequantize_into(tensor.ggml_type(), &[], &mut [])?;
        let bytes = tensor.abs_range();
        self.tensors.insert(
            native,
            Tensor {
                kind: tensor.ggml_type(),
                bytes: bytes.start as usize..bytes.end as usize,
                columns: dims[0] as usize,
                rows: dims.get(1).copied().unwrap_or(1) as usize,
                planes: dims.get(2).copied().unwrap_or(1) as usize,
                vector_order: VectorOrder::Stored,
            },
        );
        Ok(())
    }

    fn reorder(&mut self, native: &str, order: VectorOrder) {
        self.tensors
            .get_mut(native)
            .expect("new binding")
            .vector_order = order;
    }

    fn globals(&mut self, spec: &TextSpec) -> Result<(), EngError> {
        let hidden = spec.hidden_size;
        let vocab = spec.vocab_size;
        let queries = 2 * (spec.core.core_end - spec.core.core_start) as u64;
        for (native, gguf, dims) in [
            ("embed.weight", "token_embd.weight", &[hidden, vocab][..]),
            ("unembed.weight", "output.weight", &[hidden, vocab]),
            ("embed_norm.weight", "token_embd_norm.weight", &[hidden]),
            ("norm.weight", "output_norm.weight", &[hidden]),
            (
                "attn_res.output_query",
                "output_res_query.weight",
                &[hidden],
            ),
            (
                "middle_cycle_cross_pass_queries",
                "loop_res_query.weight",
                &[hidden, queries],
            ),
            (
                "middle_cycle_cross_pass_kv_queries",
                "loop_kv_res_query.weight",
                &[hidden, queries],
            ),
        ] {
            self.bind(format!("model.llm.{native}"), gguf, dims)?;
        }
        Ok(())
    }

    fn attention(
        &mut self,
        spec: &TextSpec,
        layer: usize,
        local: bool,
        taps: u64,
        rel: [u64; 2],
    ) -> Result<(), EngError> {
        let hidden = spec.hidden_size;
        let (query, kv, width) = if local {
            (
                spec.swa_num_attention_heads,
                spec.swa_num_key_value_heads,
                spec.swa_head_dim,
            )
        } else {
            (
                spec.num_attention_heads,
                spec.num_key_value_heads,
                spec.head_dim,
            )
        };
        let [rel_dim, extent] = rel;
        let native = format!("model.llm.layers.{layer}");
        let gguf = format!("blk.{layer}");
        for (name, source, dims) in [
            (
                "wq_du.weight",
                "attn_q.weight",
                &[hidden, query * width][..],
            ),
            ("wk_dv.weight", "attn_k.weight", &[hidden, kv * width]),
            ("wv_dv.weight", "attn_v.weight", &[hidden, kv * width]),
            (
                "wo_ud.weight",
                "attn_output.weight",
                &[query * width, hidden],
            ),
            (
                "wr_du.weight",
                "attn_rel.weight",
                &[hidden, query * rel_dim],
            ),
            (
                "gate_attn.weight",
                "attn_gate.weight",
                &[hidden, query * width],
            ),
            (
                "attn_sink_proj.weight",
                "attn_sink_proj.weight",
                &[hidden, query],
            ),
            ("q_l2_clip.scale", "attn_q_clip.weight", &[width, query]),
            (
                "rel_l2_clip.scale",
                "attn_rel_clip.weight",
                &[rel_dim, query],
            ),
            (
                "rel_logits_proj.proj",
                "attn_rel_bias.weight",
                &[rel_dim, extent],
            ),
        ] {
            self.bind(
                format!("{native}.attn.{name}"),
                &format!("{gguf}.{source}"),
                dims,
            )?;
        }
        self.reorder(
            &format!("{native}.attn.rel_logits_proj.proj"),
            VectorOrder::Transpose,
        );
        for plane in ["k", "v"] {
            let conv = format!("{native}.attn.{plane}_sconv");
            let name = if local {
                format!("{conv}.weight")
            } else {
                format!("{conv}.static_w")
            };
            self.bind(
                name.clone(),
                &format!("{gguf}.attn_{plane}_conv.weight"),
                &[taps, kv * width],
            )?;
            if !local {
                self.reorder(&name, VectorOrder::ConvTaps);
                self.bind(
                    format!("{conv}.weight_proj"),
                    &format!("{gguf}.attn_{plane}_conv_proj.weight"),
                    &[hidden, taps * kv],
                )?;
            }
        }
        if local {
            self.bind(
                format!("{native}.attn_norm.weight"),
                &format!("{gguf}.attn_norm.weight"),
                &[hidden],
            )?;
        } else {
            for plane in ["q", "k", "v"] {
                self.bind(
                    format!("{native}.{plane}_input_norm.weight"),
                    &format!("{gguf}.attn_{plane}_inp_norm.weight"),
                    &[hidden],
                )?;
            }
        }
        if layer > 0 {
            let dims = if local { vec![hidden] } else { vec![hidden, 3] };
            self.bind(
                format!("model.llm.attn_res.attn_queries.{}", layer - 1),
                &format!("{gguf}.attn_res_query.weight"),
                &dims,
            )?;
        }
        Ok(())
    }

    fn feed_forward(&mut self, spec: &TextSpec, layer: usize, taps: u64) -> Result<(), EngError> {
        let hidden = spec.hidden_size;
        let experts = spec.n_routed_experts;
        let intermediate = spec.intermediate_size;
        let native = format!("model.llm.layers.{layer}");
        let gguf = format!("blk.{layer}");
        for (name, source, dims) in [
            (
                "mlp.gate.weight",
                "ffn_gate_inp.weight",
                &[hidden, experts][..],
            ),
            ("mlp.gate.bias", "exp_probs_b.bias", &[experts]),
            ("mlp_norm.weight", "ffn_norm.weight", &[hidden]),
            (
                "attn_sconv.weight",
                "attn_post_conv.weight",
                &[taps, hidden],
            ),
            ("mlp_sconv.weight", "ffn_post_conv.weight", &[taps, hidden]),
            (
                "mlp.experts.w13_weight",
                "ffn_gate_up_exps.weight",
                &[hidden, 2 * intermediate, experts],
            ),
            (
                "mlp.experts.w2_weight",
                "ffn_down_exps.weight",
                &[intermediate, hidden, experts],
            ),
        ] {
            self.bind(
                format!("{native}.{name}"),
                &format!("{gguf}.{source}"),
                dims,
            )?;
        }
        for (name, stack) in [
            ("w13_weight", "ffn_gate_up_exps"),
            ("w2_weight", "ffn_down_exps"),
        ] {
            let scale = format!("{gguf}.{stack}.scale");
            if let Some(tensor) = self.header.tensor(&scale) {
                if tensor.ggml_type() != GgufType::F32 {
                    return Err(bad(format!("GGUF expert scale {scale} must be F32")));
                }
                self.bind(
                    format!("{native}.mlp.experts.{name}_scale"),
                    &scale,
                    &[experts],
                )?;
            }
        }
        self.bind(
            format!("model.llm.attn_res.mlp_queries.{layer}"),
            &format!("{gguf}.ffn_res_query.weight"),
            &[hidden],
        )?;
        Ok(())
    }

    fn controllers(&mut self, spec: &TextSpec, layer: usize) -> Result<(), EngError> {
        let hidden = spec.hidden_size;
        let rank = spec.core.gate_rank;
        for role in ["attn", "k", "v", "mlp"] {
            for (part, dims) in [
                ("down", [2 * hidden, rank]),
                ("gain", [rank, hidden]),
                ("forget", [rank, hidden]),
            ] {
                self.bind(
                    format!("model.llm.layers.{layer}.controllers.{role}.{part}_weight"),
                    &format!("blk.{layer}.loop_{role}_{part}.weight"),
                    &dims,
                )?;
            }
        }
        Ok(())
    }
}

pub(crate) struct GgufPlan {
    pub(crate) header: GgufHeader,
    pub(crate) spec: TextSpec,
    tensors: Tensors,
}

impl GgufPlan {
    pub(crate) fn open(path: &Path) -> Result<Self, EngError> {
        crate::storage_guard::require_local(path, "model weights")?;
        let file = File::open(path).map_err(|error| bad(format!("open GGUF: {error}")))?;
        let len = file
            .metadata()
            .map_err(|error| bad(format!("stat GGUF: {error}")))?
            .len();
        let header = GgufHeader::parse(&mut BufReader::new(file), len, HEADER_LIMIT)?;
        let spec = text_spec(&header)?;
        let meta = Metadata::new(&header);
        let taps = meta.integer("inkling2.shortconv.l_cache")?;
        let rel = [
            meta.integer("inkling2.attention.relative_length")?,
            meta.integer("inkling2.attention.relative_extent")?,
        ];
        // These dimensions bound all shape products in the text binder.
        if spec.hidden_size != 2304
            || spec.intermediate_size != 1152
            || spec.core.gate_rank != 64
            || spec.head_dim != 128
            || spec.swa_head_dim != 128
            || spec.n_routed_experts != 128
            || spec.vocab_size != 201_024
            || spec.unpadded_vocab_size != 200_058
            || taps != 4
            || rel != [16, 1024]
        {
            return Err(bad(
                "GGUF geometry differs from the Inkling 2 Mini text backbone",
            ));
        }
        let mut bind = Bindings {
            header: &header,
            tensors: BTreeMap::new(),
        };
        bind.globals(&spec)?;
        for layer in 0..spec.stored_layers() {
            let local = spec.local_layer_ids.contains(&layer);
            bind.attention(&spec, layer, local, taps, rel)?;
            bind.feed_forward(&spec, layer, taps)?;
            if (spec.core.core_start..spec.core.core_end).contains(&layer) {
                bind.controllers(&spec, layer)?;
            }
        }
        let tensors = bind.tensors;
        Ok(Self {
            header,
            spec,
            tensors,
        })
    }

    pub(super) fn map(self, path: &Path) -> Result<(TextSpec, GgufTextWeights), EngError> {
        let mapping = MappedFile::open(path, self.header.file_len())
            .map_err(|error| bad(format!("map GGUF: {error}")))?;
        Ok((
            self.spec,
            GgufTextWeights {
                mapping,
                tensors: self.tensors,
                #[cfg(all(feature = "native-cuda", target_os = "linux"))]
                cuda: None,
            },
        ))
    }
}

pub(super) struct GgufTextWeights {
    mapping: MappedFile,
    tensors: Tensors,
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    cuda: Option<GgufCuda>,
}

impl GgufTextWeights {
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    pub(super) fn enable_cuda(&mut self, spec: &TextSpec) -> Result<(), EngError> {
        let mut uploads = Vec::new();
        for (name, tensor) in &self.tensors {
            if tensor.kind != GgufType::Bf16
                || tensor.planes != 1
                || tensor.rows == 1
                || name == "model.llm.embed.weight"
                || !matches!(tensor.vector_order, VectorOrder::Stored)
            {
                continue;
            }
            let rows = if name == "model.llm.unembed.weight" {
                spec.unpadded_vocab_size as usize
            } else {
                tensor.rows
            };
            let (_, payload) = self.tensor(name)?;
            uploads.push(DenseUpload {
                name,
                codes: &payload[..rows * tensor.columns * 2],
                shape: [rows, tensor.columns],
            });
        }
        self.cuda = Some(GgufCuda::load(uploads, spec)?);
        Ok(())
    }

    fn tensor(&self, name: &str) -> Result<(&Tensor, &[u8]), EngError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("no GGUF binding for {name}")))?;
        let bytes = self
            .mapping
            .as_ref()
            .get(tensor.bytes.clone())
            .ok_or_else(|| bad(format!("GGUF tensor {name} escapes the mapping")))?;
        Ok((tensor, bytes))
    }

    pub(super) fn vec(&self, name: &str, len: usize) -> Result<Vec<f32>, EngError> {
        let (tensor, bytes) = self.tensor(name)?;
        if tensor.columns * tensor.rows != len || tensor.planes != 1 {
            return Err(bad(format!("GGUF vector {name} has the wrong length")));
        }
        let mut values = vec![0.0; len];
        dequantize_into(tensor.kind, bytes, &mut values)?;
        let (cols, rows) = (tensor.columns, tensor.rows);
        let transpose = |flip| {
            let values = &values;
            (0..cols)
                .flat_map(move |col| {
                    let col = if flip { cols - 1 - col } else { col };
                    (0..rows).map(move |row| values[row * cols + col])
                })
                .collect()
        };
        match tensor.vector_order {
            VectorOrder::Stored => Ok(values),
            VectorOrder::Transpose => Ok(transpose(false)),
            VectorOrder::ConvTaps => Ok(transpose(true)),
        }
    }

    pub(super) fn row(&self, name: &str, row: usize, cols: usize) -> Result<Vec<f32>, EngError> {
        let (tensor, bytes) = self.tensor(name)?;
        if tensor.columns != cols || row >= tensor.rows || tensor.planes != 1 {
            return Err(bad(format!("GGUF row {row} escapes {name}")));
        }
        let stride = bytes.len() / tensor.rows;
        let mut values = vec![0.0; cols];
        dequantize_into(tensor.kind, &bytes[row * stride..][..stride], &mut values)?;
        Ok(values)
    }

    pub(super) fn project(&self, name: &str, x: &[f32], rows: usize) -> Result<Vec<f32>, EngError> {
        let (tensor, bytes) = self.tensor(name)?;
        if tensor.columns != x.len() || rows > tensor.rows || tensor.planes != 1 {
            return Err(bad(format!("bad GGUF projection shape for {name}")));
        }
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = &self.cuda
            && let Some(output) = cuda.project(name, rows, x)
        {
            return output;
        }
        project_rows(tensor, bytes, x, rows)
    }

    pub(super) fn expert(&self, name: &str, id: usize, x: &[f32]) -> Result<Vec<f32>, EngError> {
        let (tensor, bytes) = self.tensor(name)?;
        if id >= tensor.planes || tensor.columns != x.len() {
            return Err(bad(format!("bad GGUF expert {id} shape for {name}")));
        }
        let plane = bytes.len() / tensor.planes;
        let payload = &bytes[id * plane..][..plane];
        let factor = self.expert_factor(name, id, tensor.planes)?;
        let mut output = self.expert_project(tensor, payload, factor, x)?;
        if name.ends_with(".w13_weight") {
            let mid = output.len() / 2;
            output = (0..mid)
                .flat_map(|row| [output[row], output[row + mid]])
                .collect();
        }
        Ok(output)
    }
    fn expert_factor(&self, name: &str, id: usize, experts: usize) -> Result<f32, EngError> {
        let scale = format!("{name}_scale");
        if !self.tensors.contains_key(&scale) {
            return Ok(1.0);
        }
        let factor = self.row(&scale, 0, experts)?[id];
        if !factor.is_finite() || factor < 0.0 {
            return Err(bad(format!("invalid GGUF expert {id} scale for {name}")));
        }
        Ok(factor)
    }

    fn expert_project(
        &self,
        tensor: &Tensor,
        bytes: &[u8],
        factor: f32,
        x: &[f32],
    ) -> Result<Vec<f32>, EngError> {
        #[cfg(all(feature = "native-cuda", target_os = "linux"))]
        if let Some(cuda) = &self.cuda {
            let view = ExpertView {
                kind: tensor.kind,
                shape: [tensor.rows, tensor.columns],
                bytes,
                factor,
            };
            if let Some(output) = cuda.expert(&view, x) {
                return output;
            }
        }
        let mut output = project_rows(tensor, bytes, x, tensor.rows)?;
        if factor != 1.0 {
            for value in &mut output {
                *value *= factor;
            }
        }
        Ok(output)
    }
}

fn project_rows(
    tensor: &Tensor,
    bytes: &[u8],
    x: &[f32],
    rows: usize,
) -> Result<Vec<f32>, EngError> {
    let stride = bytes.len() / tensor.rows;
    let mut w = vec![0.0; x.len()];
    bytes
        .chunks_exact(stride)
        .take(rows)
        .map(|row| {
            dequantize_into(tensor.kind, row, &mut w)?;
            Ok(super::cpu_forward::dot(&w, x))
        })
        .collect()
}
