//! Tensor validation and bounded reads for the two fixed-loop GGUF formats.

use std::{fs::File, io::BufReader, ops::Range, path::Path, time::SystemTime};

use super::timing::DenseIo;
use super::{
    CONV_TAPS, CORE_END, CORE_START, EXPERTS, HEAD_DIM, HEADER_LIMIT, HIDDEN, INTERMEDIATE,
    KV_HEADS, LAYERS, QUERY_HEADS, checkpoint_error,
    gguf_metadata::{self, Layout},
    is_local, read_exact_at,
};
use crate::{
    EngError,
    family::mapped_file::MappedFile,
    weights::{GgufHeader, GgufTensor, GgufType, MetaValue},
};

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) const DENSE_MAPPING_ENV: &str = "FIFI_INKLING_GGUF_MMAP_DENSE";
const ROWS_PER_READ: usize = 256;
const ATTENTION_WORKER_STACK_BYTES: usize = 256 << 10;
const MAX_ATTENTION_PROJECTION_DIM: usize = QUERY_HEADS * HEAD_DIM as usize;
pub(super) const LARGEST_PLANE_F32_BYTES: usize = 2 * HIDDEN * INTERMEDIATE * 4;
const PADDED_VOCAB: u64 = 201_024;
const RELATIVE_FEATURES: u64 = 16;
const RELATIVE_EXTENT: u64 = 1024;

#[derive(Clone, Copy)]
enum Kind {
    F32,
    Dense,
    Expert,
}

impl Kind {
    fn accepts(self, dtype: GgufType) -> bool {
        use GgufType::*;
        match self {
            Self::F32 => dtype == F32,
            Self::Dense => matches!(
                dtype,
                F32 | F16 | Bf16 | Q80 | Q4K | Q5K | Q6K | Iq3Xxs | Iq4Xs
            ),
            Self::Expert => matches!(
                dtype,
                F32 | F16
                    | Bf16
                    | Q80
                    | Q4K
                    | Q5K
                    | Q6K
                    | Nvfp4
                    | Q20
                    | Iq1S
                    | Iq2Xxs
                    | Iq2Xs
                    | Iq2S
                    | Iq3Xxs
                    | Iq3S
                    | Iq4Xs
            ),
        }
    }
}

pub(crate) fn validate_tensors(header: &GgufHeader) -> Result<(), EngError> {
    let layout = Layout::from_header(header)
        .ok_or_else(|| checkpoint_error("unsupported Inkling GGUF architecture"))?;
    let hidden = HIDDEN as u64;
    let query = QUERY_HEADS as u64;
    let q_width = query * HEAD_DIM;
    let kv_width = KV_HEADS * HEAD_DIM;
    let experts = EXPERTS as u64;
    let intermediate = INTERMEDIATE as u64;

    let expect = |name: &str, shape: &[u64], kind: Kind| -> Result<GgufType, EngError> {
        let tensor = header
            .tensor(name)
            .ok_or_else(|| checkpoint_error(format!("missing GGUF tensor {name}")))?;
        let dtype = tensor.ggml_type();
        if tensor.dims() != shape || !kind.accepts(dtype) {
            return Err(checkpoint_error(format!(
                "GGUF tensor {name} has GGML dimensions {:?} and type {}, expected dimensions {shape:?} and a compatible type",
                tensor.dims(),
                dtype.name(),
            )));
        }
        Ok(dtype)
    };

    for name in ["token_embd.weight", "output.weight"] {
        expect(name, &[hidden, PADDED_VOCAB], Kind::Dense)?;
    }
    for name in [
        "token_embd_norm.weight",
        "output_norm.weight",
        "output_res_query.weight",
    ] {
        expect(name, &[hidden], Kind::F32)?;
    }
    if layout == Layout::Fork {
        for name in ["loop_res_query.weight", "loop_kv_res_query.weight"] {
            expect(
                name,
                &[hidden, (2 * (CORE_END - CORE_START)) as u64],
                Kind::F32,
            )?;
        }
    }

    for i in 0..LAYERS {
        let layer = |suffix: &str, shape: &[u64], kind: Kind| {
            expect(&format!("blk.{i}.{suffix}"), shape, kind)
        };
        for (name, shape) in [
            ("attn_q.weight", [hidden, q_width]),
            ("attn_k.weight", [hidden, kv_width]),
            ("attn_v.weight", [hidden, kv_width]),
            ("attn_gate.weight", [hidden, q_width]),
            ("attn_output.weight", [q_width, hidden]),
            ("attn_rel.weight", [hidden, RELATIVE_FEATURES * query]),
            ("attn_sink_proj.weight", [hidden, query]),
        ] {
            layer(name, &shape, Kind::Dense)?;
        }
        for (name, shape) in [
            ("attn_q_clip.weight", [HEAD_DIM, query]),
            ("attn_rel_bias.weight", [RELATIVE_FEATURES, RELATIVE_EXTENT]),
            ("attn_rel_clip.weight", [RELATIVE_FEATURES, query]),
            ("attn_k_conv.weight", [CONV_TAPS, kv_width]),
            ("attn_v_conv.weight", [CONV_TAPS, kv_width]),
            ("attn_post_conv.weight", [CONV_TAPS, hidden]),
            ("ffn_post_conv.weight", [CONV_TAPS, hidden]),
            ("ffn_gate_inp.weight", [hidden, experts]),
        ] {
            layer(name, &shape, Kind::F32)?;
        }
        for (name, width) in [
            ("exp_probs_b.bias", experts),
            ("ffn_norm.weight", hidden),
            ("ffn_res_query.weight", hidden),
        ] {
            layer(name, &[width], Kind::F32)?;
        }
        if is_local(i) {
            layer("attn_norm.weight", &[hidden], Kind::F32)?;
            if i != 0 {
                layer("attn_res_query.weight", &[hidden], Kind::F32)?;
            }
        } else {
            for stream in ["q", "k", "v"] {
                layer(
                    &format!("attn_{stream}_inp_norm.weight"),
                    &[hidden],
                    Kind::F32,
                )?;
            }
            for stream in ["k", "v"] {
                layer(
                    &format!("attn_{stream}_conv_proj.weight"),
                    &[hidden, CONV_TAPS * KV_HEADS],
                    Kind::Dense,
                )?;
            }
            layer("attn_res_query.weight", &[hidden, 3], Kind::F32)?;
        }
        for (prefix, width, height) in [
            ("ffn_gate_up_exps", hidden, 2 * intermediate),
            ("ffn_down_exps", intermediate, hidden),
        ] {
            let dtype = layer(
                &format!("{prefix}.weight"),
                &[width, height, experts],
                Kind::Expert,
            )?;
            let scale = format!("{prefix}.scale");
            if dtype == GgufType::Nvfp4 || header.tensor(&format!("blk.{i}.{scale}")).is_some() {
                layer(&scale, &[experts], Kind::F32)?;
            }
        }
        if (CORE_START..CORE_END).contains(&i) {
            let streams = match layout {
                Layout::Mainline => ["q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"],
                Layout::Fork => ["attn", "k", "v", "mlp"],
            };
            for stream in streams {
                layer(
                    &format!("loop_{stream}_down.weight"),
                    &[2 * hidden, 64],
                    Kind::F32,
                )?;
                for suffix in ["gain", "forget"] {
                    layer(
                        &format!("loop_{stream}_{suffix}.weight"),
                        &[64, hidden],
                        Kind::F32,
                    )?;
                }
            }
            if layout == Layout::Mainline {
                layer("loop_msg_query.weight", &[hidden, 4], Kind::F32)?;
            }
        }
    }
    // The shared text graph fixes these inference constants. Reject a header
    // that advertises different inference semantics for the same tensor shapes.
    let arch = layout.architecture();
    for (key, expected) in [
        ("vocab_size", MetaValue::U32(PADDED_VOCAB as u32)),
        (
            "vocab_size_unpadded",
            MetaValue::U32(super::UNPADDED_VOCAB as u32),
        ),
        (
            "attention.relative_length",
            MetaValue::U32(RELATIVE_FEATURES as u32),
        ),
        (
            "attention.relative_extent",
            MetaValue::U32(RELATIVE_EXTENT as u32),
        ),
        ("attention.layer_norm_rms_epsilon", MetaValue::F32(1e-6)),
        (
            "attention.qk_clip",
            MetaValue::F32(2.0 * (HEAD_DIM as f32).sqrt()),
        ),
        ("attention.relative_clip", MetaValue::F32(8.0)),
        ("attention.log_scale_floor", MetaValue::U32(8192)),
        ("attention.log_scale_alpha", MetaValue::F32(0.1)),
        ("shortconv.l_cache", MetaValue::U32(CONV_TAPS as u32)),
        ("expert_weights_scale", MetaValue::F32(super::ROUTE_SCALE)),
        ("activation.situ_beta", MetaValue::F32(super::GATE_CAP)),
        ("activation.situ_linear_beta", MetaValue::F32(super::UP_CAP)),
        ("attn_res.block_size", MetaValue::U32(6)),
        ("final_logit_softcapping", MetaValue::F32(30.0)),
        ("rope.dimension_count", MetaValue::U32(0)),
    ] {
        let key = format!("{arch}.{key}");
        if header.meta(&key) != Some(&expected) {
            return Err(checkpoint_error(format!(
                "GGUF metadata {key} must be {expected:?}"
            )));
        }
    }
    Ok(())
}

/// A validated single-file checkpoint. All reads are positioned and confined to one tensor.
pub struct Weights {
    file: File,
    header: GgufHeader,
    modified: Option<SystemTime>,
    dense_io: Option<DenseIo>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self, EngError> {
        crate::storage_guard::require_local(path, "model weights")?;
        let file = File::open(path).map_err(checkpoint_error)?;
        let mut reader = BufReader::new(&file);
        let metadata = file.metadata().map_err(checkpoint_error)?;
        let header = GgufHeader::parse(&mut reader, metadata.len(), HEADER_LIMIT)?;
        gguf_metadata::inspect(&header, &mut reader)?;
        validate_tensors(&header)?;
        Ok(Self {
            file,
            header,
            modified: metadata.modified().ok(),
            dense_io: DenseIo::start_if_enabled(),
        })
    }

    pub(super) fn map_projection_source(&self) -> Result<MappedFile, EngError> {
        let modified = self
            .modified
            .ok_or_else(|| checkpoint_error("GGUF modification time unavailable"))?;
        let file = self.file.try_clone().map_err(checkpoint_error)?;
        MappedFile::from_file(file, self.header.file_len(), modified).map_err(checkpoint_error)
    }

    pub fn header(&self) -> &GgufHeader {
        &self.header
    }

    pub fn read_range(&self, name: &str, offset: u64, dst: &mut [u8]) -> Result<(), EngError> {
        self.read_range_classified(name, offset, dst, false)
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn dense_io_snapshot(&self) -> super::timing::DenseIoStats {
        self.dense_io
            .as_ref()
            .map(DenseIo::snapshot)
            .unwrap_or_default()
    }

    fn read_range_classified(
        &self,
        name: &str,
        offset: u64,
        dst: &mut [u8],
        dense: bool,
    ) -> Result<(), EngError> {
        let tensor = self.tensor(name)?;
        if offset
            .checked_add(dst.len() as u64)
            .is_none_or(|end| end > tensor.byte_len())
        {
            return Err(EngError::BadRequest(format!(
                "range exceeds Inkling GGUF tensor {name}"
            )));
        }
        self.check_len()?;
        read_exact_at(&self.file, dst, tensor.abs_offset() + offset).map_err(checkpoint_error)?;
        if dense && let Some(io) = &self.dense_io {
            io.file_read(dst.len());
        }
        self.check_len()
    }

    fn tensor(&self, name: &str) -> Result<&GgufTensor, EngError> {
        self.header
            .tensor(name)
            .ok_or_else(|| checkpoint_error(format!("missing GGUF tensor {name}")))
    }

    fn check_len(&self) -> Result<(), EngError> {
        if self.file.metadata().map_err(checkpoint_error)?.len() != self.header.file_len() {
            return Err(checkpoint_error(
                "GGUF file length changed after validation",
            ));
        }
        Ok(())
    }
}

impl Weights {
    pub(super) fn layout(&self) -> Layout {
        Layout::from_header(&self.header).expect("validated Inkling GGUF layout")
    }

    pub(super) fn values(&self, tensor: &GgufTensor, dense: bool) -> Result<Vec<f32>, EngError> {
        let mut bytes = self.buffer(tensor.name(), tensor.byte_len() as usize)?;
        self.read_range_classified(tensor.name(), 0, &mut bytes, dense)?;
        crate::weights::dequantize(tensor.ggml_type(), &bytes)
    }

    pub(super) fn row(&self, name: &str, row: usize) -> Result<Vec<f32>, EngError> {
        let tensor = self.tensor(name)?;
        let (width, height, dtype, row_bytes) = matrix(tensor)?;
        if tensor.dims().len() != 2 || row >= height {
            return Err(EngError::BadRequest(format!(
                "row {row} exceeds Inkling GGUF tensor {name}"
            )));
        }
        let mut bytes = self.buffer(name, row_bytes)?;
        self.read_range_classified(name, (row * row_bytes) as u64, &mut bytes, true)?;
        let mut values = vec![0.; width];
        crate::weights::dequantize_into(dtype, &bytes, &mut values)?;
        Ok(values)
    }

    pub(super) fn project(
        &self,
        name: &str,
        plane: usize,
        rows: std::ops::Range<usize>,
        input: &[f32],
        dense: bool,
    ) -> Result<Vec<f32>, EngError> {
        self.project_with_workers(name, plane, rows, input, dense, 1)
    }

    pub(super) fn project_with_workers(
        &self,
        name: &str,
        plane: usize,
        rows: Range<usize>,
        input: &[f32],
        dense: bool,
        workers: usize,
    ) -> Result<Vec<f32>, EngError> {
        let tensor = self.tensor(name)?;
        let (width, height, dtype, row_bytes) = matrix(tensor)?;
        let planes = tensor.dims().get(2).copied().unwrap_or(1) as usize;
        if plane >= planes || rows.start > rows.end || rows.end > height || input.len() != width {
            return Err(EngError::BadRequest(format!(
                "projection exceeds Inkling GGUF tensor {name}"
            )));
        }
        project_row_groups(rows, workers, |rows| {
            let mut bytes = self.buffer(name, rows.len().min(ROWS_PER_READ) * row_bytes)?;
            let mut result = Vec::with_capacity(rows.len());
            let mut values = vec![0.; width];
            for start in (rows.start..rows.end).step_by(ROWS_PER_READ) {
                let count = ROWS_PER_READ.min(rows.end - start);
                let bytes = &mut bytes[..count * row_bytes];
                self.read_range_classified(
                    name,
                    ((plane * height + start) * row_bytes) as u64,
                    bytes,
                    dense,
                )?;
                for row in bytes.chunks_exact(row_bytes) {
                    crate::weights::dequantize_into(dtype, row, &mut values)?;
                    result.push(values.iter().zip(input).map(|(w, x)| w * x).sum());
                }
            }
            Ok(result)
        })
    }

    pub(super) fn project_mapped(
        &self,
        name: &str,
        rows: Range<usize>,
        input: &[f32],
        mapping: &MappedFile,
        admitted_bytes: u64,
        workers: usize,
    ) -> Result<Vec<f32>, EngError> {
        let tensor = self.tensor(name)?;
        let (width, height, dtype, row_bytes) = matrix(tensor)?;
        if tensor.dims().len() != 2
            || rows.start > rows.end
            || rows.end > height
            || input.len() != width
        {
            return Err(EngError::BadRequest(format!(
                "projection exceeds Inkling GGUF tensor {name}"
            )));
        }
        let offset = (rows.start * row_bytes) as u64;
        let end = (rows.end * row_bytes) as u64;
        if end > admitted_bytes || end > tensor.byte_len() {
            return Err(EngError::BadRequest(format!(
                "projection exceeds mapped Inkling GGUF tensor {name}"
            )));
        }
        mapping.check_unchanged().map_err(checkpoint_error)?;
        let start = (tensor.abs_offset() + offset) as usize;
        let stop = (tensor.abs_offset() + end) as usize;
        let bytes = mapping.as_ref().get(start..stop).ok_or_else(|| {
            checkpoint_error(format!("mapped GGUF tensor {name} is out of bounds"))
        })?;
        let result = project_row_groups(0..rows.len(), workers, |range| {
            project_packed_rows(
                dtype,
                row_bytes,
                &bytes[range.start * row_bytes..range.end * row_bytes],
                input,
            )
        })?;
        if let Some(io) = &self.dense_io {
            io.mapped_access(bytes.len());
        }
        mapping.check_unchanged().map_err(checkpoint_error)?;
        Ok(result)
    }

    fn buffer(&self, name: &str, len: usize) -> Result<Vec<u8>, EngError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| {
            EngError::MemoryPlan(format!("Inkling GGUF tensor {name} needs {len} bytes"))
        })?;
        bytes.resize(len, 0);
        Ok(bytes)
    }

    /// Portable projection for selected experts. Only one expert plane is streamed at a time.
    pub fn moe_row(
        &self,
        layer: usize,
        input: &[f32],
        selected: &[usize],
        routes: &[f32],
    ) -> Result<Vec<half::bf16>, EngError> {
        self.moe_row_with(
            layer,
            input,
            selected,
            routes,
            |name, plane, rows, input| self.project(name, plane, 0..rows, input, false),
        )
    }

    pub(super) fn moe_row_with(
        &self,
        layer: usize,
        input: &[f32],
        selected: &[usize],
        routes: &[f32],
        mut project: impl FnMut(&str, usize, usize, &[f32]) -> Result<Vec<f32>, EngError>,
    ) -> Result<Vec<half::bf16>, EngError> {
        let plan = self.expert_visit(layer, selected, routes)?;
        let mut merged = vec![0.; HIDDEN];
        for (&expert, &route) in selected.iter().zip(routes) {
            let fused = project(&plan.gate, expert, 2 * INTERMEDIATE, input)?;
            let activated = plan.activate(expert, &fused);
            let projected = project(&plan.down, expert, HIDDEN, &activated)?;
            plan.accumulate(&mut merged, expert, route, &projected);
        }
        finish_expert_visit(merged)
    }

    pub(super) fn expert_visit(
        &self,
        layer: usize,
        selected: &[usize],
        routes: &[f32],
    ) -> Result<ExpertVisit, EngError> {
        if layer >= LAYERS
            || selected.len() != super::TOP_K
            || routes.len() != super::TOP_K
            || selected.iter().any(|&expert| expert >= EXPERTS)
        {
            return Err(EngError::BadRequest(
                "bad Inkling GGUF expert selection".into(),
            ));
        }
        let gate = format!("blk.{layer}.ffn_gate_up_exps");
        let down = format!("blk.{layer}.ffn_down_exps");
        let scale = |prefix: &str| -> Result<Vec<f32>, EngError> {
            self.header.tensor(&format!("{prefix}.scale")).map_or_else(
                || Ok(vec![1.; EXPERTS]),
                |tensor| self.values(tensor, false),
            )
        };
        let gate_scale = scale(&gate)?;
        let down_scale = scale(&down)?;
        Ok(ExpertVisit {
            gate: format!("{gate}.weight"),
            down: format!("{down}.weight"),
            gate_scale,
            down_scale,
        })
    }
}

pub(super) struct ExpertVisit {
    pub(super) gate: String,
    pub(super) down: String,
    gate_scale: Vec<f32>,
    down_scale: Vec<f32>,
}

impl ExpertVisit {
    pub(super) fn activate(&self, expert: usize, fused: &[f32]) -> Vec<f32> {
        use super::{GATE_CAP, UP_CAP};
        let (gate_values, up_values) = fused.split_at(INTERMEDIATE);
        let s = self.gate_scale[expert];
        gate_values
            .iter()
            .zip(up_values)
            .map(|(&g, &u)| {
                let g = g * s;
                let u = u * s;
                GATE_CAP * (g / GATE_CAP).tanh() / (1. + (-g).exp()) * UP_CAP * (u / UP_CAP).tanh()
            })
            .collect()
    }

    pub(super) fn accumulate(
        &self,
        merged: &mut [f32],
        expert: usize,
        route: f32,
        projected: &[f32],
    ) {
        let s = route * self.down_scale[expert];
        for (out, &value) in merged.iter_mut().zip(projected) {
            *out += value * s;
        }
    }
}

pub(super) fn finish_expert_visit(merged: Vec<f32>) -> Result<Vec<half::bf16>, EngError> {
    if merged.iter().any(|v| !v.is_finite()) {
        return Err(checkpoint_error("nonfinite Inkling GGUF expert output"));
    }
    Ok(merged.into_iter().map(half::bf16::from_f32).collect())
}

fn project_row_groups(
    rows: Range<usize>,
    workers: usize,
    project: impl Fn(Range<usize>) -> Result<Vec<f32>, EngError> + Sync,
) -> Result<Vec<f32>, EngError> {
    if workers <= 1 || rows.len() <= 1 {
        return project(rows);
    }
    let rows_per_worker = rows.len().div_ceil(workers);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for start in rows.clone().step_by(rows_per_worker) {
            let end = (start + rows_per_worker).min(rows.end);
            let project = &project;
            handles.push(
                std::thread::Builder::new()
                    .stack_size(ATTENTION_WORKER_STACK_BYTES)
                    .spawn_scoped(scope, move || project(start..end))
                    .map_err(|error| {
                        EngError::MemoryPlan(format!("Inkling attention worker: {error}"))
                    })?,
            );
        }
        let mut out = Vec::with_capacity(rows.len());
        for handle in handles {
            out.extend(handle.join().expect("Inkling attention worker panicked")?);
        }
        Ok(out)
    })
}

fn project_packed_rows(
    dtype: GgufType,
    row_bytes: usize,
    bytes: &[u8],
    input: &[f32],
) -> Result<Vec<f32>, EngError> {
    let rows = bytes.chunks_exact(row_bytes);
    let result = match dtype {
        GgufType::F32 => rows
            .map(|row| super::dense::dot_f32_le(row, input))
            .collect(),
        GgufType::Bf16 => rows
            .map(|row| super::dense::dot_bf16_le_scalar(row, input))
            .collect(),
        _ => {
            let mut values = vec![0.; input.len()];
            rows.map(|row| {
                crate::weights::dequantize_into(dtype, row, &mut values)?;
                Ok(values.iter().zip(input).map(|(w, x)| w * x).sum())
            })
            .collect::<Result<Vec<_>, EngError>>()?
        }
    };
    Ok(result)
}

pub(super) fn projection_scratch_bytes(attention_workers: usize) -> u64 {
    // Serial scratch includes the largest FP32 controller chunk and logits row.
    let serial_buffer = ROWS_PER_READ * 2 * HIDDEN * 4;
    let base = serial_buffer + PADDED_VOCAB as usize * 4 + 256 * HIDDEN * 4;
    if attention_workers == 1 {
        return base as u64;
    }
    // Each worker can hold one 256-row packed chunk with up to a 3072-wide
    // F32 row, one decoded row, and a fixed stack. Joining also holds one
    // extra full attention output. Only one projection runs at a time.
    let worker_buffer = ROWS_PER_READ * MAX_ATTENTION_PROJECTION_DIM * 4;
    let buffers = serial_buffer.max(attention_workers * worker_buffer);
    let extra = attention_workers
        * (ATTENTION_WORKER_STACK_BYTES + MAX_ATTENTION_PROJECTION_DIM * 4)
        + MAX_ATTENTION_PROJECTION_DIM * 4;
    (base - serial_buffer + buffers + extra) as u64
}

pub(super) fn packed_mlx_enabled() -> bool {
    std::env::var_os("FIFI_INKLING_GGUF_PACKED_MLX")
        .map_or(super::factory::NATIVE_MLX, |value| value == "1")
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn dense_mapping_enabled(native: bool) -> bool {
    std::env::var_os(DENSE_MAPPING_ENV).map_or(native, |value| value == "1")
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn single_copy_experts_enabled() -> bool {
    packed_mlx_enabled()
        && std::env::var_os("FIFI_INKLING_GGUF_SINGLE_COPY_EXPERTS")
            .is_none_or(|value| value == "1")
}

pub(super) fn batched_experts_enabled() -> bool {
    packed_mlx_enabled()
        && std::env::var_os("FIFI_INKLING_GGUF_BATCH_EXPERTS")
            .map_or(super::factory::NATIVE_MLX, |value| value == "1")
}

// NVFP4 has the largest block byte/element ratio among the six packed formats.
// A visit holds a single group's input references at dispatch. Hold one more
// group in the ledger for released projections pending the next sync.
const MAX_PACKED_GROUP_BYTES: u64 = (2 * HIDDEN * INTERMEDIATE / 64 * 36 * super::TOP_K) as u64;
// Fused/activation/down rows, inputs, projection results, concatenate output,
// and host readback/decoded rows fit within eight widest F32 groups.
const GROUP_ROW_WORKSPACE_BYTES: u64 = (8 * super::TOP_K * 2 * INTERMEDIATE * 4) as u64;
const GGUF_BATCH_EXPERT_BYTES: u64 = 2 * MAX_PACKED_GROUP_BYTES + GROUP_ROW_WORKSPACE_BYTES;
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) const GGUF_BATCH_RETIRE_BYTES: u64 =
    LARGEST_PLANE_F32_BYTES as u64 + MAX_PACKED_GROUP_BYTES + GROUP_ROW_WORKSPACE_BYTES;

pub(super) fn native_scratch_bytes(attention_workers: usize) -> u64 {
    // Reserve the existing general plane workspace, plus four resident U32
    // codebooks and a second copy to cover their host upload in packed mode.
    const CODEBOOK_BYTES: u64 = (2048 + 2 * (256 + 512 + 1024)) * 4;
    projection_scratch_bytes(attention_workers)
        + 4 * LARGEST_PLANE_F32_BYTES as u64
        + if packed_mlx_enabled() {
            2 * CODEBOOK_BYTES
        } else {
            0
        }
        + if batched_experts_enabled() {
            GGUF_BATCH_EXPERT_BYTES
        } else {
            0
        }
}

pub(crate) fn preflight_plan(context: Option<u32>) -> Result<super::HeaderPlan, EngError> {
    use crate::family::plan_decision::{PlanDecision, PlanFixity};

    let context = context.unwrap_or(super::DEFAULT_CONTEXT);
    let state = super::state_plan(context)?;
    let attention_workers = super::dense::attention_cpu_workers();
    let staging = native_scratch_bytes(attention_workers);
    let charges = super::base_charges(0, staging, &state);
    let expert_workspace = if batched_experts_enabled() {
        format!(
            "batched GGUF projection workspace including {GGUF_BATCH_EXPERT_BYTES} B for grouped and deferred buffers"
        )
    } else {
        "one GGUF expert plane".to_string()
    };
    let plan = format!(
        "{} B for {expert_workspace}, BF16 state at {context} tokens and decoder scratch ({attention_workers} CPU attention workers)",
        charges.iter().map(|&(_, bytes)| bytes).sum::<u64>(),
    );
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    let (chosen, budget_fit) = if let Some(device) = super::device_budget(
        super::factory::check_gguf_backend().is_ok(),
        super::GGUF_SMALL_WORKING_SET,
        &charges,
    )? {
        match super::working_set_remaining(device, &charges) {
            Ok(headroom) => {
                let chosen = if dense_mapping_enabled(true) {
                    format!(
                        "{plan}, up to {headroom} B shared by mapped packed dense projections and expert cache under {device}; exact dense reservation is selected at load; MLX allocator pool is process-wide and outside these logical cache capacities"
                    )
                } else if single_copy_experts_enabled() {
                    format!(
                        "{plan}, up to {headroom} B of Metal expert retention under {device}; one retained copy when the checkpoint's expert tensors all have direct packed kernels; MLX allocator pool is process-wide and outside this logical capacity"
                    )
                } else {
                    let ram = headroom / 4;
                    format!(
                        "{plan}, {ram} B for packed RAM and {} B for Metal expert planes under {device}; MLX allocator pool is process-wide and outside these logical cache capacities",
                        headroom - ram
                    )
                };
                (chosen, Some(true))
            }
            Err(error) => (format!("{plan}, {}", error.describe()), Some(false)),
        }
    } else {
        (
            format!(
                "{plan}, automatic Metal budget disabled; base allocations unchecked and expert caches disabled"
            ),
            None,
        )
    };
    #[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
    let (chosen, budget_fit) = (
        format!("{plan}, Metal budget unavailable in this build"),
        None,
    );
    Ok(super::HeaderPlan {
        decisions: vec![PlanDecision {
            name: "Apple Silicon GGUF working set".into(),
            chosen,
            alternatives: Vec::new(),
            fixed_at: PlanFixity::Load,
        }],
        budget_fit,
    })
}

fn matrix(tensor: &GgufTensor) -> Result<(usize, usize, GgufType, usize), EngError> {
    let &[width, height, ..] = tensor.dims() else {
        return Err(checkpoint_error(format!(
            "GGUF tensor {} is not a matrix",
            tensor.name()
        )));
    };
    let dtype = tensor.ggml_type();
    Ok((
        width as usize,
        height as usize,
        dtype,
        (width / dtype.block_elems() * dtype.block_bytes()) as usize,
    ))
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) struct ExpertPlane {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) dtype: GgufType,
    pub(super) bytes: std::sync::Arc<Vec<u8>>,
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
impl Weights {
    pub(super) fn read_expert(&self, name: &str, expert: usize) -> Result<ExpertPlane, EngError> {
        let tensor = self.tensor(name)?;
        let (width, height, dtype, row_bytes) = matrix(tensor)?;
        if tensor.dims().get(2).copied() != Some(EXPERTS as u64) || expert >= EXPERTS {
            return Err(EngError::BadRequest(format!(
                "invalid expert {expert} for Inkling GGUF tensor {name}"
            )));
        }
        let mut bytes = self.buffer(name, height * row_bytes)?;
        self.read_range(name, (expert * bytes.len()) as u64, &mut bytes)?;
        Ok(ExpertPlane {
            width,
            height,
            dtype,
            bytes: std::sync::Arc::new(bytes),
        })
    }
}

#[cfg(test)]
mod mapped_projection_tests {
    use super::*;

    #[test]
    fn gguf_attention_worker_admission_covers_pread_buffers_and_stacks() {
        assert_eq!(projection_scratch_bytes(1), 7_881_984);
        assert_eq!(
            projection_scratch_bytes(4) - projection_scratch_bytes(1),
            8_974_336
        );
        assert!(projection_scratch_bytes(2) < projection_scratch_bytes(4));
    }

    #[test]
    fn parallel_quantized_attention_preserves_uneven_q6_k_and_q8_0_rows() {
        let input: Vec<_> = (0..256).map(|i| (i % 11) as f32 / 16. - 0.25).collect();
        for (dtype, row_bytes) in [(GgufType::Q6K, 210), (GgufType::Q80, 272)] {
            let mut weights = vec![0; row_bytes * 11];
            for (index, row) in weights.chunks_exact_mut(row_bytes).enumerate() {
                row.fill((index * 7 + 1) as u8);
                let scale = half::f16::from_f32(0.25).to_bits().to_le_bytes();
                if dtype == GgufType::Q6K {
                    row[208..210].copy_from_slice(&scale);
                } else {
                    for block in row.as_chunks_mut::<34>().0 {
                        block[..2].copy_from_slice(&scale);
                    }
                }
            }
            let run = |workers| {
                project_row_groups(0..11, workers, |range| {
                    project_packed_rows(
                        dtype,
                        row_bytes,
                        &weights[range.start * row_bytes..range.end * row_bytes],
                        &input,
                    )
                })
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
            };
            let serial = run(1);
            assert_ne!(serial[0], serial[10]);
            assert_eq!(serial, run(4));
        }
    }

    #[test]
    fn mapped_projection_matches_buffered_gates_and_bounds_the_selected_rows() {
        use std::io::Write;

        let mut artifact = tempfile::NamedTempFile::new().unwrap();
        let name = "output.weight";
        let mut header = b"GGUF".to_vec();
        header.extend(3_u32.to_le_bytes());
        header.extend(1_u64.to_le_bytes());
        header.extend(0_u64.to_le_bytes());
        header.extend((name.len() as u64).to_le_bytes());
        header.extend(name.as_bytes());
        header.extend(2_u32.to_le_bytes());
        header.extend(3_u64.to_le_bytes());
        header.extend(257_u64.to_le_bytes());
        header.extend(0_u32.to_le_bytes());
        header.extend(0_u64.to_le_bytes());
        header.resize(header.len().next_multiple_of(32), 0);
        artifact.write_all(&header).unwrap();
        for _ in 0..257 {
            for weight in [1.0_f32, -2.0, 3.0] {
                artifact.write_all(&weight.to_le_bytes()).unwrap();
            }
        }
        let file = File::open(artifact.path()).unwrap();
        let meta = file.metadata().unwrap();
        let header =
            GgufHeader::parse(&mut BufReader::new(&file), meta.len(), HEADER_LIMIT).unwrap();
        let weights = Weights {
            file,
            header,
            modified: meta.modified().ok(),
            dense_io: Some(DenseIo::default()),
        };
        let mapping = weights.map_projection_source().unwrap();
        let input = [0.5, 0.25, -0.75];
        let before = weights.project(name, 0, 0..257, &input, true).unwrap();
        let after = weights
            .project_mapped(name, 0..1, &input, &mapping, 12, 1)
            .unwrap();
        assert_eq!(before[0].to_bits(), after[0].to_bits());
        assert!(
            weights
                .project_mapped(name, 1..2, &input, &mapping, 12, 1)
                .is_err()
        );
        // Expert callers use the same read helpers without charging dense bytes.
        weights.project(name, 0, 0..257, &input, false).unwrap();
        weights.read_range(name, 0, &mut [0; 12]).unwrap();
        let tensor = weights.tensor(name).unwrap();
        weights.values(tensor, false).unwrap();
        weights.values(tensor, true).unwrap();
        weights.row(name, 0).unwrap();
        let io = weights.dense_io.as_ref().unwrap();
        assert_eq!(io.snapshot().file_ranges, 4);
        assert_eq!(io.snapshot().file_bytes, 2 * 257 * 12 + 12);
        assert_eq!(io.snapshot().mapped_accesses, 1);
        assert_eq!(io.snapshot().mapped_span_bytes, 12);
        let before_io = io.snapshot();
        let parallel = weights
            .project_with_workers(name, 0, 0..257, &input, true, 4)
            .unwrap();
        let mapped = weights
            .project_mapped(name, 0..257, &input, &mapping, 257 * 12, 4)
            .unwrap();
        let bits = |values: Vec<f32>| values.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        assert_eq!(bits(before.clone()), bits(parallel));
        assert_eq!(bits(before), bits(mapped));
        let delta = io.snapshot().since(before_io);
        assert_eq!(delta.file_bytes, 257 * 12);
        assert_eq!(delta.file_ranges, 4);
        assert_eq!(delta.mapped_span_bytes, 257 * 12);
        assert_eq!(delta.mapped_accesses, 1);
        artifact.as_file().set_len(0).unwrap();
        assert!(
            weights
                .project_mapped(name, 0..1, &input, &mapping, 12, 1)
                .is_err()
        );
    }

    #[test]
    fn mapped_packed_dots_preserve_reference_accumulation() {
        let input: Vec<_> = (0..256).map(|i| ((i % 11) as f32 - 5.) / 8.).collect();
        let weights = input.iter().rev().copied();
        let f32_bytes: Vec<_> = weights.clone().flat_map(f32::to_le_bytes).collect();
        let bf16_bytes: Vec<_> = weights
            .flat_map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        let mut iq4_block = vec![0x33; 136];
        iq4_block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
        for (dtype, row) in [
            (GgufType::F32, f32_bytes),
            (GgufType::Bf16, bf16_bytes),
            (GgufType::Iq4Xs, iq4_block),
        ] {
            let mut values = vec![0.; input.len()];
            crate::weights::dequantize_into(dtype, &row, &mut values).unwrap();
            let reference: f32 = values.iter().zip(&input).map(|(w, x)| w * x).sum();
            let mapped = project_packed_rows(dtype, row.len(), &row, &input).unwrap();
            assert_eq!(mapped[0].to_bits(), reference.to_bits(), "{dtype:?}");
        }
    }
}
