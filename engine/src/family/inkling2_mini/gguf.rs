//! Tensor validation and bounded reads for the two fixed-loop GGUF formats.

use std::{fs::File, io::BufReader, ops::Range, path::Path, time::SystemTime};

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

pub(super) const DENSE_MAPPING_ENV: &str = "FIFI_INKLING_GGUF_MMAP_DENSE";
const ROWS_PER_READ: usize = 256;
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

    pub(super) fn values(&self, tensor: &GgufTensor) -> Result<Vec<f32>, EngError> {
        let mut bytes = self.buffer(tensor.name(), tensor.byte_len() as usize)?;
        self.read_range(tensor.name(), 0, &mut bytes)?;
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
        self.read_range(name, (row * row_bytes) as u64, &mut bytes)?;
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
    ) -> Result<Vec<f32>, EngError> {
        let tensor = self.tensor(name)?;
        let (width, height, dtype, row_bytes) = matrix(tensor)?;
        let planes = tensor.dims().get(2).copied().unwrap_or(1) as usize;
        if plane >= planes || rows.start > rows.end || rows.end > height || input.len() != width {
            return Err(EngError::BadRequest(format!(
                "projection exceeds Inkling GGUF tensor {name}"
            )));
        }
        let mut bytes = self.buffer(name, rows.len().min(ROWS_PER_READ) * row_bytes)?;
        let mut result = Vec::with_capacity(rows.len());
        let mut values = vec![0.; width];
        for start in (rows.start..rows.end).step_by(ROWS_PER_READ) {
            let count = ROWS_PER_READ.min(rows.end - start);
            let bytes = &mut bytes[..count * row_bytes];
            self.read_range(name, ((plane * height + start) * row_bytes) as u64, bytes)?;
            for row in bytes.chunks_exact(row_bytes) {
                crate::weights::dequantize_into(dtype, row, &mut values)?;
                result.push(values.iter().zip(input).map(|(w, x)| w * x).sum());
            }
        }
        Ok(result)
    }

    pub(super) fn project_mapped(
        &self,
        name: &str,
        rows: Range<usize>,
        input: &[f32],
        mapping: &MappedFile,
        admitted_bytes: u64,
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
        let result = project_packed_rows(dtype, row_bytes, bytes, input)?;
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
            |name, plane, rows, input| self.project(name, plane, 0..rows, input),
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
        use super::{GATE_CAP, TOP_K, UP_CAP};
        if layer >= LAYERS || selected.len() != TOP_K || routes.len() != TOP_K {
            return Err(EngError::BadRequest(
                "bad Inkling GGUF expert selection".into(),
            ));
        }
        let gate = format!("blk.{layer}.ffn_gate_up_exps");
        let down = format!("blk.{layer}.ffn_down_exps");
        let scale = |prefix: &str| -> Result<Vec<f32>, EngError> {
            self.header
                .tensor(&format!("{prefix}.scale"))
                .map_or_else(|| Ok(vec![1.; EXPERTS]), |tensor| self.values(tensor))
        };
        let gate_scale = scale(&gate)?;
        let down_scale = scale(&down)?;
        let gate = format!("{gate}.weight");
        let down = format!("{down}.weight");
        let mut merged = vec![0.; HIDDEN];
        for (&expert, &route) in selected.iter().zip(routes) {
            if expert >= EXPERTS {
                return Err(EngError::BadRequest(
                    "Inkling GGUF expert ID out of range".into(),
                ));
            }
            let fused = project(&gate, expert, 2 * INTERMEDIATE, input)?;
            let (gate_values, up_values) = fused.split_at(INTERMEDIATE);
            let s = gate_scale[expert];
            let activated: Vec<_> = gate_values
                .iter()
                .zip(up_values)
                .map(|(&g, &u)| {
                    let g = g * s;
                    let u = u * s;
                    GATE_CAP * (g / GATE_CAP).tanh() / (1. + (-g).exp())
                        * UP_CAP
                        * (u / UP_CAP).tanh()
                })
                .collect();
            let projected = project(&down, expert, HIDDEN, &activated)?;
            let s = route * down_scale[expert];
            for (out, value) in merged.iter_mut().zip(projected) {
                *out += value * s;
            }
        }
        if merged.iter().any(|v| !v.is_finite()) {
            return Err(checkpoint_error("nonfinite Inkling GGUF expert output"));
        }
        Ok(merged.into_iter().map(half::bf16::from_f32).collect())
    }
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

pub(super) fn projection_scratch_bytes() -> u64 {
    // Largest possible FP32 row (the controller input) times the chunk, plus
    // one output logits row and conservatively sized attention and MoE temporaries.
    (ROWS_PER_READ * 2 * HIDDEN * 4 + PADDED_VOCAB as usize * 4 + 256 * HIDDEN * 4) as u64
}

pub(super) fn packed_mlx_enabled() -> bool {
    std::env::var_os("FIFI_INKLING_GGUF_PACKED_MLX").is_some_and(|value| value == "1")
}

pub(super) fn native_scratch_bytes() -> u64 {
    // Reserve the existing general plane workspace, plus four resident U32
    // codebooks and a second copy to cover their host upload in packed mode.
    const CODEBOOK_BYTES: u64 = (2048 + 2 * (256 + 512 + 1024)) * 4;
    projection_scratch_bytes()
        + 4 * LARGEST_PLANE_F32_BYTES as u64
        + if packed_mlx_enabled() {
            2 * CODEBOOK_BYTES
        } else {
            0
        }
}

pub(crate) fn preflight_plan(context: Option<u32>) -> Result<super::HeaderPlan, EngError> {
    use crate::family::plan_decision::{PlanDecision, PlanFixity};

    let context = context.unwrap_or(super::DEFAULT_CONTEXT);
    let state = super::state_plan(context)?;
    let staging = native_scratch_bytes();
    let charges = super::base_charges(0, staging, &state);
    let plan = format!(
        "{} B for one GGUF expert plane, BF16 state at {context} tokens and decoder scratch",
        charges.iter().map(|&(_, bytes)| bytes).sum::<u64>(),
    );
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    let (chosen, budget_fit) = if let Some(device) =
        crate::family::mlx_budget::DeviceBudget::detect_with_default(
            super::factory::check_gguf_backend().is_ok(),
        )? {
        match super::working_set_remaining(device, &charges) {
            Ok(headroom) => {
                let chosen = if std::env::var_os(DENSE_MAPPING_ENV)
                    .is_some_and(|value| value == "1")
                {
                    format!(
                        "{plan}, up to {headroom} B shared by mapped packed dense projections and expert cache under {device}; exact dense reservation is selected at load; MLX allocator pool is process-wide and outside these logical cache capacities"
                    )
                } else {
                    let ram = headroom / 4;
                    format!(
                        "{plan}, {ram} B for packed RAM and {} B for expanded Metal expert planes under {device}; MLX allocator pool is process-wide and outside these logical cache capacities",
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
    pub(super) bytes: Vec<u8>,
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
            bytes,
        })
    }
}

#[cfg(test)]
mod mapped_projection_tests {
    use super::*;

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
        header.extend(2_u64.to_le_bytes());
        header.extend(0_u32.to_le_bytes());
        header.extend(0_u64.to_le_bytes());
        header.resize(header.len().next_multiple_of(32), 0);
        artifact.write_all(&header).unwrap();
        for weight in [1.0_f32, -2.0, 3.0, 4.0, 5.0, 1.0] {
            artifact.write_all(&weight.to_le_bytes()).unwrap();
        }
        let file = File::open(artifact.path()).unwrap();
        let meta = file.metadata().unwrap();
        let header =
            GgufHeader::parse(&mut BufReader::new(&file), meta.len(), HEADER_LIMIT).unwrap();
        let weights = Weights {
            file,
            header,
            modified: meta.modified().ok(),
        };
        let mapping = weights.map_projection_source().unwrap();
        let input = [0.5, 0.25, -0.75];
        let before = weights.project(name, 0, 0..1, &input).unwrap();
        let after = weights
            .project_mapped(name, 0..1, &input, &mapping, 12)
            .unwrap();
        assert_eq!(before[0].to_bits(), after[0].to_bits());
        assert!(
            weights
                .project_mapped(name, 1..2, &input, &mapping, 12)
                .is_err()
        );
        artifact.as_file().set_len(0).unwrap();
        assert!(
            weights
                .project_mapped(name, 0..1, &input, &mapping, 12)
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
