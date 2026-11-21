//! The device trunk of a Flash-Next GGUF: the tensors the decode step keeps resident, uploaded
//! under the names the step reads (the pack names of `exl3_pack`), each in the form the kernel
//! that reads it takes.
//!
//! - Quantized matrices stay as stored blocks (`CudaWeight::Native`) for the matvec kernels.
//! - The hyper-connection tensors are widened to F16 (the kernels read half-precision weights); the
//!   gamma is stored `1 + w` and the kernel adds the 1, so the upload subtracts it.
//! - The vectors the scan, the router and the norms read are widened to F32; the indexer gammas
//!   lose their 1 as well. The query and key norm gammas are stored `1 + w`, the form the attention
//!   prep reads, and go up as stored. `ssm_a` is `-exp(A_log)`, the decay the scan reads.
//! - The two BF16 indexer projections stay BF16 (the dense matvec reads them).
//! - The embedding, the PLE block and the n-gram table are read on the host; the routed experts
//!   go up as native stacks when asked (a small model), and are the expert tiers' otherwise.
//!
//! The DeltaNet value heads stay in the file's tiled order: the scan runs in its tiled form and
//! no tensor is reordered.

use super::{
    GgufModel,
    tensors::{Convention, Mapped, Role, Site},
};
use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        exl3_tensors::f32_le_bytes,
        project::cuda::{CudaWeight, CudaWeights},
    },
    kernels::cuda::{CudaContext, QuantTensor},
    weights::{GgufType, dequantize},
};

/// The pack name of the hyper-connection mixer that feeds `lm_head`.
const MIXER: &str = "model.language_model.hyper_connection_mixer";

/// How one tensor goes to the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Upload {
    Native,
    /// Widened to F16; `minus_one` returns a `1 + w` gamma to `w`.
    Half {
        minus_one: bool,
    },
    /// Widened to F32.
    Float {
        minus_one: bool,
    },
    /// The stored 16-bit floats as they are.
    Raw,
}

/// The key of the routed-expert stack of `layer`, `part` one of `gate`, `up`, `down`.
pub(in crate::family::qwen::flash_next) fn expert_key(layer: usize, part: &str) -> String {
    format!("layer {layer} routed {part} stack")
}

fn site_prefix(site: Site, layer: Option<usize>) -> String {
    match (site, layer) {
        (Site::Mixer, _) => MIXER.to_string(),
        (Site::Attn, Some(layer)) => layer_name(layer, "attn_hyper_connection"),
        (Site::Mlp, Some(layer)) => layer_name(layer, "mlp_hyper_connection"),
        _ => unreachable!("a layer site has a layer"),
    }
}

/// The name the step reads a tensor under and how it is uploaded; `None` for a tensor the device
/// does not hold.
fn plan_of(entry: &Mapped) -> Option<(String, Upload)> {
    let layer = entry.layer;
    let at = |suffix: &str| layer_name(layer.unwrap_or(0), suffix);
    let native = |name: String| Some((name, Upload::Native));
    let float = |name: String| Some((name, Upload::Float { minus_one: false }));
    match entry.role {
        Role::Head => native("lm_head".to_string()),
        Role::HyperNorm(site) => Some((
            format!("{}.hc_norm.weight", site_prefix(site, layer)),
            Upload::Half { minus_one: true },
        )),
        Role::HyperDown(site) => Some((
            format!("{}.input_mix_weight_down.weight", site_prefix(site, layer)),
            Upload::Half { minus_one: false },
        )),
        Role::HyperUp(site) => Some((
            format!("{}.input_mix_weight_up.weight", site_prefix(site, layer)),
            Upload::Half { minus_one: false },
        )),
        Role::HyperInject(site) => Some((
            format!("{}.block_inject_weight.weight", site_prefix(site, layer)),
            Upload::Half { minus_one: false },
        )),
        Role::DeltaQkv => native(at("linear_attn.in_proj_qkv")),
        Role::DeltaZ => native(at("linear_attn.in_proj_z")),
        Role::DeltaOut => native(at("linear_attn.out_proj")),
        Role::DeltaBeta => float(at("linear_attn.in_proj_b.weight")),
        Role::DeltaAlpha => float(at("linear_attn.in_proj_a.weight")),
        Role::DeltaConv => float(at("linear_attn.conv1d.weight")),
        Role::DeltaDecay => float(at("linear_attn.A_log")),
        Role::DeltaDtBias => float(at("linear_attn.dt_bias")),
        Role::DeltaNorm => float(at("linear_attn.norm.weight")),
        Role::QsaQ => native(at("self_attn.q_proj")),
        Role::QsaK => native(at("self_attn.k_proj")),
        Role::QsaV => native(at("self_attn.v_proj")),
        Role::QsaO => native(at("self_attn.o_proj")),
        Role::QsaQNorm => float(at("self_attn.q_norm.weight")),
        Role::QsaKNorm => float(at("self_attn.k_norm.weight")),
        Role::QsaIndexQNorm => Some((
            at("self_attn.indexer.q_layernorm.weight"),
            Upload::Float { minus_one: true },
        )),
        Role::QsaIndexKNorm => Some((
            at("self_attn.indexer.k_layernorm.weight"),
            Upload::Float { minus_one: true },
        )),
        Role::QsaIndexQ => Some((at("self_attn.indexer.index_qk_proj.q"), Upload::Raw)),
        Role::QsaIndexK => Some((at("self_attn.indexer.index_qk_proj.k"), Upload::Raw)),
        Role::Router => float(at("mlp.gate.weight")),
        Role::SharedScalar => float(at("mlp.shared_expert_gate.weight")),
        Role::SharedGate => native(at("mlp.shared_expert.gate_proj")),
        Role::SharedUp => native(at("mlp.shared_expert.up_proj")),
        Role::SharedDown => native(at("mlp.shared_expert.down_proj")),
        Role::Embedding
        | Role::NgramTable
        | Role::PleKey
        | Role::PleValue
        | Role::PleNormKey
        | Role::PleNormQuery
        | Role::PleNormConv
        | Role::PleConv
        | Role::ExpertGate
        | Role::ExpertUp
        | Role::ExpertDown => None,
    }
}

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF trunk: {}", detail.as_ref()))
}

/// F32 values to the little-endian F16 bytes of a half-precision buffer, round to nearest even.
fn f16_le_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| f32_to_f16(value).to_le_bytes())
        .collect()
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity or NaN.
        return sign | 0x7c00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let rounded = (m + (1 << (shift - 1)) - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let rounded = (mantissa + 0x0fff + ((mantissa >> 13) & 1)) >> 13;
    sign | (((exponent as u32) << 10) + rounded) as u16
}

/// Uploads the resident trunk of `model` on `ctx`, and with `resident_experts` the routed experts
/// as native stacks under [`expert_key`]. `chunk_limit` is the host bytes one native read holds.
pub(in crate::family::qwen::flash_next) fn gguf_trunk(
    model: &'static GgufModel,
    ctx: &CudaContext,
    resident_experts: bool,
    chunk_limit: usize,
) -> Result<CudaWeights, EngError> {
    let weights = &*model.weights;
    let experts = model.config().dims.experts as usize;
    let mut out: Vec<CudaWeight> = Vec::new();
    for entry in model.map().entries() {
        let expert_part = match entry.role {
            Role::ExpertGate => Some("gate"),
            Role::ExpertUp => Some("up"),
            Role::ExpertDown => Some("down"),
            _ => None,
        };
        let planned = match (expert_part, resident_experts) {
            (Some(part), true) => {
                Some((expert_key(entry.layer.unwrap_or(0), part), Upload::Native))
            }
            (Some(_), false) => None,
            (None, _) => plan_of(entry),
        };
        let Some((name, how)) = planned else {
            continue;
        };
        let upload = match how {
            Upload::Native => {
                let (k, rows) = match entry.dims.as_slice() {
                    [k, rows] => (*k as usize, *rows as usize),
                    [k, rows, stack] => (*k as usize, *rows as usize * *stack as usize),
                    other => {
                        return Err(refuse(format!("{} has dimensions {other:?}", entry.name)));
                    }
                };
                let tensor = QuantTensor {
                    name: &name,
                    ty: entry.ty,
                    n_rows: rows,
                    k,
                    source_len: entry.bytes as usize,
                };
                let source = entry.name.clone();
                let mut read =
                    |offset: usize, buf: &mut [u8]| weights.read_range(&source, offset as u64, buf);
                let native = ctx.upload_native(tensor, chunk_limit, &mut read)?;
                let native = if expert_part.is_some() {
                    native.with_experts(Some(experts))
                } else {
                    native
                };
                CudaWeight::Native {
                    name,
                    weight: native,
                }
            }
            Upload::Raw => {
                if !matches!(entry.ty, GgufType::Bf16 | GgufType::F16) {
                    return Err(refuse(format!(
                        "{} is {}, the dense matvec reads BF16 or F16",
                        entry.name,
                        entry.ty.name()
                    )));
                }
                let (_, bytes) = weights.read_tensor(&entry.name)?;
                CudaWeight::Raw {
                    name,
                    ty: entry.ty,
                    buffer: ctx.upload(&bytes)?,
                }
            }
            Upload::Half { minus_one } => {
                let values = widened(weights, entry, minus_one)?;
                CudaWeight::Raw {
                    name,
                    ty: GgufType::F16,
                    buffer: ctx.upload(&f16_le_bytes(&values))?,
                }
            }
            Upload::Float { minus_one } => {
                let values = widened(weights, entry, minus_one)?;
                CudaWeight::Raw {
                    name,
                    ty: GgufType::F32,
                    buffer: ctx.upload(&f32_le_bytes(&values))?,
                }
            }
        };
        out.push(upload);
    }
    CudaWeights::from_weights(out)
}

/// The tensor decoded to F32, with a `1 + w` gamma returned to `w` when `minus_one`.
fn widened(
    weights: &crate::weights::GgufWeights,
    entry: &Mapped,
    minus_one: bool,
) -> Result<Vec<f32>, EngError> {
    if minus_one && entry.convention != Convention::GammaOnePlusW {
        return Err(refuse(format!("{} is not a `1 + w` gamma", entry.name)));
    }
    let (ty, bytes) = weights.read_tensor(&entry.name)?;
    let mut values = dequantize(ty, &bytes)?;
    if minus_one {
        for value in &mut values {
            *value -= 1.0;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::f32_to_f16;

    #[test]
    fn half_conversion_matches_known_values() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e9), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        // 2^-24, the smallest subnormal.
        assert_eq!(f32_to_f16(5.960_464_5e-8), 1);
        // A value between two halves rounds to even: 1 + 2^-11 is the tie below 1 + 2^-10.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
    }
}
