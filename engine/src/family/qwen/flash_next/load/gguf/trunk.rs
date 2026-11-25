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

use std::collections::HashSet;

use super::{
    GgufModel,
    tensors::{Convention, Form, Mapped, Slot},
};
use crate::{
    EngError,
    family::{
        exl3_pack::layer_name,
        exl3_tensors::f32_le_bytes,
        project::cuda::{CudaWeight, CudaWeights},
        recipe::{Global, Role as Wanted, Site, SiteTensor},
    },
    kernels::cuda::{CudaContext, QuantTensor},
    weights::{GgufType, dequantize},
};

/// The pack name of the hyper-connection mixer that feeds `lm_head`.
const MIXER: &str = "model.language_model.hyper_connection_mixer";

/// The tensors of a hyper-connection site, under its pack name.
const HYPER_TAILS: [&str; 4] = [
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
    "block_inject_weight.weight",
];

/// The trunk tensors of the mixer that feeds `lm_head`, under its pack name (no inject).
const MIXER_TAILS: [&str; 3] = [
    "hc_norm.weight",
    "input_mix_weight_down.weight",
    "input_mix_weight_up.weight",
];

/// The pack names the step reads in every layer: the two hyper-connection sites and the MoE.
const EVERY_LAYER: [&str; 13] = [
    "mlp.gate.weight",
    "mlp.shared_expert_gate.weight",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "attn_hyper_connection.hc_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight",
    "attn_hyper_connection.block_inject_weight.weight",
    "mlp_hyper_connection.hc_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight",
];

/// The pack names of the gated DeltaNet tensors the step reads in a DeltaNet layer.
const DELTA: [&str; 9] = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
];

/// The pack names of the QSA attention tensors the step reads in an attention layer.
const QSA: [&str; 10] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.indexer.q_layernorm.weight",
    "self_attn.indexer.k_layernorm.weight",
    "self_attn.indexer.index_qk_proj.q",
    "self_attn.indexer.index_qk_proj.k",
];

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
        (Site::Head, _) => MIXER.to_string(),
        (Site::Mixer, Some(layer)) => layer_name(layer, "attn_hyper_connection"),
        (Site::Ffn, Some(layer)) => layer_name(layer, "mlp_hyper_connection"),
        _ => unreachable!("a layer site has a layer"),
    }
}

/// The pack name of one tensor of a hyper-connection site and how it goes up: the norm gamma is
/// stored `1 + w` and goes up as `w` in F16, the down, up and inject matrices as F16.
fn hyper_plan(site: Site, part: SiteTensor, layer: Option<usize>) -> (String, Upload) {
    let prefix = site_prefix(site, layer);
    match part {
        SiteTensor::Norm => (
            format!("{prefix}.hc_norm.weight"),
            Upload::Half { minus_one: true },
        ),
        SiteTensor::Down => (
            format!("{prefix}.input_mix_weight_down.weight"),
            Upload::Half { minus_one: false },
        ),
        SiteTensor::Up => (
            format!("{prefix}.input_mix_weight_up.weight"),
            Upload::Half { minus_one: false },
        ),
        SiteTensor::Inject => (
            format!("{prefix}.block_inject_weight.weight"),
            Upload::Half { minus_one: false },
        ),
    }
}

/// The name the step reads a tensor under and how it is uploaded; `None` for a tensor the device
/// does not hold.
fn plan_of(entry: &Mapped) -> Option<(String, Upload)> {
    let layer = entry.layer;
    let at = |suffix: &str| layer_name(layer.unwrap_or(0), suffix);
    let native = |name: String| Some((name, Upload::Native));
    let float = |name: String| Some((name, Upload::Float { minus_one: false }));
    let role = match entry.role {
        Slot::Global(Global::Head) => return native("lm_head".to_string()),
        Slot::Global(Global::HeadSite(part)) => {
            return Some(hyper_plan(Site::Head, part, None));
        }
        Slot::Global(_) => return None,
        Slot::Layer(role) => role,
    };
    match role {
        Wanted::Hyper(site, part) => Some(hyper_plan(site, part, layer)),
        Wanted::DeltaQkv => native(at("linear_attn.in_proj_qkv")),
        Wanted::DeltaGate => native(at("linear_attn.in_proj_z")),
        Wanted::DeltaOut => native(at("linear_attn.out_proj")),
        Wanted::DeltaBeta => float(at("linear_attn.in_proj_b.weight")),
        Wanted::DeltaAlpha => float(at("linear_attn.in_proj_a.weight")),
        Wanted::DeltaConv => float(at("linear_attn.conv1d.weight")),
        Wanted::DeltaDecay => float(at("linear_attn.A_log")),
        Wanted::DeltaDtBias => float(at("linear_attn.dt_bias")),
        Wanted::DeltaNorm => float(at("linear_attn.norm.weight")),
        Wanted::QProj => native(at("self_attn.q_proj")),
        Wanted::KProj => native(at("self_attn.k_proj")),
        Wanted::VProj => native(at("self_attn.v_proj")),
        Wanted::OProj => native(at("self_attn.o_proj")),
        Wanted::QNorm => float(at("self_attn.q_norm.weight")),
        Wanted::KNorm => float(at("self_attn.k_norm.weight")),
        Wanted::IndexQNorm => Some((
            at("self_attn.indexer.q_layernorm.weight"),
            Upload::Float { minus_one: true },
        )),
        Wanted::IndexKNorm => Some((
            at("self_attn.indexer.k_layernorm.weight"),
            Upload::Float { minus_one: true },
        )),
        Wanted::IndexProj => Some((at("self_attn.indexer.index_qk_proj.q"), Upload::Raw)),
        Wanted::IndexKey => Some((at("self_attn.indexer.index_qk_proj.k"), Upload::Raw)),
        Wanted::Router => float(at("mlp.gate.weight")),
        Wanted::SharedRouterGate => float(at("mlp.shared_expert_gate.weight")),
        Wanted::SharedGate => native(at("mlp.shared_expert.gate_proj")),
        Wanted::SharedUp => native(at("mlp.shared_expert.up_proj")),
        Wanted::SharedDown => native(at("mlp.shared_expert.down_proj")),
        Wanted::AttnNorm
        | Wanted::FfnNorm
        | Wanted::DenseGate
        | Wanted::DenseUp
        | Wanted::DenseDown
        | Wanted::PleKey
        | Wanted::PleValue
        | Wanted::PleNormKey
        | Wanted::PleNormQuery
        | Wanted::PleNormConv
        | Wanted::PleConv
        | Wanted::ExpertGate
        | Wanted::ExpertUp
        | Wanted::ExpertDown => None,
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

/// One tensor of the trunk plan: the mapped entry, the name the step reads it under and how
/// it goes up.
#[derive(Debug, Clone)]
struct Planned {
    entry: Mapped,
    name: String,
    how: Upload,
}

/// The part of the routed-expert stack `role` is a tensor of, when it is one.
fn expert_part(role: Slot) -> Option<&'static str> {
    match role {
        Slot::Layer(Wanted::ExpertGate) => Some("gate"),
        Slot::Layer(Wanted::ExpertUp) => Some("up"),
        Slot::Layer(Wanted::ExpertDown) => Some("down"),
        _ => None,
    }
}

/// The resident trunk of `model`: every mapped tensor the step reads on the card, and the
/// routed experts of the layers `resident_experts` names as native stacks under [`expert_key`].
/// No device memory moves: [`check_plan`] refuses an incomplete plan before the upload starts.
fn plan(model: &GgufModel, resident_experts: &dyn Fn(usize) -> bool) -> Vec<Planned> {
    let mut out = Vec::new();
    for entry in model.map().entries() {
        let layer = entry.layer.unwrap_or(0);
        let expert_part = expert_part(entry.role);
        let planned = match expert_part {
            Some(part) if resident_experts(layer) => {
                Some((expert_key(layer, part), Upload::Native))
            }
            Some(_) => None,
            None => plan_of(entry),
        };
        if let Some((name, how)) = planned {
            out.push(Planned {
                entry: entry.clone(),
                name,
                how,
            });
        }
    }
    out
}

/// The pack names the step reads outside the layers and in each of the model's layers of its
/// kind, the routed expert stacks of the layers `resident_experts` names included.
fn required(model: &GgufModel, resident_experts: &dyn Fn(usize) -> bool) -> Vec<String> {
    let mut out: Vec<String> = vec!["lm_head".to_string()];
    out.extend(MIXER_TAILS.iter().map(|tail| format!("{MIXER}.{tail}")));
    for (layer, &attention) in model.config().kinds().iter().enumerate() {
        let at = |suffix: &str| layer_name(layer, suffix);
        out.extend(EVERY_LAYER.iter().map(|tail| at(tail)));
        let mixer: &[&str] = if attention { &QSA } else { &DELTA };
        out.extend(mixer.iter().map(|suffix| at(suffix)));
        if resident_experts(layer) {
            out.extend(["gate", "up", "down"].map(|part| expert_key(layer, part)));
        }
    }
    out
}

/// Whether `how` carries `entry` to the form its kernel reads: a role stored as the device
/// reads it goes up as stored, and a `1 + w` gamma goes up less one for a kernel that reads
/// `w` and as stored for the one that adds the 1 itself ([`Form::NormPlusOne`]).
fn convention_applied(entry: &Mapped, how: Upload) -> bool {
    match (entry.convention, how) {
        (Convention::AsStored, Upload::Native | Upload::Raw) => true,
        (Convention::AsStored, Upload::Half { minus_one } | Upload::Float { minus_one }) => {
            !minus_one
        }
        (Convention::GammaOnePlusW, Upload::Native | Upload::Raw) => false,
        (Convention::GammaOnePlusW, Upload::Half { minus_one } | Upload::Float { minus_one }) => {
            minus_one == (entry.form != Form::NormPlusOne)
        }
    }
}

/// Refuses a plan the step could not run on, before any upload: every name the step reads
/// outside the layers and in layers `0..layers` must be planned (the first missing one is
/// named), and every planned tensor must reach the form its kernel reads. The map already
/// refuses a file without a tensor; these catch a role the plan does not cover, or covers
/// without the value convention of its file.
fn check_plan(
    model: &GgufModel,
    planned: &[Planned],
    resident_experts: &dyn Fn(usize) -> bool,
) -> Result<(), EngError> {
    let names: HashSet<&str> = planned.iter().map(|tensor| tensor.name.as_str()).collect();
    if let Some(missing) = required(model, resident_experts)
        .into_iter()
        .find(|name| !names.contains(name.as_str()))
    {
        return Err(refuse(format!(
            "the step reads {missing}, and the plan has no tensor under that name"
        )));
    }
    for Planned { entry, name, how } in planned {
        if !convention_applied(entry, *how) {
            let reads = if entry.form == Form::NormPlusOne {
                "`1 + w` and takes the gamma as stored"
            } else {
                "`w` and takes the gamma less one"
            };
            return Err(refuse(format!(
                "{name} ({}) is stored as `1 + w`; its kernel reads {reads}",
                entry.name
            )));
        }
    }
    Ok(())
}

/// Uploads the resident trunk of `model` on `ctx`, and the routed experts of the layers
/// `resident_experts` names as native stacks under [`expert_key`]. `chunk_limit` is the host bytes one native read holds.
pub(in crate::family::qwen::flash_next) fn gguf_trunk(
    model: &GgufModel,
    ctx: &CudaContext,
    resident_experts: &dyn Fn(usize) -> bool,
    chunk_limit: usize,
) -> Result<CudaWeights, EngError> {
    let weights = &*model.weights;
    let planned = plan(model, resident_experts);
    check_plan(model, &planned, resident_experts)?;
    let mut out: Vec<CudaWeight> = Vec::new();
    for Planned { entry, name, how } in &planned {
        let (name, how) = (name.clone(), *how);
        let expert_part = expert_part(entry.role);
        let upload = match how {
            Upload::Native => {
                let (k, rows, stack) = match entry.dims.as_slice() {
                    [k, rows] => (*k as usize, *rows as usize, None),
                    [k, rows, stack] => (
                        *k as usize,
                        *rows as usize * *stack as usize,
                        Some(*stack as usize),
                    ),
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
                // The recorded depth is the tensor's own stack dimension, which the MoE block
                // checks against its rows and the dims' expert count.
                let native = match (expert_part, stack) {
                    (Some(_), Some(depth)) => native.with_experts(Some(depth)),
                    (Some(_), None) => {
                        return Err(refuse(format!(
                            "{} is an expert stack without a stack dimension",
                            entry.name
                        )));
                    }
                    (None, _) => native,
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
    use super::*;
    use crate::family::qwen::flash_next::load::gguf::smallmodel::write_small;

    fn small_model() -> GgufModel {
        let small = write_small(7);
        GgufModel::open(&small.first)
            .expect("open")
            .expect("a qwen4exp model")
    }

    /// The plan of the small model covers every name the step reads, and a plan that lacks one
    /// of them is refused by that name before any upload. (A file that lacks the tensor is
    /// refused by the map at open; this is the plan's own check, which needs no card.)
    #[test]
    fn a_plan_without_a_trunk_tensor_the_step_reads_is_refused_by_its_name() {
        let model = small_model();
        // Layers from 0 read their experts from the card; the others stream them.
        let resident = |layer: usize| layer.is_multiple_of(2);
        let planned = plan(&model, &resident);
        check_plan(&model, &planned, &resident).expect("the plan is complete");
        assert!(planned.iter().any(|t| t.name == expert_key(0, "up")));
        assert!(!planned.iter().any(|t| t.name == expert_key(1, "up")));
        for name in [
            layer_name(1, "mlp.shared_expert_gate.weight"),
            layer_name(3, "self_attn.indexer.index_qk_proj.q"),
            format!("{MIXER}.input_mix_weight_up.weight"),
            expert_key(2, "down"),
        ] {
            let lacking: Vec<Planned> = planned
                .iter()
                .filter(|tensor| tensor.name != name)
                .cloned()
                .collect();
            let error = format!(
                "{:?}",
                check_plan(&model, &lacking, &resident).expect_err("a lacking plan")
            );
            assert!(error.contains(&name), "{name}: {error}");
        }
        // The experts of a streamed layer are not trunk tensors.
        let lacking: Vec<Planned> = planned
            .iter()
            .filter(|tensor| tensor.name != expert_key(1, "up"))
            .cloned()
            .collect();
        check_plan(&model, &lacking, &resident).expect("layer 1 streams its experts");
    }

    /// A role the file stores as `1 + w` whose kernel reads `w` goes up less one; a plan that
    /// would carry it up as stored is refused before the upload, naming it, and so is a `1 + w`
    /// gamma returned to `w` for the attention prep, which reads it as stored. The roles with
    /// the convention today pass the same check.
    #[test]
    fn a_gamma_stored_as_one_plus_w_is_refused_when_its_upload_skips_the_convention() {
        let model = small_model();
        let resident = |_: usize| true;
        let mut planned = plan(&model, &resident);
        check_plan(&model, &planned, &resident).expect("every convention is applied");
        // The beta of layer 0 as if its file stored `1 + w`: its kernel reads `w`.
        let beta = planned
            .iter()
            .position(|tensor| {
                tensor.entry.role == Slot::Layer(Wanted::DeltaBeta) && tensor.entry.layer == Some(0)
            })
            .expect("the beta of layer 0");
        planned[beta].entry.convention = Convention::GammaOnePlusW;
        let error = format!(
            "{:?}",
            check_plan(&model, &planned, &resident).expect_err("a convention not applied")
        );
        assert!(
            error.contains("blk.0.ssm_beta.weight")
                && error.contains("linear_attn.in_proj_b.weight"),
            "{error}"
        );
        // The query norm of layer 3 returned to `w`: its prep reads `1 + w`.
        planned[beta].entry.convention = Convention::AsStored;
        let q_norm = planned
            .iter()
            .position(|tensor| {
                tensor.entry.role == Slot::Layer(Wanted::QNorm) && tensor.entry.layer == Some(3)
            })
            .expect("the query norm of layer 3");
        planned[q_norm].how = Upload::Float { minus_one: true };
        let error = format!(
            "{:?}",
            check_plan(&model, &planned, &resident).expect_err("a convention not applied")
        );
        assert!(error.contains("self_attn.q_norm.weight"), "{error}");
    }

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
