//! A small `qwen4exp` GGUF with random weights, for the checks that compare the device with the
//! host: the head geometry the kernels fix (256-value attention heads, 128-value indexer and
//! DeltaNet heads, 64 rotary values) with few heads, four layers (three DeltaNet, one QSA, the
//! PLE block in layer 1), 16 experts of which 10 are picked, and every tensor in a type the CUDA
//! kernels read (Q8_0 matrices, F32 vectors, BF16 for the indexer).

use std::path::PathBuf;

use super::{
    config::GgufConfig,
    tensors::Slot,
    testkit::{LIMITS, TensorSpec, kv_f32, kv_str, kv_u32, kv_u32s, kv_u64s, write_gguf},
};
use crate::{
    family::recipe::{Global, Role as Wanted, SiteTensor},
    weights::{GgufType, GgufWeights},
};

pub(super) const HIDDEN: usize = 256;
pub(super) const VOCAB: usize = 512;
pub(super) const LAYERS: usize = 4;
pub(super) const EXPERTS: usize = 16;
pub(super) const TOP_K: usize = 10;
pub(super) const INTER: usize = 128;
const PLE_HEAD_DIM: usize = 16;
const PRIMES: [u64; 16] = [
    61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
];

fn head_ranges() -> (Vec<u64>, Vec<u64>, u64) {
    let mut offsets = Vec::new();
    let mut at = 0;
    for prime in PRIMES {
        offsets.push(at);
        at += prime;
    }
    (offsets, PRIMES.to_vec(), at)
}

/// The metadata of the small model.
pub(super) fn small_kvs() -> Vec<Vec<u8>> {
    small_kvs_for(LAYERS, 8)
}

/// The metadata of a small model of `layers` layers (a QSA layer closes every group of four)
/// whose indexer keeps `top_k` positions.
pub(super) fn small_kvs_for(layers: usize, top_k: u32) -> Vec<Vec<u8>> {
    let (offsets, vocab, _) = head_ranges();
    vec![
        kv_str("general.architecture", "qwen4exp"),
        kv_u32("qwen4exp.block_count", layers as u32),
        kv_u32("qwen4exp.context_length", 4096),
        kv_u32("qwen4exp.embedding_length", HIDDEN as u32),
        kv_u32("qwen4exp.attention.head_count", 2),
        kv_u32("qwen4exp.attention.head_count_kv", 1),
        kv_u32s("qwen4exp.rope.dimension_sections", &[11, 11, 10, 0]),
        kv_f32("qwen4exp.rope.freq_base", 10_000_000.0),
        kv_f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32("qwen4exp.expert_count", EXPERTS as u32),
        kv_u32("qwen4exp.expert_used_count", TOP_K as u32),
        kv_u32("qwen4exp.attention.key_length", 256),
        kv_u32("qwen4exp.attention.value_length", 256),
        kv_u32("qwen4exp.expert_feed_forward_length", INTER as u32),
        kv_u32("qwen4exp.expert_shared_feed_forward_length", INTER as u32),
        kv_u32("qwen4exp.ssm.conv_kernel", 4),
        kv_u32("qwen4exp.ssm.state_size", 128),
        kv_u32("qwen4exp.ssm.group_count", 2),
        kv_u32("qwen4exp.ssm.time_step_rank", 4),
        kv_u32("qwen4exp.ssm.inner_size", 512),
        kv_u32("qwen4exp.full_attention_interval", 4),
        kv_u32("qwen4exp.rope.dimension_count", 64),
        kv_u32("qwen4exp.hyper_connection.count", 4),
        kv_u32("qwen4exp.hyper_connection.low_rank", 32),
        kv_u32("qwen4exp.attention.indexer.head_count", 2),
        kv_u32("qwen4exp.attention.indexer.key_length", 128),
        kv_u32("qwen4exp.attention.indexer.top_k", top_k),
        kv_u32s(
            "qwen4exp.attention.compress_ratios",
            &(0..layers)
                .map(|layer| if layer % 4 == 3 { 4 } else { 0 })
                .collect::<Vec<u32>>(),
        ),
        kv_u32s("qwen4exp.ple.layers", &[1]),
        kv_u32("qwen4exp.ple.ngram_size", 3),
        kv_u32("qwen4exp.ple.heads_per_ngram", 8),
        kv_u32("qwen4exp.ple.conv_kernel", 4),
        kv_u32("qwen4exp.ple.eos_token_id", 3),
        kv_u32(
            "qwen4exp.embedding_length_per_layer_input",
            PLE_HEAD_DIM as u32,
        ),
        kv_u64s(
            "qwen4exp.ple.layer_multipliers",
            &[1_000_003, 7_654_321, 98_765_431],
        ),
        kv_u64s("qwen4exp.ple.head_offsets", &offsets),
        kv_u64s("qwen4exp.ple.head_vocab_sizes", &vocab),
    ]
}

/// A deterministic generator of small floats.
pub(super) struct Lcg(pub(super) u64);

impl Lcg {
    pub(super) fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn half_bits(value: f32) -> u16 {
    // Round to nearest even from f32.
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let m = (mantissa | 0x80_0000) >> (1 - exponent);
        let rounded = (m + 0x0fff + ((m >> 13) & 1)) >> 13;
        return sign | rounded as u16;
    }
    let rounded = (mantissa + 0x0fff + ((mantissa >> 13) & 1)) >> 13;
    sign | (((exponent as u32) << 10) + rounded) as u16
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// Quantizes `values` (a whole number of 32-value blocks) to Q8_0: a half scale `amax / 127`
/// and 32 signed bytes per block.
fn q8_0(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 32 * 34);
    for block in values.as_chunks::<32>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&half_bits(d).to_le_bytes());
        out.extend(
            block
                .iter()
                .map(|v| (v * id).round().clamp(-127.0, 127.0) as i8 as u8),
        );
    }
    out
}

/// Quantizes `values` (a whole number of 256-value blocks) to Q6_K with one scale a block: the
/// half is `amax / 31`, the sixteen sub-scales are 1, and a quant is the value over the half,
/// rounded and held to -32..=31. The device and the host read the same bytes, so the block's
/// precision is not what a device test compares.
fn q6_k(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() / 256 * 210);
    for block in values.as_chunks::<256>().0 {
        let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
        let half = if amax == 0.0 { 1.0 } else { amax / 31.0 };
        let (mut low, mut high) = ([0u8; 128], [0u8; 64]);
        for (i, &value) in block.iter().enumerate() {
            let quant = (value / half).round().clamp(-32.0, 31.0) as i32 + 32;
            let (side, quarter, lane) = (i / 128, (i % 128) / 32, i % 32);
            let nibble = (quant & 0xF) as u8;
            let at = 64 * side + lane + 32 * (quarter & 1);
            if quarter < 2 {
                low[at] |= nibble;
            } else {
                low[at] |= nibble << 4;
            }
            high[32 * side + lane] |= (((quant >> 4) & 3) as u8) << (2 * quarter);
        }
        out.extend_from_slice(&low);
        out.extend_from_slice(&high);
        out.extend_from_slice(&[1u8; 16]);
        out.extend_from_slice(&half_bits(half).to_le_bytes());
    }
    out
}

/// What the stored values of a role look like.
fn values_for(role: Slot, count: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut draw = |scale: f32, shift: f32| -> Vec<f32> {
        (0..count).map(|_| shift + scale * rng.unit()).collect()
    };
    use SiteTensor as Part;
    match role {
        // 1 + w gammas.
        Slot::Layer(Wanted::Hyper(_, Part::Norm))
        | Slot::Global(Global::HeadSite(Part::Norm))
        | Slot::Layer(
            Wanted::QNorm
            | Wanted::KNorm
            | Wanted::IndexQNorm
            | Wanted::IndexKNorm
            | Wanted::PleNormKey
            | Wanted::PleNormQuery
            | Wanted::PleNormConv,
        ) => draw(0.2, 1.0),
        // -exp(A_log).
        Slot::Layer(Wanted::DeltaDecay) => draw(0.5, -1.0),
        Slot::Layer(Wanted::DeltaDtBias) => draw(0.3, 0.0),
        Slot::Layer(Wanted::DeltaNorm) => draw(0.2, 1.0),
        Slot::Layer(Wanted::DeltaConv | Wanted::PleConv) => draw(0.4, 0.0),
        Slot::Layer(Wanted::Router | Wanted::SharedRouterGate) => draw(0.3, 0.0),
        Slot::Layer(Wanted::Hyper(_, Part::Inject)) => draw(0.2, 0.0),
        Slot::Global(Global::NgramTable | Global::Embedding) => draw(0.5, 0.0),
        Slot::Global(Global::Head) => draw(0.3, 0.0),
        Slot::Layer(Wanted::Hyper(_, Part::Down | Part::Up))
        | Slot::Global(Global::HeadSite(Part::Down | Part::Up)) => draw(0.15, 0.0),
        Slot::Layer(Wanted::IndexProj | Wanted::IndexKey) => draw(0.1, 0.0),
        Slot::Layer(Wanted::ExpertGate | Wanted::ExpertUp | Wanted::ExpertDown) => draw(0.12, 0.0),
        _ => draw(0.08, 0.0),
    }
}

fn encode(ty: GgufType, values: &[f32]) -> Vec<u8> {
    match ty {
        GgufType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgufType::F16 => values
            .iter()
            .flat_map(|&v| half_bits(v).to_le_bytes())
            .collect(),
        GgufType::Bf16 => values
            .iter()
            .flat_map(|&v| bf16_bits(v).to_le_bytes())
            .collect(),
        GgufType::Q80 => q8_0(values),
        GgufType::Q6K => q6_k(values),
        other => panic!("the small model has no encoder for {other:?}"),
    }
}

/// The small model written to a fresh directory (one file), with its first path.
pub(in crate::family::qwen::flash_next) struct Small {
    pub(in crate::family::qwen::flash_next) dir: tempfile::TempDir,
    pub(in crate::family::qwen::flash_next) first: PathBuf,
}

pub(in crate::family::qwen::flash_next) fn write_small(seed: u64) -> Small {
    write_small_with(seed, LAYERS, 8)
}

/// [`write_small`] with `layers` layers and an indexer that keeps `top_k` positions.
pub(super) fn write_small_with(seed: u64, layers: usize, top_k: u32) -> Small {
    write_small_as(seed, layers, top_k, GgufType::Q80)
}

/// [`write_small`] with the DeltaNet out-projection matrices stored as `out_proj`: the Q8_0 of
/// every other matrix, or a K type, blocks of 256 values over heads of 128.
pub(super) fn write_small_out_proj(seed: u64, out_proj: GgufType) -> Small {
    write_small_as(seed, LAYERS, 8, out_proj)
}

fn write_small_as(seed: u64, layers: usize, top_k: u32, out_proj: GgufType) -> Small {
    let dir = tempfile::tempdir().expect("tempdir");
    let kvs = small_kvs_for(layers, top_k);
    let config = {
        let probe = dir.path().join("probe.gguf");
        let embed = TensorSpec {
            name: "token_embd.weight".into(),
            dims: vec![HIDDEN as u64, VOCAB as u64],
            ty: GgufType::Q80,
            payload: None,
        };
        write_gguf(&probe, &kvs, &[embed]);
        let weights = GgufWeights::open_any_type(&probe, LIMITS, None).expect("probe");
        let config = GgufConfig::read(&weights).expect("config");
        std::fs::remove_file(&probe).expect("remove probe");
        config
    };
    let mut rng = Lcg(seed);
    let (_, _, rows) = head_ranges();
    let mut tensors = Vec::new();
    for item in super::tensors::inventory(&config) {
        let mut dims = item.dims.clone();
        if item.role == Slot::Global(Global::NgramTable) {
            dims[1] = rows;
        }
        if item.role == Slot::Global(Global::Embedding) || item.role == Slot::Global(Global::Head) {
            dims[1] = VOCAB as u64;
        }
        let ty = match item.role {
            Slot::Layer(Wanted::DeltaOut) => out_proj,
            Slot::Global(Global::NgramTable) => GgufType::F16,
            Slot::Layer(Wanted::IndexProj | Wanted::IndexKey) => GgufType::Bf16,
            Slot::Layer(Wanted::Hyper(_, SiteTensor::Inject)) => GgufType::F32,
            _ if item.vector => GgufType::F32,
            _ => GgufType::Q80,
        };
        let count: usize = dims.iter().product::<u64>() as usize;
        let values = values_for(item.role, count, &mut rng);
        tensors.push(TensorSpec {
            name: item.name,
            dims,
            ty,
            payload: Some(encode(ty, &values)),
        });
    }
    let path = dir.path().join("small.gguf");
    write_gguf(&path, &kvs, &tensors);
    Small { dir, first: path }
}

mod tests {
    use super::*;

    /// A block of the kit's Q6_K decodes, by the CPU oracle the host forward reads, to its
    /// values within half a quant's step (and the half's own rounding).
    #[test]
    fn a_q6_k_block_decodes_to_its_values_within_half_a_step() {
        let values: Vec<f32> = (0..512)
            .map(|i| ((i * 37) % 101) as f32 / 250.0 - 0.2)
            .collect();
        let bytes = q6_k(&values);
        assert_eq!(bytes.len(), 2 * 210);
        let decoded = crate::weights::dequantize(GgufType::Q6K, &bytes).expect("decode Q6_K");
        for (block, want) in decoded.chunks(256).zip(values.chunks(256)) {
            let amax = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = amax / 31.0;
            for (&got, &value) in block.iter().zip(want) {
                assert!(
                    (got - value).abs() <= 0.6 * step,
                    "{value} decodes to {got}, step {step}"
                );
            }
        }
    }
}
