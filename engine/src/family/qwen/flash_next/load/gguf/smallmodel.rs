//! A small `qwen4exp` GGUF with random weights, for the checks that compare the device with the
//! host: the head geometry the kernels fix (256-value attention heads, 128-value indexer and
//! DeltaNet heads, 64 rotary values) with few heads, four layers (three DeltaNet, one QSA, the
//! PLE block in layer 1), 16 experts of which 10 are picked, and every tensor in a type the CUDA
//! kernels read (Q8_0 matrices, F32 vectors, BF16 for the indexer).

use std::path::PathBuf;

use super::{
    config::GgufConfig,
    tensors::Role,
    testkit::{LIMITS, TensorSpec, kv_f32, kv_str, kv_u32, kv_u32s, kv_u64s, write_gguf},
};
use crate::weights::{GgufType, GgufWeights};

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
    let (offsets, vocab, _) = head_ranges();
    vec![
        kv_str("general.architecture", "qwen4exp"),
        kv_u32("qwen4exp.block_count", LAYERS as u32),
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
        kv_u32("qwen4exp.attention.indexer.top_k", 8),
        kv_u32s("qwen4exp.attention.compress_ratios", &[0, 0, 0, 4]),
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

/// What the stored values of a role look like.
fn values_for(role: Role, count: usize, rng: &mut Lcg) -> Vec<f32> {
    let mut draw = |scale: f32, shift: f32| -> Vec<f32> {
        (0..count).map(|_| shift + scale * rng.unit()).collect()
    };
    match role {
        // 1 + w gammas.
        Role::HyperNorm(_)
        | Role::QsaQNorm
        | Role::QsaKNorm
        | Role::QsaIndexQNorm
        | Role::QsaIndexKNorm
        | Role::PleNormKey
        | Role::PleNormQuery
        | Role::PleNormConv => draw(0.2, 1.0),
        // -exp(A_log).
        Role::DeltaDecay => draw(0.5, -1.0),
        Role::DeltaDtBias => draw(0.3, 0.0),
        Role::DeltaNorm => draw(0.2, 1.0),
        Role::DeltaConv | Role::PleConv => draw(0.4, 0.0),
        Role::Router => draw(0.3, 0.0),
        Role::SharedScalar => draw(0.3, 0.0),
        Role::HyperInject(_) => draw(0.2, 0.0),
        Role::NgramTable => draw(0.5, 0.0),
        Role::Embedding => draw(0.5, 0.0),
        Role::Head => draw(0.3, 0.0),
        Role::HyperDown(_) | Role::HyperUp(_) => draw(0.15, 0.0),
        Role::QsaIndexQ | Role::QsaIndexK => draw(0.1, 0.0),
        Role::ExpertGate | Role::ExpertUp | Role::ExpertDown => draw(0.12, 0.0),
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
        other => panic!("the small model has no encoder for {other:?}"),
    }
}

/// The small model written to a fresh directory (one file), with its first path.
pub(super) struct Small {
    pub(super) dir: tempfile::TempDir,
    pub(super) first: PathBuf,
}

pub(super) fn write_small(seed: u64) -> Small {
    let dir = tempfile::tempdir().expect("tempdir");
    let kvs = small_kvs();
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
        if item.role == Role::NgramTable {
            dims[1] = rows;
        }
        if item.role == Role::Embedding || item.role == Role::Head {
            dims[1] = VOCAB as u64;
        }
        let ty = match item.role {
            Role::NgramTable => GgufType::F16,
            Role::QsaIndexQ | Role::QsaIndexK => GgufType::Bf16,
            Role::HyperInject(_) => GgufType::F32,
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
