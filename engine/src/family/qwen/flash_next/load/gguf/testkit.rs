//! Synthetic Flash-Next GGUF files for the tests: the metadata and the tensor table of the
//! real model at full size, written sparse (a tensor of 28 GB costs a hole in the file), and
//! optionally split over several shard files in the layout split models are published in (shard 1 holds the
//! metadata and no tensors).

use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::{
    config::{GgufConfig, NgramHash},
    tensors::{Convention, Form, Mapped, Place, Role},
};
use crate::weights::{GgufLimits, GgufType, GgufWeights};

const ALIGN: u64 = 32;

pub(super) fn string(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

pub(super) fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

pub(super) fn kv_f32(key: &str, value: f32) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&6u32.to_le_bytes());
    out.extend_from_slice(&value.to_le_bytes());
    out
}

pub(super) fn kv_str(key: &str, value: &str) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&string(value));
    out
}

/// An array of unsigned 32-bit values (GGUF element type 4).
pub(super) fn kv_u32s(key: &str, values: &[u32]) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// An array of unsigned 64-bit values (GGUF element type 10).
pub(super) fn kv_u64s(key: &str, values: &[u64]) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&9u32.to_le_bytes());
    out.extend_from_slice(&10u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// One tensor to write. A `None` payload is a hole of the right size.
pub(super) struct TensorSpec {
    pub(super) name: String,
    pub(super) dims: Vec<u64>,
    pub(super) ty: GgufType,
    pub(super) payload: Option<Vec<u8>>,
}

impl TensorSpec {
    pub(super) fn bytes(&self) -> u64 {
        let elements: u64 = self.dims.iter().product();
        elements / self.ty.block_elems() * self.ty.block_bytes()
    }
}

/// Writes one GGUF file; returns its length.
pub(super) fn write_gguf(path: &Path, kvs: &[Vec<u8>], tensors: &[TensorSpec]) -> u64 {
    let mut head = b"GGUF".to_vec();
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        head.extend_from_slice(kv);
    }
    let mut offsets = Vec::new();
    let mut at = 0u64;
    for tensor in tensors {
        head.extend_from_slice(&string(&tensor.name));
        head.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            head.extend_from_slice(&dim.to_le_bytes());
        }
        head.extend_from_slice(&tensor.ty.id().to_le_bytes());
        head.extend_from_slice(&at.to_le_bytes());
        offsets.push(at);
        at = (at + tensor.bytes()).div_ceil(ALIGN) * ALIGN;
    }
    let data_start = (head.len() as u64).div_ceil(ALIGN) * ALIGN;
    head.resize(data_start as usize, 0);
    let file = File::create(path).expect("create gguf");
    file.write_all_at(&head, 0).expect("write header");
    for (tensor, offset) in tensors.iter().zip(&offsets) {
        if let Some(payload) = &tensor.payload {
            assert_eq!(payload.len() as u64, tensor.bytes(), "{}", tensor.name);
            file.write_all_at(payload, data_start + offset)
                .expect("write payload");
        }
    }
    let len = data_start + at;
    file.set_len(len).expect("size the file");
    len
}

pub(super) const LIMITS: GgufLimits = GgufLimits {
    max_header_bytes: 1 << 26,
    max_tensor_bytes: 1 << 36,
};

/// The routed-expert types of each layer of the 3-bit mixed-quant file: gate and up share a type,
/// down has its own.
pub(super) const REAL_EXPERT_TYPES: [(GgufType, GgufType); 48] = {
    use GgufType::{Iq2S as S, Iq3Xxs as X, Iq4Nl as N, Iq4Xs as XS, Q20 as Q};
    [
        (S, N),
        (S, N),
        (S, N),
        (S, N),
        (S, N),
        (S, N),
        (S, N),
        (S, N),
        (S, Q),
        (S, N),
        (S, Q),
        (S, N),
        (S, N),
        (S, Q),
        (X, N),
        (S, N),
        (X, N),
        (X, N),
        (X, N),
        (S, N),
        (X, N),
        (S, N),
        (X, N),
        (S, N),
        (S, N),
        (X, N),
        (S, N),
        (X, N),
        (X, N),
        (X, N),
        (X, N),
        (X, N),
        (S, N),
        (X, N),
        (S, N),
        (X, N),
        (S, N),
        (X, N),
        (X, N),
        (S, N),
        (X, N),
        (X, N),
        (X, N),
        (S, N),
        (S, N),
        (X, N),
        (X, N),
        (XS, N),
    ]
};

/// What to write.
#[derive(Clone)]
pub(super) struct Spec {
    /// Shard files to split the tensors over (one writes a single file).
    pub(super) shards: usize,
    /// Per-layer routed-expert types: gate and up, then down.
    pub(super) expert_types: [(GgufType, GgufType); 48],
    /// Tensors to leave out.
    pub(super) omit: Vec<String>,
    /// Tensors to write with other dimensions.
    pub(super) reshape: Vec<(String, Vec<u64>)>,
    /// Tensors to write with another type.
    pub(super) retype: Vec<(String, GgufType)>,
    /// Tensors no role claims.
    pub(super) extra: Vec<String>,
    /// Rows of the n-gram table.
    pub(super) ngram_rows: u64,
    /// Whether the file carries its own output matrix.
    pub(super) output: bool,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            shards: 1,
            expert_types: REAL_EXPERT_TYPES,
            omit: Vec::new(),
            reshape: Vec::new(),
            retype: Vec::new(),
            extra: Vec::new(),
            ngram_rows: 320_001_536,
            output: true,
        }
    }
}

/// The head offsets and vocabulary sizes of the real file's 16 n-gram heads.
pub(super) fn head_ranges() -> (Vec<u64>, Vec<u64>) {
    let primes: [u64; 16] = [
        20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063, 20_000_069,
        20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147, 20_000_153, 20_000_159,
        20_000_161, 20_000_171,
    ];
    let mut offsets = Vec::new();
    let mut at = 0;
    for prime in primes {
        offsets.push(at);
        at += prime;
    }
    (offsets, primes.to_vec())
}

/// The metadata of the real model, without the tokenizer arrays.
pub(super) fn flash_kvs() -> Vec<Vec<u8>> {
    let (offsets, vocab) = head_ranges();
    let ratios: Vec<u32> = (0..48).map(|l| if l % 4 == 3 { 4 } else { 0 }).collect();
    vec![
        kv_str("general.architecture", "qwen4exp"),
        kv_u32("qwen4exp.block_count", 48),
        kv_u32("qwen4exp.context_length", 262_144),
        kv_u32("qwen4exp.embedding_length", 2560),
        kv_u32("qwen4exp.attention.head_count", 24),
        kv_u32("qwen4exp.attention.head_count_kv", 2),
        kv_u32s("qwen4exp.rope.dimension_sections", &[11, 11, 10, 0]),
        kv_f32("qwen4exp.rope.freq_base", 10_000_000.0),
        kv_f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32("qwen4exp.expert_count", 512),
        kv_u32("qwen4exp.expert_used_count", 10),
        kv_u32("qwen4exp.attention.key_length", 256),
        kv_u32("qwen4exp.attention.value_length", 256),
        kv_u32("qwen4exp.expert_feed_forward_length", 640),
        kv_u32("qwen4exp.expert_shared_feed_forward_length", 640),
        kv_u32("qwen4exp.ssm.conv_kernel", 4),
        kv_u32("qwen4exp.ssm.state_size", 128),
        kv_u32("qwen4exp.ssm.group_count", 16),
        kv_u32("qwen4exp.ssm.time_step_rank", 48),
        kv_u32("qwen4exp.ssm.inner_size", 6144),
        kv_u32("qwen4exp.full_attention_interval", 4),
        kv_u32("qwen4exp.rope.dimension_count", 64),
        kv_u32("qwen4exp.hyper_connection.count", 4),
        kv_u32("qwen4exp.hyper_connection.low_rank", 320),
        kv_u32("qwen4exp.attention.indexer.head_count", 4),
        kv_u32("qwen4exp.attention.indexer.key_length", 128),
        kv_u32("qwen4exp.attention.indexer.top_k", 2048),
        kv_u32s("qwen4exp.attention.compress_ratios", &ratios),
        kv_u32s("qwen4exp.ple.layers", &[1]),
        kv_u32("qwen4exp.ple.ngram_size", 3),
        kv_u32("qwen4exp.ple.heads_per_ngram", 8),
        kv_u32("qwen4exp.ple.conv_kernel", 4),
        kv_u32("qwen4exp.ple.eos_token_id", 248_044),
        kv_u32("qwen4exp.embedding_length_per_layer_input", 160),
        kv_u64s(
            "qwen4exp.ple.layer_multipliers",
            &[23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071],
        ),
        kv_u64s("qwen4exp.ple.head_offsets", &offsets),
        kv_u64s("qwen4exp.ple.head_vocab_sizes", &vocab),
    ]
}

fn split_kvs(shard: usize, shards: usize, total: usize) -> Vec<Vec<u8>> {
    if shards <= 1 {
        return Vec::new();
    }
    let u16kv = |key: &str, value: u16| {
        let mut out = string(key);
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
        out
    };
    let _ = shard;
    vec![
        u16kv("split.no", shard as u16),
        u16kv("split.count", shards as u16),
        {
            let mut out = string("split.tensors.count");
            out.extend_from_slice(&5u32.to_le_bytes());
            out.extend_from_slice(&(total as i32).to_le_bytes());
            out
        },
    ]
}

/// The whole tensor list of the model for `spec`, in the order the executor reads them.
pub(super) fn flash_tensors(spec: &Spec, config: &GgufConfig) -> Vec<TensorSpec> {
    let mut out = Vec::new();
    for item in super::tensors::inventory(config) {
        let name = item.name;
        if spec.omit.contains(&name) || (!spec.output && name == "output.weight") {
            continue;
        }
        let mut dims = item.dims;
        let mut ty = match item.role {
            Role::ExpertGate | Role::ExpertUp => spec.expert_types[item.layer.unwrap()].0,
            Role::ExpertDown => spec.expert_types[item.layer.unwrap()].1,
            Role::HyperDown(_)
            | Role::HyperUp(_)
            | Role::PleKey
            | Role::PleValue
            | Role::SharedDown => GgufType::Q80,
            Role::HyperInject(_) => GgufType::F32,
            Role::QsaIndexQ | Role::QsaIndexK => GgufType::Bf16,
            Role::NgramTable => GgufType::Iq4Nl,
            _ if item.vector => GgufType::F32,
            _ => GgufType::Q6K,
        };
        if item.role == Role::NgramTable {
            dims[1] = spec.ngram_rows;
        }
        if let Some((_, other)) = spec.reshape.iter().find(|(n, _)| *n == name) {
            dims = other.clone();
        }
        if let Some((_, other)) = spec.retype.iter().find(|(n, _)| *n == name) {
            ty = *other;
        }
        out.push(TensorSpec {
            name,
            dims,
            ty,
            payload: None,
        });
    }
    for name in &spec.extra {
        out.push(TensorSpec {
            name: name.clone(),
            dims: vec![64],
            ty: GgufType::F32,
            payload: None,
        });
    }
    out
}

/// A written model: the directory, the path of shard 1 and the open checkpoint.
pub(super) struct Written {
    pub(super) dir: tempfile::TempDir,
    pub(super) first: PathBuf,
}

impl Written {
    pub(super) fn open(&self) -> GgufWeights {
        GgufWeights::open_any_type(&self.first, LIMITS, None).expect("open the synthetic model")
    }

    /// Writes `bytes` at `offset` into the data of tensor `name`, in whichever shard holds it.
    pub(super) fn patch(&self, name: &str, offset: u64, bytes: &[u8]) {
        let weights = self.open();
        let tensor = weights.header().tensor(name).expect("tensor to patch");
        assert!(offset + bytes.len() as u64 <= tensor.byte_len());
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&weights.paths()[tensor.shard()])
            .expect("open the shard for writing");
        file.write_all_at(bytes, tensor.abs_offset() + offset)
            .expect("patch");
    }
}

/// Writes the model of `spec` into a fresh directory. With several shards, shard 1 holds the
/// metadata and no tensors and the rest share the tensors in table order.
pub(super) fn write_flash(spec: &Spec) -> Written {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = {
        // The shapes come from the metadata alone, so a header-only file gives the config.
        let probe = dir.path().join("probe.gguf");
        let kvs = flash_kvs();
        let embed = TensorSpec {
            name: "token_embd.weight".into(),
            dims: vec![2560, 248_320],
            ty: GgufType::Q6K,
            payload: None,
        };
        write_gguf(&probe, &kvs, &[embed]);
        let weights = GgufWeights::open_any_type(&probe, LIMITS, None).expect("probe");
        let config = GgufConfig::read(&weights).expect("config");
        std::fs::remove_file(&probe).expect("remove probe");
        config
    };
    let tensors = flash_tensors(spec, &config);
    let total = tensors.len();
    let first = if spec.shards <= 1 {
        let path = dir.path().join("flash.gguf");
        write_gguf(&path, &flash_kvs(), &tensors);
        path
    } else {
        let per_shard = total.div_ceil(spec.shards - 1);
        let mut rest = tensors;
        let name = |n: usize| {
            dir.path()
                .join(format!("flash-{n:05}-of-{:05}.gguf", spec.shards))
        };
        let mut kvs = flash_kvs();
        kvs.extend(split_kvs(0, spec.shards, total));
        write_gguf(&name(1), &kvs, &[]);
        for shard in 1..spec.shards {
            let take = per_shard.min(rest.len());
            let part: Vec<TensorSpec> = rest.drain(..take).collect();
            write_gguf(
                &name(shard + 1),
                &split_kvs(shard, spec.shards, total),
                &part,
            );
        }
        name(1)
    };
    Written { dir, first }
}

/// A one-tensor file holding only an n-gram table of `rows` rows of 160 values in `ty`, and
/// the mapped tensor and hash constants that describe it.
pub(super) struct TinyTable {
    pub(super) _dir: tempfile::TempDir,
    pub(super) weights: GgufWeights,
    pub(super) table: Mapped,
    pub(super) hash: NgramHash,
}

pub(super) fn tiny_table(ty: GgufType, rows: u64, payload: Vec<u8>) -> TinyTable {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("table.gguf");
    let tensor = TensorSpec {
        name: "per_layer_token_embd.weight".into(),
        dims: vec![160, rows],
        ty,
        payload: Some(payload),
    };
    let bytes = tensor.bytes();
    write_gguf(
        &path,
        &[kv_str("general.architecture", "qwen4exp")],
        &[tensor],
    );
    let weights = GgufWeights::open_any_type(&path, LIMITS, None).expect("open the table");
    let table = Mapped {
        role: Role::NgramTable,
        layer: None,
        name: "per_layer_token_embd.weight".into(),
        ty,
        dims: vec![160, rows],
        bytes,
        shard: 0,
        place: Place::Ngram,
        form: Form::None,
        convention: Convention::AsStored,
    };
    TinyTable {
        _dir: dir,
        weights,
        table,
        hash: NgramHash {
            multipliers: vec![1, 2, 3],
            offsets: vec![0],
            vocab: vec![rows as i64],
        },
    }
}
