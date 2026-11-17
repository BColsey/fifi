//! Tests for the raon1 opener. A tiny spec with a distinct value in every field
//! exercises every rejection path on synthetic files; an always-run golden table
//! pins the real spec to the published Q4_K_M file role by role; the env-gated
//! test opens the real artifact.

use std::path::PathBuf;

use super::*;

const ALIGN: usize = 32;

/// All 18 scalar fields mutually distinct (streams and the derived mix count of 17
/// and projection width of 480 included), so a generator that reads the wrong field
/// fails some assertion. Block rules kept: 32 for Q8_0 and Q5_0 rows (the projection
/// width 480, q_lora 288, k_head 64, moe_inter 352), 256 for Q4_K and Q6_K rows
/// (hidden 512, kv_latent 768, dense_inter 1024).
fn tiny_spec() -> Raon1Spec {
    Raon1Spec {
        layers: 3,
        leading_dense: 1,
        hidden: 512,
        vocab: 640,
        heads: 5,
        head_dim: 96,
        q_lora: 288,
        key_mla: 160,
        kv_latent: 768,
        kv_a_width: 832,
        k_head: 64,
        v_head: 80,
        time_step_rank: 208,
        conv_kernel: 9,
        experts: 6,
        moe_inter: 352,
        dense_inter: 1024,
        streams: 4,
        mla_layers: &[2],
        down_q5_0_layers: &[1],
    }
}

/// A unique temporary file path per call, so parallel tests never share one.
fn temp_path(tag: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "raon1-{tag}-{:x}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ))
}

fn string(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(text.as_bytes());
    out
}

fn kv_string(key: &str, value: &str) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&8u32.to_le_bytes()); // string
    out.extend_from_slice(&string(value));
    out
}

fn kv_u32(key: &str, value: u32) -> Vec<u8> {
    let mut out = string(key);
    out.extend_from_slice(&4u32.to_le_bytes()); // u32
    out.extend_from_slice(&value.to_le_bytes());
    out
}

/// The pinned metadata of one spec, as the converter writes it (u32 keys).
fn metadata(spec: Raon1Spec, architecture: &str) -> Vec<Vec<u8>> {
    let mut kvs = vec![kv_string("general.architecture", architecture)];
    let pairs: [(&str, u64); 20] = [
        ("raon1.block_count", spec.layers as u64),
        ("raon1.embedding_length", spec.hidden),
        ("raon1.vocab_size", spec.vocab),
        ("raon1.attention.head_count", spec.heads),
        ("raon1.attention.q_lora_rank", spec.q_lora),
        ("raon1.attention.kv_lora_rank", spec.kv_latent),
        ("raon1.attention.key_length", spec.kv_a_width),
        ("raon1.attention.key_length_mla", spec.key_mla),
        ("raon1.attention.value_length", spec.kv_latent),
        ("raon1.attention.value_length_mla", spec.v_head),
        ("raon1.kda.head_dim", spec.head_dim),
        ("raon1.ssm.conv_kernel", spec.conv_kernel),
        ("raon1.ssm.time_step_rank", spec.time_step_rank),
        ("raon1.expert_count", spec.experts),
        ("raon1.expert_used_count", 6),
        ("raon1.expert_shared_count", 1),
        ("raon1.expert_feed_forward_length", spec.moe_inter),
        ("raon1.feed_forward_length", spec.dense_inter),
        ("raon1.leading_dense_block_count", spec.leading_dense as u64),
        ("raon1.hyper_connection.count", spec.streams),
    ];
    kvs.extend(pairs.iter().map(|(key, value)| kv_u32(key, *value as u32)));
    kvs
}

/// The 20 pinned keys, for loops that corrupt one at a time.
const PINNED_KEYS: [&str; 20] = [
    "raon1.block_count",
    "raon1.embedding_length",
    "raon1.vocab_size",
    "raon1.attention.head_count",
    "raon1.attention.q_lora_rank",
    "raon1.attention.kv_lora_rank",
    "raon1.attention.key_length",
    "raon1.attention.key_length_mla",
    "raon1.attention.value_length",
    "raon1.attention.value_length_mla",
    "raon1.kda.head_dim",
    "raon1.ssm.conv_kernel",
    "raon1.ssm.time_step_rank",
    "raon1.expert_count",
    "raon1.expert_used_count",
    "raon1.expert_shared_count",
    "raon1.expert_feed_forward_length",
    "raon1.feed_forward_length",
    "raon1.leading_dense_block_count",
    "raon1.hyper_connection.count",
];

#[derive(Clone)]
struct FileTensor {
    name: String,
    ty: GgufType,
    dims: Vec<u64>,
}

impl FileTensor {
    fn byte_len(&self) -> usize {
        let elements = self.dims.iter().product::<u64>();
        let blocks = elements / self.ty.block_elems();
        (blocks * self.ty.block_bytes()) as usize
    }
}

/// The file tensors of a spec, in inventory order, down-expert types read from
/// the spec's own layer table.
fn file_tensors(spec: Raon1Spec) -> Vec<FileTensor> {
    spec.inventory()
        .into_iter()
        .map(|expected| FileTensor {
            name: expected.name,
            ty: expected.ty,
            dims: expected.dims,
        })
        .collect()
}

/// One deterministic byte per position; the content is dequantizable junk, only the
/// block structure must be valid.
fn gguf_file(kvs: &[Vec<u8>], tensors: &[FileTensor]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for kv in kvs {
        out.extend_from_slice(kv);
    }
    let mut offsets = Vec::new();
    let mut offset = 0u64;
    for tensor in tensors {
        offsets.push(offset);
        offset += tensor.byte_len() as u64;
        offset = offset.div_ceil(ALIGN as u64) * ALIGN as u64;
    }
    for (tensor, offset) in tensors.iter().zip(&offsets) {
        out.extend_from_slice(&string(&tensor.name));
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ty.id().to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }
    while !out.len().is_multiple_of(ALIGN) {
        out.push(0);
    }
    for tensor in tensors {
        let start = out.len();
        let len = tensor.byte_len();
        out.resize(start + len, 0);
        for (index, byte) in out[start..].iter_mut().enumerate() {
            // bytes stay <= 0x3E so no F16/F32 halfword lands in the NaN/Inf
            // exponent range while still varying across every quant block
            *byte = (index % 63) as u8;
        }
        while !out.len().is_multiple_of(ALIGN) {
            out.push(0);
        }
    }
    out
}

fn open(kvs: &[Vec<u8>], tensors: &[FileTensor], spec: Raon1Spec) -> Result<(), EngError> {
    let path = temp_path("open");
    std::fs::write(&path, gguf_file(kvs, tensors)).unwrap();
    let result = open_with_spec(&path, None, spec).map(|model| {
        let _ = model;
    });
    let _ = std::fs::remove_file(&path);
    result
}

/// The error text of a rejection, or the panic that there was none.
fn rejection(result: Result<(), EngError>) -> String {
    match result {
        Ok(()) => panic!("the open must fail"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    }
}

fn replace(tensors: &mut [FileTensor], name: &str, ty: GgufType, dims: Vec<u64>) {
    let tensor = tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("no tensor {name} in the file"));
    tensor.ty = ty;
    tensor.dims = dims;
}

#[test]
fn the_tiny_spec_opens() {
    let spec = tiny_spec();
    open(&metadata(spec, "raon1"), &file_tensors(spec), spec).unwrap();
}

#[test]
fn a_truncated_file_fails() {
    let spec = tiny_spec();
    let file = gguf_file(&metadata(spec, "raon1"), &file_tensors(spec));
    let path = temp_path("truncated");
    std::fs::write(&path, &file[..file.len() / 2]).unwrap();
    let error = match open_with_spec(&path, None, spec) {
        Ok(_) => panic!("a truncated file must fail"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    let _ = std::fs::remove_file(&path);
    assert!(
        error.contains("beyond the end of the file") || error.contains("truncated"),
        "{error}"
    );
}

#[test]
fn a_duplicate_tensor_name_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    let first = tensors[1].clone();
    tensors.insert(1, first);
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("duplicate tensor name"), "{error}");
    assert!(error.contains(&tensors[1].name), "{error}");
}

#[test]
fn a_missing_architecture_key_fails() {
    let spec = tiny_spec();
    let mut kvs = metadata(spec, "raon1");
    kvs.remove(0);
    let error = rejection(open(&kvs, &file_tensors(spec), spec));
    assert!(error.contains("architecture"), "{error}");
}

#[test]
fn a_wrong_architecture_fails_naming_both() {
    let spec = tiny_spec();
    let error = rejection(open(&metadata(spec, "llama"), &file_tensors(spec), spec));
    assert!(error.contains("llama"), "{error}");
    assert!(error.contains("raon1"), "{error}");
}

#[test]
fn every_pinned_key_is_required_with_its_value() {
    let spec = tiny_spec();
    let tensors = file_tensors(spec);
    for key in PINNED_KEYS {
        let mut kvs = metadata(spec, "raon1");
        let pos = kvs
            .iter()
            .position(|kv| kv.starts_with(&string(key)))
            .unwrap_or_else(|| panic!("key {key} not written by the builder"));
        let old = kvs.remove(pos);
        // missing: named as missing
        let error = rejection(open(&kvs, &tensors, spec));
        assert!(
            error.contains(key) && error.contains("missing"),
            "{key}: {error}"
        );
        // wrong value: named with both values
        kvs.insert(pos, corrupt_u32(&old));
        let error = rejection(open(&kvs, &tensors, spec));
        assert!(error.contains(key), "{key}: {error}");
        assert!(error.contains("expected"), "{key}: {error}");
        kvs.remove(pos);
        kvs.insert(pos, old);
    }
}

/// One u32 metadata pair with its value incremented, so the open names the key.
fn corrupt_u32(kv: &[u8]) -> Vec<u8> {
    let mut out = kv.to_vec();
    let last = out.len();
    let value = u32::from_le_bytes(out[last - 4..].try_into().unwrap());
    out[last - 4..].copy_from_slice(&(value + 1).to_le_bytes());
    out
}

#[test]
fn a_non_integer_pinned_key_fails() {
    let spec = tiny_spec();
    let mut kvs = metadata(spec, "raon1");
    let pos = kvs
        .iter()
        .position(|kv| kv.starts_with(&string("raon1.block_count")))
        .unwrap();
    let old = kvs.remove(pos);
    let value = "three";
    let mut out = string("raon1.block_count");
    out.extend_from_slice(&8u32.to_le_bytes()); // string
    out.extend_from_slice(&string(value));
    kvs.insert(pos, out);
    let error = rejection(open(&kvs, &file_tensors(spec), spec));
    assert!(
        error.contains("raon1.block_count") && error.contains("not a u32 or u64 integer"),
        "{error}"
    );
    let _ = old;
}

#[test]
fn a_missing_tensor_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    let name = "blk.2.attn_v_b.weight";
    tensors.retain(|tensor| tensor.name != name);
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains(name) && error.contains("missing"), "{error}");
}

#[test]
fn an_extra_tensor_fails_naming_it() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    tensors.push(FileTensor {
        name: "blk.0.attn_k_b.weight".to_owned(), // MLA tensor on an RDA layer
        ty: GgufType::Q80,
        dims: vec![spec.k_head, spec.kv_latent, spec.heads],
    });
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.0.attn_k_b.weight") && error.contains("not in the raon1 inventory"),
        "{error}"
    );
}

#[test]
fn a_wrong_type_fails_naming_the_tensor_and_both_types() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "token_embd.weight",
        GgufType::F16,
        vec![spec.hidden, spec.vocab],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("token_embd.weight"), "{error}");
    assert!(error.contains("F16"), "{error}");
    assert!(error.contains("Q4_K"), "{error}");
}

#[test]
fn a_swapped_dim_pair_fails_naming_both_shapes() {
    let spec = tiny_spec();
    let mut tensors = file_tensors(spec);
    // attn_q_b is not square on the tiny spec (288 x 384), so the swap changes dims
    replace(
        &mut tensors,
        "blk.2.attn_q_b.weight",
        GgufType::Q80,
        vec![spec.heads * spec.key_mla, spec.q_lora],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(error.contains("blk.2.attn_q_b.weight"), "{error}");
    assert!(error.contains("has dims"), "{error}");
    assert!(error.contains("expected"), "{error}");
}

#[test]
fn a_down_expert_type_swap_fails_naming_the_layer_tensor() {
    let spec = tiny_spec();
    // layer 1 is Q5_0 in the spec's table; Q8_0 there must fail.
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "blk.1.ffn_down_exps.weight",
        GgufType::Q80,
        vec![spec.moe_inter, spec.hidden, spec.experts],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.1.ffn_down_exps.weight") && error.contains("expected Q5_0"),
        "{error}"
    );
    // layer 2 is Q8_0 in the spec's table; Q5_0 there must fail.
    let mut tensors = file_tensors(spec);
    replace(
        &mut tensors,
        "blk.2.ffn_down_exps.weight",
        GgufType::Q50,
        vec![spec.moe_inter, spec.hidden, spec.experts],
    );
    let error = rejection(open(&metadata(spec, "raon1"), &tensors, spec));
    assert!(
        error.contains("blk.2.ffn_down_exps.weight") && error.contains("expected Q8_0"),
        "{error}"
    );
}

#[test]
fn read_f32_of_an_unknown_name_fails() {
    let spec = tiny_spec();
    let path = temp_path("read-unknown");
    std::fs::write(
        &path,
        gguf_file(&metadata(spec, "raon1"), &file_tensors(spec)),
    )
    .unwrap();
    let model = open_with_spec(&path, None, spec).unwrap();
    let _ = std::fs::remove_file(&path);
    let error = match model.read_f32("blk.9.attn_norm.weight") {
        Ok(values) => panic!(
            "an unknown tensor must not read, got {} values",
            values.len()
        ),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    assert!(error.contains("blk.9.attn_norm.weight"), "{error}");
}

#[test]
fn read_f32_dequantizes_every_quantized_type_of_the_file() {
    let spec = tiny_spec();
    let path = temp_path("read-types");
    std::fs::write(
        &path,
        gguf_file(&metadata(spec, "raon1"), &file_tensors(spec)),
    )
    .unwrap();
    let model = open_with_spec(&path, None, spec).unwrap();
    let _ = std::fs::remove_file(&path);
    let cases: [(&str, GgufType, u64); 6] = [
        ("blk.0.attn_norm.weight", GgufType::F32, spec.hidden),
        (
            "blk.1.hc_attn_fn.weight",
            GgufType::F16,
            spec.streams * spec.hidden * spec.mix_count(),
        ),
        ("token_embd.weight", GgufType::Q4K, spec.hidden * spec.vocab),
        (
            "blk.1.attn_q.weight",
            GgufType::Q6K,
            spec.hidden * spec.proj_width(),
        ),
        (
            "blk.1.ffn_down_exps.weight",
            GgufType::Q50,
            spec.moe_inter * spec.hidden * spec.experts,
        ),
        (
            "blk.2.ffn_down_exps.weight",
            GgufType::Q80,
            spec.moe_inter * spec.hidden * spec.experts,
        ),
    ];
    for (name, want_ty, elements) in cases {
        let (found_ty, _) = model.weights.read_tensor(name).unwrap();
        assert_eq!(found_ty, want_ty, "{name}");
        let values = model.read_f32(name).unwrap();
        assert_eq!(values.len() as u64, elements, "{name}");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} dequantized to a non-finite value"
        );
    }
}

/// Dims of the roles whose fields alias on the real model, written as literals
/// from the field table — NOT from spec.inventory(). These are the roles at
/// stake: a generator that reads the wrong field (ssm_norm on time_step_rank
/// instead of head_dim) still opens its own generated file, and only these
/// hand-written rows catch it. RDA roles on layer 1, MLA roles on layer 2.
#[test]
fn the_tiny_spec_pins_hand_written_dims_for_the_aliasing_roles() {
    let spec = tiny_spec();
    let tensors = file_tensors(spec);
    let oracle: &[(&str, GgufType, &[u64])] = &[
        ("blk.1.ssm_norm.weight", GgufType::F32, &[96]),
        ("blk.1.ssm_f_a.weight", GgufType::F16, &[512, 208]),
        ("blk.1.ssm_f_b.weight", GgufType::F16, &[208, 480]),
        ("blk.1.ssm_a", GgufType::F32, &[5]),
        ("blk.1.hc_attn_fn.weight", GgufType::F16, &[2048, 17]),
        ("blk.1.hc_ffn_fn.weight", GgufType::F16, &[2048, 17]),
        ("blk.1.ffn_gate_inp.weight", GgufType::F32, &[512, 6]),
        ("blk.2.attn_q_b.weight", GgufType::Q80, &[288, 800]),
        ("blk.2.attn_k_b.weight", GgufType::Q80, &[64, 768, 5]),
        ("blk.2.attn_v_b.weight", GgufType::Q6K, &[768, 80, 5]),
    ];
    for (name, want_ty, want_dims) in oracle {
        let tensor = tensors
            .iter()
            .find(|tensor| &tensor.name == name)
            .unwrap_or_else(|| panic!("{name} missing from the inventory"));
        assert_eq!(&tensor.ty, want_ty, "{name}");
        assert_eq!(tensor.dims, *want_dims, "{name}");
    }
}

/// The tiny file's unpadded source-byte total, as a literal: any drift in a
/// role's type or dims moves it.
#[test]
fn the_tiny_spec_source_bytes_total_12_073_968() {
    let spec = tiny_spec();
    let total: u64 = file_tensors(spec)
        .iter()
        .map(|tensor| tensor.byte_len() as u64)
        .sum();
    assert_eq!(total, 12_073_968);
}

/// The layer index of a "blk.N." name, for scoped role counting.
fn layer_of(name: &str) -> usize {
    name.trim_start_matches("blk.")
        .split('.')
        .next()
        .and_then(|index| index.parse().ok())
        .unwrap_or(0)
}

/// The real spec, pinned role by role to the published Q4_K_M file. This runs
/// without the 19.7 GB artifact: a generator that reads the wrong field fails
/// here, not only in the env-gated test.
#[test]
fn the_real_spec_matches_the_published_q4_k_m_role_table() {
    let spec = RAON_1_MINI_PREVIEW;
    let inventory = spec.inventory();
    assert_eq!(inventory.len(), 731);

    let h = 2560u64;
    let w = 5120u64;
    let mix = 17u64;
    let vocab = 259_840u64;
    let rows: &[(&str, GgufType, &[u64], usize, &str)] = &[
        // (name suffix, type, dims, expected count, layer-kind scope)
        // global
        ("token_embd.weight", GgufType::Q4K, &[h, vocab], 1, "token"),
        ("output_norm.weight", GgufType::F32, &[h], 1, "token"),
        ("output.weight", GgufType::Q6K, &[h, vocab], 1, "token"),
        // every layer
        ("attn_norm.weight", GgufType::F32, &[h], 25, "every"),
        ("ffn_norm.weight", GgufType::F32, &[h], 25, "every"),
        ("hc_attn_base.weight", GgufType::F32, &[mix], 25, "every"),
        ("hc_ffn_base.weight", GgufType::F32, &[mix], 25, "every"),
        (
            "hc_attn_fn.weight",
            GgufType::F16,
            &[10240, mix],
            25,
            "every",
        ),
        (
            "hc_ffn_fn.weight",
            GgufType::F16,
            &[10240, mix],
            25,
            "every",
        ),
        ("hc_attn_scale.weight", GgufType::F32, &[4], 25, "every"),
        ("hc_ffn_scale.weight", GgufType::F32, &[4], 25, "every"),
        // RDA layers (18)
        ("attn_q.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_k.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_v.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("attn_output.weight", GgufType::Q80, &[w, h], 18, "rda"),
        ("ssm_a", GgufType::F32, &[40], 18, "rda"),
        ("ssm_beta.weight", GgufType::F16, &[h, 40], 18, "rda"),
        ("ssm_rda.weight", GgufType::F16, &[h, 40], 18, "rda"),
        ("ssm_conv1d_k.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_conv1d_q.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_conv1d_v.weight", GgufType::F32, &[4, 1, w], 18, "rda"),
        ("ssm_dt.bias", GgufType::F32, &[w], 18, "rda"),
        ("ssm_f_a.weight", GgufType::F16, &[h, 128], 18, "rda"),
        ("ssm_f_b.weight", GgufType::F16, &[128, w], 18, "rda"),
        ("ssm_g.weight", GgufType::Q6K, &[h, w], 18, "rda"),
        ("ssm_norm.weight", GgufType::F32, &[128], 18, "rda"),
        // MLA layers (7)
        ("attn_gate.weight", GgufType::Q80, &[h, w], 7, "mla"),
        ("attn_q_a.weight", GgufType::Q6K, &[h, 576], 7, "mla"),
        ("attn_q_a_norm.weight", GgufType::F32, &[576], 7, "mla"),
        ("attn_q_b.weight", GgufType::Q80, &[576, 7680], 7, "mla"),
        ("attn_kv_a_mqa.weight", GgufType::Q6K, &[h, 576], 7, "mla"),
        ("attn_kv_a_norm.weight", GgufType::F32, &[512], 7, "mla"),
        ("attn_k_b.weight", GgufType::Q80, &[128, 512, 40], 7, "mla"),
        ("attn_v_b.weight", GgufType::Q6K, &[512, 128, 40], 7, "mla"),
        ("attn_output.weight", GgufType::Q80, &[w, h], 7, "mla"),
        // MoE layers (24)
        ("ffn_gate_inp.weight", GgufType::F32, &[h, 128], 24, "moe"),
        (
            "ffn_gate_exps.weight",
            GgufType::Q4K,
            &[h, 1088, 128],
            24,
            "moe",
        ),
        (
            "ffn_up_exps.weight",
            GgufType::Q4K,
            &[h, 1088, 128],
            24,
            "moe",
        ),
        (
            "ffn_down_exps.weight",
            GgufType::Q50,
            &[1088, h, 128],
            12,
            "moe",
        ),
        (
            "ffn_down_exps.weight",
            GgufType::Q80,
            &[1088, h, 128],
            12,
            "moe",
        ),
        (
            "ffn_gate_shexp.weight",
            GgufType::Q6K,
            &[h, 1088],
            24,
            "moe",
        ),
        ("ffn_up_shexp.weight", GgufType::Q6K, &[h, 1088], 24, "moe"),
        (
            "ffn_down_shexp.weight",
            GgufType::Q80,
            &[1088, h],
            24,
            "moe",
        ),
        ("exp_probs_b.bias", GgufType::F32, &[128], 24, "moe"),
        // leading dense layer (1)
        ("ffn_gate.weight", GgufType::Q4K, &[h, 12_032], 1, "dense"),
        ("ffn_up.weight", GgufType::Q4K, &[h, 12_032], 1, "dense"),
        ("ffn_down.weight", GgufType::Q6K, &[12_032, h], 1, "dense"),
    ];

    let mut covered = 0usize;
    for (suffix, ty, dims, count, scope) in rows {
        let found = inventory
            .iter()
            .filter(|tensor| {
                if !tensor.name.ends_with(suffix)
                    || tensor.ty != *ty
                    || tensor.dims.as_slice() != *dims
                {
                    return false;
                }
                // scopes pick the layer kind, so roles shared by RDA and MLA (attn_output)
                // count separately instead of matching each other's layers
                match *scope {
                    "token" => !tensor.name.starts_with("blk."),
                    "every" => tensor.name.starts_with("blk."),
                    "rda" => {
                        tensor.name.starts_with("blk.")
                            && spec.kind(layer_of(&tensor.name)) == RaonKind::Rda
                    }
                    "mla" => {
                        tensor.name.starts_with("blk.")
                            && spec.kind(layer_of(&tensor.name)) == RaonKind::Mla
                    }
                    "moe" => {
                        tensor.name.starts_with("blk.")
                            && layer_of(&tensor.name) >= spec.leading_dense
                            && layer_of(&tensor.name) < spec.layers
                    }
                    "dense" => {
                        tensor.name.starts_with("blk.")
                            && layer_of(&tensor.name) < spec.leading_dense
                    }
                    other => unreachable!("unknown scope {other}"),
                }
            })
            .count();
        assert_eq!(
            found, *count,
            "role {suffix} as {ty:?} {dims:?} in scope {scope}"
        );
        covered += found;
    }

    assert_eq!(covered, inventory.len(), "every tensor is covered by a row");

    // the two down-expert layer lists, exactly
    let q5_0: Vec<usize> = (1..spec.layers)
        .filter(|&layer| spec.down_type(layer) == GgufType::Q50)
        .collect();
    assert_eq!(q5_0, vec![3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19]);
    let mla: Vec<usize> = (0..spec.layers)
        .filter(|&layer| spec.kind(layer) == RaonKind::Mla)
        .collect();
    assert_eq!(mla, vec![3, 7, 11, 15, 19, 23, 24]);

    // the byte total of the published conversion, from the table alone
    let total: u64 = inventory
        .iter()
        .map(|tensor| tensor.byte_len(tensor.ty).unwrap())
        .sum();
    assert_eq!(total, 19_674_383_528);
}

/// The published Q4_K_M file when RAON_Q4_KM points at it: the strict open of
/// the real 19,686,434,240-byte artifact. Skipped when the variable is unset.
#[test]
fn the_published_q4_k_m_file_opens_strictly() {
    let Ok(path) = std::env::var("RAON_Q4_KM") else {
        eprintln!("skipped: RAON_Q4_KM is unset");
        return;
    };
    let path = PathBuf::from(path);
    // SHA-256 over the first header_len bytes of the published conversion, checked
    // by the engine's hasher
    const HEADER_SHA: &str = "b8e287e3eae14db453bf8f13addace471cc2472324021369cd809dcac6cf9f17";
    let model = open_raon_1_mini_preview(&path, Some(HEADER_SHA))
        .unwrap_or_else(|error| panic!("the published file must open: {error:?}"));
    assert_eq!(model.weights.header().tensors().len(), 731);

    // one read per type present in the file, with length and finite checks
    let cases: [(&str, GgufType, usize); 6] = [
        ("blk.0.attn_norm.weight", GgufType::F32, 2560),
        ("blk.3.hc_attn_fn.weight", GgufType::F16, 10240 * 17),
        ("blk.0.ffn_gate.weight", GgufType::Q4K, 2560 * 12_032),
        ("blk.3.attn_q_a.weight", GgufType::Q6K, 2560 * 576),
        (
            "blk.3.ffn_down_exps.weight",
            GgufType::Q50,
            1088 * 2560 * 128,
        ),
        ("blk.3.attn_q_b.weight", GgufType::Q80, 576 * 7680),
    ];
    for (name, want_ty, elements) in cases {
        let (found_ty, _) = model.weights.read_tensor(name).unwrap();
        assert_eq!(found_ty, want_ty, "{name}");
        let values = model.read_f32(name).unwrap();
        assert_eq!(values.len(), elements, "{name}");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} dequantized to a non-finite value"
        );
    }
}

#[test]
fn a_wrong_header_hash_is_refused() {
    let Ok(path) = std::env::var("RAON_Q4_KM") else {
        eprintln!("skipped: RAON_Q4_KM is unset");
        return;
    };
    let error = match open_raon_1_mini_preview(Path::new(&path), Some(&"0".repeat(64))) {
        Ok(_) => panic!("a wrong header hash must be refused"),
        Err(EngError::Checkpoint(text)) => text,
        Err(other) => panic!("expected a Checkpoint rejection, got {other:?}"),
    };
    assert!(error.to_lowercase().contains("sha256"), "{error}");
}

/// The fork's frozen 42-token prompt (fixture-run1.log), the exact input the
/// `fixtures1` dumps were produced from.
const FIXTURE1_TOKENS: [u32; 42] = [
    166863, 54935, 7821, 44, 11456, 189096, 257706, 142907, 189286, 9305, 39154, 166294, 107408,
    51684, 46, 224037, 18624, 166295, 11234, 166471, 167757, 142907, 22233, 5407, 5066, 46, 254733,
    27335, 44, 4565, 46, 124319, 156431, 44, 235664, 73629, 177858, 9678, 444, 2452, 44, 3248,
];

/// The fork's top-3 final logits over the same prompt (fixtures1.top20).
const FIXTURE1_TOP3: [(u32, f32); 3] = [(44, 21.714_62), (58, 19.871_094), (369, 19.577_736)];

/// Reads a fixture file as little-endian f32 in ggml linear order.
fn read_fixture_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let bytewidth = 4; // little-endian f32
    bytes
        .chunks_exact(bytewidth)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Fixture provenance. The dumps came from the KRAFTON fork at commit
/// `fb895662`, its `examples/eval-callback` with the additive dump patch
/// (RAON_DUMP_DIR / RAON_DUMP_FILTER), run over the frozen 42-token prompt.
/// The fork commit is printed with the run.
const FIXTURE1_FORK_COMMIT: &str = "fb895662";

/// Reads the fixture env pair in one go: both unset skips the test; a
/// half-set pair fails loudly instead of silently skipping.
fn fixture_env() -> Option<(String, String)> {
    let model = std::env::var("RAON_Q4_KM");
    let fixtures = std::env::var("RAON_FIXTURES1");
    match (model, fixtures) {
        (Ok(m), Ok(f)) => Some((m, f)),
        (Err(_), Err(_)) => {
            eprintln!("skipped: RAON_Q4_KM and RAON_FIXTURES1 are unset");
            None
        }
        (m, f) => panic!(
            "fixture pair half-set: RAON_Q4_KM={m:?} RAON_FIXTURES1={f:?} \
             (both are needed to run the fixture tests)"
        ),
    }
}

/// The fork's own top-k selection, extracted from its biased-score dump by
/// the same descending, ties-to-lower-index rule the reference uses.
fn fork_selection(sel: &[f32], ne: usize, nt: usize, nu: usize) -> Vec<u8> {
    let mut ids = vec![0u8; nu * nt];
    for t in 0..nt {
        let mut idx: Vec<usize> = (0..ne).collect();
        idx.sort_by(|&a, &b| sel[b + t * ne].total_cmp(&sel[a + t * ne]));
        for (slot, &e) in idx.iter().enumerate().take(nu) {
            ids[slot + t * nu] = u8::try_from(e).expect("expert id fits u8");
        }
    }
    ids
}

/// The stages whose values are bounded near 1 (sigmoid/softmax outputs,
/// normalized routing weights, helmert comb entries, the per-head beta).
fn is_bound1(base: &str) -> bool {
    matches!(
        base,
        "hc_pre"
            | "hc_post"
            | "hc_comb"
            | "ffn_moe_probs"
            | "ffn_moe_probs_biased"
            | "ffn_moe_weights"
            | "ffn_moe_weights_norm"
            | "ffn_moe_weights_sum"
            | "ffn_moe_weights_sum_clamped"
            | "rda_beta"
    )
}

/// Tolerance for the UNFORCED whole-graph pass (stage A), as a fraction of
/// the stage's value range. The reference's own layer outputs feed the next
/// layer, so the fork-vs-reference f32/f64 accumulation difference rides the
/// fold amplification chain: measured worst 1.8e-2 of range (rda_g1-2) in
/// the pre-flip region, 1.2e-2 absolute on routing weights (layer 4). The
/// chain table sits above that; the tight value oracle for every layer is
/// stage B's `oracle_tolerance`.
fn chain_tolerance(base: &str) -> f32 {
    if is_bound1(base) { 3e-2 } else { 5e-2 }
}

/// Tolerance for the TEACHER-FORCED pass (stage B), ~10x above the worst
/// measured diff/range per class over all of fixtures1 (the `measure ...`
/// lines print when the test runs with the env pair set):
/// - bound-1 stages: worst 2.7e-4 of range (rda_beta); the class is 37x
///   above it and still <= ~1e-2 absolute, tight enough that a wrong
///   constant or a gamma/rho slip in the mHC fold fails.
/// - everything else: worst 6.3e-3 of range (ffn_moe_out); the class is 8x
///   above it. A wrong formula moves O(range).
fn oracle_tolerance(base: &str) -> f32 {
    if is_bound1(base) { 1e-2 } else { 5e-2 }
}

/// One teacher-forced stage compare: positional, finite, per-class tolerance.
fn compare_stage(where_: &str, name: &str, mine: &[f32], fork: &[f32]) {
    assert_eq!(mine.len(), fork.len(), "{where_} {name}: lengths differ");
    assert!(
        mine.iter().all(|v| v.is_finite()),
        "{where_} {name}: reference produced non-finite values"
    );
    let mut worst = 0.0f32;
    let mut worst_at = 0usize;
    for (i, (&a, &b)) in mine.iter().zip(fork.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > worst {
            worst = diff;
            worst_at = i;
        }
    }
    let range = fork.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let base = name.rsplit_once('-').map_or(name, |(b, _)| b);
    let tol = oracle_tolerance(base) * range;
    eprintln!(
        "measure {where_} {name}: worst/range {:+.3e} (tol {:.0e})",
        worst / range,
        tol
    );
    if std::env::var("RAON_MEASURE").is_ok() {
        return;
    }
    assert!(
        worst <= tol,
        "{where_} stage {name} diverged: abs diff {worst:.3e} vs tol {tol:.3e} \
         at [{worst_at}] (reference {}, fork {})",
        mine[worst_at],
        fork[worst_at],
    );
}

/// The published Q4_K_M file against the fork's frozen fixtures1 stage dumps,
/// in two stages.
///
/// Stage A runs the whole graph unforced: it pins the embedding, layer 0
/// completely, and layer 1 up to the known top-k boundary flip (t=35,
/// experts 73/124, selection gap 1.2e-3); after the flip the rest is
/// structure-only (names, lengths, finiteness).
///
/// Stage B teacher-forces: every layer runs from the fork's own boundary
/// dumps (`l_out-{n-1}` / `hc_attn_out-{n}`), and each MoE layer's top-k is
/// forced to the fork's own selection, so nothing can cascade and EVERY
/// layer — all seven MLA layers included — plus the output head is
/// value-compared. Skipped when RAON_Q4_KM or RAON_FIXTURES1 is unset.
#[test]
fn the_host_reference_matches_the_fork_fixtures() {
    let Some((model_path, fixtures)) = fixture_env() else {
        return;
    };
    eprintln!("fixtures1 provenance: fork commit {FIXTURE1_FORK_COMMIT}");
    const HEADER_SHA: &str = "b8e287e3eae14db453bf8f13addace471cc2472324021369cd809dcac6cf9f17";
    let model = open_raon_1_mini_preview(std::path::Path::new(&model_path), Some(HEADER_SHA))
        .unwrap_or_else(|error| panic!("the published file must open: {error:?}"));

    // The semantic fixture sequence: every dump except pure view/reshape nodes,
    // in manifest (graph) order.
    let mut files: Vec<(u64, std::path::PathBuf)> = std::fs::read_dir(&fixtures)
        .unwrap_or_else(|e| panic!("read {}: {e}", fixtures))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?.to_owned();
            if !name.ends_with(".f32") || name.contains("(view)") || name.contains("(reshaped)") {
                return None;
            }
            let number: u64 = name.split('-').next()?.parse().ok()?;
            Some((number, path))
        })
        .collect();
    files.sort_by_key(|(number, _)| *number);
    assert_eq!(files.len(), 877, "the semantic fixture count is pinned");

    let mut emitted: Vec<(String, Vec<f32>)> = Vec::new();
    // Set once the first top-k boundary flip is excused: after it, the flip's
    // moe_out perturbation (~0.15 of range) spreads through attention into
    // every later stage, so detailed parity degrades to a coarse sanity bar.
    // The flip evidence itself (five of six expert columns matching exactly)
    // is what pins the formulas; the cascade is chaotic, not wrong.
    let mut cascade = false;
    reference::forward(&model, &FIXTURE1_TOKENS, &mut |name, data| {
        emitted.push((name.to_owned(), data.to_vec()))
    })
    .unwrap_or_else(|error| panic!("the reference must run: {error:?}"));
    assert_eq!(emitted.len(), files.len(), "stage counts must agree");

    for (idx, ((name, mine), (_, path))) in emitted.iter().zip(files.iter()).enumerate() {
        let fixture_name = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .trim_start_matches(|c: char| c.is_ascii_digit() || c == '-')
            .trim_end_matches(".f32")
            .to_owned();
        assert_eq!(
            name, &fixture_name,
            "stage {idx}: name order diverged from the fork graph"
        );
        let fork = read_fixture_f32(path);
        assert_eq!(mine.len(), fork.len(), "stage {name}: lengths differ");
        // Once a selection flip cascaded, the fixture stream is no longer
        // reproducible by an independent-precision implementation: the flip's
        // ~0.15 moe_out perturbation rides the 260x fold amplification and
        // eventually overflows the scan's f32-representable envelope. Later
        // stages pin structure only (the name/length asserts above); the
        // engine's own Raon-1 implementation validates against this reference
        // directly, where the reference's own selections are consistent and
        // the stream stays finite.
        if cascade {
            continue;
        }
        assert!(
            mine.iter().all(|v| v.is_finite()),
            "stage {name}: reference produced non-finite values"
        );
        // The reference's dots accumulate F64 where the fork accumulates
        // F32, so per-stage diffs are that accumulation noise; cancellation-
        // heavy fold chains amplify it (see `chain_tolerance`). A wrong
        // formula moves values by O(range); noise does not.
        let range = fork.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        let base = name.rsplit_once('-').map_or(&name[..], |(b, _)| b);
        let tol = chain_tolerance(base) * range;
        // Slot-ordered MoE dumps pair each token's six expert columns with the
        // top-k ORDER, and that order is chaotic at near-ties: the fork ranks
        // on its own F32 logits, the oracle on F64 ones, and selection gaps
        // as small as 1.2e-3 (layer 1, t=35, experts 73/124) flip slot pairs
        // while the emitted weights and the order-independent moe_out stay
        // within tolerance. Those stages compare per token after greedy slot
        // matching instead of positionally.
        let base: &str = name.rsplit_once('-').map_or(name, |(b, _)| b);
        // Selection-dependent stages: the routing weights and normalized
        // forms are 6-wide per token (the greedy matching below treats them
        // as feat=1 vectors), the sums are one value per token (a 1-wide
        // frame), and the per-expert slot dumps are feat-wide. All depend on
        // the biased top-k SET, chaotic at near-ties (the fork ranks on its
        // f32 scores, this reference on f64 ones); over-tolerance tokens are
        // suspects excused only via the moe_out verification below.
        let selection_dependent = matches!(
            base,
            "ffn_moe_weights"
                | "ffn_moe_weights_norm"
                | "ffn_moe_weights_sum"
                | "ffn_moe_weights_sum_clamped"
                | "ffn_moe_gate"
                | "ffn_moe_up"
                | "ffn_moe_situ"
                | "ffn_moe_down"
                | "ffn_moe_weighted"
        );
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        // (stage name, token, per-token matched cost) for tokens whose match
        // exceeds tolerance: boundary flips of the top-k SET.
        let mut suspects: Vec<(&str, usize, f32)> = Vec::new();
        if selection_dependent {
            const NT: usize = 42;
            let nu = if matches!(base, "ffn_moe_weights_sum" | "ffn_moe_weights_sum_clamped") {
                1
            } else {
                6
            };
            let feat = fork.len() / (nu * NT);
            assert_eq!(fork.len(), feat * nu * NT, "stage {name} layout");
            for t in 0..NT {
                let mut pairs: Vec<(f32, usize, usize)> = Vec::with_capacity(nu * nu);
                for i in 0..nu {
                    for j in 0..nu {
                        let mut c = 0.0f32;
                        for f in 0..feat {
                            let a = mine[f + i * feat + t * feat * nu];
                            let b = fork[f + j * feat + t * feat * nu];
                            c = c.max((a - b).abs());
                        }
                        pairs.push((c, i, j));
                    }
                }
                pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
                let mut mine_taken = vec![false; nu];
                let mut fork_taken = vec![false; nu];
                let mut token_worst = 0.0f32;
                let mut bad_pairs = 0usize;
                for &(c, i, j) in &pairs {
                    if mine_taken[i] || fork_taken[j] {
                        continue;
                    }
                    mine_taken[i] = true;
                    fork_taken[j] = true;
                    token_worst = token_worst.max(c);
                    if c > tol {
                        bad_pairs += 1;
                    }
                    if mine_taken.iter().filter(|&&v| v).count() == nu {
                        break;
                    }
                }
                // A boundary flip breaks exactly one expert pair; anything
                // else is a real divergence and must fail here.
                assert!(
                    bad_pairs <= 1,
                    "stage {name} t={t}: {bad_pairs} slot pairs over tol, \
                     (token_worst {token_worst:.3e}) - not a top-k boundary flip"
                );
                if bad_pairs == 1 {
                    suspects.push((name, t, token_worst));
                } else {
                    worst = worst.max(token_worst);
                }
            }
            for (suspect_stage, t, cost) in &suspects {
                assert!(
                    suspects.len() <= 4,
                    "more than four excused flips in one stage smells systemic"
                );
                let layer = suspect_stage.rsplit_once('-').expect("layer suffix").1;
                // Gap evidence: the excuse is legitimate only for a genuine
                // near-tie. The fork's own biased scores at this token must
                // put the 6th and 7th ranked experts within 5e-3 (the two
                // measured flips sit at 9.8e-4 and 2.4e-3; every non-flip
                // margin in these columns is >= 3e-2, so the bound is 2x
                // above the largest flip and 6x below the smallest margin);
                // a real routing divergence moves the boundary by O(0.1).
                let biased_name = format!("ffn_moe_probs_biased-{layer}");
                let fork_biased = files
                    .iter()
                    .find(|(_, p)| {
                        p.file_name()
                            .and_then(|f| f.to_str())
                            .is_some_and(|f| f.ends_with(&format!("-{biased_name}.f32")))
                    })
                    .map(|(_, p)| read_fixture_f32(p))
                    .unwrap_or_else(|| panic!("fixture for {biased_name} missing"));
                let ne = fork_biased.len() / 42;
                let mine_biased = emitted
                    .iter()
                    .find(|(n, _)| n == &biased_name)
                    .map(|(_, d)| d)
                    .unwrap_or_else(|| panic!("{biased_name} missing for suspect check"));
                // Rank both sides' biased scores for this token, descending;
                // expert ids are the positions in the ranked index order.
                let rank_desc = |scores: &[f32]| {
                    let mut idx: Vec<usize> = (0..ne).collect();
                    idx.sort_by(|&a, &b| scores[b + t * ne].total_cmp(&scores[a + t * ne]));
                    idx
                };
                let fork_rank = rank_desc(&fork_biased);
                let mine_rank = rank_desc(mine_biased);
                let gap = fork_biased[fork_rank[5] + t * ne] - fork_biased[fork_rank[6] + t * ne];
                assert!(
                    gap < 5e-3,
                    "slot suspect {suspect_stage} t={t} has no near-tie: \
                     the fork's biased-score gap at rank 6/7 is {gap:.3e}"
                );
                // Identity: the swap must be exactly the fork's 7th expert
                // in place of its 6th, and ranks 0..=4 must be untouched on
                // both sides. A near tie at 6/7 with any other set change is
                // a real divergence hiding behind the tie.
                assert!(
                    mine_rank[5] == fork_rank[6]
                        && mine_rank[6] == fork_rank[5]
                        && mine_rank[..5] == fork_rank[..5],
                    "slot suspect {suspect_stage} t={t} changed the wrong experts: \
                     the reference ranks {} and {} where the fork ranks {} and {}",
                    mine_rank[5],
                    mine_rank[6],
                    fork_rank[5],
                    fork_rank[6]
                );
                let out_name = format!("ffn_moe_out-{layer}");
                let mine_out = emitted
                    .iter()
                    .find(|(n, _)| n == &out_name)
                    .map(|(_, d)| d)
                    .unwrap_or_else(|| panic!("{out_name} missing for suspect check"));
                let fork_out = files
                    .iter()
                    .find(|(_, p)| {
                        p.file_name()
                            .and_then(|f| f.to_str())
                            .is_some_and(|f| f.ends_with(&format!("-{out_name}.f32")))
                    })
                    .map(|(_, p)| read_fixture_f32(p))
                    .unwrap_or_else(|| panic!("fixture for {out_name} missing"));
                let h = fork_out.len() / 42;
                let mut oworst = 0.0f32;
                for k in 0..h {
                    let d = (mine_out[k + t * h] - fork_out[k + t * h]).abs();
                    oworst = oworst.max(d);
                }
                let orange = 1.0
                    + fork_out[t * h..(t + 1) * h]
                        .iter()
                        .fold(0.0f32, |m, v| m.max(v.abs()));
                // A single expert swap moves this token's moe_out by up to
                // ~0.15 of range on this fixture (measured); 0.3 admits that
                // swap while still rejecting a wrong formula, which moves
                // values by O(range) across every token.
                assert!(
                    oworst <= 3e-1 * orange,
                    "slot suspect {suspect_stage} t={t} (cost {cost:.3e}) is a real \
                     divergence: moe_out diff {oworst:.3e} vs {}",
                    3e-1 * orange
                );
                eprintln!(
                    "excused selection flip {suspect_stage} t={t} cost {cost:.3e} \
                     (moe_out diff {oworst:.3e}, fork rank-6/7 score gap {gap:.3e})"
                );
            }
            if !suspects.is_empty() {
                cascade = true;
            }
        } else {
            for (i, (&a, &b)) in mine.iter().zip(fork.iter()).enumerate() {
                let diff = (a - b).abs();
                if diff > worst {
                    worst = diff;
                    worst_at = i;
                }
            }
        }
        if idx % 100 == 0 {
            eprintln!("stage {idx} {name}: worst/range {:+.3e}", worst / range);
        }
        if std::env::var("RAON_MEASURE").is_ok() {
            continue;
        }
        assert!(
            worst <= tol,
            "stage {name} diverged: abs diff {worst:.3e} vs tol {tol:.3e} at [{worst_at}] \
             (reference {}, fork {})",
            mine[worst_at],
            fork[worst_at],
        );
    }

    // Stage B: teacher-forced, every layer and the head value-compared. The
    // fixture cursor walks the same ordered dump list stage A proved matches
    // the reference's emission order; files[0] is embd, which stage B feeds
    // directly instead of emitting.
    let spec = model.spec();
    let hy =
        reference::hyper(&model, spec).unwrap_or_else(|error| panic!("hyper must read: {error:?}"));
    let h = spec.hidden as usize;
    let hc = hy.streams;
    let nt = FIXTURE1_TOKENS.len();
    let fixture = |name: &str| -> Vec<f32> {
        files
            .iter()
            .find(|(_, p)| {
                p.file_name()
                    .and_then(|f| f.to_str())
                    .is_some_and(|f| f.ends_with(&format!("-{name}.f32")))
            })
            .map(|(_, p)| read_fixture_f32(p))
            .unwrap_or_else(|| panic!("fixture {name} missing"))
    };
    let embd = fixture("embd");
    let mut cursor = 1usize;
    for layer in 0..spec.layers {
        let inp: Vec<f32> = if layer == 0 {
            let mut v = vec![0.0f32; h * hc * nt];
            for t in 0..nt {
                for s in 0..hc {
                    for k in 0..h {
                        v[k + s * h + t * h * hc] = embd[k + t * h];
                    }
                }
            }
            v
        } else {
            fixture(&format!("l_out-{}", layer - 1))
        };
        let mut emitted_attn: Vec<(String, Vec<f32>)> = Vec::new();
        reference::attn_half(&model, spec, &hy, &inp, layer, &mut |name, data| {
            emitted_attn.push((name.to_owned(), data.to_vec()))
        })
        .unwrap_or_else(|error| panic!("attn half {layer} must run: {error:?}"));
        for (name, mine) in &emitted_attn {
            let (_, path) = &files[cursor];
            cursor += 1;
            let fork = read_fixture_f32(path);
            compare_stage(&format!("attn half {layer}"), name, mine, &fork);
        }

        let inp = fixture(&format!("hc_attn_out-{layer}"));
        let forced = if layer >= spec.leading_dense {
            let sel = fixture(&format!("ffn_moe_probs_biased-{layer}"));
            Some(fork_selection(
                &sel,
                spec.experts as usize,
                nt,
                hy.used_experts,
            ))
        } else {
            None
        };
        let mut emitted_ffn: Vec<(String, Vec<f32>)> = Vec::new();
        reference::ffn_half(
            &model,
            spec,
            &hy,
            &inp,
            layer,
            &mut |name, data| emitted_ffn.push((name.to_owned(), data.to_vec())),
            forced.as_deref(),
        )
        .unwrap_or_else(|error| panic!("ffn half {layer} must run: {error:?}"));
        for (name, mine) in &emitted_ffn {
            let (_, path) = &files[cursor];
            cursor += 1;
            let fork = read_fixture_f32(path);
            compare_stage(&format!("ffn half {layer}"), name, mine, &fork);
        }
    }

    let last = fixture(&format!("l_out-{}", spec.layers - 1));
    let mut emitted_head: Vec<(String, Vec<f32>)> = Vec::new();
    let logits = reference::head(&model, spec, &hy, &last, nt, &mut |name, data| {
        emitted_head.push((name.to_owned(), data.to_vec()))
    })
    .unwrap_or_else(|error| panic!("head must run: {error:?}"));
    for (name, mine) in &emitted_head {
        let (_, path) = &files[cursor];
        cursor += 1;
        let fork = read_fixture_f32(path);
        compare_stage("head", name, mine, &fork);
    }
    assert_eq!(
        cursor,
        files.len(),
        "the teacher-forced walk covered every dump"
    );
    for &(tok, value) in FIXTURE1_TOP3.iter() {
        // Absolute 5e-2 on logits of range ~22 (top-3 span): the teacher-
        // forced layers already match per stage, so this absorbs only the
        // f64-vs-f32 head drift; a wrong head misranks the top-3 by O(1).
        assert!(
            (logits[tok as usize] - value).abs() <= 5e-2,
            "teacher-forced logit {tok}: {} vs {value}",
            logits[tok as usize]
        );
    }
}

/// `f16_roundtrip` against f16-exact and subnormal values (the Q8_0 scale
/// rounding the fork's quantizer applies).
#[test]
fn f16_roundtrip_pins_known_values() {
    assert_eq!(reference::f16_roundtrip(1.0), 1.0);
    assert_eq!(reference::f16_roundtrip(0.1), 0.099_975_586);
    assert_eq!(reference::f16_roundtrip(-2.5), -2.5);
    assert_eq!(reference::f16_roundtrip(1e-5), 1.001358e-5); // subnormal
    assert_eq!(reference::f16_roundtrip(0.0), 0.0);
}

/// The layer-1 MoE expert-124 gate column against the fork's slot dump
/// (`00074-ffn_moe_gate-1.f32`, token 35, slot 5): measured reference-vs-
/// fork worst diff ~2e-2 absolute on the 1088-wide column, which pins the
/// engine-side expert slice + quantized dot in isolation.
#[test]
fn the_moe_expert_gate_column_pins_the_fork_slot() {
    let Some((model_path, fixtures)) = fixture_env() else {
        return;
    };
    const HEADER_SHA: &str = "b8e287e3eae14db453bf8f13addace471cc2472324021369cd809dcac6cf9f17";
    let model = open_raon_1_mini_preview(std::path::Path::new(&model_path), Some(HEADER_SHA))
        .unwrap_or_else(|error| panic!("the published file must open: {error:?}"));
    // layer 1 geometry: hidden 2560, moe_inter 1088, 128 experts, 6 used, 42 tokens.
    let (gate_w, gate_ty) = model
        .read_f32_typed("blk.1.ffn_gate_exps.weight")
        .expect("exps gate");
    let (h, mi, e, t, u) = (2560usize, 1088usize, 124usize, 35usize, 5usize);
    let cur = read_fixture_f32(
        std::path::Path::new(&fixtures)
            .join("00062-ffn_norm-1.f32")
            .as_path(),
    );
    let xt = &cur[t * h..(t + 1) * h];
    let span = h * mi;
    let g = reference::fork_mv(&gate_w[e * span..][..span], gate_ty, xt, h, mi);
    let gate_fix = read_fixture_f32(
        std::path::Path::new(&fixtures)
            .join("00074-ffn_moe_gate-1.f32")
            .as_path(),
    );
    let mut worst = 0.0f32;
    for f in 0..mi {
        let fork = gate_fix[f + u * mi + t * mi * 6];
        worst = worst.max((g[f] - fork).abs());
    }
    eprintln!("expert-124 gate col worst diff: {worst:.3e}");
    assert!(worst <= 5e-2, "expert gate column drifted: {worst}");
}

/// One dense-FFN gate element recomputed against the fork's Q8_K-quantized
/// dot, pinning `fork_mm`'s numerics in isolation from the whole graph.
/// Measured on `00027-ffn_norm-0.f32` token 6, output row 1893: exact F64
/// 15.113298, Q8_K-quantized 15.101180, fork dump 15.101177.
#[test]
fn the_dense_ffn_gate_element_pins_the_quantized_dot() {
    let Some((model_path, fixtures)) = fixture_env() else {
        return;
    };
    const HEADER_SHA: &str = "b8e287e3eae14db453bf8f13addace471cc2472324021369cd809dcac6cf9f17";
    let model = open_raon_1_mini_preview(std::path::Path::new(&model_path), Some(HEADER_SHA))
        .unwrap_or_else(|error| panic!("the published file must open: {error:?}"));
    let cur = read_fixture_f32(
        std::path::Path::new(&fixtures)
            .join("00027-ffn_norm-0.f32")
            .as_path(),
    );
    // blk.0 geometry: hidden 2560 in, dense_inter 12032 out, 42 tokens.
    let gate = reference::fork_mm(&model, "blk.0.ffn_gate.weight", &cur, 2560, 12032, 42)
        .unwrap_or_else(|error| panic!("the mm must run: {error:?}"));
    let at = 74085; // o=1893, t=6 in [n, nt] layout
    assert_eq!(gate.len(), 12032 * 42);
    assert!(
        (gate[at] - 15.101_18).abs() <= 5e-3,
        "quantized gate element: {} vs fork 15.101177",
        gate[at]
    );
}
