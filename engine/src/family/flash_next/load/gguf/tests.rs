use super::{
    GgufModel,
    config::{ARCHITECTURE, GgufConfig},
    plan::Card,
    tensors::{KernelGap, Place, Role, TensorMap},
    testkit::{REAL_EXPERT_TYPES, Spec, flash_kvs, write_flash, write_gguf},
};
use crate::{
    EngError,
    family::flash_next::host_plan::HostMemory,
    family::flash_next::load::{Dims, tests::CONFIG},
    weights::GgufType,
};

const GIB: u64 = 1 << 30;

fn open(spec: &Spec) -> (super::testkit::Written, GgufModel) {
    let written = write_flash(spec);
    let model = GgufModel::open(&written.first)
        .expect("open")
        .expect("a qwen4exp model");
    (written, model)
}

/// The EXL3 configuration of the same model, as the pack path reads it.
fn exl3_dims() -> Dims {
    let mut dims = Dims::read(CONFIG).expect("the fixture config");
    // The pack splits its n-gram table over parts; the GGUF holds one tensor.
    dims.ngram_parts = None;
    dims
}

#[test]
fn the_metadata_gives_the_dimensions_the_exl3_configuration_gives() {
    let written = write_flash(&Spec::default());
    let weights = written.open();
    let config = GgufConfig::read(&weights).expect("config");
    assert_eq!(config.dims, exl3_dims());
    assert_eq!(config.layers(), 48);
    assert_eq!(config.ple_layer(), 1);
    assert_eq!(config.max_positions, 262_144);
    assert_eq!(config.rope_sections, [11, 11, 10, 0]);
    assert_eq!(
        config.hash.multipliers,
        [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071]
    );
    assert_eq!(config.hash.offsets.len(), 16);
    assert_eq!(config.hash.offsets[1], 20_000_003);
    assert_eq!(config.hash.vocab[0], 20_000_003);
    assert_eq!(config.ngram_rows_needed, 320_001_446);
    config
        .model_shape()
        .expect("the shape constructors accept it");
}

#[test]
fn every_tensor_of_the_real_layout_is_mapped_with_its_place_and_bytes() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    assert_eq!(map.total(), 1224, "tensors written");
    assert_eq!(map.mapped(), map.total());
    assert!(map.unmapped().is_empty(), "{:?}", map.unmapped());
    // The expert bytes the type table implies: 1,638,400 weights per matrix per expert.
    let per_matrix = |ty: GgufType| 640u64 * 2560 / ty.block_elems() * ty.block_bytes() * 512;
    let experts: u64 = REAL_EXPERT_TYPES
        .iter()
        .map(|&(gate_up, down)| 2 * per_matrix(gate_up) + per_matrix(down))
        .sum();
    assert_eq!(map.file_bytes(Place::Experts), experts);
    assert_eq!(model.expert_bytes(), experts);
    assert_eq!(map.file_bytes(Place::Ngram), 320_001_536 * 90);
    assert_eq!(
        map.get(Role::Embedding, None).expect("embedding").bytes,
        521_472_000
    );
    // Layer 1 holds the PLE block; no other layer does.
    assert!(map.get(Role::PleKey, Some(1)).is_some());
    assert!(map.get(Role::PleKey, Some(5)).is_none());
    // Layer 3 attends, layer 0 runs DeltaNet.
    assert!(map.get(Role::QsaQ, Some(3)).is_some() && map.get(Role::DeltaQkv, Some(3)).is_none());
    assert!(map.get(Role::DeltaQkv, Some(0)).is_some() && map.get(Role::QsaQ, Some(0)).is_none());
    // The mixer has no inject projection; every layer site does.
    assert!(
        map.get(Role::HyperInject(super::tensors::Site::Mixer), None)
            .is_none()
    );
    assert!(
        map.get(Role::HyperInject(super::tensors::Site::Attn), Some(7))
            .is_some()
    );
}

#[test]
fn the_device_trunk_counts_each_tensor_in_the_form_the_executor_holds_it() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    let at = |role: Role, layer: Option<usize>| map.get(role, layer).expect("mapped");
    // Quantized matrices stay as stored blocks.
    let qkv = at(Role::DeltaQkv, Some(0));
    assert_eq!(qkv.device_bytes(), qkv.bytes);
    // Hyper-connection matrices are widened to half precision: 10240 x 320 elements.
    assert_eq!(
        at(Role::HyperDown(super::tensors::Site::Attn), Some(0)).device_bytes(),
        10240 * 320 * 2
    );
    // The router is F32 as stored.
    assert_eq!(at(Role::Router, Some(0)).device_bytes(), 2560 * 512 * 4);
    // The embedding and the experts are not on the card.
    assert_eq!(at(Role::Embedding, None).device_bytes(), 0);
    assert_eq!(at(Role::ExpertGate, Some(0)).device_bytes(), 0);
    let trunk = map.device_bytes();
    assert!(trunk > 2 * GIB && trunk < 6 * GIB, "trunk {trunk}");
}

#[test]
fn a_model_without_its_own_output_matrix_uses_the_embedding() {
    let (_written, model) = open(&Spec {
        output: false,
        ..Spec::default()
    });
    let head = model.map().get(Role::Head, None).expect("head");
    assert_eq!(head.name, "token_embd.weight");
    assert_eq!(model.map().mapped(), model.map().total());
}

fn refusal(spec: &Spec) -> String {
    let written = write_flash(spec);
    match GgufModel::open(&written.first) {
        Err(error) => error.describe(),
        Ok(_) => panic!("expected a refusal"),
    }
}

#[test]
fn a_missing_tensor_is_named() {
    let text = refusal(&Spec {
        omit: vec!["blk.5.ffn_up_exps.weight".into()],
        ..Spec::default()
    });
    assert!(
        text.contains("blk.5.ffn_up_exps.weight is missing"),
        "{text}"
    );
    assert!(text.contains("1 of the model's"), "{text}");
}

#[test]
fn a_tensor_of_another_shape_is_named_with_both_shapes() {
    let text = refusal(&Spec {
        reshape: vec![("blk.3.attn_k.weight".into(), vec![2560, 256])],
        ..Spec::default()
    });
    assert!(
        text.contains(
            "blk.3.attn_k.weight has dimensions [2560, 256], the model needs [2560, 512]"
        ),
        "{text}"
    );
}

#[test]
fn a_vector_stored_as_a_quantized_type_is_refused() {
    let text = refusal(&Spec {
        retype: vec![("blk.0.ffn_gate_inp.weight".into(), GgufType::Q80)],
        ..Spec::default()
    });
    assert!(text.contains("blk.0.ffn_gate_inp.weight is Q8_0"), "{text}");
}

#[test]
fn more_than_five_problems_are_counted_and_five_are_listed() {
    let omit: Vec<String> = (0..8)
        .map(|l| format!("blk.{l}.ffn_gate_inp.weight"))
        .collect();
    let text = refusal(&Spec {
        omit,
        ..Spec::default()
    });
    assert!(text.contains("8 of the model's"), "{text}");
    assert_eq!(text.matches("is missing").count(), 5, "{text}");
}

#[test]
fn a_tensor_no_role_claims_is_reported_and_not_refused() {
    let (_written, model) = open(&Spec {
        extra: vec!["mtp.0.eh_proj.weight".into()],
        ..Spec::default()
    });
    assert_eq!(model.map().unmapped(), ["mtp.0.eh_proj.weight"]);
    assert_eq!(model.map().mapped() + 1, model.map().total());
    assert!(
        model.summary().contains("1 unmapped"),
        "{}",
        model.summary()
    );
}

#[test]
fn a_split_set_maps_like_one_file() {
    let (_one, single) = open(&Spec::default());
    let (written, split) = open(&Spec {
        shards: 4,
        ..Spec::default()
    });
    assert!(split.weights.is_split());
    assert_eq!(split.weights.paths().len(), 4);
    assert_eq!(split.map().mapped(), split.map().total());
    let key = |m: &super::tensors::Mapped| (m.name.clone(), m.ty, m.bytes, m.place);
    let a: Vec<_> = single.map().entries().iter().map(key).collect();
    let b: Vec<_> = split.map().entries().iter().map(key).collect();
    assert_eq!(a, b);
    // The tensors sit in shards 2 to 4; shard 1 holds only metadata.
    let shards: std::collections::BTreeSet<usize> =
        split.map().entries().iter().map(|m| m.shard).collect();
    assert_eq!(shards, [1, 2, 3].into());
    // Opening from the last shard finds the same set.
    let last = written.dir.path().join("flash-00004-of-00004.gguf");
    let again = GgufModel::open(&last).expect("open").expect("model");
    assert_eq!(again.first_path(), written.first);
}

#[test]
fn a_directory_opens_the_flash_next_file_and_skips_other_models() {
    let written = write_flash(&Spec::default());
    // A vision projector of another architecture in the same directory.
    let other = written.dir.path().join("mmproj-F16.gguf");
    write_gguf(
        &other,
        &[super::testkit::kv_str("general.architecture", "clip")],
        &[],
    );
    let model = GgufModel::open(written.dir.path())
        .expect("open")
        .expect("model");
    assert_eq!(model.first_path(), written.first);
    // A directory with no file of this architecture is not an error.
    let empty = tempfile::tempdir().expect("tempdir");
    write_gguf(
        &empty.path().join("m.gguf"),
        &[super::testkit::kv_str("general.architecture", "other")],
        &[],
    );
    assert!(GgufModel::open(empty.path()).expect("open").is_none());
    let nothing = tempfile::tempdir().expect("tempdir");
    assert!(GgufModel::open(nothing.path()).expect("open").is_none());
}

#[test]
fn another_architecture_is_refused_by_the_config_reader() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    write_gguf(
        &path,
        &[super::testkit::kv_str("general.architecture", "other")],
        &[],
    );
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains(r#"general.architecture is "other", expected "qwen4exp""#),
        "{error}"
    );
    assert_eq!(ARCHITECTURE, "qwen4exp");
}

#[test]
fn a_missing_metadata_key_is_named() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    let kvs: Vec<Vec<u8>> = flash_kvs()
        .into_iter()
        .filter(|kv| !kv.windows(20).any(|w| w == b"hyper_connection.cou"))
        .collect();
    let embed = super::testkit::TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![2560, 248_320],
        ty: GgufType::Q6K,
        payload: None,
    };
    write_gguf(&path, &kvs, &[embed]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains("metadata key qwen4exp.hyper_connection.count is missing"),
        "{error}"
    );
}

#[test]
fn the_real_file_has_no_kernel_gaps() {
    // Its expert types (IQ2_S, IQ3_XXS, IQ4_NL, Q2_0, IQ4_XS, Q5_0) are all read by the
    // vendored or the codebook expert kernel.
    let (_written, model) = open(&Spec::default());
    let gaps = model.kernel_gaps();
    assert!(gaps.is_empty(), "{gaps:?}");
}

#[test]
fn the_refusal_names_each_type_without_a_kernel_and_the_types_that_have_one() {
    let (_written, model) = open(&Spec::default());
    let gaps = [KernelGap {
        ty: GgufType::Q41,
        tensors: 3,
        bytes: 3 << 30,
        first: "blk.0.ffn_down_exps.weight".to_owned(),
    }];
    let message = model.kernel_refusal(&gaps).describe();
    assert!(
        message.contains("cannot be served on CUDA yet"),
        "{message}"
    );
    assert!(
        message.contains("Q4_1") && message.contains("blk.0.ffn_down_exps.weight"),
        "{message}"
    );
    assert!(
        message.contains("dense and routed-expert kernels read Q4_K"),
        "{message}"
    );
}

#[test]
fn a_native_matrix_is_not_a_gap_anywhere_and_q4_1_is_one() {
    use super::tensors::has_decode_kernel;
    assert!(has_decode_kernel(Place::Device, GgufType::Q3K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Q3K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Q4K));
    assert!(has_decode_kernel(Place::Experts, GgufType::Iq3Xxs));
    assert!(!has_decode_kernel(Place::Experts, GgufType::Q41));
    assert_eq!(GgufType::native_types().len(), 18);
}

#[test]
fn a_file_of_supported_types_has_no_gaps() {
    let all = |_: Place, _: GgufType| true;
    let (_written, model) = open(&Spec::default());
    assert!(model.map().kernel_gaps(all).is_empty());
    assert_eq!(GgufType::native_types().len(), 18);
}

#[test]
fn the_n_gram_rows_come_from_the_ends_of_the_28_gb_table() {
    let (written, model) = open(&Spec::default());
    // Row 320,001,535 is the last row: its blocks start past byte 2^34.
    let last = 320_001_535u64;
    let mut block = [0u8; 18];
    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // scale 1.0
    block[2] = 0x98; // element 0: value index 8 (+1), element 16: index 9 (+13)
    written.patch("per_layer_token_embd.weight", last * 90, &block);
    let rows = model.ngram().expect("open the table");
    assert_eq!(
        (rows.rows(), rows.row_dim(), rows.row_bytes()),
        (320_001_536, 160, 90)
    );
    let row = rows.row(last).expect("last row");
    assert_eq!((row[0], row[16]), (1.0, 13.0));
    assert!(
        row[1..16].iter().all(|&v| v == -127.0),
        "zero nibbles decode to the first value"
    );
    // A row nobody wrote decodes to zero: the scale is zero.
    assert!(rows.row(0).expect("first row").iter().all(|&v| v == 0.0));
    assert!(rows.row(last + 1).is_err());
}

fn card(free: u64, total: u64, available: u64) -> Card {
    Card {
        free_vram: free,
        host: HostMemory::from_pair(total, available),
        ring_budget: 0,
    }
}

#[test]
fn the_plan_on_a_24_gib_card_and_a_62_gib_host_does_not_refuse() {
    let (_written, model) = open(&Spec::default());
    let sizes = model.sizes().expect("sizes");
    // 83.6 GB in the files: the experts and the n-gram table dominate.
    let total = sizes.expert_bytes + sizes.ngram_bytes + sizes.embedding_bytes + sizes.trunk_bytes;
    assert!(
        total > 70 * 1_000_000_000 && total < 90 * 1_000_000_000,
        "{total}"
    );
    // A 24 GiB card with about 1.2 GiB held by the desktop, a 62 GiB host with 40 GiB free.
    for (available, label) in [
        (40 * GIB, "40 GiB free"),
        (20 * GIB, "20 GiB free"),
        (16 * GIB, "16 GiB free"),
    ] {
        let plan = model
            .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, available))
            .expect(label);
        let load = plan
            .at_context(
                4096,
                super::super::super::moe::context_budget::KvFormat::F16,
                11,
            )
            .unwrap_or_else(|error| panic!("{label}: {}", error.describe()));
        assert!(load.context.plan.slots_per_layer() >= 11, "{label}");
        eprintln!("{label}: {load}; tier {:?}", load.context.plan.store_tier());
    }
}

#[test]
fn a_host_that_cannot_pin_the_cold_experts_plans_the_disk_tier() {
    use crate::family::expert_bank::bank_tier::TierChoice;
    let (_written, model) = open(&Spec::default());
    let kv = super::super::super::moe::context_budget::KvFormat::F16;
    let tier = |available| {
        model
            .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, available))
            .and_then(|plan| plan.at_context(4096, kv, 11))
            .map(|load| load.context.plan.store_tier())
    };
    // A host with room for the whole store pins it; the same card with 16 GiB free (the 8 GiB OS
    // floor and the process terms leave no room for a pinned store) reads from disk.
    assert!(matches!(
        tier(200 * GIB),
        Ok(TierChoice::Inclusive | TierChoice::Exclusive { .. })
    ));
    assert!(matches!(tier(16 * GIB), Ok(TierChoice::Disk)));
}

#[test]
fn the_tier_shape_uses_the_mean_expert_and_the_ring_the_largest_record() {
    let (_written, model) = open(&Spec::default());
    let sizes = model.sizes().expect("sizes");
    let records = 48 * 512;
    assert_eq!(
        sizes.shape.expert_bytes,
        sizes.expert_bytes.div_ceil(records)
    );
    assert!(sizes.record_bytes_max > sizes.shape.expert_bytes);
    // Layer 47: IQ4_XS gate and up, IQ4_NL down.
    let xs = 640u64 * 2560 / 256 * 136;
    let nl = 640u64 * 2560 / 32 * 18;
    assert_eq!(sizes.record_bytes_max, 2 * xs + nl);
}

#[test]
fn the_tensor_map_type_is_send() {
    fn needs_send<T: Send>() {}
    needs_send::<TensorMap>();
    let _ = EngError::BadRequest(String::new());
}

/// Real-file checks: set FIFI_FLASH_GGUF to the model directory (or its first shard) and run
/// with `--nocapture`. Skipped when unset. Set FIFI_FLASH_EXL3 to the EXL3 pack directory to
/// also compare the configuration and the n-gram hash constants with the pack's.
fn real_model() -> Option<GgufModel> {
    let path = std::env::var("FIFI_FLASH_GGUF").ok()?;
    Some(
        GgufModel::open(std::path::Path::new(&path))
            .expect("open the real model")
            .expect("a qwen4exp model at FIFI_FLASH_GGUF"),
    )
}

#[test]
fn real_file_maps_every_tensor_and_agrees_with_the_exl3_configuration() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let map = model.map();
    eprintln!("{}", model.summary());
    eprintln!("tensors mapped {} of {}", map.mapped(), map.total());
    assert_eq!(map.mapped(), map.total(), "unmapped: {:?}", map.unmapped());
    let sizes = model.sizes().expect("sizes");
    eprintln!("{sizes:?}");
    for gap in model.kernel_gaps() {
        eprintln!("gap {gap:?}");
    }
    eprintln!("{}", model.kernel_refusal(&model.kernel_gaps()).describe());
    if let Ok(exl3) = std::env::var("FIFI_FLASH_EXL3") {
        let config =
            std::fs::read(std::path::Path::new(&exl3).join("config.json")).expect("config");
        let mut want = Dims::read(&config).expect("exl3 dims");
        want.ngram_parts = None;
        assert_eq!(model.config().dims, want);
        eprintln!("dims equal the EXL3 config.json: {:?}", want);
    }
    // The three default-context lines for a 24 GiB card and a 62 GiB host.
    let kv = super::super::super::moe::context_budget::KvFormat::F16;
    let plan = model
        .load_plan(card(22 * GIB + GIB / 2, 62 * GIB, 40 * GIB))
        .expect("plan");
    let load = plan.at_context(4096, kv, 11).expect("context");
    eprintln!("{load}; tier {:?}", load.context.plan.store_tier());
}

#[test]
fn real_file_ngram_rows_decode_to_finite_values() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let rows = model.ngram().expect("table");
    assert_eq!(
        (rows.rows(), rows.row_dim(), rows.row_bytes()),
        (320_001_536, 160, 90)
    );
    let mut sum = 0f64;
    for row in [0u64, 1, 20_000_003, 160_000_000, 320_001_445] {
        let values = rows.row(row).expect("row");
        assert!(values.iter().all(|v| v.is_finite()), "row {row}");
        sum += values.iter().map(|v| f64::from(v.abs())).sum::<f64>();
    }
    assert!(sum > 0.0, "all sampled rows are zero");
    eprintln!("mean |value| over five rows: {}", sum / 800.0);
}

#[test]
fn real_file_host_forward_of_the_first_layers() {
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let layers: usize = std::env::var("FIFI_FLASH_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let host = super::host::HostModel::load(&model, layers).expect("load the layers");
    let tokens = [9707u32, 11, 1879];
    let run = super::host::run(&model, &host, &tokens, 8).expect("host forward");
    let rms = super::host::rms_by_layer(&run);
    for (layer, values) in &rms {
        eprintln!("layer {layer}: rms per token {values:?}");
        assert!(
            values.iter().all(|v| v.is_finite() && *v > 0.0),
            "layer {layer}"
        );
    }
    // Different tokens at different positions give different residuals.
    assert_ne!(run.layer_out[0][layers - 1], run.layer_out[2][layers - 1]);
    assert!(run.logits.iter().flatten().all(|v| v.is_finite()));
    // A second run reproduces the first.
    let again = super::host::run(&model, &host, &tokens[..1], 3).expect("again");
    assert_eq!(again.layer_out[0], run.layer_out[0]);
}

/// The whole model on the host: a plain-text prompt and the five most likely next tokens, of which the first must be Paris.
/// Needs FIFI_FLASH_GGUF, FIFI_FLASH_EXL3 (for `tokenizer.json`) and FIFI_FLASH_FULL=1; it
/// decodes every layer of every token at CPU speed (minutes).
#[test]
fn real_file_host_forward_continues_a_sentence() {
    let (Some(model), Ok(exl3), Ok(_)) = (
        real_model(),
        std::env::var("FIFI_FLASH_EXL3"),
        std::env::var("FIFI_FLASH_FULL"),
    ) else {
        eprintln!("skipped: FIFI_FLASH_GGUF, FIFI_FLASH_EXL3 or FIFI_FLASH_FULL is unset");
        return;
    };
    let tokenizer =
        tokenizers::Tokenizer::from_file(std::path::Path::new(&exl3).join("tokenizer.json"))
            .expect("tokenizer.json");
    let prompt = "The capital of France is";
    let ids = tokenizer
        .encode(prompt, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    eprintln!("prompt ids {ids:?}");
    let host = super::host::HostModel::load(&model, model.config().layers()).expect("load");
    let run = super::host::run(&model, &host, &ids, 12).expect("host forward");
    let logits = run.logits.last().expect("logits");
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<(String, f32)> = order[..5]
        .iter()
        .map(|&id| {
            (
                tokenizer.decode(&[id as u32], false).unwrap_or_default(),
                logits[id],
            )
        })
        .collect();
    eprintln!("top five after {prompt:?}: {top:?}");
    assert_eq!(top[0].0.trim(), "Paris", "top five: {top:?}");
}

#[test]
fn an_unrelated_gguf_that_does_not_open_leaves_the_exl3_start_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    // A partial download of another model: the magic and nothing else.
    std::fs::write(
        dir.path().join("other-model-Q4_K_M.gguf"),
        b"GGUF\x03\0\0\0",
    )
    .expect("write");
    assert!(GgufModel::open(dir.path()).expect("not an error").is_none());
    // A truncated set whose first shard is not ours either.
    std::fs::write(dir.path().join("part-00002-of-00003.gguf"), b"GGUF").expect("write");
    assert!(GgufModel::open(dir.path()).expect("not an error").is_none());
}

#[test]
fn a_truncated_file_that_names_the_architecture_is_an_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("Qwen3.8-Flash-Next-Q3.gguf"),
        b"GGUF\x03\0\0\0",
    )
    .expect("write");
    let error = GgufModel::open(dir.path()).expect_err("the model is there and unusable");
    assert!(error.describe().contains("gguf"), "{}", error.describe());
    // The same by the first bytes: the architecture key leads the metadata.
    let other = tempfile::tempdir().expect("tempdir");
    let mut head = b"GGUF\x03\0\0\0\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0".to_vec();
    head.extend_from_slice(
        b"\x14\0\0\0\0\0\0\0general.architecture\x08\0\0\0\x08\0\0\0\0\0\0\0qwen4exp",
    );
    std::fs::write(other.path().join("model.gguf"), head).expect("write");
    assert!(GgufModel::open(other.path()).is_err());
}

#[test]
fn a_set_with_a_truncated_shard_is_an_error_named_for_the_set() {
    let written = write_flash(&Spec {
        shards: 3,
        ..Spec::default()
    });
    let last = written.dir.path().join("flash-00003-of-00003.gguf");
    let len = std::fs::metadata(&last).expect("stat").len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&last)
        .expect("open")
        .set_len(len / 2)
        .expect("truncate");
    let error = GgufModel::open(written.dir.path()).expect_err("an unusable set");
    assert!(
        error.describe().contains("extends beyond"),
        "{}",
        error.describe()
    );
}

#[test]
fn a_file_that_says_the_expert_weights_are_not_normalised_is_refused_by_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    let mut kvs = flash_kvs();
    let mut flag = super::testkit::string("qwen4exp.expert_weights_norm");
    flag.extend_from_slice(&7u32.to_le_bytes());
    flag.push(0);
    kvs.push(flag);
    let embed = super::testkit::TensorSpec {
        name: "token_embd.weight".into(),
        dims: vec![2560, 248_320],
        ty: GgufType::Q6K,
        payload: None,
    };
    write_gguf(&path, &kvs, &[embed]);
    let weights = crate::weights::GgufWeights::open_any_type(&path, super::testkit::LIMITS, None)
        .expect("open");
    let error = GgufConfig::read(&weights).expect_err("refused").describe();
    assert!(
        error.contains("qwen4exp.expert_weights_norm is false"),
        "{error}"
    );
}
