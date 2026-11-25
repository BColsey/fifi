use super::{
    GgufModel,
    config::{ARCHITECTURE, GgufConfig},
    plan::Card,
    tensors::{KernelGap, Place, Slot, TensorMap},
    testkit::{REAL_EXPERT_TYPES, Spec, flash_kvs, write_flash, write_gguf},
};
use crate::{
    EngError,
    family::qwen::flash_next::host_plan::HostMemory,
    family::qwen::flash_next::load::{Dims, tests::CONFIG},
    family::recipe::{Global, Role as Wanted, Site, SiteTensor, WeightMap},
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
        map.get(Slot::Global(Global::Embedding), None)
            .expect("embedding")
            .bytes,
        521_472_000
    );
    // Layer 1 holds the PLE block; no other layer does.
    assert!(map.get(Slot::Layer(Wanted::PleKey), Some(1)).is_some());
    assert!(map.get(Slot::Layer(Wanted::PleKey), Some(5)).is_none());
    // Layer 3 attends, layer 0 runs DeltaNet.
    assert!(
        map.get(Slot::Layer(Wanted::QProj), Some(3)).is_some()
            && map.get(Slot::Layer(Wanted::DeltaQkv), Some(3)).is_none()
    );
    assert!(
        map.get(Slot::Layer(Wanted::DeltaQkv), Some(0)).is_some()
            && map.get(Slot::Layer(Wanted::QProj), Some(0)).is_none()
    );
    // The mixer has no inject projection; every layer site does.
    assert!(
        map.get(Slot::hyper(Site::Head, SiteTensor::Inject), None)
            .is_none()
    );
    assert!(
        map.get(Slot::hyper(Site::Mixer, SiteTensor::Inject), Some(7))
            .is_some()
    );
}

#[test]
fn the_device_trunk_counts_each_tensor_in_the_form_the_executor_holds_it() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    let at = |role: Slot, layer: Option<usize>| map.get(role, layer).expect("mapped");
    // Quantized matrices stay as stored blocks.
    let qkv = at(Slot::Layer(Wanted::DeltaQkv), Some(0));
    assert_eq!(qkv.device_bytes(), qkv.bytes);
    // Hyper-connection matrices are widened to half precision: 10240 x 320 elements.
    assert_eq!(
        at(Slot::hyper(Site::Mixer, SiteTensor::Down), Some(0)).device_bytes(),
        10240 * 320 * 2
    );
    // The router is F32 as stored.
    assert_eq!(
        at(Slot::Layer(Wanted::Router), Some(0)).device_bytes(),
        2560 * 512 * 4
    );
    // The embedding and the experts are not on the card.
    assert_eq!(at(Slot::Global(Global::Embedding), None).device_bytes(), 0);
    assert_eq!(
        at(Slot::Layer(Wanted::ExpertGate), Some(0)).device_bytes(),
        0
    );
    let trunk = map.device_bytes();
    assert!(trunk > 2 * GIB && trunk < 6 * GIB, "trunk {trunk}");
}

#[test]
fn a_model_without_its_own_output_matrix_uses_the_embedding() {
    let (_written, model) = open(&Spec {
        output: false,
        ..Spec::default()
    });
    let head = model
        .map()
        .get(Slot::Global(Global::Head), None)
        .expect("head");
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
        message.contains("dense and routed-expert kernels read Q4_0, Q5_0"),
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
                crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16,
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
    let kv = crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16;
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
/// with `--nocapture`. Skipped with the inventory's SKIPPING notice when the location is
/// unresolved, and refused under FIFI_TEST_REQUIRE_DATA=1. Set FIFI_FLASH_EXL3 to the EXL3 pack
/// directory to also compare the configuration and the n-gram hash constants with the pack's.
fn real_model() -> Option<GgufModel> {
    let path = crate::test_paths::or_skip("FIFI_FLASH_GGUF")?;
    Some(
        GgufModel::open(&path)
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
    let kv = crate::family::qwen::flash_next::moe::context_budget::KvFormat::F16;
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

/// The real GGUF's shard offsets and tensor lengths reach the same staging slot bytes
/// from file and pinned sources. Read back every byte, in bounded chunks.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
#[test]
#[ignore = "requires CUDA (2 GiB) and a real split Flash-Next GGUF at FIFI_FLASH_GGUF"]
fn real_gguf_expert_stacks_stage_byte_exact_from_file_and_pinned_memory() {
    use crate::family::expert_bank::{
        layer_stream::{ExpertTransport, UploadSet},
        layer_stream_card::{CardTransport, StackHome, StackSpan},
    };
    use crate::kernels::cuda::{CudaContext, device_present};
    use std::sync::Arc;

    let model = real_model().expect("set FIFI_FLASH_GGUF to a split Flash-Next checkpoint");
    assert!(model.weights.is_split(), "a split checkpoint is required");
    assert!(device_present("real Flash-Next stacks"), "CUDA is required");
    let experts = model.config().dims.experts as usize;
    let roles = [Wanted::ExpertGate, Wanted::ExpertUp, Wanted::ExpertDown];
    let layer = (0..model.config().layers())
        .find(|&layer| {
            roles.iter().all(|&role| {
                model
                    .map()
                    .get(role, Some(layer))
                    .is_some_and(|stack| stack.shard != 0)
            })
        })
        .expect("a layer with all three expert stacks outside the first shard");
    let stacks = roles.map(|role| {
        let mapped = model.map().get(role, Some(layer)).expect("expert stack");
        let tensor = model
            .weights
            .header()
            .tensor(&mapped.name)
            .expect("GGUF tensor");
        assert_eq!(tensor.shard(), mapped.shard, "{}", mapped.name);
        tensor
    });
    let total = stacks.iter().map(|tensor| tensor.byte_len()).sum::<u64>();
    assert!(total <= 2 * GIB, "staging exceeds the 2 GiB test budget");
    let ctx = CudaContext::new().expect("CUDA context");
    let compute = ctx.stream().expect("compute stream");

    let mut want = vec![0u8; 8 << 20];
    let mut got = vec![0u8; want.len()];
    for pinned in [false, true] {
        let mut store = Vec::new();
        let mut spans = Vec::new();
        for tensor in stacks {
            let bytes = usize::try_from(tensor.byte_len()).expect("stack size");
            let home = if pinned {
                let mut buf = ctx.pinned_buf(bytes).expect("pinned stack");
                buf.fill(bytes, |out| model.weights.read_range(tensor.name(), 0, out))
                    .expect("fill pinned stack");
                let index = store.len();
                store.push(buf);
                StackHome::Pinned {
                    buf: index,
                    offset: 0,
                }
            } else {
                StackHome::File {
                    file: Arc::new(model.weights.file_of(tensor).try_clone().expect("shard")),
                    direct: None,
                    offset: tensor.abs_offset(),
                }
            };
            spans.push(StackSpan {
                bytes,
                home,
                shape: None,
            });
        }
        let mut transport =
            CardTransport::new(&ctx, &compute, experts, vec![spans], store, 1).expect("transport");
        let ticket = transport.begin(0, &UploadSet::Whole, 0).expect("upload");
        let report = transport.wait(ticket).expect("copy complete");
        assert_eq!(report.bytes, total, "pinned={pinned}");
        for (stack, tensor) in stacks.into_iter().enumerate() {
            let staged = transport.stack(0, 0, stack).expect("staged stack");
            assert_eq!(
                staged.bytes() as u64,
                tensor.byte_len(),
                "{}",
                tensor.name()
            );
            for offset in (0..staged.bytes()).step_by(want.len()) {
                let len = want.len().min(staged.bytes() - offset);
                model
                    .weights
                    .read_range(tensor.name(), offset as u64, &mut want[..len])
                    .expect("GGUF bytes");
                staged
                    .buffer()
                    .read_at(&compute, staged.offset() + offset, &mut got[..len])
                    .expect("device bytes");
                assert!(
                    got[..len] == want[..len],
                    "{} at byte {offset}, pinned={pinned}",
                    tensor.name()
                );
            }
        }
        transport.release(0).expect("release staging slot");
        eprintln!("layer {layer}: {total} bytes verified, pinned={pinned}");
    }
}

/// The rate of the real file's expert extents: one buffered `pread` of a stack against the
/// direct parallel reads of every candidate plan, on a layer far from the ones other runs read
/// (`FIFI_FLASH_EXTENT_LAYER`, default 40), and the bytes of the best plan against the buffered
/// bytes. A direct read bypasses the page cache, so its rates are cold-cache rates by
/// construction; the buffered rate is for the pages as found.
#[test]
fn real_file_extents_read_in_parallel_with_direct_io() {
    use crate::family::expert_bank::{
        extent_read::{ExtentPlan, read_extent, span_of},
        host_tier::{SECTOR, open_direct},
    };
    use std::os::unix::fs::FileExt;
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let layer: usize = std::env::var("FIFI_FLASH_EXTENT_LAYER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(40);
    let experts = model.config().dims.experts as usize;
    let lens: Vec<u64> = model
        .weights
        .paths()
        .iter()
        .map(|path| std::fs::metadata(path).expect("shard").len())
        .collect();
    let dims = &model.config().dims;
    let index = crate::weights::GgufExpertIndex::new(
        model.weights.header(),
        &lens,
        &crate::weights::ExpertTensorNames::standard(),
        model.config().layers(),
        experts,
        (dims.hidden as usize, dims.moe_inter as usize),
        None,
    )
    .expect("index");
    let slices = index.slices(layer, 0).expect("slices");
    let part = slices.parts().next().expect("a stack");
    let bytes = ((part.file_range.end - part.file_range.start) as usize) * experts;
    let offset = part.file_range.start;
    let path = &model.weights.paths()[part.shard];
    let buffered = std::fs::File::open(path).expect("open");
    let direct = open_direct(path).expect("a drive that takes O_DIRECT");
    eprintln!(
        "layer {layer}: one stack of {} MiB at offset {offset} of {}",
        bytes >> 20,
        path.display()
    );
    let mut want = vec![0u8; bytes];
    let started = std::time::Instant::now();
    let mut done = 0usize;
    while done < bytes {
        let n = buffered
            .read_at(&mut want[done..], offset + done as u64)
            .expect("pread");
        assert!(n > 0, "end of file");
        done += n;
    }
    let seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "buffered pread, one thread: {:.2} GB/s ({seconds:.2} s)",
        bytes as f64 / seconds / 1e9
    );
    let (span, delta) = span_of(offset, bytes);
    let mut backing = vec![0u8; span + SECTOR as usize];
    let shift = (SECTOR as usize - backing.as_ptr() as usize % SECTOR as usize) % SECTOR as usize;
    let mut best: Option<(f64, ExtentPlan)> = None;
    for plan in ExtentPlan::CANDIDATES {
        let dst = &mut backing[shift..shift + span];
        let stats = read_extent(&direct, offset, bytes, dst, plan).expect("direct read");
        eprintln!(
            "direct {} threads x {} KiB: {:.2} GB/s ({:.2} s)",
            plan.threads,
            plan.chunk >> 10,
            stats.gbps(),
            stats.seconds
        );
        if best.is_none_or(|(rate, _)| stats.gbps() > rate) {
            best = Some((stats.gbps(), plan));
        }
        assert!(
            backing[shift + delta..shift + delta + bytes] == want[..],
            "{plan:?} read other bytes than the buffered read"
        );
    }
    let (rate, plan) = best.expect("a plan");
    eprintln!(
        "best: {} threads x {} KiB at {rate:.2} GB/s",
        plan.threads,
        plan.chunk >> 10
    );
}

#[test]
fn the_weight_map_names_each_recipe_role_and_gives_its_stored_type() {
    let (_written, model) = open(&Spec::default());
    let map = model.map();
    // A DeltaNet layer holds the gated recurrence, no attention and no indexer.
    assert_eq!(
        map.layer(0, Wanted::DeltaQkv).as_deref(),
        Some("blk.0.attn_qkv.weight")
    );
    assert_eq!(map.layer(0, Wanted::QProj), None);
    assert_eq!(map.layer(0, Wanted::IndexProj), None);
    // Every fourth layer is the QSA layer, with the indexer's two matrices.
    assert_eq!(
        map.layer(3, Wanted::QProj).as_deref(),
        Some("blk.3.attn_q.weight")
    );
    assert!(map.layer(3, Wanted::IndexProj).is_some());
    assert!(map.layer(3, Wanted::IndexKey).is_some());
    assert_eq!(map.layer(3, Wanted::DeltaQkv), None);
    // The hyper-connection sites carry the norm the plain model holds as a separate role.
    assert_eq!(
        map.layer(5, Wanted::Hyper(Site::Mixer, SiteTensor::Down))
            .as_deref(),
        Some("blk.5.hc_attn_down.weight")
    );
    assert_eq!(
        map.layer(5, Wanted::Hyper(Site::Ffn, SiteTensor::Inject))
            .as_deref(),
        Some("blk.5.hc_ffn_inject.weight")
    );
    assert_eq!(map.layer(5, Wanted::AttnNorm), None);
    // Only the layer that carries the per-layer embedding step has its tensors.
    assert!(map.layer(1, Wanted::PleKey).is_some());
    assert_eq!(map.layer(2, Wanted::PleKey), None);
    // The stored type is the file's own, per layer and role.
    let expert = model
        .map()
        .get(Slot::Layer(Wanted::ExpertGate), Some(2))
        .expect("expert gate");
    assert_eq!(map.storage(2, Wanted::ExpertGate), Some(expert.ty));
    assert!(map.storage(2, Wanted::DeltaQkv).is_some());
    assert_eq!(
        map.global(Global::Embedding).as_deref(),
        Some("token_embd.weight")
    );
    assert!(map.global(Global::NgramTable).is_some());
    assert!(map.global(Global::HeadSite(SiteTensor::Norm)).is_some());
    assert_eq!(map.global(Global::FinalNorm), None);
}

#[test]
fn the_feeds_of_a_synthetic_model_have_the_widths_the_driver_expects() {
    use crate::family::qwen::flash_next::cuda::driver::TokenFeeds;
    let written = write_flash(&Spec::default());
    let model: &'static GgufModel = Box::leak(Box::new(
        GgufModel::open(&written.first)
            .expect("open")
            .expect("a qwen4exp model"),
    ));
    let mut feeds = super::feeds::gguf_feeds(model, 2).expect("feeds");
    let start = feeds.begin(5).expect("start vectors");
    assert_eq!(start.stream.len(), 4 * 2560);
    assert_eq!(start.ple_embedding.len(), 16 * 160);
    let update = feeds.ple_update(5, &start.stream).expect("ple update");
    assert_eq!(update.len(), 4 * 2560);
    assert!(update.iter().all(|v| v.is_finite()));
    let refused = feeds.begin(248_320).expect_err("outside the vocabulary");
    assert!(
        refused
            .describe()
            .contains("outside the 248320 embedding rows"),
        "{}",
        refused.describe()
    );
}

/// The GGUF feed against the host step: the same tokens through the feed (preloaded F32
/// projections, gathered rows) and through `ple::step` over a projector that decodes from the
/// file on every call. Needs FIFI_FLASH_GGUF.
#[test]
fn real_file_feed_equals_the_host_ple_step() {
    use crate::family::qwen::flash_next::{cuda::driver::TokenFeeds, ple};
    let Some(model) = real_model() else {
        eprintln!("skipped: FIFI_FLASH_GGUF is unset");
        return;
    };
    let model: &'static GgufModel = Box::leak(Box::new(model));
    let mut feeds = super::feeds::gguf_feeds(model, 8).expect("feeds");
    let host = super::host::HostModel::load(model, 2).expect("host layers");
    let ngram = model.ngram().expect("table");
    let mut projector = super::host::HostProjector::new(&model.weights, &ngram, 8);
    let weights = host.view();
    let mut state = ple::PleState::new(&host.shape.ple);
    let tokens = [760u32, 6511, 314, 9338, 369];
    let mut worst = 0f32;
    for (at, &token) in tokens.iter().enumerate() {
        let start = feeds.begin(token).expect("start");
        // The host embedding: the row of the same table through the host projector.
        let row = crate::family::project::Projector::rows(&mut projector, weights.embed, &[token])
            .expect("embedding row");
        assert_eq!(start.stream, row.repeat(4), "token {at}");
        // A stream that differs per token and branch.
        let stream: Vec<f32> = (0..start.stream.len())
            .map(|i| ((i * 31 + at * 7) % 101) as f32 * 0.01 - 0.5)
            .collect();
        let got = feeds.ple_update(token, &stream).expect("feed update");
        let want = ple::step(
            &mut projector,
            &host.shape.ple,
            &weights.ple,
            &mut state,
            token,
            &stream,
        )
        .expect("host step");
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            worst = worst.max((g - w).abs());
        }
    }
    eprintln!("max abs difference between the feed and the host step: {worst}");
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn the_plans_uniform_expert_is_the_mean_of_the_layers_own_records() {
    let (_written, model) = open(&Spec::default());
    let sizes = model.sizes().expect("sizes");
    let layers = 48u64;
    assert_eq!(sizes.layer_record_bytes.len(), 48);
    let sum: u64 = sizes.layer_record_bytes.iter().sum();
    // One slot in every layer costs the sum of the layers' records when each layer's slot is
    // its own record size, and the plan charges `slot_cost`: the same to within rounding.
    assert!(sizes.shape.slot_cost() >= sum);
    assert!(sizes.shape.slot_cost() - sum < layers);
    // Slots as large as the largest record in every layer would cost more: the plan is exact
    // only for a bank sized per layer.
    assert!(layers * sizes.record_bytes_max > sum + sum / 20);
    // The records come from the shared index, layer by layer.
    let index = model.expert_index().expect("index");
    for (layer, &bytes) in sizes.layer_record_bytes.iter().enumerate() {
        assert_eq!(index.record_bytes(layer).unwrap(), bytes);
    }
}

#[test]
fn the_record_store_reads_the_first_and_last_expert_of_the_real_layout() {
    use crate::weights::expert_store::{ExpertRecordStore, RecordSite};
    let (written, model) = open(&Spec::default());
    let index = model.expert_index().expect("index");
    // Write a marker into the last expert's down slice of layer 47 and read it back through
    // the store: the offset is past 2^32 bytes in a sparse file.
    let slices = index.slices(47, 511).expect("slices");
    written.patch(slices.down.name, slices.down.tensor_offset, &[7, 8, 9]);
    let store = model.record_store().expect("store");
    let record = store
        .read_records(&[RecordSite {
            layer: 47,
            expert: 511,
        }])
        .expect("read");
    let gate = slices.gate.as_ref().expect("a gate stack");
    let gate_up = gate.file_range.end - gate.file_range.start + slices.up.file_range.end
        - slices.up.file_range.start;
    assert_eq!(
        &record[0][gate_up as usize..gate_up as usize + 3],
        &[7, 8, 9]
    );
    assert_eq!(
        store.record_bytes_at(47).expect("layer 47"),
        index.record_bytes(47).unwrap()
    );
    assert_eq!(store.record_bytes(), index.max_record_bytes().unwrap());
}

#[test]
fn the_small_model_opens_maps_and_runs_the_host_forward() {
    let small = super::smallmodel::write_small(7);
    let model = GgufModel::open(&small.first)
        .expect("open")
        .expect("a qwen4exp model");
    assert_eq!(model.map().mapped(), model.map().total());
    assert!(model.map().unmapped().is_empty());
    assert_eq!(model.config().layers(), 4);
    assert!(model.kernel_gaps().is_empty(), "{:?}", model.kernel_gaps());
    let host = super::host::HostModel::load(&model, 4).expect("host layers");
    let run = super::host::run(&model, &host, &[5, 9, 200], 2).expect("host forward");
    assert_eq!(run.logits.len(), 3);
    assert!(run.logits.iter().flatten().all(|v| v.is_finite()));
    // Different tokens give different logits.
    assert_ne!(run.logits[0], run.logits[1]);
}
