//! Header-only inspection of EXL3 packs: a synthetic pack of the Qwen3.6 layout at small widths,
//! built from the same shape arithmetic with its byte total counted independently, and the
//! refusals a changed header must raise. The real packs are checked by the survey test.

use std::path::Path;

use super::{
    exl3_pack::{expert_name, inspect_pack_split},
    fixtures::{self, Entry},
    host::{AttnDims, DeltaDims, HostShape, MoeDims},
};
use crate::family::exl3_pack::{ExpertPart, Marker, codebook_name, layer_name};
use crate::{
    EngError,
    family::{AttentionKind, QWEN3_6_35B_A3B},
    weights::exl3::Checkpoint,
};

/// A DeltaNet layer and an attention layer at the smallest widths the format allows.
fn shape() -> HostShape {
    HostShape {
        hidden: 128,
        vocab: 128,
        kinds: vec![AttentionKind::DeltaNet, AttentionKind::Full],
        eps: 1e-6,
        delta: DeltaDims {
            key_heads: 2,
            key_dim: 128,
            value_heads: 4,
            value_dim: 128,
            conv_kernel: 4,
        },
        attn: AttnDims {
            heads_q: 2,
            heads_kv: 1,
            head_dim: 128,
            rotary_dim: 64,
            theta: 1e7,
        },
        dense_inter: None,
        moe: MoeDims {
            experts: 3,
            top_k: 2,
            inter: 128,
            shared_inter: 128,
            renorm: true,
        },
    }
}

const WORDS: u32 = 64;
const HEAD_WORDS: u32 = 96;

fn entry(name: &str, dtype: &'static str, shape: &[u32]) -> Entry {
    let width = if dtype == "F16" || dtype == "BF16" || dtype == "I16" {
        2
    } else {
        4
    };
    let bytes = shape.iter().map(|&d| d as usize).product::<usize>().max(1) * width;
    (name.to_owned(), dtype, shape.to_vec(), vec![0; bytes])
}

fn matrix(out: &mut Vec<Entry>, prefix: &str, (k, n): (u32, u32), (words, marker): (u32, &str)) {
    out.push(entry(
        &format!("{prefix}.trellis"),
        "I16",
        &[k / 16, n / 16, words],
    ));
    out.push(entry(&format!("{prefix}.suh"), "F16", &[k]));
    out.push(entry(&format!("{prefix}.svh"), "F16", &[n]));
    out.push(entry(&format!("{prefix}.{marker}"), "I32", &[]));
}

fn pack(shape: &HostShape) -> Vec<Entry> {
    let (h, v) = (shape.hidden as u32, shape.vocab as u32);
    let (m, d, a) = (&shape.moe, &shape.delta, &shape.attn);
    let mut out = vec![
        entry("model.language_model.embed_tokens.weight", "BF16", &[v, h]),
        entry("model.language_model.norm.weight", "BF16", &[h]),
    ];
    matrix(&mut out, "lm_head", (h, v), (HEAD_WORDS, "mcg"));
    for (layer, kind) in shape.kinds.iter().enumerate() {
        for norm in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            out.push(entry(&layer_name(layer, norm), "BF16", &[h]));
        }
        let at = |suffix: &str| layer_name(layer, suffix);
        match kind {
            AttentionKind::DeltaNet => {
                let z = (d.value_heads * d.value_dim) as u32;
                matrix(
                    &mut out,
                    &at("linear_attn.in_proj_qkv"),
                    (h, d.conv_dim() as u32),
                    (WORDS, "mcg"),
                );
                matrix(
                    &mut out,
                    &at("linear_attn.in_proj_z"),
                    (h, z),
                    (WORDS, "mcg"),
                );
                matrix(
                    &mut out,
                    &at("linear_attn.out_proj"),
                    (z, h),
                    (WORDS, "mcg"),
                );
                let heads = d.value_heads as u32;
                for gate in ["in_proj_a", "in_proj_b"] {
                    out.push(entry(
                        &at(&format!("linear_attn.{gate}.weight")),
                        "F16",
                        &[heads, h],
                    ));
                }
                out.push(entry(&at("linear_attn.A_log"), "BF16", &[heads]));
                out.push(entry(&at("linear_attn.dt_bias"), "BF16", &[heads]));
                out.push(entry(
                    &at("linear_attn.conv1d.weight"),
                    "BF16",
                    &[d.conv_dim() as u32, 1, d.conv_kernel as u32],
                ));
                out.push(entry(
                    &at("linear_attn.norm.weight"),
                    "BF16",
                    &[d.value_dim as u32],
                ));
            }
            _ => {
                let (q, kv) = (
                    (a.heads_q * a.head_dim) as u32,
                    (a.heads_kv * a.head_dim) as u32,
                );
                matrix(
                    &mut out,
                    &at("self_attn.q_proj"),
                    (h, 2 * q),
                    (WORDS, "mcg"),
                );
                matrix(&mut out, &at("self_attn.k_proj"), (h, kv), (WORDS, "mcg"));
                matrix(&mut out, &at("self_attn.v_proj"), (h, kv), (WORDS, "mcg"));
                matrix(&mut out, &at("self_attn.o_proj"), (q, h), (WORDS, "mcg"));
                for norm in ["q_norm", "k_norm"] {
                    out.push(entry(
                        &at(&format!("self_attn.{norm}.weight")),
                        "BF16",
                        &[a.head_dim as u32],
                    ));
                }
            }
        }
        out.push(entry(&at("mlp.gate.weight"), "F16", &[m.experts as u32, h]));
        out.push(entry(&at("mlp.shared_expert_gate.weight"), "F16", &[1, h]));
        for part in ExpertPart::ALL {
            let (k, n) = match part {
                ExpertPart::Down => (m.inter as u32, h),
                _ => (h, m.inter as u32),
            };
            for e in 0..m.experts {
                matrix(
                    &mut out,
                    &expert_name(layer, e, part),
                    (k, n),
                    (WORDS, "mcg"),
                );
            }
            let (k, n) = match part {
                ExpertPart::Down => (m.shared_inter as u32, h),
                _ => (h, m.shared_inter as u32),
            };
            matrix(
                &mut out,
                &super::exl3_pack::shared_name(layer, part),
                (k, n),
                (WORDS, "mcg"),
            );
        }
    }
    out
}

/// The device bytes counted from the entries alone, independently of the loader's walk.
fn counted(entries: &[Entry]) -> u64 {
    entries
        .iter()
        .map(|(name, _, _, bytes)| {
            let stored = bytes.len() as u64;
            if name.ends_with(".mul1") || name.ends_with(".mcg") {
                0
            } else if name.ends_with(".trellis") || name.contains("embed_tokens") {
                stored
            } else {
                stored * 2
            }
        })
        .sum()
}

fn refusal(entries: Vec<Entry>) -> String {
    let source = fixtures::checkpoint(entries);
    match inspect_pack_split(&source, &shape()) {
        Err(EngError::Checkpoint(message)) => message,
        other => panic!("expected a pack refusal, got {other:?}"),
    }
}

fn position(entries: &[Entry], name: &str) -> usize {
    entries
        .iter()
        .position(|e| e.0 == name)
        .unwrap_or_else(|| panic!("{name}"))
}

#[test]
fn a_pack_of_the_model_shape_is_counted_from_its_headers() {
    let entries = pack(&shape());
    let expected = counted(&entries);
    let report = inspect_pack_split(&fixtures::checkpoint(entries), &shape()).unwrap();
    assert_eq!(report.device_bytes, expected);
    assert_eq!(report.codebook, Marker::Mcg);
    // The head at 6 bits and every other matrix at 4: the rate is per matrix, the experts of one
    // projection share one.
    assert_eq!(report.rates.get(&12), Some(&1));
    assert_eq!(report.matrices, report.rates.values().sum::<usize>());
    // 1 head + 3 + 4 mixer matrices + 2 layers x 3 projections x (3 experts + 1 shared).
    assert_eq!(report.matrices, 1 + 3 + 4 + 2 * 3 * 4);
}

/// The walk's report carries the routed experts' device bytes in `expert_bytes` (the field a
/// banked load's dense/routed split reads); a synthetic pack with experts must produce a
/// positive total there.
#[test]
fn the_walk_totals_routed_expert_bytes() {
    let entries = pack(&shape());
    let report = inspect_pack_split(&fixtures::checkpoint(entries), &shape()).unwrap();
    assert!(report.expert_bytes > 0, "the synthetic pack has experts");
}

#[test]
fn each_header_change_is_a_named_refusal() {
    let base = pack(&shape());
    let experts = |part| expert_name(0, 1, part);
    let mut cases: Vec<(&str, Vec<Entry>, &str)> = Vec::new();

    let mut e = base.clone();
    e.remove(position(
        &e,
        &format!("{}.trellis", experts(ExpertPart::Down)),
    ));
    cases.push(("a missing expert", e, "missing tensor"));

    let mut e = base.clone();
    let at = position(&e, &format!("{}.trellis", experts(ExpertPart::Up)));
    e[at] = entry(&e[at].0.clone(), "I16", &[8, 8, 48]);
    cases.push(("an expert of another rate", e, "differ in rate"));

    let mut e = base.clone();
    let at = position(&e, "model.language_model.layers.1.self_attn.q_proj.trellis");
    e[at] = entry(&e[at].0.clone(), "I16", &[8, 8, WORDS]);
    cases.push(("a q_proj of another width", e, "the model needs"));

    let mut e = base.clone();
    e.push(entry("lm_head.su", "F16", &[128]));
    cases.push(("an old component", e, "old component"));

    let mut e = base.clone();
    let at = position(&e, &format!("{}.mcg", experts(ExpertPart::Gate)));
    e[at].0 = format!("{}.mul1", experts(ExpertPart::Gate));
    cases.push(("a second codebook", e, "another codebook"));

    let mut e = base.clone();
    e.remove(position(&e, "lm_head.mcg"));
    cases.push(("no marker", e, "codebook markers"));

    let mut e = base.clone();
    let at = position(
        &e,
        "model.language_model.layers.0.linear_attn.in_proj_z.suh",
    );
    e[at] = entry(&e[at].0.clone(), "F32", &[128]);
    cases.push(("a float suh", e, "is not F16"));

    let mut e = base.clone();
    let at = position(&e, &layer_name(1, "mlp.gate.weight"));
    e[at] = entry(&e[at].0.clone(), "F16", &[4, 128]);
    cases.push(("a router of the wrong depth", e, "the model needs"));

    let mut e = base.clone();
    let at = position(&e, &layer_name(0, "input_layernorm.weight"));
    e[at] = entry(&e[at].0.clone(), "I32", &[128]);
    cases.push(("an integer norm", e, "has dtype"));

    for dtype in ["F16", "F32"] {
        let mut e = base.clone();
        let at = position(&e, "model.language_model.embed_tokens.weight");
        let dims = e[at].2.clone();
        e[at] = entry(&e[at].0.clone(), dtype, &dims);
        cases.push(("an embedding that is not BF16", e, "the step needs Bf16"));
    }

    for (what, entries, wanted) in cases {
        let message = refusal(entries);
        assert!(message.contains(wanted), "{what}: {message}");
    }
}

#[test]
fn a_model_whose_widths_are_not_multiples_of_128_is_refused() {
    let mut narrow = shape();
    narrow.hidden = 192;
    let source = fixtures::checkpoint(pack(&narrow));
    assert!(matches!(
        inspect_pack_split(&source, &narrow),
        Err(EngError::Checkpoint(m)) if m.contains("multiple of 128")
    ));
}

/// The real packs: `FIFI_EXL3_CHECKPOINT` names a pack directory and `FIFI_EXL3_EXPECT_BYTES` the
/// device bytes an independent sum over its headers gives (trellises and the embedding as stored,
/// every other half-width tensor widened, markers, the vision tower and the MTP head not at all).
/// Prints the report.
#[test]
#[ignore = "reads the headers of a real pack in FIFI_EXL3_CHECKPOINT"]
fn the_real_pack_matches_the_model_and_its_independent_byte_sum() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let checkpoint = Checkpoint::open(Path::new(&dir)).expect("open the pack");
    QWEN3_6_35B_A3B
        .check_architecture_config(&checkpoint.model_config)
        .expect("the pack config is the model's");
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
    let report = inspect_pack_split(&checkpoint.source, &shape).expect("the pack fits the model");
    println!("{report:?}");
    let expected: u64 = std::env::var("FIFI_EXL3_EXPECT_BYTES")
        .expect("FIFI_EXL3_EXPECT_BYTES")
        .parse()
        .unwrap();
    assert_eq!(report.device_bytes, expected);
}

/// The coverage split: a read a visit stands for is neither unrecorded nor unread, a matrix
/// visit stands for its component suffixes (including the legacy ones the reader must refuse),
/// and a visit no read touches is listed unread.
#[test]
fn coverage_splits_reads_from_visits() {
    use crate::family::exl3_pack::{Visit, VisitKind, coverage};
    use std::collections::BTreeSet;

    let visits = vec![
        Visit {
            name: "model.norm.weight".into(),
            kind: VisitKind::Plain {
                dtype: crate::tensor::Dtype::F16,
                shape: vec![128],
                keep: false,
            },
            bytes: 512,
        },
        Visit {
            name: "lm_head".into(),
            kind: VisitKind::Matrix {
                k: 128,
                n: 128,
                rate: 4,
            },
            bytes: 4096,
        },
    ];
    let reads = BTreeSet::from([
        "model.norm.weight".to_owned(),
        "lm_head.trellis".to_owned(),
        "lm_head.mcg".to_owned(),
    ]);
    let (unrecorded, unread) = coverage(&reads, &visits);
    assert!(unrecorded.is_empty(), "covered reads: {unrecorded:?}");
    assert!(unread.is_empty(), "both visits touched: {unread:?}");

    let legacy = BTreeSet::from(["lm_head.su".to_owned()]);
    let (unrecorded, _unread) = coverage(&legacy, &visits);
    assert!(
        unrecorded.is_empty(),
        "a legacy component is the matrix visit's: {unrecorded:?}"
    );

    let stray = BTreeSet::from(["model.vision.patch.weight".to_owned()]);
    let (unrecorded, unread) = coverage(&stray, &visits);
    assert_eq!(unrecorded, vec!["model.vision.patch.weight".to_owned()]);
    assert_eq!(
        unread,
        vec!["model.norm.weight".to_owned(), "lm_head".to_owned()],
        "untouched visits listed"
    );
}

/// The refusal and the report the served load shows, pinned as pure functions of the name
/// sets: [`coverage_verdict`] builds both without a projector, so the words of the check are
/// testable on the host, capped at twenty names plus a count and naming the pack.
#[test]
fn coverage_verdict_words_refuse_report_and_cap() {
    use std::collections::BTreeSet;

    use crate::family::exl3_pack::{Visit, VisitKind, coverage_verdict};

    let visits = vec![
        Visit {
            name: "model.norm.weight".into(),
            kind: VisitKind::Plain {
                dtype: crate::tensor::Dtype::F16,
                shape: vec![128],
                keep: false,
            },
            bytes: 256,
        },
        Visit {
            name: "lm_head".into(),
            kind: VisitKind::Matrix {
                k: 128,
                n: 128,
                rate: 4,
            },
            bytes: 4096,
        },
    ];

    // A clean load: every read stood for a visit, every visit touched, nothing printed.
    let reads = BTreeSet::from(["model.norm.weight".to_owned(), "lm_head.svh".to_owned()]);
    let verdict = coverage_verdict(&reads, &visits, "/packs/4.00bpw");
    verdict.result.expect("covered load");
    assert!(
        verdict.report.is_none(),
        "nothing to report: {:?}",
        verdict.report
    );

    // Unrecorded reads refuse, capped at twenty names plus the count, naming the pack.
    // The stray names are zero-padded so the set's sorted order is the numeric order and the
    // one name left out is known.
    let mut drifted = BTreeSet::new();
    for index in 0..21 {
        drifted.insert(format!("stray.{index:02}"));
    }
    let verdict = coverage_verdict(&drifted, &visits, "/packs/4.00bpw");
    let error = format!("{:?}", verdict.result.expect_err("21 stray reads"));
    assert!(error.contains("/packs/4.00bpw"), "names the pack: {error}");
    assert!(error.contains("21 read tensors"), "counts them: {error}");
    assert!(error.contains("stray.00"), "shows the first: {error}");
    assert!(error.contains("(+1)"), "caps at twenty: {error}");
    assert!(
        !error.contains("stray.20"),
        "the 21st is only counted: {error}"
    );
    assert_eq!(
        verdict.report.as_deref(),
        Some(
            "EXL3 load of /packs/4.00bpw: 2 recorded tensors left unread: \
             model.norm.weight, lm_head"
        ),
        "both visits untouched, reported in visit order"
    );

    // No visits is itself a refusal: a non-recording walk must not feed the check.
    let verdict = coverage_verdict(&reads, &[], "/packs/4.00bpw");
    let error = format!("{:?}", verdict.result.expect_err("no visits"));
    assert!(error.contains("no recorded visits"), "says so: {error}");
}

/// The survey over every local Qwen3.6 EXL3 pack: `Q36_EXL3_PACK_DIRS` is a colon-separated
/// list of pack directories. The walk refuses a missing tensor, so a pack that walks proves
/// every visited name exists; the listed remainder is index-routed tensors no visit stands
/// for (the vision tower and the mtp head), a property of the pack the load's coverage check
/// never reports. That a load's reads stand for visits — the check's own property — is
/// established by loading the pack, not by this survey. Host-only: no projector, no device.
#[test]
#[ignore = "reads the headers of every pack in Q36_EXL3_PACK_DIRS"]
fn every_local_packs_tensors_are_walked_or_listed() {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    let dirs: Vec<PathBuf> = std::env::var_os("Q36_EXL3_PACK_DIRS")
        .expect("Q36_EXL3_PACK_DIRS")
        .to_string_lossy()
        .split(':')
        .filter(|part| !part.is_empty())
        .map(PathBuf::from)
        .collect();
    assert!(!dirs.is_empty(), "at least one pack directory");
    for dir in &dirs {
        let checkpoint = Checkpoint::open(dir).expect("open the pack");
        let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, 64).unwrap();
        let (_, visits) = super::exl3_pack::inspect_pack_split_visits(&checkpoint.source, &shape)
            .unwrap_or_else(|error| panic!("{}: the walk refuses: {error:?}", dir.display()));
        let mut stood_for = BTreeSet::new();
        for visit in &visits {
            stood_for.extend(visit.tensor_names());
        }
        let unread: Vec<&str> = checkpoint
            .source
            .names()
            .filter(|name| !stood_for.contains(*name))
            .collect();
        println!(
            "{}: {} visits, {} index tensors no visit stands for: {:?}",
            dir.display(),
            visits.len(),
            unread.len(),
            unread
        );
    }
}

/// The planning seam of [`exl3_plan`]: a directory without a pack's metadata is refused with
/// the loader's own words, with no projector and no device — the refusal a preflight turns
/// into a failing check before any upload.
#[test]
fn exl3_plan_refuses_a_directory_without_pack_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let error =
        super::exl3_pack::exl3_plan(dir.path()).expect_err("an empty directory is not a pack");
    let text = format!("{error:?}");
    assert!(
        text.contains("config.json"),
        "the refusal names the file: {text}"
    );
}

/// The marker names the seam reports are the marker enum's own: swapped arms would rename a
/// real pack's codebook in the decision rows, so the mapping is pinned, not println'd.
#[test]
fn codebook_names_are_the_markers_own() {
    assert_eq!(codebook_name(Marker::Mul1), "Mul1");
    assert_eq!(codebook_name(Marker::Mcg), "Mcg");
}

/// The seam and the loader's walk agree on the real pack: the same device bytes the ignored
/// walk test pins, read through the planning path the preflight uses.
#[test]
#[ignore = "reads the headers of a real pack in FIFI_EXL3_CHECKPOINT"]
fn exl3_plan_reports_the_real_packs_walk() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let plan = super::exl3_pack::exl3_plan(Path::new(&dir)).expect("the pack fits the model");
    let expected: u64 = std::env::var("FIFI_EXL3_EXPECT_BYTES")
        .expect("FIFI_EXL3_EXPECT_BYTES")
        .parse()
        .unwrap();
    assert_eq!(plan.device_bytes, expected);
    assert!(plan.matrices > 0);
    assert!(!plan.rates.is_empty());
    println!("{plan:?}");
}
