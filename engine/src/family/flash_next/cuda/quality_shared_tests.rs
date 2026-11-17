//! The application-quality arm: teacher-forced margins
//! of the shared EXL3 prefill against the legacy prefill over a frozen corpus
//! of repository source, on the served build.
//!
//! Pure host core: the frozen corpus, the scored-position math, and the paired
//! document-clustered bootstrap verdict (the frozen m3 core in
//! `qwen36::resident::m3_quality`, widened to the family). Three environment
//! tests carry the protocol:
//!
//! `quality_leg_scores_the_frozen_corpus` (ignored; one card leg per run, the
//! flint tf-gate leg pattern — the runner pins the arm BEFORE the process):
//! - `FIFI_FLASH_NEXT_QUALITY_ARM`: `legacy` or `shared` (required).
//! - `INFERENCE_FLASH_NEXT_SHARED_PREFILL`: `0` for legacy, `1` for shared;
//!   asserted against the arm, fail-loud on a mismatch.
//! - `FIFI_FLASH_NEXT_EXL3_PACK`: the pack directory (required).
//! - `FIFI_FLASH_NEXT_QUALITY_OUT`: output directory of `rows-<arm>.json`
//!   (required).
//!
//! Legacy pins `prefill_rows` 64, shared pins 256 (the pinned arms);
//! both run one driver at `context` 17156, f16 KV, 196 expert slots. Each
//! document: `reset`, `feed_rows` of the class context, then 768 eager
//! `logits` steps scoring the frozen true suffix. Per-arm rows land in one
//! JSON sidecar.
//!
//! `quality_corpus_freeze_stamps_the_id_stream` (ignored; host, pack env +
//! `FIFI_FLASH_NEXT_QUALITY_OUT`): tokenizes the frozen corpus with the
//! engine's own pack tokenizer and writes `freeze.json` (per-document token
//! counts, the id-stream sha256) for stamping the registration.
//!
//! `quality_verdict_decides_from_both_legs` (ignored; host,
//! `FIFI_FLASH_NEXT_QUALITY_OUT`): reads both legs' rows, computes the
//! bootstrap bounds, writes `verdict.json`. Pass: NLL 95% bound <= +0.02 nats
//! AND top-1 95% bound >= -0.5 pp.
//!
//! Host unit tests always run: corpus integrity against the committed
//! manifest, the softmax row math, and the verdict rule on synthetic rows.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::start::build_driver;
use crate::family::{
    flash_next::settings::{CudaSettings, KvFormatName},
    qwen36::resident::m3_quality::{PositionRow, bootstrap_bound, nll_rows, top1_rows},
    session::TokenStepper,
};
use crate::tokenizer::TokenizerBundle;

/// The frozen corpus manifest, committed beside the documents.
const MANIFEST_JSON: &str = include_str!("testdata/quality-corpus/manifest.json");
/// The twelve frozen documents, in manifest order (4 S, 4 M, 2 L, 2 K16).
const DOC_BYTES: [&[u8]; 12] = [
    include_bytes!("testdata/quality-corpus/doc-00.txt"),
    include_bytes!("testdata/quality-corpus/doc-01.txt"),
    include_bytes!("testdata/quality-corpus/doc-02.txt"),
    include_bytes!("testdata/quality-corpus/doc-03.txt"),
    include_bytes!("testdata/quality-corpus/doc-04.txt"),
    include_bytes!("testdata/quality-corpus/doc-05.txt"),
    include_bytes!("testdata/quality-corpus/doc-06.txt"),
    include_bytes!("testdata/quality-corpus/doc-07.txt"),
    include_bytes!("testdata/quality-corpus/doc-08.txt"),
    include_bytes!("testdata/quality-corpus/doc-09.txt"),
    include_bytes!("testdata/quality-corpus/doc-10.txt"),
    include_bytes!("testdata/quality-corpus/doc-11.txt"),
];
/// The classes in document order; the plan derives from them, never the reverse.
const DOC_CLASSES: [&str; 12] = [
    "S", "S", "S", "S", "M", "M", "M", "M", "L", "L", "K16", "K16",
];
/// The context length per class (registration: 1k/4k/8k/16k).
fn context_ids(class: &str) -> usize {
    match class {
        "S" => 1024,
        "M" => 4096,
        "L" => 8192,
        "K16" => 16384,
        other => panic!("unknown corpus class {other}"),
    }
}
/// The scored suffix: 769 tokens, the first 768 scored against the next.
const SUFFIX_IDS: usize = 769;
/// Registration margins: NLL ceiling +0.02 nats, top-1 floor -0.5 pp.
const NLL_UB: f64 = 0.02;
const TOP1_LB: f64 = -0.005;
/// The frozen bootstrap: B=10,000 draws, seed 20261031, 12 document clusters.
const DRAWS: usize = 10_000;
const SEED: u64 = 20_261_031;
const CLUSTERS: u32 = 12;
/// The pinned artifact's tokenizer vocabulary (artifact.rs): the real
/// vocabulary the head reports, narrower than the padded row.
const VOCAB_PIN: u32 = 248_077;
/// The frozen corpus stamps (the freeze run):
/// the engine-tokenizer id stream of the 12 documents, checked by the leg before
/// any card work and by the verdict against each sidecar's counts and id stamp.
const FROZEN_ID_SHA256: &str = "cc99f6eef0e0662c7efeedc7c3e90599f1d748dfc326d0fade45f13284a0e696";
const FROZEN_TOKEN_COUNTS: [usize; 12] = [
    2402, 4372, 4101, 3471, 8983, 10051, 12016, 13052, 22291, 16924, 35197, 38988,
];
/// The leg's CudaSettings pins, recorded in every sidecar and asserted by the verdict.
fn expected_pins(rows: u32) -> serde_json::Value {
    serde_json::json!({
        "INFERENCE_FLASH_NEXT_SHARED_PREFILL": if rows == 64 { "0" } else { "1" },
        "prefill_rows": rows,
        "context": 17_156,
        "kv": "f16",
        "slots": 196,
    })
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn require(name: &str) -> String {
    var(name).unwrap_or_else(|| panic!("{name} is required"))
}

/// Scores one document through any stepper: reset, the context feed, then 768
/// eager steps. `doc` is the cluster every row of this document carries — the
/// producer is stepper-generic so a host test proves the cluster wiring
/// without a card.
fn score_document<S: TokenStepper>(
    stepper: &mut S,
    doc: u32,
    ids: &[u32],
    class_context: usize,
) -> Vec<ScoredRow> {
    let suffix = &ids[class_context..class_context + SUFFIX_IDS];
    stepper.reset().expect("reset between documents");
    stepper
        .feed_rows(&ids[..class_context])
        .expect("feed the context");
    let mut rows = Vec::with_capacity(SUFFIX_IDS - 1);
    // Feeding suffix[step] yields the row that scores suffix[step+1]: 768 positions.
    for step in 0..SUFFIX_IDS - 1 {
        let row = stepper.logits(suffix[step]).expect("the logits row");
        let (logprob_true, top1_correct) = score_row(&row, suffix[step + 1], VOCAB_PIN as usize);
        rows.push(ScoredRow {
            cluster: doc,
            logprob_true,
            top1_correct,
        });
    }
    rows
}

/// The frozen id stream over the whole corpus: per-document token counts and
/// the sha256 of the concatenated little-endian u32 ids.
fn frozen_stream(bundle: &TokenizerBundle, manifest: &[ManifestEntry]) -> (Vec<usize>, String) {
    let mut counts = Vec::with_capacity(manifest.len());
    let mut stream = Sha256::new();
    for (index, entry) in manifest.iter().enumerate() {
        let text = std::str::from_utf8(DOC_BYTES[index]).expect("UTF-8 document");
        let ids = bundle.tokenizer().encode(text).expect("tokenize");
        let need = context_ids(&entry.class) + SUFFIX_IDS;
        assert!(
            ids.len() >= need,
            "document {index} ({}): {} ids < the {need} the class needs",
            entry.class,
            ids.len()
        );
        for id in &ids {
            stream.update(id.to_le_bytes());
        }
        counts.push(ids.len());
    }
    (counts, format!("{:x}", stream.finalize()))
}

/// The tokenizer bundle of the pack, pinned by its download manifest.
fn pack_tokenizer(pack: &Path) -> TokenizerBundle {
    let manifest = std::fs::read(pack.join("download-manifest.json"))
        .unwrap_or_else(|error| panic!("download-manifest.json: {error}"));
    let pin = super::super::artifact::tokenizer_pin(&manifest, VOCAB_PIN)
        .expect("the pack's tokenizer pin");
    TokenizerBundle::load(pack, &pin).expect("the pack tokenizer")
}

/// One scored position of the arm: the true token's log-probability and the
/// greedy top-1 equality, from a full logits row narrowed to the real
/// vocabulary `vocab` (the head's padded row is wider). Stable log-softmax,
/// the accumulation in f64.
fn score_row(row: &[f32], next_true: u32, vocab: usize) -> (f32, bool) {
    assert!(
        row.len() >= vocab,
        "the logits row is narrower than the vocabulary"
    );
    let row = &row[..vocab];
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f64;
    for &value in row {
        sum += f64::from((value - max).exp());
    }
    let lse = f64::from(max) + sum.ln();
    let mut argmax = 0usize;
    for (index, &value) in row.iter().enumerate() {
        if value > row[argmax] {
            argmax = index;
        }
    }
    (
        (f64::from(row[next_true as usize]) - lse) as f32,
        argmax == next_true as usize,
    )
}

/// The verdict bounds: candidate-minus-reference, one-sided 95%, document
/// clusters as the resample unit (the frozen m3 core).
fn bounds(candidate: &[PositionRow], reference: &[PositionRow]) -> (f64, f64) {
    assert_eq!(candidate.len(), reference.len(), "leg row counts");
    let clusters: Vec<u32> = candidate.iter().map(|row| row.cluster).collect();
    assert!(
        reference
            .iter()
            .zip(candidate)
            .all(|(r, c)| r.cluster == c.cluster),
        "leg rows are not cluster-aligned"
    );
    let nll = bootstrap_bound(
        &nll_rows(candidate)
            .expect("nll_rows rejects a row whose logprob_true is not a true log-probability"),
        &nll_rows(reference)
            .expect("nll_rows rejects a row whose logprob_true is not a true log-probability"),
        &clusters,
        CLUSTERS,
        DRAWS,
        SEED,
        true,
    );
    let top1 = bootstrap_bound(
        &top1_rows(candidate),
        &top1_rows(reference),
        &clusters,
        CLUSTERS,
        DRAWS,
        SEED,
        false,
    );
    (nll, top1)
}

fn passes(nll_ub: f64, top1_lb: f64) -> bool {
    nll_ub <= NLL_UB && top1_lb >= TOP1_LB
}

#[derive(serde::Deserialize)]
struct ManifestEntry {
    class: String,
    byte_sha256: String,
    byte_length: u64,
}

#[derive(serde::Deserialize)]
struct CorpusManifest {
    documents: Vec<ManifestEntry>,
}

/// Verified corpus: every document's bytes against the committed manifest.
fn documents() -> Vec<ManifestEntry> {
    let manifest: CorpusManifest =
        serde_json::from_str(MANIFEST_JSON).expect("the corpus manifest");
    let manifest = manifest.documents;
    assert_eq!(manifest.len(), DOC_BYTES.len(), "12 documents");
    for (index, entry) in manifest.iter().enumerate() {
        assert_eq!(entry.class, DOC_CLASSES[index], "document {index} class");
        assert_eq!(
            entry.byte_length as usize,
            DOC_BYTES[index].len(),
            "doc {index} size"
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(DOC_BYTES[index])),
            entry.byte_sha256,
            "document {index} bytes"
        );
    }
    manifest
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct ScoredRow {
    cluster: u32,
    logprob_true: f32,
    top1_correct: bool,
}

/// One leg's complete sidecar, as the verdict re-reads it.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Sidecar {
    arm: String,
    build: String,
    pack: String,
    id_sha: String,
    pins: serde_json::Value,
    doc_tokens: Vec<usize>,
    rows: Vec<ScoredRow>,
}

impl From<&PositionRow> for ScoredRow {
    fn from(row: &PositionRow) -> Self {
        ScoredRow {
            cluster: row.cluster,
            logprob_true: row.logprob_true,
            top1_correct: row.top1_correct,
        }
    }
}

fn rows_to_position(rows: &[ScoredRow]) -> Vec<PositionRow> {
    rows.iter()
        .map(|row| PositionRow {
            cluster: row.cluster,
            logprob_true: row.logprob_true,
            top1_correct: row.top1_correct,
        })
        .collect()
}

/// One leg on the card: the arm's pins asserted, the corpus scored, the rows
/// sidecar written. The runner (never the test) sets the switch env, so no
/// in-process env mutation exists.
#[test]
#[ignore = "a card leg; run one arm per process with its env pins"]
fn quality_leg_scores_the_frozen_corpus() {
    let arm = require("FIFI_FLASH_NEXT_QUALITY_ARM");
    let switch = require("INFERENCE_FLASH_NEXT_SHARED_PREFILL");
    let pack = PathBuf::from(require("FIFI_FLASH_NEXT_EXL3_PACK"));
    let out = PathBuf::from(require("FIFI_FLASH_NEXT_QUALITY_OUT"));
    let (want_switch, rows): (&str, u32) = match arm.as_str() {
        "legacy" => ("0", 64),
        "shared" => ("1", 256),
        other => panic!("unknown arm {other}: legacy or shared"),
    };
    assert_eq!(
        switch, want_switch,
        "the arm {arm} pins INFERENCE_FLASH_NEXT_SHARED_PREFILL={want_switch}"
    );

    let manifest = documents();
    let bundle = pack_tokenizer(&pack);
    let (doc_tokens, id_stream) = frozen_stream(&bundle, &manifest);
    // The registration stamps, checked BEFORE any card work: a corpus or
    // tokenizer drift refuses the leg rather than scoring a different stream.
    assert_eq!(
        id_stream, FROZEN_ID_SHA256,
        "the corpus id stream drifted from the frozen registration"
    );
    assert_eq!(
        doc_tokens, FROZEN_TOKEN_COUNTS,
        "the corpus token counts drifted from the frozen registration"
    );
    let ids: Vec<Vec<u32>> = manifest
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let text = std::str::from_utf8(DOC_BYTES[index]).expect("document {index} is UTF-8");
            bundle
                .tokenizer()
                .encode(text)
                .expect("tokenize the document")
                .into_iter()
                .take(context_ids(&entry.class) + SUFFIX_IDS)
                .collect()
        })
        .collect();

    // K16's context plus the suffix, rounded to the executor's multiple of 4;
    // expected_pins records 17156, so bind the two.
    let context: u32 = ((16384 + SUFFIX_IDS + 3) as u32 / 4) * 4;
    assert_eq!(context, 17_156, "the registration's context pin");
    let wanted = CudaSettings {
        context,
        kv_format: KvFormatName::F16,
        slots: Some(196),
        expert_stats: None,
        prefill_rows: rows,
        ring_rows: 32,
    };
    let mut driver = build_driver(
        &|stage| println!("stage: {stage}"),
        &pack,
        &wanted,
        VOCAB_PIN as usize,
        None,
    )
    .expect("build the driver");

    let mut scored: Vec<ScoredRow> = Vec::with_capacity(12 * (SUFFIX_IDS - 1));
    let began = std::time::Instant::now();
    for (doc, entry) in manifest.iter().enumerate() {
        let class_context = context_ids(&entry.class);
        scored.extend(score_document(
            &mut driver,
            doc as u32,
            &ids[doc],
            class_context,
        ));
        println!(
            "document {doc} ({}): scored by {:.1} s",
            entry.class,
            began.elapsed().as_secs_f64()
        );
    }
    assert_eq!(scored.len(), 12 * (SUFFIX_IDS - 1), "768 rows per document");

    // `doc_tokens` stays the FROZEN counts from frozen_stream (the full id
    // stream the registration stamps); the truncated per-leg cuts never reach
    // the sidecar, or the verdict's frozen-counts check would refuse both legs.
    let pins = expected_pins(rows);
    let out_path = out.join(format!("rows-{arm}.json"));
    if out_path.exists() {
        panic!("an existing rows-{arm}.json refuses to be overwritten; remove it deliberately");
    }
    let sidecar = Sidecar {
        arm: arm.clone(),
        build: build_stamp(),
        pack: pack.display().to_string(),
        id_sha: FROZEN_ID_SHA256.to_string(),
        pins,
        doc_tokens,
        rows: scored,
    };
    std::fs::write(
        &out_path,
        serde_json::to_vec_pretty(&sidecar).expect("serialize the rows"),
    )
    .expect("write the rows sidecar");
    println!(
        "{arm}: {} rows in {:.1} s",
        sidecar.rows.len(),
        began.elapsed().as_secs_f64()
    );
}

/// Host: stamp the frozen id stream (token counts, sha256) for the registration.
#[test]
#[ignore = "needs the pack; host-only, no card"]
fn quality_corpus_freeze_stamps_the_id_stream() {
    let pack = PathBuf::from(require("FIFI_FLASH_NEXT_EXL3_PACK"));
    let out = PathBuf::from(require("FIFI_FLASH_NEXT_QUALITY_OUT"));
    let manifest = documents();
    let bundle = pack_tokenizer(&pack);
    let (counts, id_sha) = frozen_stream(&bundle, &manifest);
    let stamp = serde_json::json!({
        "token_counts": counts,
        "id_stream_sha256": id_sha,
        "vocab_pin": VOCAB_PIN,
    });
    std::fs::write(
        out.join("freeze.json"),
        serde_json::to_vec_pretty(&stamp).expect("serialize the stamp"),
    )
    .expect("write the freeze stamp");
    println!("freeze: {stamp}");
}

/// The build stamp the runner pins; the sidecar records it so a sidecar is
/// always attributable to the binary and tree that produced it.
fn build_stamp() -> String {
    std::env::var("FIFI_FLASH_NEXT_QUALITY_BUILD").unwrap_or_else(|_| "unset".to_string())
}

/// The verdict's sidecar gate: the sidecar must BE this arm's run - the arm
/// name, the registration's pins, the frozen token counts, and exactly 768
/// rows per document cluster. A stale file or two copies of one arm refuse
/// here. Shared with the host round-trip test so a field mismatch fails on
/// the host, not after the card.
fn check_sidecar(sidecar: &Sidecar, arm: &str, rows: u32) -> Vec<PositionRow> {
    assert_eq!(sidecar.arm, arm, "the sidecar's arm");
    assert_eq!(
        sidecar.id_sha, FROZEN_ID_SHA256,
        "the {arm} sidecar's corpus id stamp"
    );
    assert_eq!(
        sidecar.pins,
        serde_json::to_value(expected_pins(rows)).unwrap(),
        "the {arm} sidecar's pins"
    );
    assert_eq!(
        sidecar.doc_tokens, FROZEN_TOKEN_COUNTS,
        "the {arm} sidecar's token counts"
    );
    assert_eq!(
        sidecar.rows.len(),
        12 * (SUFFIX_IDS - 1),
        "the {arm} sidecar's row count"
    );
    for doc in 0..12u32 {
        assert_eq!(
            sidecar.rows.iter().filter(|row| row.cluster == doc).count(),
            SUFFIX_IDS - 1,
            "the {arm} sidecar's rows for document {doc}"
        );
    }
    rows_to_position(&sidecar.rows)
}

/// Host: both legs' rows in, the bootstrap verdict out.
#[test]
#[ignore = "host; needs both legs' rows sidecars"]
fn quality_verdict_decides_from_both_legs() {
    let out = PathBuf::from(require("FIFI_FLASH_NEXT_QUALITY_OUT"));
    let read = |arm: &str, rows: u32| -> Vec<PositionRow> {
        let bytes = std::fs::read(out.join(format!("rows-{arm}.json")))
            .unwrap_or_else(|error| panic!("rows-{arm}.json: {error}"));
        let sidecar: Sidecar = serde_json::from_slice(&bytes).expect("the sidecar json");
        check_sidecar(&sidecar, arm, rows)
    };
    let candidate = read("shared", 256);
    let reference = read("legacy", 64);
    let (nll_ub, top1_lb) = bounds(&candidate, &reference);
    let receipt = serde_json::json!({
        "nll_ub": nll_ub,
        "top1_lb": top1_lb,
        "margins": {"nll_ub": NLL_UB, "top1_lb": TOP1_LB},
        "bootstrap": {"draws": DRAWS, "seed": SEED, "clusters": CLUSTERS},
        "pass": passes(nll_ub, top1_lb),
    });
    std::fs::write(
        out.join("verdict.json"),
        serde_json::to_vec_pretty(&receipt).expect("serialize the verdict"),
    )
    .expect("write the verdict");
    println!("verdict: {receipt}");
    assert!(
        passes(nll_ub, top1_lb),
        "the shared prefill failed a registration margin: {receipt}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_corpus_matches_its_committed_manifest() {
        let manifest = documents();
        let classes: Vec<&str> = manifest.iter().map(|entry| entry.class.as_str()).collect();
        assert_eq!(classes, DOC_CLASSES.to_vec());
        assert_eq!(manifest[0].byte_length, 8505);
        let total: u64 = manifest.iter().map(|entry| entry.byte_length).sum();
        assert_eq!(total, 630_900, "the frozen corpus byte total");
    }

    #[test]
    fn the_row_math_is_a_stable_log_softmax() {
        // [0, ln2, 0]: logp(ln2) = ln2 - ln(e^0 + 2 + e^0) = ln2 - ln4.
        let row = [0.0_f32, std::f32::consts::LN_2, 0.0_f32];
        let (logprob_true, top1_correct) = score_row(&row, 1, 3);
        // probs [1/4, 1/2, 1/4]: the true token's logp is ln(1/2).
        assert!((logprob_true + std::f32::consts::LN_2).abs() < 1e-6);
        assert!(top1_correct);
        let (_, wrong) = score_row(&row, 0, 3);
        assert!(!wrong);
        // A large-magnitude row stays finite (the stable shift).
        let big = [1.0e4_f32, 1.0e4 + 1.0, 1.0e4];
        let (logprob, _) = score_row(&big, 1, 3);
        assert!(logprob.is_finite() && logprob > -2.0);
        // The slice to the real vocabulary ignores padding: the same row with
        // a padded tail scores identically at vocab = 3.
        let padded: Vec<f32> = row.iter().copied().chain([0.0_f32; 5]).collect();
        let (logprob_padded, _) = score_row(&padded, 1, 3);
        assert_eq!(logprob_padded, logprob_true);
    }

    /// A host stepper whose `logits(t)` row puts all mass on t+1: every scored
    /// position is a correct top-1 at logp ~ 0. It proves the row producer's
    /// cluster wiring (suffix position must NOT become the cluster).
    struct NextTokenStepper {
        resets: usize,
        context_fed: usize,
    }

    impl TokenStepper for NextTokenStepper {
        const OPERATION: &'static str = "quality_row_producer_test";
        fn feed(&mut self, _token: u32) -> std::result::Result<(), crate::EngError> {
            self.context_fed += 1;
            Ok(())
        }
        fn fed(&self) -> usize {
            self.context_fed
        }
        fn logits(&mut self, token: u32) -> std::result::Result<Vec<f32>, crate::EngError> {
            let mut row = vec![-30.0_f32; VOCAB_PIN as usize];
            row[(token + 1) as usize] = 0.0;
            Ok(row)
        }
        fn reset(&mut self) -> std::result::Result<(), crate::EngError> {
            self.resets += 1;
            self.context_fed = 0;
            Ok(())
        }
    }

    /// Host round trip: a sidecar built the way the leg builds one (stub
    /// stepper, the same 12x768 rows, the frozen token counts the leg now
    /// records) passes the verdict's own gate; a sidecar whose token counts
    /// were the truncated per-leg cuts - the shadowing the reader caught -
    /// is refused on the host, not after both card windows.
    #[test]
    fn the_verdict_refuses_truncated_token_counts_on_the_host() {
        let scored = {
            let mut stepper = NextTokenStepper {
                resets: 0,
                context_fed: 0,
            };
            let mut scored = Vec::new();
            for doc in 0..12u32 {
                let ids: Vec<u32> = (0..(64 + SUFFIX_IDS) as u32).collect();
                scored.extend(score_document(&mut stepper, doc, &ids, 64));
            }
            assert_eq!(stepper.resets, 12);
            scored
        };
        let leg = Sidecar {
            arm: "shared".into(),
            build: "unset".into(),
            pack: "round-trip".into(),
            id_sha: FROZEN_ID_SHA256.to_string(),
            pins: serde_json::to_value(expected_pins(256)).unwrap(),
            doc_tokens: FROZEN_TOKEN_COUNTS.to_vec(),
            rows: scored,
        };
        let bytes = serde_json::to_vec(&leg).expect("serialize the leg sidecar");
        let roundtrip: Sidecar = serde_json::from_slice(&bytes).expect("read the sidecar back");
        check_sidecar(&roundtrip, "shared", 256);

        let mut truncated = roundtrip.clone();
        truncated.doc_tokens = vec![64 + SUFFIX_IDS; 12];
        let refused = std::panic::catch_unwind(|| check_sidecar(&truncated, "shared", 256));
        assert!(
            refused.is_err(),
            "truncated counts must be refused by the verdict's own gate"
        );
    }

    #[test]
    fn the_row_producer_clusters_by_document_not_position() {
        let mut stepper = NextTokenStepper {
            resets: 0,
            context_fed: 0,
        };
        // Doc 7, an S-class cut: 64 context ids then 769 suffix ids.
        let ids: Vec<u32> = (0..(64 + SUFFIX_IDS) as u32).collect();
        let rows = score_document(&mut stepper, 7, &ids, 64);
        assert_eq!(rows.len(), SUFFIX_IDS - 1);
        assert!(
            rows.iter().all(|row| row.cluster == 7),
            "the cluster is the document"
        );
        assert!(rows.iter().all(|row| row.top1_correct));
        assert!(rows.iter().all(|row| row.logprob_true > -1e-4));
        assert_eq!(stepper.resets, 1, "one reset per document");
    }

    #[test]
    fn the_verdict_rule_passes_identical_and_fails_worse_arms() {
        let row = |cluster: u32, logprob: f32| PositionRow {
            cluster,
            logprob_true: logprob,
            top1_correct: true,
        };
        let same: Vec<PositionRow> = (0..12).flat_map(|k| vec![row(k, -1.0); 64]).collect();
        let (nll, top1) = bounds(&same, &same);
        assert!(nll.abs() < 1e-12 && top1.abs() < 1e-12 && passes(nll, top1));
        let worse: Vec<PositionRow> = (0..12).flat_map(|k| vec![row(k, -1.5); 64]).collect();
        let (nll_worse, _) = bounds(&worse, &same);
        assert!((nll_worse - 0.5).abs() < 1e-9 && !passes(nll_worse, 0.0));
    }
}
