//! Teacher-forced quality gate: compares the next-token distributions of the
//! legacy q8, tiled q8, and f16-KV configurations on a fixed corpus of repo
//! source (the committed testdata snapshot), with EXACT full-vocabulary KL.
//! One child per leg (`FLINT_TF_GATE_ARM` legacy|tiled|f16, `FLINT_TF_GATE_CLASS`
//! S|M|L|K16|K32); the arms differ ONLY in the tiled knob (legacy/tiled share the q8
//! profile; f16 never reads the knob), and each leg's receipt plus the
//! engine load line on the run log prove the arm. Legs are admitted-device
//! only and fail loudly, never skip.
//!
//! Plan: eight 1,024-token contexts (class S), two 4,096 (M), one 8,192 (L),
//! each followed by a 769-token scored suffix — 24,576 context tokens plus
//! 8,459 suffix tokens (11 x 769), 33,035 ids total, 768 scored positions per
//! segment, 8,448 per arm. Class S reads the corpus from id 0, M from 14,344,
//! L from 24,074. The downstream CE gap per position is logp_next_ref minus
//! logp_next (referee minus arm, true log-probabilities <= 0): positive when
//! the arm gives the actual token less probability than the referee.
//!
//! Exact KL mechanics: the f16 referee leg writes ONE full f32 log-probability
//! grid per class (positions x 150,144 x 4 B: S 3.69 GB = 3.44 GiB, M 0.92 GB
//! = 0.86 GiB, L 0.46 GB = 0.43 GiB) to a fresh exclusive 0600 file, hashed
//! while written, with an authenticating sidecar JSON; each q8 arm leg
//! validates the sidecar (class, geometry, corpus and run identity, model
//! root, encoding, byte count) before the engine load, streams the same grid
//! in the same position order, and verifies the streamed digest and exact
//! end-of-file before publishing its receipt. KL is exact f64 from the arm's
//! own full logits — no truncation anywhere. The grid lives only during its
//! class's legs; the run script deletes it after the class's final leg.

use super::*;
use crate::family::flint::{factory, kv_plan::KvStorage, settings};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufReader, Read as _, Write as _},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const SUFFIX: usize = 769;
const SCORED_PER_SEGMENT: usize = SUFFIX - 1;
const CORPUS_IDS: usize = 33_035;
/// Ids consumed by the K classes: both end at the committed snapshot's
/// file end (47,522), so their scored 768 positions are identical ids and
/// the K16-vs-K32 pair isolates context depth alone.
const K_END: usize = 47_522;
const SNAPSHOT_SHA: &str = "854bec635304dc0999ecd1183689699e4d19f3c6d67df2bc584cd99c67c2fcaf";
const VOCAB: usize = 150_144;
const GRID_ENCODING: &str = "le-f32";
/// The referee grid stores f32 log-probs; the mass of a legitimate row
/// drifts only through f32 rounding over the vocabulary, orders of magnitude
/// below this bar. A corrupted row (wrong encoding, NaN, truncation) lands
/// far outside it.
const GRID_MASS_TOL: f64 = 1e-2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SegmentClass {
    S,
    M,
    L,
    K16,
    K32,
}

impl SegmentClass {
    fn context(&self) -> usize {
        match self {
            SegmentClass::S => 1024,
            SegmentClass::M => 4096,
            SegmentClass::L => 8192,
            SegmentClass::K16 => 16_384,
            SegmentClass::K32 => 32_768,
        }
    }
    fn count(&self) -> usize {
        match self {
            SegmentClass::S => 8,
            SegmentClass::M => 2,
            SegmentClass::L | SegmentClass::K16 | SegmentClass::K32 => 1,
        }
    }
    /// First corpus id this class reads: S 0, M 8x1,793 = 14,344, L +2x4,865.
    /// K classes read back from the file end: K32 spans [13,985, K_END),
    /// K16 spans [30,369, K_END) — K32 nests K16's ids and carries all M
    /// and L ids plus the final 359 S ids (S is [0, 14,344)) as context,
    /// K16 overlaps only the last 2,666 L ids, and no S/M/L leg ever
    /// scored past 33,035.
    fn start(&self) -> usize {
        match self {
            SegmentClass::S => 0,
            SegmentClass::M => 8 * (1024 + SUFFIX),
            SegmentClass::L => 8 * (1024 + SUFFIX) + 2 * (4096 + SUFFIX),
            SegmentClass::K16 => K_END - (16_384 + SUFFIX),
            SegmentClass::K32 => K_END - (32_768 + SUFFIX),
        }
    }
    fn context_limit(&self) -> u32 {
        match self {
            SegmentClass::S => 4096,
            SegmentClass::M => 8192,
            SegmentClass::L => 16_384,
            SegmentClass::K16 => 32_768,
            // K32 reads 32,768 + 769 = 33,537 ids; 36,864 covers them with
            // chunk margin. 65,536 admitted to 2x over-provisioned KV and
            // failed the f16 referee's device admission by ~348 MiB on the
            // 24 GB card (need 24,619,116,804 vs 24,254,349,312 free).
            SegmentClass::K32 => 36_864,
        }
    }
    /// Run only the first `raw` segments (design: the duplicate leg re-runs
    /// S segments 0-1; absent means the whole class).
    fn segment_limit(raw: Option<&str>, class: Self) -> usize {
        match raw {
            None => class.count(),
            Some(text) => {
                let limit: usize = text
                    .parse()
                    .unwrap_or_else(|error| panic!("FLINT_TF_GATE_SEGMENTS: {error}"));
                assert!(
                    (1..=class.count()).contains(&limit),
                    "segment limit {limit} outside 1..={}",
                    class.count()
                );
                limit
            }
        }
    }
    /// The corpus cursor after `limit` segments of this class (the leg's
    /// endpoint): consecutive classes tile the plan without overlap.
    fn corpus_end(&self, limit: usize) -> usize {
        self.start() + limit * (self.context() + SUFFIX)
    }
    fn as_str(&self) -> &'static str {
        match self {
            SegmentClass::S => "S",
            SegmentClass::M => "M",
            SegmentClass::L => "L",
            SegmentClass::K16 => "K16",
            SegmentClass::K32 => "K32",
        }
    }
}

/// The receipt header every leg writes, extracted so a model-free test
/// can run the REAL serializer: `corpus_ids_len` must be the length of
/// the ids the leg hashed (`used.len()`), never a class constant.
struct ReceiptHeader<'a> {
    arm: &'a str,
    class: SegmentClass,
    profile: &'a str,
    armed: Option<bool>,
    storage: KvStorage,
    run_id: &'a str,
    segments: usize,
    corpus_ids_hash: &'a str,
    used: &'a [u32],
    started: u128,
    load_ms: f64,
    grid: Option<&'a str>,
}

impl ReceiptHeader<'_> {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "leg": "q8_tf_gate_teacher_forced",
            "arm": self.arm,
            "class": self.class.as_str(),
            "profile": self.profile,
            "armed": self.armed,
            "resolved_route": self.armed.map(|armed| self.profile == "q8" && armed),
            "kv_storage": format!("{:?}", self.storage),
            "expert_cache": "off (FIFI_FLINT_CUDA_EXPERT_CACHE=0 at the precondition)",
            "run_id": self.run_id,
            "segments_run": self.segments,
            "context_limit": self.class.context_limit(),
            "prefill_chunk_size": 8,
            "vocab": VOCAB,
            "corpus_snapshot_sha256": SNAPSHOT_SHA,
            "corpus_ids_sha256": self.corpus_ids_hash,
            "corpus_ids_len": self.used.len(),
            "started_utc_ms": self.started,
            "load_ms": self.load_ms,
            "grid": self.grid,
            "note": "the load line lives on the run log stderr; arms differ only in the tiled knob; CE gap = logp_next_ref - logp_next (positive when the arm is worse)",
        })
    }
}

/// The referee's authenticating sidecar: an arm leg accepts a grid only if
/// every field matches its own expectation, so a stale, edited, wrong-class,
/// or wrong-geometry grid cannot be scored.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct GridSidecar {
    leg: String,
    arm: String,
    class: String,
    run_id: String,
    model: String,
    encoding: String,
    vocab: usize,
    positions: usize,
    segments: usize,
    scored_per_segment: usize,
    corpus_snapshot_sha256: String,
    corpus_ids_sha256: String,
    bytes: u64,
    grid_sha256: String,
}

/// What an arm leg expects of the grid, derived from its own class, corpus,
/// run identity, and model root — never from the sidecar itself.
struct GridExpectation<'a> {
    class: SegmentClass,
    run_id: &'a str,
    model: &'a Path,
    vocab: usize,
    corpus_snapshot_sha256: &'a str,
    corpus_ids_sha256: &'a str,
}

fn check_sidecar(
    sidecar: &GridSidecar,
    expect: &GridExpectation,
) -> std::result::Result<(), String> {
    let model = expect
        .model
        .canonicalize()
        .map_err(|error| format!("model root canonicalise: {error}"))?;
    let fields = [
        (sidecar.leg.as_str(), "q8_tf_gate_teacher_forced", "leg"),
        (sidecar.arm.as_str(), "f16", "arm"),
        (sidecar.class.as_str(), expect.class.as_str(), "class"),
        (sidecar.run_id.as_str(), expect.run_id, "run identity"),
        (
            sidecar.model.as_str(),
            model.to_str().unwrap_or_default(),
            "model root",
        ),
        (sidecar.encoding.as_str(), GRID_ENCODING, "encoding"),
        (
            sidecar.corpus_snapshot_sha256.as_str(),
            expect.corpus_snapshot_sha256,
            "corpus snapshot",
        ),
        (
            sidecar.corpus_ids_sha256.as_str(),
            expect.corpus_ids_sha256,
            "corpus ids",
        ),
    ];
    for (got, want, name) in fields {
        if got != want {
            return Err(format!(
                "grid {name} mismatch: sidecar {got:?} != expected {want:?}"
            ));
        }
    }
    let geometry = [
        (sidecar.vocab, expect.vocab, "vocab"),
        (sidecar.segments, expect.class.count(), "segments"),
        (
            sidecar.scored_per_segment,
            SCORED_PER_SEGMENT,
            "scored per segment",
        ),
        (
            sidecar.positions,
            expect.class.count() * SCORED_PER_SEGMENT,
            "positions",
        ),
    ];
    for (got, want, name) in geometry {
        if got != want {
            return Err(format!("grid {name} mismatch: {got} != {want}"));
        }
    }
    if sidecar.bytes != (sidecar.positions * sidecar.vocab * 4) as u64 {
        return Err(format!(
            "grid byte count {} is not positions x vocab x 4",
            sidecar.bytes
        ));
    }
    Ok(())
}

/// Streamed, digested reader over the referee grid. Open checks the exact
/// byte count; `finish` checks exact end-of-file and the sidecar's digest, so
/// a trailing extra row or one changed byte both fail before any receipt is
/// published.
struct GridStream {
    file: BufReader<fs::File>,
    hasher: Sha256,
    rows_read: usize,
    rows_total: usize,
    digest: String,
}

impl GridStream {
    fn open(
        path: &Path,
        positions: usize,
        row_bytes: usize,
        digest: &str,
    ) -> std::io::Result<Self> {
        let len = fs::symlink_metadata(path)?.len();
        let want = (positions * row_bytes) as u64;
        if len != want {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("grid {path:?} is {len} bytes, expected exactly {want}"),
            ));
        }
        Ok(Self {
            file: BufReader::with_capacity(1 << 20, fs::File::open(path)?),
            hasher: Sha256::new(),
            rows_read: 0,
            rows_total: positions,
            digest: digest.to_owned(),
        })
    }

    fn next_row(&mut self, row_bytes: &mut [u8], row: &mut Vec<f32>) {
        assert!(
            self.rows_read < self.rows_total,
            "grid exhausted after {} rows",
            self.rows_total
        );
        self.file
            .read_exact(&mut row_bytes[..])
            .expect("grid row read");
        self.hasher.update(&row_bytes[..]);
        bytes_to_row(&row_bytes[..], row);
        self.rows_read += 1;
    }

    /// Read and hash the unread declared rows: a duplicate leg scores only
    /// its own rows, but the digest and end-of-file must still cover the
    /// whole grid before that leg publishes its receipt.
    fn drain(&mut self, row_bytes: &mut [u8], row: &mut Vec<f32>) {
        while self.rows_read < self.rows_total {
            self.next_row(row_bytes, row);
        }
    }
    fn finish(mut self) -> std::result::Result<(), String> {
        if self.rows_read != self.rows_total {
            return Err(format!(
                "streamed {} of {} rows",
                self.rows_read, self.rows_total
            ));
        }
        let mut tail = [0u8; 1];
        let extra = self.file.read(&mut tail).expect("grid tail read");
        if extra != 0 {
            return Err("grid has trailing bytes beyond the declared rows".to_owned());
        }
        let streamed = format!("{:x}", self.hasher.finalize());
        if streamed != self.digest {
            return Err(format!(
                "grid digest mismatch: streamed {streamed} != sidecar {}",
                self.digest
            ));
        }
        Ok(())
    }
}

fn snapshot_bytes() -> &'static [u8] {
    let bytes = include_bytes!("testdata/tf-gate-corpus.txt");
    assert_eq!(
        format!("{:x}", Sha256::digest(bytes)),
        SNAPSHOT_SHA,
        "the committed corpus snapshot changed"
    );
    bytes
}

pub(super) fn corpus_ids(tokenizer: &crate::tokenizer::Tokenizer) -> Vec<u32> {
    let text = std::str::from_utf8(snapshot_bytes()).expect("snapshot is valid UTF-8");
    tokenizer.encode(text).expect("corpus tokenisation")
}

fn ids_sha(ids: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// f32 little-endian bytes of a row (no unsafe casts, no new dependencies).
fn row_to_bytes(row: &[f32], buffer: &mut Vec<u8>) {
    buffer.clear();
    buffer.reserve(row.len() * 4);
    for value in row {
        buffer.extend_from_slice(&value.to_le_bytes());
    }
}

fn bytes_to_row(bytes: &[u8], row: &mut Vec<f32>) {
    assert_eq!(bytes.len() % 4, 0);
    row.clear();
    row.reserve(bytes.len() / 4);
    for chunk in bytes.as_chunks::<4>().0.iter() {
        row.push(f32::from_le_bytes(*chunk));
    }
}

fn row_lse(logits: &[f32]) -> f64 {
    let max = f64::from(logits.iter().copied().fold(f32::MIN, f32::max));
    let sum: f64 = logits
        .iter()
        .map(|&value| (f64::from(value) - max).exp())
        .sum();
    max + sum.ln()
}

/// Normalised log-probabilities of one full-vocab logits row, f64 -> f32.
fn logprob_grid_row(logits: &[f32]) -> Vec<f32> {
    let lse = row_lse(logits);
    assert!(lse.is_finite(), "non-finite lse");
    logits
        .iter()
        .map(|&value| (f64::from(value) - lse) as f32)
        .collect()
}

/// Exact f64 KL(P || Q): P from the referee's stored f32 log-probs, Q from
/// the arm's own logits with its own f64 lse. One fixed summation order, so
/// the duplicate leg must reproduce every value bitwise. Degenerate inputs
/// are rejected before they can become a receipt row: non-finite reference
/// values, a reference row that is not a distribution, and an arm probability
/// of exactly zero where the referee has mass (an infinite KL) — each fails
/// loudly rather than clamping or emitting a silent NaN that JSON would
/// serialise as null.
fn kl_validated(ref_logp: &[f32], logits: &[f32]) -> std::result::Result<f64, String> {
    if ref_logp.len() != logits.len() {
        return Err(format!(
            "vocab mismatch {} vs {}",
            ref_logp.len(),
            logits.len()
        ));
    }
    for &lp in ref_logp {
        if !lp.is_finite() {
            return Err(format!("non-finite reference log-prob {lp}"));
        }
    }
    let lse = row_lse(logits);
    if !lse.is_finite() {
        return Err(format!("non-finite arm lse {lse}"));
    }
    let mut mass = 0.0f64;
    let mut kl = 0.0f64;
    for (&lp_ref, &logit) in ref_logp.iter().zip(logits.iter()) {
        let p = f64::from(lp_ref).exp();
        mass += p;
        // A referee probability so small it underflows f64 contributes 0.
        if p == 0.0 {
            continue;
        }
        let q = (f64::from(logit) - lse).exp();
        if q == 0.0 {
            return Err(
                "arm probability underflows to zero where the referee has mass: infinite KL"
                    .to_owned(),
            );
        }
        kl += p * (f64::from(lp_ref) - (f64::from(logit) - lse));
    }
    if (mass - 1.0).abs() > GRID_MASS_TOL {
        return Err(format!(
            "reference row mass {mass}, expected 1 +/- {GRID_MASS_TOL}"
        ));
    }
    if !kl.is_finite() {
        return Err(format!("non-finite KL {kl}"));
    }
    Ok(kl)
}

/// Create the referee grid exclusively: fails if the destination exists (or
/// is a symlink), never truncates, and is 0600 from the first byte.
/// Deletes a newly created referee grid unless publication disarms it: a
/// panic or write failure mid-run leaves no partial grid behind.
struct GridGuard(Option<PathBuf>);
impl GridGuard {
    fn disarm(mut self) {
        self.0 = None;
    }
}
impl Drop for GridGuard {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

fn create_grid_exclusive(path: &Path) -> std::io::Result<fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

fn publish(mut temp: tempfile::NamedTempFile, dest: &Path) -> std::io::Result<()> {
    temp.flush()?;
    temp.persist_noclobber(dest)
        .map(|_| ())
        .map_err(std::io::Error::from)
}

#[test]
fn tf_gate_segment_plan_counts() {
    // Independent literals: the class geometry is spelled out here, not
    // derived from the helpers it checks.
    assert_eq!(SCORED_PER_SEGMENT, 768);
    assert_eq!(SegmentClass::S.context(), 1024);
    assert_eq!(SegmentClass::M.context(), 4096);
    assert_eq!(SegmentClass::L.context(), 8192);
    assert_eq!(
        [SegmentClass::S, SegmentClass::M, SegmentClass::L]
            .iter()
            .map(|class| class.count() * SCORED_PER_SEGMENT)
            .sum::<usize>(),
        8448
    );
    assert_eq!(SegmentClass::S.start(), 0);
    assert_eq!(SegmentClass::M.start(), 8 * (1024 + SUFFIX));
    assert_eq!(SegmentClass::M.start(), 14_344);
    assert_eq!(
        SegmentClass::L.start(),
        SegmentClass::M.start() + 2 * (4096 + SUFFIX)
    );
    assert_eq!(SegmentClass::L.start(), 24_074);
    assert_eq!(SegmentClass::L.start() + 8192 + SUFFIX, CORPUS_IDS);
    assert_eq!(24_576 + 11 * SUFFIX, CORPUS_IDS, "context + suffix ids");
    assert_eq!(SegmentClass::K16.context(), 16_384);
    assert_eq!(SegmentClass::K32.context(), 32_768);
    assert_eq!(SegmentClass::K16.context_limit(), 32_768);
    assert_eq!(SegmentClass::K32.context_limit(), 36_864);
    assert_eq!(SegmentClass::K16.start(), 47_522 - (16_384 + SUFFIX));
    assert_eq!(SegmentClass::K32.start(), 47_522 - (32_768 + SUFFIX));
    assert_eq!(SegmentClass::K16.start(), 30_369);
    assert_eq!(SegmentClass::K32.start(), 13_985);
    assert!(SegmentClass::K32.start() < SegmentClass::K16.start());
    assert_eq!(K_END - SUFFIX, 46_753);
    const _: () = assert!(K_END - SUFFIX > CORPUS_IDS, "scored ids are fresh");
}

/// A revert to the CORPUS_IDS constant for `corpus_ids_len` fails the K
/// assertions below.
#[test]
fn tf_gate_receipt_header_publishes_the_hashed_length() {
    for (class, len) in [
        (SegmentClass::S, CORPUS_IDS),
        (SegmentClass::M, CORPUS_IDS),
        (SegmentClass::L, CORPUS_IDS),
        (SegmentClass::K16, K_END),
        (SegmentClass::K32, K_END),
    ] {
        let used = vec![0u32; len];
        let header = ReceiptHeader {
            arm: "f16",
            class,
            profile: "f16",
            armed: None,
            storage: KvStorage::F16,
            run_id: "t",
            segments: 1,
            corpus_ids_hash: "x",
            used: &used,
            started: 0,
            load_ms: 0.0,
            grid: None,
        }
        .to_json();
        assert_eq!(
            header["corpus_ids_len"].as_u64(),
            Some(len as u64),
            "{class:?} must publish its own hash scope"
        );
        assert_eq!(header["class"].as_str(), Some(class.as_str()));
    }
}

#[test]
fn tf_gate_logprob_and_kl_round_trip() {
    let logits: Vec<f32> = (0..64).map(|i| ((i % 13) as f32 - 6.0) / 8.0).collect();
    let row = logprob_grid_row(&logits);
    assert_eq!(row.len(), logits.len());
    let mass: f64 = row.iter().map(|&lp| f64::from(lp).exp()).sum();
    assert!((mass - 1.0).abs() < 1e-5, "row mass {mass}");
    // Self-KL is within the f32 storage round-trip of zero, not exactly zero:
    // the grid stores f32 log-probs, the arm path recomputes f64 from logits.
    assert!(kl_validated(&row, &logits).unwrap().abs() < 1e-6);
    // A genuinely different distribution (a uniform logit shift is
    // softmax-invariant) gives strictly positive KL; identical inputs are
    // bitwise reproducible.
    let worse: Vec<f32> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.1 * ((i % 3) as f32 - 1.0))
        .collect();
    let kl = kl_validated(&row, &worse).unwrap();
    assert!(kl > 1e-4, "perturbed KL {kl}");
    assert_eq!(kl, kl_validated(&row, &worse).unwrap());
    let mut buffer = Vec::new();
    row_to_bytes(&row, &mut buffer);
    let mut back = Vec::new();
    bytes_to_row(&buffer, &mut back);
    assert_eq!(row, back);
}

/// Hand-derived asymmetric oracle: KL(P||Q) != KL(Q||P) for these P and Q,
/// so a reversed implementation cannot pass.
#[test]
fn tf_gate_kl_direction_matches_hand_arithmetic() {
    // P = [.6 .3 .05 .05] as logits; Q uniform over 4 tokens.
    let p: Vec<f32> = [0.6f64, 0.3, 0.05, 0.05]
        .iter()
        .map(|v| v.ln() as f32)
        .collect();
    let q: Vec<f32> = vec![0.25f64.ln() as f32; 4];
    let row_p = logprob_grid_row(&p);
    let row_q = logprob_grid_row(&q);
    let kl_pq = kl_validated(&row_p, &q).unwrap();
    let kl_qp = kl_validated(&row_q, &p).unwrap();
    let hand_pq =
        0.6f64 * (0.6 / 0.25f64).ln() + 0.3 * (0.3 / 0.25f64).ln() + 0.1 * (0.05 / 0.25f64).ln();
    let hand_qp =
        0.5f64 * (0.25 / 0.05f64).ln() + 0.25 * (0.25 / 0.3f64).ln() + 0.25 * (0.25 / 0.6f64).ln();
    assert!(
        (kl_pq - hand_pq).abs() < 1e-6,
        "KL(P||Q) {kl_pq} vs {hand_pq}"
    );
    assert!(
        (kl_qp - hand_qp).abs() < 1e-6,
        "KL(Q||P) {kl_qp} vs {hand_qp}"
    );
    assert!((kl_pq - kl_qp).abs() > 1e-3, "direction must matter");
    assert!((f64::from(p[0]) - row_lse(&p) - 0.6f64.ln()).abs() < 1e-6);
    assert!((f64::from(q[0]) - row_lse(&q) - 0.25f64.ln()).abs() < 1e-6);
}

#[test]
fn tf_gate_kl_rejects_degenerate_reference_rows() {
    let good = logprob_grid_row(&[0.5f32, 0.25, -0.25, -0.5]);
    // Zero support: every reference probability underflows to zero.
    let zero_row: Vec<f32> = vec![-890.0f32; 4];
    let mut nan_row = good.clone();
    nan_row[1] = f32::NAN;
    let mut inf_row = good.clone();
    inf_row[2] = f32::INFINITY;
    let mut bad_mass = good.clone();
    for lp in &mut bad_mass {
        *lp += 0.7f64.ln() as f32;
    }
    let mut starved = [0.0f32; 4];
    starved[0] = -890.0;
    for (name, row) in [
        ("zero support", &zero_row),
        ("nan", &nan_row),
        ("inf", &inf_row),
        ("bad mass", &bad_mass),
    ] {
        assert!(kl_validated(row, &good).is_err(), "{name} must be rejected");
    }
    assert!(
        kl_validated(&good, &starved).is_err(),
        "zero arm probability must be rejected as infinite KL"
    );
    assert!(kl_validated(&good, &good).is_ok());
}

#[test]
fn tf_gate_publish_never_replaces() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("receipt");
    fs::write(&dest, b"original").unwrap();
    let mut temp = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
    writeln!(temp, "new content").unwrap();
    assert!(publish(temp, &dest).is_err(), "must refuse to replace");
    assert_eq!(fs::read(&dest).unwrap(), b"original");
    let fresh = dir.path().join("fresh");
    let mut temp = tempfile::NamedTempFile::new_in(dir.path()).unwrap();
    writeln!(temp, "content").unwrap();
    publish(temp, &fresh).unwrap();
    assert_eq!(fs::read(&fresh).unwrap(), b"content\n");
}

#[test]
fn tf_gate_grid_create_is_exclusive_and_private() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let grid = dir.path().join("grid");
    let file = create_grid_exclusive(&grid).unwrap();
    drop(file);
    let mode = fs::metadata(&grid).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    fs::write(&grid, b"existing").unwrap();
    assert!(create_grid_exclusive(&grid).is_err());
    assert_eq!(fs::read(&grid).unwrap(), b"existing");
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(dir.path().join("missing"), &link).unwrap();
    assert!(create_grid_exclusive(&link).is_err());
}

/// A partial run is legal only for an ARM duplicate: a referee that stops
/// early publishes a grid no arm will accept.
fn referee_limit_ok(arm: &str, limit: usize, class: SegmentClass) -> bool {
    arm != "f16" || limit == class.count()
}

#[test]
fn tf_gate_segment_limit_parsing() {
    use SegmentClass as Sc;
    assert_eq!(Sc::segment_limit(None, Sc::S), 8);
    assert_eq!(Sc::segment_limit(None, Sc::L), 1);
    assert_eq!(Sc::segment_limit(Some("2"), Sc::S), 2);
    assert_eq!(Sc::segment_limit(Some("1"), Sc::L), 1);
    for bad in ["0", "9", "x", ""] {
        assert!(
            std::panic::catch_unwind(|| Sc::segment_limit(Some(bad), Sc::S)).is_err(),
            "limit {bad:?} must be rejected"
        );
    }
    assert!(!referee_limit_ok("f16", 2, Sc::S), "partial referee S");
    assert!(referee_limit_ok("f16", 8, Sc::S));
    assert!(referee_limit_ok("f16", 1, Sc::L));
    assert!(referee_limit_ok("legacy", 2, Sc::S), "arm duplicate");
    assert!(referee_limit_ok("tiled", 1, Sc::L));
}

#[test]
fn tf_gate_class_endpoints_tile_the_plan() {
    assert_eq!(SegmentClass::S.corpus_end(8), 14_344, "S endpoint");
    assert_eq!(SegmentClass::M.corpus_end(2), 24_074, "M endpoint");
    assert_eq!(SegmentClass::L.corpus_end(1), CORPUS_IDS, "L endpoint");
    assert_eq!(SegmentClass::K16.corpus_end(1), K_END, "K16 endpoint");
    assert_eq!(SegmentClass::K32.corpus_end(1), K_END, "K32 endpoint");
    assert_eq!(K_END, 47_522, "K classes end at the file end");
    assert_eq!(
        SegmentClass::S.corpus_end(2),
        3_586,
        "the duplicate leg reads two S segments"
    );
    assert_eq!(SegmentClass::S.corpus_end(8), SegmentClass::M.start());
    assert_eq!(SegmentClass::M.corpus_end(2), SegmentClass::L.start());
}

#[test]
fn tf_gate_snapshot_sha_is_committed() {
    assert_eq!(
        format!("{:x}", Sha256::digest(snapshot_bytes())),
        SNAPSHOT_SHA
    );
}

/// A tiny synthetic grid (vocab 64) with a correct sidecar so each negative
/// case can mutate exactly one thing.
fn synthetic_grid(dir: &Path, vocab: usize) -> (PathBuf, GridSidecar) {
    let positions = SegmentClass::S.count() * SCORED_PER_SEGMENT;
    let logits: Vec<f32> = (0..vocab).map(|i| ((i % 7) as f32 - 3.0) / 4.0).collect();
    let row = logprob_grid_row(&logits);
    let mut buffer = Vec::new();
    row_to_bytes(&row, &mut buffer);
    let mut hasher = Sha256::new();
    let grid = dir.join("grid.bin");
    let mut file = create_grid_exclusive(&grid).unwrap();
    for _ in 0..positions {
        file.write_all(&buffer).unwrap();
        hasher.update(&buffer);
    }
    drop(file);
    let sidecar = GridSidecar {
        leg: "q8_tf_gate_teacher_forced".to_owned(),
        arm: "f16".to_owned(),
        class: "S".to_owned(),
        run_id: "run-1".to_owned(),
        model: dir.canonicalize().unwrap().to_str().unwrap().to_owned(),
        encoding: GRID_ENCODING.to_owned(),
        vocab,
        positions,
        segments: positions.div_ceil(SCORED_PER_SEGMENT),
        scored_per_segment: SCORED_PER_SEGMENT,
        corpus_snapshot_sha256: "snap".to_owned(),
        corpus_ids_sha256: "ids".to_owned(),
        bytes: (positions * vocab * 4) as u64,
        grid_sha256: format!("{:x}", hasher.finalize()),
    };
    (grid, sidecar)
}

fn expect_for(sidecar: &GridSidecar) -> GridExpectation<'_> {
    GridExpectation {
        class: SegmentClass::S,
        run_id: &sidecar.run_id,
        model: Path::new(&sidecar.model),
        vocab: sidecar.vocab,
        corpus_snapshot_sha256: &sidecar.corpus_snapshot_sha256,
        corpus_ids_sha256: &sidecar.corpus_ids_sha256,
    }
}

#[test]
fn tf_gate_grid_guard_removes_partial_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let grid = dir.path().join("guard-grid");
    fs::write(&grid, b"partial-bytes").unwrap();
    let result = std::panic::catch_unwind(|| {
        let _guard = GridGuard(Some(grid.clone()));
        fs::write(&grid, b"more-partial-bytes").unwrap();
        panic!("injected referee write-path failure");
        #[allow(unreachable_code)]
        _guard.disarm();
    });
    assert!(result.is_err(), "the injected failure must have panicked");
    assert!(
        !grid.exists(),
        "a failed referee run must leave no partial grid"
    );
    fs::write(&grid, b"complete-grid").unwrap();
    let guard = GridGuard(Some(grid.clone()));
    guard.disarm();
    assert!(grid.exists(), "a disarmed guard keeps the published grid");
    assert!(
        fs::read(&grid).unwrap() == b"complete-grid",
        "the guard never rewrites the grid it protects"
    );
}

#[test]
fn tf_gate_sidecar_rejects_every_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let (grid, sidecar) = synthetic_grid(dir.path(), 64);
    let mut buffer = vec![0u8; 64 * 4];
    let mut row = Vec::new();
    assert!(check_sidecar(&sidecar, &expect_for(&sidecar)).is_ok());
    let mut cases: Vec<(String, GridSidecar)> = Vec::new();
    let mut s = sidecar.clone();
    s.class = "M".to_owned();
    cases.push(("wrong class".into(), s));
    let mut s = sidecar.clone();
    s.run_id = "run-0".to_owned();
    cases.push(("stale run identity".into(), s));
    let mut s = sidecar.clone();
    s.encoding = "be-f32".to_owned();
    cases.push(("wrong encoding".into(), s));
    let mut s = sidecar.clone();
    s.corpus_ids_sha256 = "other".to_owned();
    cases.push(("wrong corpus ids".into(), s));
    let mut s = sidecar.clone();
    s.model = "/elsewhere".to_owned();
    cases.push(("wrong model root".into(), s));
    let mut s = sidecar.clone();
    s.positions += 1;
    cases.push(("wrong positions".into(), s));
    let mut s = sidecar.clone();
    s.segments = 7;
    cases.push(("wrong segments".into(), s));
    let mut s = sidecar.clone();
    s.leg = "tiled".to_owned();
    cases.push(("wrong leg".into(), s));
    let mut s = sidecar.clone();
    s.arm = "legacy".to_owned();
    cases.push(("wrong arm".into(), s));
    let mut s = sidecar.clone();
    s.vocab += 1;
    cases.push(("wrong vocab".into(), s));
    let mut s = sidecar.clone();
    s.scored_per_segment -= 1;
    cases.push(("wrong scored rows".into(), s));
    let mut s = sidecar.clone();
    s.corpus_snapshot_sha256 = "other".to_owned();
    cases.push(("wrong corpus snapshot".into(), s));
    let mut s = sidecar.clone();
    s.bytes += 4;
    cases.push(("wrong byte length".into(), s));
    let mut s = sidecar.clone();
    s.bytes += 4;
    cases.push(("wrong byte length".into(), s));
    for (name, s) in cases {
        assert!(
            check_sidecar(&s, &expect_for(&sidecar)).is_err(),
            "{name} must be rejected"
        );
    }
    // The grid digest is not a check_sidecar field: the leg hands it to the
    // streaming reader, so a tampered digest must fail there instead.
    let mut s = sidecar.clone();
    s.grid_sha256 = "other".to_owned();
    let mut stream = GridStream::open(&grid, s.positions, 64 * 4, &s.grid_sha256).unwrap();
    for _ in 0..s.positions {
        stream.next_row(&mut buffer, &mut row);
    }
    assert!(stream.finish().is_err(), "wrong grid digest must fail");
    // Wrong class with the OTHER class's geometry is caught by the geometry
    // check too (S has 8 segments, M claims 2).
    {
        let mut s = sidecar.clone();
        s.class = "M".to_owned();
        s.segments = 2;
        s.positions = 2 * SCORED_PER_SEGMENT;
        assert!(check_sidecar(&s, &expect_for(&sidecar)).is_err());
    }
    drop(grid);
}

#[test]
fn tf_gate_grid_stream_detects_corruption() {
    let dir = tempfile::tempdir().unwrap();
    let (grid, sidecar) = synthetic_grid(dir.path(), 64);
    let positions = sidecar.positions;
    let digest = sidecar.grid_sha256.clone();
    let row_bytes = 64 * 4;
    let mut buffer = vec![0u8; row_bytes];
    let mut row = Vec::new();

    let mut stream = GridStream::open(&grid, positions, row_bytes, &digest).unwrap();
    for _ in 0..positions {
        stream.next_row(&mut buffer, &mut row);
    }
    stream.finish().unwrap();

    // One changed byte: opens (same length), fails on the digest at finish.
    let mut bytes = fs::read(&grid).unwrap();
    bytes[7] ^= 0x01;
    let mutated = dir.path().join("one-byte");
    fs::write(&mutated, &bytes).unwrap();
    let mut stream = GridStream::open(&mutated, positions, row_bytes, &digest).unwrap();
    for _ in 0..positions {
        stream.next_row(&mut buffer, &mut row);
    }
    assert!(stream.finish().is_err(), "one changed byte must fail");

    let mut bytes = fs::read(&grid).unwrap();
    let head = bytes[..256].to_vec();
    bytes.extend_from_slice(&head);
    let trailing = dir.path().join("trailing");
    fs::write(&trailing, &bytes).unwrap();
    assert!(GridStream::open(&trailing, positions, row_bytes, &digest).is_err());
    let short = dir.path().join("short");
    fs::write(&short, &fs::read(&grid).unwrap()[..1024]).unwrap();
    assert!(GridStream::open(&short, positions, row_bytes, &digest).is_err());
    assert!(GridStream::open(&dir.path().join("absent"), positions, row_bytes, &digest).is_err());

    let mut stream = GridStream::open(&grid, positions, row_bytes, "deadbeef").unwrap();
    for _ in 0..positions {
        stream.next_row(&mut buffer, &mut row);
    }
    assert!(stream.finish().is_err(), "digest mismatch must fail");

    let partial = 2 * SCORED_PER_SEGMENT;
    let mut stream = GridStream::open(&grid, positions, row_bytes, &digest).unwrap();
    for _ in 0..partial {
        stream.next_row(&mut buffer, &mut row);
    }
    stream.drain(&mut buffer, &mut row);
    stream.finish().unwrap();
    let mut bytes = fs::read(&grid).unwrap();
    let unread_at = partial * row_bytes + 5;
    bytes[unread_at] ^= 0x01;
    let unread_corrupt = dir.path().join("unread-corrupt");
    fs::write(&unread_corrupt, &bytes).unwrap();
    let mut stream = GridStream::open(&unread_corrupt, positions, row_bytes, &digest).unwrap();
    for _ in 0..partial {
        stream.next_row(&mut buffer, &mut row);
    }
    stream.drain(&mut buffer, &mut row);
    assert!(
        stream.finish().is_err(),
        "corruption in the drained rows must fail the digest"
    );
    let mut stream = GridStream::open(&grid, positions, row_bytes, "deadbeef").unwrap();
    for _ in 0..partial {
        stream.next_row(&mut buffer, &mut row);
    }
    stream.drain(&mut buffer, &mut row);
    assert!(
        stream.finish().is_err(),
        "a tampered digest must fail even for a partial read"
    );

    // Trailing bytes appended AFTER open: the opening length check cannot
    // see them, so finish's tail read must reject them. The appended
    // variant lives on its own path so later cases keep a pristine grid.
    let late_trailing = dir.path().join("trailing-late");
    fs::write(&late_trailing, fs::read(&grid).unwrap()).unwrap();
    let mut stream = GridStream::open(&late_trailing, positions, row_bytes, &digest).unwrap();
    {
        use std::io::Write as _;
        let mut late = fs::OpenOptions::new()
            .append(true)
            .open(&late_trailing)
            .unwrap();
        late.write_all(&[0u8]).unwrap();
    }
    for _ in 0..positions {
        stream.next_row(&mut buffer, &mut row);
    }
    assert!(
        stream.finish().is_err(),
        "one byte appended after open must fail finish"
    );

    let mut stream = GridStream::open(&grid, positions, row_bytes, &digest).unwrap();
    for _ in 0..positions {
        stream.next_row(&mut buffer, &mut row);
    }
    let overread = std::panic::AssertUnwindSafe(|| {
        let mut stream = stream;
        stream.next_row(&mut buffer, &mut row);
    });
    let panic = std::panic::catch_unwind(overread)
        .expect_err("reading past the last declared row must panic");
    let text = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or_default();
    assert!(
        text.contains("grid exhausted"),
        "the exhaustion guard must panic with its own message, got {text:?}"
    );
    let mut stream = GridStream::open(&grid, positions, row_bytes, &digest).unwrap();
    stream.next_row(&mut buffer, &mut row);
    assert!(
        stream.finish().is_err(),
        "finish must refuse a stream whose declared rows are not all read"
    );
}

#[test]
#[ignore = "needs the FLINT_QUAL_MODEL weights root: run on the measurement host"]
fn tf_gate_corpus_tokenises_to_the_plan() {
    let root =
        PathBuf::from(std::env::var_os("FLINT_QUAL_MODEL").expect("FLINT_QUAL_MODEL weights root"));
    let bundle = factory::load_tokenizer(&root).expect("qualification tokenizer");
    assert_eq!(
        bundle.vocab_size() as usize,
        VOCAB,
        "tokenizer vocab changed"
    );
    let ids = corpus_ids(bundle.tokenizer());
    assert!(ids.len() >= CORPUS_IDS, "corpus {} ids", ids.len());
    assert!(
        ids.len() >= K_END,
        "K classes need {K_END} ids, corpus has {}",
        ids.len()
    );
    println!(
        "tf-gate corpus: {} ids (need {CORPUS_IDS}), sha {}",
        ids.len(),
        ids_sha(&ids)
    );
}

// The explicit-rollback precondition panics before any device need, so the
// empty and "off" knob values are provable in a no-GPU child process.
#[test]
fn tf_gate_legacy_arm_rejects_non_explicit_rollback_values() {
    use std::process::Command;
    for value in ["", "off"] {
        let out = Command::new(std::env::current_exe().expect("exe"))
            .arg("--exact")
            .arg(concat!(
                "family::flint::cuda_executor::tf_gate_tests::",
                "q8_tf_gate_teacher_forced_leg"
            ))
            .arg("--ignored")
            .arg("--test-threads=1")
            .arg("--quiet")
            .env("FLINT_TF_GATE_ARM", "legacy")
            .env("FLINT_TF_GATE_CLASS", "S")
            .env("FLINT_TF_GATE_RUN", "rollback-matrix")
            .env("FIFI_FLINT_CUDA_Q8_TILED", value)
            .output()
            .expect("child run");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            !out.status.success()
                && text.contains("legacy arm requires the explicit rollback value"),
            "legacy arm must reject the non-explicit rollback value {value:?}: {}\n{text}",
            out.status
        );
    }
}

/// One admitted-device scoring leg. One leg per arm and class; the referee
/// (f16) runs first per class because it writes the grid the q8 arms read.
#[test]
#[ignore = "requires an admitted CUDA device: one leg = one arm x one class"]
fn q8_tf_gate_teacher_forced_leg() {
    let arm = std::env::var("FLINT_TF_GATE_ARM").expect("FLINT_TF_GATE_ARM legacy|tiled|f16");
    let class = match std::env::var("FLINT_TF_GATE_CLASS")
        .expect("FLINT_TF_GATE_CLASS S|M|L|K16|K32")
        .as_str()
    {
        "S" => SegmentClass::S,
        "M" => SegmentClass::M,
        "L" => SegmentClass::L,
        "K16" => SegmentClass::K16,
        "K32" => SegmentClass::K32,
        other => panic!("unknown FLINT_TF_GATE_CLASS {other}"),
    };
    let run_id = std::env::var("FLINT_TF_GATE_RUN").expect("FLINT_TF_GATE_RUN run identity");
    assert!(
        !run_id.trim().is_empty() && run_id.len() <= 64,
        "run identity"
    );
    let segment_limit = SegmentClass::segment_limit(
        std::env::var("FLINT_TF_GATE_SEGMENTS").ok().as_deref(),
        class,
    );
    assert!(
        referee_limit_ok(&arm, segment_limit, class),
        "the referee must write its whole class: FLINT_TF_GATE_SEGMENTS \
         applies to arm duplicates only"
    );
    let (profile, armed_record, storage) = match arm.as_str() {
        "legacy" => {
            let knob = std::env::var("FIFI_FLINT_CUDA_Q8_TILED").unwrap_or_default();
            assert!(
                knob == "0",
                "legacy arm requires the explicit rollback value '0' since the tiled route defaulted ON (got {knob:?})"
            );
            ("q8", Some(settings::q8_tiled_armed()), KvStorage::Q8G32)
        }
        "tiled" => {
            assert_eq!(
                std::env::var("FIFI_FLINT_CUDA_Q8_TILED").as_deref(),
                Ok("1"),
                "tiled arm must arm the knob explicitly"
            );
            ("q8", Some(settings::q8_tiled_armed()), KvStorage::Q8G32)
        }
        "f16" => {
            assert!(
                std::env::var_os("FIFI_FLINT_CUDA_Q8_TILED").is_none(),
                "f16 referee must not even set the tiled knob"
            );
            ("f16", None, KvStorage::F16)
        }
        other => panic!("unknown FLINT_TF_GATE_ARM {other}"),
    };
    if let Some(armed) = armed_record {
        assert_eq!(
            profile == "q8" && armed,
            arm == "tiled",
            "resolved route disagrees with the arm"
        );
    }
    assert_eq!(
        std::env::var("INFERENCE_CUDA_REQUIRE_DEVICE").as_deref(),
        Ok("1"),
        "an admitted device is required; this leg never silently skips"
    );
    assert_eq!(
        std::env::var("FIFI_FLINT_CUDA_EXPERT_CACHE").as_deref(),
        Ok("0"),
        "the expert host-cache changes the loading path: set FIFI_FLINT_CUDA_EXPERT_CACHE=0"
    );
    let root =
        PathBuf::from(std::env::var_os("FLINT_QUAL_MODEL").expect("FLINT_QUAL_MODEL weights root"));
    let receipt_path = PathBuf::from(
        std::env::var_os("FLINT_TF_GATE_RECEIPT").expect("FLINT_TF_GATE_RECEIPT path"),
    );
    assert!(receipt_path.is_absolute());
    assert!(
        matches!(fs::symlink_metadata(&receipt_path), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
        "receipt {receipt_path:?} must not exist before the leg"
    );
    let grid_path =
        PathBuf::from(std::env::var_os("FLINT_TF_GATE_GRID").expect("FLINT_TF_GATE_GRID path"));
    let writing_grid = arm == "f16";
    if writing_grid {
        assert!(
            matches!(fs::symlink_metadata(&grid_path), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
            "grid {grid_path:?} must not exist before the referee leg"
        );
    } else {
        assert!(grid_path.is_file(), "referee grid {grid_path:?} must exist");
    }
    let stamp = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis()
    };
    let started = stamp();

    let bundle = factory::load_tokenizer(&root).expect("qualification tokenizer");
    assert_eq!(
        bundle.vocab_size() as usize,
        VOCAB,
        "tokenizer vocab changed"
    );
    let ids = corpus_ids(bundle.tokenizer());
    assert!(ids.len() >= CORPUS_IDS, "corpus {} ids", ids.len());
    // K legs hash to their own endpoint: S/M/L retain their existing
    // 33,035-id hash binding; K legs bind the ids they actually read.
    let hash_end = match class {
        SegmentClass::K16 | SegmentClass::K32 => K_END,
        SegmentClass::S | SegmentClass::M | SegmentClass::L => CORPUS_IDS,
    };
    let used = &ids[..hash_end];
    let corpus_ids_hash = ids_sha(used);

    // Arms authenticate the referee grid (sidecar fields, exact byte count)
    // BEFORE the engine load, so a stale or edited grid fails cheaply.
    let mut grid_stream = None;
    if !writing_grid {
        let sidecar_text = fs::read_to_string(grid_path.with_extension("json"))
            .expect("referee sidecar must exist next to the grid");
        let sidecar: GridSidecar =
            serde_json::from_str(&sidecar_text).expect("referee sidecar JSON");
        let expectation = GridExpectation {
            class,
            run_id: &run_id,
            model: &root,
            vocab: VOCAB,
            corpus_snapshot_sha256: SNAPSHOT_SHA,
            corpus_ids_sha256: &corpus_ids_hash,
        };
        check_sidecar(&sidecar, &expectation)
            .unwrap_or_else(|error| panic!("referee grid rejected: {error}"));
        grid_stream = Some(
            GridStream::open(
                &grid_path,
                sidecar.positions,
                VOCAB * 4,
                &sidecar.grid_sha256,
            )
            .expect("grid open"),
        );
    }

    let options = FlintOptions {
        context_limit: class.context_limit(),
        kv_profile: profile.to_owned(),
        prefill_chunk_size: 8,

        cpu_moe_offload_layers: 0,
        cpu_moe_split_experts: 0,
        cpu_moe_threads: 0,
    };
    // The session's ACTUAL KV storage, resolved from the plan that allocates
    // it — not just the requested profile string.
    assert_eq!(
        options.plan().expect("kv plan").storage(),
        storage,
        "resolved KV storage disagrees with the arm"
    );
    let load_started = Instant::now();
    let (mut engine, mut session) = FlintExecutor::open(
        &root,
        &options,
        bundle.eos_token_ids().to_vec(),
        bundle.vocab_size() as usize,
        // Fixture template identity (parks need per-test consistency).
        [0u8; 32],
    )
    .expect("the qualification executor (loud by design)");
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let routes_before = crate::kernels::cuda::CudaContext::q8_tiled_route_count();
    let fallbacks_before = crate::kernels::cuda::CudaContext::q8_tiled_fallback_count();

    let mut temp =
        tempfile::NamedTempFile::new_in(receipt_path.parent().unwrap()).expect("receipt temp file");
    let header = ReceiptHeader {
        arm: &arm,
        class,
        profile,
        armed: armed_record,
        storage,
        run_id: &run_id,
        segments: segment_limit,
        corpus_ids_hash: &corpus_ids_hash,
        used,
        started,
        load_ms,
        grid: grid_path.file_name().and_then(|s| s.to_str()),
    }
    .to_json();
    writeln!(temp, "{}", header).unwrap();

    let grid_file =
        writing_grid.then(|| create_grid_exclusive(&grid_path).expect("referee grid create"));
    let grid_guard = writing_grid.then(|| GridGuard(Some(grid_path.clone())));
    let mut grid_hasher = writing_grid.then(Sha256::new);
    let mut grid_row = Vec::with_capacity(VOCAB);
    let mut grid_bytes = vec![0u8; VOCAB * 4];
    let mut row_buffer = Vec::with_capacity(VOCAB * 4);

    let mut scored_total = 0usize;
    let mut cursor = class.start();
    for segment in 0..segment_limit {
        let segment_started = Instant::now();
        let context = &used[cursor..cursor + class.context()];
        let suffix = &used[cursor + class.context()..cursor + class.context() + SUFFIX];
        engine.reset(&mut session).expect("reset between segments");
        engine.boundary(&mut session);
        engine
            .prefill(&mut session, context)
            .expect("segment prefill");
        // `prefill` keeps the final context token pending and unfed: consume
        // it so the first scored row conditions on the WHOLE context.
        assert_eq!(
            session.state.fed(),
            class.context() - 1,
            "prefill feeds all but the pending token"
        );
        let pending = context[context.len() - 1];
        let _context_row = engine
            .qual_logits(&mut session, pending)
            .expect("pending context token row");
        assert_eq!(
            session.state.fed(),
            class.context(),
            "the pending token must be consumed before the suffix"
        );
        let mut scored = 0usize;
        // Row j (after feeding suffix[j]) predicts suffix[j + 1]; feed all 769
        // tokens, score the first 768 rows against their known next token.
        for (fed, &token) in suffix.iter().enumerate() {
            let logits = engine.qual_logits(&mut session, token).expect("logits row");
            let Some(&actual_next) = suffix.get(fed + 1) else {
                break;
            };
            let lse = row_lse(&logits);
            let logp_next = f64::from(logits[actual_next as usize]) - lse;
            assert!(
                lse.is_finite() && logp_next.is_finite(),
                "non-finite row values at segment {segment} position {scored}"
            );
            if writing_grid {
                let norm = logprob_grid_row(&logits);
                row_to_bytes(&norm, &mut row_buffer);
                grid_file
                    .as_ref()
                    .unwrap()
                    .write_all(&row_buffer)
                    .expect("grid row write");
                grid_hasher.as_mut().unwrap().update(&row_buffer);
                writeln!(
                    temp,
                    "{}",
                    serde_json::json!({
                        "cls": class.as_str(), "seg": segment, "pos": scored,
                        "next_id": actual_next, "lse": lse, "logp_next": logp_next,
                    })
                )
                .unwrap();
            } else {
                grid_stream
                    .as_mut()
                    .unwrap()
                    .next_row(&mut grid_bytes, &mut grid_row);
                let kl = kl_validated(&grid_row, &logits)
                    .unwrap_or_else(|error| panic!("segment {segment} position {scored}: {error}"));
                writeln!(
                    temp,
                    "{}",
                    serde_json::json!({
                        "cls": class.as_str(), "seg": segment, "pos": scored,
                        "next_id": actual_next, "kl": kl, "lse": lse,
                        "logp_next": logp_next,
                        "logp_next_ref": f64::from(grid_row[actual_next as usize]),
                    })
                )
                .unwrap();
            }
            scored += 1;
        }
        assert_eq!(
            scored, SCORED_PER_SEGMENT,
            "segment {segment} scored {scored}"
        );
        assert_eq!(
            session.state.fed(),
            class.context() + SUFFIX,
            "the whole segment stream must be fed"
        );
        eprintln!(
            "TF-GATE leg {arm}/{} seg {segment}: {scored} rows in {:.1}s",
            class.as_str(),
            segment_started.elapsed().as_secs_f64()
        );
        scored_total += scored;
        cursor += class.context() + SUFFIX;
    }
    assert_eq!(
        cursor,
        class.corpus_end(segment_limit),
        "this leg ends at its own class endpoint"
    );
    if let Some(mut stream) = grid_stream.take() {
        stream.drain(&mut grid_bytes, &mut grid_row);
        stream
            .finish()
            .expect("referee grid digest and end-of-file check");
    }
    let routes_delta = crate::kernels::cuda::CudaContext::q8_tiled_route_count() - routes_before;
    let fallbacks_delta =
        crate::kernels::cuda::CudaContext::q8_tiled_fallback_count() - fallbacks_before;
    match arm.as_str() {
        "tiled" => {
            assert!(
                routes_delta > 0,
                "the tiled arm recorded {routes_delta} tiled launches: the route never ran"
            );
            assert_eq!(
                fallbacks_delta, 0,
                "the tiled arm fell back {fallbacks_delta}x"
            );
        }
        _ => {
            assert_eq!(
                routes_delta, 0,
                "only the tiled arm may launch the tile kernel"
            );
            assert_eq!(fallbacks_delta, 0, "unexpected fallbacks {fallbacks_delta}");
        }
    }
    writeln!(
        temp,
        "{}",
        serde_json::json!({
            "rows": scored_total,
            "utc_unix_ms": stamp(),
            "expect_rows": segment_limit * SCORED_PER_SEGMENT,
            "routes_delta": routes_delta,
            "fallbacks_delta": fallbacks_delta,
        })
    )
    .unwrap();
    if writing_grid {
        let sidecar = serde_json::json!({
            "leg": "q8_tf_gate_teacher_forced",
            "arm": "f16",
            "class": class.as_str(),
            "run_id": run_id,
            "model": root.canonicalize().expect("model canonical").to_str(),
            "encoding": GRID_ENCODING,
            "vocab": VOCAB,
            "positions": scored_total,
            "segments": class.count(),
            "scored_per_segment": SCORED_PER_SEGMENT,
            "corpus_snapshot_sha256": SNAPSHOT_SHA,
            "corpus_ids_sha256": corpus_ids_hash,
            "bytes": scored_total * VOCAB * 4,
            "grid_sha256": format!("{:x}", grid_hasher.take().unwrap().finalize()),
        });
        let sidecar_path = grid_path.with_extension("json");
        let mut side_temp = tempfile::NamedTempFile::new_in(grid_path.parent().unwrap()).unwrap();
        writeln!(side_temp, "{sidecar}").unwrap();
        publish(side_temp, &sidecar_path).expect("sidecar publish");
    }
    if let Some(guard) = grid_guard {
        guard.disarm();
    }
    publish(temp, &receipt_path).expect("receipt publish (loud on collision)");
    assert_eq!(scored_total, segment_limit * SCORED_PER_SEGMENT);
}
