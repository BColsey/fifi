//! The chunked prefill of an EXL3 pack against its own one-token-per-step prefill, on the whole
//! 4.00 bpw model, one test per process on the 3090 held exclusively (about 21 GiB).
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack, `FIFI_PF_PROMPT_IDS` a JSON file `{"ids": [...]}` of
//! prompt tokens and `FIFI_PF_ROWS` how many of them to feed (default 5,222). The serving
//! configuration is mirrored: Int8G32 KV with the chunk opt-in, the default 1,024-row cap and the
//! fixed admission of `start.rs`. The same weights serve two sessions in turn: the one-token path
//! produces the reference logits of the last prompt token, the chunk path the candidate.
//!
//! The chunk path evaluates the projections with the rows kernels (tensor cores from 64 rows)
//! and attention with the chunk kernels, so it is not bit-equal to the one-row path. The test
//! compares the next-token distributions: the greedy token must agree and the KL divergence of
//! the candidate from the reference stays small. Wall times are reported, not asserted.

use std::{path::Path, time::Instant};

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    resources::StepResources,
    rules::QwenRules,
    step::QwenResidentStep,
};
use crate::{
    EngError,
    family::{
        project::cuda::{CudaProjector, CudaWeights, Reserve, WeightLayout},
        session::TokenStepper,
    },
};

pub(super) const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
const CONTEXT_TOKENS: usize = 8_192;
pub(super) const CHUNK_ROWS: u32 = 1_024;
/// Decode steps after the prompt that both paths are compared on.
pub(super) const CONTINUATION: usize = 32;
/// A reference step whose two best logits are closer than this (nats) may pick the other token.
const TIE_GAP: f32 = 0.1;
/// The second prompt: this many tokens from this offset of the ids file.
pub(super) const SECOND_PROMPT: (usize, usize) = (1_000, 3_300);
/// Greatest KL divergence, in nats, of the chunk distribution from the one-token distribution
/// (measured 4.9e-4 and 1.1e-4 over 5,222 tokens).
const KL_BOUND: f64 = 1e-3;
/// Greatest KL divergence at any step of the teacher-forced continuation (measured worst: 1.4e-2,
/// at a step that copies the prompt's text and rests on exact cache values).
const CONTINUATION_KL_BOUND: f64 = 5e-2;

pub(super) fn rules() -> QwenRules {
    QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules")
}

pub(super) fn all_ids() -> Vec<u32> {
    let path = std::env::var("FIFI_PF_PROMPT_IDS").expect("FIFI_PF_PROMPT_IDS");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let value: serde_json::Value = serde_json::from_str(&text).expect("ids json");
    value["ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| id.as_u64().expect("id") as u32)
        .collect()
}

/// Tokens of the first prompt.
fn prompt_rows() -> usize {
    std::env::var("FIFI_PF_ROWS")
        .ok()
        .and_then(|rows| rows.parse::<usize>().ok())
        .unwrap_or(5_222)
}

/// Whether the session uses the default CUDA KV storage (`FIFI_PF_KV=fp16`) instead of Int8G32.
pub(super) fn fp16_kv() -> bool {
    std::env::var("FIFI_PF_KV").is_ok_and(|kv| kv == "fp16")
}

pub(super) fn options() -> StepOptions {
    StepOptions {
        kv_mode: if fp16_kv() {
            KvMode::Fp16
        } else {
            KvMode::Int8G32
        },
        context_tokens: CONTEXT_TOKENS,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

/// A softmax of `logits` over the sampler vocabulary, in f64.
fn distribution(logits: &[f32]) -> Vec<f64> {
    let row = &logits[..SAMPLER_VOCAB];
    let top = row.iter().copied().fold(f32::MIN, f32::max);
    let weights: Vec<f64> = row.iter().map(|v| f64::from(v - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

pub(super) fn argmax(row: &[f32]) -> usize {
    row[..SAMPLER_VOCAB]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .unwrap()
}

/// `KL(reference || candidate)` in nats.
fn kl(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q.max(1e-300)).ln())
        .sum()
}

struct Run {
    logits: Vec<f32>,
    /// The tokens fed after the prompt, one decode step each: the greedy tokens of the session,
    /// or the forced ones.
    continuation: Vec<u32>,
    /// The logits row after the prompt and after each token of the continuation.
    rows: Vec<Vec<f32>>,
    seconds: f64,
    steps: usize,
}

/// One session over the shared weights: `chunked` admits the arena and feeds the head rows in
/// chunks; otherwise every prompt token is one step. Returns the logits of the last token.
fn session(
    projector: &'static CudaProjector,
    weights: &'static CudaWeights,
    free_after_load: usize,
    ids: &[u32],
    (chunked, forced): (bool, Option<&[u32]>),
) -> Result<Run, EngError> {
    let opts = options();
    let res: &'static StepResources = Box::leak(Box::new(StepResources::new(
        projector.context(),
        dims(),
        &inventory(),
        &opts,
    )?));
    let mut step = QwenResidentStep::new(projector.context(), res, weights, &opts, rules())?;
    let (head, last) = ids.split_at(ids.len() - 1);
    let began = Instant::now();
    if chunked && fp16_kv() {
        // The serving configuration of the default KV: the arena at the default cap, the prompt
        // through the session's own feed (the chunk sizes of `m_pos`).
        let verdict = step.admit_prefill(
            PrefillBudget {
                device_free: free_after_load.saturating_sub(1 << 30),
                extras: Vec::new(),
            },
            true,
            false,
            false,
        )?;
        println!("admission: {verdict:?}");
        assert!(
            format!("{verdict:?}").starts_with("Ready"),
            "the chunk path must be admitted: {verdict:?}"
        );
        step.feed_rows(head)?;
    } else if chunked {
        step.set_kv8_chunked_prefill(true)?;
        step.set_m_sess_cap(CHUNK_ROWS);
        let verdict = step.admit_prefill(
            PrefillBudget {
                device_free: free_after_load.saturating_sub(FIXED_ADMISSION_MARGIN),
                extras: Vec::new(),
            },
            true,
            true,
            false,
        )?;
        println!("admission: {verdict:?}");
        assert!(
            format!("{verdict:?}").starts_with("Ready"),
            "the chunk path must be admitted: {verdict:?}"
        );
        // One call per chunk-sized piece: the same partition as one call, with a time per piece
        // that is visible while the test runs.
        let rows = CHUNK_ROWS as usize;
        for piece in head.chunks(rows) {
            let began = Instant::now();
            step.feed_rows_chunked(piece, rows)?;
            println!(
                "chunk piece of {} rows, tokens {}: {:.3} s",
                piece.len(),
                step.tokens(),
                began.elapsed().as_secs_f64()
            );
        }
    } else {
        for &token in head {
            step.feed(token)?;
        }
    }
    let logits = step.logits(last[0])?;
    let seconds = began.elapsed().as_secs_f64();
    let steps = step.last_prefill_report().steps().len();
    assert_eq!(step.tokens(), ids.len(), "every prompt row is committed");
    let mut continuation = Vec::with_capacity(CONTINUATION);
    let mut rows = vec![logits.clone()];
    for index in 0..CONTINUATION {
        let token = match forced {
            Some(tokens) => tokens[index],
            None => u32::try_from(argmax(&rows[index])).expect("a token id"),
        };
        continuation.push(token);
        rows.push(step.logits(token)?);
    }
    Ok(Run {
        logits,
        rows,
        continuation,
        seconds,
        steps,
    })
}

/// Runs the prompt again with every stage waited for, and prints the milliseconds and calls of each
/// stage over the whole prefill. The waits make this run slower than the untimed one: the buckets
/// locate time, they do not measure it.
fn report_split(
    projector: &'static CudaProjector,
    weights: &'static CudaWeights,
    free: usize,
    ids: &[u32],
) {
    super::prefill::exl3::split::enable();
    let timed = session(projector, weights, free, ids, (true, None)).expect("timed session");
    let (seconds, calls) = super::prefill::exl3::split::take();
    let total: f64 = seconds.values().sum();
    println!(
        "timed run: {} rows in {:.3} s, {:.3} s in {} buckets",
        ids.len() - 1,
        timed.seconds,
        total,
        seconds.len()
    );
    for (bucket, secs) in &seconds {
        println!("  {bucket:>20}: {:8.1} ms", secs * 1e3);
    }
    for (bucket, count) in &calls {
        println!("  {bucket:>20}: {count:8} calls");
    }
}

/// The top two logits' gap of a row: a step with a gap below [`TIE_GAP`] may legitimately pick
/// the other token under the numerics of the other path.
fn gap(row: &[f32]) -> f32 {
    let mut top = [f32::MIN; 2];
    for &value in &row[..SAMPLER_VOCAB] {
        if value > top[0] {
            top = [value, top[0]];
        } else if value > top[1] {
            top[1] = value;
        }
    }
    top[0] - top[1]
}

/// Checks every step of `candidate` against `reference`: finite logits, a greedy flip only at a
/// near-tie of the reference, the prompt's distribution within [`KL_BOUND`] and no step beyond
/// [`CONTINUATION_KL_BOUND`].
pub(super) fn assert_rows(reference_rows: &[Vec<f32>], candidate_rows: &[Vec<f32>]) {
    let mut flips = 0;
    let mut worst = (0usize, 0.0f64);
    for (step, (reference, candidate)) in reference_rows.iter().zip(candidate_rows).enumerate() {
        assert!(
            candidate[..SAMPLER_VOCAB].iter().all(|v| v.is_finite()),
            "step {step}: finite logits"
        );
        let divergence = kl(&distribution(reference), &distribution(candidate));
        let (top_chunk, top_token) = (argmax(candidate), argmax(reference));
        let margin = gap(reference);
        println!(
            "step {step}: KL {divergence:.3e} gap {margin:.3} greedy {top_chunk} vs {top_token}"
        );
        if divergence > worst.1 {
            worst = (step, divergence);
        }
        if top_chunk != top_token {
            flips += 1;
            assert!(margin < TIE_GAP, "step {step}: a flip at a gap of {margin}");
        }
        if step == 0 {
            assert!(
                divergence < KL_BOUND,
                "the prompt's distribution: KL {divergence}"
            );
        }
    }
    println!(
        "{flips} near-tie flips in {} steps; worst KL {:.3e} at step {}",
        reference_rows.len(),
        worst.1,
        worst.0
    );
    assert!(
        worst.1 < CONTINUATION_KL_BOUND,
        "step {}: KL {}",
        worst.0,
        worst.1
    );
}

/// One prompt through both paths. The one-token session picks its greedy continuation; the chunk
/// session is then fed THE SAME tokens (teacher forcing: a flip at a near-tie must not change what
/// the later steps are compared on), and every step compares the next-token distribution and, where
/// the reference's top two logits are further apart than [`TIE_GAP`], the greedy token.
fn compare(
    (projector, weights, free): (&'static CudaProjector, &'static CudaWeights, usize),
    ids: &[u32],
    split: bool,
) {
    assert!(
        ids.len() > 2 * CHUNK_ROWS as usize,
        "the prompt spans several chunks"
    );
    let token = session(projector, weights, free, ids, (false, None)).expect("one-token session");
    let forced = Some(token.continuation.as_slice());
    let chunk = session(projector, weights, free, ids, (true, forced)).expect("chunked session");
    if split {
        report_split(projector, weights, free, &ids[..2_049]);
    }
    println!(
        "{} rows: chunked {:.3} s ({} records) vs one-token {:.3} s",
        ids.len(),
        chunk.seconds,
        chunk.steps,
        token.seconds
    );
    assert_rows(&token.rows, &chunk.rows);
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn chunked_prefill_matches_the_one_token_path_on_the_real_pack() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let ids = all_ids();
    let first: Vec<u32> = ids.iter().take(prompt_rows()).copied().collect();
    let (from, rows) = SECOND_PROMPT;
    let second: Vec<u32> = ids.iter().skip(from).take(rows).copied().collect();
    let projector = CudaProjector::new().expect("projector");
    let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("load");
    let free = projector.memory_info().expect("memory").free;
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let rig = (projector, weights, free);
    if std::env::var_os("FIFI_PF_TIMING_ONLY").is_some() {
        // Timing without the one-token references: the chunked run, then the timed stage split.
        let chunk = session(projector, weights, free, &first, (true, None)).expect("chunked");
        println!("{} rows: chunked {:.3} s", first.len(), chunk.seconds);
        report_split(projector, weights, free, &first);
        return;
    }
    compare(rig, &first, std::env::var_os("FIFI_PF_SPLIT").is_some());
    compare(rig, &second, false);
}
