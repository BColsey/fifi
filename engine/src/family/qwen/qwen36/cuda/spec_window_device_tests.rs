//! What a verify window of `r` rows costs on the whole 4.00 bpw EXL3 model, against `r` single-row
//! steps, at a fixed depth. One test per process on the GPU held exclusively (about 21 GiB).
//!
//! `FIFI_EXL3_CHECKPOINT` names the pack, `FIFI_SPEC_TEXT` a UTF-8 file of real source text and
//! `FIFI_SPEC_TOKENIZER` the `tokenizer.json` that encodes it. `FIFI_SPEC_DEPTH` is the depth in
//! committed rows (default 8,192) and `FIFI_SPEC_ROWS` the window sizes, comma separated.
//!
//! The text is read as a teacher-forced stream: every window and every single-row step is fed the
//! next tokens of the text, so the routing is that of real code. A window is the chunk entry
//! (`feed_rows_chunked`) over `r` rows, which ends with a status read; a single-row step is
//! `run_token` without logits, which ends with its read-back. Both wait for the device, so the
//! wall time of the call is the cost of the window. Windows and single-row runs alternate inside
//! one session, so slow drift hits both arms alike. The chunk entry computes no head: the cost of
//! the head rows is not in the window figure.
//!
//! A second part compares the distribution after a window with the one after the same rows fed
//! one by one, from the same saved state.

use std::{path::Path, time::Instant};

use super::{
    common_tests::{dims, inventory},
    dims::{KvMode, StepOptions},
    exl3_load::load_exl3_dir,
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    resources::StepResources,
    rules::QwenRules,
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        project::cuda::{CudaProjector, CudaWeights, Reserve, WeightLayout},
        session::TokenStepper,
    },
    tokenizer::Tokenizer,
};

const SAMPLER_VOCAB: usize = 248_077;
const HEAD_ROWS: usize = 248_320;
const EOS: [u32; 2] = [248_046, 248_044];
const CHUNK_ROWS: usize = 1_024;
/// Repetitions of each window size after the warm-up pair.
const REPS: usize = 7;
const WARMUP: usize = 2;
/// Windows compared with the one-by-one run, per size.
const TRIALS: usize = 6;

fn rules() -> QwenRules {
    QwenRules::new(EOS.to_vec(), SAMPLER_VOCAB, HEAD_ROWS).expect("rules")
}

fn options(context_tokens: usize) -> StepOptions {
    StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: WeightLayout::Exl3,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn window_sizes() -> Vec<usize> {
    let text = std::env::var("FIFI_SPEC_ROWS").unwrap_or_else(|_| "1,2,3,4,6,8,16".to_owned());
    text.split(',')
        .map(|part| part.trim().parse().expect("a window size"))
        .collect()
}

/// The token stream: the text, encoded, long enough for the depth and every window.
fn stream(need: usize) -> Vec<u32> {
    let path = std::env::var("FIFI_SPEC_TOKENIZER").expect("FIFI_SPEC_TOKENIZER");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let tokenizer = Tokenizer::from_bytes(&bytes).expect("tokenizer");
    let path = std::env::var("FIFI_SPEC_TEXT").expect("FIFI_SPEC_TEXT");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let ids = tokenizer.encode(&text).expect("encode");
    assert!(
        ids.len() >= need,
        "the text holds {} tokens, the run needs {need}",
        ids.len()
    );
    assert!(
        ids.iter().all(|&id| (id as usize) < SAMPLER_VOCAB),
        "an id outside the sampler vocabulary"
    );
    ids
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn millis(began: Instant) -> f64 {
    began.elapsed().as_secs_f64() * 1e3
}

/// The distribution of a logits row over the sampler vocabulary.
fn distribution(logits: &[f32]) -> Vec<f64> {
    let row = &logits[..SAMPLER_VOCAB];
    let top = row.iter().copied().fold(f32::MIN, f32::max);
    let weights: Vec<f64> = row.iter().map(|v| f64::from(v - top).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.iter().map(|w| w / total).collect()
}

fn argmax(row: &[f32]) -> usize {
    row[..SAMPLER_VOCAB]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .expect("a row")
}

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

fn kl(reference: &[f64], candidate: &[f64]) -> f64 {
    reference
        .iter()
        .zip(candidate)
        .filter(|(p, _)| **p > 0.0)
        .map(|(p, q)| p * (p / q.max(1e-300)).ln())
        .sum()
}

struct Rig {
    projector: &'static CudaProjector,
    weights: &'static CudaWeights,
}

/// A session over the shared weights. `chunked` admits the chunk arena; `capture` captures the
/// single-row body into a graph.
fn session(
    rig: &Rig,
    context_tokens: usize,
    (chunked, capture): (bool, bool),
) -> Result<QwenResidentStep<'static>, EngError> {
    let opts = options(context_tokens);
    let res: &'static StepResources = Box::leak(Box::new(StepResources::new(
        rig.projector.context(),
        dims(),
        &inventory(),
        &opts,
    )?));
    let mut step =
        QwenResidentStep::new(rig.projector.context(), res, rig.weights, &opts, rules())?;
    step.set_capture_config(CaptureConfig {
        capture,
        section_marks: false,
    })?;
    if chunked {
        step.set_kv8_chunked_prefill(true)?;
        step.set_m_sess_cap(CHUNK_ROWS as u32);
        // The free memory now, after any session built before this one.
        let free = rig.projector.memory_info()?.free;
        let verdict = step.admit_prefill(
            PrefillBudget {
                device_free: free.saturating_sub(FIXED_ADMISSION_MARGIN),
                extras: Vec::new(),
            },
            true,
            true,
            false,
        )?;
        assert!(
            format!("{verdict:?}").starts_with("Ready"),
            "the chunk path must be admitted: {verdict:?}"
        );
    }
    Ok(step)
}

/// One window of `rows` through the chunk entry, in milliseconds.
fn window_ms(step: &mut QwenResidentStep<'_>, rows: &[u32]) -> Result<f64, EngError> {
    let began = Instant::now();
    step.feed_rows_chunked(rows, CHUNK_ROWS)?;
    Ok(millis(began))
}

/// `rows.len()` single-row steps, in milliseconds for the whole run.
fn solo_ms(step: &mut QwenResidentStep<'_>, rows: &[u32]) -> Result<f64, EngError> {
    let began = Instant::now();
    for &token in rows {
        step.run_token(token, false)?;
    }
    Ok(millis(began))
}

/// The cost cells of the chunk window against single-row steps on `step` (the chunk session).
/// `solo` is the same measurement on a captured single-row session when one exists.
fn cost_cells(
    step: &mut QwenResidentStep<'_>,
    mut solo: Option<&mut QwenResidentStep<'_>>,
    text: &[u32],
    sizes: &[usize],
) -> Result<(), EngError> {
    let depth = step.tokens();
    for &r in sizes {
        let (mut windows, mut solos, mut captured) = (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..WARMUP + REPS {
            let at = step.tokens();
            let w = window_ms(step, &text[at..at + r])?;
            let at = step.tokens();
            let s = solo_ms(step, &text[at..at + r])?;
            let c = match solo.as_deref_mut() {
                Some(other) => {
                    let at = other.tokens();
                    Some(solo_ms(other, &text[at..at + r])?)
                }
                None => None,
            };
            if rep >= WARMUP {
                windows.push(w);
                solos.push(s / r as f64);
                if let Some(c) = c {
                    captured.push(c / r as f64);
                }
            }
        }
        let (w, s) = (median(&mut windows), median(&mut solos));
        let c = if captured.is_empty() {
            f64::NAN
        } else {
            median(&mut captured)
        };
        println!(
            "CELL depth={depth} r={r} window_ms={w:.3} eager_step_ms={s:.3} \
             captured_step_ms={c:.3} W_over_eager={:.3} W_over_captured={:.3}",
            w / s,
            w / c
        );
    }
    Ok(())
}

/// A window of `r` rows and the same rows one by one, from the same saved state: the greedy
/// token and the divergence after the last row.
fn compare(step: &mut QwenResidentStep<'_>, text: &[u32], r: usize) -> Result<(), EngError> {
    let (mut flips, mut worst) = (0usize, 0.0f64);
    for trial in 0..TRIALS {
        let saved = step.export_state()?;
        let at = step.tokens();
        step.feed_rows_chunked(&text[at..at + r], CHUNK_ROWS)?;
        let candidate = step.logits(text[at + r])?;
        step.import_state(&saved)?;
        for &token in &text[at..at + r] {
            step.run_token(token, false)?;
        }
        let reference = step.logits(text[at + r])?;
        let divergence = kl(&distribution(&reference), &distribution(&candidate));
        let (top_window, top_solo, margin) =
            (argmax(&candidate), argmax(&reference), gap(&reference));
        if top_window != top_solo {
            flips += 1;
        }
        worst = worst.max(divergence);
        println!(
            "MISMATCH r={r} trial={trial} depth={at} KL={divergence:.3e} gap={margin:.3} \
             window_token={top_window} solo_token={top_solo}"
        );
    }
    println!("MISMATCH_SUMMARY r={r} flips={flips}/{TRIALS} worst_KL={worst:.3e}");
    Ok(())
}

fn run_case(rig: &Rig, depth: usize) -> Result<(), EngError> {
    let sizes = window_sizes();
    let need = depth + 4 * (WARMUP + REPS) * sizes.iter().sum::<usize>() + 1_024;
    let text = stream(need);
    let context = depth + need.saturating_sub(depth) + 2_048;
    // The captured single-row session is built first, so the chunk admission sees what is left.
    let mut captured = match session(rig, context, (false, true)) {
        Ok(other) => Some(other),
        Err(error) => {
            println!("CAPTURED_SESSION_UNAVAILABLE {error:?}");
            None
        }
    };
    let mut step = session(rig, context, (true, false))?;
    let began = Instant::now();
    for piece in text[..depth].chunks(CHUNK_ROWS) {
        step.feed_rows_chunked(piece, CHUNK_ROWS)?;
    }
    println!(
        "PREFILL depth={} wall_s={:.2}",
        step.tokens(),
        began.elapsed().as_secs_f64()
    );
    // The captured session continues from the same committed state.
    if let Some(other) = captured.as_mut() {
        other.import_state(&step.export_state()?)?;
    }
    cost_cells(&mut step, captured.as_mut(), &text, &sizes)?;
    for r in [4, 8] {
        compare(&mut step, &text, r)?;
    }
    for r in [8, 16] {
        super::prefill::exl3::split::enable();
        let at = step.tokens();
        step.feed_rows_chunked(&text[at..at + r], CHUNK_ROWS)?;
        let (seconds, calls) = super::prefill::exl3::split::take();
        for (bucket, secs) in &seconds {
            println!(
                "SPLIT r={r} {bucket} ms={:.3} calls={}",
                secs * 1e3,
                calls.get(bucket).copied().unwrap_or(0)
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "the GPU, held exclusively (21 GiB): loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn verify_window_cost_on_the_real_pack() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let depth = env_usize("FIFI_SPEC_DEPTH", 8_192);
    let projector = CudaProjector::new().expect("projector");
    let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("load");
    let rig = Rig {
        projector: Box::leak(Box::new(projector)),
        weights: Box::leak(Box::new(weights)),
    };
    run_case(&rig, depth).expect("window cost run");
}
