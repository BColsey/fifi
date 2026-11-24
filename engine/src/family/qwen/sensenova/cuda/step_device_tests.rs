//! The slice-C decode parity rig: the device trunk executor
//! ([`LoopStep`](super::step::LoopStep)) against the CPU reference
//! (`family/qwen/sensenova/step.rs`) on the released Q4_K_M checkpoint — the same file,
//! the same tokens, both paths opened once and stepped in lockstep. The 24-token smoke
//! prompt is the CPU lane's vendor-oracle prompt (`What is the capital of France?…`, the
//! exact ids of the lane's smoke script); the rig then continues greedily, where every
//! drawn id must match the reference's.
//!
//! The slice-D tests ride the same checkpoint without the CPU reference (which costs ~2 min
//! per step here): the prefill of the prompt against the decode walk above, the prefill
//! throughput, and the kernel-level check that the YaRN frequency table reaches
//! `attn_prep_chunk`.
//!
//! Needs a device and `SENSENOVA_GGUF` (the released SenseNova-Loop-9B GGUF); without
//! either the rig skips with a note.

use std::rc::Rc;
use std::sync::Arc;

use super::super::step::LoopState;
use super::super::weights::LoopWeights;
use super::step::{Carried, LoopStep, bytes_range, floats, le_f32};
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::cuda::{BorrowedLoader, CudaWeights, LoadOptions, WeightLayout};
use crate::family::project::gguf::GgufProjector;
use crate::kernels::cuda::prefill::attn_prep_chunk::{
    AttnPrepChunkCall, AttnPrepChunkShape, PrepChunkMode,
};
use crate::kernels::cuda::resident::attn_prep::RopeParams;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, device_present};
use crate::weights::{GgufLimits, GgufWeights};

/// The smoke prompt's exact ids (the vendor-oracle run of the CPU lane).
const PROMPT: [u32; 24] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
];
const GREEDY: usize = 8;
const CACHE_ROWS: usize = 512;
/// The prompt's chunk cuts for the prefill gate: every cut ends where a walk row ends.
const CUTS: [usize; 6] = [1, 2, 3, 6, 12, 24];
/// The decode steps both cards take after the prompt.
const CONTINUE: usize = 4;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    // A plain fold skips NaN (f32::max ignores it), which passed a NaN logits row.
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            if x.is_finite() && y.is_finite() {
                (x - y).abs()
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0, f32::max)
}

/// The largest magnitude of a row; NaN when the row holds a non-finite value.
fn worst_abs(row: &[f32]) -> f32 {
    row.iter().fold(0.0f32, |m, &v| {
        if v.is_finite() {
            m.max(v.abs())
        } else {
            f32::NAN
        }
    })
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

fn download_f32(context: &CudaContext, buffer: &DeviceBuffer) -> Result<Vec<f32>, EngError> {
    let mut bytes = vec![0u8; buffer.bytes()];
    context.download(buffer, &mut bytes)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// The released checkpoint opened for both paths: the context, the device weights, the
/// CPU-side weights and the projector. `None` (with a note) when the rig cannot run here.
type Rig = (Rc<CudaContext>, CudaWeights, LoopWeights, GgufProjector);

fn open_rig(what: &str) -> Option<Rig> {
    if !device_present(what) {
        return None;
    }
    let Ok(path) = std::env::var("SENSENOVA_GGUF") else {
        eprintln!("skipped: set SENSENOVA_GGUF to the released Loop-9B GGUF");
        return None;
    };
    let file = Arc::new(
        GgufWeights::open_any_type(
            &std::path::PathBuf::from(path),
            GgufLimits {
                max_header_bytes: 32 << 20,
                max_tensor_bytes: 8 << 30,
            },
            None,
        )
        .unwrap(),
    );
    let weights = LoopWeights::open(file.clone()).unwrap();
    let ctx = Rc::new(CudaContext::new().unwrap());
    let loaded = super::load::load(
        &BorrowedLoader::new(&ctx).unwrap(),
        &file,
        weights.plan(),
        LoadOptions {
            layout: WeightLayout::Native,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let projector = weights.projector();
    Some((ctx, loaded, weights, projector))
}

fn open_card<'a>(
    ctx: &Rc<CudaContext>,
    loaded: &'a CudaWeights,
    weights: &LoopWeights,
    cache_rows: usize,
) -> LoopStep<'a> {
    LoopStep::open(
        ctx.clone(),
        loaded,
        weights.shape(),
        weights.plan(),
        weights.rope(),
        cache_rows,
    )
    .unwrap()
}

/// Little-endian f32 words out of raw bytes (a carried f32 buffer).
fn le_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The K/V planes' IEEE binary16, little endian, decoded to f32.
fn le_f16_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| {
            let bits = u16::from_le_bytes(*c);
            let sign = (bits as u32 & 0x8000) << 16;
            let exponent = (bits >> 10) & 0x1f;
            let fraction = (bits & 0x3ff) as u32;
            if exponent == 0 {
                (fraction as f32 * 2f32.powi(-24)).copysign(f32::from_bits(sign))
            } else {
                f32::from_bits(sign | ((exponent as u32 + 112) << 23) | (fraction << 13))
            }
        })
        .collect()
}

/// Every carried buffer of one card, downloaded in the order of [`LoopStep::carried`] (the
/// K/V planes decoded from fp16).
fn carried_snapshot(
    ctx: &CudaContext,
    card: &LoopStep<'_>,
    rows: usize,
) -> Vec<(String, Vec<f32>)> {
    card.stream().synchronize().unwrap();
    card.carried(rows)
        .unwrap()
        .into_iter()
        .map(|(piece, buf, at, len)| {
            let mut bytes = vec![0u8; buf.bytes()];
            ctx.download(buf, &mut bytes).unwrap();
            let decode = if matches!(piece, Carried::K(_) | Carried::V(_)) {
                le_f16_vec
            } else {
                le_f32_vec
            };
            (format!("{piece:?}"), decode(&bytes[at..at + len]))
        })
        .collect()
}

/// The max difference of every carried buffer between two cards, in the order of
/// [`LoopStep::carried`].
fn carried_diffs(
    ctx: &CudaContext,
    a: &LoopStep<'_>,
    b: &LoopStep<'_>,
    rows: usize,
) -> Vec<(String, f32)> {
    diff_snapshots(
        &carried_snapshot(ctx, a, rows),
        &carried_snapshot(ctx, b, rows),
    )
}

/// [`carried_diffs`] over two downloaded states.
fn diff_snapshots(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)]) -> Vec<(String, f32)> {
    a.iter()
        .zip(b)
        .map(|((name, x), (_, y))| (name.clone(), max_abs_diff(x, y)))
        .collect()
}

/// The card's logits row once the stream drained.
fn drain_row(ctx: &CudaContext, card: &LoopStep<'_>) -> Vec<f32> {
    card.stream().synchronize().unwrap();
    download_f32(ctx, card.logits_row()).unwrap()
}

/// One token through both paths: the reference's logits row and the card's. Traced
/// steps also snapshot the residual row after every visit on both sides.
#[allow(clippy::too_many_arguments)]
fn both(
    ctx: &CudaContext,
    projector: &mut GgufProjector,
    weights: &LoopWeights,
    state: &mut LoopState,
    card: &mut LoopStep<'_>,
    token: u32,
    pos: u32,
    cpu_trace: Option<&mut Vec<Vec<f32>>>,
    card_trace: Option<&mut Vec<Vec<f32>>>,
) -> Result<(Vec<f32>, Vec<f32>, u128, u128), EngError> {
    let t0 = std::time::Instant::now();
    let hidden = match cpu_trace {
        Some(t) => super::super::step::trunk_traced(projector, weights, state, token, pos, t)?,
        None => super::super::step::trunk(projector, weights, state, token, pos)?,
    };
    let reference = super::super::step::logits(projector, weights, &hidden)?;
    let cpu_us = t0.elapsed().as_micros();
    let t1 = std::time::Instant::now();
    let embed = projector.rows(weights.token_embd(), &[token])?;
    match card_trace {
        Some(t) => card.decode_traced(&embed, pos as usize, t)?,
        None => card.decode(&embed, pos as usize)?,
    }
    card.stream().synchronize()?;
    let drawn = download_f32(ctx, card.logits_row())?;
    let card_us = t1.elapsed().as_micros();
    Ok((reference, drawn, cpu_us, card_us))
}

fn top2_margin(row: &[f32]) -> f32 {
    let mut top = [f32::NEG_INFINITY; 2];
    for &v in row {
        if v > top[0] {
            top[1] = top[0];
            top[0] = v;
        } else if v > top[1] {
            top[1] = v;
        }
    }
    top[0] - top[1]
}

#[test]
fn card_trunk_matches_the_cpu_reference() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova step parity") else {
        return;
    };
    let plan_probe = weights.plan();
    eprintln!(
        "mhc plan: start {} end {} iterations {} pre_scale {} fixed {}",
        plan_probe.mhc_start,
        plan_probe.mhc_end,
        plan_probe.mhc_iterations,
        plan_probe.mhc_pre_scale,
        plan_probe.mhc_fixed_identity
    );
    let mut state = LoopState::new(weights.shape(), weights.plan());
    let mut card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);

    // Prompt pass: every step's row stays close; no argmax demand mid-prompt. The last
    // step's reference row seeds the greedy draw. Step 0 and the first step whose
    // predecessor drifted past 1.0 dump their per-visit residual diff.
    let mut worst = 0.0f32;
    let mut last = Vec::new();
    let mut prev_diff = f32::INFINITY;
    let mut traced_bad = false;
    // pos-0 per-visit resid diffs (gate 1) and the pos-1/2 full-row diffs (gate 2).
    let mut pos0_resid: Vec<f32> = Vec::new();
    let mut early = [0.0f32; 2];
    // The debug knob shortens the prompt (the CPU reference costs ~2 min per step).
    let prompt_steps = std::env::var("SENSENOVA_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|n| (1..=PROMPT.len()).contains(n))
        .unwrap_or(PROMPT.len());
    for (pos, &token) in PROMPT.iter().take(prompt_steps).enumerate() {
        let trace_this = pos == 0 || (prev_diff > 1.0 && !traced_bad);
        let mut cpu_trace = Vec::new();
        let mut card_trace = Vec::new();
        let (reference, drawn, cpu_us, card_us) = both(
            &ctx,
            &mut projector,
            &weights,
            &mut state,
            &mut card,
            token,
            pos as u32,
            trace_this.then_some(&mut cpu_trace),
            trace_this.then_some(&mut card_trace),
        )
        .unwrap();
        assert_eq!(reference.len(), drawn.len(), "the vocab widths differ");
        let diff = max_abs_diff(&reference, &drawn);
        eprintln!(
            "pos {pos:2} diff {diff:9.4} margin {:-8.3} cpu_ms {:6} card_ms {:6}",
            top2_margin(&reference),
            cpu_us / 1000,
            card_us / 1000
        );
        if trace_this {
            // pos 0 traces unconditionally (prev_diff is INFINITY there); it must not
            // consume the one drift trace.
            if pos > 0 {
                traced_bad = true;
            }
            for (il, pair) in cpu_trace.chunks(2).zip(card_trace.chunks(2)).enumerate() {
                let (b, r) = (
                    max_abs_diff(&pair.0[0], &pair.1[0]),
                    max_abs_diff(&pair.0[1], &pair.1[1]),
                );
                if pos == 0 {
                    pos0_resid.push(r);
                }
                if r > 1e-3 || b > 1e-3 || il < 6 {
                    eprintln!("  visit {il:2} branch {b:9.4} resid {r:9.4}");
                }
            }
        }
        if pos == 1 || pos == 2 {
            early[pos - 1] = diff;
        }
        worst = worst.max(diff);
        prev_diff = diff;
        last = reference;
    }
    eprintln!("prompt worst row diff {worst:.4}");

    // Greedy continuation: ids are reported against the reference's, not asserted — the
    // reference is the pure-f32 CPU lane and the measured token-parity horizon is ~24
    // tokens (gate 3's comment).
    let mut token = argmax(&last);
    for pos in PROMPT.len()..PROMPT.len() + GREEDY {
        let (reference, drawn, cpu_us, card_us) = both(
            &ctx,
            &mut projector,
            &weights,
            &mut state,
            &mut card,
            token,
            pos as u32,
            None,
            None,
        )
        .unwrap();
        let (reference_id, drawn_id) = (argmax(&reference), argmax(&drawn));
        eprintln!(
            "pos {pos:2} diff {:9.4} margin {:-8.3} cpu_ms {:6} card_ms {:6} ids {reference_id} {drawn_id}",
            max_abs_diff(&reference, &drawn),
            top2_margin(&reference),
            cpu_us / 1000,
            card_us / 1000
        );
        worst = worst.max(max_abs_diff(&reference, &drawn));
        token = reference_id;
    }
    eprintln!("greedy worst row diff {worst:.4}");

    // Gate 1 — the trunk math at fresh state: every pos-0 visit row sits at q8 noise;
    // the seed visits measure <=0.13 and the drift compounds visit-to-visit through the
    // residual stream to 1.54 at the final visit (deterministic across runs), so 2.0
    // bounds the amplification while a single broken visit (the residual rotation bug:
    // 13.4 at visit 1, compounding to 158) still fails it by an order of magnitude.
    let worst_visit = pos0_resid.iter().copied().fold(0.0f32, f32::max);
    assert_eq!(pos0_resid.len(), 56, "the pos-0 trace is not whole");
    assert!(
        worst_visit < 2.0,
        "pos-0 visit drift: worst {worst_visit:.4}"
    );

    // Gate 2 — the scoring path with one and two tokens of history: the softmax and the
    // cached k/v are active (they are invisible at pos 0, where softmax over one token is
    // the identity) while the recurrence compounding has not yet engaged. A rope, mscale
    // or cache-layout bug lands here at O(1)+; q8 noise measures 0.32-0.36.
    assert!(early[0] < 1.0, "pos-1 row drift {}", early[0]);
    assert!(early[1] < 1.0, "pos-2 row drift {}", early[1]);

    // Gate 3 — structural: every row of the whole run is finite and bounded. The card's q8
    // activations and fp16 KV seed ~1e-2 row noise that BOTH recurrences (the GDN state and
    // the KV cache) compound at a measured ~1.35x per token, reaching ~14 by token 16 and
    // flipping the first greedy argmax at token 25 (margin 2.8, drift 3.7) — so no
    // f32-tight row bound or token-identity assert can hold past that horizon.
    assert!(
        worst.is_finite() && worst < 16.0,
        "the logits rows diverged: worst {worst:.4}"
    );
}

/// The slice-D prefill gate, device-internal: the decode walk that the rig above
/// validated is the reference, so the same prompt through `prefill` on a fresh card, cut
/// into consecutive chunks, must land where the walk lands at every cut. Then both cards
/// decode the walk's greedy ids, which proves the prefilled state (the recurrent states,
/// the conv rings, the cache rows) feeds decode.
#[test]
fn card_prefill_matches_the_card_decode_walk() {
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova prefill parity") else {
        return;
    };
    let mut walked = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let mut prefilled = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let h = weights.shape().hidden;
    let embed = projector.rows(weights.token_embd(), &PROMPT).unwrap();
    assert_eq!(embed.len(), PROMPT.len() * h, "the prompt embed rows");

    // The walk: the logits row at every prompt position.
    let walk_rows: Vec<Vec<f32>> = PROMPT
        .iter()
        .enumerate()
        .map(|(pos, _)| {
            walked.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
            drain_row(&ctx, &walked)
        })
        .collect();

    // The prefill: consecutive chunks, each ending where a walk row ends.
    let mut start = 0;
    let mut diffs = [0.0f32; CUTS.len()];
    for (diff, &end) in diffs.iter_mut().zip(&CUTS) {
        let t0 = std::time::Instant::now();
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        let row = drain_row(&ctx, &prefilled);
        *diff = max_abs_diff(&walk_rows[end - 1], &row);
        eprintln!(
            "prefill [{start:2}..{end:2}] {:6} ms: row diff vs the walk at pos {} {:.4}, ids {} {}",
            t0.elapsed().as_micros() / 1000,
            end - 1,
            *diff,
            argmax(&walk_rows[end - 1]),
            argmax(&row)
        );
        start = end;
    }

    // The continuation: both cards take the walk's greedy ids.
    let mut token = argmax(&walk_rows[PROMPT.len() - 1]);
    let mut continuation = [0.0f32; CONTINUE];
    for (i, pos) in (PROMPT.len()..PROMPT.len() + CONTINUE).enumerate() {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        walked.decode(&embed, pos).unwrap();
        prefilled.decode(&embed, pos).unwrap();
        let (a, b) = (drain_row(&ctx, &walked), drain_row(&ctx, &prefilled));
        continuation[i] = max_abs_diff(&a, &b);
        eprintln!(
            "continuation pos {pos}: row diff {:.4}, ids {} {}",
            continuation[i],
            argmax(&a),
            argmax(&b)
        );
        token = argmax(&a);
    }
    let continuation_worst = continuation.iter().copied().fold(0.0f32, f32::max);
    let worst = diffs.iter().copied().fold(0.0f32, f32::max);
    eprintln!("prefill worst row diff {worst:.4}; continuation worst {continuation_worst:.4}");

    // Gate 1 — one token through both paths: the dense projections (MMQ) against the
    // matvecs, the chunk kernels against the step kernels: rounding only.
    assert!(
        diffs[0] < 0.5,
        "prefill of one token: row diff {}",
        diffs[0]
    );
    // Gate 2 — two and three tokens of history: the cached k/v, the softmax over several
    // keys and the recurrent state are active (the bounds are the decode rig's).
    assert!(
        diffs[1] < 1.0 && diffs[2] < 1.0,
        "prefill with history: row diffs {} {}",
        diffs[1],
        diffs[2]
    );
    // Gate 3 — structural: the compounding over the prompt stays finite and bounded, and
    // the continuation adds no jump of its own.
    assert!(
        worst.is_finite() && worst < 16.0,
        "the prefilled rows diverged: worst {worst:.4}"
    );
    assert!(
        continuation_worst.is_finite() && continuation_worst < 16.0,
        "the rows diverged after the prompt: worst {continuation_worst:.4}"
    );
}

/// The prefill throughput: a 1100-token prompt (one full 1024-row chunk and the
/// remainder) three times end to end. Reports tokens per second and asserts the logits row
/// is finite; the floor guards against a silent collapse of the chunk path.
#[test]
fn card_prefill_throughput() {
    const TOKENS: usize = 1100;
    /// Three prompts end to end.
    const CACHE: usize = 3 * TOKENS;
    /// A quarter of the measured rate (see the lane's NOW card).
    const FLOOR: f64 = 200.0;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova prefill throughput")
    else {
        return;
    };
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();
    let mut card = open_card(&ctx, &loaded, &weights, CACHE);
    let mut best = 0.0f64;
    for (run, p0) in [0, TOKENS, 2 * TOKENS].into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        card.prefill(&embed, p0).unwrap();
        card.stream().synchronize().unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let row = download_f32(&ctx, card.logits_row()).unwrap();
        let worst = worst_abs(&row);
        assert!(worst.is_finite(), "run {run}: the logits row is not finite");
        let rate = TOKENS as f64 / secs;
        best = best.max(rate);
        eprintln!(
            "prefill run {run}: {TOKENS} tokens in {:.1} ms = {:.0} tok/s (worst |logit| {worst:.1})",
            secs * 1e3,
            rate
        );
    }
    assert!(
        best > FLOOR,
        "prefill throughput {best:.0} tok/s is under the floor {FLOOR}"
    );
}

/// The prefill over a long prompt: 1061 pseudo-random tokens walked one by one on one card
/// and prefilled on the other in chunks of 1, 63, 192, 256, 512 and 37 rows (a full 1024-row
/// chunk among them), then four greedy steps on both. Reports the row diff at every cut: the
/// short gate's diff grew with prompt position, and only a long chunk shows whether it keeps
/// growing (a chunk kernel accumulating error) or levels (decaying state).
#[test]
fn card_prefill_long_prompt_matches_the_walk() {
    /// The prompt: one full 1024-row chunk and a remainder.
    const TOKENS: usize = 1024 + 37;
    /// The chunk cuts: every cut ends where a walk row ends.
    const CUTS: [usize; 6] = [1, 64, 256, 512, 1024, TOKENS];
    /// The cards' cache rows: the prompt and the continuation.
    const ROWS: usize = 2048;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova long prefill parity")
    else {
        return;
    };
    let mut walked = open_card(&ctx, &loaded, &weights, ROWS);
    let mut prefilled = open_card(&ctx, &loaded, &weights, ROWS);
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();

    // The walk: queued ahead, drained for its row at every cut position only.
    let mut walk_rows: Vec<Vec<f32>> = Vec::new();
    for pos in 0..TOKENS {
        walked.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        if CUTS.contains(&(pos + 1)) {
            walk_rows.push(drain_row(&ctx, &walked));
        }
    }

    // The prefill: consecutive chunks, each ending where a walk row ends.
    let mut start = 0;
    let mut diffs = [0.0f32; CUTS.len()];
    for ((diff, &end), walk) in diffs.iter_mut().zip(&CUTS).zip(&walk_rows) {
        let t0 = std::time::Instant::now();
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        let row = drain_row(&ctx, &prefilled);
        *diff = max_abs_diff(walk, &row);
        eprintln!(
            "long prefill [{start:4}..{end:4}] {:6} ms: row diff vs the walk at pos {} {:.4}, ids {} {}",
            t0.elapsed().as_micros() / 1000,
            end - 1,
            *diff,
            argmax(walk),
            argmax(&row)
        );
        start = end;
    }

    // The continuation: both cards take the walk's greedy ids.
    let mut token = argmax(&walk_rows[CUTS.len() - 1]);
    let mut continuation = [0.0f32; CONTINUE];
    for (i, pos) in (TOKENS..TOKENS + CONTINUE).enumerate() {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        walked.decode(&embed, pos).unwrap();
        prefilled.decode(&embed, pos).unwrap();
        let (a, b) = (drain_row(&ctx, &walked), drain_row(&ctx, &prefilled));
        continuation[i] = max_abs_diff(&a, &b);
        eprintln!(
            "long continuation pos {pos}: row diff {:.4}, ids {} {}",
            continuation[i],
            argmax(&a),
            argmax(&b)
        );
        token = argmax(&a);
    }
    let worst = diffs.iter().copied().fold(0.0f32, f32::max);
    let continuation_worst = continuation.iter().copied().fold(0.0f32, f32::max);
    eprintln!("long prefill worst row diff {worst:.4}; continuation worst {continuation_worst:.4}");
    assert!(
        worst.is_finite() && worst < 16.0,
        "the long prefilled rows diverged: worst {worst:.4}"
    );
    assert!(
        continuation_worst.is_finite() && continuation_worst < 16.0,
        "the rows diverged after the long prompt: worst {continuation_worst:.4}"
    );
}

/// Where the long-prompt divergence lives, against the walk's own noise floor: two cards walk
/// the same 64 pseudo-random tokens, the second from a first token whose largest embedding
/// value sits one q8 step higher; a third card then prefills the same tokens in chunks of 1,
/// 1, 3, 6, 13 and 40 rows. Prints the logits row diff of both pairs at every cut, then the
/// max difference of every carried buffer for both pairs: the q8 activations make every
/// step's rounding a discrete choice, so the walk against its own one-step perturbation is
/// the spread the prefill has to be judged against.
#[test]
fn card_long_prompt_divergence_localized() {
    const TOKENS: usize = 64;
    /// The chunk cuts: every cut ends where a walk row ends.
    const CUTS: [usize; 6] = [1, 2, 5, 11, 24, TOKENS];
    const ROWS: usize = 256;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova long prompt divergence")
    else {
        return;
    };
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();

    // The perturbed walk: the first token's largest embedding value one q8 step (1/127 of
    // the row's largest) higher.
    let (k, top) = embed[..h]
        .iter()
        .enumerate()
        .map(|(i, x)| (i, x.abs()))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    let mut nudged = embed.clone();
    nudged[k] += top / 127.0;

    let mut walk = open_card(&ctx, &loaded, &weights, ROWS);
    let mut walk_nudged = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut rows, mut nudged_rows) = (Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        walk.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        walk_nudged
            .decode(&nudged[pos * h..(pos + 1) * h], pos)
            .unwrap();
        if CUTS.contains(&(pos + 1)) {
            rows.push(drain_row(&ctx, &walk));
            nudged_rows.push(drain_row(&ctx, &walk_nudged));
        }
    }
    let floor: Vec<f32> = rows
        .iter()
        .zip(&nudged_rows)
        .map(|(a, b)| max_abs_diff(a, b))
        .collect();
    let floor_carried = carried_diffs(&ctx, &walk, &walk_nudged, TOKENS);
    drop(walk_nudged);

    let mut prefilled = open_card(&ctx, &loaded, &weights, ROWS);
    let mut start = 0;
    let mut prefill = [0.0f32; CUTS.len()];
    for ((diff, &end), row) in prefill.iter_mut().zip(&CUTS).zip(&rows) {
        prefilled
            .prefill(&embed[start * h..end * h], start)
            .unwrap();
        *diff = max_abs_diff(row, &drain_row(&ctx, &prefilled));
        start = end;
    }
    let prefill_carried = carried_diffs(&ctx, &walk, &prefilled, TOKENS);

    eprintln!(
        "row diff at pos {:?}: walk vs one q8 step in token 0: {floor:.3?}; prefill vs walk: {prefill:.3?}",
        CUTS.map(|c| c - 1)
    );
    eprintln!("carried buffer      walk vs nudged   prefill vs walk");
    for ((name, a), (_, b)) in floor_carried.iter().zip(&prefill_carried) {
        eprintln!("  {name:<16} {a:14.5}  {b:14.5}");
    }
    let finite = |d: &f32| d.is_finite();
    assert!(
        floor.iter().chain(&prefill).all(finite)
            && floor_carried
                .iter()
                .chain(&prefill_carried)
                .map(|(_, d)| d)
                .all(finite),
        "a non-finite value reached the logits or the carried state"
    );
}

/// Which stage breaks the second token. The long-prompt probe showed a one-row prefill at
/// position 1 already off the floor, so row counts are not the question: the walk takes two
/// tokens; a card prefills them as two one-row chunks; a third as one two-row chunk. The
/// carried buffers after each chunk stand next to the walk's state at that stage and to the
/// floor there, which a second walk from a nudged first token gives. A third card issues the
/// two chunks back to back and must reach the synchronized card's state: a blocking row upload
/// landed at once, and the chunk before wrote its final residual over it.
#[test]
fn card_second_token_from_carried_state_localized() {
    const TOKENS: usize = 2;
    const ROWS: usize = 256;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova second token") else {
        return;
    };
    let h = weights.shape().hidden;
    let vocab = weights.shape().vocab;
    let ids: Vec<u32> = (1..=TOKENS as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) >> 7)
        .map(|id| id % vocab as u32)
        .collect();
    let embed = projector.rows(weights.token_embd(), &ids).unwrap();
    let mut nudged_embed = embed.clone();
    nudged_embed[h] += 1.0 / 127.0;

    const _: () = assert!(TOKENS == 2);
    let mut walk = open_card(&ctx, &loaded, &weights, ROWS);
    let mut nudged = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut walk_rows, mut walk_states, mut floor_rows, mut floors) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        walk.decode(&embed[pos * h..(pos + 1) * h], pos).unwrap();
        nudged
            .decode(&nudged_embed[pos * h..(pos + 1) * h], pos)
            .unwrap();
        walk_rows.push(drain_row(&ctx, &walk));
        floor_rows.push(max_abs_diff(&walk_rows[pos], &drain_row(&ctx, &nudged)));
        walk_states.push(carried_snapshot(&ctx, &walk, pos + 1));
        floors.push(carried_diffs(&ctx, &walk, &nudged, pos + 1));
    }
    // The walk and one more card at a time: a fourth model instance on the card ran out of
    // memory.
    drop(nudged);

    let mut singles = open_card(&ctx, &loaded, &weights, ROWS);
    let (mut singles_rows, mut after_chunk) = (Vec::new(), Vec::new());
    for pos in 0..TOKENS {
        singles
            .prefill(&embed[pos * h..(pos + 1) * h], pos)
            .unwrap();
        singles_rows.push(max_abs_diff(&walk_rows[pos], &drain_row(&ctx, &singles)));
        let state = carried_snapshot(&ctx, &singles, pos + 1);
        after_chunk.push(diff_snapshots(&walk_states[pos], &state));
    }
    let singles_last = carried_snapshot(&ctx, &singles, TOKENS);
    drop(singles);

    let last = TOKENS - 1;
    let mut pair = open_card(&ctx, &loaded, &weights, ROWS);
    pair.prefill(&embed, 0).unwrap();
    let pair_diff = max_abs_diff(&walk_rows[last], &drain_row(&ctx, &pair));
    let pair_state = carried_snapshot(&ctx, &pair, TOKENS);
    let pair_carried = diff_snapshots(&walk_states[last], &pair_state);

    drop(pair);
    let mut burst = open_card(&ctx, &loaded, &weights, ROWS);
    for pos in 0..TOKENS {
        burst.prefill(&embed[pos * h..(pos + 1) * h], pos).unwrap();
    }
    let burst_row = max_abs_diff(&walk_rows[last], &drain_row(&ctx, &burst));
    let burst_state = carried_snapshot(&ctx, &burst, TOKENS);
    let burst_vs_synced = diff_snapshots(&singles_last, &burst_state)
        .iter()
        .map(|(_, d)| *d)
        .fold(0.0_f32, f32::max);

    eprintln!(
        "row diff after token 0: floor {:.4}, one-row chunk {:.4}; after token 1: floor {:.4}, two one-row chunks {:.4}, one two-row chunk {:.4}, two back to back {:.4} (their state against the synchronized pair: {:.5})",
        floor_rows[0],
        singles_rows[0],
        floor_rows[last],
        singles_rows[last],
        pair_diff,
        burst_row,
        burst_vs_synced
    );
    let stages = [
        ("floor 1", &floors[0]),
        ("chunk 1", &after_chunk[0]),
        ("floor 2", &floors[last]),
        ("chunks 1+2", &after_chunk[last]),
        ("one 2-row", &pair_carried),
    ];
    let cells = |i: usize| -> String {
        stages
            .iter()
            .map(|(_, column)| format!("{:12.5}", column[i].1))
            .collect()
    };
    eprintln!(
        "carried buffer   {}",
        stages
            .iter()
            .map(|(title, _)| format!("{title:>12}"))
            .collect::<String>()
    );
    for (i, (name, _)) in stages[0].1.iter().enumerate() {
        eprintln!("  {name:<16}{}", cells(i));
    }
    let finite = |d: &f32| d.is_finite();
    let diffs = floor_rows
        .iter()
        .chain(&singles_rows)
        .chain([&pair_diff, &burst_row, &burst_vs_synced])
        .chain(
            stages
                .iter()
                .flat_map(|(_, column)| column.iter().map(|(_, d)| d)),
        );
    assert!(
        diffs.into_iter().all(finite),
        "a non-finite value reached the logits or the carried state"
    );
    assert!(
        burst_vs_synced < 1e-4,
        "chunks issued back to back diverge from the same chunks with a synchronize between them: {burst_vs_synced:.5}"
    );
}

/// The YaRN table reaches the kernel (owed by slice B2): the kernel derives
/// `rope_base^(-2i/n_rot)` when it gets no table, so passing that table must change nothing
/// beyond rounding, and a zero table must not rotate: its rows do not depend on the position,
/// and its first row equals the derived table's first row at position 0, where nothing rotates
/// either.
#[test]
fn attn_prep_chunk_applies_the_frequency_table() {
    if !device_present("attn_prep_chunk frequency table") {
        return;
    }
    const M: usize = 2;
    const HQ: usize = 4;
    const HKV: usize = 2;
    const HD: usize = 256;
    const N_ROT: usize = 64;
    const ROWS: usize = 32;
    const BASE: f32 = 5_000_000.0;
    let ld = HQ * 2 * HD + 2 * HKV * HD;
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    // Deterministic projection rows in [-1, 1].
    let mut seed = 0x9e37_79b9u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
    };
    let proj: Vec<f32> = (0..M * ld).map(|_| noise()).collect();
    let norm = ctx.upload(&le_f32(&vec![1.0; HD])).unwrap();
    let run = |p0: i64, table: Option<&[f32]>| -> Vec<f32> {
        let cache_bytes = ROWS * HKV * HD * 2;
        let proj_dev = ctx.upload(&le_f32(&proj)).unwrap();
        let k_cache = ctx.upload(&vec![0u8; cache_bytes]).unwrap();
        let v_cache = ctx.upload(&vec![0u8; cache_bytes]).unwrap();
        let q = ctx.upload(&vec![0u8; M * HQ * HD * 4]).unwrap();
        let inv_freq = table.map(|t| ctx.upload(&le_f32(t)).unwrap());
        ctx.attn_prep_chunk(
            &stream,
            AttnPrepChunkCall {
                shape: AttnPrepChunkShape {
                    m: M,
                    n_head: HQ,
                    n_kv_head: HKV,
                    head_dim: HD,
                    n_rot: N_ROT,
                    rows: ROWS,
                    ld,
                    p0,
                },
                rope: RopeParams {
                    eps: 1e-6,
                    rope_base: BASE,
                },
                proj: floats(&proj_dev, 0, M * ld).unwrap(),
                q_norm: floats(&norm, 0, HD).unwrap(),
                k_norm: floats(&norm, 0, HD).unwrap(),
                q: floats(&q, 0, M * HQ * HD).unwrap(),
                mode: PrepChunkMode::AppendFp16 {
                    k_cache: bytes_range(&k_cache, cache_bytes).unwrap(),
                    v_cache: bytes_range(&v_cache, cache_bytes).unwrap(),
                },
                status: None,
                inv_freq: inv_freq.as_ref().map(|t| floats(t, 0, N_ROT / 2).unwrap()),
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        download_f32(&ctx, &q).unwrap()
    };
    let derived_at_0 = run(0, None);
    let derived_at_8 = run(8, None);
    let zeros = vec![0.0; N_ROT / 2];
    let zero_at_8 = run(8, Some(&zeros));
    let zero_at_0 = run(0, Some(&zeros));
    let table: Vec<f32> = (0..N_ROT / 2)
        .map(|i| 1.0 / BASE.powf(2.0 * i as f32 / N_ROT as f32))
        .collect();
    let table_at_8 = run(8, Some(&table));

    let row = HQ * HD;
    let unrotated = max_abs_diff(&zero_at_8[..row], &derived_at_0[..row]);
    // Row 1 carries other projection values than row 0, so its unrotated form is its own
    // zero-table row at position 1.
    let still = max_abs_diff(&zero_at_8[row..], &zero_at_0[row..]);
    let rotated = max_abs_diff(&zero_at_8, &derived_at_8);
    let same_table = max_abs_diff(&table_at_8, &derived_at_8);
    eprintln!(
        "zero table at pos 8: row 0 vs derived at pos 0: {unrotated:.6}; row 1 vs zero table at pos 0: {still:.6}; vs derived at pos 8: {rotated:.4}; the standard table vs derived: {same_table:.7}"
    );
    assert!(rotated > 1e-3, "pos 8 is not rotated: {rotated:.6}");
    assert!(
        unrotated < 1e-6 && still < 1e-6,
        "a zero table must not rotate: {unrotated:.6}, row 1 vs zero table at pos 0: {still:.6}"
    );
    assert!(
        same_table < 1e-4,
        "the standard table differs from the derived one: {same_table:.7}"
    );
}

/// The decode rate on its own: the prompt through `prefill`, then [`STEPS`] greedy steps as
/// the serve loop runs them (every step drains for its token), then as many again queued
/// ahead without a drain. The difference is the launch-side cost; the run is also what the
/// host's `nsys` profiles (the decode census in the lane's NOW card).
#[test]
fn card_decode_trace() {
    const STEPS: usize = 64;
    let Some((ctx, loaded, weights, mut projector)) = open_rig("sensenova decode trace") else {
        return;
    };
    let mut card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let prompt = projector.rows(weights.token_embd(), &PROMPT).unwrap();
    card.prefill(&prompt, 0).unwrap();
    let mut token = argmax(&drain_row(&ctx, &card));

    // The ceiling: the block and head weights every step reads (the loop layers once per
    // revisit) over the card's rated bandwidth.
    const RATED_GB_S: f64 = 936.0;
    let gib = card.step_weight_bytes() as f64 / 1e9;
    eprintln!(
        "decode step reads {gib:.2} GB of block and head weights (the small F32 tensors are not counted): at the rated {RATED_GB_S} GB/s that alone is {:.1} ms = {:.0} tok/s",
        gib / RATED_GB_S * 1e3,
        RATED_GB_S / gib
    );

    let t0 = std::time::Instant::now();
    for pos in PROMPT.len()..PROMPT.len() + STEPS {
        let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
        card.decode(&embed, pos).unwrap();
        token = argmax(&drain_row(&ctx, &card));
    }
    let serialized = t0.elapsed().as_secs_f64() * 1e3 / STEPS as f64;

    let embed = projector.rows(weights.token_embd(), &[token]).unwrap();
    card.stream().synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for pos in PROMPT.len() + STEPS..PROMPT.len() + 2 * STEPS {
        card.decode(&embed, pos).unwrap();
    }
    card.stream().synchronize().unwrap();
    let pipelined = t0.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    eprintln!(
        "decode: {serialized:.1} ms/step with the per-step drain, {pipelined:.1} ms/step queued ahead"
    );
    assert!(serialized.is_finite() && pipelined.is_finite());
}

/// Device microbenchmark, not a gate: the decode step's weight sweep at 1, 2, 4 and 8
/// activation columns. A batched step reads the weights once for all of its tokens, so
/// this is the weight term of one token of a batch on this card. The glue, the launch gaps
/// and the per-sequence attention state of a real batched step are not in it.
#[test]
fn card_decode_sweep_columns() {
    let Some((ctx, loaded, weights, _projector)) = open_rig("sensenova decode sweep columns")
    else {
        return;
    };
    let card = open_card(&ctx, &loaded, &weights, CACHE_ROWS);
    let gib = card.step_weight_bytes() as f64 / 1e9;
    eprintln!("sweep reads {gib:.2} GB per pass: median and minimum of 5 groups of 5 sweeps");
    for ncols in [1, 2, 4, 8] {
        let mut sorted = card.sweep_micros(ncols, 5, 5).unwrap();
        sorted.sort_by(f64::total_cmp);
        let (mid, min) = (sorted[sorted.len() / 2], sorted[0]);
        eprintln!(
            "sweep {ncols} column(s): median {mid:.0} us, min {min:.0} us = {:.0} GB/s; weight term {:.2} ms per token = {:.0} tok/s from this term alone",
            gib / (mid * 1e-6),
            mid * 1e-6 * 1e3 / ncols as f64,
            ncols as f64 / (mid * 1e-6),
        );
    }
}
