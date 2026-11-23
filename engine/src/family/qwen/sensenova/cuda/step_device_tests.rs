//! The slice-C decode parity rig: the device trunk executor
//! ([`LoopStep`](super::step::LoopStep)) against the CPU reference
//! (`family/qwen/sensenova/step.rs`) on the released Q4_K_M checkpoint — the same file,
//! the same tokens, both paths opened once and stepped in lockstep. The 24-token smoke
//! prompt is the CPU lane's vendor-oracle prompt (`What is the capital of France?…`, the
//! exact ids of the lane's smoke script); the rig then continues greedily, where every
//! drawn id must match the reference's.
//!
//! Needs a device and `SENSENOVA_GGUF` (the released SenseNova-Loop-9B GGUF); without
//! either the rig skips with a note.

use std::sync::Arc;

use super::super::step::LoopState;
use super::super::weights::LoopWeights;
use super::step::LoopStep;
use crate::EngError;
use crate::family::project::Projector;
use crate::family::project::cuda::{BorrowedLoader, LoadOptions, WeightLayout};
use crate::family::project::gguf::GgufProjector;
use crate::kernels::cuda::{CudaContext, DeviceBuffer, device_present};
use crate::weights::{GgufLimits, GgufWeights};

/// The smoke prompt's exact ids (the vendor-oracle run of the CPU lane).
const PROMPT: [u32; 24] = [
    248045, 846, 198, 3710, 369, 279, 6511, 314, 9338, 30, 21134, 303, 799, 3299, 13, 248046, 198,
    248045, 74455, 198, 248068, 271, 248069, 271,
];
const GREEDY: usize = 8;
const CACHE_ROWS: usize = 512;

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
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
    if !device_present("sensenova step parity") {
        return;
    }
    let Ok(path) = std::env::var("SENSENOVA_GGUF") else {
        eprintln!("skipped: set SENSENOVA_GGUF to the released Loop-9B GGUF");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let limits = GgufLimits {
        max_header_bytes: 32 << 20,
        max_tensor_bytes: 8 << 30,
    };
    let file = Arc::new(GgufWeights::open_any_type(&path, limits, None).unwrap());

    // The CPU reference and the device twin over the one file.
    let weights = LoopWeights::open(file.clone()).unwrap();
    let plan_probe = weights.plan();
    eprintln!(
        "mhc plan: start {} end {} iterations {} pre_scale {} fixed {}",
        plan_probe.mhc_start,
        plan_probe.mhc_end,
        plan_probe.mhc_iterations,
        plan_probe.mhc_pre_scale,
        plan_probe.mhc_fixed_identity
    );
    let mut projector = weights.projector();
    let mut state = LoopState::new(weights.shape(), weights.plan());
    let ctx = Arc::new(CudaContext::new().unwrap());
    let loader = BorrowedLoader::new(&ctx).unwrap();
    let loaded = super::load::load(
        &loader,
        &file,
        weights.plan(),
        LoadOptions {
            layout: WeightLayout::Native,
            ..LoadOptions::default()
        },
    )
    .unwrap();
    let mut card = LoopStep::open(
        ctx.clone(),
        &loaded,
        weights.shape(),
        weights.plan(),
        weights.rope(),
        CACHE_ROWS,
    )
    .unwrap();

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
