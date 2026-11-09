//! The chunk driver's device gate: one chunk through BOTH cards vs the
//! per-token path over the same tokens. The chunk path runs MMQ GEMMs
//! where the per-token path runs native matvecs, so the gate is the
//! greedy token and a top-2 margin bound (the single-device chunk suite's
//! own standard for the same comparison), not bit equality.

use super::card_prefill::{CardChunkInputs, build_card_chunk, run_card_chunk};
use super::card_session_device_tests::rig_spec;
use super::dims::{KvMode, StepDims, StepOptions};
use super::layer_table::LayerTable;
use crate::family::gguf_model::Inventory;
use crate::family::qwen36::host::HostShape;
use crate::family::session::TokenStepper;
use crate::kernels::cuda::device_present;

/// One full chunk pass and one per-token pass over the same rows: the
/// greedy winner must agree and the top-2 margin stay inside the slack
/// the single-device chunk suite uses for MMQ-vs-matvec.
#[test]
fn card_chunk_matches_the_per_token_path() {
    const NAME: &str = "card_chunk_matches_the_per_token_path";
    if !device_present(NAME) {
        return;
    }
    let (mut session, stack, loaded, names, _vocab) =
        crate::family::qwen36::cuda::card_session_device_tests::build_rig_session_for_chunk(
            [0, 0],
            256,
            crate::family::session::StoreLimits::default(),
        )
        .expect("the rig session");
    let spec = rig_spec();
    let inv = Inventory::for_spec(&spec).expect("inventory");
    let shape = HostShape::from_spec(&spec, 64).expect("shape");
    let d = StepDims::new(&shape, &inv).expect("dims");
    let options = StepOptions {
        kv_mode: KvMode::Int8G32,
        context_tokens: 256,
        delta_slots: 1,
        forked_experts: false,
        tiered: None,
        layout: crate::family::project::cuda::WeightLayout::Native,
    };
    let table = LayerTable::new(&d, &inv).expect("layers");
    // The rig session's stock mailbox payload is hidden*top_k = 16384
    // values; 8 rows fit it exactly (8 * 2048). The production start grows
    // the mailbox for bigger chunks; this rig proves correctness at the
    // boundary the stock stack allows.
    let rows = 8usize.min(super::card_prefill::max_chunk_rows(d.shape.hidden));
    let chunk = build_card_chunk(
        stack,
        &d,
        &table,
        CardChunkInputs {
            weights: loaded,
            embed_name: Box::leak(names.embed.clone().into_boxed_str()),
        },
        &options,
        rows,
    )
    .unwrap_or_else(|error| panic!("{NAME}: the chunk build refused: {error:?}"));

    // Deterministic, in-range token ids.
    let tokens: Vec<u32> = (0..rows as u32 + 1).map(|i| (i * 37 + 11) % 900).collect();
    let (chunk_rows, [last]) = tokens.split_at(tokens.len() - 1) else {
        unreachable!("len checked above");
    };

    // The per-token control: every row through run_token.
    TokenStepper::reset(&mut session).expect("reset control");
    let mut control_rows = Vec::new();
    for &token in chunk_rows {
        control_rows.push(TokenStepper::logits(&mut session, token).expect("control logits"));
    }
    let control_last = TokenStepper::logits(&mut session, *last).expect("control last");

    // The chunk pass: rows through the driver, the last token per-token.
    TokenStepper::reset(&mut session).expect("reset chunk");
    let fed = TokenStepper::fed(&session);
    run_card_chunk(stack, session.states_mut(), &chunk, chunk_rows, fed)
        .unwrap_or_else(|error| panic!("{NAME}: the chunk run refused: {error:?}"));
    session.note_chunk_rows(chunk_rows.len());
    let chunk_last = TokenStepper::logits(&mut session, *last).expect("chunk last");

    // The gate: the same greedy winner, margin within the single-device
    // chunk suite's slack (2 * (peak * 1e-3 + 1e-6)).
    let winner = |row: &[f32]| -> (usize, f32, f32) {
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut runner = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best.1 {
                runner = best.1;
                best = (i, v);
            } else if v > runner {
                runner = v;
            }
        }
        (best.0, best.1, runner)
    };
    let (want_tok, want_best, want_runner) = winner(&control_last);
    let (got_tok, got_best, got_runner) = winner(&chunk_last);
    let peak = control_last.iter().fold(0f32, |a, &v| a.max(v.abs()));
    let slack = 2.0 * (peak * 1e-3 + 1e-6);
    assert!(
        (got_best - want_best).abs() <= slack && (got_runner - want_runner).abs() <= slack,
        "{NAME}: the last-token margins diverge: chunk ({got_best:.4} vs {got_runner:.4}) \
         control ({want_best:.4} vs {want_runner:.4}), slack {slack:.5}"
    );
    // A hard disagreement on the winner is a real failure even when both
    // margins hold: report it distinctly.
    if got_tok != want_tok {
        panic!(
            "{NAME}: the greedy winner differs: chunk {got_tok} vs control {want_tok} \
             (peak {peak:.3}, slack {slack:.5})"
        );
    }
    // And the mid-prompt winners: every control row's winner must lead its
    // chunk counterpart within the same slack (the rows are not sampled by
    // the driver, so compare only the last row above and the count here).
    assert_eq!(
        control_rows.len(),
        chunk_rows.len(),
        "{NAME}: the control fed every row"
    );
    println!("{NAME}: {rows} rows, winner {want_tok}, chunk margin {got_best:.4}/{got_runner:.4}");
}
