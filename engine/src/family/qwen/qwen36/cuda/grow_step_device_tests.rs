//! A sequence part grown on a real step: the resident Qwen3.6 step over the real weights runs a
//! sequence in a part of 1024 cells, the part is grown to 3072 cells between two tokens (a graph
//! baked over the old planes is dropped and the next token warms and captures again), and the
//! rows after the growth equal the rows of a part that had 3072 cells from its first token. A
//! growth the card cannot hold is refused with the part as it was, and the part still decodes.
//!
//! The device phase is a plain `Result` function (only `?` inside) that returns host data; the
//! owners drop on return and only then do the asserts run (the isolation rule of
//! `native_device_rig.rs`). The test is ignored and run by name on the card held exclusively.

use super::common_tests;
use super::dims::KvMode;
use super::kv8_step_device_tests::{full_options, load_full_native, rules};
use super::native_device_rig::device_present;
use super::resources::StepResources;
use crate::EngError;
use crate::family::qwen::qwen36::cuda::step::QwenResidentStep;

const SMALL: usize = 1024;
const LARGE: usize = 3072;
/// Tokens fed before the growth: below the small part's cells.
const BEFORE: usize = 1000;
/// Tokens in all: the sequence crosses the 1024 cells of the small part.
const TOTAL: usize = 1100;

fn hash(row: &[u8]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &byte| {
        (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

fn ids() -> Vec<u32> {
    (0..TOTAL as u32)
        .map(|i| 1_000 + (i * 7919) % 200_000)
        .collect()
}

/// What one mode's phase observed.
struct Observed {
    mode: KvMode,
    reference: Vec<u64>,
    candidate: Vec<u64>,
    refusal: Option<String>,
    cells_after_refusal: usize,
    cells_after_growth: usize,
}

fn phase(
    context: &crate::kernels::cuda::CudaContext,
    weights: &crate::family::project::cuda::CudaWeights,
    mode: KvMode,
) -> Result<Observed, EngError> {
    let dims = common_tests::dims();
    let inventory = common_tests::inventory();
    let options = full_options(mode);
    let res = StepResources::new(context, dims.clone(), &inventory, &options)?;
    let mut step = QwenResidentStep::new(context, &res, weights, &options, rules()?)?;
    let ids = ids();
    let mut large = step.new_seq(LARGE)?;
    let mut small = step.new_seq(SMALL)?;

    let run = |step: &mut QwenResidentStep<'_>, from: usize, to: usize| {
        let mut rows = Vec::with_capacity(to - from);
        for &id in &ids[from..to] {
            step.run_token(id, true)?;
            rows.push(hash(step.row_bytes()));
        }
        Ok::<_, EngError>(rows)
    };

    // The reference: 3072 cells from the first token.
    step.swap_seq(&mut large);
    let reference = run(&mut step, 0, TOTAL)?;
    step.swap_seq(&mut large);

    // The candidate: 1024 cells, grown after BEFORE tokens.
    step.swap_seq(&mut small);
    let mut candidate = run(&mut step, 0, BEFORE)?;
    step.swap_seq(&mut small);
    // The card cannot hold a part of twice its free memory: refused, the part as it was.
    let per_cell = (step.seq_bytes_at(2 * SMALL) - step.seq_bytes_at(SMALL)) / SMALL as u64;
    let free = step.free_device_bytes().unwrap_or(0);
    let refusal = step
        .grow_seq(&mut small, (free * 2 / per_cell) as usize)
        .err()
        .map(|error| format!("{error:?}"));
    let cells_after_refusal = small.cells();
    step.grow_seq(&mut small, LARGE)?;
    let cells_after_growth = small.cells();
    step.swap_seq(&mut small);
    candidate.extend(run(&mut step, BEFORE, TOTAL)?);
    step.swap_seq(&mut small);
    Ok(Observed {
        mode,
        reference,
        candidate,
        refusal,
        cells_after_refusal,
        cells_after_growth,
    })
}

fn observe() -> Result<Vec<Observed>, EngError> {
    let (projector, weights) = load_full_native()?;
    let mut seen = Vec::new();
    for mode in [KvMode::Int8G32, KvMode::Fp16] {
        seen.push(phase(projector.context(), &weights, mode)?);
    }
    Ok(seen)
}

#[test]
#[ignore = "the 3090, held exclusively: loads the real model; run by name"]
fn a_part_grown_between_two_tokens_gives_the_rows_of_one_sized_for_the_run() {
    if !device_present("a_part_grown_between_two_tokens") {
        return;
    }
    let seen = observe().expect("device phase");
    for run in seen {
        let mode = run.mode;
        assert_eq!(run.reference.len(), TOTAL);
        assert_eq!(run.candidate.len(), TOTAL);
        let first_difference = (0..TOTAL).find(|&at| run.reference[at] != run.candidate[at]);
        assert_eq!(
            first_difference, None,
            "{mode:?}: the row of token {first_difference:?} differs from the part that had 3072 cells"
        );
        let refusal = run.refusal.expect("a growth beyond the card is refused");
        eprintln!("{mode:?}: growth beyond the card refused: {refusal}");
        assert_eq!(
            run.cells_after_refusal, SMALL,
            "{mode:?}: the refusal kept the planes"
        );
        assert_eq!(run.cells_after_growth, LARGE);
    }
}
