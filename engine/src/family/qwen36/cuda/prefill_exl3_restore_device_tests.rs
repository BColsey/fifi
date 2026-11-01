//! A saved prefix of the EXL3 pack restored under chunked prefill against the cold chunked prefill
//! of the same prompt, on the whole 4.00 bpw model, one test per process on the 3090 held
//! exclusively (about 21 GiB).
//!
//! The store of a step with `prefix_reuse` keeps a pinned host snapshot and a device tip of the
//! committed state (`start.rs` enables both by default). A snapshot is taken after the first `cut`
//! prompt rows were fed in chunks, the state is reset, and the snapshot is restored either from the
//! device tip or, after a foreign prompt overwrote the tip, from the pinned host copy. The rest of
//! the prompt is then fed in chunks (starting off the cold partition when `cut` is not a multiple of
//! the chunk size) or one token per step, and the next-token distributions of the prompt and of a
//! teacher-forced continuation are compared with the cold chunked run by the bounds of the
//! chunk-versus-token test in `prefill_exl3_step_device_tests.rs`.

use std::path::Path;

use super::kv8::snapshot;
use crate::family::qwen36::cuda::{
    common_tests::{dims, inventory},
    exl3_load::load_exl3_dir,
    prefill_driver::{FIXED_ADMISSION_MARGIN, PrefillBudget},
    prefill_exl3_step_device_tests::{
        CHUNK_ROWS, CONTINUATION, SECOND_PROMPT, all_ids, argmax, assert_rows, fp16_kv, options,
        rules,
    },
    prefill_plan::Extra,
    resources::StepResources,
    state::pinned::PinnedSnapshot,
    step::QwenResidentStep,
};
use crate::{
    EngError,
    family::{
        project::cuda::{CudaProjector, CudaWeights, Reserve},
        session::TokenStepper,
    },
};

/// Prompt rows of every case.
const PROMPT_ROWS: usize = 3_300;
/// Rows of the first chunk of a cold prefill: the cap of the Int8G32 chunk path, or the first
/// `m_pos` tier at the default KV.
fn first_chunk() -> usize {
    if fp16_kv() {
        2_048
    } else {
        CHUNK_ROWS as usize
    }
}

/// Restore points: one on the boundary of the cold partition, one inside a chunk of it and one
/// that leaves a partial chunk before it.
fn cuts() -> [usize; 3] {
    if fp16_kv() {
        [1_024, 2_048, 2_500]
    } else {
        [1_024, 1_280, 2_500]
    }
}
/// The restore point of the cases that continue one token per step.
const ONE_TOKEN_CUT: usize = 2_500;

fn bad(message: &str) -> EngError {
    EngError::BadRequest(format!("EXL3 prefix restore: {message}"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// The rest of the prompt in chunks from the restored position.
    Chunked,
    /// The rest of the prompt one token per step.
    OneToken,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Case {
    cut: usize,
    /// A foreign prompt overwrote the device tip, so the restore comes from the pinned host copy.
    foreign: bool,
    tail: Tail,
}

fn cases() -> Vec<Case> {
    let mut list = Vec::new();
    for cut in cuts() {
        for foreign in [false, true] {
            list.push(Case {
                cut,
                foreign,
                tail: Tail::Chunked,
            });
        }
    }
    for foreign in [false, true] {
        list.push(Case {
            cut: ONE_TOKEN_CUT,
            foreign,
            tail: Tail::OneToken,
        });
    }
    list
}

struct Scored {
    continuation: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

/// The logits row after the prompt's last token and after each of [`CONTINUATION`] further tokens:
/// the greedy ones, or the `forced` ones.
fn score(
    step: &mut QwenResidentStep<'_>,
    last: u32,
    forced: Option<&[u32]>,
) -> Result<Scored, EngError> {
    let mut rows = vec![step.logits(last)?];
    let mut continuation = Vec::with_capacity(CONTINUATION);
    for index in 0..CONTINUATION {
        let token = match forced {
            Some(tokens) => tokens[index],
            None => u32::try_from(argmax(&rows[index])).map_err(|_| bad("token id"))?,
        };
        continuation.push(token);
        rows.push(step.logits(token)?);
    }
    Ok(Scored { continuation, rows })
}

/// Whether every logit of every row has the same bits in both.
fn same_bits(left: &[Vec<f32>], right: &[Vec<f32>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn feed_chunked(step: &mut QwenResidentStep<'_>, tokens: &[u32]) -> Result<(), EngError> {
    if fp16_kv() {
        return step.feed_rows(tokens);
    }
    let rows = CHUNK_ROWS as usize;
    for piece in tokens.chunks(rows) {
        step.feed_rows_chunked(piece, rows)?;
    }
    Ok(())
}

/// Restores `saved` and returns whether the device tip served it.
fn restore(step: &mut QwenResidentStep<'_>, saved: &PinnedSnapshot) -> Result<bool, EngError> {
    let pool = step
        .prefix_pins
        .as_mut()
        .ok_or_else(|| bad("the pinned pool is absent"))?;
    pool.restore(
        step.context,
        &step.stream,
        &mut step.state,
        saved,
        step.device_tip.as_ref(),
    )
}

/// One case: the first `cut` rows in chunks, a snapshot, a reset, a foreign overwrite if asked, the
/// restore, then the tail of the prompt and the continuation `forced` by the cold run.
fn restored(
    step: &mut QwenResidentStep<'_>,
    (head, last): (&[u32], u32),
    other: &[u32],
    (case, forced): (Case, &[u32]),
) -> Result<Scored, EngError> {
    let Case { cut, foreign, tail } = case;
    step.reset_state()?;
    feed_chunked(step, &head[..cut])?;
    let saved = snapshot(step)?;
    step.reset_state()?;
    if foreign {
        if other[..cut] == head[..cut] {
            return Err(bad("the foreign prompt is not distinct"));
        }
        feed_chunked(step, &other[..cut])?;
    }
    let device = restore(step, &saved)?;
    if device == foreign || step.tokens() != cut {
        return Err(bad("restore source or recovered position differs"));
    }
    match tail {
        Tail::Chunked => feed_chunked(step, &head[cut..])?,
        Tail::OneToken => {
            for &token in &head[cut..] {
                step.feed(token)?;
            }
        }
    }
    score(step, last, Some(forced))
}

fn open_step(
    (projector, weights, free): (&'static CudaProjector, &'static CudaWeights, usize),
) -> Result<QwenResidentStep<'static>, EngError> {
    let opts = options();
    let res: &'static StepResources = Box::leak(Box::new(StepResources::new(
        projector.context(),
        dims(),
        &inventory(),
        &opts,
    )?));
    let mut step = QwenResidentStep::new(projector.context(), res, weights, &opts, rules())?;
    if !fp16_kv() {
        step.set_kv8_chunked_prefill(true)?;
        step.set_m_sess_cap(CHUNK_ROWS);
    }
    step.enable_pinned_prefix()?;
    let tip = step.enable_device_tip()?;
    if tip == 0 {
        return Err(bad("the device tip fell back to the host"));
    }
    let verdict = step.admit_prefill(
        PrefillBudget {
            device_free: free.saturating_sub(if fp16_kv() {
                1 << 30
            } else {
                FIXED_ADMISSION_MARGIN
            }),
            extras: vec![Extra {
                name: "device-prefix-delta",
                fixed: tip,
                bytes_per_token: 0,
            }],
        },
        true,
        !fp16_kv(),
        false,
    )?;
    if !format!("{verdict:?}").starts_with("Ready") {
        return Err(bad("the chunk path was not admitted"));
    }
    Ok(step)
}

#[test]
fn the_cases_cover_aligned_and_split_cuts_both_routes_and_both_tails() {
    let list = cases();
    let rows = first_chunk();
    assert!(list.iter().any(|c| c.cut % rows == 0));
    assert!(list.iter().any(|c| c.cut % rows != 0 && c.cut < 2 * rows));
    assert!(list.iter().any(|c| c.cut % rows != 0 && c.cut > rows));
    for tail in [Tail::Chunked, Tail::OneToken] {
        for foreign in [false, true] {
            assert!(list.iter().any(|c| c.tail == tail && c.foreign == foreign));
        }
    }
    assert!(list.iter().all(|c| c.cut > 0 && c.cut < PROMPT_ROWS - 1));
}

#[test]
#[ignore = "the 3090, held exclusively (21 GiB): loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn a_restored_prefix_matches_the_cold_chunked_prefill_on_the_real_pack() {
    let dir = std::env::var_os("FIFI_EXL3_CHECKPOINT").expect("FIFI_EXL3_CHECKPOINT");
    let ids = all_ids();
    let prompt: Vec<u32> = ids.iter().take(PROMPT_ROWS).copied().collect();
    let other: Vec<u32> = ids
        .iter()
        .skip(SECOND_PROMPT.0)
        .take(PROMPT_ROWS)
        .copied()
        .collect();
    assert_eq!(prompt.len(), PROMPT_ROWS, "the ids file is long enough");
    assert_eq!(other.len(), PROMPT_ROWS, "the ids file is long enough");
    let projector = CudaProjector::new().expect("projector");
    let weights = load_exl3_dir(&projector, Path::new(&dir), Reserve::default()).expect("load");
    let free = projector.memory_info().expect("memory").free;
    let projector: &'static CudaProjector = Box::leak(Box::new(projector));
    let weights: &'static CudaWeights = Box::leak(Box::new(weights));
    let mut step = open_step((projector, weights, free)).expect("step");
    let (head, last) = prompt.split_at(PROMPT_ROWS - 1);
    let last = last[0];
    step.reset_state().expect("reset");
    feed_chunked(&mut step, head).expect("cold chunked prefill");
    let cold = score(&mut step, last, None).expect("cold continuation");
    for case in cases() {
        let run = restored(&mut step, (head, last), &other, (case, &cold.continuation))
            .unwrap_or_else(|e| panic!("{case:?}: {e:?}"));
        println!("case {case:?}");
        assert_rows(&cold.rows, &run.rows);
        // The partition of the rest equals the cold one at a chunk boundary, so every row is the
        // same value, not merely close.
        if case.tail == Tail::Chunked && case.cut % first_chunk() == 0 {
            assert!(
                same_bits(&cold.rows, &run.rows),
                "{case:?}: a restore at a chunk boundary must reproduce the cold rows exactly"
            );
        }
    }
}
