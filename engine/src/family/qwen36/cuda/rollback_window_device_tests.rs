//! The device rollback of the verify window on the whole 4.00 bpw EXL3 model: the cases the D1
//! checks do not name. Device tests that COMPILE on the host build of the CUDA container and RUN
//! only on a device, one process per test on the GPU held exclusively. NONE HAS RUN.
//!
//! | test | what |
//! |---|---|
//! | `unarmed` | a step that wants the device rollback and holds no slots refuses a window before any change |
//! | `small_reserve` | slots that no admission charges and the reserve cannot hold refuse arming and leave the step |
//! | `cancel` | a reset with a window open discards it; the next job equals a fresh baseline |
//! | `late_poison` | a state poisoned between the window and its commit stays poisoned |
//! | `sixteen_steps` | sixteen ordinary decode steps after a rejected window equal the uninterrupted baseline, bit for bit, and so does the exported state |
//!
//! Environment as `d1_window_device_tests`: `FIFI_EXL3_CHECKPOINT`, `FIFI_SPEC_TEXT`,
//! `FIFI_SPEC_TOKENIZER`, `FIFI_D1_KV`, `FIFI_D1_CAPTURE`.

use super::{
    d1_window_device_tests::{
        arm_rollback, build_step, capture_from_env, exported_difference, kv_mode_from_env,
        load_pack, step_options, text_stream,
    },
    step::{CaptureConfig, QwenResidentStep},
};
use crate::{
    EngError,
    family::{
        qwen36::{
            rollback::Cover,
            settings::{QWEN36_MAX_WINDOW_ROWS, RollbackMode},
        },
        sampler::Sampler,
        session::{DecodeSession, TokenStepper},
        speculate::{d1_oracle::first_bit_difference, driver::WindowModel, window::Commit},
    },
};

const CONTEXT: usize = 2_048;
const PREFIX: usize = 600;
const ROWS: usize = 4;

fn fixture(mode: RollbackMode) -> (QwenResidentStep<'static>, Vec<u32>) {
    let (_tokenizer, stream) = text_stream(PREFIX + 64);
    let mut step = build_step(
        &load_pack(),
        &step_options(kv_mode_from_env(), CONTEXT),
        CaptureConfig {
            capture: capture_from_env(),
            section_marks: false,
        },
    )
    .expect("step");
    arm_rollback(&mut step, mode);
    (step, stream)
}

fn prefill(step: &mut QwenResidentStep<'static>, stream: &[u32]) -> DecodeSession {
    let mut session = DecodeSession::default();
    session.reset(step).expect("reset");
    session
        .prefill(step, &stream[..PREFIX], CONTEXT)
        .expect("prefill");
    session
}

/// A verdict that keeps `kept` of the window's rows.
fn keeps(base: usize, kept: usize) -> Commit {
    Commit {
        committed_rows: base + kept,
        kept_rows: kept,
        discarded_rows: ROWS - kept,
        state_slot: kept - 1,
    }
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn unarmed_a_step_that_wants_the_device_rollback_refuses_a_window_before_any_change() {
    let (mut step, stream) = fixture(RollbackMode::Host);
    step.request_device_rollback();
    let _session = prefill(&mut step, &stream);
    let before = step.export_state().expect("export");
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    let error = step.run_window(inputs).expect_err("no slots are armed");
    assert!(
        format!("{error:?}").contains("no rollback slots"),
        "{error:?}"
    );
    let after = step.export_state().expect("the state is still valid");
    assert_eq!(exported_difference(&before, &after), None);
    // The step goes on as a plain one-row step.
    step.run_token(inputs[0], false).expect("a one-row step");
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn small_reserve_slots_that_no_admission_charges_and_the_reserve_cannot_hold_refuse_arming() {
    let (mut step, stream) = fixture(RollbackMode::Host);
    let _session = prefill_fresh_only(&mut step);
    // The arming call of a fresh step: refused by the reserve check before any allocation.
    let (mut fresh, _) = fixture(RollbackMode::Host);
    let error = fresh
        .enable_device_rollback(
            QWEN36_MAX_WINDOW_ROWS,
            Cover::Reserve { headroom: 1 << 20 },
            0,
        )
        .expect_err("a 1 MiB reserve cannot hold 7 slots");
    assert!(matches!(error, EngError::MemoryPlan(_)), "{error:?}");
    assert!(!fresh.device_rollback_armed(), "the step is untouched");
    // And a step that has already run is not armed late.
    let error = step
        .enable_device_rollback(QWEN36_MAX_WINDOW_ROWS, Cover::Arena, 0)
        .expect_err("armed after the first token");
    assert!(format!("{error:?}").contains("already run"), "{error:?}");
    let _ = stream;
}

/// Runs one token so the step counts as started.
fn prefill_fresh_only(step: &mut QwenResidentStep<'static>) -> u32 {
    step.run_token(1, false).expect("one token")
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn cancel_a_reset_with_a_window_open_discards_it_and_the_next_job_is_clean() {
    let (mut step, stream) = fixture(RollbackMode::Device);
    let _session = prefill(&mut step, &stream);
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    step.run_window(inputs).expect("window");
    step.reset_state().expect("the cancel resets the session");
    let error = step
        .commit_window(&keeps(PREFIX - 1, 2))
        .expect_err("no window");
    assert!(format!("{error:?}").contains("no window"), "{error:?}");
    assert_eq!(step.tokens(), 0);
    // The next job: a window runs again over a fresh prefill and rolls back to the baseline.
    let _next = prefill(&mut step, &stream);
    let base = step.tokens();
    step.run_window(inputs).expect("a window on the next job");
    step.commit_window(&keeps(base, 1)).expect("rollback");
    let after = step.export_state().expect("export");
    let (mut fresh, _) = fixture(RollbackMode::Host);
    let _ = prefill(&mut fresh, &stream);
    fresh.run_token(inputs[0], false).expect("one row");
    let want = fresh.export_state().expect("export");
    assert_eq!(exported_difference(&after, &want), None);
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn late_poison_a_state_poisoned_between_the_window_and_its_commit_stays_poisoned() {
    let (mut step, stream) = fixture(RollbackMode::Device);
    let _session = prefill(&mut step, &stream);
    let base = step.tokens();
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    step.run_window(inputs).expect("window");
    step.poison_state();
    assert!(step.commit_window(&keeps(base, 2)).is_err());
    assert!(step.export_state().is_err(), "it was not healed");
    assert!(
        step.run_token(inputs[0], false).is_err(),
        "and it refuses steps"
    );
    step.reset_state().expect("a reset is the way back");
    assert!(step.export_state().is_ok());
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT"]
fn sixteen_steps_after_a_rejected_window_equal_the_uninterrupted_baseline() {
    let (mut step, stream) = fixture(RollbackMode::Device);
    let _session = prefill(&mut step, &stream);
    let base = step.tokens();
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    for kept in 1..ROWS {
        let _session = prefill(&mut step, &stream);
        step.run_window(inputs).expect("window");
        step.window_greedy_probe();
        step.commit_window(&keeps(base, kept)).expect("rollback");
        let mut got = Vec::new();
        for token in &stream[PREFIX - 1 + kept..PREFIX - 1 + kept + 16] {
            got.push(step.logits(*token).expect("decode"));
        }
        let state = step.export_state().expect("export");

        let (mut reference, _) = fixture(RollbackMode::Host);
        let _ = prefill(&mut reference, &stream);
        for token in &inputs[..kept] {
            reference.run_token(*token, false).expect("baseline row");
        }
        for (at, token) in stream[PREFIX - 1 + kept..PREFIX - 1 + kept + 16]
            .iter()
            .enumerate()
        {
            let want = reference.logits(*token).expect("baseline decode");
            assert_eq!(
                first_bit_difference(&got[at], &want),
                None,
                "kept {kept}: decode step {at} after the rollback"
            );
        }
        let want = reference.export_state().expect("export");
        assert_eq!(exported_difference(&state, &want), None, "kept {kept}");
    }
}

impl QwenResidentStep<'_> {
    /// Draws the greedy token of row 0 of the open window, as the driver does, to prove that
    /// the window answers before its verdict.
    fn window_greedy_probe(&mut self) {
        let drawn = self.draw(
            0,
            &mut Sampler::greedy(),
            super::d1_window_device_tests::SAMPLER_VOCAB,
        );
        assert!(drawn.is_ok(), "{drawn:?}");
    }
}
