//! The recovery gate of a failed window commit on the REAL step. COMPILED-ONLY: this is a device
//! test that builds in the CUDA container and runs only on a device, in a GPU window nobody has
//! authorised yet. NONE HAS RUN.
//!
//! | test | what |
//! |---|---|
//! | `a_failed_commit_leaves_a_step_that_only_a_reset_recovers` | after a host commit fails (an injected refusal before `begin_import`), decode, export, import, the pinned-prefix restore and the pinned-prefix save are all refused with the "reset" message, and `reset_state` clears the gate so decode works again |
//!
//! It is a child of the step so that it can reach the pinned pool and the injection field.
//! Environment as `d1_window_device_tests`: `FIFI_EXL3_CHECKPOINT`, `FIFI_SPEC_TEXT`,
//! `FIFI_SPEC_TOKENIZER`, `FIFI_D1_KV`, `FIFI_D1_CAPTURE`.

use super::super::{
    d1_window_device_tests::{
        build_step, capture_from_env, kv_mode_from_env, load_pack, step_options, text_stream,
    },
    prefix::CudaPrefillPath,
    state::pinned::PinnedSnapshot,
};
use super::{CaptureConfig, QwenResidentStep};
use crate::{
    EngError,
    family::{
        session::{DecodeSession, SaveKind},
        speculate::{d1_oracle::Fault, driver::WindowModel, window::Commit},
    },
};

const CONTEXT: usize = 2_048;
const PREFIX: usize = 600;
const ROWS: usize = 4;

/// A pinned host snapshot of the committed state (no device tip), taken the way the prefix tests do.
fn pinned_snapshot(step: &mut QwenResidentStep<'_>) -> Result<PinnedSnapshot, EngError> {
    let pool = step
        .prefix_pins
        .as_mut()
        .ok_or_else(|| EngError::BadRequest("the pinned pool is absent".into()))?;
    let pending = pool.begin(step.context, &step.stream, &step.state, None)?;
    pool.finish(step.context, &step.stream, &step.state, pending)
}

/// The refusal must come from the gate, not from some other missing piece.
fn refused_by_the_gate<T>(what: &str, result: Result<T, EngError>) {
    let error = match result {
        Ok(_) => panic!("{what} must be refused while the recovery gate is set"),
        Err(error) => error,
    };
    assert!(
        format!("{error:?}").contains("reset_state"),
        "{what}: refused, but not by the gate: {error:?}"
    );
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT. COMPILED-ONLY, never run"]
fn a_failed_commit_leaves_a_step_that_only_a_reset_recovers() {
    let (_tokenizer, stream) = text_stream(PREFIX + 64);
    let loaded = load_pack();
    let mut step = build_step(
        &loaded,
        &step_options(kv_mode_from_env(), CONTEXT),
        CaptureConfig {
            capture: capture_from_env(),
            section_marks: false,
        },
    )
    .expect("step");
    step.enable_pinned_prefix().expect("the pinned pool");
    assert!(step.pinned_prefix_enabled(), "the pinned pool was refused");
    let mut session = DecodeSession::default();
    session.reset(&mut step).expect("reset");
    session
        .prefill(&mut step, &stream[..PREFIX], CONTEXT)
        .expect("prefill");
    // What a restore would write back, taken while the state is healthy.
    let pinned = pinned_snapshot(&mut step).expect("a pinned snapshot");
    let exported = step.export_state().expect("an exported state");
    let base = step.tokens();
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    // A partial verdict whose import is refused before `begin_import`.
    step.window_fault = Some(Fault::CheckImport);
    step.run_window(inputs).expect("the window runs");
    let keeps = Commit {
        committed_rows: base + 1,
        kept_rows: 1,
        discarded_rows: ROWS - 1,
        state_slot: 0,
    };
    step.commit_window(&keeps).expect_err("the check fails");

    // 17.2: every consumer of the live state is refused, and so is every restore.
    refused_by_the_gate("a decode step", step.run_token(inputs[0], false));
    refused_by_the_gate("an export", step.export_state());
    refused_by_the_gate("an import", step.import_state(&exported));
    refused_by_the_gate(
        "the pinned-prefix restore (17.5-1)",
        step.restore_pinned_prefix(&pinned),
    );
    let path = CudaPrefillPath::Steps(step.state_layout().kv.mode);
    refused_by_the_gate(
        "the pinned-prefix save",
        step.stage_pinned_prefix(&session, &stream[..PREFIX], SaveKind::Tip, path),
    );
    refused_by_the_gate("a prefill chunk", step.run_chunk(base, &inputs[..1]));

    // Only a reset recovers it; afterwards the same window runs and commits.
    step.reset_state().expect("the reset");
    step.run_token(inputs[0], false)
        .expect("decode works after the reset");
    step.import_state(&exported)
        .expect("an import works after the reset");
    step.run_window(inputs).expect("the same window runs again");
    step.commit_window(&keeps).expect("and commits");
}

#[test]
#[ignore = "the GPU, held exclusively: loads the real 4.00 bpw pack in FIFI_EXL3_CHECKPOINT. COMPILED-ONLY, never run"]
fn an_unresolved_window_refuses_every_restore_and_only_a_reset_recovers() {
    let (_tokenizer, stream) = text_stream(PREFIX + 64);
    let loaded = load_pack();
    let mut step = build_step(
        &loaded,
        &step_options(kv_mode_from_env(), CONTEXT),
        CaptureConfig {
            capture: capture_from_env(),
            section_marks: false,
        },
    )
    .expect("step");
    step.enable_pinned_prefix().expect("the pinned pool");
    assert!(step.pinned_prefix_enabled(), "the pinned pool was refused");
    let mut session = DecodeSession::default();
    session.reset(&mut step).expect("reset");
    session
        .prefill(&mut step, &stream[..PREFIX], CONTEXT)
        .expect("prefill");
    let pinned = pinned_snapshot(&mut step).expect("a pinned snapshot");
    let exported = step.export_state().expect("an exported state");
    let inputs = &stream[PREFIX - 1..PREFIX - 1 + ROWS];
    // The rows ran and no verdict came (a draw failed in the driver, say): the window is
    // unresolved. DESIGN 17.10: no restore closes it silently.
    step.run_window(inputs).expect("the window runs");
    refused_by_the_gate("an import", step.import_state(&exported));
    refused_by_the_gate(
        "the pinned-prefix restore",
        step.restore_pinned_prefix(&pinned),
    );
    // Only a reset recovers it.
    step.reset_state().expect("the reset");
    step.import_state(&exported)
        .expect("an import works after the reset");
}
