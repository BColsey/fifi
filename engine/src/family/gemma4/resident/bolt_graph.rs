//! The host rules of the Bolt decode graph, as pure code.
//!
//! The graph replays the one-token decode body (embed, the per-layer planes, every layer, the
//! output norm, the logits projection and the softcap) as one captured CUDA graph instead of
//! about a thousand single launches per token. This file holds what needs no device: whether
//! the graph is wanted, the arming refusals, and the replay accounting. The eager path stays
//! the reference and the fallback; every capture refusal or failure pins the mode to eager.
//!
//! Arming refusals, all sticky:
//!
//! * `BOLT_TRACE_DIR` set. The trace hook reads device vectors back inside the body, and a
//!   capture refuses a synchronous copy; a traced step and a graphed step are different bodies.
//! * A step with more than one token. Only the one-token decode body is captured: a prefill
//!   batch is bandwidth-bound GEMV work that gains nothing from graph launch, and its row count
//!   changes the shape of the body.
//! * A hidden step (no logits). It skips the shared-KV layers, so it is a second body; it
//!   always runs eagerly.
//!
//! The per-step inputs (the token id, the position, the KV length) stay host-written words: the
//! body reads them from fixed device buffers (`Scratch::ids`, `Scratch::pos`,
//! `Scratch::kv_len`), which the step fills in place between replays, never reallocating. The
//! kernels that depend on the position or the length read device words, so the captured body
//! needs no per-step kernel argument.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the receipt getters are not read outside tests yet"
    )
)]

use crate::EngError;

/// The environment variable that arms the decode graph. Default off until the
/// graph is qualified; the refusal messages name it.
pub(crate) const DECODE_GRAPH_VAR: &str = "FIFI_BOLT_DECODE_GRAPH";

/// Parses the arming value: absent, `0` and `off` disarm; `1` and `on` arm;
/// anything else is an error, so a typo or an empty value refuses loudly
/// instead of silently serving eager.
pub(crate) fn parse_switch(value: Option<&str>) -> Result<bool, EngError> {
    match value {
        None | Some("0") | Some("off") => Ok(false),
        Some("1") | Some("on") => Ok(true),
        Some(other) => Err(EngError::BadRequest(format!(
            "bolt decode graph: {DECODE_GRAPH_VAR} accepts 1, on, 0 or off; got {other:?}"
        ))),
    }
}

/// Whether the environment asks for the decode graph; a bad value fails the
/// load instead of arming or disarming by accident.
pub(crate) fn decode_graph_wanted() -> Result<bool, EngError> {
    parse_switch(std::env::var(DECODE_GRAPH_VAR).ok().as_deref())
}

fn refuse(why: &str) -> EngError {
    EngError::BadRequest(format!("bolt decode graph: {why}"))
}

/// What the arming of the decode graph knows. The step builds it from its own state so the
/// check runs per step, not only at load.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DecodeGraphFacts {
    /// `BOLT_TRACE_DIR` is set: the body would read vectors back mid-step.
    pub(crate) trace_armed: bool,
    /// The step carries more than one token (a prefill batch).
    pub(crate) multi_token: bool,
    /// The step skips the output norm and the logits projection (a hidden decode).
    pub(crate) hidden: bool,
}

/// The arming rules. `Ok` means this step may run the captured body.
pub(crate) fn check_decode_graph(facts: DecodeGraphFacts) -> Result<(), EngError> {
    if facts.trace_armed {
        return Err(refuse(
            "unset BOLT_TRACE_DIR; a traced step reads back and cannot be captured",
        ));
    }
    if facts.multi_token {
        return Err(refuse(
            "the captured body is the one-token decode; a multi-token step runs eagerly",
        ));
    }
    if facts.hidden {
        return Err(refuse(
            "a hidden step skips the shared layers and runs eagerly",
        ));
    }
    Ok(())
}

/// The sticky mode of the decode graph: wanted, refused (with the reason), and the accounting
/// the pin tests and the receipt read. A refused mode never becomes armed again in this
/// process; the eager path is the reference and the fallback.
#[derive(Debug, Default)]
pub(crate) struct DecodeGraphMode {
    wanted: bool,
    refusal: Option<String>,
    warm: bool,
    replays: u64,
    eager_steps: u64,
}

impl DecodeGraphMode {
    /// A fresh mode for the process; `wanted` comes from the environment once.
    pub(crate) fn new(wanted: bool) -> Self {
        Self {
            wanted,
            ..Self::default()
        }
    }

    /// Whether the graph can still run.
    pub(crate) fn active(&self) -> bool {
        self.wanted && self.refusal.is_none()
    }

    /// The sticky refusal, if any.
    pub(crate) fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }

    /// Pins the mode to eager with `why`; a second refusal keeps the first reason.
    pub(crate) fn refuse(&mut self, why: &str) {
        if self.refusal.is_none() {
            self.refusal = Some(why.to_owned());
        }
    }

    /// Whether one clean eager logits step has warmed the kernels the capture
    /// records: a first launch loads its module, which a capture forbids.
    pub(crate) fn warm(&self) -> bool {
        self.warm
    }

    /// One clean eager step ran; the next clean step may capture.
    pub(crate) fn note_warm(&mut self) {
        self.warm = true;
    }

    /// One replay ran.
    pub(crate) fn note_replay(&mut self) {
        self.replays = self.replays.saturating_add(1);
    }

    /// One step ran the eager body (before the capture, after a refusal, or unwanted).
    pub(crate) fn note_eager(&mut self) {
        self.eager_steps = self.eager_steps.saturating_add(1);
    }

    /// Replays since load; the receipt prints this with the refusal.
    pub(crate) fn replays(&self) -> u64 {
        self.replays
    }

    /// Eager steps since load.
    pub(crate) fn eager_steps(&self) -> u64 {
        self.eager_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(trace_armed: bool, multi_token: bool, hidden: bool) -> DecodeGraphFacts {
        DecodeGraphFacts {
            trace_armed,
            multi_token,
            hidden,
        }
    }

    #[test]
    fn a_clean_single_token_logits_step_is_admitted() {
        assert!(check_decode_graph(facts(false, false, false)).is_ok());
    }

    #[test]
    fn every_refusal_names_its_cause() {
        let traced = format!(
            "{:?}",
            check_decode_graph(facts(true, false, false)).unwrap_err()
        );
        assert!(traced.contains("BOLT_TRACE_DIR"));
        assert!(traced.contains("unset"));
        let batch = format!(
            "{:?}",
            check_decode_graph(facts(false, true, false)).unwrap_err()
        );
        assert!(batch.contains("one-token"));
        let hidden = format!(
            "{:?}",
            check_decode_graph(facts(false, false, true)).unwrap_err()
        );
        assert!(hidden.contains("hidden"));
    }

    #[test]
    fn the_switch_accepts_one_and_on_only() {
        assert!(!parse_switch(None).unwrap());
        assert!(!parse_switch(Some("0")).unwrap());
        assert!(!parse_switch(Some("off")).unwrap());
        assert!(parse_switch(Some("1")).unwrap());
        assert!(parse_switch(Some("on")).unwrap());
        for bad in ["", "false", "yes", "ON"] {
            let error = format!("{:?}", parse_switch(Some(bad)).unwrap_err());
            assert!(
                error.contains("got"),
                "the refusal must quote the bad value {bad:?}"
            );
        }
    }

    #[test]
    fn warm_latches_after_one_clean_eager_step() {
        let mut mode = DecodeGraphMode::new(true);
        assert!(!mode.warm());
        mode.note_warm();
        mode.refuse("later");
        assert!(mode.warm(), "warmth outlives a refusal");
    }

    #[test]
    fn a_refusal_is_sticky_and_keeps_the_first_reason() {
        let mut mode = DecodeGraphMode::new(true);
        assert!(mode.active());
        mode.refuse("first");
        mode.refuse("second");
        assert!(!mode.active());
        assert_eq!(mode.refusal(), Some("first"));
    }

    /// Fields drop in declaration order, and the graph must be destroyed
    /// before every buffer whose address it baked. By NAME, never by line:
    /// the lines keep moving.
    #[test]
    fn the_graph_is_declared_before_every_buffer_it_bakes_in() {
        let source = include_str!("bolt.rs");
        let start = source
            .find("pub(crate) struct Bolt {")
            .expect("the Bolt struct");
        let end = source[start..].find("\n}").expect("the struct's end");
        let fields = &source[start..start + end];
        let at = |name: &str| {
            fields
                .find(&format!("{name}:"))
                .unwrap_or_else(|| panic!("field `{name}` missing from Bolt"))
        };
        let graph = at("graph");
        for buffer in ["ctx", "layers", "embed", "scratch", "logits", "rope_freqs"] {
            assert!(graph < at(buffer), "the graph must drop before `{buffer}`");
        }
    }

    #[test]
    fn an_unwanted_mode_is_never_active_but_counts_eager_steps() {
        let mut mode = DecodeGraphMode::new(false);
        assert!(!mode.active());
        mode.note_eager();
        assert_eq!(mode.eager_steps(), 1);
        assert_eq!(mode.replays(), 0);
    }

    #[test]
    fn replay_and_eager_counts_accumulate_saturating() {
        let mut mode = DecodeGraphMode::new(true);
        mode.note_replay();
        mode.note_replay();
        mode.note_eager();
        assert_eq!(mode.replays(), 2);
        assert_eq!(mode.eager_steps(), 1);
    }
}
