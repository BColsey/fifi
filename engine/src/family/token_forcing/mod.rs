//! Token forcing: a qualification gate that judges every generated token in its own context.
//!
//! Two greedy runs that differ in one early token stop being comparable: every later token
//! follows a different history. Token forcing, after SpecPilot (arXiv 2611.02501, section 3.3),
//! instead feeds a system's own output back through a reference model on the same weights and
//! asks, at each position, whether the reference would have chosen the same token from that
//! exact history. A mismatch is a token that differs from the reference's argmax; its margin is
//! the log-probability gap to that argmax.
//!
//! One scorer, the accepted build on the same weight pack, reads every output (`score`): the
//! prompt goes through the executor's own prompt path and each generated token through its
//! one-row step, the eager path of `TokenStepper::logits`. Three executions take part, all
//! generating greedily from the same prompts:
//!
//! - the reference: generation on the scorer's own numeric path (SpecPilot's autoregressive
//!   run). Its mismatch rate is near zero by construction;
//! - the baseline: a second legal execution of the accepted build that differs from the scorer
//!   in reduction order only: another prefill chunk size, the one setting it may change
//!   (SpecPilot's stock system);
//! - the candidate: the execution under test.
//!
//! The near-tie threshold `delta` is the largest log-probability gap between the reference's
//! and the baseline's tokens at their first departure, rounded up to a multiple of 1/8 nat
//! (`gate::calibrate`, which states the contract). When they never depart, the legal margin
//! measured is zero and delta takes 1/8, the strictest value; token identity says nothing about
//! the executions' logits, and delta then claims nothing about them. The reference and the
//! baseline must be executions of the scorer's build and settings, recorded from the settings
//! their executors were opened with, and the reference must match its scorer at every token.
//! The baseline is the comparison of both rules, because a lossless candidate's mismatches are
//! near ties of the same kind. A candidate is rejected when
//!
//! - its mismatch rate exceeds the baseline's by more than 0.5 points at one-sided 95%
//!   confidence (Newcombe's score interval for a difference of rates), or
//! - its unexplained mismatches with a margin above `2 * delta` are more frequent than the
//!   baseline's (exact conditional binomial test, one-sided, 0.05). An audit explains such an
//!   event when one prefill of the whole context, or the baseline execution, gives the
//!   generated token a margin of at most `2 * delta`, so that it is no event there
//!   (`gate::Audit`). At equal exposure and no baseline events, five unexplained events are
//!   the fewest that reject (four give p = 1/16).
//!
//! A pass also needs enough evidence (`gate::GateConfig`): for qualification, four requests of
//! 4,096 generated tokens each for the reference, the baseline and the candidate, a delta of at
//! most 0.5 nats, and every baseline event audited; otherwise the verdict is inconclusive. The
//! report (`gate::Report`, schema `fifi.token_forcing.v2`) carries the digests of its input
//! files, the counts, the bounds, a mismatch list with each margin class, and mismatch counts
//! by output depth.
//!
//! Scope. The gate compares executions of the same weights in the same format, and refuses
//! anything else: the difference between two weight sets or formats is a quality question for
//! the non-inferiority gate, not a defect test. It adds to the bitwise tests, the NLL
//! non-inferiority gate, top-1 agreement, exact copy and retrieval; it replaces none of them.
//!
//! Files. A device run (for Flint CUDA: `flint::cuda_executor::token_forcing_tests`) writes one
//! scored file per execution and the report. A scored file (`gate::ScoredSystem`) holds the
//! execution's identity, the scorer's identity, and per request the name, prompt length,
//! SHA-256 of the prompt ids (little-endian `u32`), one `score::TokenRecord` per generated
//! token, and the audits of its large-margin mismatches. `run::verdict_from_files` recomputes
//! the verdict from the three files on any host (`run::entry::token_forcing_verdict`). It
//! checks the numbers rather than trusting them: a record or an audit that contradicts itself,
//! a duplicate audit, an audit of a token that is no event, or one file in two roles (the three
//! labels must differ) is refused.
//!
//! Before a qualification, a probe (`probe`) measures on each task's reference output how far
//! the injected fault reaches, so that the fault self-test is known to have power. Its file
//! carries the SHA-256 of each output it read, and `probe::check_probe` refuses it for a
//! reference that generated anything else.
//!
//! An independent reference, such as stock exllamav3 on the same pack, plugs in at that file
//! boundary rather than as a model inside the engine: it writes the three scored files under
//! its own scorer identity, and the offline verdict evaluates them. Its audits are optional, but
//! without them a baseline event makes the verdict inconclusive and a candidate event counts as
//! unexplained.

pub(crate) mod gate;
pub(crate) mod probe;
pub(crate) mod run;
pub(crate) mod score;
pub(crate) mod stats;

mod fixture;
mod tests;
