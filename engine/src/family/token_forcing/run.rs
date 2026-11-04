//! One qualification run end to end, and the offline verdict from the files a run wrote.

use super::gate::{
    AuditPlan, ExecutionIdentity, GateConfig, Generation, Report, ScoredSystem, audit_system,
    calibrate, evaluate, score_system,
};
use super::score::ForcedModel;
use crate::EngError;
use serde::Serialize;
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

/// The identities of the scorer and the three executions.
#[derive(Clone, Debug)]
pub(crate) struct Identities {
    pub(crate) scorer: ExecutionIdentity,
    pub(crate) reference: ExecutionIdentity,
    pub(crate) baseline: ExecutionIdentity,
    pub(crate) candidate: ExecutionIdentity,
}

/// What each execution generated, request by request in one order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Outputs<'a> {
    pub(crate) reference: &'a [Generation],
    pub(crate) baseline: &'a [Generation],
    pub(crate) candidate: &'a [Generation],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RunSettings {
    pub(crate) config: GateConfig,
    /// The tokenizer-visible vocabulary (`TokenizerBundle::vocab_size`).
    pub(crate) vocab: usize,
    /// The most events audited per execution.
    pub(crate) audit_cap: usize,
}

/// A finished run: the report and the three scored files behind it.
#[derive(Clone, Debug)]
pub(crate) struct Run {
    pub(crate) report: Report,
    pub(crate) reference: ScoredSystem,
    pub(crate) baseline: ScoredSystem,
    pub(crate) candidate: ScoredSystem,
}

/// Scores the three executions, calibrates delta, audits the baseline's and the candidate's
/// large-margin mismatches, and evaluates the gate. `models` holds the scorer and, when the
/// harness has one, a model of the baseline execution for the audits.
pub(crate) fn qualify(
    models: (&mut dyn ForcedModel, Option<&mut dyn ForcedModel>),
    identities: &Identities,
    outputs: Outputs<'_>,
    settings: RunSettings,
) -> Result<Run> {
    let (scorer, mut baseline_model) = models;
    let score = |model: &mut dyn ForcedModel, system: &ExecutionIdentity, generations| {
        score_system(
            model,
            (system, &identities.scorer),
            generations,
            settings.vocab,
        )
    };
    let reference = score(&mut *scorer, &identities.reference, outputs.reference)?;
    let mut baseline = score(&mut *scorer, &identities.baseline, outputs.baseline)?;
    let delta = calibrate(&reference, &baseline)?.delta;
    let mut candidate = score(&mut *scorer, &identities.candidate, outputs.candidate)?;
    let plan = AuditPlan {
        threshold: 2.0 * delta,
        vocab: settings.vocab,
        cap: settings.audit_cap,
    };
    for (scored, generations) in [
        (&mut baseline, outputs.baseline),
        (&mut candidate, outputs.candidate),
    ] {
        // The cast shortens the trait object's lifetime to this reborrow; `as_deref_mut` cannot,
        // because `&mut` is invariant in its target inside the `Option`.
        let baseline_audit = baseline_model
            .as_mut()
            .map(|model| &mut **model as &mut dyn ForcedModel);
        audit_system(scored, (&mut *scorer, baseline_audit), generations, plan)?;
    }
    let report = evaluate(settings.config, &reference, &baseline, &candidate)?;
    Ok(Run {
        report,
        reference,
        baseline,
        candidate,
    })
}

/// Recomputes the verdict from the scored files of a device run.
pub(crate) fn verdict_from_files(
    reference: &Path,
    baseline: &Path,
    candidate: &Path,
    config: GateConfig,
) -> Result<Report> {
    evaluate(
        config,
        &read_scored(reference)?,
        &read_scored(baseline)?,
        &read_scored(candidate)?,
    )
}

pub(crate) fn read_scored(path: &Path) -> Result<ScoredSystem> {
    let text = std::fs::read_to_string(path).map_err(|error| io_error(path, &error.to_string()))?;
    serde_json::from_str(&text).map_err(|error| io_error(path, &error.to_string()))
}

/// Writes `value` as pretty JSON to a new file; an existing file is never replaced.
pub(crate) fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    use std::io::Write as _;
    let text =
        serde_json::to_string_pretty(value).map_err(|error| io_error(path, &error.to_string()))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| io_error(path, &error.to_string()))?;
    file.write_all(text.as_bytes())
        .and_then(|()| file.write_all(b"\n"))
        .map_err(|error| io_error(path, &error.to_string()))
}

fn io_error(path: &Path, message: &str) -> EngError {
    EngError::BadRequest(format!("token forcing: {}: {message}", path.display()))
}

/// The offline verdict: recomputes a device run's verdict from its scored files on any host,
/// writes the report to a new file, and fails unless the verdict is a pass.
///
/// ```text
/// TF_REFERENCE=<RUN_DIR>/reference.json TF_BASELINE=<RUN_DIR>/baseline.json \
/// TF_CANDIDATE=<RUN_DIR>/candidate.json TF_REPORT=<OUT_DIR>/report.json \
/// cargo test --release --lib family::token_forcing::run::entry::token_forcing_verdict \
///   -- --ignored --exact --nocapture
/// ```
mod entry {
    use super::super::gate::{GateConfig, Verdict};
    use super::{verdict_from_files, write_new_json};
    use std::path::PathBuf;

    fn path(name: &str) -> PathBuf {
        std::env::var_os(name)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| panic!("{name} must name a file"))
            .into()
    }

    #[test]
    #[ignore = "reads the scored files of a device run named by TF_REFERENCE, TF_BASELINE and TF_CANDIDATE"]
    fn token_forcing_verdict() {
        let report = verdict_from_files(
            &path("TF_REFERENCE"),
            &path("TF_BASELINE"),
            &path("TF_CANDIDATE"),
            GateConfig::QUALIFICATION,
        )
        .unwrap_or_else(|error| panic!("the gate refused the files: {error:?}"));
        let out = path("TF_REPORT");
        write_new_json(&out, &report).unwrap_or_else(|error| panic!("{error:?}"));
        println!(
            "TOKEN_FORCING verdict={} report={}",
            serde_json::to_string(&report.verdict).expect("a verdict serializes"),
            out.display()
        );
        assert_eq!(report.verdict, Verdict::Pass, "see {}", out.display());
    }
}
