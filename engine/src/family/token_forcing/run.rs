//! A qualification run: the reference and the baseline scored and calibrated once, each
//! candidate judged against them, and the offline verdict from the files a run wrote.

use super::gate::{
    AuditPlan, ExecutionIdentity, GateConfig, Generation, Report, ScoredSystem, audit_system,
    calibrate, evaluate, score_system,
};
use super::score::ForcedModel;
use crate::EngError;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

type Result<T> = std::result::Result<T, EngError>;

/// The identities of the scorer and of the two executions every candidate is compared with.
#[derive(Clone, Debug)]
pub(crate) struct Identities {
    pub(crate) scorer: ExecutionIdentity,
    pub(crate) reference: ExecutionIdentity,
    pub(crate) baseline: ExecutionIdentity,
}

/// What the reference and the baseline generated, request by request in one order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Outputs<'a> {
    pub(crate) reference: &'a [Generation],
    pub(crate) baseline: &'a [Generation],
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RunSettings {
    pub(crate) config: GateConfig,
    /// The tokenizer-visible vocabulary (`TokenizerBundle::vocab_size`).
    pub(crate) vocab: usize,
    /// The most baseline events audited.
    pub(crate) audit_cap: usize,
}

/// The scorer and, when the harness has one, a model of the baseline execution for the audits.
pub(crate) type Models<'m> = (&'m mut dyn ForcedModel, Option<&'m mut dyn ForcedModel>);

/// One execution under test.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Candidate<'a> {
    pub(crate) identity: &'a ExecutionIdentity,
    pub(crate) generations: &'a [Generation],
    /// The most events audited. Events past it count as unexplained.
    pub(crate) audit_cap: usize,
}

/// The reference and the baseline of a run, scored, calibrated and audited once and shared by
/// every candidate judged against them.
#[derive(Clone, Debug)]
pub(crate) struct Comparison {
    pub(crate) reference: ScoredSystem,
    pub(crate) baseline: ScoredSystem,
    config: GateConfig,
    vocab: usize,
    /// Twice the calibrated delta: the margin above which a mismatch is audited.
    threshold: f64,
}

/// One candidate's report and its scored file.
#[derive(Clone, Debug)]
pub(crate) struct Run {
    pub(crate) report: Report,
    pub(crate) candidate: ScoredSystem,
}

impl Comparison {
    /// Scores the reference and the baseline, calibrates delta, and audits the baseline's
    /// large-margin mismatches.
    pub(crate) fn prepare(
        models: Models<'_>,
        identities: &Identities,
        outputs: Outputs<'_>,
        settings: RunSettings,
    ) -> Result<Self> {
        let (scorer, baseline_model) = models;
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
        let threshold = 2.0 * calibrate(&reference, &baseline)?.delta;
        let plan = AuditPlan {
            threshold,
            vocab: settings.vocab,
            cap: settings.audit_cap,
        };
        audit_system(
            &mut baseline,
            (scorer, baseline_model),
            outputs.baseline,
            plan,
        )?;
        Ok(Self {
            reference,
            baseline,
            config: settings.config,
            vocab: settings.vocab,
            threshold,
        })
    }

    /// Scores and audits one candidate and evaluates the gate.
    pub(crate) fn judge(&self, models: Models<'_>, candidate: Candidate<'_>) -> Result<Run> {
        let (scorer, baseline_model) = models;
        let mut scored = score_system(
            &mut *scorer,
            (candidate.identity, &self.reference.scorer),
            candidate.generations,
            self.vocab,
        )?;
        let plan = AuditPlan {
            threshold: self.threshold,
            vocab: self.vocab,
            cap: candidate.audit_cap,
        };
        audit_system(
            &mut scored,
            (scorer, baseline_model),
            candidate.generations,
            plan,
        )?;
        let report = evaluate(self.config, &self.reference, &self.baseline, &scored)?;
        Ok(Run {
            report,
            candidate: scored,
        })
    }
}

/// Recomputes the verdict from the scored files of a device run.
/// The report names each file by the SHA-256 of its bytes.
pub(crate) fn verdict_from_files(
    reference: &Path,
    baseline: &Path,
    candidate: &Path,
    config: GateConfig,
) -> Result<Report> {
    let (reference, reference_sha256) = read_scored(reference)?;
    let (baseline, baseline_sha256) = read_scored(baseline)?;
    let (candidate, candidate_sha256) = read_scored(candidate)?;
    let mut report = evaluate(config, &reference, &baseline, &candidate)?;
    report.inputs = input_digests([reference_sha256, baseline_sha256, candidate_sha256]);
    Ok(report)
}

/// The `Report::inputs` of three scored files, given in the order reference, baseline,
/// candidate.
pub(crate) fn input_digests(digests: [String; 3]) -> BTreeMap<String, String> {
    ["reference", "baseline", "candidate"]
        .into_iter()
        .map(str::to_owned)
        .zip(digests)
        .collect()
}

/// A scored file and the SHA-256 of its bytes.
pub(crate) fn read_scored(path: &Path) -> Result<(ScoredSystem, String)> {
    let bytes = std::fs::read(path).map_err(|error| io_error(path, &error.to_string()))?;
    let scored =
        serde_json::from_slice(&bytes).map_err(|error| io_error(path, &error.to_string()))?;
    Ok((scored, format!("{:x}", Sha256::digest(&bytes))))
}

/// Writes `value` as pretty JSON to a new file, and returns the SHA-256 of the bytes written;
/// an existing file is never replaced.
pub(crate) fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<String> {
    use std::io::Write as _;
    let mut text =
        serde_json::to_string_pretty(value).map_err(|error| io_error(path, &error.to_string()))?;
    text.push('\n');
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| io_error(path, &error.to_string()))?;
    file.write_all(text.as_bytes())
        .map_err(|error| io_error(path, &error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(text.as_bytes())))
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
