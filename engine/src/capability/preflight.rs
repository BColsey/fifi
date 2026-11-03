//! The compatibility verdict for one model directory on this machine,
//! computed before anything is loaded.
//!
//! Qwen3.6 gets the deep path because its validators are reusable as-is:
//! option checking is `Q36Settings::new` (the exact function the factory
//! runs at start), the tokenizer verdict is `TokenizerBundle::load` with
//! the family's own pin, and the byte estimate uses
//! `settings::cache_growth_per_position` and `settings::delta_state_bytes`
//! — the authorities the session admission itself charges by. Other
//! servable families get an honest shallow path (inventory, tokenizer
//! presence, no invented estimate) until their validators are composed
//! the same way.

use serde::Serialize;
use std::path::{Path, PathBuf};

use super::{Probe, inspect::ModelInspection, probe::HardwareFacts};
use crate::memory::MemoryPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Supported, and the estimated plan fits the probed budget.
    Ready,
    /// Supported and fits, with warnings the user should read first.
    ReadyWithAdjustments,
    /// Supported, but the estimated plan exceeds the probed budget.
    ResourceConstrained,
    /// Implemented but not qualified end to end for this combination.
    Experimental,
    /// This build cannot serve this model directory.
    Unsupported,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Ready => "ready",
            Verdict::ReadyWithAdjustments => "ready with adjustments",
            Verdict::ResourceConstrained => "resource-constrained",
            Verdict::Experimental => "experimental",
            Verdict::Unsupported => "unsupported",
        }
    }

    /// Whether serving should be attempted with these settings. The CLI
    /// exit code follows this; nothing is auto-adjusted either way.
    pub fn servable(self) -> bool {
        matches!(
            self,
            Verdict::Ready | Verdict::ReadyWithAdjustments | Verdict::Experimental
        )
    }
}

/// The three independent facts behind the summary verdict. A single
/// verdict alone conflates them: a model can be *compatible* but
/// *unqualified*, or qualified but *not fitting* this machine right now.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compatibility {
    Supported,
    Unsupported,
    /// Inspection was not deep enough to decide (shallow path).
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Qualification {
    /// Measured and gated for this combination (see doc/explanation/status.md).
    Qualified,
    /// Implemented, but the quality gate for this combination has not passed.
    Experimental,
    Unverified,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceFit {
    Fits,
    DoesNotFit,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Dimensions {
    pub compatibility: Compatibility,
    pub qualification: Qualification,
    pub resource_fit: ResourceFit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
    Info,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
}

impl Check {
    fn new(name: &str, status: CheckStatus, detail: String) -> Self {
        Self {
            name: name.to_string(),
            status,
            detail,
        }
    }
}

/// The memory estimate: checkpoint **file** bytes plus closed-form cache
/// and state bytes. File bytes are not promised to equal runtime-resident
/// weight allocations — residency is each loader's decision (the CUDA
/// GGUF arm uploads the pinned file's tensors; tiered and host-offload
/// modes deliberately do not) — so the weight term is an upper-bound
/// planning figure, labeled as file bytes wherever rendered. Workspace, activations, and driver overhead are **not**
/// in `total_bytes`; on the CUDA path they enter through the runtime
/// admission charges ([`Estimate::admission_charges_bytes`], modeled on the
/// load-time ledger), and load-time admission remains the authority.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Estimate {
    pub weight_bytes: u64,
    pub kv_bytes_per_position: u64,
    pub context_limit: u32,
    pub kv_bytes: u64,
    pub fixed_state_bytes: u64,
    pub total_bytes: u64,
    /// What the estimate was compared against, in the budget's own words
    /// (for example "free VRAM of device 0, measured now").
    pub budget_kind: String,
    pub budget_free_bytes: Option<u64>,
    /// The admission ceiling the demand was compared against: the measured
    /// free bytes on the CUDA path (the margin is inside the demand), the
    /// `MemoryPolicy` share of them on the MLX path.
    pub budget_limit_bytes: Option<u64>,
    /// Whether `budget_limit_bytes` is the `MemoryPolicy` share of the free
    /// bytes (MLX) rather than the free bytes themselves (CUDA).
    pub budget_limit_is_policy_share: bool,
    /// The largest context whose estimate fits the budget, when that is
    /// smaller than the requested one. A proposal, never applied.
    pub largest_fitting_context: Option<u32>,
    /// The runtime admission demand charged after the weights on the CUDA
    /// path: cache, fixed state, prefix device tip, staged KV8 rows, part
    /// charge, spare minimum, admission margin, and the measured runtime
    /// overhead floor. Zero on the MLX path, which the runtime budgets by
    /// the memory policy share instead.
    pub admission_charges_bytes: u64,
    /// The named terms behind `admission_charges_bytes`, for the rendered
    /// breakdown. Empty when the estimate has no charge model.
    pub admission_charge_terms: Vec<(String, u64)>,
    /// The estimate's own fit call against the measured budget.
    pub fit: ResourceFit,
    /// `true` when `fit` is `Unknown` because the demand lands inside the
    /// measurement band around the budget boundary (±`ADMISSION_UNCERTAINTY_BAND_BYTES`):
    /// load-time admission decides, not the estimate.
    pub borderline: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PreflightReport {
    pub model_dir: PathBuf,
    pub tokenizer_dir: PathBuf,
    pub family: Option<String>,
    pub verdict: Verdict,
    pub dimensions: Dimensions,
    pub checks: Vec<Check>,
    pub estimate: Option<Estimate>,
    /// The single most useful next action, in one sentence.
    pub next_step: String,
}

/// The candidate settings a preflight evaluates. `None` fields take the
/// family's own resolved defaults, exactly as a config file without the
/// key would.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Candidate {
    pub family: Option<String>,
    pub context_limit: Option<u32>,
    pub kv_dtype: Option<String>,
}

/// Run a preflight. `facts` is probed by the caller so one probe can
/// serve a whole report. Inspection failures (not a directory, unreadable)
/// are the only `Err`: everything about a readable directory, however
/// wrong for serving, comes back as a verdict with reasons.
pub fn preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    candidate: &Candidate,
    facts: &HardwareFacts,
) -> Result<PreflightReport, crate::EngError> {
    let inspection = super::inspect::inspect(model_dir, tokenizer_dir)?;
    Ok(evaluate(
        model_dir,
        tokenizer_dir,
        &inspection,
        candidate,
        facts,
    ))
}

fn evaluate(
    model_dir: &Path,
    tokenizer_dir: &Path,
    inspection: &ModelInspection,
    candidate: &Candidate,
    facts: &HardwareFacts,
) -> PreflightReport {
    let mut checks = Vec::new();
    inventory_checks(inspection, &mut checks);

    let family = identify_family(inspection, candidate, &mut checks);
    let family_name = family.as_deref();
    if family_name == Some(crate::family::qwen36::factory::FAMILY) {
        // `None` when this binary has no GPU backend at all;
        // `backend_of_build`'s fallback arm must not masquerade as one.
        let backend =
            (!facts.backends.none()).then(crate::family::qwen36::factory::backend_of_build);
        return q36_preflight(
            model_dir,
            tokenizer_dir,
            inspection,
            candidate,
            facts,
            backend,
            checks,
        );
    }
    shallow_preflight(model_dir, tokenizer_dir, inspection, family, checks)
}

fn inventory_checks(inspection: &ModelInspection, checks: &mut Vec<Check>) {
    for fact in &inspection.gguf_files {
        checks.push(Check::new(
            "gguf header",
            CheckStatus::Pass,
            format!(
                "{}: {} bytes, {} tensors{}",
                fact.file,
                fact.file_bytes,
                fact.tensor_count,
                fact.architecture
                    .as_deref()
                    .map_or(String::new(), |arch| format!(", architecture {arch}")),
            ),
        ));
    }
    for error in &inspection.gguf_errors {
        checks.push(Check::new("gguf header", CheckStatus::Fail, error.clone()));
    }
    if inspection.safetensors_files > 0 {
        checks.push(Check::new(
            "safetensors shards",
            CheckStatus::Info,
            format!(
                "{} files, {} bytes (contents are the loader's check, not inspected here)",
                inspection.safetensors_files, inspection.safetensors_bytes
            ),
        ));
    }
    if inspection.weight_bytes() == 0 {
        checks.push(Check::new(
            "weights",
            CheckStatus::Fail,
            "no .gguf or .safetensors weight files in the directory".to_string(),
        ));
    }
    if inspection.tokenizer_missing.is_empty() {
        checks.push(Check::new(
            "tokenizer files",
            CheckStatus::Pass,
            "all four tokenizer files are present".to_string(),
        ));
    } else {
        checks.push(Check::new(
            "tokenizer files",
            CheckStatus::Fail,
            format!(
                "missing {} in the tokenizer directory",
                inspection.tokenizer_missing.join(", ")
            ),
        ));
    }
}

/// The servable families this build recognizes, from the one list the
/// server config validates against.
fn servable_families() -> &'static [&'static str] {
    &crate::server::config::FAMILIES
}

fn identify_family(
    inspection: &ModelInspection,
    candidate: &Candidate,
    checks: &mut Vec<Check>,
) -> Option<String> {
    if let Some(family) = &candidate.family {
        if servable_families().contains(&family.as_str()) {
            checks.push(Check::new(
                "family",
                CheckStatus::Pass,
                format!("{family} (given) is recognized by this build"),
            ));
            return Some(family.clone());
        }
        checks.push(Check::new(
            "family",
            CheckStatus::Fail,
            format!(
                "family {family} is not recognized by this build (supported: {})",
                servable_families().join(", ")
            ),
        ));
        return None;
    }
    // Inference, deliberately narrow: only a `config.json` model_type that
    // a family's spec itself accepts. Anything else asks the user to name
    // the family rather than guessing at GB-scale consequences.
    let spec = crate::family::spec::QWEN3_6_35B_A3B;
    if inspection
        .config_model_types
        .iter()
        .any(|t| spec.model_types.contains(&t.as_str()))
    {
        checks.push(Check::new(
            "family",
            CheckStatus::Pass,
            format!(
                "config.json model_type matches {}",
                crate::family::qwen36::factory::FAMILY
            ),
        ));
        return Some(crate::family::qwen36::factory::FAMILY.to_string());
    }
    checks.push(Check::new(
        "family",
        CheckStatus::Fail,
        format!(
            "the family cannot be identified from this directory; pass --family (supported: {})",
            servable_families().join(", ")
        ),
    ));
    None
}

fn any_failed(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == CheckStatus::Fail)
}

fn any_warned(checks: &[Check]) -> bool {
    checks.iter().any(|c| c.status == CheckStatus::Warn)
}

fn shallow_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    _inspection: &ModelInspection,
    family: Option<String>,
    mut checks: Vec<Check>,
) -> PreflightReport {
    let (verdict, next_step) = if any_failed(&checks) || family.is_none() {
        (
            Verdict::Unsupported,
            "Fix the failed checks above; nothing here is loaded until they pass.".to_string(),
        )
    } else {
        checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            "no byte estimate is composed for this family yet; load-time admission will check the plan and refuse with byte counts if it does not fit".to_string(),
        ));
        (
            Verdict::Experimental,
            "The directory shape is plausible but this family is not qualified end to end from preflight; serve it and read the engine start lines for the admission result.".to_string(),
        )
    };
    let dimensions = Dimensions {
        compatibility: if verdict == Verdict::Unsupported {
            Compatibility::Unsupported
        } else {
            Compatibility::Unverified
        },
        qualification: Qualification::Unverified,
        resource_fit: ResourceFit::Unknown,
    };
    PreflightReport {
        model_dir: model_dir.to_path_buf(),
        tokenizer_dir: tokenizer_dir.to_path_buf(),
        family,
        verdict,
        dimensions,
        checks,
        estimate: None,
        next_step,
    }
}

fn q36_preflight(
    model_dir: &Path,
    tokenizer_dir: &Path,
    inspection: &ModelInspection,
    candidate: &Candidate,
    facts: &HardwareFacts,
    backend: Option<crate::family::qwen36::settings::Backend>,
    mut checks: Vec<Check>,
) -> PreflightReport {
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};

    let family = Some(crate::family::qwen36::factory::FAMILY.to_string());
    let finish = |checks: Vec<Check>, estimate: Option<Estimate>, qualification: Qualification| {
        let dimensions = q36_dimensions(&checks, estimate.as_ref(), qualification);
        let (verdict, next_step) = q36_verdict(&checks, estimate.as_ref(), qualification);
        PreflightReport {
            model_dir: model_dir.to_path_buf(),
            tokenizer_dir: tokenizer_dir.to_path_buf(),
            family: family.clone(),
            verdict,
            dimensions,
            checks,
            estimate,
            next_step,
        }
    };

    let Some(backend) = backend else {
        checks.push(Check::new(
            "backend",
            CheckStatus::Fail,
            "no GPU backend is compiled into this build (build with --features native-cuda on Linux or native-mlx on Apple Silicon); this binary cannot serve this family".to_string(),
        ));
        return finish(checks, None, Qualification::Unverified);
    };
    checks.push(Check::new(
        "backend",
        CheckStatus::Pass,
        format!(
            "{} backend is compiled in{}",
            match backend {
                Backend::Cuda => "CUDA",
                Backend::Mlx => "MLX",
            },
            match (backend, facts.cuda.device_count.as_ref()) {
                (Backend::Cuda, Some(n)) if *n == 0 => ", but no CUDA device is present",
                _ => "",
            }
        ),
    ));
    if backend == Backend::Cuda && facts.cuda.device_count.as_ref().is_some_and(|n| *n > 1) {
        checks.push(Check::new(
            "devices",
            CheckStatus::Info,
            "more than one CUDA device is present; this estimate is for device 0 alone (the default). A cuda_devices plan is admitted by its own placement planner at load, with per-device budgets this check does not reproduce".to_string(),
        ));
    }
    if backend == Backend::Cuda && facts.cuda.device_count.as_ref() == Some(&0) {
        checks.last_mut().expect("backend check").status = CheckStatus::Fail;
        return finish(checks, None, Qualification::Unverified);
    }

    // The factory's own settings resolution: identical validation of the
    // candidate options, including every range and combination rule.
    let options = Q36Options {
        context_limit: candidate.context_limit,
        kv_dtype: candidate.kv_dtype.clone(),
        ..Q36Options::default()
    };
    let env = Q36Env {
        cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
    };
    let settings = match Q36Settings::new(model_dir, tokenizer_dir, Some(&options), &env, backend) {
        Ok(settings) => {
            checks.push(Check::new(
                "settings",
                CheckStatus::Pass,
                format!(
                    "context {}, kv_dtype {}, activation {} resolve under the {} rules",
                    settings.context_limit,
                    crate::family::qwen36::settings::kv_dtype_name(settings.kv),
                    crate::family::qwen36::settings::activation_name(settings.activation),
                    match backend {
                        Backend::Cuda => "CUDA",
                        Backend::Mlx => "MLX",
                    },
                ),
            ));
            settings
        }
        Err(error) => {
            checks.push(Check::new(
                "settings",
                CheckStatus::Fail,
                format!(
                    "the engine would refuse these settings: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, Qualification::Unverified);
        }
    };

    // The tokenizer verdict is the loader's: bundle load under the
    // family's pin refuses any file that differs from the tested release.
    match crate::family::qwen36::settings::tokenizer_pin()
        .and_then(|pin| crate::tokenizer::TokenizerBundle::load(tokenizer_dir, &pin))
    {
        Ok(_) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Pass,
            "all four tokenizer files match the tested release (SHA-256 pin)".to_string(),
        )),
        Err(error) => checks.push(Check::new(
            "tokenizer pin",
            CheckStatus::Fail,
            format!(
                "the tokenizer would be refused at load: {}",
                super::describe_error(&error)
            ),
        )),
    }

    // Qualification of this exact combination, from the project's own
    // gates: the EXL3 pack's quality against the GGUF engine is not yet
    // measured (doc/explanation/status.md), so it stays experimental
    // until that gate passes.
    let qualification = match settings.weights {
        crate::family::qwen36::settings::WeightFormat::Exl3 => Qualification::Experimental,
        crate::family::qwen36::settings::WeightFormat::Gguf => Qualification::Qualified,
    };

    // Format vs. directory shape: the resolved weight format must have
    // its files here. Exact checkpoint identity stays the loader's check.
    use crate::family::qwen36::settings::WeightFormat;
    match settings.weights {
        WeightFormat::Gguf if backend == Backend::Cuda => {
            checks.push(pinned_gguf_check(model_dir, inspection));
        }
        WeightFormat::Gguf if inspection.gguf_files.is_empty() => checks.push(Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is gguf but no readable .gguf file is in the directory".to_string(),
        )),
        WeightFormat::Exl3 if inspection.safetensors_files == 0 => checks.push(Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is exl3 but no .safetensors shards are in the directory".to_string(),
        )),
        _ => checks.push(Check::new(
            "weight format",
            CheckStatus::Pass,
            format!(
                "{} checkpoint files are present; exact file identity is verified by the loader",
                crate::family::qwen36::settings::weight_format_name(settings.weights)
            ),
        )),
    }

    if any_failed(&checks) {
        return finish(checks, None, qualification);
    }

    let spec = settings.spec;
    // The CUDA GGUF loader reads exactly the pinned file, so only its
    // bytes are the planning figure; other layouts use the whole
    // inventory of weight files found.
    let weight_bytes = if backend == Backend::Cuda && settings.weights == WeightFormat::Gguf {
        pinned_gguf_bytes(inspection).unwrap_or_else(|| inspection.weight_bytes())
    } else {
        inspection.weight_bytes()
    };
    let estimate = match estimate_q36(&spec, &settings, weight_bytes, backend, facts) {
        Ok(estimate) => estimate,
        Err(error) => {
            checks.push(Check::new(
                "memory estimate",
                CheckStatus::Fail,
                format!(
                    "the estimate could not be computed: {}",
                    super::describe_error(&error)
                ),
            ));
            return finish(checks, None, qualification);
        }
    };
    let demand = estimate
        .weight_bytes
        .saturating_add(estimate.admission_charges_bytes);
    match (estimate.fit, estimate.borderline) {
        (ResourceFit::Fits, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Pass,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) fits the {} budget",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::DoesNotFit, _) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Fail,
            format!(
                "estimated demand {} bytes (weights {} + runtime admission charges {}) exceed the {} budget; load-time admission is authoritative",
                demand,
                estimate.weight_bytes,
                estimate.admission_charges_bytes,
                estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, true) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated demand {} bytes lands inside the measurement band around the {} budget; load-time admission decides",
                demand, estimate.budget_kind
            ),
        )),
        (ResourceFit::Unknown, false) => checks.push(Check::new(
            "memory estimate",
            CheckStatus::Warn,
            format!(
                "estimated {} bytes; no budget could be probed ({}), so fit is unverified",
                estimate.total_bytes, estimate.budget_kind
            ),
        )),
    }
    finish(checks, Some(estimate), qualification)
}

/// The pinned CUDA checkpoint check, through the loader's own gate
/// (`check_gguf_identity`: pinned name + exact size; the full-file
/// SHA-256 stays the loader's opt-in, since hashing 20 GB is not a
/// preflight default). Only a CUDA build has the gate at all — and a
/// CUDA backend implies that build — so the non-CUDA arm below is
/// reachable only from tests that inject the backend.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn pinned_gguf_check(model_dir: &Path, _inspection: &ModelInspection) -> Check {
    use crate::family::qwen36::cuda::start::{GGUF_NAME, check_gguf_identity};
    match check_gguf_identity(&model_dir.join(GGUF_NAME), false) {
        Ok(()) => Check::new(
            "weight format",
            CheckStatus::Pass,
            format!("the pinned CUDA checkpoint {GGUF_NAME} is present at its pinned size"),
        ),
        Err(error) => Check::new(
            "weight format",
            CheckStatus::Fail,
            format!(
                "the loader would refuse this checkpoint: {}",
                super::describe_error(&error)
            ),
        ),
    }
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn pinned_gguf_check(_model_dir: &Path, inspection: &ModelInspection) -> Check {
    if inspection.gguf_files.is_empty() {
        Check::new(
            "weight format",
            CheckStatus::Fail,
            "weight_format is gguf but no readable .gguf file is in the directory".to_string(),
        )
    } else {
        Check::new(
            "weight format",
            CheckStatus::Warn,
            "the pinned-file identity check lives in the CUDA build; this binary cannot run it"
                .to_string(),
        )
    }
}

/// The pinned file's bytes from the inventory, when this build knows
/// the pinned name (same gate as `pinned_gguf_check`).
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn pinned_gguf_bytes(inspection: &ModelInspection) -> Option<u64> {
    use crate::family::qwen36::cuda::start::GGUF_NAME;
    inspection
        .gguf_files
        .iter()
        .find(|fact| fact.file == GGUF_NAME)
        .map(|fact| fact.file_bytes)
}

#[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
fn pinned_gguf_bytes(_inspection: &ModelInspection) -> Option<u64> {
    None
}

fn q36_dimensions(
    checks: &[Check],
    estimate: Option<&Estimate>,
    qualification: Qualification,
) -> Dimensions {
    // A failed memory estimate is a resource fact, not a compatibility
    // one; every other failed check here means this build cannot serve
    // the directory as asked.
    let incompatible = checks
        .iter()
        .any(|c| c.status == CheckStatus::Fail && c.name != "memory estimate");
    let resource_fit = match estimate {
        Some(estimate) => estimate.fit,
        None => ResourceFit::Unknown,
    };
    Dimensions {
        compatibility: if incompatible {
            Compatibility::Unsupported
        } else {
            Compatibility::Supported
        },
        qualification,
        resource_fit,
    }
}

fn q36_verdict(
    checks: &[Check],
    estimate: Option<&Estimate>,
    qualification: Qualification,
) -> (Verdict, String) {
    if let Some(estimate) = estimate
        && estimate.fit == ResourceFit::DoesNotFit
    {
        let next = match estimate.largest_fitting_context {
            Some(fitting) if fitting >= 2 => format!(
                "Free memory, or serve with a smaller context: the largest estimated fitting context is {fitting} (requested {}). Nothing was changed for you.",
                estimate.context_limit
            ),
            _ => "Free memory or use a smaller checkpoint: even a minimal context does not fit the probed budget.".to_string(),
        };
        return (Verdict::ResourceConstrained, next);
    }
    if any_failed(checks) {
        return (
            Verdict::Unsupported,
            "Fix the failed checks above; preflight ran the same validators the loader runs, so it would refuse the same way.".to_string(),
        );
    }
    if let Some(estimate) = estimate
        && estimate.borderline
    {
        return (
            Verdict::ReadyWithAdjustments,
            "The estimate lands inside the measurement band around the budget, so load-time admission decides: try serving, or pick a smaller context for certainty.".to_string(),
        );
    }
    if qualification == Qualification::Experimental {
        return (
            Verdict::Experimental,
            "Servable, but this combination has not passed its quality gate; treat output quality as unmeasured against the qualified path.".to_string(),
        );
    }
    if any_warned(checks) {
        return (
            Verdict::ReadyWithAdjustments,
            "Read the warnings above, then serve; load-time admission re-checks the plan."
                .to_string(),
        );
    }
    (
        Verdict::Ready,
        "Serve it: enginesrv --config <file> with this directory as model_dir (load-time admission re-checks the plan).".to_string(),
    )
}

fn estimate_q36(
    spec: &crate::family::spec::FamilySpec,
    settings: &crate::family::qwen36::settings::Q36Settings,
    weight_bytes: u64,
    backend: crate::family::qwen36::settings::Backend,
    facts: &HardwareFacts,
) -> Result<Estimate, crate::EngError> {
    use crate::family::qwen36::admission_sketch as sketch;
    use crate::family::qwen36::settings::{
        Backend, KvDtype, cache_growth_per_position, delta_state_bytes,
    };

    let per_position = cache_growth_per_position(spec, settings.kv, settings.activation)?;
    let kv_bytes = per_position
        .checked_mul(u64::from(settings.context_limit))
        .ok_or_else(|| {
            crate::EngError::MemoryPlan("preflight KV estimate overflows u64".to_string())
        })?;
    let state_bytes = delta_state_bytes(spec)?;
    let weights = weight_bytes;
    let total = weights
        .checked_add(kv_bytes)
        .and_then(|sum| sum.checked_add(state_bytes))
        .ok_or_else(|| {
            crate::EngError::MemoryPlan("preflight total estimate overflows u64".to_string())
        })?;

    let (budget_kind, free, limit_is_policy_share) = match backend {
        Backend::Cuda => (
            "free VRAM of device 0, measured now, minus the runtime admission charges below (estimate; load-time admission is authoritative)".to_string(),
            facts.cuda.first_device_free_bytes.clone(),
            false,
        ),
        Backend::Mlx => (
            "available unified memory, measured now".to_string(),
            facts.host.mem_available_bytes.clone(),
            true,
        ),
    };
    let policy = MemoryPolicy::default_share();

    // Staged KV8 rows and the device-tip prefix copy are the two runtime
    // charges the closed-form figures above do not carry; only the Q8 cache
    // dequants into a staging cache (F32 and Bf16 are read natively).
    let staging_at = |context: u32| -> Result<u64, crate::EngError> {
        if settings.kv != KvDtype::Q8 {
            return Ok(0);
        }
        let rows = sketch::fixed_staging_rows(usize::try_from(context).unwrap_or(usize::MAX));
        u64::try_from(sketch::kv8_staging_bytes(
            rows,
            spec.heads_kv as usize,
            spec.head_dim as usize,
        )?)
        .map_err(|_| crate::EngError::MemoryPlan("staging bytes exceed u64".to_string()))
    };
    let cuda_demand_at = |context: u32| -> Result<sketch::AdmissionSketch, crate::EngError> {
        let kv = per_position
            .checked_mul(u64::from(context))
            .ok_or_else(|| {
                crate::EngError::MemoryPlan("preflight KV estimate overflows u64".to_string())
            })?;
        Ok(sketch::admission_demand(
            kv,
            state_bytes,
            state_bytes,
            staging_at(context)?,
        ))
    };

    let (free_bytes, limit_bytes, charges, charge_terms, fit, borderline) = match (&free, backend) {
        (Probe::Measured(free), Backend::Cuda) => {
            let demand = cuda_demand_at(settings.context_limit)?;
            let charges = demand.total_after_weights()?;
            let slack = i128::from(*free) - i128::from(weights) - i128::from(charges);
            let band = i128::from(sketch::ADMISSION_UNCERTAINTY_BAND_BYTES);
            let (fit, borderline) = if slack >= band {
                (ResourceFit::Fits, false)
            } else if slack >= -band {
                (ResourceFit::Unknown, true)
            } else {
                (ResourceFit::DoesNotFit, false)
            };
            (
                Some(*free),
                Some(*free),
                charges,
                vec![
                    ("staged kv8 rows".to_string(), demand.staging_bytes),
                    ("part charge".to_string(), demand.part_charge_bytes),
                    ("spare minimum".to_string(), demand.spare_min_bytes),
                    ("admission margin".to_string(), demand.margin_bytes),
                    (
                        "runtime overhead floor".to_string(),
                        demand.overhead_floor_bytes,
                    ),
                ],
                fit,
                borderline,
            )
        }
        (Probe::Measured(free), Backend::Mlx) => {
            let limit = policy.limit(*free);
            let fit = if total <= limit {
                ResourceFit::Fits
            } else {
                ResourceFit::DoesNotFit
            };
            (Some(*free), Some(limit), 0, Vec::new(), fit, false)
        }
        (Probe::Unavailable(_), _) => (None, None, 0, Vec::new(), ResourceFit::Unknown, false),
    };

    // A proposal only when the requested context is not a confident fit; the
    // search charges the same paper demand the runtime ledger charges, band
    // included. Not even a minimal context fitting yields None, and the
    // verdict then says so in its own words.
    let proposal_needed = per_position > 0 && (fit == ResourceFit::DoesNotFit || borderline);
    let largest_fitting = if !proposal_needed {
        None
    } else if backend == Backend::Cuda {
        free_bytes.and_then(|free| {
            largest_fitting_cuda(free, weights, &cuda_demand_at, settings.context_limit)
        })
    } else {
        let for_kv = limit_bytes
            .unwrap_or(0)
            .saturating_sub(weights.saturating_add(state_bytes));
        let positions = for_kv / per_position;
        (positions >= 2).then_some(u32::try_from(positions).unwrap_or(u32::MAX))
    };
    Ok(Estimate {
        weight_bytes: weights,
        kv_bytes_per_position: per_position,
        context_limit: settings.context_limit,
        kv_bytes,
        fixed_state_bytes: state_bytes,
        total_bytes: total,
        budget_kind: match &free {
            Probe::Measured(_) => budget_kind,
            Probe::Unavailable(reason) => format!("{budget_kind}; unavailable: {reason}"),
        },
        budget_free_bytes: free_bytes,
        budget_limit_bytes: limit_bytes,
        budget_limit_is_policy_share: limit_is_policy_share,
        largest_fitting_context: largest_fitting,
        admission_charges_bytes: charges,
        admission_charge_terms: charge_terms,
        fit,
        borderline,
    })
}

/// The largest context (a multiple of 1024, capped at the requested context)
/// whose paper demand leaves at least the uncertainty band of slack against
/// the measured budget. `None` when not even a minimal context fits — the
/// honest answer the old division rule could not give (it proposed 0).
fn largest_fitting_cuda(
    free: u64,
    weights: u64,
    cuda_demand_at: &dyn Fn(
        u32,
    ) -> Result<
        crate::family::qwen36::admission_sketch::AdmissionSketch,
        crate::EngError,
    >,
    requested: u32,
) -> Option<u32> {
    use crate::family::qwen36::admission_sketch as sketch;
    let band = i128::from(sketch::ADMISSION_UNCERTAINTY_BAND_BYTES);
    let fits = |context: u32| -> bool {
        let Ok(demand) = cuda_demand_at(context) else {
            return false;
        };
        let Ok(charges) = demand.total_after_weights() else {
            return false;
        };
        i128::from(free) - i128::from(weights) - i128::from(charges) >= band
    };
    const MIN_CONTEXT: u32 = 1024;
    const STEP: u32 = 1024;
    if requested <= MIN_CONTEXT || !fits(MIN_CONTEXT) {
        return None;
    }
    let mut low = MIN_CONTEXT;
    let mut high = requested;
    while high - low > STEP {
        let mid = low + (high - low) / 2 / STEP * STEP;
        if mid <= low {
            break;
        }
        if fits(mid) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Some(low)
}

#[cfg(test)]
mod tests {
    use super::{
        Check, CheckStatus, Compatibility, Estimate, Qualification, ResourceFit, Verdict,
        q36_dimensions, q36_verdict,
    };

    fn estimate(total: u64, limit: Option<u64>, fitting: Option<u32>) -> Estimate {
        let fit = match limit {
            Some(limit) if total <= limit => ResourceFit::Fits,
            Some(_) => ResourceFit::DoesNotFit,
            None => ResourceFit::Unknown,
        };
        Estimate {
            weight_bytes: 100,
            kv_bytes_per_position: 10,
            context_limit: 1000,
            kv_bytes: total.saturating_sub(100),
            fixed_state_bytes: 0,
            total_bytes: total,
            budget_kind: "test budget".into(),
            budget_free_bytes: limit,
            budget_limit_bytes: limit,
            budget_limit_is_policy_share: true,
            largest_fitting_context: fitting,
            admission_charges_bytes: 0,
            admission_charge_terms: Vec::new(),
            fit,
            borderline: false,
        }
    }

    #[test]
    fn an_over_budget_estimate_is_resource_constrained_with_a_proposal() {
        let (verdict, next) = q36_verdict(
            &[],
            Some(&estimate(2000, Some(1000), Some(90))),
            Qualification::Qualified,
        );
        assert_eq!(verdict, Verdict::ResourceConstrained);
        assert!(
            next.contains("largest estimated fitting context is 90"),
            "{next}"
        );
        assert!(next.contains("Nothing was changed"), "{next}");
    }

    #[test]
    fn failures_outrank_warnings_and_warnings_downgrade_ready() {
        let check = |status| Check::new("x", status, String::new());
        assert_eq!(
            q36_verdict(&[check(CheckStatus::Pass)], None, Qualification::Qualified).0,
            Verdict::Ready
        );
        assert_eq!(
            q36_verdict(
                &[check(CheckStatus::Pass), check(CheckStatus::Warn)],
                None,
                Qualification::Qualified
            )
            .0,
            Verdict::ReadyWithAdjustments
        );
        assert_eq!(
            q36_verdict(
                &[check(CheckStatus::Pass), check(CheckStatus::Fail)],
                None,
                Qualification::Qualified
            )
            .0,
            Verdict::Unsupported
        );
    }

    fn cuda_facts(free_vram: u64) -> super::super::probe::HardwareFacts {
        use crate::capability::Probe;
        use crate::capability::probe::{CompiledBackends, CudaFacts, HardwareFacts, HostFacts};
        HardwareFacts {
            backends: CompiledBackends {
                cuda: true,
                mlx: false,
            },
            cuda: CudaFacts {
                device_count: Probe::Measured(1),
                first_device_integrated: Probe::Measured(false),
                first_device_free_bytes: Probe::Measured(free_vram),
                first_device_total_bytes: Probe::Measured(24u64 << 30),
            },
            host: HostFacts {
                mem_total_bytes: Probe::Unavailable("test".into()),
                mem_available_bytes: Probe::Unavailable("test".into()),
                cpu_threads: 1,
            },
        }
    }

    fn check_named<'a>(report: &'a super::PreflightReport, name: &str) -> &'a super::Check {
        report
            .checks
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no check named {name} in {:?}", report.checks))
    }

    #[test]
    fn a_wrong_tokenizer_fails_the_pin_check() {
        use crate::family::qwen36::settings::Backend;
        let dir = tempfile::tempdir().unwrap().keep();
        std::fs::write(dir.join("weights.safetensors"), vec![0u8; 8]).unwrap();
        for name in crate::capability::inspect::TOKENIZER_FILES {
            std::fs::write(dir.join(name), "{}").unwrap();
        }
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &super::Candidate::default(),
            &cuda_facts(24u64 << 30),
            Some(Backend::Cuda),
            Vec::new(),
        );
        assert_eq!(
            check_named(&report, "tokenizer pin").status,
            CheckStatus::Fail
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
        assert_eq!(report.dimensions.compatibility, Compatibility::Unsupported);
    }

    #[test]
    fn a_refused_setting_fails_before_any_estimate() {
        use crate::family::qwen36::settings::Backend;
        let dir = tempfile::tempdir().unwrap().keep();
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let candidate = super::Candidate {
            kv_dtype: Some("q4".to_string()),
            ..super::Candidate::default()
        };
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &candidate,
            &cuda_facts(24u64 << 30),
            Some(Backend::Cuda),
            Vec::new(),
        );
        assert_eq!(check_named(&report, "settings").status, CheckStatus::Fail);
        assert!(report.estimate.is_none());
        assert_eq!(report.dimensions.resource_fit, ResourceFit::Unknown);
    }

    #[test]
    fn no_compiled_backend_is_unsupported_not_unverified() {
        let dir = tempfile::tempdir().unwrap().keep();
        let inspection = crate::capability::inspect::inspect(&dir, &dir).unwrap();
        let report = super::q36_preflight(
            &dir,
            &dir,
            &inspection,
            &super::Candidate::default(),
            &cuda_facts(0),
            None,
            Vec::new(),
        );
        assert_eq!(report.verdict, Verdict::Unsupported);
        assert_eq!(report.dimensions.qualification, Qualification::Unverified);
    }

    #[test]
    fn dimensions_keep_resource_failure_out_of_compatibility() {
        let dims = q36_dimensions(
            &[Check::new(
                "memory estimate",
                CheckStatus::Fail,
                String::new(),
            )],
            Some(&estimate(2000, Some(1000), Some(90))),
            Qualification::Qualified,
        );
        assert_eq!(dims.compatibility, Compatibility::Supported);
        assert_eq!(dims.resource_fit, ResourceFit::DoesNotFit);
        let dims = q36_dimensions(
            &[Check::new(
                "tokenizer pin",
                CheckStatus::Fail,
                String::new(),
            )],
            None,
            Qualification::Unverified,
        );
        assert_eq!(dims.compatibility, Compatibility::Unsupported);
        assert_eq!(dims.resource_fit, ResourceFit::Unknown);
    }

    #[test]
    fn q36_estimate_uses_the_admission_authorities() {
        use crate::family::qwen36::settings::{Backend, Q36Env, Q36Settings};
        use std::path::Path;

        // Resolved exactly as the factory resolves a bare entry on CUDA.
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            None,
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("default CUDA settings resolve");
        let facts = cuda_facts(24u64 << 30);
        let est = super::estimate_q36(
            &settings.spec,
            &settings,
            20_000_000_000,
            Backend::Cuda,
            &facts,
        )
        .expect("estimate");
        // The CUDA default cache is the F32 control: 40,960 B/position,
        // and the DeltaNet state of the real spec is 65,863,680 bytes —
        // the same figures the settings admission charges by.
        assert_eq!(est.kv_bytes_per_position, 40_960);
        assert_eq!(est.fixed_state_bytes, 65_863_680);
        assert_eq!(est.context_limit, 32_768);
        assert_eq!(
            est.total_bytes,
            20_000_000_000 + 40_960 * 32_768 + 65_863_680
        );
        // 24 GiB free against the full runtime charge model is a confident
        // fit for the 32K F32 default — the case the 850-permille share rule
        // used to refuse even though load-time admission serves it.
        assert_eq!(est.fit, ResourceFit::Fits);
        assert!(!est.borderline);
        assert_eq!(est.largest_fitting_context, None);
    }

    #[test]
    fn the_estimate_matches_the_gpu_qualification_receipts() {
        use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
        use std::path::Path;

        // The 2026-11-03 qualification host: a display-attached RTX 3090,
        // probe free 22.35 GiB, the pinned 19.02 GiB Q4_K_M checkpoint.
        let free = 23_997_186_988u64;
        let weights = 20_419_565_568u64;
        let facts = cuda_facts(free);
        let at = |context: u32, kv: &str| {
            let options: Q36Options = serde_json::from_value(serde_json::json!({
                "context_limit": context,
                "kv_dtype": kv,
            }))
            .expect("options");
            let settings = Q36Settings::new(
                Path::new("m"),
                Path::new("t"),
                Some(&options),
                &Q36Env::default(),
                Backend::Cuda,
            )
            .expect("settings");
            super::estimate_q36(&settings.spec, &settings, weights, Backend::Cuda, &facts)
                .expect("estimate")
        };

        // The 32K F32 default is servable: the old rule called it
        // resource-constrained; the runtime ledger admits it.
        let est = at(32_768, "f32");
        assert_eq!(est.fit, ResourceFit::Fits);
        assert!(!est.borderline);
        assert_eq!(est.largest_fitting_context, None);

        // The shipped 262K Q8 profile on this host: the runtime refused with
        // a 223 MiB shortfall and the estimate must agree, with a real
        // proposal (never the old 0) inside the model's context bound.
        let est = at(262_144, "q8");
        assert_eq!(est.fit, ResourceFit::DoesNotFit);
        assert!(!est.borderline);
        let fitting = est.largest_fitting_context.expect("a proposal");
        assert!(fitting >= 131_072, "{fitting}");
        assert!(fitting < 262_144, "{fitting}");
        assert_eq!(fitting % 1024, 0);
        // The proposal must itself be a confident fit under the same rule.
        let proposed = at(fitting, "q8");
        assert_eq!(proposed.fit, ResourceFit::Fits);

        // An oversized 1M F32 request: refused, but with a usable proposal
        // instead of the degenerate 0 the old division rule produced.
        let est = at(1_048_576, "f32");
        assert_eq!(est.fit, ResourceFit::DoesNotFit);
        let fitting = est.largest_fitting_context.expect("a proposal");
        assert!(fitting >= 60_000, "{fitting}");

        // A free figure landing the 262K Q8 demand inside the measurement
        // band is borderline, not a claim either way.
        let demand_262k_q8 = 3_855_868_032u64;
        let facts = cuda_facts(weights + demand_262k_q8 - (64 << 20));
        let options: Q36Options = serde_json::from_value(serde_json::json!({
            "context_limit": 262_144,
            "kv_dtype": "q8",
        }))
        .expect("options");
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(&options),
            &Q36Env::default(),
            Backend::Cuda,
        )
        .expect("settings");
        let est = super::estimate_q36(&settings.spec, &settings, weights, Backend::Cuda, &facts)
            .expect("estimate");
        assert_eq!(est.fit, ResourceFit::Unknown);
        assert!(est.borderline);
        let (verdict, next) = q36_verdict(&[], Some(&est), Qualification::Qualified);
        assert_eq!(verdict, Verdict::ReadyWithAdjustments);
        assert!(next.contains("measurement band"), "{next}");
    }

    #[test]
    fn largest_fitting_search_is_none_when_nothing_fits() {
        use crate::family::qwen36::admission_sketch;
        // A paper demand proportional to context, as the real one is.
        let demand = |context: u32| {
            Ok(admission_sketch::admission_demand(
                u64::from(context) * 1024,
                0,
                0,
                0,
            ))
        };
        // Beyond the fixed charges, free leaves the band plus 10 MiB of KV
        // (10,240 positions at 1024 B each): the search proposes the largest
        // multiple of 1024 whose slack still clears the band.
        let free = admission_sketch::RUNTIME_OVERHEAD_FLOOR_BYTES
            + admission_sketch::FIXED_ADMISSION_MARGIN_BYTES as u64
            + admission_sketch::PREFILL_SPARE_MIN_BYTES as u64
            + admission_sketch::PREFILL_PART_CHARGE_BYTES as u64
            + admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES
            + 10 * 1024 * 1024;
        let got = super::largest_fitting_cuda(free, 0, &demand, 1_048_576);
        assert!(got.is_some_and(|c| c % 1024 == 0 && (9_000..=10_240).contains(&c)));
        // Weights alone beyond the budget: not even a minimal context fits.
        let got = super::largest_fitting_cuda(free, free + 1, &demand, 1_048_576);
        assert_eq!(got, None);
    }

    #[test]
    fn servable_excludes_constrained_and_unsupported() {
        assert!(Verdict::Ready.servable());
        assert!(Verdict::Experimental.servable());
        assert!(!Verdict::ResourceConstrained.servable());
        assert!(!Verdict::Unsupported.servable());
    }
}
