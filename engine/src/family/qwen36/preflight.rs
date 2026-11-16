//! The Qwen3.6 preflight rows: the loader's own settings resolution, header
//! walk and admission arithmetic rendered as decision rows, with the same
//! signature the shared [`crate::family::plan_rows::RowsSource`] shape
//! carries (the arm calls the entry directly; a const in this file stops
//! compiling the day the two drift apart). Only the rows live here; the
//! capability arm that calls this entry keeps its checks and its stored
//! estimate, so the report's figures and its rows each have one home.

use crate::EngError;
use crate::capability::Probe;
use crate::family::admission_sketch::BandVerdict;
use crate::family::plan_decision::{PlanAlternative, PlanDecision, PlanFixity};
use crate::family::plan_rows::{RowsFit, RowsRequest};
use crate::family::qwen36::admission_sketch::{cuda_demand_model, cuda_verdict_and_proposal};
use crate::family::qwen36::settings::{
    Backend, KvDtype, Q36Env, Q36Options, Q36Settings, kv_dtype_name,
};

// The arm calls this entry directly; this line stops compiling the day the
// entry and the shared RowsSource shape drift apart.
const _: Option<crate::family::plan_rows::RowsSource> = Some(preflight_rows);

/// The rows for one candidate directory: the estimate's own figures retold
/// through the same functions the capability arm calls. The row's context
/// proposal is a second CALL of the same search the estimate runs, on the
/// same demand model and the same free bytes — the value is not passed from
/// the arm because the shared request carries the inputs, never another
/// layer's results. The arm ignores the note and the fit: its estimate, not
/// the rows, states them for this family.
pub(crate) fn preflight_rows(
    request: &RowsRequest,
) -> Result<(Vec<PlanDecision>, String, RowsFit), EngError> {
    // The arm resolves the build's own backend; the request's
    // compiled-backend facts say the same thing in a real build (a build
    // with the CUDA backend compiled is the build whose EXL3 rows these
    // are) and the two can differ only under injected test facts, where
    // the facts are what lets a plain host build exercise the CUDA rows.
    let backend = if request.facts.backends.cuda {
        Backend::Cuda
    } else {
        Backend::Mlx
    };
    let options = Q36Options {
        context_limit: request.context,
        kv_dtype: request.kv_dtype.map(|name| name.to_string()),
        ..Q36Options::default()
    };
    let env = Q36Env {
        cache_limit: std::env::var("Q36_CACHE_LIMIT").ok(),
        kv_geometric: std::env::var("Q36_KV_GEOMETRIC").ok(),
        spec_step_window: std::env::var("Q36_SPEC_STEP_WINDOW").ok(),
    };
    let settings = Q36Settings::new(
        request.model_dir,
        request.tokenizer_dir,
        Some(&options),
        &env,
        backend,
    )?;
    let plan = crate::family::qwen36::exl3_plan(request.model_dir)?;
    let disk_bytes = crate::capability::inspect::inspect(request.model_dir, request.tokenizer_dir)?
        .weight_bytes();
    let (rows, fit) = rows_from_walk(
        &settings,
        &plan,
        disk_bytes,
        &request.facts.cuda.first_device_free_bytes,
    )?;
    Ok((rows, String::new(), fit))
}

/// The rows and the fit from figures already walked: everything past the
/// pack walk, so a host test with a synthetic plan runs the same
/// composition the entry hands the arm — the same demand model, the same
/// shared verdict-and-proposal call, the same renderer.
pub(crate) fn rows_from_walk(
    settings: &Q36Settings,
    plan: &crate::family::exl3_pack::PackPlan,
    disk_bytes: u64,
    free_bytes: &Probe<u64>,
) -> Result<(Vec<PlanDecision>, RowsFit), EngError> {
    let (per_position, _state_bytes, demand_at) =
        cuda_demand_model(&settings.spec, settings.kv, settings.activation)?;

    // One gate and one search, shared with the capability estimate: the
    // entry never re-implements the rule, it calls the same function with
    // the same inputs.
    let proposal = match free_bytes {
        Probe::Measured(free) => Some(cuda_verdict_and_proposal(
            *free,
            plan.device_bytes,
            &demand_at,
            per_position,
            settings.context_limit,
        )?),
        // No probed free bytes: no verdict, no proposal — the arm's
        // estimate states the same as `Unknown`.
        Probe::Unavailable(_) => None,
    };
    let (fit, largest_row) = match proposal {
        None => (RowsFit::Unknown, None),
        Some((BandVerdict::Fits, largest)) => (RowsFit::Fits, largest),
        Some((_, Some(largest))) => (
            RowsFit::Constrained {
                requested: settings.context_limit,
                largest,
            },
            Some(largest),
        ),
        Some((BandVerdict::Borderline | BandVerdict::DoesNotFit, None)) => (RowsFit::Unknown, None),
    };
    Ok((
        render_rows(
            plan,
            disk_bytes,
            settings.context_limit,
            largest_row,
            settings.kv,
            per_position,
        ),
        fit,
    ))
}

/// The EXL3 decision rows: what the loader's own header walk found in the
/// pack, and the context the same paper charges admit. The walk already ran
/// for the weight figure; these rows only retell it, so the record and the
/// estimate cannot disagree.
fn render_rows(
    plan: &crate::family::exl3_pack::PackPlan,
    disk_bytes: u64,
    context_limit: u32,
    largest_fitting: Option<u32>,
    kv: KvDtype,
    per_position: u64,
) -> Vec<PlanDecision> {
    // Rates are keyed by twice the bits per weight; render them as bpw.
    let rates = if plan.rates.is_empty() {
        "none".to_string()
    } else {
        plan.rates
            .iter()
            .map(|(twice_bits, count)| format!("{:.1} bpw x {count}", f64::from(*twice_bits) / 2.0))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let weights_row = PlanDecision {
        name: "weight figure".to_string(),
        chosen: format!(
            "{} B device bytes from the pack walk ({} quantized matrices, rates {rates}, \
             codebook {}); {} B on disk",
            plan.device_bytes, plan.matrices, plan.codebook, disk_bytes
        ),
        alternatives: vec![PlanAlternative {
            option: format!("the {disk_bytes} B the files occupy on disk"),
            verdict: "refused".to_string(),
            reason: format!(
                "the load widens scales and plain vectors to F32, so the walk's {} B is the \
                 figure the device holds",
                plan.device_bytes
            ),
        }],
        fixed_at: PlanFixity::Load,
    };
    let mut context_alternatives = Vec::new();
    if let Some(largest) = largest_fitting {
        context_alternatives.push(PlanAlternative {
            option: format!("{largest} positions"),
            verdict: "fits".to_string(),
            reason: "largest context the same paper charges admit; a proposal, never applied"
                .to_string(),
        });
    }
    let context_row = PlanDecision {
        name: "context".to_string(),
        chosen: format!("{context_limit} positions"),
        alternatives: context_alternatives,
        fixed_at: PlanFixity::Start,
    };
    let kv_row = PlanDecision {
        name: "kv cache".to_string(),
        chosen: format!(
            "{kv} at {per_position} B per position",
            kv = kv_dtype_name(kv)
        ),
        alternatives: Vec::new(),
        fixed_at: PlanFixity::Start,
    };
    vec![weights_row, context_row, kv_row]
}

#[cfg(test)]
mod tests {
    use super::render_rows;
    use crate::family::exl3_pack::PackPlan;
    use crate::family::plan_decision::PlanFixity;
    use crate::family::qwen36::settings::KvDtype;
    use std::collections::BTreeMap;

    #[test]
    fn q36_exl3_decisions_retell_the_pack_walk_and_the_context_proposal() {
        let plan = PackPlan {
            device_bytes: 21_000_000_000,
            expert_bytes: 20_000_000_000,
            matrices: 1_976,
            rates: BTreeMap::from([(10u8, 976), (12u8, 1_000)]),
            codebook: "Mcg",
        };
        // An over-budget candidate carries the proposal the rows must
        // retell; every text is pinned in full.
        let decisions = render_rows(&plan, 20_000_000_000, 1000, Some(90), KvDtype::F32, 10);

        let weights = decisions
            .iter()
            .find(|decision| decision.name == "weight figure")
            .expect("weight figure row");
        assert_eq!(
            weights.chosen,
            "21000000000 B device bytes from the pack walk (1976 quantized matrices, rates 5.0 bpw x 976, 6.0 bpw x 1000, codebook Mcg); 20000000000 B on disk"
        );
        assert_eq!(weights.fixed_at, PlanFixity::Load);
        assert_eq!(weights.alternatives.len(), 1);
        assert_eq!(
            weights.alternatives[0].option,
            "the 20000000000 B the files occupy on disk"
        );
        assert_eq!(weights.alternatives[0].verdict, "refused");
        assert_eq!(
            weights.alternatives[0].reason,
            "the load widens scales and plain vectors to F32, so the walk's 21000000000 B is the figure the device holds"
        );

        let context = decisions
            .iter()
            .find(|decision| decision.name == "context")
            .expect("context row");
        assert_eq!(context.chosen, "1000 positions");
        assert_eq!(context.fixed_at, PlanFixity::Start);
        assert_eq!(context.alternatives.len(), 1);
        assert_eq!(context.alternatives[0].option, "90 positions");
        assert_eq!(context.alternatives[0].verdict, "fits");
        assert_eq!(
            context.alternatives[0].reason,
            "largest context the same paper charges admit; a proposal, never applied"
        );

        let kv = decisions
            .iter()
            .find(|decision| decision.name == "kv cache")
            .expect("kv cache row");
        assert_eq!(kv.chosen, "f32 at 10 B per position");
        assert!(kv.alternatives.is_empty());
        assert_eq!(kv.fixed_at, PlanFixity::Start);

        // A fitting candidate proposes nothing, so the context row has no
        // alternative.
        let decisions = render_rows(&plan, 20_000_000_000, 1000, None, KvDtype::Q8, 10);
        let context = decisions
            .iter()
            .find(|decision| decision.name == "context")
            .expect("context row");
        assert!(context.alternatives.is_empty());
        let kv = decisions
            .iter()
            .find(|decision| decision.name == "kv cache")
            .expect("kv cache row");
        assert_eq!(kv.chosen, "q8 at 10 B per position");
    }
}
