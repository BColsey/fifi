//! The paper version of the fixed CUDA admission ledger.
//!
//! `enginesrv check` cannot run the real CUDA admission: that path measures
//! free VRAM in the serving process *after* the weights are resident and
//! charges its budget from the device. This module is the pure arithmetic of
//! the same ledger, shared by both sides so they cannot drift apart:
//!
//! * the policy constants and the staging-row rule here are canonical —
//!   `cuda::prefill_driver` and `cuda::prefill_plan` alias them; the neutral
//!   half (search step and wrapper, staging arithmetic, sketch type, the
//!   kernel-tied part charge) lives in `family::admission_sketch`, and
//!   `prefill_plan` const-asserts the part charge equals the kernel cap;
//! * the terms a paper estimate cannot know exactly (the prefill chunk arena,
//!   the decode arena, eager module workspace, and the serving process's own
//!   CUDA context beyond a fresh probe) collapse into
//!   [`RUNTIME_OVERHEAD_FLOOR_BYTES`], a measured floor with provenance, and
//!   the boundary is reported through [`ADMISSION_UNCERTAINTY_BAND_BYTES`]
//!   instead of a precise claim;
//! * load-time admission stays the authority: within the band an estimate is
//!   borderline, not a verdict either way.

// The neutral search step, sketch type, kernel-tied part charge, and the
// fixed-CUDA admission policy constants live in the family home; private use
// only, so the export surface stays alias-free.
use crate::family::admission_sketch::{
    ADMISSION_UNCERTAINTY_BAND_BYTES, AdmissionSketch, FIXED_ADMISSION_MARGIN_BYTES,
    PREFILL_PART_CHARGE_BYTES, PREFILL_SPARE_MIN_BYTES,
};

/// The full-stage tier bound: the maximum rows one default stage carries and
/// the boundary above which [`fixed_staging_rows`] shrinks to the 32K tail —
/// a staging tier bound, not a context cap. Canonical value.
pub(crate) const FULL_STAGE_TIER_ROWS: usize = 131_072;

/// The staged KV8 rows a context of `context_limit` positions pins at boot:
/// the whole context below the tier, the 32K tail above it. Matches
/// `cuda::prefill_driver::fixed_staging_rows` by construction (it delegates
/// here). Pinned by the driver tests: 262144 -> 32768, 131072 -> 131072.
pub(crate) fn fixed_staging_rows(context_limit: usize) -> usize {
    if context_limit <= FULL_STAGE_TIER_ROWS {
        context_limit
    } else {
        32_768
    }
}

/// What an estimate adds on top of the exact terms because it runs outside the
/// serving process: the prefill chunk arena (~58-61 MiB at the 512/1024-row
/// tiers), the decode arena (~1.1 MiB), eager module workspace, and the
/// serving process's own CUDA context beyond a fresh probe. Measured on the
/// 24 GiB RTX 3090 host of the 2026-11-03 qualification: a server holding the
/// 19.0 GiB Q4_K_M checkpoint showed ~275 MiB less free-after-weights than a
/// fresh probe-context process, plus ~555 MiB against `nvidia-smi` free
/// (n=1). Charged as one floor, never as a precise term.
pub(crate) const RUNTIME_OVERHEAD_FLOOR_BYTES: u64 =
    crate::family::mem::DEVICE.runtime_overhead_floor;

/// Compose the sketch from the exact state terms. `kv_bytes` is the whole
/// cache at the requested context; `fixed_state_bytes` the model's fixed
/// DeltaNet state; `prefix_tip_bytes` the device-tip prefix copy the CUDA arm
/// charges on top (the same `delta_state_bytes` figure when that arm is the
/// default); `staging_bytes` zero unless the KV cache is Q8.
pub(crate) fn admission_demand(
    kv_bytes: u64,
    fixed_state_bytes: u64,
    prefix_tip_bytes: u64,
    staging_bytes: u64,
) -> AdmissionSketch {
    AdmissionSketch {
        state_bytes: kv_bytes
            .checked_add(fixed_state_bytes)
            .and_then(|sum| sum.checked_add(prefix_tip_bytes))
            .unwrap_or(u64::MAX),
        staging_bytes,
        part_charge_bytes: PREFILL_PART_CHARGE_BYTES as u64,
        spare_min_bytes: PREFILL_SPARE_MIN_BYTES as u64,
        margin_bytes: FIXED_ADMISSION_MARGIN_BYTES as u64,
        overhead_floor_bytes: RUNTIME_OVERHEAD_FLOOR_BYTES,
    }
}

/// The KV bytes one position costs on the CUDA arm: the fp16 control cache for `f32` (half the
/// F32 figure of the spec, two bytes per value) and the int8 form with fp16 group scales for
/// `q8`. These are `KvPool::bytes_per_token` of the two `KvMode`s, 20,480 and 10,880 for
/// `QWEN3_6_35B_A3B`; the loader pins both in its start tests.
pub(crate) fn cuda_kv_per_position(
    spec: &crate::family::spec::FamilySpec,
    kv: crate::family::qwen::qwen36::settings::KvDtype,
) -> Result<u64, crate::EngError> {
    use crate::family::qwen::qwen36::settings::{Activation, KvDtype, cache_growth_per_position};

    match kv {
        KvDtype::Q8 => cache_growth_per_position(spec, KvDtype::Q8, Activation::Bf16),
        KvDtype::F32 | KvDtype::Bf16 => {
            cache_growth_per_position(spec, KvDtype::F32, Activation::Bf16).map(|bytes| bytes / 2)
        }
    }
}

/// The CUDA admission demand as a function of the candidate context, on
/// the paper math above. The per-position cache rate and the fixed state
/// come back beside the closure because every caller needs the figures as
/// well as the demand; the preflight estimate and the preflight rows both
/// call this one builder, so the model is never re-implemented at a call
/// site.
#[allow(
    clippy::type_complexity,
    reason = "the builder hands back the figures and the closure together; an impl-Trait closure has no nameable type to factor out"
)]
pub(crate) fn cuda_demand_model(
    spec: &crate::family::spec::FamilySpec,
    kv: crate::family::qwen::qwen36::settings::KvDtype,
    activation: crate::family::qwen::qwen36::settings::Activation,
) -> Result<
    (
        u64,
        u64,
        impl Fn(u32) -> Result<AdmissionSketch, crate::EngError> + use<'_>,
    ),
    crate::EngError,
> {
    let per_position =
        crate::family::qwen::qwen36::settings::cache_growth_per_position(spec, kv, activation)?;
    demand_model(spec, kv, per_position)
}

/// [`cuda_demand_model`] with the exact CUDA cache rate ([`cuda_kv_per_position`]): what the card
/// start charges for the cache form it would build.
#[allow(
    clippy::type_complexity,
    reason = "the builder hands back the figures and the closure together; an impl-Trait closure has no nameable type to factor out"
)]
pub(crate) fn card_demand_model(
    spec: &crate::family::spec::FamilySpec,
    kv: crate::family::qwen::qwen36::settings::KvDtype,
) -> Result<
    (
        u64,
        u64,
        impl Fn(u32) -> Result<AdmissionSketch, crate::EngError> + use<'_>,
    ),
    crate::EngError,
> {
    demand_model(spec, kv, cuda_kv_per_position(spec, kv)?)
}

#[allow(
    clippy::type_complexity,
    reason = "the builder hands back the figures and the closure together; an impl-Trait closure has no nameable type to factor out"
)]
fn demand_model(
    spec: &crate::family::spec::FamilySpec,
    kv: crate::family::qwen::qwen36::settings::KvDtype,
    per_position: u64,
) -> Result<
    (
        u64,
        u64,
        impl Fn(u32) -> Result<AdmissionSketch, crate::EngError> + use<'_>,
    ),
    crate::EngError,
> {
    use crate::family::admission_sketch::kv8_staging_bytes;
    use crate::family::qwen::qwen36::settings::{KvDtype, delta_state_bytes};

    let state_bytes = delta_state_bytes(spec)?;
    // Staged KV8 rows and the device-tip prefix copy are the two runtime
    // charges the closed-form figures above do not carry; only the Q8 cache
    // dequants into a staging cache (F32 and Bf16 are read natively).
    let staging_at = move |context: u32| -> Result<u64, crate::EngError> {
        if kv != KvDtype::Q8 {
            return Ok(0);
        }
        let rows = fixed_staging_rows(usize::try_from(context).unwrap_or(usize::MAX));
        u64::try_from(kv8_staging_bytes(
            rows,
            spec.heads_kv as usize,
            spec.head_dim as usize,
        )?)
        .map_err(|_| crate::EngError::MemoryPlan("staging bytes exceed u64".to_string()))
    };
    let demand_at = move |context: u32| -> Result<AdmissionSketch, crate::EngError> {
        let kv_bytes = per_position
            .checked_mul(u64::from(context))
            .ok_or_else(|| {
                crate::EngError::MemoryPlan("preflight KV estimate overflows u64".to_string())
            })?;
        Ok(admission_demand(
            kv_bytes,
            state_bytes,
            state_bytes,
            staging_at(context)?,
        ))
    };
    Ok((per_position, state_bytes, demand_at))
}

/// Whether `context` leaves at least the uncertainty band of slack against the measured budget.
fn admits_cuda(
    free: u64,
    weights: u64,
    cuda_demand_at: &dyn Fn(u32) -> Result<AdmissionSketch, crate::EngError>,
    context: u32,
) -> bool {
    let Ok(demand) = cuda_demand_at(context) else {
        return false;
    };
    let Ok(charges) = demand.total_after_weights() else {
        return false;
    };
    i128::from(free) - i128::from(weights) - i128::from(charges)
        >= i128::from(ADMISSION_UNCERTAINTY_BAND_BYTES)
}

/// The largest context (a multiple of 1024, capped at the requested context)
/// whose paper demand leaves at least the uncertainty band of slack against
/// the measured budget. `None` when not even a minimal context fits — the
/// honest answer the old division rule could not give (it proposed 0).
pub(crate) fn largest_fitting_cuda(
    free: u64,
    weights: u64,
    cuda_demand_at: &dyn Fn(u32) -> Result<AdmissionSketch, crate::EngError>,
    requested: u32,
) -> Option<u32> {
    let fits = |context: u32| admits_cuda(free, weights, cuda_demand_at, context);
    crate::family::admission_sketch::largest_fitting_context(requested, &fits)
}

/// The model's maximum context in positions: the ceiling of a card-chosen context.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) const MODEL_MAX_CONTEXT: u32 =
    crate::family::qwen::qwen36::settings::MLX_DEFAULT_CONTEXT_LIMIT;

/// A card-chosen start: the cache form and the context it serves.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) type CudaFit =
    crate::family::card_fit::CardFit<crate::family::qwen::qwen36::settings::KvDtype>;

/// The context up to which the fp16 cache is kept over the int8 cache: the owner's 128K prompt
/// tier. The fp16 cache is the exact control and prefills faster; the int8 cache holds almost
/// twice the positions, so it is chosen only when fp16 cannot reach this context.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) const FP16_SUFFICIENT_CONTEXT: u32 = 131_072;

/// The cache form and context a card carries, from the free bytes before the weights are
/// uploaded and the weight bytes the load pins. `kv` and `context` name what the config fixed;
/// an absent one is chosen by the shared card-fit rule (`family::card_fit`), fed with this
/// family's charges: the cache at the candidate context, the DeltaNet state, the staged int8
/// rows and the admission margins, never less than `load_need`, the cache, work and headroom
/// the load itself admits. fp16 is preferred; the ceiling is the named context, else
/// [`MODEL_MAX_CONTEXT`].
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn choose_cuda_fit(
    spec: &crate::family::spec::FamilySpec,
    free: u64,
    weights: u64,
    int8_usable: bool,
    kv: Option<crate::family::qwen::qwen36::settings::KvDtype>,
    context: Option<u32>,
    load_need: &dyn Fn(crate::family::qwen::qwen36::settings::KvDtype, u32) -> Option<u64>,
) -> Result<Option<CudaFit>, crate::EngError> {
    use crate::family::card_fit::{CacheForm, CardBytes, ContextLimits, plan_card_fit};
    use crate::family::qwen::qwen36::settings::KvDtype;

    let (_, _, fp16_demand) = card_demand_model(spec, KvDtype::F32)?;
    let (_, _, int8_demand) = card_demand_model(spec, KvDtype::Q8)?;
    // What the load's own admission asks for beside the weights (the cache, the work and the
    // headroom it keeps) is a floor under the sketch's sum: a plan that passes the sketch and
    // fails the load would send the start down to the tiered rung.
    let fp16 = |context: u32| {
        let sketch = fp16_demand(context).ok()?.total_after_weights().ok()?;
        Some(sketch.max(load_need(KvDtype::F32, context)?))
    };
    let int8 = |context: u32| {
        let sketch = int8_demand(context).ok()?.total_after_weights().ok()?;
        Some(sketch.max(load_need(KvDtype::Q8, context)?))
    };
    let candidates = [
        CacheForm {
            form: KvDtype::F32,
            charge: &fp16,
        },
        CacheForm {
            form: KvDtype::Q8,
            charge: &int8,
        },
    ];
    let named: Vec<CacheForm<'_, KvDtype>> = candidates
        .into_iter()
        .filter(|candidate| kv.is_none_or(|fixed| fixed == candidate.form))
        // The int8 cache runs through the chunk prefill arena; a step that cannot use the arena
        // (the chunked prefill that gives int8 its arena is off) refuses it at the prefill admission. A start that left the form to
        // the card only plans a form the step runs; a named form is the operator's. Which forms
        // are eligible is decided here; how long a context each holds is the fit's.
        .filter(|candidate| candidate.form != KvDtype::Q8 || int8_usable || kv == Some(KvDtype::Q8))
        .collect();
    Ok(plan_card_fit(
        CardBytes {
            free,
            weights,
            band: ADMISSION_UNCERTAINTY_BAND_BYTES,
        },
        &named,
        ContextLimits {
            ceiling: context.unwrap_or(MODEL_MAX_CONTEXT),
            // A named context is the target: the first form that reaches it serves it, and
            // when none does the longest context any form holds is served.
            sufficient: context.unwrap_or(FP16_SUFFICIENT_CONTEXT),
        },
    ))
}

/// The configured context's verdict against the measured budget, and the
/// largest fitting context when a proposal is worth one: the single gate
/// and search behind both the capability estimate and the preflight rows.
/// The demand at the configured context is evaluated here and (as a second
/// call of the same pure closure) by callers that compose its charge terms
/// into a receipt.
pub(crate) fn cuda_verdict_and_proposal(
    free: u64,
    weights: u64,
    demand_at: &dyn Fn(u32) -> Result<AdmissionSketch, crate::EngError>,
    per_position: u64,
    context_limit: u32,
) -> Result<(crate::family::admission_sketch::BandVerdict, Option<u32>), crate::EngError> {
    let charges = demand_at(context_limit)?.total_after_weights()?;
    let slack = i128::from(free) - i128::from(weights) - i128::from(charges);
    let verdict = crate::family::admission_sketch::band_verdict(
        slack,
        i128::from(ADMISSION_UNCERTAINTY_BAND_BYTES),
    );
    let largest = if crate::family::admission_sketch::proposal_needed(per_position, verdict) {
        largest_fitting_cuda(free, weights, demand_at, context_limit)
    } else {
        None
    };
    Ok((verdict, largest))
}

#[cfg(test)]
mod tests {
    /// c0 goldens (P4): the charge terms of `admission_demand` and its checked
    /// total as literal cells. `state_bytes` folds kv + fixed + prefix tip.
    #[test]
    fn admission_demand_golden_cells() {
        let sketch = admission_demand(1 << 20, 2 << 20, 3 << 20, 4 << 20);
        assert_eq!(sketch.state_bytes, 6 << 20);
        assert_eq!(sketch.staging_bytes, 4 << 20);
        assert_eq!(sketch.part_charge_bytes, PREFILL_PART_CHARGE_BYTES as u64);
        assert_eq!(sketch.spare_min_bytes, PREFILL_SPARE_MIN_BYTES as u64);
        assert_eq!(sketch.margin_bytes, FIXED_ADMISSION_MARGIN_BYTES as u64);
        assert_eq!(sketch.overhead_floor_bytes, RUNTIME_OVERHEAD_FLOOR_BYTES);
        let total = sketch.total_after_weights().expect("fits u64");
        assert_eq!(
            total,
            (6u64 << 20)
                + (4u64 << 20)
                + PREFILL_PART_CHARGE_BYTES as u64
                + PREFILL_SPARE_MIN_BYTES as u64
                + FIXED_ADMISSION_MARGIN_BYTES as u64
                + RUNTIME_OVERHEAD_FLOOR_BYTES
        );
        // The composition saturates its state instead of panicking, and the
        // checked total then REPORTS the overflow rather than wrapping.
        let huge = admission_demand(u64::MAX, u64::MAX, u64::MAX, 0);
        assert_eq!(huge.state_bytes, u64::MAX);
        assert!(huge.total_after_weights().is_err());
    }

    use super::largest_fitting_cuda;

    /// c0 goldens (P4): `largest_fitting_cuda` on a linear charge model. The
    /// demand carries the four canonical constants (part 64 MiB, spare 64 MiB,
    /// margin 256 MiB, floor 384 MiB, so 768 MiB fixed) plus 1 MiB per lattice
    /// step, and the admit rule is `free - weights - charges >= band`
    /// (band 128 MiB), all in i128. Free values sit exactly on the boundary of
    /// a chosen step count and one byte/one step either side.
    #[test]
    fn largest_fitting_cuda_golden_grid() {
        const MIB: u64 = 1 << 20;
        let demand = |context: u32| {
            Ok(
                crate::family::qwen::qwen36::admission_sketch::admission_demand(
                    (u64::from(context) / 1024) * MIB,
                    0,
                    0,
                    0,
                ),
            )
        };
        let band = crate::family::qwen::qwen36::admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES;
        let fixed = 768 * MIB + band; // part+spare+margin+floor+band
        // free(fitting steps) = fixed + steps*MiB. Boundary at 97 steps.
        let at = |steps: u64| fixed + steps * MIB;
        let grid: [(u64, u32, Option<u32>); 8] = [
            (at(96), 100_000, Some(98_304)),     // exactly 96 admitting steps
            (at(97) - 1, 100_000, Some(98_304)), // one byte under 97 steps: still 96
            (at(97), 100_000, Some(98_304)),     // 97 steps strand at 98_304 (off-lattice request)
            (at(99), 100_000, Some(98_304)),     // 99 steps strand too: the request is off-lattice
            (at(128), 131_072, Some(130_048)),   // 128 steps = 131_072 == request: never asked
            (at(128), 262_144, Some(131_072)),   // 128 steps, room to return it
            (at(0) - 1, 2048, None),             // under even the first probe
            (at(1), 1024, None),                 // request at the step floor
        ];
        for (free, requested, expected) in grid {
            assert_eq!(
                largest_fitting_cuda(free, 0, &demand, requested),
                expected,
                "free={free} requested={requested}"
            );
        }
    }

    #[test]
    fn largest_fitting_search_is_none_when_nothing_fits() {
        use crate::family::admission_sketch::PREFILL_PART_CHARGE_BYTES;
        use crate::family::qwen::qwen36::admission_sketch;
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
            + PREFILL_PART_CHARGE_BYTES as u64
            + admission_sketch::ADMISSION_UNCERTAINTY_BAND_BYTES
            + 10 * 1024 * 1024;
        let got = largest_fitting_cuda(free, 0, &demand, 1_048_576);
        assert!(got.is_some_and(|c| c % 1024 == 0 && (9_000..=10_240).contains(&c)));
        // Weights alone beyond the budget: not even a minimal context fits.
        let got = largest_fitting_cuda(free, free + 1, &demand, 1_048_576);
        assert_eq!(got, None);
    }

    use super::*;

    #[test]
    fn staging_rows_follow_the_tier_rule() {
        assert_eq!(fixed_staging_rows(32_768), 32_768);
        assert_eq!(fixed_staging_rows(131_072), 131_072);
        assert_eq!(fixed_staging_rows(131_073), 32_768);
        assert_eq!(fixed_staging_rows(262_144), 32_768);
    }

    #[test]
    fn the_sketch_sums_its_terms_checked() {
        let sketch = admission_demand(1_000, 65_863_680, 65_863_680, 67_108_864);
        let total = sketch.total_after_weights().unwrap();
        assert_eq!(
            total,
            1_000
                + 65_863_680
                + 65_863_680
                + 67_108_864
                + PREFILL_PART_CHARGE_BYTES as u64
                + PREFILL_SPARE_MIN_BYTES as u64
                + FIXED_ADMISSION_MARGIN_BYTES as u64
                + RUNTIME_OVERHEAD_FLOOR_BYTES
        );
        assert!(
            admission_demand(u64::MAX, 1, 0, 0)
                .total_after_weights()
                .is_err()
        );
    }

    use crate::family::QWEN3_6_35B_A3B;
    use crate::family::qwen::qwen36::settings::KvDtype;

    #[test]
    fn the_cuda_cache_rates_are_the_two_kv_pool_layouts() {
        // KvPool::bytes_per_token of Fp16 and Int8G32, pinned in the start tests.
        assert_eq!(
            cuda_kv_per_position(&QWEN3_6_35B_A3B, KvDtype::F32).unwrap(),
            20_480
        );
        assert_eq!(
            cuda_kv_per_position(&QWEN3_6_35B_A3B, KvDtype::Q8).unwrap(),
            10_880
        );
    }

    /// The free bytes at which exactly `context` positions of `kv` fit beside `weights`.
    fn free_for(kv: KvDtype, weights: u64, context: u32) -> u64 {
        let (_, _, demand_at) = card_demand_model(&QWEN3_6_35B_A3B, kv).unwrap();
        weights
            + demand_at(context).unwrap().total_after_weights().unwrap()
            + ADMISSION_UNCERTAINTY_BAND_BYTES
    }

    const WEIGHTS: u64 = 20_408_576_512;

    #[test]
    fn a_roomy_card_keeps_fp16_at_the_model_maximum() {
        let free = free_for(KvDtype::F32, WEIGHTS, MODEL_MAX_CONTEXT);
        let fit = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            None,
            &|_, _| Some(0),
        )
        .unwrap();
        assert_eq!(
            fit,
            Some(CudaFit {
                form: KvDtype::F32,
                context: MODEL_MAX_CONTEXT
            })
        );
    }

    #[test]
    fn fp16_is_kept_where_it_reaches_the_sufficient_context() {
        let free = free_for(KvDtype::F32, WEIGHTS, 150_528);
        let fit = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            None,
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        assert_eq!(fit.form, KvDtype::F32);
        assert_eq!(fit.context, 150_528);
    }

    #[test]
    fn int8_is_chosen_where_fp16_falls_short_of_the_sufficient_context() {
        let free = free_for(KvDtype::F32, WEIGHTS, 100_352);
        let fit = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            None,
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        assert_eq!(fit.form, KvDtype::Q8);
        assert!(fit.context > 150_000, "{}", fit.context);
        // The chosen context is the largest one that fits: one step more does not.
        let (_, _, demand_at) = card_demand_model(&QWEN3_6_35B_A3B, KvDtype::Q8).unwrap();
        assert!(admits_cuda(free, WEIGHTS, &demand_at, fit.context));
        assert!(
            fit.context == MODEL_MAX_CONTEXT
                || !admits_cuda(free, WEIGHTS, &demand_at, fit.context + 1024)
        );
    }

    #[test]
    fn a_named_cache_form_or_context_is_taken_as_given() {
        let free = free_for(KvDtype::F32, WEIGHTS, 100_352);
        // The form is fixed: only that form is weighed, whatever the other would reach.
        let fixed = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            Some(KvDtype::F32),
            None,
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        assert_eq!((fixed.form, fixed.context), (KvDtype::F32, 100_352));
        // The context is fixed and fp16 holds it: fp16 serves exactly that context.
        let named = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            Some(65_536),
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        assert_eq!((named.form, named.context), (KvDtype::F32, 65_536));
        // The context is fixed, fp16 cannot hold it and int8 can: int8 serves it.
        let wide = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            Some(180_224),
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        assert_eq!((wide.form, wide.context), (KvDtype::Q8, 180_224));
    }

    #[test]
    fn a_named_context_above_what_fp16_holds_goes_to_the_form_that_reaches_it() {
        // fp16 holds 150,528 positions, more than the sufficient context, yet the named
        // context is beyond it: int8 reaches the named value, so int8 serves it.
        let free = free_for(KvDtype::F32, WEIGHTS, 150_528);
        let named = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            Some(262_144),
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("fits");
        let (_, _, int8_demand) = card_demand_model(&QWEN3_6_35B_A3B, KvDtype::Q8).unwrap();
        assert!(admits_cuda(free, WEIGHTS, &int8_demand, 262_144));
        assert_eq!((named.form, named.context), (KvDtype::Q8, 262_144));
    }

    #[test]
    fn a_named_context_no_form_reaches_starts_at_the_longest_that_fits() {
        // No form holds 262,144 positions on this card: the start does not refuse, it takes
        // the form that holds the most and reports a context below the named value.
        let free = free_for(KvDtype::F32, WEIGHTS, 100_352);
        let (_, _, int8_demand) = card_demand_model(&QWEN3_6_35B_A3B, KvDtype::Q8).unwrap();
        assert!(!admits_cuda(free, WEIGHTS, &int8_demand, 262_144));
        let named = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            Some(262_144),
            &|_, _| Some(0),
        )
        .unwrap()
        .expect("a smaller context fits");
        assert_eq!(named.form, KvDtype::Q8);
        assert!(
            named.context < 262_144 && named.context > 100_352,
            "{}",
            named.context
        );
        assert!(admits_cuda(free, WEIGHTS, &int8_demand, named.context));
    }

    #[test]
    fn a_card_that_carries_nothing_answers_none() {
        let free = free_for(KvDtype::Q8, WEIGHTS, 1024) - 1;
        let fit = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            true,
            None,
            None,
            &|_, _| Some(0),
        )
        .unwrap();
        assert_eq!(fit, None);
        assert_eq!(
            choose_cuda_fit(
                &QWEN3_6_35B_A3B,
                WEIGHTS - 1,
                WEIGHTS,
                true,
                None,
                None,
                &|_, _| Some(0)
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn int8_is_not_planned_where_the_step_cannot_run_it() {
        // fp16 reaches 100,352 positions on this card, short of the sufficient context, so a
        // step that can run the int8 cache gets it (above); one that cannot (the chunked prefill
        // that gives int8 its arena is off) gets the longest fp16 context the card holds.
        let free = free_for(KvDtype::F32, WEIGHTS, 100_352);
        let none = &|_, _| Some(0);
        let fit = choose_cuda_fit(&QWEN3_6_35B_A3B, free, WEIGHTS, false, None, None, none)
            .unwrap()
            .expect("fits");
        assert_eq!((fit.form, fit.context), (KvDtype::F32, 100_352));
        // The same card, a named context fp16 cannot hold: fp16's longest, not an int8 start
        // the prefill would refuse.
        let named = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            false,
            None,
            Some(262_144),
            none,
        )
        .unwrap()
        .expect("fits");
        assert_eq!((named.form, named.context), (KvDtype::F32, 100_352));
        // A named int8 cache is the operator's and stays.
        let asked = choose_cuda_fit(
            &QWEN3_6_35B_A3B,
            free,
            WEIGHTS,
            false,
            Some(KvDtype::Q8),
            None,
            none,
        )
        .unwrap()
        .expect("fits");
        assert_eq!(asked.form, KvDtype::Q8);
    }
}
