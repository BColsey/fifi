//! The engine factory for Q36 and dense Qwen3.8 entries of the `enginesrv` config.
//!
//! It follows the interface of the Flash-Next factory (`flash_next/factory.rs`) and returns the
//! same `StartedEngine`. The server start-up code (`server/run.rs`) calls it for the entries of
//! this family with the process-wide `prefix_reuse` flag of the configuration file. Order of
//! checks: the settings (`settings.rs`), the `prefix_reuse` session-cap rule when the flag is on
//! and the arm is MLX (`Q36Settings::check_prefix_reuse_cap`), then the backend. A build with
//! neither `native-mlx` on Apple Silicon nor `native-cuda` on Linux stops at the backend check with
//! a clear error naming both features and never touches a file.

use super::settings::{Backend, Q36Env, Q36Options, Q36Settings};
use crate::family::mem::EntryFit;
use crate::{
    EngError,
    family::{FamilySpec, QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE, flash_next::factory::StartedEngine},
    sched::worker::{Engine, ReadyNote},
    tokenizer::TokenizerBundle,
};
use std::path::Path;

/// The family descriptor owns the name.
pub(crate) const FAMILY: &str = QWEN3_6_35B_A3B.name;
/// The dense family shares the factory, but not its dimensions or quantization defaults.
pub(crate) const DENSE_FAMILY: &str = QWEN3_8_27B_DENSE.name;

pub(crate) fn descriptor(family: &str) -> Option<FamilySpec> {
    match family {
        FAMILY => Some(QWEN3_6_35B_A3B),
        DENSE_FAMILY => Some(QWEN3_8_27B_DENSE),
        _ => None,
    }
}

/// The engine state is `Loading` until the weights are resident.
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    let (started, _settings) = start_engine_for_family(
        FAMILY,
        model_dir,
        tokenizer_dir,
        options,
        prefix_reuse,
        ready_note,
    )?;
    Ok(started)
}

pub(crate) fn start_engine_for_family(
    family: &str,
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<(StartedEngine, Q36Settings), EngError> {
    let spec = descriptor(family)
        .ok_or_else(|| EngError::BadRequest(format!("unsupported Q36-style family {family}")))?;
    let env = Q36Env::from_process(model_dir);
    start_engine_with(
        model_dir,
        tokenizer_dir,
        options,
        &env,
        prefix_reuse,
        spec,
        ready_note,
    )
}

pub(crate) fn backend_of_build() -> Backend {
    if cfg!(all(
        feature = "native-mlx",
        target_os = "macos",
        target_arch = "aarch64"
    )) {
        Backend::Mlx
    } else if cfg!(all(feature = "native-cuda", target_os = "linux")) {
        Backend::Cuda
    } else {
        Backend::Mlx
    }
}

/// The settings of an entry with the process-wide `prefix_reuse` flag applied:
/// `Q36Settings::new` builds the checked settings and leaves the flag false, because no config
/// member or environment key carries the process-wide flag, so the flag travels in the settings
/// from here — the MLX `start` reads it into the `prefix_store` limits of the rules. The
/// builder is pure: the factory and its test share it, and the server's
/// effective-configuration parity test resolves through it so the retained
/// rows cannot drift from what a start resolves.
pub(crate) fn settings_with_prefix_reuse(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
    backend: Backend,
    prefix_reuse: bool,
    spec: FamilySpec,
) -> Result<Q36Settings, EngError> {
    let mut settings =
        Q36Settings::new_for_family(model_dir, tokenizer_dir, options, env, backend, spec)?;
    settings.prefix_reuse = prefix_reuse;
    // An MLX start that sets no session cap takes the cap the prefix store needs, instead of
    // refusing the default cap.
    // The raise stays inside the memory budget: a cap that would take the model, the reserve and
    // one entry past it is left alone, and the cap rule then refuses with what it needs.
    if prefix_reuse
        && backend == Backend::Mlx
        && options.is_none_or(|options| options.session_cap_bytes.is_none())
        && entry_fit(&settings, env).is_none_or(|fit| fit.fits())
    {
        settings.raise_session_cap_for_prefix_reuse()?;
    }
    Ok(settings)
}

/// The fit of one saved prefix in the budget the settings resolved: `None` when the host memory,
/// the budget or the entry size is not known. A start that streams its experts counts one stage
/// of them, not the whole stacks.
fn entry_fit(settings: &Q36Settings, env: &Q36Env) -> Option<EntryFit> {
    let (host, budget) = (env.host?, settings.prefill_budget?);
    let entry = settings.prefix_entry_bytes().ok()?;
    let resident = match (settings.mlx_staged_cap, env.experts) {
        (Some(_), Some(experts)) => host.staged(experts).model_bytes,
        _ => host.model_bytes,
    };
    Some(EntryFit::of(
        resident,
        settings.prefill_reserve,
        entry,
        budget,
    ))
}

/// Whether an entry that sets no `prefix_reuse` keeps conversation prefixes on a Mac, and why
/// not when it does not. Reuse is on when the host memory holds the model, the session reserve
/// and one saved prefix entry at the context the start will use, and the configuration does not
/// pin a smaller session cap; otherwise the start keeps the plain full prefill.
pub(crate) fn mlx_auto_prefix_reuse(
    family: &str,
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
) -> (bool, String) {
    let Some(spec) = descriptor(family) else {
        return (false, "unsupported family".into());
    };
    if backend_of_build() != Backend::Mlx {
        return (false, "not an Apple Silicon start".into());
    }
    let env = Q36Env::from_process(model_dir);
    mlx_auto_prefix_reuse_with(spec, model_dir, tokenizer_dir, options, &env)
}

fn mlx_auto_prefix_reuse_with(
    spec: FamilySpec,
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
) -> (bool, String) {
    let off = |why: String| (false, why);
    if spec.feed_forward.is_some() {
        return off("the dense family keeps one session".into());
    }
    let settings = match settings_with_prefix_reuse(
        model_dir,
        tokenizer_dir,
        options,
        env,
        Backend::Mlx,
        true,
        spec,
    ) {
        Ok(settings) => settings,
        Err(error) => return off(format!("the settings do not resolve ({error:?})")),
    };
    if settings.check_prefix_reuse_cap(true).is_err() {
        return off("the configured session_cap_bytes is below what a saved prefix needs".into());
    }
    let Some(fit) = entry_fit(&settings, env) else {
        return off("the host memory, the prefill budget or the prefix entry is not known".into());
    };
    if !fit.fits() {
        return off(format!(
            "the model, the session reserve and one saved prefix need {} bytes and the budget is {}",
            fit.needed, fit.budget
        ));
    }
    (
        true,
        format!(
            "the model, the session reserve and one saved prefix need {} of {} bytes",
            fit.needed, fit.budget
        ),
    )
}

/// The Linux tests reach `settings_with_prefix_reuse` and the rules; the call from this
/// function is covered by the Mac served smoke (prefix-reuse row).
fn start_engine_with(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&Q36Options>,
    env: &Q36Env,
    prefix_reuse: bool,
    spec: FamilySpec,
    ready_note: ReadyNote,
) -> Result<(StartedEngine, Q36Settings), EngError> {
    let settings = settings_with_prefix_reuse(
        model_dir,
        tokenizer_dir,
        options,
        env,
        backend_of_build(),
        prefix_reuse,
        spec,
    )?;
    settings.check_prefix_reuse_cap(prefix_reuse)?;
    // A card-chosen context and cache form are known only once the engine thread has read the
    // card, so the settings the caller gets are the ones the engine serves.
    let (engine, bundle, settings) = launch(&settings, ready_note)?;
    Ok((
        StartedEngine {
            engine,
            bundle,
            context_limit: settings.context_limit,
        },
        settings,
    ))
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, Q36Settings), EngError> {
    let (engine, bundle) = super::start::start(settings, ready_note)?;
    Ok((engine, bundle, settings.clone()))
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
fn launch(
    settings: &Q36Settings,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, Q36Settings), EngError> {
    super::cuda::start::start(settings, ready_note)
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
fn launch(
    settings: &Q36Settings,
    _ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, Q36Settings), EngError> {
    if settings.spec.feed_forward.is_some() {
        return Err(EngError::BadRequest(format!(
            "engine config: this build has no {} backend; build with --features native-mlx on Apple Silicon",
            settings.spec.name
        )));
    }
    Err(EngError::BadRequest(
        "engine config: this build has no Q36 backend; build with --features native-mlx on Apple Silicon or --features native-cuda on Linux"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{FAMILY, Q36Env, start_engine_with};
    use crate::EngError;
    use crate::family::qwen36::settings::Q36Options;
    use std::path::Path;

    fn message<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn start(options: Option<&Q36Options>, env: &Q36Env) -> Result<(), EngError> {
        start_engine_with(
            Path::new("/a"),
            Path::new("/a"),
            options,
            env,
            false,
            crate::family::QWEN3_6_35B_A3B,
            None,
        )
        .map(|_| ())
    }

    fn start_with_reuse(
        options: Option<&Q36Options>,
        env: &Q36Env,
        flag: bool,
    ) -> Result<(), EngError> {
        start_engine_with(
            Path::new("/a"),
            Path::new("/a"),
            options,
            env,
            flag,
            crate::family::QWEN3_6_35B_A3B,
            None,
        )
        .map(|_| ())
    }

    fn error_text(result: Result<(), EngError>) -> String {
        match result {
            Err(EngError::ModelOutput(text)) => text,
            Err(EngError::BadRequest(text))
            | Err(EngError::Checkpoint(text))
            | Err(EngError::MemoryPlan(text))
            | Err(EngError::ContextLimit(text)) => text,
            Err(EngError::PromptTooLong {
                prompt_tokens,
                limit_tokens,
            }) => EngError::prompt_too_long_message(prompt_tokens, limit_tokens),
            Err(EngError::Backend { operation, message }) => format!("{operation}: {message}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key), so the cap rule has no CUDA
    // premise.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_below_the_required_session_cap_is_refused_with_the_numbers_and_the_hint()
     {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(4 << 30),
            ..Q36Options::default()
        };
        // A configured cap of 4 GiB covers one live session of the default 8-bit cache at the
        // default limit of 262,144 positions (2,917,990,400 bytes with the DeltaNet state), but
        // not the entry beside it (5,770,117,120 bytes together). A cap that the configuration
        // sets stays exact; only an unset cap is raised to what the prefix store needs.
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(
            text.contains("at least 5770117120 bytes")
                && text.contains("context limit 262144")
                && text.contains("the cap is 4294967296 bytes")
                && text.contains("raise session_cap_bytes"),
            "{text}"
        );
    }

    // An unset session cap is raised to what the prefix store needs instead of refusing.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn an_unset_session_cap_is_raised_for_prefix_reuse() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(!text.contains("raise session_cap_bytes"), "{text}");
    }

    // The CUDA arm ignores `session_cap_bytes` (an MLX-only key), so the cap rule has no CUDA
    // premise.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_prefix_reuse_entry_at_the_required_session_cap_passes_the_rule() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(5_770_117_120),
            ..Q36Options::default()
        };
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), true));
        assert!(!text.contains("raise session_cap_bytes"), "{text}");
    }

    #[test]
    fn the_session_cap_rule_is_off_without_the_prefix_reuse_flag() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(1),
            ..Q36Options::default()
        };
        let text = error_text(start_with_reuse(Some(&options), &Q36Env::default(), false));
        assert!(!text.contains("raise session_cap_bytes"), "{text}");
    }

    /// The pure settings builder carries the flag for both values: deleting the line that sets
    /// `settings.prefix_reuse` kept every other test green, so the flag is asserted here, on
    /// the builder the factory itself uses.
    ///
    /// Mutation check: removing `settings.prefix_reuse = prefix_reuse;` fails the `true` row.
    #[test]
    fn the_settings_builder_carries_the_prefix_reuse_flag_for_both_values() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        for flag in [false, true] {
            let settings = super::settings_with_prefix_reuse(
                Path::new("/a"),
                Path::new("/a"),
                Some(&options),
                &Q36Env::default(),
                super::backend_of_build(),
                flag,
                crate::family::QWEN3_6_35B_A3B,
            )
            .unwrap_or_else(|_| panic!("flag {flag}: the settings must build"));
            assert_eq!(settings.prefix_reuse, flag, "flag {flag}");
        }
    }

    #[test]
    fn the_family_name_is_the_documented_value() {
        assert_eq!(FAMILY, "qwen3_6_35b_a3b");
    }

    #[test]
    fn the_factory_selects_the_descriptor_named_by_the_config() {
        use crate::family::{QWEN3_6_35B_A3B, QWEN3_8_27B_DENSE};

        for (name, spec, cap) in [
            ("qwen3_6_35b_a3b", QWEN3_6_35B_A3B, 4 << 30),
            ("qwen3_8_27b_dense", QWEN3_8_27B_DENSE, 9_283_698_688),
        ] {
            let options = Q36Options {
                cache_limit_bytes: Some(1024),
                session_cap_bytes: Some(cap),
                ..Q36Options::default()
            };
            let selected = super::descriptor(name).expect("known family");
            let settings = super::settings_with_prefix_reuse(
                Path::new("/a"),
                Path::new("/a"),
                Some(&options),
                &Q36Env::default(),
                super::Backend::Mlx,
                false,
                selected,
            )
            .unwrap();
            assert_eq!(settings.spec, spec);
            assert_eq!(settings.context_limit, 262_144);
            assert_eq!(settings.session_cap, cap);
        }
        assert!(super::descriptor("unknown").is_none());
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn a_dense_entry_without_a_backend_names_only_the_supported_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            session_cap_bytes: Some(9_283_698_688),
            ..Q36Options::default()
        };
        let text = message(start_engine_with(
            Path::new("/nonexistent"),
            Path::new("/nonexistent"),
            Some(&options),
            &Q36Env::default(),
            false,
            crate::family::QWEN3_8_27B_DENSE,
            None,
        ));
        assert_eq!(
            text,
            "engine config: this build has no qwen3_8_27b_dense backend; build with --features native-mlx on Apple Silicon"
        );
    }

    // The three tests below pin the MLX reading and the fallback refusal of the builds
    // WITHOUT the CUDA arm; on a `native-cuda` Linux build the arm exists and the entry
    // reaches the CUDA start, so the pinned premises are false there by design.
    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn an_entry_with_a_missing_setting_is_refused_before_the_backend() {
        assert_eq!(
            message(start(None, &Q36Env::default())),
            "engine config: cache_limit_bytes is missing and Q36_CACHE_LIMIT is not set"
        );
    }

    #[cfg(not(all(feature = "native-cuda", target_os = "linux")))]
    #[test]
    fn a_bad_member_is_refused_by_name_before_the_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            prefill_chunk: Some(0),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: prefill_chunk must be between 1 and 2048, found 0"
        );
    }

    #[test]
    fn a_bad_activation_dtype_is_refused_by_name_before_the_backend() {
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            activation_dtype: Some("f16".to_string()),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: activation_dtype must be f32 or bf16, got \"f16\""
        );
    }

    #[cfg(not(any(
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
        all(feature = "native-cuda", target_os = "linux")
    )))]
    #[test]
    fn a_complete_entry_in_a_build_without_the_backend_names_both_features() {
        // The FULL fallback text, by equality: an include_str! pin of this file's own literals
        // cannot fail, so the no-backend build — where the text is the error — pins it
        // outright. The old single-feature spelling fails here.
        let options = Q36Options {
            cache_limit_bytes: Some(1024),
            ..Q36Options::default()
        };
        assert_eq!(
            message(start(Some(&options), &Q36Env::default())),
            "engine config: this build has no Q36 backend; build with --features native-mlx \
             on Apple Silicon or --features native-cuda on Linux"
        );
    }

    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    #[test]
    fn the_cuda_arm_is_selected_and_the_fallback_names_both_features() {
        // A complete CUDA entry needs no cache limit: an EMPTY qwen36 object leaves the
        // settings checks and fails in the CUDA start's tokenizer load, never at a refusal.
        let text = match start(None, &Q36Env::default()) {
            Err(EngError::Checkpoint(text)) => text,
            other => panic!("expected the CUDA start's tokenizer refusal, got {other:?}"),
        };
        assert!(text.contains("tokenizer.json"), "{text}");
        assert!(text.contains("is missing"), "{text}");
        assert!(!text.contains("no Q36 backend"), "{text}");
        // The fallback arm is not compiled in this build: its full text is pinned by equality
        // in `a_complete_entry_in_a_build_without_the_backend_names_both_features`, which runs
        // on the no-backend build where the text IS the error.
    }

    #[test]
    fn a_mac_start_keeps_prefixes_when_the_host_holds_one_saved_entry() {
        use super::super::host_plan::{ExpertFacts, HostFacts};
        use super::{QWEN3_6_35B_A3B, mlx_auto_prefix_reuse_with};
        use std::path::Path;
        const GIB: u64 = 1 << 30;
        let env = |memory_gib: u64| Q36Env {
            host: Some(HostFacts {
                memory_bytes: memory_gib * GIB,
                metal_bytes: None,
                ceiling_bytes: None,
                model_bytes: 20_401_094_656,
            }),
            experts: Some(ExpertFacts {
                total_bytes: 18_119_393_280,
                stage_bytes: 452_984_832,
            }),
            ..Q36Env::default()
        };
        let decide = |memory_gib: u64, options: Option<&Q36Options>| {
            mlx_auto_prefix_reuse_with(
                QWEN3_6_35B_A3B,
                Path::new("/a"),
                Path::new("/a"),
                options,
                &env(memory_gib),
            )
        };
        for memory_gib in [16, 24, 36, 64] {
            let (on, reason) = decide(memory_gib, None);
            println!("{memory_gib} GiB host: prefix reuse {on}: {reason}");
            assert!(on, "{memory_gib} GiB: {reason}");
        }
        // A configured cap below what a saved prefix needs keeps reuse off instead of refusing.
        let small_cap = Q36Options {
            session_cap_bytes: Some(1 << 30),
            ..Q36Options::default()
        };
        let (on, reason) = decide(36, Some(&small_cap));
        assert!(!on && reason.contains("session_cap_bytes"), "{reason}");
        // A host with no room for the saved entry keeps reuse off.
        let tight = Q36Options {
            context_limit: Some(262_144),
            ..Q36Options::default()
        };
        let mut crowded = env(36);
        crowded.host.as_mut().unwrap().model_bytes = 28 * GIB;
        let (on, reason) = mlx_auto_prefix_reuse_with(
            QWEN3_6_35B_A3B,
            Path::new("/a"),
            Path::new("/a"),
            Some(&tight),
            &Q36Env {
                experts: None,
                ..crowded
            },
        );
        assert!(!on, "{reason}");
    }

    /// The automatic reuse decision reads the budget the load enforces, so a working-set ceiling
    /// below the host budget is the number it checks against.
    #[test]
    fn automatic_reuse_checks_the_working_set_ceiling_not_the_host_budget() {
        use super::super::host_plan::{ExpertFacts, HostFacts};
        use super::{QWEN3_6_35B_A3B, mlx_auto_prefix_reuse_with};
        use std::path::Path;
        const GIB: u64 = 1 << 30;
        let env = |ceiling_mib: Option<u64>| Q36Env {
            host: Some(HostFacts {
                memory_bytes: 36 * GIB,
                metal_bytes: Some(27 * GIB),
                ceiling_bytes: ceiling_mib.map(|mib| mib << 20),
                model_bytes: 20_401_094_656,
            }),
            experts: Some(ExpertFacts {
                total_bytes: 18_119_393_280,
                stage_bytes: 452_984_832,
            }),
            ..Q36Env::default()
        };
        // A start that names its context, as the served 12 GiB arm does: the default context of
        // 262,144 positions needs more than 12 GiB for the reserve and one entry.
        let options = Q36Options {
            context_limit: Some(32_768),
            ..Q36Options::default()
        };
        let decide = |ceiling| {
            mlx_auto_prefix_reuse_with(
                QWEN3_6_35B_A3B,
                Path::new("/a"),
                Path::new("/a"),
                Some(&options),
                &env(ceiling),
            )
        };
        let (on, reason) = decide(None);
        assert!(on && reason.contains(&(27 * GIB).to_string()), "{reason}");
        let (on, reason) = decide(Some(12288));
        assert!(
            reason.contains(&(12 * GIB).to_string()),
            "the 12 GiB ceiling is the budget: {reason}"
        );
        assert!(
            on,
            "the staged model, the reserve and one entry fit 12 GiB: {reason}"
        );
    }
}
