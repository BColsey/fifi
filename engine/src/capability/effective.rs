//! The effective configuration: every resolved value with where it came
//! from.
//!
//! Resolution itself stays exactly where it always was —
//! `Q36Settings::new` and its per-setting rules, with their current
//! per-key precedence (a config member, else one named environment
//! fallback where a setting has one, else a default). This module adds
//! no rule of its own: given the options, the environment fallback, and
//! the resolved settings, it names each principal value's origin. That
//! keeps preflight, startup, and this view in agreement by construction
//! — there is one resolver, and three renderers of its result.
//!
//! Origins are deliberately coarse and honest. `Derived` is only for
//! values computed from other resolved settings by the resolver itself
//! (today: the MLX prefill reserve and the dense session cap).
//! `Ignored` is for members the active backend reads past, reported
//! from the resolver's own `ignored_mlx_keys` list. One exception is
//! computed here rather than disclosed by the resolver: `compile_program_limit`
//! refuses an explicit CUDA value instead of ignoring it, so it can never
//! appear in `ignored_mlx_keys`; its row reports `IgnoredOnThisBackend` on
//! CUDA by construction (only the default is CUDA-reachable).

use serde::Serialize;

use crate::family::qwen::qwen36::settings::{Backend, Q36Options, Q36Settings};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// Set in the configuration file (or the equivalent CLI candidate).
    ConfigFile,
    /// Read from the setting's named environment variable.
    Environment,
    /// Computed by the resolver from other resolved settings.
    Derived,
    /// The resolver's built-in default for the active backend.
    Default,
    /// Present in the file but not read by the active backend.
    IgnoredOnThisBackend,
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::ConfigFile => "config file",
            Origin::Environment => "environment",
            Origin::Derived => "derived",
            Origin::Default => "default",
            Origin::IgnoredOnThisBackend => "ignored on this backend",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EffectiveSetting {
    pub key: &'static str,
    /// The resolved value, rendered exactly (bytes stay bytes).
    pub value: String,
    pub origin: Origin,
}

fn row(key: &'static str, value: String, origin: Origin) -> EffectiveSetting {
    EffectiveSetting { key, value, origin }
}

fn present<T>(value: &Option<T>) -> Origin {
    if value.is_some() {
        Origin::ConfigFile
    } else {
        Origin::Default
    }
}

/// The principal Qwen3.6 settings with origins. `env_cache_limit` is the
/// `Q36_CACHE_LIMIT` fallback exactly as the factory reads it; callers
/// in a diagnostic process read the real environment, tests inject.
pub(crate) fn q36_effective(
    options: &Q36Options,
    env_cache_limit: Option<&str>,
    backend: Backend,
    settings: &Q36Settings,
) -> Vec<EffectiveSetting> {
    use crate::family::qwen::qwen36::settings::{
        activation_name, kv_dtype_name, weight_format_name,
    };
    let ignored = |key: &str| settings.ignored_mlx_keys.contains(&key);
    let on_arm = |key: &str, origin: Origin| {
        if ignored(key) {
            Origin::IgnoredOnThisBackend
        } else {
            origin
        }
    };
    let dense = settings.spec.feed_forward.is_some();
    let derived = |key: &str| settings.derived.contains(&key);
    let from_host = |key: &str, member: Origin| {
        if derived(key) {
            Origin::Derived
        } else {
            member
        }
    };

    let cache_origin = if options.cache_limit_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && env_cache_limit.is_some() {
        Origin::Environment
    } else if derived("cache_limit_bytes") {
        Origin::Derived
    } else {
        Origin::Default
    };
    let reserve_origin = if options.prefill_session_reserve_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && settings.prefill_budget.is_some() {
        Origin::Derived
    } else {
        Origin::Default
    };
    let cap_origin = if options.session_cap_bytes.is_some() {
        Origin::ConfigFile
    } else if dense && backend == Backend::Mlx {
        Origin::Derived
    } else {
        Origin::Default
    };

    let mut rows = vec![
        row(
            "weight_format",
            weight_format_name(settings.weights).to_string(),
            present(&options.weight_format),
        ),
        row(
            "gguf_sha256",
            settings
                .gguf_sha256
                .clone()
                .unwrap_or_else(|| "off".to_string()),
            present(&options.gguf_sha256),
        ),
        if settings.context_auto {
            row(
                "context_limit",
                "the largest the free card carries, chosen at load".to_string(),
                Origin::Derived,
            )
        } else {
            row(
                "context_limit",
                settings.context_limit.to_string(),
                from_host("context_limit", present(&options.context_limit)),
            )
        },
        if settings.kv_auto {
            row(
                "kv_dtype",
                "f32 (fp16) or q8, chosen from the free card at load".to_string(),
                Origin::Derived,
            )
        } else {
            row(
                "kv_dtype",
                kv_dtype_name(settings.kv).to_string(),
                present(&options.kv_dtype),
            )
        },
        row(
            "activation_dtype",
            activation_name(settings.activation).to_string(),
            present(&options.activation_dtype),
        ),
        row(
            "cache_limit_bytes",
            settings.cache_limit.to_string(),
            on_arm("cache_limit_bytes", cache_origin),
        ),
        row(
            "session_cap_bytes",
            settings.session_cap.to_string(),
            on_arm("session_cap_bytes", cap_origin),
        ),
        row(
            "prefill_budget_bytes",
            settings
                .prefill_budget
                .map_or_else(|| "off".to_string(), |b| b.to_string()),
            from_host(
                "prefill_budget_bytes",
                present(&options.prefill_budget_bytes),
            ),
        ),
        row(
            "prefill_session_reserve_bytes",
            settings.prefill_reserve.to_string(),
            reserve_origin,
        ),
        row(
            "prefill_chunk",
            settings.prefill_chunk.to_string(),
            on_arm("prefill_chunk", present(&options.prefill_chunk)),
        ),
        row(
            "max_seqs",
            settings.max_seqs.to_string(),
            from_host("max_seqs", present(&options.max_seqs)),
        ),
        row(
            "prefix_store_entries",
            settings
                .prefix_store_entries
                .map_or_else(|| "default (6)".to_string(), |n| n.to_string()),
            on_arm(
                "prefix_store_entries",
                present(&options.prefix_store_entries),
            ),
        ),
        row(
            "compile",
            settings.compile.to_string(),
            on_arm("compile", from_host("compile", present(&options.compile))),
        ),
        row(
            "compile_program_limit",
            settings.compile_program_limit.get().to_string(),
            if backend == Backend::Cuda {
                Origin::IgnoredOnThisBackend
            } else {
                present(&options.compile_program_limit)
            },
        ),
        row(
            "capture",
            settings.capture.to_string(),
            present(&options.capture),
        ),
    ];
    if backend == Backend::Mlx && super::probe::CompiledBackends::of_this_build().mlx {
        rows.push(row("backend", "mlx".into(), Origin::Derived));
        append_mlx_inputs(&mut rows, false);
        if !dense {
            rows.push(disk_experts_row());
            rows.push(row(
                "mlx_expert_placement",
                match settings.mlx_staged_cap {
                    Some(_) => "disk (derived: the resident model does not fit the host budget)",
                    None => "by the device budget, resident when it fits",
                }
                .into(),
                if settings.mlx_staged_cap.is_some() {
                    Origin::Derived
                } else {
                    Origin::Default
                },
            ));
        } else {
            rows.push(mlx_switch(
                "mlx_disk_dense_ffn",
                "FIFI_Q36_MLX_DISK_DENSE_FFN",
            ));
        }
    }
    rows
}

/// A Mac switch is on only for the exact value `1`. Any other value is
/// reported as off, with the value named, so the row says what the start
/// will do with it.
fn mlx_switch(key: &'static str, name: &str) -> EffectiveSetting {
    let value = std::env::var_os(name);
    row(key, mlx_switch_text(value.as_deref()), env_origin(&value))
}

/// The Qwen disk-experts override: automatic when unset, `on` for `1`, `off` for `0`, and a
/// value that is neither is named as ignored.
fn disk_experts_row() -> EffectiveSetting {
    use crate::family::qwen::qwen36::settings::DiskExperts;
    let value = std::env::var_os("FIFI_Q36_MLX_DISK_EXPERTS");
    let (setting, ignored) = DiskExperts::of(value.as_deref());
    let label = match setting {
        DiskExperts::Auto => "auto",
        DiskExperts::On => "on",
        DiskExperts::Off => "off",
    };
    let text = match ignored {
        Some(reason) => format!("{label} (ignored: {reason})"),
        None => label.to_string(),
    };
    row("mlx_disk_experts", text, env_origin(&value))
}

/// A switch that is on only for `1`, rendered with the first condition the executor also needs
/// when it is not met, and with the value named when it is not `1`.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn conditional_switch_text(value: Option<&std::ffi::OsStr>, unmet: &[(bool, &str)]) -> String {
    match value {
        None => "false".into(),
        Some(flag) if flag == "1" => match unmet.iter().find(|(met, _)| !met) {
            None => "true".into(),
            Some((_, why)) => format!("true (ignored: {why})"),
        },
        Some(flag) => format!("false (ignored: {:?} is not 1)", flag.to_string_lossy()),
    }
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn request_prefill_text(
    value: Option<&std::ffi::OsStr>,
    device_sizing: bool,
    chained_prefill: bool,
) -> String {
    conditional_switch_text(
        value,
        &[
            (device_sizing, "device sizing is off"),
            (!chained_prefill, "chained prefill is on"),
        ],
    )
}

#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn request_kv_text(
    value: Option<&std::ffi::OsStr>,
    compact_kv: bool,
    device_sizing: bool,
) -> String {
    conditional_switch_text(
        value,
        &[
            (compact_kv, "compact KV is off"),
            (device_sizing, "device sizing is off"),
        ],
    )
}

/// The pair path needs request-sized prefill, the shared-page pipeline and disk-backed experts;
/// the last is decided when the model opens, so a pair that meets the other two reports it.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn prefill_pair_text(
    value: Option<&std::ffi::OsStr>,
    request_prefill: bool,
    pipeline: bool,
) -> String {
    let parsed = matches!(
        value.and_then(std::ffi::OsStr::to_str),
        Some("1" | "auto" | "waves64")
    );
    match value {
        None => "false".into(),
        Some(_) if !parsed => format!(
            "false (ignored: {:?} is not 1, auto or waves64)",
            value.map(|v| v.to_string_lossy()).unwrap_or_default()
        ),
        Some(_) => match [
            (request_prefill, "request prefill is not active"),
            (pipeline, "the shared-page pipeline is off"),
        ]
        .iter()
        .find(|(met, _)| !met)
        {
            Some((_, why)) => format!("true (ignored: {why})"),
            None => "true (applies only with disk-backed experts)".into(),
        },
    }
}

/// The policy the executor will run: off whenever the pair path is off.
#[cfg(any(
    test,
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
))]
fn prefill_pair_policy_text(parsed: &str, request_prefill: bool, pipeline: bool) -> String {
    if parsed != "off" && !(request_prefill && pipeline) {
        format!("off (the {parsed} policy is not active: the pair path is off)")
    } else {
        parsed.to_string()
    }
}

/// The expert staging switches are automatic: unset, the start turns them on when its experts
/// stream from disk. A set variable decides (`on` is the one enabling value, `0` is off) and
/// any other value is named as ignored and counted off.
fn staging_switch_text(value: Option<&std::ffi::OsStr>, on: &str) -> String {
    match value {
        None => "auto (on when the experts stream from disk)".into(),
        Some(value) if value == on => "on".into(),
        Some(value) if value == "0" => "off".into(),
        Some(value) => format!("off (ignored: {:?} is not {on})", value.to_string_lossy()),
    }
}

fn mlx_switch_text(value: Option<&std::ffi::OsStr>) -> String {
    match value {
        None => "false".into(),
        Some(flag) if flag == "1" => "true".into(),
        Some(flag) => format!("false (ignored: {:?} is not 1)", flag.to_string_lossy()),
    }
}

fn env_origin<T>(value: &Option<T>) -> Origin {
    if value.is_some() {
        Origin::Environment
    } else {
        Origin::Default
    }
}

/// The expert read-run policy. A value other than `prefill` is reported as off with the value
/// named. For Flint, FIFI_FLINT_MLX_EXPERT_READ_RUNS (off, `prefill` or `1`) takes precedence
/// when set, and the row says so.
fn read_runs_row(flint: bool) -> EffectiveSetting {
    let legacy = flint
        .then(|| std::env::var_os("FIFI_FLINT_MLX_EXPERT_READ_RUNS"))
        .flatten();
    if legacy.is_some() {
        return row(
            "mlx_expert_read_runs",
            legacy_read_runs_text(legacy.as_deref()),
            Origin::Environment,
        );
    }
    let shared = std::env::var_os("FIFI_MLX_EXPERT_READ_RUNS");
    row(
        "mlx_expert_read_runs",
        read_runs_text(shared.as_deref()),
        env_origin(&shared),
    )
}

fn read_runs_text(value: Option<&std::ffi::OsStr>) -> String {
    staging_switch_text(value, "prefill")
}

fn legacy_read_runs_text(value: Option<&std::ffi::OsStr>) -> String {
    let name = "FIFI_FLINT_MLX_EXPERT_READ_RUNS";
    match value {
        Some(value) if value == "prefill" => format!("prefill ({name} takes precedence)"),
        Some(value) if value == "1" => format!("on ({name} takes precedence)"),
        Some(value) => format!(
            "off ({name} takes precedence; ignored: {:?} is not prefill or 1)",
            value.to_string_lossy()
        ),
        None => "off".into(),
    }
}

/// Whether the shared expert overlaps the CPU reads. For Flint, FIFI_FLINT_MLX_EXPERT_PIPELINE
/// takes precedence when set.
fn overlap_row(flint: bool) -> EffectiveSetting {
    let legacy = flint
        .then(|| std::env::var_os("FIFI_FLINT_MLX_EXPERT_PIPELINE"))
        .flatten();
    match legacy {
        Some(value) => row(
            "mlx_overlap_expert_reads",
            format!(
                "{} (FIFI_FLINT_MLX_EXPERT_PIPELINE takes precedence)",
                value == "1"
            ),
            Origin::Environment,
        ),
        None => {
            let shared = std::env::var_os("FIFI_MLX_OVERLAP_EXPERT_READS");
            row(
                "mlx_overlap_expert_reads",
                staging_switch_text(shared.as_deref(), "1"),
                env_origin(&shared),
            )
        }
    }
}

fn append_mlx_inputs(rows: &mut Vec<EffectiveSetting>, flint: bool) {
    rows.push(mlx_switch("mlx_auto_budget", "FIFI_MLX_AUTO_BUDGET"));
    let positioned = std::env::var_os("FIFI_MLX_POSITIONED_READS");
    rows.push(row(
        "mlx_positioned_reads",
        staging_switch_text(positioned.as_deref(), "1"),
        env_origin(&positioned),
    ));
    rows.push(read_runs_row(flint));
    rows.push(overlap_row(flint));
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        use crate::family::mlx_budget;
        let (value, origin) = match mlx_budget::configured_ceiling() {
            Ok(None) => ("off".into(), Origin::Default),
            Ok(Some(bytes)) => (bytes.to_string(), Origin::Environment),
            Err(_) => ("invalid".into(), Origin::Environment),
        };
        rows.push(row("mlx_device_ceiling_bytes", value, origin));
        rows.push(row(
            "mlx_device_sizing",
            mlx_budget::enabled().to_string(),
            Origin::Derived,
        ));
    }
}

pub(crate) fn flash_mlx_effective(
    options: &crate::family::qwen::flash_next::settings::FlashOptions,
    env: &crate::family::qwen::flash_next::settings::FlashEnv,
    settings: &crate::family::qwen::flash_next::settings::FlashSettings,
) -> Vec<EffectiveSetting> {
    use crate::family::qwen::flash_next::factory::CONTEXT_LIMIT;
    let cache_origin = if options.cache_limit_bytes.is_some() {
        Origin::ConfigFile
    } else {
        env_origin(&env.cache_limit)
    };
    let mut rows = vec![
        row("backend", "mlx".into(), Origin::Derived),
        row("context_limit", CONTEXT_LIMIT.to_string(), Origin::Default),
        row("kv_dtype", "f32".into(), Origin::Default),
        row(
            "cache_limit_bytes",
            settings.cache_limit.to_string(),
            cache_origin,
        ),
        row(
            "prefill_budget_bytes",
            settings
                .prefill_budget
                .map_or_else(|| "off".into(), |bytes| bytes.to_string()),
            present(&options.prefill_budget_bytes),
        ),
        row(
            "prefill_chunk",
            settings.prefill_chunk.to_string(),
            present(&options.prefill_chunk),
        ),
        mlx_switch("mlx_sparse_prefill", "FIFI_FLASH_NEXT_MLX_SPARSE_PREFILL"),
        mlx_switch("mlx_capacity_kv", "FIFI_FLASH_NEXT_MLX_CAPACITY_KV"),
    ];
    append_mlx_inputs(&mut rows, false);
    rows
}

pub(crate) fn flash_mlx_config_effective(
    model_dir: &std::path::Path,
    tokenizer_dir: &std::path::Path,
    options: Option<&crate::family::qwen::flash_next::settings::FlashOptions>,
) -> Result<Vec<EffectiveSetting>, crate::EngError> {
    use crate::family::qwen::flash_next::{
        factory::check_mlx_settings,
        settings::{FlashEnv, FlashOptions, FlashSettings},
    };
    let env = FlashEnv::capture();
    let settings = FlashSettings::new(model_dir, tokenizer_dir, options, &env)?;
    check_mlx_settings(&settings)?;
    let default = FlashOptions::default();
    Ok(flash_mlx_effective(
        options.unwrap_or(&default),
        &env,
        &settings,
    ))
}

pub(crate) fn flint_effective_for_build() -> Result<Vec<EffectiveSetting>, crate::EngError> {
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        use crate::family::flint::factory;
        let context = std::env::var_os("FIFI_FLINT_MLX_CONTEXT_LIMIT");
        let rows = std::env::var_os("FIFI_FLINT_MLX_BATCH_PREFILL");
        let cache = std::env::var_os("FIFI_FLINT_MLX_CACHE_LIMIT");
        let request_kv = std::env::var_os("FIFI_FLINT_MLX_REQUEST_KV");
        let request_prefill = std::env::var_os("FIFI_FLINT_MLX_REQUEST_PREFILL");
        let prefill_pair = std::env::var_os("FIFI_FLINT_MLX_PREFILL_PAIR");
        let conditions = factory::switch_conditions();
        let (sizing, chained, compact, pipeline) = (
            conditions.device_sizing,
            conditions.chained_prefill,
            conditions.compact_kv,
            conditions.pipeline,
        );
        let request_prefill_on = factory::request_prefill_enabled(request_prefill.as_deref());
        let pair_policy = factory::prefill_pair_policy(prefill_pair.as_deref());
        let mut result = vec![
            row("backend", "mlx".into(), Origin::Derived),
            row(
                "context_limit",
                factory::context_limit().to_string(),
                env_origin(&context),
            ),
            row("kv_dtype", "bf16".into(), Origin::Default),
            row(
                "prefill_chunk",
                factory::prefill_chunk_rows(factory::os_setting(rows.as_deref())).to_string(),
                env_origin(&rows),
            ),
            row(
                "cache_limit_bytes",
                factory::cache_limit_with(factory::os_setting(cache.as_deref())).to_string(),
                env_origin(&cache),
            ),
            row(
                "mlx_request_prefill",
                request_prefill_text(request_prefill.as_deref(), sizing, chained),
                env_origin(&request_prefill),
            ),
            row(
                "mlx_prefill_pair",
                prefill_pair_text(prefill_pair.as_deref(), request_prefill_on, pipeline),
                env_origin(&prefill_pair),
            ),
            row(
                "mlx_prefill_pair_policy",
                prefill_pair_policy_text(pair_policy.label(), request_prefill_on, pipeline),
                env_origin(&prefill_pair),
            ),
            mlx_switch(
                "mlx_cancel_prefill_pair",
                "FIFI_FLINT_MLX_CANCEL_PREFILL_PAIR",
            ),
            mlx_switch("mlx_compact_kv", "FIFI_FLINT_MLX_COMPACT_KV"),
            row(
                "mlx_request_kv",
                request_kv_text(request_kv.as_deref(), compact, sizing),
                env_origin(&request_kv),
            ),
        ];
        append_mlx_inputs(&mut result, true);
        Ok(result)
    }
    #[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        Err(crate::EngError::BadRequest(
            "engine config: Flint requires --features native-mlx on Apple Silicon".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{Origin, q36_effective};
    use crate::family::qwen::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
    use std::path::Path;

    fn resolve(options: &Q36Options, backend: Backend) -> Q36Settings {
        Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(options),
            &Q36Env::default(),
            backend,
        )
        .expect("resolves")
    }

    fn origin(rows: &[super::EffectiveSetting], key: &str) -> Origin {
        rows.iter()
            .find(|r| r.key == key)
            .unwrap_or_else(|| panic!("no row {key}"))
            .origin
    }

    #[test]
    fn q36_program_budget_reports_the_resolved_limit_and_origin() {
        for (limit, expected, source) in [
            (None, "8", Origin::Default),
            (Some(12), "12", Origin::ConfigFile),
        ] {
            let options = Q36Options {
                cache_limit_bytes: Some(1024),
                compile_program_limit: limit,
                ..Q36Options::default()
            };
            let settings = resolve(&options, Backend::Mlx);
            let rows = q36_effective(&options, None, Backend::Mlx, &settings);
            let row = rows
                .iter()
                .find(|row| row.key == "compile_program_limit")
                .unwrap();
            assert_eq!(row.value, expected);
            assert_eq!(row.origin, source);
        }
        let options = Q36Options::default();
        let settings = resolve(&options, Backend::Cuda);
        let rows = q36_effective(&options, None, Backend::Cuda, &settings);
        assert_eq!(
            origin(&rows, "compile_program_limit"),
            Origin::IgnoredOnThisBackend
        );
    }

    #[test]
    fn q36_content_pin_reports_the_normalized_value_and_its_origin() {
        for pin in [None, Some("A".repeat(64))] {
            let options = Q36Options {
                gguf_sha256: pin.clone(),
                ..Q36Options::default()
            };
            let settings = resolve(&options, Backend::Cuda);
            let rows = q36_effective(&options, None, Backend::Cuda, &settings);
            let row = rows.iter().find(|row| row.key == "gguf_sha256").unwrap();
            assert_eq!(
                row.value,
                if pin.is_some() {
                    "a".repeat(64)
                } else {
                    "off".to_string()
                }
            );
            assert_eq!(
                row.origin,
                if pin.is_some() {
                    Origin::ConfigFile
                } else {
                    Origin::Default
                }
            );
        }
    }

    #[test]
    fn origins_distinguish_file_derived_and_default() {
        let options = Q36Options {
            cache_limit_bytes: Some(4_294_967_296),
            context_limit: Some(262_144),
            prefill_budget_bytes: Some(32_212_254_720),
            kv_dtype: Some("q8".into()),
            ..Q36Options::default()
        };
        let settings = resolve(&options, Backend::Mlx);
        let rows = q36_effective(&options, None, Backend::Mlx, &settings);
        assert_eq!(origin(&rows, "context_limit"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "cache_limit_bytes"), Origin::ConfigFile);
        assert_eq!(
            origin(&rows, "prefill_session_reserve_bytes"),
            Origin::Derived
        );
        assert_eq!(origin(&rows, "activation_dtype"), Origin::Default);
        assert_eq!(origin(&rows, "session_cap_bytes"), Origin::Default);
        let reserve = rows
            .iter()
            .find(|r| r.key == "prefill_session_reserve_bytes")
            .expect("row");
        assert_eq!(reserve.value, settings.prefill_reserve.to_string());
    }

    #[test]
    fn the_environment_fallback_and_ignored_keys_are_named() {
        let options = Q36Options {
            session_cap_bytes: Some(1000),
            compile: Some(true),
            ..Q36Options::default()
        };
        let env = Q36Env {
            cache_limit: Some("2048".to_string()),
            ..Q36Env::default()
        };
        let settings = Q36Settings::new(
            Path::new("m"),
            Path::new("t"),
            Some(&options),
            &env,
            Backend::Mlx,
        )
        .expect("resolves via env");
        let rows = q36_effective(&options, Some("2048"), Backend::Mlx, &settings);
        assert_eq!(origin(&rows, "cache_limit_bytes"), Origin::Environment);

        let cuda = resolve(&options, Backend::Cuda);
        let rows = q36_effective(&options, None, Backend::Cuda, &cuda);
        assert_eq!(
            origin(&rows, "session_cap_bytes"),
            Origin::IgnoredOnThisBackend
        );
        assert_eq!(origin(&rows, "compile"), Origin::IgnoredOnThisBackend);
    }

    #[test]
    fn a_mac_switch_names_a_value_other_than_one_as_ignored() {
        use std::ffi::OsStr;
        assert_eq!(super::mlx_switch_text(None), "false");
        assert_eq!(super::mlx_switch_text(Some(OsStr::new("1"))), "true");
        for other in ["true", "0", "", "yes"] {
            let text = super::mlx_switch_text(Some(OsStr::new(other)));
            assert!(
                text.starts_with("false (ignored: ") && text.contains(&format!("{other:?}")),
                "{other:?}: {text}"
            );
        }
    }

    #[test]
    fn the_read_run_rows_name_ignored_values_and_the_flint_variable_that_wins() {
        use std::ffi::OsStr;
        assert_eq!(
            super::read_runs_text(None),
            "auto (on when the experts stream from disk)"
        );
        assert_eq!(super::read_runs_text(Some(OsStr::new("prefill"))), "on");
        assert_eq!(
            super::read_runs_text(Some(OsStr::new("1"))),
            "off (ignored: \"1\" is not prefill)"
        );
        assert_eq!(
            super::legacy_read_runs_text(Some(OsStr::new("1"))),
            "on (FIFI_FLINT_MLX_EXPERT_READ_RUNS takes precedence)"
        );
        assert_eq!(
            super::legacy_read_runs_text(Some(OsStr::new("prefill"))),
            "prefill (FIFI_FLINT_MLX_EXPERT_READ_RUNS takes precedence)"
        );
        let bad = super::legacy_read_runs_text(Some(OsStr::new("yes")));
        assert!(
            bad.starts_with("off (FIFI_FLINT_MLX_EXPERT_READ_RUNS takes precedence; ignored: ")
        );
    }

    #[test]
    fn the_flint_request_rows_name_the_condition_that_stops_them() {
        use std::ffi::OsStr;
        let one = Some(OsStr::new("1"));
        assert_eq!(super::request_prefill_text(None, true, false), "false");
        assert_eq!(super::request_prefill_text(one, true, false), "true");
        assert_eq!(
            super::request_prefill_text(one, false, false),
            "true (ignored: device sizing is off)"
        );
        assert_eq!(
            super::request_prefill_text(one, true, true),
            "true (ignored: chained prefill is on)"
        );
        assert_eq!(
            super::request_prefill_text(Some(OsStr::new("yes")), true, false),
            "false (ignored: \"yes\" is not 1)"
        );
        assert_eq!(super::request_kv_text(one, true, true), "true");
        assert_eq!(
            super::request_kv_text(one, false, true),
            "true (ignored: compact KV is off)"
        );
        assert_eq!(
            super::request_kv_text(one, true, false),
            "true (ignored: device sizing is off)"
        );
        assert_eq!(super::request_kv_text(None, false, false), "false");
    }

    #[test]
    fn the_flint_pair_rows_follow_the_executors_conditions() {
        use std::ffi::OsStr;
        let value = |text: &'static str| Some(OsStr::new(text));
        assert_eq!(super::prefill_pair_text(None, true, true), "false");
        for text in ["1", "auto", "waves64"] {
            assert_eq!(
                super::prefill_pair_text(value(text), true, true),
                "true (applies only with disk-backed experts)"
            );
        }
        assert_eq!(
            super::prefill_pair_text(value("auto"), false, true),
            "true (ignored: request prefill is not active)"
        );
        assert_eq!(
            super::prefill_pair_text(value("auto"), true, false),
            "true (ignored: the shared-page pipeline is off)"
        );
        assert_eq!(
            super::prefill_pair_text(value("2"), true, true),
            "false (ignored: \"2\" is not 1, auto or waves64)"
        );
        assert_eq!(super::prefill_pair_policy_text("off", true, true), "off");
        assert_eq!(super::prefill_pair_policy_text("auto", true, true), "auto");
        assert_eq!(
            super::prefill_pair_policy_text("auto", true, false),
            "off (the auto policy is not active: the pair path is off)"
        );
    }

    #[test]
    fn the_staging_rows_are_automatic_unless_a_variable_decides() {
        use std::ffi::OsStr;
        assert_eq!(
            super::staging_switch_text(None, "1"),
            "auto (on when the experts stream from disk)"
        );
        assert_eq!(super::staging_switch_text(Some(OsStr::new("1")), "1"), "on");
        assert_eq!(
            super::staging_switch_text(Some(OsStr::new("0")), "1"),
            "off"
        );
        assert_eq!(
            super::staging_switch_text(Some(OsStr::new("1")), "prefill"),
            "off (ignored: \"1\" is not prefill)"
        );
    }
}
