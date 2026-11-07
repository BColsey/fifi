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
//! from the resolver's own `ignored_mlx_keys` list.

use serde::Serialize;

use crate::family::qwen36::settings::{Backend, Q36Options, Q36Settings};

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
    use crate::family::qwen36::settings::{activation_name, kv_dtype_name, weight_format_name};
    let ignored = |key: &str| settings.ignored_mlx_keys.contains(&key);
    let on_arm = |key: &str, origin: Origin| {
        if ignored(key) {
            Origin::IgnoredOnThisBackend
        } else {
            origin
        }
    };
    let dense = settings.spec.feed_forward.is_some();

    let cache_origin = if options.cache_limit_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && env_cache_limit.is_some() {
        Origin::Environment
    } else {
        Origin::Default
    };
    let reserve_origin = if options.prefill_session_reserve_bytes.is_some() {
        Origin::ConfigFile
    } else if backend == Backend::Mlx && options.prefill_budget_bytes.is_some() {
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

    vec![
        row(
            "weight_format",
            weight_format_name(settings.weights).to_string(),
            present(&options.weight_format),
        ),
        row(
            "context_limit",
            settings.context_limit.to_string(),
            present(&options.context_limit),
        ),
        row(
            "kv_dtype",
            kv_dtype_name(settings.kv).to_string(),
            present(&options.kv_dtype),
        ),
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
            present(&options.prefill_budget_bytes),
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
            present(&options.max_seqs),
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
            on_arm("compile", present(&options.compile)),
        ),
        row(
            "capture",
            settings.capture.to_string(),
            present(&options.capture),
        ),
    ]
}

/// The principal Flint settings with origins, in the same three-part
/// shape as the Qwen3.6 rows: the `flint` config object (a field set
/// there versus the CUDA default), the family's environment switches
/// (`FlintEnv`, read by the caller so diagnostic processes and tests
/// inject rather than capture), and the resolved values the switches
/// produce. Flint's MLX path ignores the `flint` object; these rows are
/// the CUDA view.
pub(crate) fn flint_effective(
    options: &crate::family::flint::settings::FlintOptions,
    env: &crate::family::flint::budget::FlintEnv,
) -> Vec<EffectiveSetting> {
    use crate::family::flint::{
        budget::{BankRequest, ExpertCacheMode, LayerMajor},
        settings::FlintOptions,
    };

    // A config field counts as set when it differs from the no-`flint`
    // default; FlintOptions has no Option fields because a present
    // `flint` object must stay explicit.
    let defaults = FlintOptions::cuda_default();
    let from_config = |value: &str, default: &str| {
        if value == default {
            Origin::Default
        } else {
            Origin::ConfigFile
        }
    };

    let mut rows = vec![
        row(
            "context_limit",
            options.context_limit.to_string(),
            if options.context_limit == defaults.context_limit {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "kv_profile",
            options.kv_profile.clone(),
            from_config(&options.kv_profile, &defaults.kv_profile),
        ),
        row(
            "prefill_chunk_size",
            options.prefill_chunk_size.to_string(),
            if options.prefill_chunk_size == defaults.prefill_chunk_size {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_offload_layers",
            options.cpu_moe_offload_layers.to_string(),
            if options.cpu_moe_offload_layers == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_split_experts",
            options.cpu_moe_split_experts.to_string(),
            if options.cpu_moe_split_experts == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
        row(
            "cpu_moe_threads",
            if options.cpu_moe_threads == 0 {
                "default (6)".to_owned()
            } else {
                options.cpu_moe_threads.to_string()
            },
            if options.cpu_moe_threads == 0 {
                Origin::Default
            } else {
                Origin::ConfigFile
            },
        ),
    ];
    // The environment switches resolve or refuse; an invalid value shows
    // as the refusal text rather than a made-up setting.
    let switch = |set: bool| {
        if set {
            Origin::Environment
        } else {
            Origin::Default
        }
    };
    let mode = env
        .expert_cache_mode()
        .map(ExpertCacheMode::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|error| format!("refused: {error:?}"));
    rows.push(row(
        "expert_cache",
        mode,
        switch(env.expert_cache.is_some()),
    ));
    let request = env.bank_request().map_or_else(
        |error| format!("refused: {error:?}"),
        |request| match request {
            BankRequest::Fit => "fit".to_owned(),
            BankRequest::Exactly(count) => format!("exactly {count}"),
        },
    );
    rows.push(row(
        "expert_slots",
        request,
        switch(env.expert_slots.is_some()),
    ));
    let layer_major = env.layer_major().map_or_else(
        |error| format!("refused: {error:?}"),
        |major| match major {
            LayerMajor::Off => "off".to_owned(),
            LayerMajor::Tiles => "tiles (1)".to_owned(),
            LayerMajor::Jobs => "jobs".to_owned(),
        },
    );
    rows.push(row(
        "layer_major",
        layer_major,
        switch(env.layer_major.is_some()),
    ));
    rows.push(row(
        "alloc_cache",
        if env.alloc_cache_enabled() {
            "on"
        } else {
            "off"
        }
        .to_owned(),
        switch(env.alloc_cache.is_some()),
    ));
    rows
}

#[cfg(test)]
mod tests {
    use super::{Origin, q36_effective};
    use crate::family::qwen36::settings::{Backend, Q36Env, Q36Options, Q36Settings};
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
    fn flint_rows_name_config_env_and_default_origins() {
        use crate::family::flint::budget::FlintEnv;
        use crate::family::flint::settings::FlintOptions;

        let rows = super::flint_effective(&FlintOptions::cuda_default(), &FlintEnv::default());
        assert_eq!(origin(&rows, "context_limit"), Origin::Default);
        let context = rows.iter().find(|r| r.key == "context_limit").unwrap();
        assert_eq!(context.value, "131072");
        assert_eq!(origin(&rows, "kv_profile"), Origin::Default);
        // The alloc cache is default-on; the row says so without a var set.
        assert_eq!(origin(&rows, "alloc_cache"), Origin::Default);
        assert!(
            rows.iter()
                .any(|r| r.key == "alloc_cache" && r.value == "on")
        );

        let mut options = FlintOptions::cuda_default();
        options.kv_profile = "q8".to_owned();
        options.cpu_moe_threads = 4;
        let env = FlintEnv {
            expert_slots: Some("226".to_owned()),
            layer_major: Some(std::ffi::OsString::from("jobs")),
            ..FlintEnv::default()
        };
        let rows = super::flint_effective(&options, &env);
        assert_eq!(origin(&rows, "kv_profile"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "cpu_moe_threads"), Origin::ConfigFile);
        assert_eq!(origin(&rows, "expert_slots"), Origin::Environment);
        let slots = rows.iter().find(|r| r.key == "expert_slots").unwrap();
        assert_eq!(slots.value, "exactly 226");
        assert_eq!(origin(&rows, "layer_major"), Origin::Environment);
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert_eq!(major.value, "jobs");

        // A refused switch value renders as the refusal, not a setting.
        let env = FlintEnv {
            layer_major: Some(std::ffi::OsString::from("yes")),
            ..FlintEnv::default()
        };
        let rows = super::flint_effective(&FlintOptions::cuda_default(), &env);
        let major = rows.iter().find(|r| r.key == "layer_major").unwrap();
        assert!(major.value.starts_with("refused:"), "{}", major.value);
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
}
