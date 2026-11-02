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
