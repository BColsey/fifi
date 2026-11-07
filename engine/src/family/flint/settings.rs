//! Explicit options of the first CUDA serving profile; legacy MLX entries omit them.

use super::kv_plan::{KvPlan, KvStorage};
use crate::EngError;
use serde::Deserialize;

// Above eight rows EXL3 changes kernels and reduction order; this is a numeric cap.
pub(crate) const PREFILL_MAX_ROWS: usize = 8;

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn default_on_selected(value: Option<&std::ffi::OsStr>) -> bool {
    match value {
        None => true,
        Some(value) => value
            .to_str()
            .is_some_and(|value| crate::tokenizer::env_allows_value(Some(value))),
    }
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn late_health_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn packed_pointwise_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn grouped_gather_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

/// FIFI_FLINT_CPU_MOE_OVERLAP: whether the CPU experts of a split/offload
/// layer run concurrently with the layer's device pass (the device pass, the
/// shared expert and the scaled gates enqueue while the CPU workers compute,
/// instead of the host-serial serve). Default off; only read when a CPU MoE
/// mode is armed, so the sequential path stays byte-identical without it.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn cpu_moe_overlap_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_off_selected(value)
}

// The layer-major switch and its parse live in `family::flint::budget`
// (host-buildable plan inventory); re-exported under the same gate the
// CUDA executor historically imported it under.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) use crate::family::flint::budget::{LayerMajor, layer_major_from_env};

// The hi/lo switch parses in the budget layer too (host-buildable). The executor open resolves it
// together with the layer-major mode and stores the decision here, so the tall projections of this
// process read the resolved state, not the raw variable.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) use crate::family::flint::budget::{TALL_HILO_VAR, tall_hilo_from_env};

/// Whether the tall projections of the jobs pass of this process split their activations into a hi
/// and a scaled lo F16 part. Set once per executor open from the resolved prefill order
/// ([`set_tall_hilo`]); off until an open has resolved it.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn tall_hilo_armed() -> bool {
    TALL_HILO.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn set_tall_hilo(on: bool) {
    TALL_HILO.store(on, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
static TALL_HILO: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn q8_tiled_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

/// Shared selector body for the default-OFF strict switches: `1`, `true`,
/// `on` (after trim + lowercase) arm; unset or any other text — including a
/// value that is not valid UTF-8 — stays OFF.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
fn default_off_selected(value: Option<&std::ffi::OsStr>) -> bool {
    value
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|value| crate::tokenizer::env_allows_value(Some(value)))
}

#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn kv_device_f16_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

/// FIFI_FLINT_CUDA_Q8_TILED, read once per process.
/// Values: `1`, `true`, `on` (after trim + lowercase) keep the route armed; unset
/// also arms it (default ON). Any other text — including a value that is not
/// valid UTF-8 — switches it OFF. Rollback: set `FIFI_FLINT_CUDA_Q8_TILED=0`
/// (or empty, or any non-accepting text) and restart. The tile kernel differs
/// from the legacy q8 kernel in reduction order within the 2^-14-class bound
/// (not bitwise), so served text can differ from the rolled-back configuration.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn q8_tiled_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED
        .get_or_init(|| q8_tiled_selected(std::env::var_os("FIFI_FLINT_CUDA_Q8_TILED").as_deref()))
}

/// FIFI_FLINT_TOKEN_HEALTH selector: unset, `1`, `true` and `on` (after trim +
/// lowercase) arm it (default ON); any other text — including empty and
/// non-UTF-8 — switches it OFF. Armed, per-frame FFN health defers to one
/// token-end ring read, which changes when (not whether) a fault surfaces.
/// Rollback: `FIFI_FLINT_TOKEN_HEALTH=0` and a restart.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn token_health_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

/// FIFI_FLINT_TOKEN_HEALTH, read once per process: shared by the session
/// scratch charge, session construction and the executor load line, so the
/// printed state can never disagree with the behaviour.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn token_health_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        token_health_selected(std::env::var_os("FIFI_FLINT_TOKEN_HEALTH").as_deref())
    })
}

/// FIFI_FLINT_DEVICE_PICKS selector: `1`, `true`, `on` (after trim + lowercase) arm it; unset or
/// any other text — including non-UTF-8 — leaves it OFF. Armed, a single-row decode step over a
/// fully resident load keeps the router's picks on the device: the routed projections read the
/// expert ids from device memory, so the layer no longer reads the ids back or waits for them.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn device_picks_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_off_selected(value)
}

/// FIFI_FLINT_DEVICE_PICKS, read once per process.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn device_picks_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        device_picks_selected(std::env::var_os("FIFI_FLINT_DEVICE_PICKS").as_deref())
    })
}

/// FIFI_FLINT_CUDA_ALLOC_LAZY_RETURN, read once per process: with the allocation cache on, a
/// returned block goes idle without a device-wide synchronize while the adapter has one stream.
/// Default off; `1`, `true` and `on` arm it, any other value leaves it off.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn alloc_lazy_return_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        default_off_selected(std::env::var_os("FIFI_FLINT_CUDA_ALLOC_LAZY_RETURN").as_deref())
    })
}

/// FIFI_FLINT_CUDA_FUSED_PUBLISH, read once per process.
/// Values: `1`, `true`, `on` (after trim + lowercase) keep the arm on; unset
/// also arms it (default ON). Any other text — including a value that is not
/// valid UTF-8 — switches it OFF. Rollback: set `FIFI_FLINT_CUDA_FUSED_PUBLISH=0`
/// (or empty, or any non-accepting text) and restart. Armed rounds each
/// published value at the producer kernel with the identical bf16 roundtrip the
/// separate publish kernel applies, so the rolled-back path is bit-identical.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn fused_publish_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        fused_publish_selected(std::env::var_os("FIFI_FLINT_CUDA_FUSED_PUBLISH").as_deref())
    })
}

/// Default-ON selector: unset, `1`, `true` and `on` (after trim + lowercase)
/// arm the switch; empty, any other text and non-UTF-8 values roll it back.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn fused_publish_selected(value: Option<&std::ffi::OsStr>) -> bool {
    default_on_selected(value)
}

/// The load-line token for the fused-publish switch, appended after the
/// resolved-switch suffix so the printed state is the state the kernels read.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn fused_publish_suffix(armed: bool) -> String {
    format!(" fused_publish={}", if armed { "on" } else { "off" })
}

/// FIFI_FLINT_CUDA_KV_DEVICE_F16, read once per process.
/// Values: `1`, `true`, `on` (after trim + lowercase) keep the switch on; unset also
/// arms it (default ON). Any other text — including empty and a value that is not
/// valid UTF-8 — switches it OFF. Rollback: `FIFI_FLINT_CUDA_KV_DEVICE_F16=0` and a
/// restart.
/// Armed, the acceptance guard on the new K and V rows of every layer runs on the
/// device instead of on a host copy of the rows, and the F16 profile also writes
/// the accepted half-words into the cache there. The accepted set and the stored
/// half-words are those of the host path (finite, low 16 bits clear, finite after
/// round-to-nearest-even conversion to F16); only where the guard runs changes.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn kv_device_f16_armed() -> bool {
    use std::sync::OnceLock;
    static ARMED: OnceLock<bool> = OnceLock::new();
    *ARMED.get_or_init(|| {
        kv_device_f16_selected(std::env::var_os("FIFI_FLINT_CUDA_KV_DEVICE_F16").as_deref())
    })
}

/// Resolved-once switch states shared by the executor's load line and the
/// routed kernels: one memoized read per process, so the printed state can
/// never disagree with the behaviour.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn packed_pointwise_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        packed_pointwise_selected(std::env::var_os("FIFI_FLINT_CUDA_PACKED_POINTWISE").as_deref())
    })
}

#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn grouped_gather_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        grouped_gather_selected(std::env::var_os("FIFI_FLINT_CUDA_GROUPED_GATHER").as_deref())
    })
}

/// The CPU-MoE overlap switch, resolved once like the kernel switches above.
#[cfg(all(feature = "native-cuda", target_os = "linux"))]
pub(crate) fn cpu_moe_overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        cpu_moe_overlap_selected(std::env::var_os("FIFI_FLINT_CPU_MOE_OVERLAP").as_deref())
    })
}

/// The q8-tile route's resolved state for a load line: the route only exists
/// for the q8 KV profile, so an f16 engine resolves off WITHOUT reading the
/// knob — `armed` is a closure so the process-wide OnceLock stays untouched
/// until the first q8 engine load actually needs it.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn resolved_q8_tiled(kv_profile: &str, armed: impl FnOnce() -> bool) -> bool {
    kv_profile == "q8" && armed()
}

/// The load-line tail for the four resolved switch states.
#[cfg(any(test, all(feature = "native-cuda", target_os = "linux")))]
pub(crate) fn resolved_switch_suffix(
    packed_pointwise: bool,
    grouped_gather: bool,
    q8_tiled: bool,
    kv_device_f16: bool,
) -> String {
    format!(
        " packed_pointwise={} grouped_gather={} q8_tiled={} kv_device_f16={}",
        if packed_pointwise { "on" } else { "off" },
        if grouped_gather { "on" } else { "off" },
        if q8_tiled { "on" } else { "off" },
        if kv_device_f16 { "on" } else { "off" }
    )
}

pub(crate) fn default_chunk_size() -> u32 {
    8
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlintOptions {
    pub(crate) context_limit: u32,
    pub(crate) kv_profile: String,
    #[serde(default = "default_chunk_size")]
    pub(crate) prefill_chunk_size: u32,
    /// Routed experts of the first N MoE layers run on the CPU (999 = all); default off.
    #[serde(default)]
    #[cfg_attr(
        not(all(feature = "native-cuda", target_os = "linux")),
        allow(dead_code, reason = "read by the CUDA executor")
    )]
    pub(crate) cpu_moe_offload_layers: u32,
    /// Experts per layer served on the CPU instead of the device; default off, exclusive with
    /// `cpu_moe_offload_layers`.
    #[serde(default)]
    #[cfg_attr(
        not(all(feature = "native-cuda", target_os = "linux")),
        allow(dead_code, reason = "read by the CUDA executor")
    )]
    pub(crate) cpu_moe_split_experts: u32,
    /// CPU expert worker threads; 0 picks the default.
    #[serde(default)]
    #[cfg_attr(
        not(all(feature = "native-cuda", target_os = "linux")),
        allow(dead_code, reason = "read by the CUDA executor")
    )]
    pub(crate) cpu_moe_threads: u32,
}

/// Context of a CUDA Flint engine configured without a `flint` object. It needs the expert bank
/// on a 24 GB card; shorter contexts keep every expert on the GPU.
pub(crate) const DEFAULT_CUDA_CONTEXT: u32 = 131_072;

impl FlintOptions {
    /// The options of a CUDA engine with no `flint` object: the f16 profile, never q8.
    pub(crate) fn cuda_default() -> Self {
        Self {
            context_limit: DEFAULT_CUDA_CONTEXT,
            kv_profile: "f16".to_owned(),
            prefill_chunk_size: default_chunk_size(),
            cpu_moe_offload_layers: 0,
            cpu_moe_split_experts: 0,
            cpu_moe_threads: 0,
        }
    }

    pub(crate) fn plan(&self) -> Result<KvPlan, EngError> {
        self.plan_rows(self.prefill_chunk_size as usize)
    }

    /// [`Self::plan`] with rings sized for appends of `rows` rows, which a layer-major jobs pass
    /// makes in one piece. `rows` is at least the prefill chunk size.
    pub(crate) fn plan_rows(&self, rows: usize) -> Result<KvPlan, EngError> {
        let (k_storage, v_storage) = match self.kv_profile.as_str() {
            "f16" => (KvStorage::F16, KvStorage::F16),
            "q8" => (KvStorage::Q8G32, KvStorage::Q8G32),
            // K-only q8, V f16: the mixed discriminator profile.
            "q8k" => (KvStorage::Q8G32, KvStorage::F16),
            _ => {
                return Err(EngError::BadRequest(
                    "Flint CUDA requires explicit kv_profile f16, q8 or q8k".into(),
                ));
            }
        };
        if !(1..=PREFILL_MAX_ROWS).contains(&(self.prefill_chunk_size as usize)) {
            return Err(EngError::BadRequest(
                "Flint CUDA prefill_chunk_size must be in 1..=8".into(),
            ));
        }
        KvPlan::new(
            self.context_limit as usize,
            rows.max(self.prefill_chunk_size as usize),
            true,
        )
        .map(|plan| plan.with_kv_storage(k_storage, v_storage))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_picks_selector_is_default_off_and_strict() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, false),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("On"), true),
            (Some("0"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                device_picks_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!device_picks_selected(Some(OsStr::from_bytes(b"\xff"))));
        }
    }

    #[test]
    fn fused_publish_selector_is_default_on_with_strict_rollback() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                fused_publish_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!fused_publish_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn fused_publish_load_line_token_matches_the_switch() {
        assert_eq!(fused_publish_suffix(false), " fused_publish=off");
        assert_eq!(fused_publish_suffix(true), " fused_publish=on");
    }

    #[test]
    fn cpu_moe_overlap_selector_is_fail_safe_default_off() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, false),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                cpu_moe_overlap_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
    }

    #[test]
    fn packed_pointwise_selector_is_fail_safe_default_on() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                packed_pointwise_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!packed_pointwise_selected(Some(OsStr::from_bytes(
                b"on\xff"
            ))));
        }
    }

    #[test]
    fn q8_tiled_resolves_off_outside_the_q8_profile() {
        assert!(resolved_q8_tiled("q8", || true));
        assert!(!resolved_q8_tiled("q8", || false));
        // Outside the q8 profile the knob is never read at all.
        assert!(!resolved_q8_tiled("f16", || panic!("knob read on f16")));
    }

    #[test]
    fn resolved_switch_suffix_matches_the_load_line_format() {
        assert_eq!(
            resolved_switch_suffix(true, true, true, false),
            " packed_pointwise=on grouped_gather=on q8_tiled=on kv_device_f16=off"
        );
        assert_eq!(
            resolved_switch_suffix(false, true, false, true),
            " packed_pointwise=off grouped_gather=on q8_tiled=off kv_device_f16=on"
        );
        assert_eq!(
            resolved_switch_suffix(false, false, true, false),
            " packed_pointwise=off grouped_gather=off q8_tiled=on kv_device_f16=off"
        );
        assert_eq!(
            resolved_switch_suffix(true, true, false, true),
            " packed_pointwise=on grouped_gather=on q8_tiled=off kv_device_f16=on"
        );
        assert_eq!(
            resolved_switch_suffix(true, false, true, false),
            " packed_pointwise=on grouped_gather=off q8_tiled=on kv_device_f16=off"
        );
    }

    #[test]
    fn grouped_gather_selector_table() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("0"), false),
            (Some(""), false),
            (Some("off"), false),
            (Some(" \t\n"), false),
            (Some("2"), false),
            (Some("yes"), false),
        ] {
            assert_eq!(
                grouped_gather_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!grouped_gather_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn layer_major_env_is_strict_and_unset_leaves_the_choice_to_the_engine() {
        use std::ffi::OsStr;

        // Unset is no choice; only the exact trimmed "off", "1" and "jobs" name a pass.
        assert_eq!(layer_major_from_env(None).unwrap(), None);
        for value in ["off", " off\n"] {
            assert_eq!(
                layer_major_from_env(Some(OsStr::new(value))).unwrap(),
                Some(LayerMajor::Off),
                "{value:?}"
            );
        }
        for value in ["1", " 1 ", "\u{2003}1"] {
            assert_eq!(
                layer_major_from_env(Some(OsStr::new(value))).unwrap(),
                Some(LayerMajor::Tiles),
                "{value:?}"
            );
        }
        for value in ["jobs", " jobs\n"] {
            assert_eq!(
                layer_major_from_env(Some(OsStr::new(value))).unwrap(),
                Some(LayerMajor::Jobs),
                "{value:?}"
            );
        }
        // Everything else is an open-time error, never a silent fallback, including the
        // near-miss forms an operator might type for "off".
        for value in [
            "0", "", " ", "true", "on", "yes", "1.0", "01", "+1", "-1", "one", "l", "Jobs", "job",
            "jobs1", "Off", "OFF", "false", "no", "none",
        ] {
            assert!(
                layer_major_from_env(Some(OsStr::new(value))).is_err(),
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(layer_major_from_env(Some(OsStr::from_bytes(b"1\xff"))).is_err());
        }
    }

    #[test]
    fn a_jobs_plan_sizes_the_sliding_rings_for_a_piece_and_the_default_for_a_chunk() {
        let options = FlintOptions::cuda_default();
        assert_eq!(options.prefill_chunk_size, 8);
        let narrow = options.plan().unwrap();
        let piece = options.plan_rows(1024).unwrap();
        // Layer 2 slides over a 513-row window: a ring holds the window plus the rows of one append
        // beyond the first.
        assert_eq!(narrow.cache_rows(2).unwrap(), 513 + 7);
        assert_eq!(piece.cache_rows(2).unwrap(), 513 + 1023);
        // The plan never goes below the prefill chunk, and a plan for the chunk is the default.
        assert_eq!(options.plan_rows(1).unwrap(), narrow);
        assert_eq!(options.plan_rows(8).unwrap(), narrow);
        // A layer that attends to the whole context keeps its full capacity either way.
        let full = crate::family::flint::config::FULL_LAYERS[0] as usize;
        assert_eq!(
            piece.cache_rows(full).unwrap(),
            narrow.cache_rows(full).unwrap()
        );
        // A chunk size outside 1..=8 stays refused whatever the piece.
        let bad = FlintOptions {
            prefill_chunk_size: 9,
            ..options
        };
        assert!(bad.plan_rows(1024).is_err());
    }

    #[test]
    fn q8_tiled_selector_is_default_on_with_strict_rollback() {
        use std::ffi::OsStr;

        // Unset arms the route (default ON). The accept list is identical to
        // the default-OFF switches; only the unset case differs.
        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                q8_tiled_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!q8_tiled_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn token_health_selector_is_default_on_with_strict_rollback() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" 1 "), true),
            (Some("true"), true),
            (Some("ON"), true),
            (Some("\tOn\r\n"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                token_health_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!token_health_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn kv_device_f16_selector_is_default_on_with_strict_rollback() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" TrUe "), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some("false"), false),
            (Some(""), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
        ] {
            assert_eq!(
                kv_device_f16_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!kv_device_f16_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn late_health_selector_is_fail_safe_default_on() {
        use std::ffi::OsStr;

        for (value, expected) in [
            (None, true),
            (Some("1"), true),
            (Some(" 1 "), true),
            (Some("true"), true),
            (Some("TRUE"), true),
            (Some(" TrUe "), true),
            (Some("on"), true),
            (Some("ON"), true),
            (Some("\tOn\r\n"), true),
            (Some("\u{2003}On\u{00a0}"), true),
            (Some("0"), false),
            (Some("off"), false),
            (Some(" OFF "), false),
            (Some("false"), false),
            (Some("FALSE"), false),
            (Some(""), false),
            (Some(" \t\n"), false),
            (Some("yes"), false),
            (Some("2"), false),
            (Some("true!"), false),
            (Some("on\0"), false),
            (Some("ＴＲＵＥ"), false),
            (Some("oＮ"), false),
        ] {
            assert_eq!(
                late_health_selected(value.map(OsStr::new)),
                expected,
                "{value:?}"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(!late_health_selected(Some(OsStr::from_bytes(b"on\xff"))));
        }
    }

    #[test]
    fn cuda_profile_requires_explicit_supported_format_and_valid_context() {
        for kv_profile in ["f16", "q8", "q8k"] {
            for context in [0, 262_145] {
                assert!(
                    FlintOptions {
                        context_limit: context,
                        kv_profile: kv_profile.into(),
                        prefill_chunk_size: 1,

                        cpu_moe_offload_layers: 0,
                        cpu_moe_split_experts: 0,
                        cpu_moe_threads: 0,
                    }
                    .plan()
                    .is_err()
                );
            }
        }
        for kv_profile in ["q4", "f32", "", "F16", "Q8"] {
            assert!(
                FlintOptions {
                    context_limit: 128,
                    kv_profile: kv_profile.into(),
                    prefill_chunk_size: 1,

                    cpu_moe_offload_layers: 0,
                    cpu_moe_split_experts: 0,
                    cpu_moe_threads: 0,
                }
                .plan()
                .is_err()
            );
        }
        let plan = FlintOptions {
            context_limit: 8192,
            kv_profile: "f16".into(),
            prefill_chunk_size: 1,

            cpu_moe_offload_layers: 0,
            cpu_moe_split_experts: 0,
            cpu_moe_threads: 0,
        }
        .plan()
        .unwrap();
        assert_eq!(plan.context_rows(), 8192);
        assert_eq!(plan.cache_rows(0).unwrap(), 8192);
        assert_eq!(plan.cache_rows(2).unwrap(), 513);
        assert_eq!(plan.bytes(2048).unwrap(), (14 * 8192 + 35 * 513) * 2048);
    }

    #[test]
    fn chunk_option_defaults_to_eight_and_admits_bounded_ring() {
        assert_eq!(
            PREFILL_MAX_ROWS, 8,
            "raising the cap changes numeric kernels"
        );
        let serial: FlintOptions = serde_json::from_str(
            r#"{"context_limit":4096,"kv_profile":"f16","prefill_chunk_size":1}"#,
        )
        .unwrap();
        assert_eq!(serial.prefill_chunk_size, 1);
        assert_eq!(serial.plan().unwrap().cache_rows(2).unwrap(), 513);
        let batch: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        assert_eq!(batch.prefill_chunk_size, 8);
        assert_eq!(batch.plan().unwrap().cache_rows(2).unwrap(), 520);
        for rows in 1..=8 {
            let options = FlintOptions {
                prefill_chunk_size: rows,
                ..serial.clone()
            };
            assert_eq!(
                options.plan().unwrap().cache_rows(2).unwrap(),
                512 + rows as usize
            );
        }
        assert_eq!(
            batch.plan().unwrap().bytes(2048).unwrap()
                - serial.plan().unwrap().bytes(2048).unwrap(),
            501_760
        );
        for rows in [0, 9, u32::MAX] {
            assert!(
                FlintOptions {
                    prefill_chunk_size: rows,
                    ..serial.clone()
                }
                .plan()
                .is_err()
            );
        }
    }

    #[test]
    fn cuda_q8_profile_is_opt_in_and_charges_payload_and_scales() {
        use crate::family::flint::kv_plan::KvStorage;

        let q8: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"q8"}"#).unwrap();
        assert_eq!(q8.prefill_chunk_size, 8);
        let plan = q8.plan().unwrap();
        assert_eq!(plan.storage(), KvStorage::Q8G32);
        assert_eq!(plan.storage().row_bytes(), 1088);
        assert_eq!(plan.bytes(plan.storage().row_bytes()).unwrap(), 82_191_872);
        assert_eq!(plan.cache_rows(0).unwrap(), 4096);
        assert_eq!(plan.cache_rows(2).unwrap(), 520);

        let f16: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        let control = f16.plan().unwrap();
        assert_eq!(control.storage(), KvStorage::F16);
        assert_eq!(
            control.bytes(control.storage().row_bytes()).unwrap(),
            154_714_112
        );
    }

    #[test]
    fn cuda_default_is_131072_tokens_of_f16_and_a_valid_plan() {
        use crate::family::flint::kv_plan::KvStorage;

        let options = FlintOptions::cuda_default();
        assert_eq!(options.context_limit, 131_072);
        assert_eq!(options.kv_profile, "f16");
        assert_eq!(options.prefill_chunk_size, 8);
        let plan = options.plan().unwrap();
        assert_eq!(plan.storage(), KvStorage::F16);
        assert_eq!(plan.context_rows(), 131_072);
    }

    #[test]
    fn cuda_q8k_profile_quantizes_k_and_keeps_v_f16() {
        use crate::family::flint::kv_plan::KvStorage;

        let q8k: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"q8k"}"#).unwrap();
        let plan = q8k.plan().unwrap();
        assert_eq!(plan.k_storage(), KvStorage::Q8G32);
        assert_eq!(plan.v_storage(), KvStorage::F16);
        // Per-plane per-position charge: K 512 payload + 32 scales, V 1024 payload.
        assert_eq!(plan.per_plane_row_bytes(), (544, 1024));
        let (k, v) = plan.per_plane_row_bytes();
        // The summed charge sits between the uniform profiles.
        let q8: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"q8"}"#).unwrap();
        let f16: FlintOptions =
            serde_json::from_str(r#"{"context_limit":4096,"kv_profile":"f16"}"#).unwrap();
        let (q8_total, f16_total) = (
            q8.plan().unwrap().storage().row_bytes(),
            f16.plan().unwrap().storage().row_bytes(),
        );
        assert_eq!(k + v, 1568);
        assert!(q8_total < k + v && k + v < f16_total);
        // Same cache geometry as the uniform profiles: only the byte charge changes.
        assert_eq!(plan.cache_rows(2).unwrap(), 520);
        assert_eq!(
            plan.cache_rows(2).unwrap(),
            q8.plan().unwrap().cache_rows(2).unwrap()
        );
    }

    #[test]
    fn cuda_options_reject_missing_or_unknown_profile_keys() {
        for text in [
            r#"{"context_limit":128}"#,
            r#"{"context_limit":128,"kv_profile":"f16","typo":1}"#,
        ] {
            assert!(serde_json::from_str::<FlintOptions>(text).is_err());
        }
    }

    /// The per-plane admission charge (`reserve_with` sums `k_row + v_row`)
    /// must equal the legacy single-storage charge main used, so the default
    /// f16 (and q8) admission figures are unchanged by the mixed-profile plan.
    #[test]
    fn uniform_profiles_keep_the_legacy_admission_charge() {
        for (profile, context) in [
            ("f16", 4096u32),
            ("f16", 16384),
            ("f16", 32768),
            ("q8", 4096),
            ("q8", 32768),
        ] {
            let options: FlintOptions = serde_json::from_str(&format!(
                r#"{{"context_limit":{context},"kv_profile":"{profile}"}}"#
            ))
            .unwrap();
            let plan = options.plan().unwrap();
            let (k, v) = plan.per_plane_row_bytes();
            assert_eq!(k + v, plan.storage().row_bytes(), "{profile}@{context}");
        }
    }
}
