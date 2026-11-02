//! Hardware capability reporting and model preflight: what this build can
//! serve, on this machine, before any model is loaded.
//!
//! This module only composes facts the engine already measures or checks
//! elsewhere (device queries in `kernels::cuda`, the family settings
//! validators, the tokenizer pins, the KV sizing authority, the shared
//! `MemoryPolicy`). It introduces no second memory planner and no second
//! compatibility rule: where a load-time check exists, preflight calls the
//! same function the loader calls, so the two cannot drift apart.
//!
//! Evidence classes, the rule of `family::qwen36::cuda::placement` applied
//! here too: every reported number is labeled. `Measured` was read from the
//! machine just now (free VRAM, available RAM, file sizes). `Property`
//! comes from a device or checkpoint descriptor. `CompiledIn` is a fact
//! about this binary. `Derived` is arithmetic over the above (an estimate),
//! and the report says what the arithmetic leaves out. Nothing here is a
//! benchmark result, and nothing is presented as one.
//!
//! Probing is for short-lived diagnostic invocations (`enginesrv doctor`,
//! `enginesrv check`). The CUDA queries latch the backend terminal on a
//! native failure for the rest of the process (`kernels::cuda`), so a
//! serving process must not call `probe`; it already reads the same facts
//! through its own context at load time.

pub mod cli;
pub mod inspect;
pub mod preflight;
pub mod probe;

use serde::Serialize;

/// One probed value, or why it could not be probed. Unavailable is a
/// first-class answer: a report states it instead of guessing.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "evidence", rename_all = "snake_case")]
pub enum Probe<T> {
    /// Read from the machine during this probe.
    Measured(T),
    /// The probe could not produce a value; the string says why.
    Unavailable(String),
}

impl<T> Probe<T> {
    pub fn as_ref(&self) -> Option<&T> {
        match self {
            Probe::Measured(value) => Some(value),
            Probe::Unavailable(_) => None,
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Probe::Measured(_) => None,
            Probe::Unavailable(reason) => Some(reason),
        }
    }
}

/// The one-line rendering of an engine error. `EngError` has no
/// `Display` by design (each surface renders it its own way, e.g.
/// `server::run::describe`); diagnostics use the message itself, with the
/// backend operation kept because it names the failing subsystem.
pub(crate) fn describe_error(error: &crate::EngError) -> String {
    error.describe()
}

/// GiB with two decimals, the unit every rendered report uses for memory.
pub fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// MiB with no decimals, for smaller quantities such as fixed state.
pub fn mib(bytes: u64) -> String {
    format!("{:.0} MiB", bytes as f64 / (1u64 << 20) as f64)
}

#[cfg(test)]
mod tests {
    use super::{Probe, gib, mib};

    #[test]
    fn probe_reports_value_or_reason_never_both() {
        let measured: Probe<u64> = Probe::Measured(7);
        assert_eq!(measured.as_ref(), Some(&7));
        assert_eq!(measured.reason(), None);
        let missing: Probe<u64> = Probe::Unavailable("no device".into());
        assert_eq!(missing.as_ref(), None);
        assert_eq!(missing.reason(), Some("no device"));
    }

    #[test]
    fn byte_rendering_uses_binary_units() {
        assert_eq!(gib(1 << 30), "1.00 GiB");
        assert_eq!(gib(24_576 * 1024 * 1024), "24.00 GiB");
        assert_eq!(mib(65_863_680), "63 MiB");
    }
}
