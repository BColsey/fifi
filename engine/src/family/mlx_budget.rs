//! Startup working-set budget for an opted-in Apple Silicon model.

use std::fmt;

use crate::{
    EngError,
    family::{device_budget::DeviceReserve, mem::MacMemory},
};

/// Family reserves leave this much room for memory outside their tensor estimates.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const PRESSURE_MARGIN: u64 = crate::family::mem::DEVICE.unified_pressure_margin;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceBudget {
    recommended: u64,
    ceiling: Option<u64>,
}

impl DeviceBudget {
    pub(crate) fn new(bytes: u64) -> Result<Self, EngError> {
        if bytes == 0 {
            return Err(EngError::MemoryPlan(
                "Metal did not report a recommended working set".into(),
            ));
        }
        Ok(Self {
            recommended: bytes,
            ceiling: None,
        })
    }

    #[cfg(any(
        test,
        all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")
    ))]
    /// The budget of `memory`, the one value the plan and the load both read: Metal reports the
    /// working set and the ceiling field holds the rest of the rule (`MacMemory::budget`).
    pub(crate) fn from_memory(memory: MacMemory) -> Result<Self, EngError> {
        let device = Self::new(memory.metal_bytes.unwrap_or(0))?;
        Ok(Self {
            ceiling: Some(memory.budget()),
            ..device
        })
    }

    /// The budget of a start that opted in (`FIFI_MLX_AUTO_BUDGET=1` or a ceiling variable).
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect() -> Result<Option<Self>, EngError> {
        if !enabled() {
            return Ok(None);
        }
        crate::family::mem::ceiling_from_env()?;
        let memory = crate::family::mem::mac_memory().ok_or_else(|| {
            EngError::MemoryPlan("the unified memory of this host is unknown".into())
        })?;
        Self::from_memory(memory).map(Some)
    }

    /// The budget of a start that sets none and whose resident model does not fit the host
    /// memory: the host budget, which makes the family choose by fit (streaming its routed
    /// experts) instead of importing everything. `None` when the model fits or the memory is
    /// unknown, so a start that fits behaves as it always did.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect_or_implied(model_dir: &std::path::Path) -> Result<Option<Self>, EngError> {
        if let Some(device) = Self::detect()? {
            return Ok(Some(device));
        }
        let (Some(memory), Some(model)) = (
            crate::family::mem::mac_memory(),
            crate::family::mem::shard_bytes(model_dir),
        ) else {
            return Ok(None);
        };
        let Some(budget) = memory.implied_budget(model) else {
            return Ok(None);
        };
        eprintln!(
            "mlx: the resident model needs {model} bytes and the budget is {budget}; sizing from the budget"
        );
        Self::from_memory(memory).map(Some)
    }

    pub(crate) fn bytes(self) -> u64 {
        self.recommended.min(self.ceiling.unwrap_or(u64::MAX))
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn ceiling_report(self) -> super::mlx_report::MlxDeviceCeiling {
        super::mlx_report::MlxDeviceCeiling {
            metal_working_set_bytes: self.recommended,
            effective_device_budget_bytes: self.bytes(),
            configured_device_ceiling_bytes: self.ceiling,
        }
    }

    /// A configured upper bound can make the working set smaller.
    pub(crate) fn prefill(self, configured: Option<u64>) -> u64 {
        self.bytes().min(configured.unwrap_or(u64::MAX))
    }

    /// Refuse a model when its mandatory packed payloads alone exceed the
    /// budget left for weights. The loader still charges the full resident
    /// tensors and staging pages after import.
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn check_packed_payload_floor(
        self,
        family: &str,
        ckpt: &crate::weights::StreamedCheckpoint,
        reserve: u64,
        is_resident: impl Fn(&str) -> bool,
    ) -> Result<(), EngError> {
        let weights = packed_payload_floor(ckpt, is_resident).ok_or_else(|| {
            EngError::MemoryPlan(format!("{family} packed payload bytes overflow"))
        })?;
        self.remaining(
            family,
            &[
                ("packed_weight_payload_floor_bytes", weights),
                ("session_reserve_bytes", reserve),
            ],
        )?;
        Ok(())
    }

    /// The space left after the resident charges and the family's reserve.
    pub(crate) fn remaining(
        self,
        family: &str,
        charges: &[(&'static str, u64)],
    ) -> Result<u64, EngError> {
        let reserve = charges
            .iter()
            .fold(DeviceReserve::new(0), |plan, &(label, bytes)| {
                plan.push(label, bytes)
            });
        let total = reserve.checked_total().ok_or_else(|| {
            EngError::MemoryPlan(format!("{family} MLX startup charges overflow u64"))
        })?;
        self.bytes().checked_sub(total).ok_or_else(|| {
            let terms = reserve.charges().iter().map(|(name, bytes)| format!("{name}={bytes}"))
                .collect::<Vec<_>>().join(" ");
            EngError::MemoryPlan(format!(
                "{family} requires {total} bytes for the resident model and session reserve. Metal recommends {} bytes, with {} bytes admitted ({terms})",
                self.recommended,
                self.bytes(),
            ))
        })
    }
}

impl fmt::Display for DeviceBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "metal_working_set_bytes={} effective_device_budget_bytes={}",
            self.recommended,
            self.bytes()
        )?;
        if let Some(ceiling) = self.ceiling {
            write!(f, " configured_device_ceiling_bytes={ceiling}")?;
        }
        Ok(())
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enabled() -> bool {
    std::env::var_os("FIFI_MLX_AUTO_BUDGET").is_some_and(|value| value == "1")
        || std::env::var_os(crate::family::mem::CEILING_MIB).is_some()
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn configured_ceiling() -> Result<Option<u64>, EngError> {
    crate::family::mem::ceiling_from_env()
}

/// The packed `.weight` payload bytes that `is_resident` keeps in memory, read from the
/// checkpoint headers alone; `None` on overflow.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(super) fn packed_payload_floor(
    ckpt: &crate::weights::StreamedCheckpoint,
    is_resident: impl Fn(&str) -> bool,
) -> Option<u64> {
    ckpt.names()
        .filter(|key| key.ends_with(".weight") && is_resident(key))
        .try_fold(0u64, |total, key| {
            total.checked_add(ckpt.stored_byte_len(key)?)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::mem::parse_ceiling;
    use std::ffi::OsStr;

    #[test]
    fn metal_working_set_caps_explicit_budget_and_checks_base_charges() {
        let small = DeviceBudget::new(12 << 30).unwrap();
        let large = DeviceBudget::new(24 << 30).unwrap();
        assert_eq!(small.prefill(None), small.bytes());
        assert_eq!(small.prefill(Some(16 << 30)), 12 << 30);
        assert_eq!(large.prefill(Some(16 << 30)), 16 << 30);
        let charges = [("weights", 11 << 30), ("session", 2 << 30)];
        assert!(matches!(
            small.remaining("fixture", &charges),
            Err(EngError::MemoryPlan(_))
        ));
        assert_eq!(large.remaining("fixture", &charges).unwrap(), 11 << 30);
        assert!(
            large
                .remaining("fixture", &[("one", u64::MAX), ("two", 1)])
                .is_err()
        );
        assert!(DeviceBudget::new(0).is_err());
    }
    #[test]
    fn explicit_ceiling_bounds_every_family_without_overriding_metal() {
        let ceiling = |mib| Some(parse_ceiling(OsStr::new(mib)).unwrap());
        let device = DeviceBudget::new(12 << 30).unwrap();
        let constrained = DeviceBudget {
            ceiling: ceiling("4096"),
            ..device
        };
        assert_eq!(constrained.bytes(), 4 << 30);
        assert_eq!(constrained.prefill(Some(2 << 30)), 2 << 30);
        assert_eq!(constrained.prefill(Some(8 << 30)), 4 << 30);
        assert!(
            constrained
                .remaining("fixture", &[("weights", 5 << 30)])
                .is_err()
        );
        let larger = DeviceBudget {
            ceiling: ceiling("16384"),
            ..device
        };
        assert_eq!(larger.bytes(), device.bytes());
        for invalid in ["", "0", "-1", "4.5", "17592186044416"] {
            assert!(parse_ceiling(OsStr::new(invalid)).is_err(), "{invalid}");
        }
    }
}
