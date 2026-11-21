//! Startup working-set budget for an opted-in Apple Silicon model.

use std::{ffi::OsStr, fmt};

use crate::{EngError, family::device_budget::DeviceReserve};

const CEILING_MIB: &str = "FIFI_MLX_WORKING_SET_MIB";

/// Family reserves leave this much room for memory outside their tensor estimates.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const PRESSURE_MARGIN: u64 = 1 << 30;

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

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect() -> Result<Option<Self>, EngError> {
        Self::detect_with_default(false)
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect_with_default(auto_default: bool) -> Result<Option<Self>, EngError> {
        if enabled_with_default(auto_default) {
            let ceiling = std::env::var_os(CEILING_MIB);
            let ceiling = ceiling.as_deref().map(parse_ceiling).transpose()?;
            let device = Self::new(crate::kernels::MlxKernels::recommended_working_set_bytes())?;
            Ok(Some(Self { ceiling, ..device }))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn bytes(self) -> u64 {
        self.recommended.min(self.ceiling.unwrap_or(u64::MAX))
    }

    /// A configured upper bound can make the working set smaller.
    pub(crate) fn prefill(self, configured: Option<u64>) -> u64 {
        self.bytes().min(configured.unwrap_or(u64::MAX))
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

fn parse_ceiling(value: &OsStr) -> Result<u64, EngError> {
    value
        .to_str()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|mib| mib.checked_mul(1 << 20))
        .filter(|&bytes| bytes != 0)
        .ok_or_else(|| {
            EngError::BadRequest(format!(
                "{CEILING_MIB} must be a positive whole number of MiB that fits in u64 bytes"
            ))
        })
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enabled() -> bool {
    enabled_with_default(false)
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enabled_with_default(auto_default: bool) -> bool {
    std::env::var_os("FIFI_MLX_AUTO_BUDGET").map_or(auto_default, |value| value == "1")
        || std::env::var_os(CEILING_MIB).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

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
