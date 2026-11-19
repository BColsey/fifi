//! Startup working-set budget for an opted-in Apple Silicon model.

use crate::{EngError, family::device_budget::DeviceReserve};

/// Family reserves leave this much room for memory outside their tensor estimates.
#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) const PRESSURE_MARGIN: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DeviceBudget {
    bytes: u64,
}

impl DeviceBudget {
    pub(crate) fn new(bytes: u64) -> Result<Self, EngError> {
        if bytes == 0 {
            return Err(EngError::MemoryPlan(
                "Metal did not report a recommended working set".into(),
            ));
        }
        Ok(Self { bytes })
    }

    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn detect() -> Result<Option<Self>, EngError> {
        if enabled() {
            Self::new(crate::kernels::MlxKernels::recommended_working_set_bytes()).map(Some)
        } else {
            Ok(None)
        }
    }

    pub(crate) fn bytes(self) -> u64 {
        self.bytes
    }

    /// A configured upper bound can make the working set smaller.
    pub(crate) fn prefill(self, configured: Option<u64>) -> u64 {
        self.bytes.min(configured.unwrap_or(self.bytes))
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
        self.bytes.checked_sub(total).ok_or_else(|| {
            let terms = reserve.charges().iter().map(|(name, bytes)| format!("{name}={bytes}"))
                .collect::<Vec<_>>().join(" ");
            EngError::MemoryPlan(format!(
                "{family} requires {total} bytes for the resident model and session reserve. Metal recommends {} bytes ({terms})",
                self.bytes
            ))
        })
    }
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn enabled() -> bool {
    std::env::var_os("FIFI_MLX_AUTO_BUDGET").is_some_and(|value| value == "1")
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
}
