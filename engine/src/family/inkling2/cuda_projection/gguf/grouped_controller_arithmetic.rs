//! Arithmetic selection for the admitted grouped prefill controller chain only.

use std::{ffi::OsStr, sync::OnceLock};

use crate::kernels::cuda::resident::inkling_controller::Arithmetic;

pub(super) fn select() -> Arithmetic {
    static MODE: OnceLock<Arithmetic> = OnceLock::new();
    *MODE.get_or_init(|| {
        from_flag(std::env::var_os("FIFI_INKLING_CUDA_CONTROLLER_CPU_PARITY").as_deref())
    })
}

fn from_flag(value: Option<&OsStr>) -> Arithmetic {
    if value == Some(OsStr::new("1")) {
        Arithmetic::CpuParity
    } else {
        Arithmetic::Generic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_opt_in_selects_cpu_parity() {
        assert_eq!(from_flag(Some(OsStr::new("1"))), Arithmetic::CpuParity);
    }

    #[test]
    fn unset_or_other_values_keep_generic_arithmetic() {
        assert_eq!(from_flag(None), Arithmetic::Generic);
        for value in ["", "0", "true", "01", " 1", "1 "] {
            assert_eq!(from_flag(Some(OsStr::new(value))), Arithmetic::Generic);
        }
    }
}
