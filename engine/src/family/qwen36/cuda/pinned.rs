//! The pinned model's host-side facts shared by the stage 1 gate and the serve start: the rotary
//! dimension, the free-after-load floor of the memory admission, and the head width of the pinned
//! model. The gate's qualified loop and the server's CUDA start read the SAME constants, so a
//! divergence between the two admission checks is impossible. Pure host arithmetic: no context, no
//! device, so a missing file or a bad rule fails in seconds, not after a 20 GB upload.

use super::super::host::HostShape;
use super::dims::StepDims;
use crate::{
    EngError,
    family::{QWEN3_6_35B_A3B, gguf_model::Inventory},
};

/// Rotary dimension of the pinned model (head_dim 256 times the partial-rotary factor 0.25);
/// the GGUF carries no config.json, and the resident-step test fixtures name the same 64.
/// Aliases the ungated single pinned fact in `settings.rs`.
pub(super) const ROTARY_DIM: u32 = super::super::settings::ROTARY_DIM;
/// Less than this many bytes free after the load is a typed refusal, not an error.
/// The default floor; `min_free_after_load` carries the `Q36_CUDA_MIN_FREE_MIB` override of it.
const MIN_FREE_AFTER_LOAD_DEFAULT: usize = 1 << 30;
const MIN_FREE_MIB_VAR: &str = "Q36_CUDA_MIN_FREE_MIB";
const MIN_FREE_MIB_FLOOR: usize = 256;
const MIN_FREE_MIB_CAP: usize = 4096;

/// `Q36_CUDA_MIN_FREE_MIB` in MiB: 256 to 4096, anything else (including a non-number) is a
/// typed refusal, so a misspelled promise cannot silently widen or narrow the floor. `None`
/// keeps the 1 GiB default.
fn min_free_mib(value: Option<&std::ffi::OsStr>) -> Result<usize, EngError> {
    let Some(value) = value else {
        return Ok(MIN_FREE_AFTER_LOAD_DEFAULT);
    };
    let mib = value
        .to_str()
        .and_then(|text| text.parse::<usize>().ok())
        .filter(|mib| (MIN_FREE_MIB_FLOOR..=MIN_FREE_MIB_CAP).contains(mib))
        .ok_or_else(|| {
            EngError::BadRequest(format!(
                "{MIN_FREE_MIB_VAR} must be {MIN_FREE_MIB_FLOOR} to {MIN_FREE_MIB_CAP} MiB"
            ))
        })?;
    mib.checked_mul(1 << 20).ok_or_else(|| {
        EngError::BadRequest(format!(
            "{MIN_FREE_MIB_VAR} must be {MIN_FREE_MIB_FLOOR} to {MIN_FREE_MIB_CAP} MiB"
        ))
    })
}

/// The free-after-load floor of the memory admission, the same bytes for the gate's qualified
/// loop and the server's CUDA start: the default, or the first read of `Q36_CUDA_MIN_FREE_MIB`
/// cached for the process. Parsed once so the two admission checks cannot diverge mid-run.
pub(super) fn min_free_after_load() -> Result<usize, EngError> {
    static FLOOR: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let parsed =
        FLOOR.get_or_init(|| min_free_mib(std::env::var_os(MIN_FREE_MIB_VAR).as_deref()).ok());
    parsed.ok_or_else(|| {
        EngError::BadRequest(format!(
            "{MIN_FREE_MIB_VAR} must be {MIN_FREE_MIB_FLOOR} to {MIN_FREE_MIB_CAP} MiB"
        ))
    })
}

/// The default floor in bytes: 1 GiB. The settings constants test pins it.
#[cfg(test)]
pub(super) fn min_free_after_load_default() -> usize {
    MIN_FREE_AFTER_LOAD_DEFAULT
}

pub(super) fn head_rows_of() -> Result<usize, EngError> {
    let inventory = Inventory::for_spec(&QWEN3_6_35B_A3B)?;
    let shape = HostShape::from_spec(&QWEN3_6_35B_A3B, ROTARY_DIM)?;
    Ok(StepDims::new(&shape, &inventory)?.shape.vocab)
}

#[cfg(test)]
mod floor_tests {
    use super::*;

    #[test]
    fn the_floor_keeps_the_default_and_parses_the_mib_range() {
        assert_eq!(min_free_mib(None), Ok(MIN_FREE_AFTER_LOAD_DEFAULT));
        assert_eq!(min_free_mib(Some("512".as_ref())), Ok(512 << 20));
        assert_eq!(min_free_mib(Some("256".as_ref())), Ok(256 << 20));
        assert_eq!(min_free_mib(Some("4096".as_ref())), Ok(4096 << 20));
        for bad in ["255", "4097", "0", "1GiB", "", "512.0"] {
            assert!(min_free_mib(Some(bad.as_ref())).is_err(), "{bad}");
        }
    }
}
