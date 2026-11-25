//! The refusals of the GGUF start that need no device: pure decisions between the entry's
//! options, the environment and the model, so they are checked on the host.

use crate::EngError;

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF start: {}", detail.as_ref()))
}

/// A cache with fewer slots per layer than the router picks fails every token: a pick without a
/// slot finds no victim to evict. Refused at the start, before any device work, naming both.
pub(in crate::family::qwen::flash_next) fn check_slots(
    slots: usize,
    top_k: usize,
) -> Result<(), EngError> {
    if slots < top_k {
        return Err(refuse(format!(
            "the card holds {slots} expert slots per layer, fewer than the {top_k} the router picks"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cache_with_fewer_slots_than_the_router_picks_is_refused() {
        let error = check_slots(8, 10).expect_err("eight slots cannot serve ten picks");
        assert!(format!("{error:?}").contains("8 expert slots"), "{error:?}");
        assert!(format!("{error:?}").contains("the 10 the router picks"), "{error:?}");
        assert!(check_slots(10, 10).is_ok());
    }
}
