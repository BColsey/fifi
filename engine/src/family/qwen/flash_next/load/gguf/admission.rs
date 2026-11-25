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

/// The entry must set `cuda_decode` to true for the GGUF start, exactly as the pack start
/// requires it: the start promoted an unset value to true and served a card the entry never asked
/// for. The refusal is the pack start's text.
pub(in crate::family::qwen::flash_next) fn check_cuda_decode(
    cuda_decode: Option<bool>,
) -> Result<(), EngError> {
    if cuda_decode != Some(true) {
        return Err(EngError::BadRequest(
            "engine config: the Flash-Next CUDA decoder is off; set flash_next.cuda_decode to true to serve on a card (the MLX backend needs --features native-mlx on Apple Silicon)"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_or_false_cuda_decode_is_refused_like_the_pack_start() {
        for value in [None, Some(false)] {
            let Err(EngError::BadRequest(text)) = check_cuda_decode(value) else {
                panic!("{value:?} must be refused");
            };
            assert_eq!(
                text,
                "engine config: the Flash-Next CUDA decoder is off; set flash_next.cuda_decode to true to serve on a card (the MLX backend needs --features native-mlx on Apple Silicon)"
            );
        }
        assert!(check_cuda_decode(Some(true)).is_ok());
    }

    #[test]
    fn a_cache_with_fewer_slots_than_the_router_picks_is_refused() {
        let error = check_slots(8, 10).expect_err("eight slots cannot serve ten picks");
        assert!(format!("{error:?}").contains("8 expert slots"), "{error:?}");
        assert!(format!("{error:?}").contains("the 10 the router picks"), "{error:?}");
        assert!(check_slots(10, 10).is_ok());
    }
}
