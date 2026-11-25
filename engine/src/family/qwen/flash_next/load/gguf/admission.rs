//! The parts of the GGUF start that need no device: pure decisions between the entry's
//! options, the environment and the model, and the digests of the files it serves, so they are
//! checked on the host.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::EngError;

fn refuse(detail: impl AsRef<str>) -> EngError {
    EngError::Checkpoint(format!("Flash-Next GGUF start: {}", detail.as_ref()))
}

/// The digest of a tokenizer file of `dir`: an absent file pins zeros and is reported by the
/// load; one that cannot be read is an error, not a zero digest that would pass for the empty
/// file and match any other unreadable one.
fn file_digest(path: &Path) -> Result<[u8; 32], EngError> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Sha256::digest(bytes).into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok([0u8; 32]),
        Err(error) => Err(refuse(format!(
            "{} cannot be read: {error}",
            path.display()
        ))),
    }
}

/// The pin of the tokenizer files found in `dir`: their own digests, so the load matches and
/// logs nothing, and the vocabulary the model's head holds.
pub(in crate::family::qwen::flash_next) fn observed_pin(
    dir: &Path,
    vocab: u32,
) -> Result<crate::tokenizer::TokenizerPin, EngError> {
    let digest = |name: &str| file_digest(&dir.join(name));
    Ok(crate::tokenizer::TokenizerPin {
        tokenizer_json: digest("tokenizer.json")?,
        tokenizer_config_json: digest("tokenizer_config.json")?,
        chat_template_jinja: digest("chat_template_jinja")?,
        generation_config_json: digest("generation_config.json")?,
        vocab_size: vocab,
    })
}

/// The tokenizer term of the prefix identity: the digest of the served `tokenizer.json`, whose
/// read error propagates instead of becoming the digest of the empty file.
pub(in crate::family::qwen::flash_next) fn tokenizer_digest(
    dir: &Path,
) -> Result<[u8; 32], EngError> {
    std::fs::read(dir.join("tokenizer.json"))
        .map(|bytes| Sha256::digest(bytes).into())
        .map_err(|error| {
            refuse(format!(
                "{} cannot be read: {error}",
                dir.join("tokenizer.json").display()
            ))
        })
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

/// The GGUF start is new behaviour whose device tests are not qualified: it runs only when the
/// switch `FIFI_FLASH_NEXT_GGUF` equals exactly `1`. Without it the start refuses as it did
/// before it read GGUF weights, and names the switch.
pub(in crate::family::qwen::flash_next) fn check_opt_in(
    opt_in: bool,
    first_path: &std::path::Path,
) -> Result<(), EngError> {
    if !opt_in {
        return Err(refuse(format!(
            "{} is planned and every matrix type has a decode kernel, but the GGUF start is \
             not qualified yet: it runs only when FIFI_FLASH_NEXT_GGUF equals 1",
            first_path.display()
        )));
    }
    Ok(())
}

/// The admission of the GGUF start: `cuda_decode` set to true, then the switch. Both are read by
/// the caller (the entry and the environment) and passed in, so tests never change the process.
pub(in crate::family::qwen::flash_next) fn admit(
    cuda_decode: Option<bool>,
    opt_in: bool,
    first_path: &std::path::Path,
) -> Result<(), EngError> {
    check_cuda_decode(cuda_decode)?;
    check_opt_in(opt_in, first_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unreadable tokenizer file is refused; an absent one still pins zeros for the load to
    /// report.
    #[test]
    fn an_unreadable_tokenizer_file_is_an_error_not_a_zero_digest() {
        let dir = tempfile::tempdir().expect("tempdir");
        let absent = observed_pin(dir.path(), 8).expect("absent files pin zeros");
        assert_eq!(
            (
                absent.tokenizer_json,
                absent.tokenizer_config_json,
                absent.chat_template_jinja,
                absent.generation_config_json,
                absent.vocab_size
            ),
            ([0u8; 32], [0u8; 32], [0u8; 32], [0u8; 32], 8)
        );
        let file = dir.path().join("tokenizer.json");
        std::fs::write(&file, b"{}").expect("write");
        std::fs::set_permissions(&file, std::os::unix::fs::PermissionsExt::from_mode(0))
            .expect("unreadable");
        let Err(error) = observed_pin(dir.path(), 8) else {
            panic!("an unreadable file is refused");
        };
        assert!(format!("{error:?}").contains("tokenizer.json"), "{error:?}");
        assert!(tokenizer_digest(dir.path()).is_err());
    }
    use std::path::Path;

    /// The switch alone, the entry alone, neither and both: only both serve.
    #[test]
    fn the_gguf_start_needs_cuda_decode_and_the_switch() {
        let path = Path::new("/models/flash-next.gguf");
        let both = admit(Some(true), true, path);
        assert!(both.is_ok(), "{both:?}");
        // The switch off: the start's own refusal, naming the switch and the file.
        let Err(error) = admit(Some(true), false, path) else {
            panic!("without the switch the start must be refused");
        };
        let text = format!("{error:?}");
        assert!(text.contains("not qualified yet"), "{text}");
        assert!(text.contains("FIFI_FLASH_NEXT_GGUF"), "{text}");
        assert!(text.contains("flash-next.gguf"), "{text}");
        // The entry not set to true: the pack's refusal, whatever the switch says.
        for (cuda_decode, opt_in) in [(None, true), (Some(false), true), (None, false)] {
            let Err(EngError::BadRequest(text)) = admit(cuda_decode, opt_in, path) else {
                panic!("{cuda_decode:?} must be refused");
            };
            assert!(text.contains("cuda_decode"), "{text}");
        }
    }

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
        assert!(
            format!("{error:?}").contains("the 10 the router picks"),
            "{error:?}"
        );
        assert!(check_slots(10, 10).is_ok());
    }
}
