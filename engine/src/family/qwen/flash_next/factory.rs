//! The engine factory for `family: "qwen3_8_flash_next"` entries of the `enginesrv` config.
//!
//! The server start-up code (`server/run.rs`) calls it for the entries of this family. Order of
//! checks: the settings (`settings.rs`), then the backend. A build without `native-mlx` on Apple
//! Silicon stops at the backend check with a clear error and never touches a file.

use super::settings::{FlashEnv, FlashOptions, FlashSettings};
use crate::{
    EngError,
    family::QWEN3_8_FLASH_NEXT,
    sched::worker::{Engine, ReadyNote},
    tokenizer::TokenizerBundle,
};
use std::path::Path;

pub(crate) const FAMILY: &str = QWEN3_8_FLASH_NEXT.name;

/// The most tokens of prompt plus completion that the server accepts for this family
/// (`qsa::MAX_POSITIONS` of the resident decoder). `start.rs` compares it with
/// `FlashExecutor::context_limit` when the engine thread opens the executor, so a changed
/// decoder limit stops the load instead of misleading the 409 check.
pub(crate) const CONTEXT_LIMIT: u32 = 32_768;

pub(crate) struct StartedEngine {
    /// The engine thread. The caller stops it with `Engine::shutdown`.
    pub(crate) engine: Engine,
    pub(crate) bundle: TokenizerBundle,
    pub(crate) context_limit: u32,
}

/// The engine state is `Loading` until the weights are resident.
pub(crate) fn start_engine(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlashOptions>,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<StartedEngine, EngError> {
    start_engine_resolved(
        model_dir,
        tokenizer_dir,
        options,
        &FlashEnv::capture(),
        prefix_reuse,
        ready_note,
    )
    .map(|(engine, _)| engine)
}

/// Starts one entry. The second element is the entry's resolved settings for a pack; a GGUF
/// checkpoint has none (no manifest, no `config.json`) and answers `None`: its start plans from
/// its own metadata and takes its own path, so no settings are made up for it.
pub(crate) fn start_engine_resolved(
    model_dir: &Path,
    tokenizer_dir: &Path,
    options: Option<&FlashOptions>,
    env: &FlashEnv,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<(StartedEngine, Option<FlashSettings>), EngError> {
    // A directory that starts from a pack keeps the pack's start, whatever GGUF lies beside it,
    // unless `INFERENCE_FLASH_NEXT_GGUF_SERVE` is on: the staged start pins its tokenizer through
    // the same manifest lookup and cannot run without one, so with the switch on a GGUF
    // checkpoint takes that start whether or not a manifest is found. Where a manifest is found
    // the switch is read here, so one that is set and is not an enable word is refused before
    // the directory is opened, a pack's as well as a checkpoint's. Without the switch a GGUF
    // checkpoint alone has no manifest and no `config.json`: it plans from its own metadata and
    // tensor table and takes its own start, without settings of the pack's kind.
    #[cfg(all(feature = "native-cuda", target_os = "linux"))]
    if !(super::settings::pack_start(options, env, model_dir, tokenizer_dir)
        && !env.gguf_serve()?)
        && let Some(model) = super::load::gguf::GgufModel::open(model_dir)?
    {
        use super::load::gguf::start::StartInput;
        // `INFERENCE_FLASH_NEXT_GGUF_SERVE` selects the staged GGUF decode modules; without it
        // the start takes the path behind `FIFI_FLASH_NEXT_GGUF` (`check_opt_in`). A mis-set
        // switch is refused before the card is opened: here where no manifest was found, at the
        // gate otherwise.
        let (engine, bundle, context_limit) = if env.gguf_serve()? {
            super::load::gguf::start::start_staged(
                model,
                StartInput {
                    model_dir,
                    tokenizer_dir,
                    options,
                    env,
                    ready_note,
                },
            )?
        } else {
            super::load::gguf::start::start(
                model,
                options,
                env.gguf_opt_in(),
                tokenizer_dir,
                ready_note,
            )?
        };
        return Ok((
            StartedEngine {
                engine,
                bundle,
                context_limit,
            },
            None,
        ));
    }
    let settings = FlashSettings::new(model_dir, tokenizer_dir, options, env)?;
    if settings.manifest_found_in_pack {
        eprintln!(
            "flash-next: using the pack's own manifest {}",
            settings.manifest.display()
        );
    }
    // The CUDA decoder serves the positions its caches hold (the memory plan's choice when the
    // config sets none); the MLX one a fixed limit.
    let (engine, bundle, context_limit) = launch(&settings, prefix_reuse, ready_note)?;
    Ok((
        StartedEngine {
            engine,
            bundle,
            context_limit,
        },
        Some(settings),
    ))
}

pub(crate) fn check_mlx_settings(settings: &FlashSettings) -> Result<(), EngError> {
    if settings.cuda.is_some() {
        return Err(EngError::BadRequest(
            "engine config: cuda_decode needs a build with --features native-cuda on Linux; this build serves Flash-Next on MLX"
                .to_string(),
        ));
    }
    Ok(())
}

#[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
fn launch(
    settings: &FlashSettings,
    prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    check_mlx_settings(settings)?;
    let (engine, bundle) = super::start::start(settings, prefix_reuse, ready_note)?;
    Ok((engine, bundle, CONTEXT_LIMIT))
}

/// A CUDA build serves Flash-Next on the card unless the entry sets `cuda_decode` to false, which
/// it refuses: the MLX backend is not in this build.
#[cfg(all(
    feature = "native-cuda",
    target_os = "linux",
    not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))
))]
fn launch(
    settings: &FlashSettings,
    _prefix_reuse: bool,
    ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    if settings.cuda.is_none() {
        return Err(EngError::BadRequest(
            "engine config: the Flash-Next CUDA decoder is off; set flash_next.cuda_decode to true to serve on a card (the MLX backend needs --features native-mlx on Apple Silicon)"
                .to_string(),
        ));
    }
    super::cuda::start::start(settings, ready_note)
}

#[cfg(not(any(
    all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"),
    all(feature = "native-cuda", target_os = "linux")
)))]
fn launch(
    _settings: &FlashSettings,
    _prefix_reuse: bool,
    _ready_note: ReadyNote,
) -> Result<(Engine, TokenizerBundle, u32), EngError> {
    Err(EngError::BadRequest(
        "engine config: this build has no Flash-Next backend; build with --features native-mlx on Apple Silicon"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{CONTEXT_LIMIT, FAMILY, FlashEnv, start_engine_resolved};
    use crate::{EngError, family::qwen::flash_next::settings::FlashOptions};
    use std::path::Path;

    fn message<T>(result: Result<T, EngError>) -> String {
        match result {
            Err(EngError::BadRequest(text)) => text,
            Err(other) => panic!("expected BadRequest, got {other:?}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    fn start(options: Option<&FlashOptions>, env: &FlashEnv) -> Result<(), EngError> {
        start_engine_resolved(Path::new("/a"), Path::new("/a"), options, env, false, None)
            .map(|_| ())
    }

    #[test]
    fn the_family_name_and_the_context_limit_are_the_documented_values() {
        assert_eq!(FAMILY, "qwen3_8_flash_next");
        assert_eq!(CONTEXT_LIMIT, 32_768);
    }

    #[test]
    fn an_entry_with_a_missing_setting_is_refused_before_the_backend() {
        assert_eq!(
            message(start(None, &FlashEnv::default())),
            "engine config: manifest is missing: set the manifest key or \
             FLASH_NEXT_ARTIFACT_MANIFEST, or put download-manifest.json in the model directory \
             (looked for /a/download-manifest.json)"
        );
    }

    #[cfg(not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64")))]
    #[test]
    fn a_complete_entry_in_a_build_without_the_backend_names_the_feature() {
        use std::path::PathBuf;
        let options = FlashOptions {
            manifest: Some(PathBuf::from("/does/not/exist/download-manifest.json")),
            cache_limit_bytes: Some(1024),
            ..FlashOptions::default()
        };
        let text = message(start(Some(&options), &FlashEnv::default()));
        assert!(text.contains("native-mlx"), "{text}");
    }

    /// A directory that starts from a pack keeps the pack's start: the small GGUF beside the
    /// pack's manifest does not switch the directory to the GGUF start.
    #[test]
    fn a_pack_directory_keeps_the_pack_start_beside_a_gguf() {
        use super::super::settings::pack_start;
        let small = super::super::load::gguf::smallmodel::write_small(7);
        let dir = small.dir.path();
        let env = FlashEnv::default();
        assert!(!pack_start(None, &env, dir, dir), "a GGUF alone");
        std::fs::write(dir.join("download-manifest.json"), "{}").expect("the pack's manifest");
        assert!(pack_start(None, &env, dir, dir), "a pack beside the GGUF");
        // Through the factory the pack's start answers (on this build its backend refusal);
        // the GGUF start's own refusal is not what an operator sees.
        let Err(error) = start_engine_resolved(dir, dir, None, &env, false, None) else {
            panic!("no backend serves the pack here");
        };
        let text = format!("{error:?}");
        eprintln!("the pack's start answers: {text}");
        assert!(!text.contains("GGUF"), "{text}");
    }

    /// A CUDA build with the switch off names the switch (and the MLX alternative).
    #[cfg(all(
        feature = "native-cuda",
        target_os = "linux",
        not(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))
    ))]
    #[test]
    fn a_cuda_build_with_the_switch_off_names_cuda_decode() {
        use std::path::PathBuf;
        let options = FlashOptions {
            manifest: Some(PathBuf::from("/does/not/exist/download-manifest.json")),
            cache_limit_bytes: Some(1024),
            cuda_decode: Some(false),
            ..FlashOptions::default()
        };
        let text = message(start(Some(&options), &FlashEnv::default()));
        assert!(text.contains("cuda_decode"), "{text}");
    }
}
