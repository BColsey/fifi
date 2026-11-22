//! `enginesrv serve --model <path>`: the simplified startup.
//!
//! A front end, not a second loader. Discovery reads the directory the
//! way `capability::inspect` already does; the chosen settings go into
//! a normal in-memory `Config` that enters the exact `--config` path,
//! so validation, admission, and serving are the existing ones. The
//! GPU is never probed here — a failed diagnostic CUDA query latches
//! the backend terminal for the rest of the process
//! (`capability::probe`), so device feasibility is left to the
//! load-time admission ledger, which is authoritative anyway.
//!
//! Selection rules: the family comes from `config.json`'s `model_type`
//! matched against the family specs (or `--family` when there is no
//! evidence); the weight format is `gguf` when a GGUF is present
//! (the qualified default — an EXL3-only directory serves `exl3` with
//! its experimental acknowledgement printed, never silently). No
//! experimental feature is armed by this path: every other option is
//! the resolver's own documented default.

use std::path::{Path, PathBuf};

use crate::capability::probe::{CompiledBackends, HardwareFacts};
use crate::capability::{Probe, describe_error, inspect, preflight};
use crate::family::inkling2_mini;
use crate::server::config::{Config, EngineConfig, StartupError};

/// The families `--model` can identify and start. One row per
/// detectable family; extending to Flint is a row here plus its
/// loader's own checks staying the authority.
pub(crate) const DETECTABLE_FAMILIES: &[(&str, &[&str])] = &[
    (
        crate::family::qwen::qwen36::factory::FAMILY,
        crate::family::QWEN3_6_35B_A3B.model_types,
    ),
    (
        crate::family::qwen::qwen36::factory::DENSE_FAMILY,
        crate::family::QWEN3_8_27B_DENSE.model_types,
    ),
    (inkling2_mini::FAMILY, &[inkling2_mini::MODEL_TYPE]),
];

/// Whether a family's engine entry carries the Qwen3.6 options block (the Qwen3.6 families);
/// every other family starts from a bare entry.
fn uses_qwen36_options(family: &str) -> bool {
    DETECTABLE_FAMILIES.iter().any(|(name, _)| *name == family)
}

/// The families that a GGUF `general.architecture` identifies when the directory has no
/// `config.json`.
const ARCHITECTURE_FAMILIES: &[(&str, &str)] = crate::server::config::GGUF_ARCHITECTURE_FAMILIES;

/// The listen address of a simplified startup, matching the shipped
/// configurations (loopback; `listen_address` rejects anything else).
const DEFAULT_LISTEN: &str = "127.0.0.1:18791";
/// The key variable of a simplified startup, matching the shipped
/// configurations; the server refuses to serve without it set.
const DEFAULT_API_KEY_ENV: &str = "ENGINESRV_API_KEY";

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ModelServeArgs {
    pub(crate) model_dir: PathBuf,
    pub(crate) tokenizer_dir: Option<PathBuf>,
    pub(crate) family: Option<String>,
    pub(crate) context_limit: Option<u32>,
    pub(crate) kv_dtype: Option<String>,
    pub(crate) listen: Option<String>,
    pub(crate) print_config: bool,
}

/// Parse the arguments of the simplified startup (`serve`, with or without `--model`).
/// The flag set is the preflight candidate surface plus the serving
/// address and `--print-config`; anything else is a usage error, not a
/// guess.
pub(crate) fn parse_args(args: &[String]) -> Result<ModelServeArgs, StartupError> {
    let mut parsed = ModelServeArgs::default();
    let mut index = 0;
    let value = |flag: &str, args: &[String], index: usize| -> Result<String, StartupError> {
        args.get(index)
            .cloned()
            .ok_or_else(|| StartupError(format!("serve --model: {flag} needs a value\n{USAGE}")))
    };
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--model" => {
                index += 1;
                parsed.model_dir = PathBuf::from(value("--model", args, index)?);
            }
            "--tokenizer-dir" => {
                index += 1;
                parsed.tokenizer_dir = Some(PathBuf::from(value("--tokenizer-dir", args, index)?));
            }
            "--family" => {
                index += 1;
                parsed.family = Some(value("--family", args, index)?);
            }
            "--context" => {
                index += 1;
                let text = value("--context", args, index)?;
                parsed.context_limit = Some(text.parse().map_err(|_| {
                    StartupError(format!(
                        "serve --model: --context needs a token count, got {text:?}"
                    ))
                })?);
            }
            "--kv" => {
                index += 1;
                parsed.kv_dtype = Some(value("--kv", args, index)?);
            }
            "--listen" => {
                index += 1;
                parsed.listen = Some(value("--listen", args, index)?);
            }
            "--print-config" => parsed.print_config = true,
            other => {
                return Err(StartupError(format!(
                    "serve --model: unknown flag {other:?}\n{USAGE}"
                )));
            }
        }
        index += 1;
    }
    // No `--model` is the default start: `resolve_model_dir` chooses a local model.
    Ok(parsed)
}

const USAGE: &str = "usage: enginesrv serve [--model <dir|file.gguf|name|org/repo[:QUANT]>] [--tokenizer-dir <dir>] [--family <name>] [--context <n>] [--kv f32|q8|bf16] [--listen <addr>] [--print-config]";

/// What discovery chose: the family, the weight format, and the
/// options that carry them. Every other option stays the resolver's
/// own default — nothing experimental is armed here.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Discovered {
    pub(crate) family: String,
    pub(crate) weight_format: &'static str,
    pub(crate) options: crate::family::qwen::qwen36::settings::Q36Options,
    pub(crate) tokenizer_dir: PathBuf,
}

/// Where the tokenizer is read from when `--tokenizer-dir` is not given: the model directory; for
/// a model path that is one GGUF file, the file itself, whose header carries the vocabulary and the
/// chat template. A sibling `tokenizer.json` is never taken for a lone file, because it may belong
/// to another model in the same directory: pass the directory to use its files.
fn default_tokenizer_dir(model: &Path) -> PathBuf {
    if crate::family::gguf_select::is_gguf_file(model) {
        let first = crate::family::gguf_select::first_shard(model);
        // An Inkling text checkpoint reads its sidecars from the parent; every other
        // bare GGUF is its own tokenizer.
        if crate::capability::inspect::gguf_is_inkling(&first) {
            return crate::family::gguf_select::directory_of(&first);
        }
        first
    } else {
        model.to_path_buf()
    }
}

/// Inspect the model path and choose family, format, and options.
/// Pure filesystem evidence; no preflight, no probing.
fn discover(args: &ModelServeArgs) -> Result<Discovered, StartupError> {
    let tokenizer_dir = args
        .tokenizer_dir
        .clone()
        .unwrap_or_else(|| default_tokenizer_dir(&args.model_dir));
    let inspection = inspect::inspect(&args.model_dir, &tokenizer_dir)
        .map_err(|error| StartupError(describe_error(&error)))?;

    let has_gguf = !inspection.gguf_files.is_empty();
    let has_safetensors = inspection.safetensors_files > 0;
    if !has_gguf && !has_safetensors {
        return Err(StartupError(format!(
            "serve --model: no .gguf or .safetensors weight files in {}",
            args.model_dir.display()
        )));
    }

    let family = select_family(args, &inspection)?;
    if !uses_qwen36_options(&family) {
        for (flag, given) in [
            ("--kv", args.kv_dtype.is_some()),
            ("--context", args.context_limit.is_some()),
        ] {
            if given {
                return Err(StartupError(format!(
                    "serve --model: {flag} is not an option of the {family} family; write a --config file"
                )));
            }
        }
    }
    let weight_format = if family == inkling2_mini::FAMILY {
        if !inspection.has_inkling_safetensors() && inspection.inkling_ggufs().next().is_some() {
            "gguf"
        } else {
            "safetensors"
        }
    } else {
        select_weight_format(has_gguf, has_safetensors)
    };

    let options = crate::family::qwen::qwen36::settings::Q36Options {
        // An MLX pack has no weight format member: the MLX arm reads the safetensors shards as
        // they are.
        weight_format: (weight_format != MLX_WEIGHTS).then(|| weight_format.to_string()),
        context_limit: args.context_limit,
        kv_dtype: args.kv_dtype.clone(),
        ..crate::family::qwen::qwen36::settings::Q36Options::default()
    };
    Ok(Discovered {
        family,
        weight_format,
        options,
        tokenizer_dir,
    })
}

/// A `--model` argument that names an existing path is that path;
/// anything else shaped like `org/repo` resolves to the hub cache
/// snapshot of its `main` ref, and anything else is left for
/// discovery to reject in its own words.
fn resolve_model_dir(args: &ModelServeArgs) -> Result<ModelServeArgs, StartupError> {
    if args.model_dir.as_os_str().is_empty() {
        return resolve_default_model(
            args,
            &super::local_models::Stores::from_env(),
            &mut |line| {
                eprintln!("{line}");
            },
        );
    }
    let text = args.model_dir.to_string_lossy();
    // A path that exists, or a link that points nowhere: the model check names what is wrong with
    // it (a dangling link) instead of reading its name as a store entry.
    if args.model_dir.exists() || std::fs::symlink_metadata(&args.model_dir).is_ok() {
        return Ok(args.clone());
    }
    // `org/repo` or `org/repo:QUANT`: the hub cache, then the hub itself.
    if let Some(wanted) = super::fetch_name::parse_wanted(&text) {
        return resolve_hub_name(args, &wanted);
    }
    if let Some(snapshot) = super::hf::cached_main_snapshot(&text) {
        eprintln!(
            "model: {text} resolved to the hub cache at {}",
            snapshot.display()
        );
        let mut resolved = args.clone();
        resolved.model_dir = snapshot;
        return Ok(resolved);
    }
    // A plain name: look in the local stores before saying it is not a directory.
    if super::local_models::is_plain_name(&text) {
        let stores = super::local_models::Stores::from_env();
        if let Some(found) = super::local_models::resolve(&text, &stores)? {
            eprintln!(
                "model: {text} resolved to {} in {}",
                found.dir.display(),
                found.store
            );
            let mut resolved = args.clone();
            resolved.model_dir = found.dir;
            return Ok(resolved);
        }
        return Err(StartupError(format!(
            "serve --model: nothing local is named {text} (looked in {}, {} and {}); download it first, for example from the model page of the web console or with `hf download <org/repo>`, or pass the directory",
            stores.models.display(),
            stores.ollama.display(),
            stores.hub.display()
        )));
    }
    Ok(args.clone())
}

/// `serve` with no `--model`: the most recently written model of the local stores that a family
/// serves. Models a family refuses are skipped, each with its reason; with none left the error
/// says where to get one. The choice and how to name another are printed.
fn resolve_default_model(
    args: &ModelServeArgs,
    stores: &super::local_models::Stores,
    say: &mut dyn FnMut(&str),
) -> Result<ModelServeArgs, StartupError> {
    let local = super::local_models::all(stores);
    let mut skipped: Vec<String> = Vec::new();
    for model in &local {
        let mut resolved = args.clone();
        resolved.model_dir = model.path.clone();
        match discover(&resolved) {
            Ok(_) => {
                for line in &skipped {
                    say(line);
                }
                say(&format!(
                    "model: no --model given; serving {} from {} (the newest of {} local); name another with --model <name>",
                    model.name,
                    model.store,
                    local.len()
                ));
                return Ok(resolved);
            }
            Err(error) => skipped.push(format!(
                "model: skipping {} ({}): {}",
                model.name,
                model.path.display(),
                error.0.lines().next().unwrap_or_default()
            )),
        }
    }
    let tried = if skipped.is_empty() {
        String::new()
    } else {
        format!("\n{}", skipped.join("\n"))
    };
    Err(StartupError(format!(
        "serve: no --model given and no model is stored locally that this engine serves (looked in {}, {} and {}){tried}\nname one: --model <org/repo> fetches a GGUF repo from the hub, --model <file.gguf> serves a file, or put a model in the models directory",
        stores.models.display(),
        stores.ollama.display(),
        stores.hub.display()
    )))
}

/// A hub repo name that is not a path: one quantization of its cached snapshot, or of the hub
/// when the cache holds none that fits, chosen by the planner.
fn resolve_hub_name(
    args: &ModelServeArgs,
    wanted: &super::fetch_name::Wanted,
) -> Result<ModelServeArgs, StartupError> {
    let mut resolved = args.clone();
    let hub = std::sync::Arc::new(super::hf::HfHub::from_env());
    // The planner reads the machine once, and only when a choice needs it.
    let facts = std::cell::OnceCell::new();
    let measured = || facts.get_or_init(crate::capability::probe::probe);
    let accepts = |stub: &Path| discover(&args_for_path(stub)).map(|_| ());
    let mut fit = |stub: &Path| planned_fit(stub, measured());
    resolved.model_dir = super::fetch_name::resolve(
        &hub,
        wanted,
        super::fetch_name::Planner {
            accepts: &accepts,
            fit: &mut fit,
        },
        &mut |line| eprintln!("{line}"),
    )?;
    Ok(resolved)
}

fn args_for_path(path: &Path) -> ModelServeArgs {
    ModelServeArgs {
        model_dir: path.to_path_buf(),
        ..ModelServeArgs::default()
    }
}

/// Whether the shared planner plans the stub's weights fully resident on this host's card at the
/// default context: the preflight estimate's own fit. A plan with no estimate is not resident; a
/// preflight that cannot run is an error.
fn planned_fit(stub: &Path, facts: &HardwareFacts) -> Result<super::fetch_name::Fit, StartupError> {
    use super::fetch_name::Fit;
    let report = preflight::preflight(stub, stub, &preflight::Candidate::default(), facts)
        .map_err(|error| StartupError(describe_error(&error)))?;
    Ok(match report.estimate {
        Some(estimate) if estimate.fit == preflight::ResourceFit::Fits => Fit::Resident,
        _ => Fit::NotResident,
    })
}

/// Discover the model, run the non-probing preflight, and build the
/// in-memory `Config` for the existing serve path. Prints the chosen
/// configuration and any qualification acknowledgement before the
/// engines start.
pub(crate) fn build_config(args: &ModelServeArgs) -> Result<Config, StartupError> {
    // An `org/repo` id names a hub-cache download (the GUI's model
    // page, or `hf download`); a literal existing path always wins.
    let args = resolve_model_dir(args)?;
    let discovered = discover(&args)?;

    // The preflight of a serving process: every check that needs no
    // device. CUDA facts stay deliberately unavailable (the module
    // docs); host facts are a /proc read the system endpoint already
    // does per request.
    let facts = HardwareFacts {
        backends: CompiledBackends::of_this_build(),
        cuda: no_cuda_facts(),
        host: crate::capability::probe::probe_host(),
    };
    let candidate = preflight::Candidate {
        family: Some(discovered.family.clone()),
        context_limit: args.context_limit,
        kv_dtype: args.kv_dtype.clone(),
        prefill_rows: None,
    };
    let report = preflight::preflight(
        &args.model_dir,
        &discovered.tokenizer_dir,
        &candidate,
        &facts,
    )
    .map_err(|error| StartupError(describe_error(&error)))?;
    match report.verdict {
        preflight::Verdict::Unsupported => {
            return Err(StartupError(format!(
                "serve --model: this build cannot serve {}:\n{}next: {}",
                args.model_dir.display(),
                failed_checks(&report),
                report.next_step
            )));
        }
        preflight::Verdict::Experimental => {
            eprintln!(
                "qualification: EXPERIMENTAL — the quality gates for this configuration have not passed; serving it because serve --model prints this acknowledgement"
            );
            eprint!("{}", warned_checks(&report));
        }
        _ => eprint!("{}", warned_checks(&report)),
    }

    let backend = crate::family::qwen::qwen36::factory::backend_of_build();
    eprintln!(
        "model: {} — family {}, {} weights, backend {}",
        args.model_dir.display(),
        discovered.family,
        discovered.weight_format,
        backend_name(backend),
    );
    eprintln!(
        "settings: the family defaults for context and KV dtype (override with --context, --kv); memory feasibility is decided at load by the runtime admission ledger — 'enginesrv check {}' prints the full preflight",
        args.model_dir.display()
    );

    Ok(config_from(&discovered, &args))
}

/// Assemble the in-memory `Config` — the document a `--config` file
/// would carry, headed for the identical serve path.
fn config_from(discovered: &Discovered, args: &ModelServeArgs) -> Config {
    let inkling = discovered.family == inkling2_mini::FAMILY;
    let simplified = !inkling && uses_qwen36_options(&discovered.family);
    let mut config = bare_config(args, &discovered.family);
    let engine = &mut config.engines[0];
    engine.tokenizer_dir = discovered.tokenizer_dir.clone();
    if simplified {
        engine.id = "q36".to_string();
        engine.qwen36 = Some(discovered.options.clone());
    }
    if inkling {
        engine.id = "inkling".to_string();
        engine.context_limit = args.context_limit;
    }
    config
}

/// The configuration a supervised worker reads for a family that has
/// no simplified startup (Flash-Next): the bare entry a `--config`
/// file carries, with the family named rather than detected and every
/// option left to its default. `serve --model` still refuses such a
/// family on its own, because a command line has nothing to name it
/// by.
pub(crate) fn bare_config(args: &ModelServeArgs, family: &str) -> Config {
    Config {
        listen: args
            .listen
            .clone()
            .unwrap_or_else(|| DEFAULT_LISTEN.to_string()),
        api_key_env: DEFAULT_API_KEY_ENV.to_string(),
        prefix_reuse: Some(false),
        sample_trace: false,
        supervised_engines: false,
        web_dir: None,
        engines: vec![EngineConfig {
            id: family.to_string(),
            model_dir: args.model_dir.clone(),
            tokenizer_dir: args
                .tokenizer_dir
                .clone()
                .unwrap_or_else(|| args.model_dir.clone()),
            family: family.to_string(),
            model: None,
            context_limit: None,
            think_replay: None,
            notice_toolwrap: None,
            kv_cache: None,
            fake: None,
            flash_next: None,
            qwen36: None,
            nemotron: None,
            gemma4: None,
            flint: None,
        }],
    }
}

/// The exact JSON of the built configuration: the same document a
/// `--config` file would carry, so a printed configuration starts
/// identically (`Config::from_file` reads it back).
pub(crate) fn config_json(config: &Config) -> String {
    use serde_json::json;
    let engine = &config.engines[0];
    if !uses_qwen36_options(&engine.family) {
        let document = json!({
            "listen": config.listen,
            "api_key_env": config.api_key_env,
            "prefix_reuse": config.prefix_reuse,
            "engines": [{
                "id": engine.id,
                "family": engine.family,
                "model_dir": engine.model_dir.display().to_string(),
                "tokenizer_dir": engine.tokenizer_dir.display().to_string(),
            }],
        });
        return serde_json::to_string_pretty(&document).expect("the built config is valid JSON");
    }
    let mut q36 = serde_json::Map::new();
    if let Some(options) = &engine.qwen36 {
        if let Some(value) = &options.weight_format {
            q36.insert("weight_format".into(), json!(value));
        }
        if let Some(value) = options.context_limit {
            q36.insert("context_limit".into(), json!(value));
        }
        if let Some(value) = &options.kv_dtype {
            q36.insert("kv_dtype".into(), json!(value));
        }
    }
    let mut entry = json!({
        "id": engine.id,
        "family": engine.family,
        "model_dir": engine.model_dir.display().to_string(),
        "tokenizer_dir": engine.tokenizer_dir.display().to_string(),
    });
    if engine.qwen36.is_some() {
        entry["qwen36"] = json!(q36);
    } else if let Some(context_limit) = engine.context_limit {
        entry["context_limit"] = json!(context_limit);
    }
    let document = json!({
        "listen": config.listen,
        "api_key_env": config.api_key_env,
        "prefix_reuse": config.prefix_reuse,
        "engines": [entry],
    });
    serde_json::to_string_pretty(&document).expect("the built config is valid JSON")
}

/// The family `serve --model` would start for this directory, from
/// the same inspection and the same selection: `Err` carries the words
/// the CLI would refuse with. The hub cache listing uses it so its
/// "servable" badge cannot disagree with the loader.
pub(crate) fn servable_family(model_dir: &Path) -> Result<String, String> {
    let inspection =
        inspect::inspect(model_dir, model_dir).map_err(|error| describe_error(&error))?;
    if inspection.gguf_files.is_empty() && inspection.safetensors_files == 0 {
        return Err("no .gguf or .safetensors weight files".to_string());
    }
    select_family(&ModelServeArgs::default(), &inspection).map_err(|error| error.0)
}

/// The Flash-Next family when the directory holds one of its packs, else
/// `None`. To every other test a pack is a Qwen3.8 snapshot, and
/// `servable_family` finds no family of this build in it: the model type
/// the spec table gives Flash-Next is what names it.
pub(crate) fn flash_next_family(model_dir: &Path) -> Option<&'static str> {
    let inspection = inspect::inspect(model_dir, model_dir).ok()?;
    let spec = &crate::family::spec::QWEN3_8_FLASH_NEXT;
    spec.model_types
        .iter()
        .any(|claimed| {
            inspection
                .config_model_types
                .iter()
                .any(|seen| seen == claimed)
        })
        .then_some(spec.name)
}

/// The memory one engine of a downloaded snapshot needs, including the
/// preflight's closed-form cache and admission terms.
pub(crate) struct TierEstimate {
    pub(crate) vram_bytes: u64,
    pub(crate) ram_bytes: u64,
    pub(crate) disk_bytes: u64,
    pub(crate) basis: String,
}

pub(crate) fn estimate_tiers(model_dir: &Path, family: &str) -> Option<TierEstimate> {
    let facts = HardwareFacts {
        backends: CompiledBackends::of_this_build(),
        cuda: no_cuda_facts(),
        host: crate::capability::probe::probe_host(),
    };
    let candidate = preflight::Candidate {
        family: Some(family.to_string()),
        ..preflight::Candidate::default()
    };
    let report = preflight::preflight(model_dir, model_dir, &candidate, &facts).ok()?;
    let estimate = report.estimate?;
    let (weights, picked) =
        if family == crate::family::qwen::qwen36::factory::FAMILY && facts.backends.cuda {
            (estimate.weight_bytes, "the GGUF selected by preflight")
        } else {
            match crate::family::gguf_select::candidates(model_dir) {
                Ok((_, candidates)) if !candidates.is_empty() => {
                    (candidates[0].bytes, "the largest GGUF quantization on disk")
                }
                _ => (estimate.weight_bytes, "the checkpoint files on disk"),
            }
        };
    let other = estimate.total_bytes.saturating_sub(estimate.weight_bytes);
    Some(TierEstimate {
        vram_bytes: weights + other + estimate.admission_charges_bytes,
        ram_bytes: weights,
        disk_bytes: weights,
        basis: format!(
            "{picked} ({weights} bytes) plus the planner's cache, state and admission terms ({} bytes) at context {}",
            other + estimate.admission_charges_bytes,
            estimate.context_limit
        ),
    })
}

fn select_family(
    args: &ModelServeArgs,
    inspection: &inspect::ModelInspection,
) -> Result<String, StartupError> {
    if let Some(family) = &args.family {
        if DETECTABLE_FAMILIES.iter().any(|(name, _)| name == family) {
            return Ok(family.clone());
        }
        return Err(StartupError(format!(
            "serve --model: family {family} has no simplified startup yet; write a --config file for it (engine/config/README.md). Detectable families: {}",
            detectable_names()
        )));
    }
    // The explicitly chosen file is authoritative even in a mixed model directory.
    if args.model_dir.is_file() && inspection.inkling_ggufs().next().is_some() {
        return Ok(inkling2_mini::FAMILY.into());
    }
    let mut matches: Vec<&str> = DETECTABLE_FAMILIES
        .iter()
        .filter(|(_, types)| {
            types
                .iter()
                .any(|t| inspection.config_model_types.iter().any(|seen| seen == t))
                || (types.contains(&inkling2_mini::MODEL_TYPE)
                    && inspection.inkling_ggufs().next().is_some())
        })
        .map(|(name, _)| *name)
        .collect();
    // A directory with no `config.json` model_type is identified by the architecture its weights
    // file declares.
    if matches.is_empty() {
        matches = ARCHITECTURE_FAMILIES
            .iter()
            .filter(|(architecture, _)| {
                inspection
                    .gguf_files
                    .iter()
                    .any(|fact| fact.architecture.as_deref() == Some(*architecture))
            })
            .map(|(_, family)| *family)
            .collect();
    }
    match matches.pop() {
        Some(name) if matches.is_empty() => Ok(name.to_string()),
        _ => {
            let seen: Vec<&str> = inspection
                .gguf_files
                .iter()
                .filter_map(|fact| fact.architecture.as_deref())
                .collect();
            if inspection.config_model_types.is_empty()
                && seen.is_empty()
                && !inspection.gguf_files.is_empty()
            {
                return Err(StartupError(format!(
                    "serve --model: {} names no general.architecture, so no family can be chosen from it; known architectures: {}; pass --family (detectable families: {}) or write a --config file",
                    inspection.weights_path.display(),
                    crate::server::config::known_gguf_architectures(),
                    detectable_names()
                )));
            }
            if inspection.config_model_types.is_empty() && !seen.is_empty() {
                let why: String = seen
                    .iter()
                    .filter_map(|architecture| {
                        crate::server::config::gguf_architecture_note(architecture)
                    })
                    .map(|note| format!(" ({note})"))
                    .collect();
                return Err(StartupError(format!(
                    "serve --model: no family serves the GGUF architecture {}{why}; known architectures: {}; pass --family or write a --config file",
                    seen.join(", "),
                    crate::server::config::known_gguf_architectures()
                )));
            }
            Err(StartupError(format!(
                "serve --model: cannot identify the family from config.json (model_type {:?}); pass --family. Detectable families: {}",
                inspection.config_model_types,
                detectable_names()
            )))
        }
    }
}

/// The weight kind a build with the MLX backend reports for a directory of safetensors shards
/// and no GGUF file.
const MLX_WEIGHTS: &str = "mlx";

/// Whether this build serves from the MLX backend (Apple Silicon with `native-mlx`).
const MLX_BUILD: bool = cfg!(all(
    feature = "native-mlx",
    target_os = "macos",
    target_arch = "aarch64"
));

fn select_weight_format(has_gguf: bool, has_safetensors: bool) -> &'static str {
    if has_gguf {
        if has_safetensors {
            eprintln!(
                "weights: the directory holds GGUF and EXL3 shards; serving the qualified GGUF default (EXL3 is experimental — select it with --config and weight_format exl3)"
            );
        }
        "gguf"
    } else if MLX_BUILD {
        MLX_WEIGHTS
    } else {
        "exl3"
    }
}

fn detectable_names() -> String {
    DETECTABLE_FAMILIES
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}

fn backend_name(backend: crate::family::qwen::qwen36::settings::Backend) -> &'static str {
    match backend {
        crate::family::qwen::qwen36::settings::Backend::Mlx => "mlx",
        crate::family::qwen::qwen36::settings::Backend::Cuda => "cuda",
    }
}

fn no_cuda_facts() -> crate::capability::probe::CudaFacts {
    let reason = "simplified startup does not probe the GPU (a failed diagnostic query latches the CUDA backend); the load-time admission ledger decides";
    crate::capability::probe::CudaFacts {
        device_count: Probe::Unavailable(reason.to_string()),
        first_device_integrated: Probe::Unavailable(reason.to_string()),
        first_device_free_bytes: Probe::Unavailable(reason.to_string()),
        first_device_total_bytes: Probe::Unavailable(reason.to_string()),
    }
}

fn failed_checks(report: &preflight::PreflightReport) -> String {
    report
        .checks
        .iter()
        .filter(|check| check.status == preflight::CheckStatus::Fail)
        .map(|check| format!("  {}: {}\n", check.name, check.detail))
        .collect()
}

fn warned_checks(report: &preflight::PreflightReport) -> String {
    report
        .checks
        .iter()
        .filter(|check| {
            check.status != preflight::CheckStatus::Pass
                && check.status != preflight::CheckStatus::Fail
        })
        .map(|check| format!("note: {}: {}\n", check.name, check.detail))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{ModelServeArgs, discover, flash_next_family, parse_args, servable_family};
    use crate::family::qwen::qwen36::settings::Q36Options;
    use crate::server::config::Config;
    use std::path::{Path, PathBuf};

    /// A GGUF v3 file with no metadata and no tensors: the header plus
    /// padding to the aligned data start, which is all `GgufHeader::parse`
    /// needs to accept the file as an inventory fact.
    fn minimal_gguf() -> Vec<u8> {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(32, 0);
        bytes
    }

    fn gguf_arch(arch: &str) -> Vec<u8> {
        let mut bytes = minimal_gguf();
        bytes.truncate(24);
        bytes[16..24].copy_from_slice(&1u64.to_le_bytes());
        let key = b"general.architecture";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key);
        bytes.extend(8u32.to_le_bytes());
        bytes.extend((arch.len() as u64).to_le_bytes());
        bytes.extend(arch.as_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes
    }

    #[test]
    fn discover_inkling_gguf_text_with_optional_sidecars() {
        let model = tempfile::tempdir().unwrap();
        let tokenizer = tempfile::tempdir().unwrap();
        for (filename, arch) in [
            ("mini.gguf", "inkling2"),
            ("mmproj.gguf", "clip"),
            ("draft.gguf", "dflash"),
        ] {
            std::fs::write(model.path().join(filename), gguf_arch(arch)).unwrap();
        }
        let found = discover(&ModelServeArgs {
            model_dir: model.path().into(),
            tokenizer_dir: Some(tokenizer.path().into()),
            ..ModelServeArgs::default()
        })
        .unwrap();
        assert_eq!(found.family, "inkling2_mini");
        assert_eq!(found.weight_format, "gguf");
        assert_eq!(found.tokenizer_dir, tokenizer.path());
    }

    #[test]
    fn explicit_inkling_gguf_selects_one_file_and_resolves_the_tokenizer() {
        let model = tempfile::tempdir().unwrap();
        let external = tempfile::tempdir().unwrap();
        let fork = model.path().join("fork.gguf");
        let mainline = model.path().join("mainline.gguf_file");
        std::fs::write(&fork, gguf_arch("inkling2")).unwrap();
        std::fs::write(&mainline, gguf_arch("inkling2_loop")).unwrap();
        std::fs::write(model.path().join("malformed-draft.gguf"), b"GGUF-broken").unwrap();
        std::fs::write(
            model.path().join("config.json"),
            r#"{"model_type":"qwen3_5_moe"}"#,
        )
        .unwrap();
        std::fs::write(model.path().join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(external.path().join("chat_template.jinja"), b"template").unwrap();

        for (path, tokenizer_dir, expected_dir, basename, present) in [
            (&fork, None, model.path(), "fork.gguf", "tokenizer.json"),
            (
                &mainline,
                Some(external.path().to_owned()),
                external.path(),
                "mainline.gguf_file",
                "chat_template.jinja",
            ),
        ] {
            let found = discover(&ModelServeArgs {
                model_dir: path.clone(),
                tokenizer_dir,
                ..ModelServeArgs::default()
            })
            .unwrap();
            assert_eq!(found.family, "inkling2_mini");
            assert_eq!(found.weight_format, "gguf");
            assert_eq!(found.tokenizer_dir, expected_dir);

            let report = crate::capability::inspect::inspect(path, &found.tokenizer_dir).unwrap();
            assert_eq!(report.model_dir, *model.path());
            assert_eq!(report.weights_path, *path);
            assert_eq!(report.files.len(), 1);
            assert_eq!(report.gguf_files.len(), 1);
            assert_eq!(report.gguf_files[0].file, basename);
            assert!(report.gguf_errors.is_empty(), "{:?}", report.gguf_errors);
            assert_eq!(report.tokenizer_present, [present]);
        }
        let directory = crate::capability::inspect::inspect(model.path(), model.path()).unwrap();
        assert_eq!(directory.inkling_ggufs().count(), 2);
        assert_eq!(directory.gguf_errors.len(), 1);
    }

    fn copy_tokenizer(dir: &Path) {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/server/testdata/tokenizer");
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            std::fs::copy(source.join(name), dir.join(name)).expect("tokenizer fixture");
        }
    }

    /// A model directory with the given weights and `config.json`
    /// `model_type`, plus the pinned tokenizer fixture.
    fn model_dir(model_type: Option<&str>, gguf: bool, safetensors: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        if gguf {
            std::fs::write(dir.path().join("model.gguf"), minimal_gguf()).expect("gguf");
        }
        if safetensors {
            std::fs::write(dir.path().join("model-00001.safetensors"), vec![0u8; 16])
                .expect("safetensors");
        }
        if let Some(model_type) = model_type {
            std::fs::write(
                dir.path().join("config.json"),
                format!(r#"{{"model_type":"{model_type}"}}"#),
            )
            .expect("config.json");
        }
        copy_tokenizer(dir.path());
        dir
    }

    fn args_for(dir: &Path) -> ModelServeArgs {
        ModelServeArgs {
            model_dir: dir.to_path_buf(),
            ..ModelServeArgs::default()
        }
    }

    #[test]
    fn a_flash_next_pack_is_named_by_its_model_type() {
        let flash = crate::family::spec::QWEN3_8_FLASH_NEXT;
        for model_type in flash.model_types {
            let pack = model_dir(Some(model_type), false, true);
            assert_eq!(flash_next_family(pack.path()), Some(flash.name));
            // The listing's special case holds only while serve --model
            // cannot start the pack itself.
            assert!(servable_family(pack.path()).is_err());
        }
        let dense = crate::family::QWEN3_8_27B_DENSE.model_types[0];
        let dense = model_dir(Some(dense), false, true);
        assert_eq!(flash_next_family(dense.path()), None);
        let unnamed = model_dir(None, false, true);
        assert_eq!(flash_next_family(unnamed.path()), None);
        assert_eq!(flash_next_family(Path::new("/nonexistent")), None);
    }

    #[test]
    fn parse_args_reads_the_surface_and_refuses_unknowns() {
        let args: Vec<String> = [
            "--model",
            "/m",
            "--family",
            "qwen3_6_35b_a3b",
            "--context",
            "4096",
            "--kv",
            "q8",
            "--listen",
            "127.0.0.1:9000",
            "--tokenizer-dir",
            "/t",
            "--print-config",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let parsed = parse_args(&args).expect("parses");
        assert_eq!(parsed.model_dir, PathBuf::from("/m"));
        assert_eq!(parsed.family.as_deref(), Some("qwen3_6_35b_a3b"));
        assert_eq!(parsed.context_limit, Some(4096));
        assert_eq!(parsed.kv_dtype.as_deref(), Some("q8"));
        assert_eq!(parsed.listen.as_deref(), Some("127.0.0.1:9000"));
        assert_eq!(parsed.tokenizer_dir, Some(PathBuf::from("/t")));
        assert!(parsed.print_config);

        let only_model = parse_args(&["--model".to_string(), "/m".to_string()]).expect("parses");
        assert_eq!(
            only_model,
            ModelServeArgs {
                model_dir: PathBuf::from("/m"),
                ..ModelServeArgs::default()
            }
        );

        let unknown = parse_args(&[
            "--model".to_string(),
            "/m".to_string(),
            "--turbo".to_string(),
        ])
        .unwrap_err();
        assert!(unknown.0.contains("unknown flag"), "{}", unknown.0);
        let bad_context = parse_args(&[
            "--model".to_string(),
            "/m".to_string(),
            "--context".to_string(),
            "huge".to_string(),
        ])
        .unwrap_err();
        assert!(bad_context.0.contains("token count"), "{}", bad_context.0);
        let missing_value = parse_args(&["--model".to_string()]).unwrap_err();
        assert!(
            missing_value.0.contains("needs a value"),
            "{}",
            missing_value.0
        );
    }

    fn safetensors_only_format() -> &'static str {
        if super::MLX_BUILD { "mlx" } else { "exl3" }
    }

    #[test]
    fn a_gguf_directory_is_detected_with_the_qualified_default_options() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.family, "qwen3_6_35b_a3b");
        assert_eq!(discovered.weight_format, "gguf");
        // Only the discovered format is set: context and KV stay the
        // resolver's own defaults — nothing experimental is armed.
        assert_eq!(
            discovered.options,
            Q36Options {
                weight_format: Some("gguf".to_string()),
                ..Q36Options::default()
            }
        );
    }

    #[test]
    fn the_dense_model_type_maps_to_the_dense_family() {
        let dir = model_dir(Some("qwen3_5"), false, true);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.family, "qwen3_8_27b_dense");
        assert_eq!(discovered.weight_format, safetensors_only_format());
    }

    #[test]
    fn an_exl3_only_directory_is_detected_as_exl3() {
        let dir = model_dir(Some("qwen3_5_moe"), false, true);
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(discovered.weight_format, safetensors_only_format());
        // A directory of safetensors shards is an MLX pack on an MLX build, which sets no
        // `weight_format` (the MLX arm refuses `exl3`), and an EXL3 pack elsewhere.
        assert_eq!(
            discovered.options.weight_format.as_deref(),
            (!super::MLX_BUILD).then_some("exl3")
        );
    }

    #[test]
    fn an_explicit_family_replaces_missing_evidence() {
        let dir = model_dir(None, true, false);
        let mut args = args_for(dir.path());
        args.family = Some("qwen3_6_35b_a3b".to_string());
        let discovered = discover(&args).expect("discovers");
        assert_eq!(discovered.family, "qwen3_6_35b_a3b");
    }

    #[test]
    fn flags_reach_the_options_verbatim() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.context_limit = Some(8192);
        args.kv_dtype = Some("q8".to_string());
        let discovered = discover(&args).expect("discovers");
        assert_eq!(discovered.options.context_limit, Some(8192));
        assert_eq!(discovered.options.kv_dtype.as_deref(), Some("q8"));
    }

    #[test]
    fn without_evidence_the_family_is_requested_not_guessed() {
        let dir = model_dir(None, true, false);
        let error = discover(&args_for(dir.path())).unwrap_err();
        assert!(error.0.contains("--family"), "{}", error.0);
        assert!(error.0.contains("qwen3_6_35b_a3b"), "{}", error.0);

        let unknown = model_dir(Some("llama"), true, false);
        let error = discover(&args_for(unknown.path())).unwrap_err();
        assert!(error.0.contains("--family"), "{}", error.0);
        assert!(error.0.contains("llama"), "{}", error.0);
    }

    #[test]
    fn flint_is_explicitly_not_simplified_yet() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.family = Some("flint".to_string());
        let error = discover(&args).unwrap_err();
        assert!(error.0.contains("no simplified startup"), "{}", error.0);
        assert!(error.0.contains("--config"), "{}", error.0);
    }

    #[test]
    fn a_directory_without_weights_is_refused() {
        let dir = model_dir(Some("qwen3_5_moe"), false, false);
        let error = discover(&args_for(dir.path())).unwrap_err();
        assert!(error.0.contains("no .gguf or .safetensors"), "{}", error.0);
    }

    #[test]
    fn a_missing_directory_is_reported_by_the_inspection() {
        let args = ModelServeArgs {
            model_dir: PathBuf::from("/nonexistent/fifi-model"),
            ..ModelServeArgs::default()
        };
        let error = discover(&args).unwrap_err();
        assert!(error.0.contains("not a directory"), "{}", error.0);
    }

    // Fails if discovery stops refusing a network-mounted model directory:
    // `discover` calls the inspection, which refuses before it lists the
    // directory. `build_config` runs `discover` first, so its refusal here is
    // the same discovery refusal; the later preflight call is not what this
    // test exercises, and the engine start is covered by the `start_engines`
    // tests.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_model_directory_on_a_network_mount_is_refused_at_discovery() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let args = args_for(dir.path());
        let error = crate::storage_guard::seam::with_nfs_at(dir.path(), || {
            (
                discover(&args).map(|_| ()).unwrap_err(),
                super::build_config(&args).map(|_| ()).unwrap_err(),
            )
        });
        assert!(
            error.0.0.contains("must be on local storage"),
            "{}",
            error.0.0
        );
        assert!(
            error.1.0.contains("must be on local storage"),
            "{}",
            error.1.0
        );
        assert!(
            discover(&args).is_ok(),
            "the same directory is fine on a local filesystem"
        );
    }

    #[test]
    fn the_printed_config_round_trips_through_the_file_path() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let mut args = args_for(dir.path());
        args.listen = Some("127.0.0.1:9100".to_string());
        let discovered = discover(&args).expect("discovers");
        let config = super::config_from(&discovered, &args);
        let printed = super::config_json(&config);

        let file = dir.path().join("printed.json");
        std::fs::write(&file, &printed).expect("write printed config");
        let reread = Config::from_file(&file).expect("the printed config is a valid file");
        assert_eq!(reread.listen, "127.0.0.1:9100");
        assert_eq!(reread.api_key_env, "ENGINESRV_API_KEY");
        assert_eq!(reread.prefix_reuse, Some(false));
        assert_eq!(reread.engines.len(), 1);
        let engine = &reread.engines[0];
        assert_eq!(engine.family, "qwen3_6_35b_a3b");
        assert_eq!(engine.model_dir, args.model_dir);
        assert_eq!(engine.qwen36, Some(discovered.options));
        // No extra fields can hide in the printed document: from_file
        // runs deny_unknown_fields, so the round-trip above is the
        // proof. The key variable is named (as in every config file);
        // its VALUE lives only in the environment.
        assert!(printed.contains("ENGINESRV_API_KEY"), "{printed}");
    }

    /// What each build does with the two fixture directories. A tokenizer that differs from the
    /// pin only warns, so it decides nothing here: the GGUF directory builds a configuration in a
    /// CUDA build, and the EXL3-only directory is refused by the weight-format check, quoted
    /// verbatim with preflight's own next-step line. Each build picks its own arm below: the CPU
    /// build's backend check, the CUDA build's weight-format check, the MLX build's settings
    /// validators. The green zero-config path is the CUDA build's, exercised by the hardware
    /// lane; the settings the CPU build WOULD resolve are covered by the discovery tests above
    /// and the parity test of `run`.
    #[test]
    fn build_config_refusals_quote_preflight_verbatim() {
        let dir = model_dir(Some("qwen3_5_moe"), true, false);
        let built = super::build_config(&args_for(dir.path()));
        if cfg!(feature = "native-mlx") {
            // A native-mlx build reaches the settings validators before the backend
            // choice and refuses the default-args fixture (no cache limit) — also a
            // verbatim preflight quote, so this arm pins that refusal instead. The
            // fixture holds no weight shard, so the host cannot derive a cache limit.
            let error = built.unwrap_err();
            assert!(error.0.contains("cannot serve"), "{}", error.0);
            assert!(
                error.0.contains("cache_limit_bytes is missing"),
                "{}",
                error.0
            );
            assert!(error.0.contains("next:"), "{}", error.0);
        } else if cfg!(all(feature = "native-cuda", target_os = "linux")) {
            // A Linux native-cuda build picks the CUDA backend. The fixture's tokenizer is not
            // the pinned one, which only warns, so the GGUF directory builds a configuration
            // for the qwen3_6_35b_a3b family with its GGUF weights.
            // (native-cuda is a no-op off Linux, engine/build.rs.)
            let config = built.expect("a GGUF directory builds a configuration");
            let engine = &config.engines[0];
            assert_eq!(engine.family, "qwen3_6_35b_a3b");
            assert_eq!(engine.model_dir, dir.path());
            let options = engine.qwen36.as_ref().expect("qwen36 options");
            assert_eq!(options.weight_format.as_deref(), Some("gguf"));
        } else {
            let error = built.unwrap_err();
            assert!(error.0.contains("cannot serve"), "{}", error.0);
            assert!(error.0.contains("no GPU backend"), "{}", error.0);
            assert!(error.0.contains("next:"), "{}", error.0);
        }

        // The EXL3-only directory: a CUDA build serves GGUF weights only and refuses it with
        // the weight-format check; an MLX build reads the shard as an MLX pack and builds a
        // configuration with no weight format member; the CPU build has no GPU backend.
        let exl3 = model_dir(Some("qwen3_5_moe"), false, true);
        let built = super::build_config(&args_for(exl3.path()));
        if cfg!(feature = "native-mlx") {
            let config = built.expect("an MLX build reads the shard as an MLX pack");
            let engine = &config.engines[0];
            assert_eq!(engine.family, "qwen3_6_35b_a3b");
            assert_eq!(engine.model_dir, exl3.path());
            let options = engine.qwen36.as_ref().expect("qwen36 options");
            assert_eq!(options.weight_format, None);
        } else {
            let error = built.unwrap_err();
            assert!(error.0.contains("cannot serve"), "{}", error.0);
            assert!(error.0.contains("next:"), "{}", error.0);
            if cfg!(all(feature = "native-cuda", target_os = "linux")) {
                assert!(error.0.contains("no .gguf weights file in"), "{}", error.0);
            } else {
                assert!(error.0.contains("no GPU backend"), "{}", error.0);
            }
        }
    }

    /// A directory of one tiny GGUF of the given architecture and nothing else.
    fn directory_of_architecture(architecture: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("model.gguf"),
            crate::tokenizer::gguf::tests::tiny_gguf(architecture, "gpt2", "raon"),
        )
        .expect("write");
        dir
    }

    #[test]
    fn every_architecture_of_the_shared_table_identifies_its_family() {
        for (architecture, family) in crate::server::config::GGUF_ARCHITECTURE_FAMILIES {
            let dir = directory_of_architecture(architecture);
            let discovered = discover(&args_for(dir.path()))
                .unwrap_or_else(|error| panic!("{architecture}: {}", error.0));
            assert_eq!(discovered.family, *family, "{architecture}");
            assert_eq!(discovered.weight_format, "gguf");
        }
    }

    #[test]
    fn an_explicit_gguf_discovers_its_own_family_and_reads_side_files_only_from_a_named_directory()
    {
        let dir = directory_of_architecture("qwen35moe");
        let file = dir.path().join("model.gguf");
        std::fs::write(
            dir.path().join("bolt.gguf"),
            crate::tokenizer::gguf::tests::tiny_gguf("gemma4", "gpt2", "raon"),
        )
        .expect("write another architecture");
        std::fs::write(dir.path().join("tokenizer.json"), "{}").expect("write tokenizer");
        let args = args_for(&file);
        let discovered = discover(&args).expect("discover explicit checkpoint");
        assert_eq!(
            discovered.family,
            crate::family::qwen::qwen36::factory::FAMILY
        );
        assert_eq!(
            discovered.tokenizer_dir, file,
            "the header supplies the vocabulary"
        );
        let named = ModelServeArgs {
            tokenizer_dir: Some(dir.path().to_path_buf()),
            ..args_for(&file)
        };
        let discovered = discover(&named).expect("discover with a named directory");
        let config = super::config_from(&discovered, &named);
        assert_eq!(config.engines[0].model_dir, file);
        assert_eq!(config.engines[0].tokenizer_dir, dir.path());
    }

    #[test]
    fn an_architecture_outside_the_table_is_refused_naming_it_and_the_known_ones() {
        for architecture in ["mamba9", "flint", "qwen4exp", "nemotron_h_moe", "raon1"] {
            let dir = directory_of_architecture(architecture);
            let error = discover(&args_for(dir.path())).unwrap_err().0;
            assert!(error.contains(architecture), "{error}");
            assert!(error.contains("known architectures: qwen35moe"), "{error}");
        }
    }

    /// The real Qwen3.6 GGUF (`QWEN36_GGUF_FILE`), reached through a symlink in a scratch
    /// directory (never in a model directory): no config.json and no --family, and the
    /// architecture names the family.
    #[test]
    #[ignore = "needs QWEN36_GGUF_FILE, the Qwen3.6 GGUF"]
    fn the_real_qwen36_gguf_alone_in_a_directory_starts_the_qwen36_family() {
        let real = std::env::var("QWEN36_GGUF_FILE").expect("QWEN36_GGUF_FILE names the file");
        let dir = tempfile::tempdir().expect("tempdir");
        std::os::unix::fs::symlink(&real, dir.path().join("model.gguf")).expect("symlink");
        let discovered = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(
            discovered.family,
            crate::family::qwen::qwen36::factory::FAMILY
        );
        assert_eq!(discovered.weight_format, "gguf");
    }

    #[test]
    fn a_row_outside_the_qwen_families_starts_from_a_bare_entry_that_prints_as_it_starts() {
        for (architecture, family) in crate::server::config::GGUF_ARCHITECTURE_FAMILIES {
            if super::uses_qwen36_options(family) {
                continue;
            }
            let dir = directory_of_architecture(architecture);
            let args = args_for(dir.path());
            let discovered = discover(&args).expect("discovers");
            let config = super::config_from(&discovered, &args);
            let engine = &config.engines[0];
            assert_eq!(engine.family, *family);
            assert!(engine.qwen36.is_none(), "{architecture}: a bare entry");
            let printed = super::config_json(&config);
            let path = dir.path().join("printed.json");
            std::fs::write(&path, &printed).expect("write");
            let again = Config::from_file(&path)
                .unwrap_or_else(|error| panic!("{architecture}: {}", error.0));
            assert_eq!(again.engines[0].family, *family);
            let mut with_flag = args_for(dir.path());
            with_flag.context_limit = Some(4096);
            let error = discover(&with_flag).unwrap_err().0;
            assert!(error.contains("--context is not an option"), "{error}");
        }
    }

    /// One tiny GGUF of `architecture` at `path`.
    fn write_architecture(path: &Path, architecture: &str) {
        std::fs::write(
            path,
            crate::tokenizer::gguf::tests::tiny_gguf(architecture, "gpt2", "raon"),
        )
        .expect("write");
    }

    #[test]
    fn a_bare_gguf_file_is_identified_by_its_own_architecture_not_its_neighbours() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_architecture(&dir.path().join("a.gguf"), "qwen35moe");
        write_architecture(&dir.path().join("b.gguf"), "gemma4");
        for (name, family) in [
            ("a.gguf", "qwen3_6_35b_a3b"),
            ("b.gguf", crate::family::gemma::gemma4::FAMILY),
        ] {
            let discovered = discover(&args_for(&dir.path().join(name))).expect("discovers");
            assert_eq!(discovered.family, family, "{name}");
            assert_eq!(discovered.weight_format, "gguf");
        }
    }

    #[test]
    fn an_ollama_blob_with_no_extension_is_served_like_any_gguf() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blob = dir.path().join("sha256-0f1e2d3c");
        write_architecture(&blob, "gemma4");
        let discovered = discover(&args_for(&blob)).expect("discovers");
        assert_eq!(discovered.family, crate::family::gemma::gemma4::FAMILY);
    }

    #[test]
    fn the_tokenizer_of_a_bare_file_is_the_file_even_with_a_tokenizer_json_beside_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("m.gguf");
        write_architecture(&file, "gemma4");
        assert_eq!(super::default_tokenizer_dir(&file), file);
        std::fs::write(dir.path().join("tokenizer.json"), "{}").expect("write");
        assert_eq!(super::default_tokenizer_dir(&file), file);
        // A directory is its own tokenizer directory.
        assert_eq!(super::default_tokenizer_dir(dir.path()), dir.path());
    }

    #[test]
    fn a_refused_architecture_is_named_with_its_reason_when_there_is_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (architecture, reason) in [
            ("raon1", "CUDA serving is not on main"),
            ("clip", "vision projector"),
            ("dflash", "draft model"),
            ("qwen4exp", "quantization kernels"),
        ] {
            let file = dir.path().join(format!("{architecture}.gguf"));
            write_architecture(&file, architecture);
            let error = discover(&args_for(&file)).unwrap_err().0;
            assert!(error.contains(reason), "{architecture}: {error}");
        }
    }

    #[test]
    fn a_gguf_that_names_no_architecture_is_refused_saying_so() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("mystery.gguf");
        std::fs::write(&file, minimal_gguf()).expect("write");
        let error = discover(&args_for(&file)).unwrap_err().0;
        assert!(error.contains("names no general.architecture"), "{error}");
    }

    #[test]
    fn two_models_in_one_directory_each_read_only_their_own_header() {
        // Side files beside a lone file may belong to the other model: neither the config.json
        // model_type nor the tokenizer.json of the directory decides for a named file.
        let dir = model_dir(Some("qwen3_5_moe"), false, false);
        write_architecture(&dir.path().join("a.gguf"), "qwen35moe");
        write_architecture(&dir.path().join("b.gguf"), "gemma4");
        let b = dir.path().join("b.gguf");
        let discovered = discover(&args_for(&b)).expect("discovers");
        assert_eq!(discovered.family, crate::family::gemma::gemma4::FAMILY);
        assert_eq!(discovered.tokenizer_dir, b);
        // Naming the directory still uses its files.
        let whole = discover(&args_for(dir.path())).expect("discovers");
        assert_eq!(whole.family, "qwen3_6_35b_a3b");
        assert_eq!(whole.tokenizer_dir, dir.path());
    }

    #[test]
    fn a_relative_path_and_a_symlink_name_the_model_like_its_absolute_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let real = dir.path().join("real").join("sha256-0f1e");
        std::fs::create_dir_all(real.parent().expect("parent")).expect("dirs");
        write_architecture(&real, "gemma4");
        let link = dir.path().join("bolt.gguf");
        std::os::unix::fs::symlink(&real, &link).expect("link");
        let relative = crate::test_paths::relative_to_cwd(&link);
        for path in [&real, &link, &relative] {
            let discovered = discover(&args_for(path)).expect("discovers");
            assert_eq!(
                discovered.family,
                crate::family::gemma::gemma4::FAMILY,
                "{path:?}"
            );
            assert_eq!(discovered.tokenizer_dir, *path, "{path:?}");
        }
    }

    #[test]
    fn a_link_into_a_crowded_directory_takes_its_own_header_and_none_of_the_neighbours() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = dir.path().join("store");
        std::fs::create_dir_all(&store).expect("store");
        let real = store.join("sha256-0f1e");
        write_architecture(&real, "gemma4");
        write_architecture(&store.join("other.gguf"), "qwen35moe");
        write_architecture(&store.join("another.gguf"), "qwen35moe");
        std::fs::write(store.join("tokenizer.json"), b"{}").expect("tokenizer");
        let shelf = dir.path().join("shelf");
        std::fs::create_dir_all(&shelf).expect("shelf");
        let link = shelf.join("bolt.gguf");
        std::os::unix::fs::symlink(&real, &link).expect("link");
        let discovered = discover(&args_for(&link)).expect("discovers");
        assert_eq!(discovered.family, crate::family::gemma::gemma4::FAMILY);
        // The side files come from the link's own directory, not the target's.
        assert_eq!(discovered.tokenizer_dir, link);
    }

    #[test]
    fn a_dangling_plain_name_is_named_a_dangling_link_not_a_store_miss() {
        // A relative name resolves against the working directory, so the link goes there under a
        // name no other test uses and is removed on every exit path.
        struct Remove(PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let name = format!("dangling-{}.gguf", std::process::id());
        let _guard = Remove(PathBuf::from(&name));
        std::os::unix::fs::symlink("no-such-target", &name).expect("link");
        let error = super::build_config(&args_for(Path::new(&name)))
            .unwrap_err()
            .0;
        assert!(error.contains("dangling symlink"), "{error}");
        assert!(!error.contains("nothing local is named"), "{error}");
    }

    fn stores_in(root: &Path) -> crate::server::local_models::Stores {
        crate::server::local_models::Stores {
            models: root.join("models"),
            ollama: root.join("ollama"),
            hub: root.join("hub"),
        }
    }

    #[test]
    fn serve_with_no_model_names_none_and_parses() {
        let args =
            parse_args(&["--listen".to_string(), "127.0.0.1:9".to_string()]).expect("parses");
        assert!(args.model_dir.as_os_str().is_empty());
        assert_eq!(parse_args(&[]).expect("parses"), ModelServeArgs::default());
    }

    #[test]
    fn the_default_model_is_the_newest_one_a_family_serves() {
        let root = tempfile::tempdir().expect("tempdir");
        let models = root.path().join("models");
        for name in ["old", "new", "newest"] {
            std::fs::create_dir_all(models.join(name)).expect("dir");
        }
        write_architecture(&models.join("old").join("old.gguf"), "qwen35moe");
        std::thread::sleep(std::time::Duration::from_millis(30));
        write_architecture(&models.join("new").join("new.gguf"), "gemma4");
        std::thread::sleep(std::time::Duration::from_millis(30));
        // The newest file is not a model any family serves: it is skipped, with its reason.
        write_architecture(
            &models.join("newest").join("newest.gguf"),
            "no-such-architecture",
        );
        let mut said = Vec::new();
        let chosen = super::resolve_default_model(
            &ModelServeArgs::default(),
            &stores_in(root.path()),
            &mut |line| said.push(line.to_string()),
        )
        .expect("chosen");
        assert_eq!(chosen.model_dir, models.join("new"));
        assert!(
            said.iter().any(|line| line.contains("skipping newest")),
            "{said:?}"
        );
        let last = said.last().expect("a line");
        assert!(
            last.contains("serving new ") && last.contains("--model <name>"),
            "{last}"
        );
    }

    #[test]
    fn with_nothing_local_the_default_start_says_where_to_get_a_model() {
        let root = tempfile::tempdir().expect("tempdir");
        let error = super::resolve_default_model(
            &ModelServeArgs::default(),
            &stores_in(root.path()),
            &mut |_| {},
        )
        .unwrap_err()
        .0;
        assert!(error.contains("no --model given"), "{error}");
        assert!(error.contains("--model <org/repo>"), "{error}");
    }
}
