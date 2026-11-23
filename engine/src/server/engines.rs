//! The runtime engine API: list, add, remove.
//!
//! `POST /v1/engines` starts one engine in the running process
//! through the exact boot path — discovery, preflight, admission on
//! the engine thread — so a model the GUI downloaded to the hub
//! cache is served without a restart. The load is asynchronous the
//! way boot is: the answer carries the id, and the state surfaces
//! (`/v1/models`, `/v1/engines`) report the loader's own stages and
//! failures. Only engines this API added can be removed by it; the
//! boot configuration's engines live and die with the process.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use serde::Deserialize;

use super::error::ServerError;
use super::state::{AddRefused, AppState, MAX_RUNTIME_ENGINES, RemoveRefused, engine_error_info};
use super::wire::state_name;
use crate::EngError;

/// GET /v1/engines — every engine with the fields the GUI's model
/// page needs: the directory the weights came from, so a cache
/// snapshot can be matched to the engine serving it.
pub(crate) async fn list(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let mut engines: Vec<serde_json::Value> = state
        .entries()
        .iter()
        .map(|entry| {
            let mut row = serde_json::json!({
                "id": entry.id,
                "model": entry.model,
                "model_dir": entry.model_dir.display().to_string(),
                "state": state_name(entry.handle.state()),
                "context_limit": entry.context_limit,
                "runtime_added": state.owns_engine(&entry.id),
                "memory_held_until_restart": entry.memory_held_until_restart(),
            });
            let cells = row.as_object_mut().expect("row is an object");
            // The stage and the failure are present only when there is
            // one, as on the models list: a refused start says why.
            if let Some(stage) = entry.handle.load_stage() {
                cells.insert("load_stage".into(), stage.into());
            }
            if let Some(error) = engine_error_info(entry)
                && let Ok(error) = serde_json::to_value(error)
            {
                cells.insert("error".into(), error);
            }
            row
        })
        .collect();
    // The supervised workers ride the same list, with the state
    // their own health poll reported.
    for worker in state.workers().snapshot() {
        let mut row = serde_json::json!({
            "id": worker.id,
            "model": worker.model,
            "model_dir": worker.model_dir.display().to_string(),
            "state": worker.status.state,
            "runtime_added": true,
            "memory_held_until_restart": false,
            "supervised": true,
        });
        let cells = row.as_object_mut().expect("row is an object");
        if let Some(stage) = worker.status.load_stage {
            cells.insert("load_stage".into(), stage.into());
        }
        if let Some(error) = worker.status.error {
            cells.insert("error".into(), serde_json::json!({ "message": error }));
        }
        if let Some(context) = worker.status.context_limit {
            cells.insert("context_limit".into(), context.into());
        }
        engines.push(row);
    }
    Json(serde_json::json!({ "engines": engines }))
}

#[derive(Deserialize)]
pub(crate) struct AddEngineRequest {
    /// The weights directory: a hub-cache snapshot path, any local
    /// model directory, or `org/repo`, which resolves to the cached
    /// snapshot the way `serve --model org/repo` does.
    model_dir: String,
    id: Option<String>,
    family: Option<String>,
    context_limit: Option<u32>,
    kv_dtype: Option<String>,
    tokenizer_dir: Option<String>,
}

/// An engine id is one plain token, never a dotted name: it becomes a header value, a log
/// field and a URL segment.
pub(crate) fn valid_engine_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && !id.starts_with('.')
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// The message of a refused derivation without the caller's paths: the
/// cause stays, the directory names become `<model_dir>` and
/// `<tokenizer_dir>`, so the answer does not map the host's filesystem.
fn scrub_paths(message: &str, model_dir: &Path, tokenizer_dir: Option<&Path>) -> String {
    let mut text = message.to_string();
    if let Some(dir) = tokenizer_dir {
        text = text.replace(&dir.display().to_string(), "<tokenizer_dir>");
    }
    text.replace(&model_dir.display().to_string(), "<model_dir>")
}

/// POST /v1/engines — start one engine at runtime.
///
/// The derivation runs on a blocking thread: `build_config` is the
/// CLI's own path and reads the model directory, and the runtime's
/// workers must not pay for that. `start_one` is the boot loop's
/// body, so every boot-time check and note applies unchanged.
pub(crate) async fn add(
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddEngineRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ServerError> {
    let requested = body.model_dir.trim().to_string();
    if requested.is_empty() {
        return Err(EngError::BadRequest("model_dir is required".into()).into());
    }
    if let Some(id) = &body.id
        && !valid_engine_id(id)
    {
        return Err(EngError::BadRequest(
            "id must be 1 to 64 characters of letters, digits, '.', '_' or '-'".into(),
        )
        .into());
    }
    if state.runtime_engine_count() >= MAX_RUNTIME_ENGINES {
        return Err(EngError::BadRequest(format!(
            "{MAX_RUNTIME_ENGINES} engines added at runtime are already running; stop one first"
        ))
        .into());
    }
    #[cfg(all(feature = "native-mlx", target_os = "macos", target_arch = "aarch64"))]
    if crate::family::mlx_budget::enabled() {
        // The boot path's one-MLX-model rule, held at runtime too:
        // auto-budget is process-wide and Metal reports one shared
        // working set, so a second MLX model cannot be admitted here.
        return Err(EngError::BadRequest(
            "FIFI_MLX_AUTO_BUDGET=1 or FIFI_MLX_WORKING_SET_MIB requires one MLX model per \
             server; restart with a different model instead"
                .into(),
        )
        .into());
    }
    // An `org/repo` id names the cached snapshot of its `main` ref, read
    // from this server's own hub cache.
    let mut model_dir = PathBuf::from(&requested);
    if !model_dir.exists()
        && let Some(snapshot) = state.hf().main_snapshot(&requested)
    {
        model_dir = snapshot;
    }
    let id = match &body.id {
        Some(id) => id.clone(),
        None => default_id(&model_dir, &requested, &state),
    };
    if state.engine_exists(&id) {
        return Err(EngError::BadRequest(format!(
            "engine {id} already exists; pass a different id"
        ))
        .into());
    }
    let tokenizer_dir = body.tokenizer_dir.clone().map(PathBuf::from);
    let derive = state.derive();
    let scrub_model = model_dir.clone();
    let scrub_tokenizer = tokenizer_dir.clone();
    // Under supervision the request names a family that `serve --model`
    // cannot derive (Flash-Next has no simplified startup) instead of
    // leaving it to detection: the worker reads the bare entry a boot
    // configuration carries.
    let bare_family = body.family.clone().filter(|family| {
        state.supervised_engines()
            && super::run::holds_memory_until_restart(family)
            && !crate::server::model_serve::DETECTABLE_FAMILIES
                .iter()
                .any(|(name, _)| name == family)
    });
    // The hub page adds an engine by directory alone, so a pack the request
    // names no family for is recognised by what it holds. A family the
    // request does name stands.
    let bare_family = bare_family.or_else(|| {
        (body.family.is_none() && state.supervised_engines())
            .then(|| super::model_serve::flash_next_family(&model_dir))
            .flatten()
            .map(str::to_string)
    });
    let derivation = tokio::task::spawn_blocking(move || {
        let args = crate::server::model_serve::ModelServeArgs {
            model_dir,
            tokenizer_dir,
            family: body.family.clone(),
            context_limit: body.context_limit,
            kv_dtype: body.kv_dtype.clone(),
            listen: None,
            print_config: false,
        };
        let mut config = match &bare_family {
            Some(family) => {
                if !args.model_dir.is_dir() {
                    return Err(crate::server::StartupError(format!(
                        "model_dir {} is not a directory",
                        args.model_dir.display()
                    )));
                }
                crate::server::model_serve::bare_config(&args, family)
            }
            None => derive(&args)?,
        };
        let mut engines = std::mem::take(&mut config.engines);
        if engines.is_empty() {
            return Err(crate::server::StartupError(
                "the derived configuration carries no engine".into(),
            ));
        }
        // `build_config` always assembles exactly one engine; the
        // guard above keeps that assumption honest if it ever changes.
        let engine_config = engines.swap_remove(0);
        Ok::<_, crate::server::StartupError>((config, engine_config))
    })
    .await
    .map_err(|_| EngError::BadRequest("engine derivation failed".into()))?
    .map_err(|error| {
        EngError::BadRequest(scrub_paths(
            &error.to_string(),
            &scrub_model,
            scrub_tokenizer.as_deref(),
        ))
    })?;
    let (mut config, mut engine_config) = derivation;
    engine_config.id = id.clone();
    if state.supervised_engines() && super::run::holds_memory_until_restart(&engine_config.family) {
        // A process-lifetime family starts a supervised worker, so
        // stopping the engine ends the process and frees its memory:
        // the refusal below belongs to the in-process path only.
        let model_name = engine_config.model.clone().unwrap_or_default();
        let model_dir = engine_config.model_dir.display().to_string();
        // The worker reads its engine from this configuration: back in
        // it goes, for the in-process path alone took it out.
        config.engines.push(engine_config);
        return state
            .workers()
            .spawn(&id, config, Some(model_name.clone()))
            .map_err(|error| {
                ServerError::from(EngError::BadRequest(scrub_paths(
                    &error,
                    &scrub_model,
                    scrub_tokenizer.as_deref(),
                )))
            })
            .map(|()| {
                (
                    StatusCode::CREATED,
                    Json(serde_json::json!({
                        "id": id,
                        "model": model_name,
                        "model_dir": model_dir,
                        "state": "starting",
                        "supervised": true,
                    })),
                )
            });
    }
    refuse_pinned_family(&state, &engine_config.family)?;
    let started = tokio::task::spawn_blocking(move || {
        crate::server::run::start_one(&engine_config, config.prefix_reuse)
    })
    .await
    .map_err(|_| EngError::BadRequest("engine start failed".into()))?
    .map_err(|error| {
        EngError::BadRequest(scrub_paths(
            &error.to_string(),
            &scrub_model,
            scrub_tokenizer.as_deref(),
        ))
    })?;
    let (engine, entry) = started;
    let shared = match state.try_add_engine(entry, engine) {
        Ok(shared) => shared,
        Err(AddRefused::IdTaken) => {
            return Err(EngError::BadRequest(format!(
                "engine {id} already exists; pass a different id"
            ))
            .into());
        }
        Err(AddRefused::TooMany) => {
            return Err(EngError::BadRequest(format!(
                "{MAX_RUNTIME_ENGINES} engines added at runtime are already running; stop one first"
            ))
            .into());
        }
    };
    if !cfg!(test) {
        crate::server::run::spawn_stage_reporter(std::slice::from_ref(&shared));
    }
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": shared.id,
            "model": shared.model,
            "model_dir": shared.model_dir.display().to_string(),
            "state": state_name(shared.handle.state()),
        })),
    ))
}

/// Refuses a start whose family keeps its memory for the life of the
/// process once an engine of it ran: the answer names the engine and
/// the way out, instead of contending for memory the family still holds.
fn refuse_pinned_family(state: &AppState, family: &str) -> Result<(), EngError> {
    if !super::run::holds_memory_until_restart(family) {
        return Ok(());
    }
    let Some((holder, running)) = state.pinned_holder(family) else {
        return Ok(());
    };
    Err(EngError::BadRequest(if running {
        format!(
            "the {family} family keeps its memory for the life of the process and engine \
             {holder} already serves it; this server runs one engine of the family"
        )
    } else {
        format!(
            "engine {holder} of the {family} family was stopped, but the family keeps its \
             memory until the server process exits: restart the server to free card memory, \
             then start the engine again"
        )
    }))
}

/// The engine id when the request carries none: the repo name of a hub
/// cache snapshot (its path is `models--org--name/snapshots/<commit>`,
/// so the commit directory is not a name), or the directory's own name,
/// or the repo part of an `org/name` request. When that is taken, the
/// first free `-2`, `-3`, … suffix; characters outside the id rule
/// become `-`.
fn default_id(model_dir: &Path, requested: &str, state: &AppState) -> String {
    let hub_repo = model_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("models--"))
        .and_then(|name| name.rsplit("--").next());
    let raw = hub_repo
        .or_else(|| {
            (!model_dir.exists() && requested.contains('/'))
                .then(|| requested.rsplit('/').next())
                .flatten()
        })
        .or_else(|| model_dir.file_name().and_then(|name| name.to_str()))
        .unwrap_or("engine");
    let mut base: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .take(56)
        .collect();
    if base.is_empty() {
        base = "engine".to_string();
    }
    if !state.engine_exists(&base) {
        return base;
    }
    let mut counter = 2;
    while state.engine_exists(&format!("{base}-{counter}")) {
        counter += 1;
    }
    format!("{base}-{counter}")
}

/// What a stop of a process-lifetime family answers: the jobs end and
/// the engine leaves the lists, but its memory stays allocated.
pub(crate) const HELD_MEMORY_NOTE: &str =
    "stopped; card memory returns when the server process restarts";

/// DELETE /v1/engines/{id} — stop and remove one runtime-added
/// engine. In-flight and queued jobs on it are cancelled; sessions
/// bound to it rebind on their next request. The answer says whether
/// the stop returned the engine's memory.
pub(crate) async fn remove(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    if !valid_engine_id(&id) {
        return Err(ServerError::NotFound("no such engine".into()));
    }
    if state.workers().contains(&id) {
        // A supervised worker: its process ends, so the memory it
        // held is genuinely freed — the honest answer the in-process
        // stop cannot give.
        let model = state
            .workers()
            .get(&id)
            .and_then(|worker| worker.model.clone());
        state.workers().stop(&id);
        state.drop_sessions_for(&id);
        return Ok(Json(serde_json::json!({
            "id": id,
            "model": model.unwrap_or_default(),
            "state": "stopped",
            "memory_released": true,
            "supervised": true,
        })));
    }
    let id_for_stop = id.clone();
    let stopped = tokio::task::spawn_blocking(move || state.remove_engine(&id_for_stop))
        .await
        .map_err(|_| EngError::BadRequest("engine stop did not finish".into()))?;
    match stopped {
        Ok((entry, report)) => {
            let held = entry.memory_held_until_restart();
            let mut body = serde_json::json!({
                "id": id,
                "model": entry.model,
                "state": "stopped",
                "shutdown_failure": report.failure,
                "memory_released": !held,
            });
            if held {
                body["memory_held_until_restart"] = true.into();
                body["note"] = HELD_MEMORY_NOTE.into();
            }
            Ok(Json(body))
        }
        Err(RemoveRefused::BootEngine) => Err(EngError::BadRequest(format!(
            "engine {id} was started by the configuration; only engines added at runtime can \
             be removed"
        ))
        .into()),
        Err(RemoveRefused::Unknown) => Err(ServerError::NotFound("no such engine".into())),
        Err(RemoveRefused::ShuttingDown) => {
            Err(EngError::BadRequest(format!("engine {id} is shutting down")).into())
        }
    }
}
