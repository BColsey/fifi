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
use super::state::{AppState, RemoveRefused, engine_error_info};
use super::wire::state_name;
use crate::EngError;

/// GET /v1/engines — every engine with the fields the GUI's model
/// page needs: the directory the weights came from, so a cache
/// snapshot can be matched to the engine serving it.
pub(crate) async fn list(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let engines: Vec<serde_json::Value> = state
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
    let model_dir = PathBuf::from(body.model_dir.trim());
    if model_dir.as_os_str().is_empty() {
        return Err(EngError::BadRequest("model_dir is required".into()).into());
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
    let id = match &body.id {
        Some(id) => id.clone(),
        None => default_id(&model_dir, &state),
    };
    if state.engine_exists(&id) {
        return Err(EngError::BadRequest(format!(
            "engine {id} already exists; pass a different id"
        ))
        .into());
    }
    let derivation = tokio::task::spawn_blocking(move || {
        let args = crate::server::model_serve::ModelServeArgs {
            model_dir,
            tokenizer_dir: body.tokenizer_dir.map(PathBuf::from),
            family: body.family.clone(),
            context_limit: body.context_limit,
            kv_dtype: body.kv_dtype.clone(),
            listen: None,
            print_config: false,
        };
        let mut config = crate::server::model_serve::build_config(&args)?;
        let mut engines = std::mem::take(&mut config.engines);
        if engines.is_empty() {
            return Err(crate::server::StartupError(
                "the derived configuration carries no engine".into(),
            ));
        }
        // `build_config` always assembles exactly one engine; the
        // guard above keeps that assumption honest if it ever changes.
        let engine_config = engines.swap_remove(0);
        Ok::<_, crate::server::StartupError>((config.prefix_reuse, engine_config))
    })
    .await
    .map_err(|error| EngError::BadRequest(format!("engine derivation failed: {error}")))?
    .map_err(|error| EngError::BadRequest(error.to_string()))?;
    let (file_prefix_reuse, mut engine_config) = derivation;
    engine_config.id = id.clone();
    let started = tokio::task::spawn_blocking(move || {
        crate::server::run::start_one(&engine_config, file_prefix_reuse)
    })
    .await
    .map_err(|error| EngError::BadRequest(format!("engine start failed: {error}")))?
    .map_err(|error| EngError::BadRequest(error.to_string()))?;
    let (engine, entry) = started;
    let Some(shared) = state.add_engine(entry, engine) else {
        return Err(EngError::BadRequest(format!(
            "engine {id} already exists; pass a different id"
        ))
        .into());
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

/// The engine id when the request carries none: the directory's own
/// name, and when that is taken, the first free `-2`, `-3`, …
/// suffix. Derived from the snapshot directory of a hub download,
/// this is the repo name — readable in the picker without any
/// configuration.
fn default_id(model_dir: &Path, state: &AppState) -> String {
    let base = model_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("engine")
        .to_string();
    if !state.engine_exists(&base) {
        return base;
    }
    let mut counter = 2;
    while state.engine_exists(&format!("{base}-{counter}")) {
        counter += 1;
    }
    format!("{base}-{counter}")
}

/// DELETE /v1/engines/{id} — stop and remove one runtime-added
/// engine. In-flight and queued jobs on it are cancelled; sessions
/// bound to it rebind on their next request.
pub(crate) async fn remove(
    State(state): State<Arc<AppState>>,
    UrlPath(id): UrlPath<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let id_for_stop = id.clone();
    let stopped = tokio::task::spawn_blocking(move || state.remove_engine(&id_for_stop))
        .await
        .map_err(|error| EngError::BadRequest(format!("engine stop join failed: {error}")))?;
    match stopped {
        Ok((entry, report)) => Ok(Json(serde_json::json!({
            "id": id,
            "model": entry.model,
            "state": "stopped",
            "shutdown_failure": report.failure,
        }))),
        Err(RemoveRefused::BootEngine) => Err(EngError::BadRequest(format!(
            "engine {id} was started by the configuration; only engines added at runtime can \
             be removed"
        ))
        .into()),
        Err(RemoveRefused::Unknown) => {
            Err(ServerError::NotFound(format!("engine {id} does not exist")))
        }
    }
}
