use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
#[cfg(feature = "hf")]
use futures::StreamExt;
use serde::Deserialize;
use std::collections::BTreeMap;

use crate::api::types::{ModelInfo, ModelList};
#[cfg(feature = "hf")]
use crate::error::PowerError;
use crate::model::manifest::ModelFormat;
#[cfg(feature = "hf")]
use crate::model::manifest::ModelManifest;
#[cfg(feature = "hf")]
use crate::model::registry::ModelRegistry;
use crate::server::state::AppState;

mod registration;

#[cfg(test)]
use registration::dir_size;
pub use registration::{register_handler, RegisterExternalDraftRequest, RegisterModelRequest};

/// GET /v1/models - OpenAI-compatible model listing.
pub async fn list_handler(State(state): State<AppState>) -> impl IntoResponse {
    match state.registry.list() {
        Ok(models) => {
            let model_infos: Vec<ModelInfo> = models
                .iter()
                .map(|m| ModelInfo {
                    id: m.name.clone(),
                    object: "model".to_string(),
                    created: m.created_at.timestamp(),
                    owned_by: "local".to_string(),
                    root: None,
                    parent: None,
                    context_length: m.parameters.as_ref().and_then(|p| p.context_length),
                    format: Some(model_format_name(&m.format).to_string()),
                    size_bytes: Some(m.size),
                    sha256: (!m.sha256.is_empty()).then(|| m.sha256.clone()),
                })
                .collect();

            Json(ModelList {
                object: "list".to_string(),
                data: model_infos,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "server_error",
                    "code": null
                }
            })),
        )
            .into_response(),
    }
}

/// GET /v1/models/:name - Retrieve a single model by name.
pub async fn get_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.registry.get(&name) {
        Ok(m) => Json(ModelInfo {
            id: m.name.clone(),
            object: "model".to_string(),
            created: m.created_at.timestamp(),
            owned_by: "local".to_string(),
            root: None,
            parent: None,
            context_length: m.parameters.as_ref().and_then(|p| p.context_length),
            format: Some(model_format_name(&m.format).to_string()),
            size_bytes: Some(m.size),
            sha256: (!m.sha256.is_empty()).then(|| m.sha256.clone()),
        })
        .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("model '{}' not found", name),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response(),
    }
}

fn model_format_name(format: &ModelFormat) -> &'static str {
    match format {
        ModelFormat::Gguf => "gguf",
        ModelFormat::SafeTensors => "safetensors",
        ModelFormat::HuggingFace => "huggingface",
        ModelFormat::Vision => "vision",
        ModelFormat::Remote => "remote",
    }
}

/// DELETE /v1/models/:name - Remove a model from the registry.
///
/// Does not delete the model file from disk; only deregisters it.
pub async fn delete_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    // Unload from backend if currently loaded, using the model's actual format.
    if state.is_model_loaded(&name) {
        let manifest = state.registry.get(&name).ok();
        let backend_result = match manifest.as_ref() {
            Some(manifest) => state.find_backend_for_manifest(manifest),
            None => state.backends.find_for_format(&ModelFormat::Gguf),
        };
        let backend = match backend_result {
            Ok(backend) => backend,
            Err(e) => {
                return super::openai_error(
                    "backend_unavailable",
                    &state.sanitize_error(&e.to_string()),
                )
                .into_response();
            }
        };
        if let Err(e) = backend.unload(&name).await {
            tracing::warn!(model = %name, error = %e, "Failed to unload model before deletion");
            return super::openai_error("server_error", &state.sanitize_error(&e.to_string()))
                .into_response();
        }
        state.mark_unloaded(&name);
    }

    match state.registry.remove(&name) {
        Ok(_) => (
            StatusCode::OK,
            Json(serde_json::json!({ "deleted": true, "id": name, "object": "model" })),
        )
            .into_response(),
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("model '{}' not found", name),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response(),
    }
}

/// Request body for POST /v1/models/pull.
#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    /// Model name to pull.
    ///
    /// Supported formats:
    /// - `owner/repo:Q4_K_M`          — resolves quantization via HF API
    /// - `owner/repo/file.gguf`        — direct filename
    pub name: String,
    /// If true, re-download even if already registered.
    #[serde(default)]
    pub force: bool,
    /// Model hub API token for private/gated models.
    /// Falls back to `MODELSCOPE_TOKEN` / `A3S_POWER_HUB_TOKEN` / `HF_TOKEN`.
    #[serde(default)]
    pub token: Option<String>,
    /// Unknown pull fields are preserved so handlers can reject unsupported hub
    /// policy instead of silently dropping it.
    #[serde(default, flatten)]
    pub unsupported: BTreeMap<String, serde_json::Value>,
}

impl PullModelRequest {
    fn unsupported_fields(&self) -> Vec<&str> {
        self.unsupported.keys().map(String::as_str).collect()
    }

    fn unsupported_fields_message(&self) -> Option<String> {
        let fields = self.unsupported_fields();
        if fields.is_empty() {
            None
        } else {
            Some(format!(
                "unsupported model pull field(s): {}; supported fields are name, force, and token",
                fields.join(", ")
            ))
        }
    }
}

#[cfg(feature = "hf")]
fn save_initial_pull_state(name: &str) {
    let ps = crate::model::pull_state::PullState::new(name);
    if let Err(e) = ps.save() {
        tracing::warn!(
            model = %name,
            error = %e,
            "Failed to persist initial pull state"
        );
    }
}

#[cfg(feature = "hf")]
fn mark_pull_state_done(name: &str) {
    let Some(mut ps) = crate::model::pull_state::PullState::load(name) else {
        tracing::warn!(model = %name, "Pull state missing while marking pull as done");
        return;
    };

    if let Err(e) = ps.mark_done() {
        tracing::warn!(
            model = %name,
            error = %e,
            "Failed to mark pull state as done"
        );
    }
}

#[cfg(feature = "hf")]
fn mark_pull_state_failed(name: &str, error_message: &str) {
    let Some(mut ps) = crate::model::pull_state::PullState::load(name) else {
        tracing::warn!(
            model = %name,
            error = %error_message,
            "Pull state missing while marking pull as failed"
        );
        return;
    };

    if let Err(e) = ps.mark_failed(error_message) {
        tracing::warn!(
            model = %name,
            error = %e,
            pull_error = %error_message,
            "Failed to mark pull state as failed"
        );
    }
}

#[cfg(feature = "hf")]
fn update_pull_state_progress(name: &str, completed: u64, total: u64) {
    let Some(mut ps) = crate::model::pull_state::PullState::load(name) else {
        tracing::warn!(
            model = %name,
            completed,
            total,
            "Pull state missing while updating progress"
        );
        return;
    };

    if let Err(e) = ps.update_progress(completed, total) {
        tracing::warn!(
            model = %name,
            completed,
            total,
            error = %e,
            "Failed to update pull state progress"
        );
    }
}

#[cfg(feature = "hf")]
fn register_pulled_manifest(
    registry: &ModelRegistry,
    manifest: ModelManifest,
    force: bool,
    pull_name: &str,
) -> bool {
    let manifest_name = manifest.name.clone();

    if force {
        match registry.remove(&manifest_name) {
            Ok(_) | Err(PowerError::ModelNotFound(_)) => {}
            Err(e) => {
                tracing::warn!(
                    model = %manifest_name,
                    error = %e,
                    "Failed to remove existing model before forced pull registration"
                );
            }
        }
    }

    match registry.register(manifest) {
        Ok(()) => {
            mark_pull_state_done(pull_name);
            true
        }
        Err(e) => {
            let message = format!("model registry update failed: {e}");
            tracing::error!(
                model = %manifest_name,
                error = %e,
                "Failed to register pulled model manifest"
            );
            mark_pull_state_failed(pull_name, &message);
            false
        }
    }
}

/// POST /v1/models/pull — Pull a GGUF model from remote model hub.
///
/// Streams SSE progress events while downloading:
/// ```json
/// {"status":"downloading","completed":104857600,"total":2147483648}
/// {"status":"verifying"}
/// {"status":"success","id":"owner/repo:Q4_K_M","object":"model","created":1234567890}
/// ```
///
/// Returns 200 with `{"status":"already_exists"}` if the model is already
/// registered and `force` is false.
///
/// Requires the `hf` feature; returns 501 otherwise.
pub async fn pull_handler(
    State(state): State<AppState>,
    Json(req): Json<PullModelRequest>,
) -> impl IntoResponse {
    if let Some(message) = req.unsupported_fields_message() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "unsupported_request_fields"
                }
            })),
        )
            .into_response();
    }

    // Fast path: already registered and not forcing.
    if !req.force && state.registry.exists(&req.name) {
        return axum::response::Sse::new(futures::stream::once(async move {
            Ok::<_, std::convert::Infallible>(super::sse_json_event(&serde_json::json!({
                "status": "already_exists",
                "name": req.name
            })))
        }))
        .into_response();
    }

    // Concurrent pull guard: reject duplicate in-flight pulls for the same model.
    if state.is_pulling(&req.name) {
        return axum::response::Sse::new(futures::stream::once(async move {
            Ok::<_, std::convert::Infallible>(super::sse_json_event(&serde_json::json!({
                "status": "already_pulling",
                "name": req.name
            })))
        }))
        .into_response();
    }

    #[cfg(feature = "hf")]
    {
        use crate::model::pull::hf::{pull, PullProgress};
        use tokio_stream::wrappers::ReceiverStream;

        let (tx, rx) = tokio::sync::mpsc::channel::<PullProgress>(32);
        let name = req.name.clone();
        let token = req.token.clone();
        let registry = state.registry.clone();
        let force = req.force;

        // Mark as in-flight before spawning.
        if !state.start_pull(&name) {
            return axum::response::Sse::new(futures::stream::once(async move {
                Ok::<_, std::convert::Infallible>(super::sse_json_event(&serde_json::json!({
                    "status": "already_pulling",
                    "name": req.name
                })))
            }))
            .into_response();
        }
        let state_for_cleanup = state.clone();
        let name_for_cleanup = name.clone();

        // Persist initial pull state.
        save_initial_pull_state(&name);

        // Spawn download task; progress flows through the channel.
        tokio::spawn(async move {
            let result = pull(&name, token.as_deref(), tx.clone()).await;
            // Always release the pull lock, success or failure.
            state_for_cleanup.finish_pull(&name_for_cleanup);
            match result {
                Ok(manifest) => {
                    register_pulled_manifest(registry.as_ref(), manifest, force, &name_for_cleanup);
                }
                Err(e) => {
                    tracing::error!(error = %e, model = %name_for_cleanup, "model pull failed");
                    mark_pull_state_failed(&name_for_cleanup, &e.to_string());
                }
            }
        });

        let pull_name = req.name.clone();
        let stream = ReceiverStream::new(rx).map(move |progress| {
            // Persist progress to disk on Downloading events (throttled to every 5%).
            if let PullProgress::Downloading { completed, total } = &progress {
                if *total > 0 {
                    let pct = completed * 100 / total;
                    let prev_pct = completed.saturating_sub(1024 * 1024) * 100 / total;
                    if pct / 5 != prev_pct / 5 {
                        update_pull_state_progress(&pull_name, *completed, *total);
                    }
                }
            }
            let event = match progress {
                PullProgress::Resuming { offset, total } => {
                    super::sse_json_event(&serde_json::json!({
                        "status": "resuming",
                        "offset": offset,
                        "total": total
                    }))
                }
                PullProgress::Downloading { completed, total } => {
                    super::sse_json_event(&serde_json::json!({
                        "status": "downloading",
                        "completed": completed,
                        "total": total
                    }))
                }
                PullProgress::Verifying => super::sse_json_event(&serde_json::json!({
                    "status": "verifying"
                })),
                PullProgress::Done => super::sse_json_event(&serde_json::json!({
                    "status": "success",
                    "id": req.name,
                    "object": "model",
                    "created": chrono::Utc::now().timestamp()
                })),
            };
            Ok::<_, std::convert::Infallible>(event)
        });

        axum::response::Sse::new(stream).into_response()
    }

    #[cfg(not(feature = "hf"))]
    {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "message": "model pull requires the 'hf' feature (recompile with --features hf)",
                    "type": "server_error",
                    "code": "not_implemented"
                }
            })),
        )
            .into_response()
    }
}

/// GET /v1/models/pull/:name/status — Query the persisted state of a pull operation.
///
/// Returns the last known state of a pull (pulling, done, or failed).
/// Useful after a server restart to check whether a previous download completed.
///
/// Returns 404 if no pull state exists for the given model name.
pub async fn pull_status_handler(Path(name): Path<String>) -> impl IntoResponse {
    use crate::model::pull_state::PullState;

    match PullState::load(&name) {
        Some(state) => (StatusCode::OK, Json(state)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("no pull state found for model '{name}'"),
                    "type": "not_found",
                    "code": "pull_state_not_found"
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests;
