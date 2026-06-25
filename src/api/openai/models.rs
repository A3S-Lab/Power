use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
#[cfg(feature = "hf")]
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::api::types::{ModelInfo, ModelList};
#[cfg(feature = "hf")]
use crate::error::PowerError;
use crate::model::manifest::{ModelFormat, ModelManifest};
#[cfg(feature = "hf")]
use crate::model::registry::ModelRegistry;
use crate::server::state::AppState;

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

/// DELETE /v1/models/:name - Remove a model from the registry.
///
/// Does not delete the model file from disk; only deregisters it.
pub async fn delete_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    // Unload from backend if currently loaded, using the model's actual format.
    if state.is_model_loaded(&name) {
        let format = state
            .registry
            .get(&name)
            .map(|m| m.format.clone())
            .unwrap_or(ModelFormat::Gguf);
        let backend = match state.backends.find_for_format(&format) {
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

/// Request body for POST /v1/models - register a local model file.
#[derive(Debug, Deserialize)]
pub struct RegisterModelRequest {
    /// Display name for the model.
    pub name: String,
    /// Absolute path to the model file on disk.
    pub path: String,
    /// Model format: "gguf" (default) or "safetensors".
    #[serde(default)]
    pub format: Option<String>,
}

/// Response body for POST /v1/models.
#[derive(Debug, Serialize)]
pub struct RegisterModelResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// POST /v1/models - Register a local GGUF model file.
pub async fn register_handler(
    State(state): State<AppState>,
    Json(req): Json<RegisterModelRequest>,
) -> impl IntoResponse {
    let path = std::path::PathBuf::from(&req.path);

    let format = match req.format.as_deref().unwrap_or("gguf") {
        "safetensors" => ModelFormat::SafeTensors,
        "huggingface" => ModelFormat::HuggingFace,
        _ => ModelFormat::Gguf,
    };

    if !path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("path not found: {}", req.path),
                    "type": "invalid_request_error",
                    "code": "file_not_found"
                }
            })),
        )
            .into_response();
    }

    // HuggingFace models are directories; skip SHA-256 (no single file to hash).
    // For file-based formats, compute SHA-256 for TEE integrity checks.
    let (size, sha256) = if format == ModelFormat::HuggingFace {
        if !path.is_dir() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("huggingface model path must be a directory: {}", req.path),
                        "type": "invalid_request_error",
                        "code": "not_a_directory"
                    }
                })),
            )
                .into_response();
        }
        let size = match dir_size(&path) {
            Ok(size) => size,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("failed to inspect huggingface model directory {}: {e}", req.path),
                            "type": "invalid_request_error",
                            "code": "path_unreadable"
                        }
                    })),
                )
                    .into_response();
            }
        };
        (size, String::new())
    } else {
        let metadata = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("failed to inspect model file {}: {e}", req.path),
                            "type": "invalid_request_error",
                            "code": "path_unreadable"
                        }
                    })),
                )
                    .into_response();
            }
        };

        if !metadata.is_file() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("model path must be a file: {}", req.path),
                        "type": "invalid_request_error",
                        "code": "not_a_file"
                    }
                })),
            )
                .into_response();
        }

        let size = metadata.len();
        let sha256 = match crate::model::storage::compute_sha256_file(&path) {
            Ok(h) => h,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("failed to hash model file: {e}"),
                            "type": "server_error",
                            "code": "hash_failed"
                        }
                    })),
                )
                    .into_response();
            }
        };
        (size, sha256)
    };

    let manifest = ModelManifest {
        name: req.name.clone(),
        format,
        size,
        sha256,
        parameters: None,
        created_at: chrono::Utc::now(),
        path,
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        projector_path: None,
        messages: vec![],
        family: None,
        families: None,
    };

    match state.registry.register(manifest) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "id": req.name,
                "object": "model",
                "created": chrono::Utc::now().timestamp(),
                "owned_by": "local"
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "invalid_request_error",
                    "code": "model_already_exists"
                }
            })),
        )
            .into_response(),
    }
}

/// Compute total size of a directory by summing all file sizes.
fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = std::fs::metadata(&path)?;
        let size = if metadata.is_dir() {
            dir_size(&path)?
        } else {
            metadata.len()
        };
        total = total
            .checked_add(size)
            .ok_or_else(|| std::io::Error::other("directory size overflow"))?;
    }
    Ok(total)
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
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    #[serial]
    async fn test_list_models_returns_ok() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("model-a")).unwrap();
        state.registry.register(sample_manifest("model-b")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_list_models_empty_registry() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_get_model_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("llama3")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models/llama3")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["id"], "llama3");
        assert_eq!(json["object"], "model");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_get_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    #[serial]
    async fn test_delete_model_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("to-delete"))
            .unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/v1/models/to-delete")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["deleted"], true);
        assert_eq!(json["id"], "to-delete");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_delete_loaded_model_keeps_registry_when_unload_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::unload_fails());
        state
            .registry
            .register(sample_manifest("stays-loaded"))
            .unwrap();
        state.mark_loaded("stays-loaded");

        let app = router::build(state.clone());
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/v1/models/stays-loaded")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(state.is_model_loaded("stays-loaded"));
        assert!(state.registry.get("stays-loaded").is_ok());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_delete_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method(Method::DELETE)
            .uri("/v1/models/ghost")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "my-model",
            "path": "/nonexistent/path/model.gguf"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "file_not_found");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Create a real file to register
        let model_file = dir.path().join("local.gguf");
        std::fs::write(&model_file, b"fake weights").unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state.clone());
        let body = serde_json::json!({
            "name": "local-model",
            "path": model_file.to_str().unwrap()
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["id"], "local-model");
        assert_eq!(json["object"], "model");

        // Verify SHA-256 was computed and stored (non-empty hash in the registry)
        let manifest = state.registry.get("local-model").unwrap();
        assert_eq!(manifest.size, b"fake weights".len() as u64);
        assert!(
            !manifest.sha256.is_empty(),
            "register_handler must compute and store SHA-256"
        );
        assert_eq!(
            manifest.sha256.len(),
            64,
            "SHA-256 hex string must be 64 characters"
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_file_format_requires_file() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let model_dir = dir.path().join("not-a-file-model");
        std::fs::create_dir_all(&model_dir).unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "bad-local-model",
            "path": model_dir.to_str().unwrap()
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "not_a_file");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_dir_size_sums_nested_files() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        std::fs::write(nested.join("weights.safetensors"), b"weights").unwrap();

        let size = super::dir_size(dir.path()).unwrap();

        assert_eq!(size, 9);
    }

    #[test]
    fn test_dir_size_reports_read_dir_errors() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not-a-dir");
        std::fs::write(&file_path, b"weights").unwrap();

        let err = super::dir_size(&file_path).unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::NotADirectory);
    }

    #[tokio::test]
    #[serial]
    async fn test_pull_model_already_exists_no_force() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("existing"))
            .unwrap();

        let app = router::build(state);
        let body = serde_json::json!({ "name": "existing" });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models/pull")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // pull_handler returns SSE; read the raw body and check for already_exists.
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&bytes);
        assert!(body_str.contains("already_exists"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_pull_model_backend_not_implemented() {
        // With the hf feature, pull_handler streams SSE (200 OK) and spawns a
        // background download task. Without hf, it returns 501.
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({ "name": "new-model" });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models/pull")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        #[cfg(feature = "hf")]
        // hf feature: SSE stream starts immediately (200 OK)
        assert_eq!(resp.status(), StatusCode::OK);
        #[cfg(not(feature = "hf"))]
        // no hf feature: 501 Not Implemented
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }

    #[tokio::test]
    #[serial]
    async fn test_pull_model_already_pulling() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        assert!(state.start_pull("busy-model"));

        let app = router::build(state);
        let body = serde_json::json!({ "name": "busy-model" });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models/pull")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&bytes);
        assert!(body_str.contains("already_pulling"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_pull_status_found_for_encoded_model_name() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mut pull_state = crate::model::pull_state::PullState::new("owner/repo:Q4_K_M");
        pull_state.update_progress(1024, 4096).unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models/pull/owner%2Frepo%3AQ4_K_M/status")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["name"], "owner/repo:Q4_K_M");
        assert_eq!(json["status"], "pulling");
        assert_eq!(json["completed"], 1024);
        assert_eq!(json["total"], 4096);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_pull_status_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .uri("/v1/models/pull/missing/status")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "pull_state_not_found");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[cfg(feature = "hf")]
    #[test]
    #[serial]
    fn test_register_pulled_manifest_marks_state_failed_when_registry_write_fails() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let pull_name = "owner/repo:Q4_K_M";
        crate::model::pull_state::PullState::new(pull_name)
            .save()
            .unwrap();

        std::fs::write(dir.path().join("models"), b"not a directory").unwrap();

        let registry = crate::model::registry::ModelRegistry::new();
        assert!(!super::register_pulled_manifest(
            &registry,
            sample_manifest(pull_name),
            false,
            pull_name
        ));

        let state = crate::model::pull_state::PullState::load(pull_name).unwrap();
        assert_eq!(state.status, crate::model::pull_state::PullStatus::Failed);
        assert!(state
            .error
            .as_deref()
            .is_some_and(|error| error.contains("model registry update failed")));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_safetensors_format() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let model_file = dir.path().join("model.safetensors");
        std::fs::write(&model_file, b"fake safetensors weights").unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state.clone());
        let body = serde_json::json!({
            "name": "my-safetensors",
            "path": model_file.to_str().unwrap(),
            "format": "safetensors"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let manifest = state.registry.get("my-safetensors").unwrap();
        assert_eq!(
            manifest.format,
            crate::model::manifest::ModelFormat::SafeTensors
        );
        assert_eq!(manifest.size, b"fake safetensors weights".len() as u64);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_huggingface_format() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // HuggingFace models are directories
        let model_dir = dir.path().join("my-embedding-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), b"{}").unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state.clone());
        let body = serde_json::json!({
            "name": "my-embedding",
            "path": model_dir.to_str().unwrap(),
            "format": "huggingface"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let manifest = state.registry.get("my-embedding").unwrap();
        assert_eq!(
            manifest.format,
            crate::model::manifest::ModelFormat::HuggingFace
        );
        assert_eq!(manifest.size, 4);
        // SHA-256 is empty for directory-based models
        assert!(manifest.sha256.is_empty());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[cfg(unix)]
    #[tokio::test]
    #[serial]
    async fn test_register_model_huggingface_rejects_unreadable_directory_entry() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let model_dir = dir.path().join("broken-embedding-model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        std::os::unix::fs::symlink(
            model_dir.join("missing.safetensors"),
            model_dir.join("weights.safetensors"),
        )
        .unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "bad-embedding",
            "path": model_dir.to_str().unwrap(),
            "format": "huggingface"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "path_unreadable");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_register_model_huggingface_requires_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Pass a file path instead of a directory
        let file_path = dir.path().join("model.bin");
        std::fs::write(&file_path, b"weights").unwrap();

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let body = serde_json::json!({
            "name": "bad-embedding",
            "path": file_path.to_str().unwrap(),
            "format": "huggingface"
        });
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/models")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "not_a_directory");

        std::env::remove_var("A3S_POWER_HOME");
    }
}
