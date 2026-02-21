use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
#[cfg(feature = "hf")]
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::api::types::{ModelInfo, ModelList};
use crate::model::manifest::{ModelFormat, ModelManifest};
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
        if let Ok(backend) = state.backends.find_for_format(&format) {
            let _ = backend.unload(&name).await;
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
        let size = dir_size(&path);
        (size, String::new())
    } else {
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
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
fn dir_size(path: &std::path::Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(|e| e.ok())
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                dir_size(&p)
            } else {
                std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
            }
        })
        .sum()
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
    /// HuggingFace API token for private/gated models.
    /// Falls back to the `HF_TOKEN` environment variable if not set.
    #[serde(default)]
    pub token: Option<String>,
}

/// POST /v1/models/pull — Pull a GGUF model from HuggingFace Hub.
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
            Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .json_data(serde_json::json!({
                        "status": "already_exists",
                        "name": req.name
                    }))
                    .unwrap(),
            )
        }))
        .into_response();
    }

    // Concurrent pull guard: reject duplicate in-flight pulls for the same model.
    if state.is_pulling(&req.name) {
        return axum::response::Sse::new(futures::stream::once(async move {
            Ok::<_, std::convert::Infallible>(
                axum::response::sse::Event::default()
                    .json_data(serde_json::json!({
                        "status": "already_pulling",
                        "name": req.name
                    }))
                    .unwrap(),
            )
        }))
        .into_response();
    }

    #[cfg(feature = "hf")]
    {
        use crate::model::pull::hf::{pull, PullProgress};
        use axum::response::sse::Event;
        use tokio_stream::wrappers::ReceiverStream;

        let (tx, rx) = tokio::sync::mpsc::channel::<PullProgress>(32);
        let name = req.name.clone();
        let token = req.token.clone();
        let registry = state.registry.clone();
        let force = req.force;

        // Mark as in-flight before spawning.
        state.start_pull(&name);
        let state_for_cleanup = state.clone();
        let name_for_cleanup = name.clone();

        // Persist initial pull state.
        {
            use crate::model::pull_state::PullState;
            let ps = PullState::new(&name);
            let _ = ps.save();
        }

        // Spawn download task; progress flows through the channel.
        tokio::spawn(async move {
            let result = pull(&name, token.as_deref(), tx.clone()).await;
            // Always release the pull lock, success or failure.
            state_for_cleanup.finish_pull(&name_for_cleanup);
            match result {
                Ok(manifest) => {
                    use crate::model::pull_state::PullState;
                    if force {
                        let _ = registry.remove(&manifest.name);
                    }
                    let _ = registry.register(manifest);
                    // Mark state as done; clean up after successful registration.
                    if let Some(mut ps) = PullState::load(&name_for_cleanup) {
                        let _ = ps.mark_done();
                    }
                }
                Err(e) => {
                    use crate::model::pull_state::PullState;
                    tracing::error!(error = %e, model = %name_for_cleanup, "model pull failed");
                    if let Some(mut ps) = PullState::load(&name_for_cleanup) {
                        let _ = ps.mark_failed(&e.to_string());
                    }
                }
            }
        });

        let pull_name = req.name.clone();
        let stream = ReceiverStream::new(rx).map(move |progress| {
            // Persist progress to disk on Downloading events (throttled to every 5%).
            if let PullProgress::Downloading { completed, total } = &progress {
                use crate::model::pull_state::PullState;
                if *total > 0 {
                    let pct = completed * 100 / total;
                    let prev_pct = completed.saturating_sub(1024 * 1024) * 100 / total;
                    if pct / 5 != prev_pct / 5 {
                        if let Some(mut ps) = PullState::load(&pull_name) {
                            let _ = ps.update_progress(*completed, *total);
                        }
                    }
                }
            }
            let event = match progress {
                PullProgress::Resuming { offset, total } => Event::default()
                    .json_data(serde_json::json!({
                        "status": "resuming",
                        "offset": offset,
                        "total": total
                    }))
                    .unwrap(),
                PullProgress::Downloading { completed, total } => Event::default()
                    .json_data(serde_json::json!({
                        "status": "downloading",
                        "completed": completed,
                        "total": total
                    }))
                    .unwrap(),
                PullProgress::Verifying => Event::default()
                    .json_data(serde_json::json!({ "status": "verifying" }))
                    .unwrap(),
                PullProgress::Done => Event::default()
                    .json_data(serde_json::json!({
                        "status": "success",
                        "id": req.name,
                        "object": "model",
                        "created": chrono::Utc::now().timestamp()
                    }))
                    .unwrap(),
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
    async fn test_pull_model_already_exists_no_force() {
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
        // SHA-256 is empty for directory-based models
        assert!(manifest.sha256.is_empty());

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

/// GET /v1/models/pull/:name/status — Query the persisted state of a pull operation.
///
/// Returns the last known state of a pull (pulling, done, or failed).
/// Useful after a server restart to check whether a previous download completed.
///
/// Returns 404 if no pull state exists for the given model name.
pub async fn pull_status_handler(Path(name): Path<String>) -> impl IntoResponse {
    use crate::model::pull_state::PullState;

    match PullState::load(&name) {
        Some(state) => {
            (StatusCode::OK, Json(serde_json::to_value(&state).unwrap())).into_response()
        }
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
