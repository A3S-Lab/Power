use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
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
    // Unload from backend if currently loaded
    if state.is_model_loaded(&name) {
        if let Ok(backend) = state
            .backends
            .find_for_format(&crate::model::manifest::ModelFormat::Gguf)
        {
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

    if !path.exists() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": format!("file not found: {}", req.path),
                    "type": "invalid_request_error",
                    "code": "file_not_found"
                }
            })),
        )
            .into_response();
    }

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    let manifest = ModelManifest {
        name: req.name.clone(),
        format: ModelFormat::Gguf,
        size,
        sha256: String::new(),
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

/// Request body for POST /v1/models/pull.
#[derive(Debug, Deserialize)]
pub struct PullModelRequest {
    /// Model name to pull (e.g. "llama3.2:3b").
    pub name: String,
    /// If true, re-download even if already registered.
    #[serde(default)]
    pub force: bool,
}

/// POST /v1/models/pull - Pull a model from the Ollama registry.
///
/// Delegates to the configured pull backend. Returns 501 if no pull
/// backend is available (pull support is backend-specific).
pub async fn pull_handler(
    State(state): State<AppState>,
    Json(req): Json<PullModelRequest>,
) -> impl IntoResponse {
    // If already registered and not forcing, return early.
    if !req.force && state.registry.exists(&req.name) {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "already_exists",
                "name": req.name
            })),
        )
            .into_response();
    }

    // Delegate to backend pull if supported.
    match state.backends.find_for_format(&ModelFormat::Gguf) {
        Ok(backend) => match backend.pull(&req.name).await {
            Ok(manifest) => {
                let name = manifest.name.clone();
                let created = manifest.created_at.timestamp();
                // Register the pulled manifest (ignore duplicate errors on force).
                let _ = state.registry.register(manifest);
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "status": "success",
                        "id": name,
                        "object": "model",
                        "created": created,
                        "owned_by": "local"
                    })),
                )
                    .into_response()
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("pull failed: {e}"),
                        "type": "server_error",
                        "code": "pull_failed"
                    }
                })),
            )
                .into_response(),
        },
        Err(_) => (
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": {
                    "message": "no backend available for model pull",
                    "type": "server_error",
                    "code": "not_implemented"
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
        let app = router::build(state);
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
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["status"], "already_exists");
    }

    #[tokio::test]
    async fn test_pull_model_backend_not_implemented() {
        // MockBackend doesn't implement pull, so we expect 501
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
        // Either 501 (no pull support) or 502 (pull attempted but failed)
        assert!(
            resp.status() == StatusCode::NOT_IMPLEMENTED
                || resp.status() == StatusCode::BAD_GATEWAY
        );
    }
}
