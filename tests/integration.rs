//! Integration tests for a3s-power core workflows.
//!
//! These tests exercise the HTTP API surface using axum's `oneshot` test
//! infrastructure with a mock backend, verifying that all components (router,
//! handlers, registry, storage, state) work together correctly.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::util::ServiceExt;

use a3s_power::backend::test_utils::MockBackend;
use a3s_power::backend::BackendRegistry;
use a3s_power::config::PowerConfig;
use a3s_power::model::manifest::{ModelFormat, ModelManifest, ModelParameters};
use a3s_power::model::registry::ModelRegistry;
use a3s_power::model::storage;
use a3s_power::server::router;
use a3s_power::server::state::AppState;
use a3s_power::tee::attestation::{DefaultTeeProvider, TeeType};
use std::sync::Arc;

// ============================================================================
// Helpers
// ============================================================================

/// Create an AppState with a mock backend and isolated data directory.
fn isolated_state(dir: &std::path::Path) -> AppState {
    std::env::set_var("A3S_POWER_HOME", dir);
    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(MockBackend::success()));
    AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(backends),
        Arc::new(PowerConfig::default()),
    )
}

/// Create a manifest with a real blob on disk so storage operations work.
fn manifest_with_blob(dir: &std::path::Path, name: &str, data: &[u8]) -> ModelManifest {
    std::env::set_var("A3S_POWER_HOME", dir);
    let (blob_path, sha256) = storage::store_blob(data).unwrap();
    ModelManifest {
        name: name.to_string(),
        format: ModelFormat::Gguf,
        size: data.len() as u64,
        sha256,
        parameters: Some(ModelParameters {
            context_length: Some(4096),
            embedding_length: Some(3200),
            parameter_count: Some(3_000_000_000),
            quantization: Some("Q4_K_M".to_string()),
        }),
        created_at: chrono::Utc::now(),
        path: blob_path,
        system_prompt: None,
        template_override: None,
        default_parameters: None,
        modelfile_content: None,
        license: None,
        adapter_path: None,
        projector_path: None,
        messages: vec![],
        family: Some("llama".to_string()),
        families: None,
    }
}

/// Send a JSON POST request and return (status, body_json).
async fn post_json(
    app: axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, json)
}

/// Send a GET request and return (status, body_json).
async fn get_json(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = if bytes.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null)
    };
    (status, json)
}

// ============================================================================
// Health & Metrics
// ============================================================================

#[tokio::test]
async fn test_health_returns_ok() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let (status, json) = get_json(app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());
    assert!(json["uptime_seconds"].is_number());
    assert!(json["loaded_models"].is_number());
}

#[tokio::test]
async fn test_metrics_returns_prometheus_format() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let req = Request::builder()
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("# HELP power_http_requests_total"));
    assert!(text.contains("power_models_loaded"));
}

// ============================================================================
// /v1/models
// ============================================================================

#[tokio::test]
async fn test_v1_models_empty() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let (status, json) = get_json(app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    assert!(json["data"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_v1_models_with_registered_model() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "test-model", b"fake-gguf-data");
    state.registry.register(manifest).unwrap();
    let app = router::build(state);
    let (status, json) = get_json(app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "test-model");
}

// ============================================================================
// /v1/chat/completions
// ============================================================================

#[tokio::test]
async fn test_v1_chat_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let (_, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not found"));
}

#[tokio::test]
async fn test_v1_chat_non_streaming_success() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "test", b"fake-gguf");
    state.registry.register(manifest).unwrap();
    state.mark_loaded("test");
    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "test");
    assert!(json["choices"][0]["message"]["content"].is_string());
}

// ============================================================================
// /v1/completions
// ============================================================================

#[tokio::test]
async fn test_v1_completions_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let (_, json) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "nonexistent",
            "prompt": "hi"
        }),
    )
    .await;
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not found"));
}

#[tokio::test]
async fn test_v1_completions_non_streaming_success() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "test", b"fake-gguf");
    state.registry.register(manifest).unwrap();
    state.mark_loaded("test");
    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "test",
            "prompt": "hi",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "test");
    assert!(json["choices"][0]["text"].is_string());
    assert!(json["usage"].is_object());
}

// ============================================================================
// Storage & Registry
// ============================================================================

#[tokio::test]
async fn test_model_register_and_list() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let m1 = manifest_with_blob(dir.path(), "model-a", b"data-a");
    let m2 = manifest_with_blob(dir.path(), "model-b", b"data-b");
    state.registry.register(m1).unwrap();
    state.registry.register(m2).unwrap();
    assert_eq!(state.registry.count(), 2);
    assert!(state.registry.get("model-a").is_ok());
    assert!(state.registry.get("model-b").is_ok());
}

#[tokio::test]
async fn test_model_delete_from_registry() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let m = manifest_with_blob(dir.path(), "to-delete", b"data");
    state.registry.register(m).unwrap();
    assert_eq!(state.registry.count(), 1);
    state.registry.remove("to-delete").unwrap();
    assert_eq!(state.registry.count(), 0);
    assert!(state.registry.get("to-delete").is_err());
}

// ============================================================================
// 404 for removed Ollama routes
// ============================================================================

#[tokio::test]
async fn test_ollama_routes_return_404() {
    let dir = tempfile::tempdir().unwrap();

    // All old Ollama native API routes should be gone
    for path in &[
        "/api/generate",
        "/api/chat",
        "/api/tags",
        "/api/pull",
        "/api/push",
        "/api/blobs/sha256:abc",
        "/api/embeddings",
        "/api/embed",
        "/api/ps",
        "/api/copy",
        "/api/create",
        "/api/version",
    ] {
        let req = Request::builder().uri(*path).body(Body::empty()).unwrap();
        let app_clone = router::build(isolated_state(dir.path()));
        let resp = app_clone.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "Expected 404 for removed route: {}",
            path
        );
    }
}

// ============================================================================
// Unknown route
// ============================================================================

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let req = Request::builder()
        .uri("/nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// /v1/attestation
// ============================================================================

#[tokio::test]
async fn test_attestation_returns_503_without_tee() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);
    let (status, json) = get_json(app, "/v1/attestation").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json["error"]["code"], "tee_not_enabled");
}

#[tokio::test]
async fn test_attestation_returns_report_with_tee() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());
    let provider = DefaultTeeProvider::with_type(TeeType::Simulated);
    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(MockBackend::success()));
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(backends),
        Arc::new(PowerConfig {
            tee_mode: true,
            ..Default::default()
        }),
    )
    .with_tee_provider(Arc::new(provider));
    let app = router::build(state);
    let (status, json) = get_json(app, "/v1/attestation").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["tee_type"], "simulated");
    assert!(json["report_data"].is_string());
    assert!(json["measurement"].is_string());
    assert!(json["timestamp"].is_string());
}
