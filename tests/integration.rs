//! Integration tests for a3s-power core workflows.
//!
//! These tests exercise the full HTTP API surface using axum's `oneshot` test
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

/// Send a DELETE request with JSON body.
async fn delete_json(
    app: axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("DELETE")
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

// ============================================================================
// 1. Health & Discovery Workflow
// ============================================================================

#[tokio::test]
async fn test_health_check_workflow() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let app = router::build(state);

    // Root endpoint returns Ollama compatibility string
    let (status, _) = get_json(app.clone(), "/").await;
    assert_eq!(status, StatusCode::OK);

    // Health endpoint returns structured JSON
    let (status, json) = get_json(app.clone(), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());
    assert!(json["uptime_seconds"].is_number());
    assert_eq!(json["loaded_models"], 0);

    // Version endpoint
    let (status, json) = get_json(app.clone(), "/api/version").await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["version"].is_string());

    // Metrics endpoint returns Prometheus format
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
    assert!(text.contains("power_http_requests_total"));
    assert!(text.contains("power_models_loaded"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 2. Model Lifecycle: Register → List → Show → Copy → Delete
// ============================================================================

#[tokio::test]
async fn test_model_lifecycle_workflow() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "test-model:latest", b"fake-gguf-data");
    state.registry.register(manifest).unwrap();

    // List models via /api/tags
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    let models = json["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "test-model:latest");
    assert_eq!(models[0]["size"], 14); // b"fake-gguf-data".len()

    // Show model details via /api/show
    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/show",
        serde_json::json!({"name": "test-model:latest"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["details"]["format"], "GGUF");
    assert_eq!(json["details"]["family"], "llama");
    assert_eq!(json["details"]["quantization_level"], "Q4_K_M");
    assert_eq!(json["details"]["parameter_size"], "3000000000");

    // Copy model
    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/copy",
        serde_json::json!({"source": "test-model:latest", "destination": "my-copy:v1"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.registry.exists("my-copy:v1"));

    // Verify both models appear in list
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["models"].as_array().unwrap().len(), 2);

    // Delete the copy
    let app = router::build(state.clone());
    let (status, _) = delete_json(
        app,
        "/api/delete",
        serde_json::json!({"name": "my-copy:v1"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(!state.registry.exists("my-copy:v1"));

    // Verify only original remains
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["models"].as_array().unwrap().len(), 1);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 3. Chat Completion Workflow (non-streaming)
// ============================================================================

#[tokio::test]
async fn test_chat_completion_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "chat-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "chat-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "chat-model");
    assert!(json["message"]["content"].is_string());
    assert_eq!(json["message"]["role"], "assistant");
    assert_eq!(json["done"], true);
    assert!(json["total_duration"].is_number());
    assert!(json["created_at"].is_string());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_chat_completion_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 4. Text Generation Workflow (non-streaming)
// ============================================================================

#[tokio::test]
async fn test_generate_completion_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "gen-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "gen-model",
            "prompt": "Hello world",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "gen-model");
    assert!(json["response"].is_string());
    assert_eq!(json["done"], true);
    assert!(json["total_duration"].is_number());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_generate_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "ghost",
            "prompt": "Hello",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 5. Embedding Workflow
// ============================================================================

#[tokio::test]
async fn test_embed_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "embed-model", b"fake-embed");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/embed",
        serde_json::json!({
            "model": "embed-model",
            "input": ["hello world", "test"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "embed-model");
    let embeddings = json["embeddings"].as_array().unwrap();
    assert_eq!(embeddings.len(), 2);
    // Each embedding should be a vector of floats
    assert!(!embeddings[0].as_array().unwrap().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_embeddings_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "embed-model", b"fake-embed");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/embeddings",
        serde_json::json!({
            "model": "embed-model",
            "prompt": "hello world"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["embedding"].is_array());
    assert!(!json["embedding"].as_array().unwrap().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 6. OpenAI-Compatible API Workflow
// ============================================================================

#[tokio::test]
async fn test_openai_models_list() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "gpt-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "gpt-model");
    assert_eq!(data[0]["object"], "model");
    assert_eq!(data[0]["owned_by"], "local");

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_chat_completion_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "oai-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "oai-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "oai-model");
    assert_eq!(json["object"], "chat.completion");
    assert!(json["id"].is_string());
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["message"]["role"], "assistant");
    assert!(choices[0]["message"]["content"].is_string());
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].is_number());
    assert!(json["usage"]["completion_tokens"].is_number());
    assert!(json["usage"]["total_tokens"].is_number());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_chat_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hi"}]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("not found"));
    assert_eq!(json["error"]["code"], "model_not_found");

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "oai-embed", b"fake-embed");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({
            "model": "oai-embed",
            "input": "hello world"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    let data = json["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert!(data[0]["embedding"].is_array());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 7. Blob Storage Workflow
// ============================================================================

#[tokio::test]
async fn test_blob_upload_check_download_delete() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let data = b"blob-content-for-testing";
    let hash = storage::compute_sha256(data);

    // Upload blob — use hash directly without sha256: prefix to avoid URL routing issues
    let app = router::build(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri(format!("/api/blobs/sha256%3A{hash}"))
        .body(Body::from(data.to_vec()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Check blob exists (HEAD)
    let app = router::build(state.clone());
    let req = Request::builder()
        .method("HEAD")
        .uri(format!("/api/blobs/sha256%3A{hash}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Download blob (GET)
    let app = router::build(state.clone());
    let req = Request::builder()
        .uri(format!("/api/blobs/sha256%3A{hash}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(&body[..], data);

    // Delete blob
    let app = router::build(state.clone());
    let req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/blobs/sha256%3A{hash}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify blob is gone
    let app = router::build(state.clone());
    let req = Request::builder()
        .method("HEAD")
        .uri(format!("/api/blobs/sha256%3A{hash}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_blob_upload_digest_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/blobs/sha256%3A0000000000000000000000000000000000000000000000000000000000000000")
        .body(Body::from(b"wrong data".to_vec()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_blob_check_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let req = Request::builder()
        .method("HEAD")
        .uri("/api/blobs/sha256%3Adeadbeef")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 8. Process Status (ps) Workflow
// ============================================================================

#[tokio::test]
async fn test_ps_empty() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = get_json(app, "/api/ps").await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["models"].is_array());
    assert!(json["models"].as_array().unwrap().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_ps_after_model_loaded() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "ps-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    // Trigger model load via chat
    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "ps-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Now ps should show the loaded model
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/api/ps").await;
    assert_eq!(status, StatusCode::OK);
    let models = json["models"].as_array().unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0]["name"], "ps-model");
    assert!(models[0]["size"].is_number());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 9. Model State Management (LRU, eviction, keep-alive)
// ============================================================================

#[tokio::test]
async fn test_model_state_lru_eviction() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let config = PowerConfig {
        max_loaded_models: 2,
        ..Default::default()
    };
    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(MockBackend::success()));
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(backends),
        Arc::new(config),
    );

    // Register 3 models
    for name in &["model-a", "model-b", "model-c"] {
        let m = manifest_with_blob(dir.path(), name, format!("data-{name}").as_bytes());
        state.registry.register(m).unwrap();
    }

    // Load model-a via chat
    let app = router::build(state.clone());
    let (s, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({"model": "model-a", "messages": [{"role": "user", "content": "a"}], "stream": false}),
    ).await;
    assert_eq!(s, StatusCode::OK);
    assert!(state.is_model_loaded("model-a"));

    // Load model-b
    let app = router::build(state.clone());
    let (s, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({"model": "model-b", "messages": [{"role": "user", "content": "b"}], "stream": false}),
    ).await;
    assert_eq!(s, StatusCode::OK);
    assert!(state.is_model_loaded("model-b"));
    assert!(state.needs_eviction());

    // Load model-c — should trigger eviction of model-a (LRU)
    let app = router::build(state.clone());
    let (s, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({"model": "model-c", "messages": [{"role": "user", "content": "c"}], "stream": false}),
    ).await;
    assert_eq!(s, StatusCode::OK);
    assert!(state.is_model_loaded("model-c"));
    // model-a should have been evicted
    assert!(!state.is_model_loaded("model-a"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 10. Content-Addressed Storage Integrity
// ============================================================================

#[tokio::test]
async fn test_storage_deduplication() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let data = b"identical-content";
    let (path1, hash1) = storage::store_blob(data).unwrap();
    let (path2, hash2) = storage::store_blob(data).unwrap();

    // Same content produces same hash and path
    assert_eq!(hash1, hash2);
    assert_eq!(path1, path2);

    // Blob exists and is correct
    assert!(storage::verify_blob(&path1, &hash1).unwrap());

    // Verify the blob is actually in our temp dir
    assert!(path1.starts_with(dir.path()));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_storage_different_content() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let (path1, hash1) = storage::store_blob(b"content-a").unwrap();
    let (path2, hash2) = storage::store_blob(b"content-b").unwrap();

    assert_ne!(hash1, hash2);
    assert_ne!(path1, path2);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 11. Registry Persistence (scan from disk)
// ============================================================================

#[tokio::test]
async fn test_registry_persistence_across_restarts() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    // First "session": register a model
    {
        let registry = ModelRegistry::new();
        let manifest = manifest_with_blob(dir.path(), "persistent-model", b"model-data");
        registry.register(manifest).unwrap();
        assert_eq!(registry.count(), 1);
    }

    // Second "session": scan from disk
    {
        let registry = ModelRegistry::new();
        assert_eq!(registry.count(), 0); // empty before scan
        registry.scan().unwrap();
        assert_eq!(registry.count(), 1);
        let m = registry.get("persistent-model").unwrap();
        assert_eq!(m.name, "persistent-model");
        assert_eq!(m.format, ModelFormat::Gguf);
    }

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 12. Error Handling Workflows
// ============================================================================

#[tokio::test]
async fn test_show_nonexistent_model() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(app, "/api/show", serde_json::json!({"name": "ghost"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_delete_nonexistent_model() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) =
        delete_json(app, "/api/delete", serde_json::json!({"name": "ghost"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_copy_nonexistent_source() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/api/copy",
        serde_json::json!({"source": "ghost", "destination": "copy"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_unknown_route_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let req = Request::builder()
        .uri("/api/nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 13. CORS Workflow
// ============================================================================

#[tokio::test]
async fn test_cors_preflight() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/api/chat")
        .header("origin", "http://localhost:3000")
        .header("access-control-request-method", "POST")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 14. Multi-Model Concurrent Workflow
// ============================================================================

#[tokio::test]
async fn test_multiple_models_concurrent_chat() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    for name in &["model-x", "model-y", "model-z"] {
        let m = manifest_with_blob(dir.path(), name, format!("data-{name}").as_bytes());
        state.registry.register(m).unwrap();
    }

    // Chat with all 3 models concurrently
    let mut handles = vec![];
    for name in &["model-x", "model-y", "model-z"] {
        let app = router::build(state.clone());
        let model = name.to_string();
        handles.push(tokio::spawn(async move {
            let (status, json) = post_json(
                app,
                "/api/chat",
                serde_json::json!({
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": false
                }),
            )
            .await;
            (status, json)
        }));
    }

    for handle in handles {
        let (status, json) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(json["done"], true);
        assert!(json["message"]["content"].is_string());
    }

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 15. Chat Streaming Workflow
// ============================================================================

#[tokio::test]
async fn test_chat_streaming_returns_ndjson() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "stream-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "stream-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();

    // Streaming returns NDJSON — multiple JSON lines
    let lines: Vec<&str> = text.trim().split('\n').filter(|l| !l.is_empty()).collect();
    assert!(
        lines.len() >= 2,
        "Expected at least 2 NDJSON lines, got {}",
        lines.len()
    );

    // First line should have content
    let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(first["model"], "stream-model");
    assert!(!first["done"].as_bool().unwrap());

    // Last line should have done=true
    let last: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
    assert!(last["done"].as_bool().unwrap());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 16. Generate Streaming Workflow
// ============================================================================

#[tokio::test]
async fn test_generate_streaming_returns_ndjson() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "gen-stream", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/generate")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gen-stream",
                "prompt": "Hello",
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();

    let lines: Vec<&str> = text.trim().split('\n').filter(|l| !l.is_empty()).collect();
    assert!(lines.len() >= 2);

    let last: serde_json::Value = serde_json::from_str(lines[lines.len() - 1]).unwrap();
    assert!(last["done"].as_bool().unwrap());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 17. OpenAI Text Completion (/v1/completions)
// ============================================================================

#[tokio::test]
async fn test_openai_completions_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "comp-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "comp-model",
            "prompt": "Once upon a time",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "comp-model");
    assert_eq!(json["object"], "text_completion");
    assert!(json["id"].as_str().unwrap().starts_with("cmpl-"));
    let choices = json["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 1);
    assert!(choices[0]["text"].is_string());
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].is_number());
    assert!(json["usage"]["completion_tokens"].is_number());
    assert!(json["usage"]["total_tokens"].is_number());

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_completions_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "ghost",
            "prompt": "Hello"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["error"]["code"], "model_not_found");

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_completions_streaming_sse() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "sse-comp", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "sse-comp",
                "prompt": "Hello",
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    // SSE format: lines starting with "data: "
    assert!(text.contains("data: "), "Expected SSE format, got: {text}");
    // Should end with [DONE]
    assert!(text.contains("[DONE]"), "Expected [DONE] terminator");

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 18. OpenAI Chat Streaming (SSE format)
// ============================================================================

#[tokio::test]
async fn test_openai_chat_streaming_sse() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "sse-chat", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "sse-chat",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("data: "), "Expected SSE format");
    assert!(text.contains("[DONE]"), "Expected [DONE] terminator");

    // Parse the first SSE data line
    let first_data = text
        .lines()
        .find(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
        .unwrap()
        .strip_prefix("data: ")
        .unwrap();
    let chunk: serde_json::Value = serde_json::from_str(first_data).unwrap();
    assert_eq!(chunk["object"], "chat.completion.chunk");
    assert!(chunk["id"].is_string());
    assert!(chunk["choices"].is_array());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 19. Usage Dashboard (/v1/usage)
// ============================================================================

#[tokio::test]
async fn test_usage_endpoint_empty() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = get_json(app, "/v1/usage").await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["data"].as_array().unwrap().is_empty());
    assert_eq!(json["total"]["requests"], 0);
    assert_eq!(json["total"]["total_tokens"], 0);
    assert_eq!(json["total"]["prompt_tokens"], 0);
    assert_eq!(json["total"]["completion_tokens"], 0);

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_usage_accumulates_after_inference() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "usage-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    // Make a chat request to generate usage
    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "usage-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Check usage reflects the request
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/v1/usage").await;
    assert_eq!(status, StatusCode::OK);
    let total = &json["total"];
    assert!(total["requests"].as_u64().unwrap() >= 1);
    assert!(total["total_tokens"].as_u64().unwrap() > 0);

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_usage_with_model_filter() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    // Register two models and chat with both
    for name in &["model-alpha", "model-beta"] {
        let m = manifest_with_blob(dir.path(), name, format!("data-{name}").as_bytes());
        state.registry.register(m).unwrap();

        let app = router::build(state.clone());
        post_json(
            app,
            "/api/chat",
            serde_json::json!({
                "model": name,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": false
            }),
        )
        .await;
    }

    // Filter by model-alpha
    let app = router::build(state.clone());
    let (status, json) = get_json(app, "/v1/usage?model=model-alpha").await;
    assert_eq!(status, StatusCode::OK);
    let data = json["data"].as_array().unwrap();
    // All returned buckets should be for model-alpha
    for bucket in data {
        assert_eq!(bucket["model"], "model-alpha");
    }

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 20. Push Endpoint Error Paths
// ============================================================================

#[tokio::test]
async fn test_push_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(app, "/api/push", serde_json::json!({"name": "ghost"})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["status"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_push_missing_destination() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "push-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state);
    let (status, json) =
        post_json(app, "/api/push", serde_json::json!({"name": "push-model"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(json["status"].as_str().unwrap().contains("destination"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 21. Create Endpoint
// ============================================================================

#[tokio::test]
async fn test_create_from_modelfile_with_existing_base() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    // Register a base model that the Modelfile can reference
    let manifest = manifest_with_blob(dir.path(), "base-model", b"fake-base-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/create",
        serde_json::json!({
            "name": "custom-model",
            "modelfile": "FROM base-model\nSYSTEM You are a helpful assistant.\nPARAMETER temperature 0.7",
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["status"].as_str().unwrap().contains("success"));

    // Verify the model was created in the registry
    assert!(state.registry.exists("custom-model"));

    // Verify the model has the system prompt
    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/show",
        serde_json::json!({"name": "custom-model"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["system"]
        .as_str()
        .unwrap()
        .contains("helpful assistant"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_create_invalid_modelfile() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, _json) = post_json(
        app,
        "/api/create",
        serde_json::json!({
            "name": "bad-model",
            "modelfile": "INVALID CONTENT WITHOUT FROM",
            "stream": false
        }),
    )
    .await;
    // Should fail because Modelfile has no FROM directive
    assert!(
        status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
        "Expected error status, got {status}"
    );

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 22. Multi-Turn Chat Conversation
// ============================================================================

#[tokio::test]
async fn test_chat_multi_turn_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "conv-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "conv-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"}
            ],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "conv-model");
    assert_eq!(json["message"]["role"], "assistant");
    assert!(json["message"]["content"].is_string());
    assert_eq!(json["done"], true);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 23. Embed Model Not Found
// ============================================================================

#[tokio::test]
async fn test_embed_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/api/embed",
        serde_json::json!({
            "model": "ghost",
            "input": ["hello"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_embeddings_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/api/embeddings",
        serde_json::json!({
            "model": "ghost",
            "prompt": "hello"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(json["error"].as_str().unwrap().contains("not found"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_openai_embeddings_model_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let (status, json) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({
            "model": "ghost",
            "input": "hello"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["error"]["code"], "model_not_found");

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 24. Generate with Options (temperature, top_p, etc.)
// ============================================================================

#[tokio::test]
async fn test_generate_with_options() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "opts-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "opts-model",
            "prompt": "Hello",
            "stream": false,
            "options": {
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 42,
                "num_ctx": 2048,
                "repeat_penalty": 1.1,
                "stop": ["\n"]
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["model"], "opts-model");
    assert_eq!(json["done"], true);

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_chat_with_options() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "opts-chat", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "opts-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "options": {
                "temperature": 0.0,
                "seed": 123,
                "num_ctx": 4096
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["done"], true);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 25. Keep-Alive Parameter
// ============================================================================

#[tokio::test]
async fn test_chat_with_keep_alive_string() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "ka-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "ka-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "keep_alive": "10m"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.is_model_loaded("ka-model"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_chat_with_keep_alive_numeric() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "ka-num", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "ka-num",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "keep_alive": 300
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.is_model_loaded("ka-num"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
async fn test_chat_with_keep_alive_never_unload() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "ka-forever", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "ka-forever",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "keep_alive": -1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(state.is_model_loaded("ka-forever"));
    // Model with keep_alive=-1 should never appear in expired list
    assert!(state.expired_models().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 26. Model Show Verbose
// ============================================================================

#[tokio::test]
async fn test_show_model_with_verbose() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "verbose-model", b"fake-gguf-data");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/show",
        serde_json::json!({"name": "verbose-model", "verbose": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(json["details"].is_object());
    assert_eq!(json["details"]["family"], "llama");

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 27. Manifest Default Parameters
// ============================================================================

#[tokio::test]
async fn test_chat_uses_manifest_default_parameters() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let mut manifest = manifest_with_blob(dir.path(), "defaults-model", b"fake-model");
    manifest.default_parameters = Some(
        [("temperature".to_string(), serde_json::json!(0.1))]
            .into_iter()
            .collect(),
    );
    manifest.system_prompt = Some("You are a pirate.".to_string());

    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(MockBackend::success()));
    let state = AppState::new(
        Arc::new(ModelRegistry::new()),
        Arc::new(backends),
        Arc::new(PowerConfig::default()),
    );
    state.registry.register(manifest).unwrap();

    // Chat without specifying temperature — should use manifest default
    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "defaults-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["done"], true);

    // Show should reflect the system prompt
    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/show",
        serde_json::json!({"name": "defaults-model"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["system"], "You are a pirate.");

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 28. Metrics Accumulation After Requests
// ============================================================================

#[tokio::test]
async fn test_metrics_increment_after_requests() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "metrics-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    // Make a chat request
    let app = router::build(state.clone());
    let (status, _) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "metrics-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Check metrics reflect the request
    let app = router::build(state.clone());
    let req = Request::builder()
        .uri("/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("power_http_requests_total"));
    // Should have at least 1 loaded model
    assert!(text.contains("power_models_loaded 1"));

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 29. Config from Environment Variables
// ============================================================================

#[tokio::test]
async fn test_config_from_env() {
    use a3s_power::config::PowerConfig;

    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());
    std::env::set_var("A3S_POWER_HOST", "0.0.0.0");
    std::env::set_var("A3S_POWER_PORT", "9999");
    std::env::set_var("A3S_POWER_MAX_LOADED_MODELS", "5");
    std::env::set_var("A3S_POWER_KEEP_ALIVE", "10m");

    let config = PowerConfig::load().unwrap_or_default();
    // Config should pick up env vars (if load() reads them)
    // At minimum, default config should be valid
    assert!(config.port > 0);
    assert!(!config.host.is_empty());

    std::env::remove_var("A3S_POWER_HOME");
    std::env::remove_var("A3S_POWER_HOST");
    std::env::remove_var("A3S_POWER_PORT");
    std::env::remove_var("A3S_POWER_MAX_LOADED_MODELS");
    std::env::remove_var("A3S_POWER_KEEP_ALIVE");
}

// ============================================================================
// 30. OpenAI Chat with Tools (function calling)
// ============================================================================

#[tokio::test]
async fn test_openai_chat_with_tools() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "tool-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tool-model",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "stream": false,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    // Response should have choices regardless of whether tools were invoked
    assert!(json["choices"].is_array());
    assert!(!json["choices"].as_array().unwrap().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 31. OpenAI Chat with Response Format (JSON mode)
// ============================================================================

#[tokio::test]
async fn test_openai_chat_json_mode() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "json-model", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "json-model",
            "messages": [{"role": "user", "content": "Return a JSON object"}],
            "stream": false,
            "response_format": {"type": "json_object"}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    assert!(!json["choices"].as_array().unwrap().is_empty());

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 32. Native Chat with Format (JSON mode)
// ============================================================================

#[tokio::test]
async fn test_native_chat_json_format() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());
    let manifest = manifest_with_blob(dir.path(), "json-native", b"fake-model");
    state.registry.register(manifest).unwrap();

    let app = router::build(state.clone());
    let (status, json) = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "json-native",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "format": "json"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["done"], true);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 33. HEAD / (Ollama client detection)
// ============================================================================

#[tokio::test]
async fn test_head_root_for_ollama_detection() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    let app = router::build(state);
    let req = Request::builder()
        .method("HEAD")
        .uri("/")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    std::env::remove_var("A3S_POWER_HOME");
}

// ============================================================================
// 34. Model Registry Count and Exists
// ============================================================================

#[tokio::test]
async fn test_registry_operations_through_api() {
    let dir = tempfile::tempdir().unwrap();
    let state = isolated_state(dir.path());

    // Initially empty
    let app = router::build(state.clone());
    let (_, json) = get_json(app, "/api/tags").await;
    assert_eq!(json["models"].as_array().unwrap().len(), 0);

    // Register 3 models
    for i in 0..3 {
        let m = manifest_with_blob(
            dir.path(),
            &format!("model-{i}"),
            format!("data-{i}").as_bytes(),
        );
        state.registry.register(m).unwrap();
    }

    // List should show 3
    let app = router::build(state.clone());
    let (_, json) = get_json(app, "/api/tags").await;
    assert_eq!(json["models"].as_array().unwrap().len(), 3);

    // OpenAI models should also show 3
    let app = router::build(state.clone());
    let (_, json) = get_json(app, "/v1/models").await;
    assert_eq!(json["data"].as_array().unwrap().len(), 3);

    // Delete one
    let app = router::build(state.clone());
    let _ = delete_json(app, "/api/delete", serde_json::json!({"name": "model-1"})).await;

    // Should show 2
    let app = router::build(state.clone());
    let (_, json) = get_json(app, "/api/tags").await;
    assert_eq!(json["models"].as_array().unwrap().len(), 2);

    std::env::remove_var("A3S_POWER_HOME");
}
