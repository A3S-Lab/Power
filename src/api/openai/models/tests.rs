use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
use crate::server::router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serial_test::serial;
use tower::util::ServiceExt;

mod registration;

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
    assert!(json["data"]
        .as_array()
        .unwrap()
        .iter()
        .all(|model| model["context_length"] == 4096));
    assert!(json["data"]
        .as_array()
        .unwrap()
        .iter()
        .all(|model| model["format"] == "gguf" && model["size_bytes"] == 1_000_000));
    assert!(json["data"]
        .as_array()
        .unwrap()
        .iter()
        .all(|model| model["sha256"]
            .as_str()
            .is_some_and(|value| value.starts_with("sha256-"))));

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
    assert_eq!(json["context_length"], 4096);
    assert_eq!(json["format"], "gguf");
    assert_eq!(json["size_bytes"], 1_000_000);
    assert_eq!(json["sha256"], "sha256-llama3");

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
async fn test_pull_model_rejects_unsupported_top_level_fields() {
    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state);
    let body = serde_json::json!({
        "name": "new-model",
        "revision": "main"
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/models/pull")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "unsupported_request_fields");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("revision"));
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
