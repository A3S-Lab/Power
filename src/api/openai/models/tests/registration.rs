use crate::backend::test_utils::{test_state_with_mock, MockBackend};
use crate::server::router;
use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serial_test::serial;
use tower::util::ServiceExt;

fn push_gguf_string(buffer: &mut Vec<u8>, value: &str) {
    buffer.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buffer.extend_from_slice(value.as_bytes());
}

fn write_valid_dspark_gguf(path: &std::path::Path) {
    let tensor_names = ["markov_w1.weight", "markov_w2.weight", "conf_proj.weight"];
    let mut buffer = Vec::new();
    buffer.extend_from_slice(b"GGUF");
    buffer.extend_from_slice(&3_u32.to_le_bytes());
    buffer.extend_from_slice(&(tensor_names.len() as u64).to_le_bytes());
    buffer.extend_from_slice(&3_u64.to_le_bytes());

    push_gguf_string(&mut buffer, "general.architecture");
    buffer.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_string(&mut buffer, "dflash");

    push_gguf_string(&mut buffer, "dflash.block_size");
    buffer.extend_from_slice(&4_u32.to_le_bytes());
    buffer.extend_from_slice(&16_u32.to_le_bytes());

    push_gguf_string(&mut buffer, "dflash.target_layers");
    buffer.extend_from_slice(&9_u32.to_le_bytes());
    buffer.extend_from_slice(&4_u32.to_le_bytes());
    buffer.extend_from_slice(&2_u64.to_le_bytes());
    buffer.extend_from_slice(&8_u32.to_le_bytes());
    buffer.extend_from_slice(&16_u32.to_le_bytes());

    for name in tensor_names {
        push_gguf_string(&mut buffer, name);
        buffer.extend_from_slice(&1_u32.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        buffer.extend_from_slice(&0_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
    }

    std::fs::write(path, buffer).unwrap();
}

fn write_minimal_target_gguf(path: &std::path::Path) {
    let mut buffer = Vec::new();
    buffer.extend_from_slice(b"GGUF");
    buffer.extend_from_slice(&3_u32.to_le_bytes());
    buffer.extend_from_slice(&0_u64.to_le_bytes());
    buffer.extend_from_slice(&1_u64.to_le_bytes());
    push_gguf_string(&mut buffer, "general.architecture");
    buffer.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_string(&mut buffer, "test-target");
    std::fs::write(path, buffer).unwrap();
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
async fn test_register_model_rejects_unsupported_top_level_fields() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("local.gguf");
    std::fs::write(&model_file, b"fake weights").unwrap();

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "policy-model",
        "path": model_file.to_str().unwrap(),
        "sha256": "client-supplied-pin"
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
    assert_eq!(json["error"]["code"], "unsupported_request_fields");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("sha256"));
    assert!(!state.registry.exists("policy-model"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
#[serial]
async fn test_register_model_rejects_unsupported_format() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("local.onnx");
    std::fs::write(&model_file, b"fake weights").unwrap();

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "bad-format",
        "path": model_file.to_str().unwrap(),
        "format": "onnx"
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
    assert_eq!(json["error"]["code"], "unsupported_model_format");
    assert!(json["error"]["message"].as_str().unwrap().contains("onnx"));
    assert!(!state.registry.exists("bad-format"));

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
async fn test_register_model_captures_external_dspark_identity() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("target.gguf");
    let draft_file = dir.path().join("draft.gguf");
    write_minimal_target_gguf(&model_file);
    write_valid_dspark_gguf(&draft_file);

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "dspark-target",
        "path": model_file.to_str().unwrap(),
        "external_draft": {
            "kind": "dspark",
            "path": draft_file.to_str().unwrap(),
            "source": "https://example.invalid/dspark",
            "revision": "0123456789abcdef",
            "license": "Apache-2.0"
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    let status = response.status();
    let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        status,
        StatusCode::CREATED,
        "{}",
        String::from_utf8_lossy(&response_body)
    );

    let manifest = state.registry.get("dspark-target").unwrap();
    let draft = manifest.external_draft.unwrap();
    assert_eq!(draft.kind.as_str(), "dspark");
    assert_eq!(draft.path, draft_file);
    assert_eq!(draft.size, std::fs::metadata(&draft.path).unwrap().len());
    assert_eq!(draft.sha256.len(), 64);
    assert_eq!(draft.target_sha256, manifest.sha256);
    assert_eq!(
        draft.source.as_deref(),
        Some("https://example.invalid/dspark")
    );
    assert_eq!(draft.revision.as_deref(), Some("0123456789abcdef"));
    assert_eq!(draft.license.as_deref(), Some("Apache-2.0"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
#[serial]
async fn test_register_model_rejects_client_supplied_draft_identity() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("target.gguf");
    let draft_file = dir.path().join("draft.gguf");
    std::fs::write(&model_file, b"target weights").unwrap();
    write_valid_dspark_gguf(&draft_file);

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "forged-draft-identity",
        "path": model_file.to_str().unwrap(),
        "external_draft": {
            "kind": "dspark",
            "path": draft_file.to_str().unwrap(),
            "size": 1,
            "sha256": "client-supplied-digest",
            "target_sha256": "client-supplied-target"
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "unsupported_request_fields");
    let message = json["error"]["message"].as_str().unwrap();
    assert!(message.contains("sha256"));
    assert!(message.contains("size"));
    assert!(message.contains("target_sha256"));
    assert!(!state.registry.exists("forged-draft-identity"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
#[serial]
async fn test_register_model_rejects_external_draft_for_non_gguf_target() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("target.safetensors");
    let draft_file = dir.path().join("draft.gguf");
    std::fs::write(&model_file, b"target weights").unwrap();
    write_valid_dspark_gguf(&draft_file);

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "non-gguf-target",
        "path": model_file.to_str().unwrap(),
        "format": "safetensors",
        "external_draft": {
            "kind": "dspark",
            "path": draft_file.to_str().unwrap()
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "external_draft_requires_gguf");
    assert!(!state.registry.exists("non-gguf-target"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
#[serial]
async fn test_register_model_rejects_external_draft_for_invalid_gguf_target() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("invalid-target.gguf");
    let draft_file = dir.path().join("draft.gguf");
    std::fs::write(&model_file, b"not a GGUF target").unwrap();
    write_valid_dspark_gguf(&draft_file);

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "invalid-gguf-target",
        "path": model_file.to_str().unwrap(),
        "external_draft": {
            "kind": "dspark",
            "path": draft_file.to_str().unwrap()
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "invalid_target_gguf");
    assert!(!state.registry.exists("invalid-gguf-target"));

    std::env::remove_var("A3S_POWER_HOME");
}

#[tokio::test]
#[serial]
async fn test_register_model_rejects_invalid_external_draft_contract() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("A3S_POWER_HOME", dir.path());

    let model_file = dir.path().join("target.gguf");
    let draft_file = dir.path().join("invalid-draft.gguf");
    write_minimal_target_gguf(&model_file);
    std::fs::write(&draft_file, b"not a GGUF draft").unwrap();

    let state = test_state_with_mock(MockBackend::success());
    let app = router::build(state.clone());
    let body = serde_json::json!({
        "name": "invalid-draft-contract",
        "path": model_file.to_str().unwrap(),
        "external_draft": {
            "kind": "dspark",
            "path": draft_file.to_str().unwrap()
        }
    });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["code"], "invalid_external_draft");
    assert!(!state.registry.exists("invalid-draft-contract"));

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

    let size = super::super::dir_size(dir.path()).unwrap();

    assert_eq!(size, 9);
}

#[test]
fn test_dir_size_reports_read_dir_errors() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("not-a-dir");
    std::fs::write(&file_path, b"weights").unwrap();

    let err = super::super::dir_size(&file_path).unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::NotADirectory);
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
