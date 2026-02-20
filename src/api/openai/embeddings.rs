use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::types::{EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};
use crate::server::audit::AuditEvent;
use crate::server::auth::AuthId;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// POST /v1/embeddings - OpenAI-compatible embedding generation.
pub async fn handler(
    State(state): State<AppState>,
    auth_id: Option<axum::Extension<AuthId>>,
    Json(request): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();

    // Build request context for isolation and audit tracking
    let ctx = RequestContext::new(auth_id.map(|a| a.0 .0.clone()));
    state.metrics.increment_active_requests();

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            return Json(serde_json::json!({
                "error": {
                    "message": format!("model '{}' not found", model_name),
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }))
            .into_response();
        }
    };

    let backend = match state.backends.find_for_format(&manifest.format) {
        Ok(b) => b,
        Err(e) => {
            return Json(serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "server_error",
                    "code": "backend_unavailable"
                }
            }))
            .into_response();
        }
    };

    if let Err(e) =
        crate::api::autoload::ensure_loaded(&state, &model_name, &manifest, &backend).await
    {
        return Json(serde_json::json!({
            "error": {
                "message": e.to_string(),
                "type": "server_error",
                "code": "model_load_failed"
            }
        }))
        .into_response();
    }

    let input_texts = request.input.into_vec();
    let backend_request = crate::backend::types::EmbeddingRequest {
        input: input_texts.clone(),
    };

    match backend.embed(&model_name, backend_request).await {
        Ok(result) => {
            let data: Vec<EmbeddingData> = result
                .embeddings
                .into_iter()
                .enumerate()
                .map(|(i, emb)| EmbeddingData {
                    object: "embedding".to_string(),
                    embedding: emb,
                    index: i as u32,
                })
                .collect();

            // Request isolation: clean up backend resources
            backend.cleanup_request(&model_name, &ctx).await.ok();
            state.metrics.decrement_active_requests();

            // Audit: log successful embedding
            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::success(
                    &ctx.request_id,
                    ctx.auth_id.clone(),
                    "embedding",
                    Some(model_name.clone()),
                    Some(ctx.elapsed().as_millis() as u64),
                    None,
                ));
            }

            Json(EmbeddingResponse {
                object: "list".to_string(),
                data,
                model: model_name,
                usage: EmbeddingUsage {
                    prompt_tokens: input_texts.len() as u32,
                    total_tokens: input_texts.len() as u32,
                },
            })
            .into_response()
        }
        Err(e) => {
            state.metrics.decrement_active_requests();
            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::failure(
                    &ctx.request_id,
                    ctx.auth_id.clone(),
                    "embedding",
                    Some(model_name.clone()),
                    e.to_string(),
                ));
            }
            Json(serde_json::json!({
                "error": {
                    "message": e.to_string(),
                    "type": "server_error",
                    "code": "inference_failed"
                }
            }))
            .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::model::manifest::ModelFormat;
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_openai_embeddings_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"nonexistent","input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_embeddings_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("st-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"st-model","input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_embeddings_success_single() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["data"][0]["object"], "embedding");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_embeddings_success_multiple() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","input":["hello","world"]}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"].as_array().unwrap().len(), 2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_embeddings_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","input":"hello"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mock load failure"));

        std::env::remove_var("A3S_POWER_HOME");
    }
}
