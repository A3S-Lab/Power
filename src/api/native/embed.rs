use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use std::time::Instant;

use crate::api::types::{NativeEmbedRequest, NativeEmbedResponse};
use crate::backend::types::EmbeddingRequest;
use crate::server::state::AppState;

/// POST /api/embed - Batch embedding generation (Ollama-compatible).
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<NativeEmbedRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            return Json(serde_json::json!({
                "error": format!("model '{}' not found", model_name)
            }))
            .into_response();
        }
    };

    let backend = match state.backends.find_for_format(&manifest.format) {
        Ok(b) => b,
        Err(e) => {
            return Json(serde_json::json!({ "error": e.to_string() })).into_response();
        }
    };

    let load_result = match crate::api::autoload::ensure_loaded_with_keep_alive(
        &state,
        &model_name,
        &manifest,
        &backend,
        request.keep_alive.as_deref(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            return Json(serde_json::json!({ "error": e.to_string() })).into_response();
        }
    };
    let load_duration_ns = load_result.load_duration.as_nanos() as u64;

    let input_texts = request.input.into_vec();
    let backend_request = EmbeddingRequest { input: input_texts };

    let start = Instant::now();
    match backend.embed(&model_name, backend_request).await {
        Ok(response) => {
            let total_duration_ns = start.elapsed().as_nanos() as u64 + load_duration_ns;
            Json(NativeEmbedResponse {
                model: model_name,
                embeddings: response.embeddings,
                total_duration: Some(total_duration_ns),
                load_duration: Some(load_duration_ns),
            })
            .into_response()
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::NativeEmbedInput;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::model::manifest::ModelFormat;
    use serial_test::serial;

    #[tokio::test]
    async fn test_embed_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let request = NativeEmbedRequest {
            model: "nonexistent".to_string(),
            input: NativeEmbedInput::Single("hello".to_string()),
            truncate: None,
            keep_alive: None,
        };
        let resp = handler(axum::extract::State(state), Json(request))
            .await
            .into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_embed_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("embed-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let request = NativeEmbedRequest {
            model: "embed-model".to_string(),
            input: NativeEmbedInput::Single("hello".to_string()),
            truncate: None,
            keep_alive: None,
        };
        let resp = handler(axum::extract::State(state), Json(request))
            .await
            .into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_native_embed_input_single() {
        let json = r#"{"model": "test", "input": "hello"}"#;
        let req: NativeEmbedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.into_vec(), vec!["hello"]);
    }

    #[test]
    fn test_native_embed_input_array() {
        let json = r#"{"model": "test", "input": ["hello", "world"]}"#;
        let req: NativeEmbedRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.input.into_vec(), vec!["hello", "world"]);
    }
}
