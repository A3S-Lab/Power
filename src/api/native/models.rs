use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::types::{
    DeleteRequest, NativeModelDetails, NativeModelInfo, ShowRequest, ShowResponse,
};
use crate::model::manifest::ModelManifest;
use crate::model::storage;
use crate::server::state::AppState;

/// Build a `model_info` JSON object from manifest metadata.
///
/// Mirrors Ollama's `model_info` field with architecture details.
fn build_model_info(manifest: &ModelManifest) -> Option<serde_json::Value> {
    let mut info = serde_json::Map::new();
    if let Some(ref family) = manifest.family {
        info.insert(
            "general.architecture".to_string(),
            serde_json::Value::String(family.clone()),
        );
    }
    if let Some(ref params) = manifest.parameters {
        if let Some(count) = params.parameter_count {
            info.insert(
                "general.parameter_count".to_string(),
                serde_json::Value::Number(count.into()),
            );
        }
        if let Some(ref quant) = params.quantization {
            info.insert(
                "general.file_type".to_string(),
                serde_json::Value::String(quant.clone()),
            );
        }
        if let Some(ctx) = params.context_length {
            info.insert(
                "general.context_length".to_string(),
                serde_json::Value::Number(ctx.into()),
            );
        }
    }
    if info.is_empty() {
        None
    } else {
        Some(serde_json::Value::Object(info))
    }
}

/// GET /api/tags - List local models (Ollama-compatible).
pub async fn list_handler(State(state): State<AppState>) -> impl IntoResponse {
    match state.registry.list() {
        Ok(models) => {
            let model_infos: Vec<NativeModelInfo> = models
                .iter()
                .map(|m| NativeModelInfo {
                    name: m.name.clone(),
                    modified_at: m.created_at.to_rfc3339(),
                    size: m.size,
                    digest: format!("sha256:{}", &m.sha256),
                    details: NativeModelDetails {
                        format: m.format.to_string(),
                        parameter_size: m
                            .parameters
                            .as_ref()
                            .and_then(|p| p.parameter_count)
                            .map(|c| format!("{c}")),
                        quantization_level: m
                            .parameters
                            .as_ref()
                            .and_then(|p| p.quantization.clone()),
                        family: m.family.clone(),
                        families: m.families.clone(),
                    },
                })
                .collect();
            Json(serde_json::json!({ "models": model_infos })).into_response()
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

/// POST /api/show - Show model details (Ollama-compatible).
pub async fn show_handler(
    State(state): State<AppState>,
    Json(request): Json<ShowRequest>,
) -> impl IntoResponse {
    match state.registry.get(&request.name) {
        Ok(manifest) => {
            let response = ShowResponse {
                modelfile: manifest.modelfile_content.clone().unwrap_or_default(),
                parameters: serde_json::to_string_pretty(&manifest.parameters).unwrap_or_default(),
                template: manifest.template_override.clone().unwrap_or_default(),
                details: NativeModelDetails {
                    format: manifest.format.to_string(),
                    parameter_size: manifest
                        .parameters
                        .as_ref()
                        .and_then(|p| p.parameter_count)
                        .map(|c| format!("{c}")),
                    quantization_level: manifest
                        .parameters
                        .as_ref()
                        .and_then(|p| p.quantization.clone()),
                    family: manifest.family.clone(),
                    families: manifest.families.clone(),
                },
                system: manifest.system_prompt.clone(),
                license: manifest.license.clone(),
                model_info: build_model_info(&manifest),
                modified_at: manifest.created_at.to_rfc3339(),
            };
            Json(response).into_response()
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

/// DELETE /api/delete - Delete a model (Ollama-compatible).
pub async fn delete_handler(
    State(state): State<AppState>,
    Json(request): Json<DeleteRequest>,
) -> impl IntoResponse {
    match state.registry.remove(&request.name) {
        Ok(manifest) => {
            if let Err(e) = storage::delete_blob(&manifest) {
                tracing::warn!(model = %manifest.name, "Failed to delete blob: {e}");
            }
            Json(serde_json::json!({ "status": "success" })).into_response()
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    #[serial]
    async fn test_list_handler_with_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("model-a")).unwrap();
        state.registry.register(sample_manifest("model-b")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .uri("/api/tags")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_show_handler_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("test-model"))
            .unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"test-model"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["details"]["format"], "GGUF");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_show_handler_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/show")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"nonexistent"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_delete_handler_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("to-delete"))
            .unwrap();

        let app = router::build(state.clone());
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/delete")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"to-delete"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "success");
        assert!(!state.registry.exists("to-delete"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_delete_handler_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("DELETE")
            .uri("/api/delete")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"nonexistent"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }
}
