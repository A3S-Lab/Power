use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::types::NativeModelDetails;
use crate::server::state::AppState;

/// GET /api/ps - List running (loaded) models (Ollama-compatible).
pub async fn handler(State(state): State<AppState>) -> impl IntoResponse {
    let loaded_names = state.loaded_model_names();

    let models: Vec<serde_json::Value> = loaded_names
        .iter()
        .filter_map(|name| {
            state.registry.get(name).ok().map(|manifest| {
                let expires_at = state
                    .model_expires_at(name)
                    .map(|dt| crate::api::format_ollama_timestamp(&dt));
                serde_json::json!({
                    "name": manifest.name,
                    "model": manifest.name,
                    "size": manifest.size,
                    "size_vram": manifest.size,
                    "digest": format!("sha256:{}", &manifest.sha256),
                    "details": NativeModelDetails {
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
                    "expires_at": expires_at,
                })
            })
        })
        .collect();

    Json(serde_json::json!({ "models": models }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use serial_test::serial;

    #[tokio::test]
    async fn test_ps_empty_state() {
        let state = test_state_with_mock(MockBackend::success());
        let resp = handler(State(state)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_ps_one_loaded_model() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "test");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_ps_loaded_model_not_in_registry() {
        let state = test_state_with_mock(MockBackend::success());
        state.mark_loaded("ghost-model");

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_ps_multiple_loaded_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("model-a")).unwrap();
        state.registry.register(sample_manifest("model-b")).unwrap();
        state.registry.register(sample_manifest("model-c")).unwrap();

        state.mark_loaded("model-a");
        state.mark_loaded("model-c");

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let models = json["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);

        let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"model-a"));
        assert!(names.contains(&"model-c"));
        assert!(!names.contains(&"model-b"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_ps_response_includes_digest() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &json["models"][0];

        assert!(model["digest"].as_str().unwrap().starts_with("sha256:"));
        assert_eq!(model["size"], 1_000_000);
        assert_eq!(model["size_vram"], 1_000_000);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_ps_response_includes_details() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &json["models"][0];

        assert!(model["details"].is_object());
        assert_eq!(model["details"]["format"], "GGUF");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_ps_includes_expires_at_when_keep_alive_set() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded_with_keep_alive("test", std::time::Duration::from_secs(300));

        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let model = &json["models"][0];

        // Should have expires_at field
        assert!(model["expires_at"].is_string());

        std::env::remove_var("A3S_POWER_HOME");
    }
}
