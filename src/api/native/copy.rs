use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;

use crate::api::types::CopyRequest;
use crate::server::state::AppState;

/// POST /api/copy - Copy a model under a new name (Ollama-compatible).
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<CopyRequest>,
) -> impl IntoResponse {
    let source_manifest = match state.registry.get(&request.source) {
        Ok(m) => m,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("model '{}' not found", request.source)
                })),
            )
                .into_response();
        }
    };

    let mut new_manifest = source_manifest.clone();
    new_manifest.name = request.destination.clone();
    new_manifest.created_at = chrono::Utc::now();

    match state.registry.register(new_manifest) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use serial_test::serial;

    #[tokio::test]
    #[serial]
    async fn test_copy_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("original"))
            .unwrap();

        let request = CopyRequest {
            source: "original".to_string(),
            destination: "copy".to_string(),
        };
        let resp = handler(State(state.clone()), Json(request))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(state.registry.exists("copy"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_copy_source_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let request = CopyRequest {
            source: "nonexistent".to_string(),
            destination: "copy".to_string(),
        };
        let resp = handler(State(state), Json(request)).await.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_copy_request_serialization() {
        let json = r#"{"source": "a", "destination": "b"}"#;
        let req: CopyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.source, "a");
        assert_eq!(req.destination, "b");
    }
}
