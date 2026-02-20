pub mod attestation;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;

use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::server::state::AppState;

/// Build the OpenAI-compatible API routes.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/chat/completions", post(chat::handler))
        .route("/completions", post(completions::handler))
        .route(
            "/models",
            get(models::list_handler).post(models::register_handler),
        )
        .route("/models/pull", post(models::pull_handler))
        .route(
            "/models/:name",
            get(models::get_handler).delete(models::delete_handler),
        )
        .route("/embeddings", post(embeddings::handler))
        .route("/attestation", get(attestation::handler))
}

/// Return a standard OpenAI-compatible error JSON response with the appropriate HTTP status.
///
/// Error code → HTTP status mapping:
/// - `model_not_found`      → 404
/// - `backend_unavailable`  → 503
/// - `model_load_failed`    → 503
/// - `inference_failed`     → 500
/// - `server_error`         → 500
/// - anything else          → 400
pub fn openai_error(code: &str, message: &str) -> (StatusCode, Json<serde_json::Value>) {
    let status = match code {
        "model_not_found" => StatusCode::NOT_FOUND,
        "backend_unavailable" | "model_load_failed" => StatusCode::SERVICE_UNAVAILABLE,
        "inference_failed" | "server_error" => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": code
            }
        })),
    )
}

/// Round a token count to the nearest 10 for side-channel mitigation.
pub(super) fn round_tokens(n: u32) -> u32 {
    ((n + 5) / 10) * 10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_error_structure() {
        let (status, Json(value)) = openai_error("model_not_found", "model 'x' not found");
        assert_eq!(status, StatusCode::NOT_FOUND);
        let error = value.get("error").unwrap();
        assert_eq!(error["message"], "model 'x' not found");
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(error["code"], "model_not_found");
    }

    #[test]
    fn test_openai_error_serializable() {
        let (status, Json(value)) = openai_error("server_error", "something broke");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains("something broke"));
        assert!(json.contains("server_error"));
    }

    #[test]
    fn test_openai_error_status_codes() {
        assert_eq!(openai_error("model_not_found", "").0, StatusCode::NOT_FOUND);
        assert_eq!(
            openai_error("backend_unavailable", "").0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            openai_error("model_load_failed", "").0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            openai_error("inference_failed", "").0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            openai_error("server_error", "").0,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(openai_error("unknown_code", "").0, StatusCode::BAD_REQUEST);
    }
}
