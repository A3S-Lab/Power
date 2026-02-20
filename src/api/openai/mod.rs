pub mod attestation;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;

use axum::routing::{get, post};
use axum::{Json, Router};

use crate::server::state::AppState;

/// Build the OpenAI-compatible API routes.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/chat/completions", post(chat::handler))
        .route("/completions", post(completions::handler))
        .route("/models", get(models::handler))
        .route("/embeddings", post(embeddings::handler))
        .route("/attestation", get(attestation::handler))
}

/// Return a standard OpenAI-compatible error JSON response.
pub fn openai_error(code: &str, message: &str) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_error_structure() {
        let Json(value) = openai_error("model_not_found", "model 'x' not found");
        let error = value.get("error").unwrap();
        assert_eq!(error["message"], "model 'x' not found");
        assert_eq!(error["type"], "invalid_request_error");
        assert_eq!(error["code"], "model_not_found");
    }

    #[test]
    fn test_openai_error_serializable() {
        let Json(value) = openai_error("server_error", "something broke");
        let json = serde_json::to_string(&value).unwrap();
        assert!(json.contains("something broke"));
        assert!(json.contains("server_error"));
    }
}
