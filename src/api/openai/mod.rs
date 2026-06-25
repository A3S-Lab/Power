pub mod attestation;
pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod models;

use axum::http::StatusCode;
use axum::response::sse::Event;
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
            "/models/pull/:name/status",
            get(models::pull_status_handler),
        )
        .route(
            "/models/:name",
            get(models::get_handler).delete(models::delete_handler),
        )
        .route("/embeddings", post(embeddings::handler))
        .route("/attestation", get(attestation::handler))
        .route("/logs", get(logs_handler))
}

pub(super) fn sse_json_data<T: serde::Serialize>(value: &T) -> String {
    match serde_json::to_string(value) {
        Ok(data) => data,
        Err(e) => serde_json::json!({
            "error": {
                "message": format!("failed to serialize SSE event: {e}"),
                "type": "server_error",
                "code": "sse_serialization_failed"
            }
        })
        .to_string(),
    }
}

pub(super) fn sse_json_event<T: serde::Serialize>(value: &T) -> Event {
    Event::default().data(sse_json_data(value))
}

/// GET /v1/logs — Stream server log entries as SSE.
///
/// Sends the last `N` buffered entries immediately, then streams new entries in
/// real-time as they are emitted by the tracing subscriber.
///
/// Each event data is a JSON object:
/// ```json
/// { "ts": "2024-01-01T00:00:00.000Z", "level": "INFO",
///   "target": "a3s_power::model::pull", "message": "Pulling model ..." }
/// ```
async fn logs_handler(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> impl axum::response::IntoResponse {
    use futures::StreamExt;
    use tokio_stream::wrappers::BroadcastStream;

    let recent = state.log_buffer.recent();
    let rx = state.log_buffer.subscribe();

    let recent_stream = futures::stream::iter(recent)
        .map(|entry| Ok::<_, std::convert::Infallible>(sse_json_event(&entry)));

    let live_stream = BroadcastStream::new(rx)
        .filter_map(|result| async move { log_entry_from_broadcast_result(result) })
        .map(|entry| Ok::<_, std::convert::Infallible>(sse_json_event(&entry)));

    axum::response::Sse::new(recent_stream.chain(live_stream))
}

fn log_entry_from_broadcast_result(
    result: Result<
        crate::server::log_stream::LogEntry,
        tokio_stream::wrappers::errors::BroadcastStreamRecvError,
    >,
) -> Option<crate::server::log_stream::LogEntry> {
    match result {
        Ok(entry) => Some(entry),
        Err(e) => {
            tracing::debug!(error = %e, "Log stream receiver lagged; dropped live log entries");
            None
        }
    }
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

    #[test]
    fn test_sse_json_data_serializes_values() {
        let data = sse_json_data(&serde_json::json!({"status": "ok"}));
        assert_eq!(data, r#"{"status":"ok"}"#);
    }

    #[test]
    fn test_sse_json_data_reports_serialization_failure() {
        struct FailingSerialize;

        impl serde::Serialize for FailingSerialize {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("boom"))
            }
        }

        let data = sse_json_data(&FailingSerialize);
        assert!(data.contains("sse_serialization_failed"));
        assert!(data.contains("boom"));
    }

    #[test]
    fn test_log_entry_from_broadcast_result_passes_entries() {
        let entry = crate::server::log_stream::LogEntry {
            ts: "2026-01-01T00:00:00.000Z".to_string(),
            level: "INFO".to_string(),
            target: "a3s_power::test".to_string(),
            message: "hello".to_string(),
        };

        let result = log_entry_from_broadcast_result(Ok(entry));

        assert_eq!(result.unwrap().message, "hello");
    }

    #[test]
    fn test_log_entry_from_broadcast_result_drops_lag_errors() {
        let result = log_entry_from_broadcast_result(Err(
            tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(3),
        ));

        assert!(result.is_none());
    }
}
