use axum::response::IntoResponse;
use axum::Json;

/// GET /api/version - Return server version (Ollama-compatible).
pub async fn handler() -> impl IntoResponse {
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn test_version_returns_ok() {
        let resp = handler().await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_version_body_has_version_string() {
        let resp = handler().await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["version"].is_string());
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }
}
