use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;

use crate::server::state::AppState;

/// Query parameters for `GET /v1/attestation`.
#[derive(Debug, Deserialize)]
pub struct AttestationQuery {
    /// Optional client-supplied hex-encoded nonce to bind into the attestation report.
    pub nonce: Option<String>,
    /// Optional model name to bind its SHA-256 hash into the attestation report.
    /// Ties the attestation to the specific model being served.
    pub model: Option<String>,
}

/// Decode a hex string to bytes, returning an error response on invalid input.
fn decode_hex_nonce(hex: &str) -> Result<Vec<u8>, (StatusCode, axum::Json<serde_json::Value>)> {
    if !hex.len().is_multiple_of(2) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": {
                    "message": "nonce must be an even-length hex string",
                    "type": "invalid_request_error",
                    "code": "invalid_nonce"
                }
            })),
        ));
    }
    (0..hex.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("nonce contains invalid hex character at position {i}"),
                            "type": "invalid_request_error",
                            "code": "invalid_nonce"
                        }
                    })),
                )
            })
        })
        .collect()
}

/// GET /v1/attestation — generate and return a TEE attestation report.
///
/// Optional `?nonce=<hex>` query parameter binds a client nonce into the report
/// to prevent replay attacks.
///
/// Optional `?model=<name>` query parameter binds the model's SHA-256 hash into
/// the report, tying the attestation to the specific model being served.
///
/// Returns 503 if TEE mode is not enabled or no TEE provider is configured.
pub async fn handler(
    State(state): State<AppState>,
    Query(params): Query<AttestationQuery>,
) -> impl IntoResponse {
    let provider = match &state.tee_provider {
        Some(p) => p.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "TEE mode is not enabled",
                        "type": "tee_unavailable",
                        "code": "tee_not_enabled"
                    }
                })),
            )
                .into_response();
        }
    };

    // Decode nonce if provided
    let nonce_bytes = match params.nonce.as_deref() {
        Some(hex) => match decode_hex_nonce(hex) {
            Ok(b) => Some(b),
            Err(resp) => return resp.into_response(),
        },
        None => None,
    };

    // Look up model hash if model name provided
    let model_hash_bytes: Option<Vec<u8>> = if let Some(ref model_name) = params.model {
        match state.registry.get(model_name) {
            Ok(manifest) => {
                // Parse "sha256:<hex>" or raw hex from model_hashes config
                let hash_str = state
                    .config
                    .model_hashes
                    .get(model_name)
                    .cloned()
                    .unwrap_or_else(|| manifest.sha256.clone());

                let hex = hash_str.strip_prefix("sha256:").unwrap_or(&hash_str);
                decode_hex_nonce(hex).ok()
            }
            Err(_) => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!("model '{}' not found", model_name),
                            "type": "invalid_request_error",
                            "code": "model_not_found"
                        }
                    })),
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    let report_result = if model_hash_bytes.is_some() {
        provider
            .attestation_report_with_model(nonce_bytes.as_deref(), model_hash_bytes.as_deref())
            .await
    } else {
        provider.attestation_report(nonce_bytes.as_deref()).await
    };

    match report_result {
        Ok(report) => {
            state.metrics.increment_tee_attestation();
            (StatusCode::OK, Json(serde_json::to_value(&report).unwrap())).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Failed to generate attestation report: {e}"),
                    "type": "tee_error",
                    "code": "attestation_failed"
                }
            })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::model::registry::ModelRegistry;
    use crate::tee::attestation::DefaultTeeProvider;
    use axum::extract::State;
    use std::sync::Arc;

    fn no_nonce() -> Query<AttestationQuery> {
        Query(AttestationQuery {
            nonce: None,
            model: None,
        })
    }

    fn with_nonce(hex: &str) -> Query<AttestationQuery> {
        Query(AttestationQuery {
            nonce: Some(hex.to_string()),
            model: None,
        })
    }

    fn test_state_no_tee() -> AppState {
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
    }

    fn test_state_simulated() -> AppState {
        std::env::set_var("A3S_TEE_SIMULATE", "1");
        let provider = DefaultTeeProvider::detect();
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                tee_mode: true,
                ..Default::default()
            }),
        )
        .with_tee_provider(Arc::new(provider))
    }

    #[tokio::test]
    async fn test_attestation_no_tee_returns_503() {
        let state = test_state_no_tee();
        let resp = handler(State(state), no_nonce()).await.into_response();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "tee_not_enabled");
    }

    #[tokio::test]
    async fn test_attestation_with_simulated_tee() {
        let state = test_state_simulated();
        let resp = handler(State(state), no_nonce()).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["tee_type"], "simulated");
        assert!(json["report_data"].is_string());
        assert!(json["measurement"].is_string());
        assert!(json["timestamp"].is_string());
        // No nonce in response when not supplied
        assert!(json["nonce"].is_null());
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_report_has_correct_fields() {
        let state = test_state_simulated();
        let resp = handler(State(state), no_nonce()).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Simulated report has 64 bytes of 0xAA for report_data
        let report_data = json["report_data"].as_str().unwrap();
        assert_eq!(report_data.len(), 128); // 64 bytes = 128 hex chars
        assert!(report_data.chars().all(|c| c == 'a'));

        // Simulated report has 48 bytes of 0xBB for measurement
        let measurement = json["measurement"].as_str().unwrap();
        assert_eq!(measurement.len(), 96); // 48 bytes = 96 hex chars
        assert!(measurement.chars().all(|c| c == 'b'));

        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_error_json_structure() {
        let state = test_state_no_tee();
        let resp = handler(State(state), no_nonce()).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Verify OpenAI-style error structure
        assert!(json["error"]["message"].is_string());
        assert!(json["error"]["type"].is_string());
        assert!(json["error"]["code"].is_string());
    }

    #[tokio::test]
    async fn test_attestation_with_nonce_binds_to_report() {
        let state = test_state_simulated();
        // nonce = [0x01, 0x02, 0x03] → hex "010203"
        let resp = handler(State(state), with_nonce("010203"))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // nonce reflected in response
        assert_eq!(json["nonce"], "010203");
        // report_data starts with nonce bytes
        let report_data = json["report_data"].as_str().unwrap();
        assert!(report_data.starts_with("010203"));
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[tokio::test]
    async fn test_attestation_invalid_hex_nonce_returns_400() {
        let state = test_state_simulated();
        let resp = handler(State(state), with_nonce("xyz"))
            .await
            .into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "invalid_nonce");
        std::env::remove_var("A3S_TEE_SIMULATE");
    }

    #[test]
    fn test_decode_hex_nonce_valid() {
        let bytes = decode_hex_nonce("deadbeef").unwrap();
        assert_eq!(bytes, vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn test_decode_hex_nonce_odd_length_fails() {
        assert!(decode_hex_nonce("abc").is_err());
    }

    #[test]
    fn test_decode_hex_nonce_invalid_chars_fails() {
        assert!(decode_hex_nonce("zzzz").is_err());
    }

    #[test]
    fn test_decode_hex_nonce_empty_is_valid() {
        let bytes = decode_hex_nonce("").unwrap();
        assert!(bytes.is_empty());
    }
}
