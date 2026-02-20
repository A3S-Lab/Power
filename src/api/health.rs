use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::server::state::AppState;
use crate::tee::attestation::TeeType;

/// TEE status in health response.
#[derive(Debug, Serialize, Deserialize)]
pub struct TeeStatus {
    pub enabled: bool,
    #[serde(rename = "type")]
    pub tee_type: TeeType,
    pub models_verified: bool,
    /// Whether hardware attestation reports can be generated.
    pub attestation_available: bool,
}

/// Response body for GET /health.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub loaded_models: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tee: Option<TeeStatus>,
}

/// GET /health — server health check.
pub async fn handler(State(state): State<AppState>) -> impl IntoResponse {
    let tee = state.tee_provider.as_ref().map(|provider| {
        let tee_type = provider.tee_type();
        let attestation_available = matches!(tee_type, TeeType::SevSnp | TeeType::Tdx);
        TeeStatus {
            enabled: true,
            tee_type,
            models_verified: !state.config.model_hashes.is_empty(),
            attestation_available,
        }
    });

    let resp = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.uptime().as_secs(),
        loaded_models: state.loaded_model_count(),
        tee,
    };
    (StatusCode::OK, Json(resp))
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

    fn test_state() -> AppState {
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
    }

    fn test_state_tee() -> AppState {
        let config = PowerConfig {
            tee_mode: true,
            redact_logs: true,
            ..Default::default()
        };
        let provider = DefaultTeeProvider::with_type(crate::tee::attestation::TeeType::Simulated);
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        )
        .with_tee_provider(Arc::new(provider))
    }

    #[tokio::test]
    async fn test_health_handler_returns_ok() {
        let state = test_state();
        let resp = handler(State(state)).await.into_response();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_response_has_version() {
        let state = test_state();
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn test_health_reflects_loaded_models() {
        let state = test_state();
        state.mark_loaded("model-a");
        state.mark_loaded("model-b");
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(health.loaded_models, 2);
    }

    #[tokio::test]
    async fn test_health_no_tee_by_default() {
        let state = test_state();
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert!(health.tee.is_none());
    }

    #[tokio::test]
    async fn test_health_tee_enabled() {
        let state = test_state_tee();
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        let tee = health.tee.unwrap();
        assert!(tee.enabled);
    }

    #[tokio::test]
    async fn test_health_tee_not_in_json_when_disabled() {
        let state = test_state();
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.get("tee").is_none());
    }

    #[test]
    fn test_health_response_serialization() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 42,
            loaded_models: 3,
            tee: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"uptime_seconds\":42"));
        assert!(json.contains("\"loaded_models\":3"));
        assert!(!json.contains("\"tee\""));

        let deser: HealthResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.status, "ok");
        assert_eq!(deser.loaded_models, 3);
    }

    #[test]
    fn test_health_response_with_tee_serialization() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            version: "0.2.0".to_string(),
            uptime_seconds: 10,
            loaded_models: 1,
            tee: Some(TeeStatus {
                enabled: true,
                tee_type: TeeType::Simulated,
                models_verified: true,
                attestation_available: false,
            }),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"tee\""));
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"type\":\"simulated\""));
        assert!(json.contains("\"models_verified\":true"));
    }

    #[test]
    fn test_tee_status_serialization() {
        let status = TeeStatus {
            enabled: true,
            tee_type: TeeType::SevSnp,
            models_verified: false,
            attestation_available: true,
        };
        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"type\":\"sev-snp\""));
        assert!(json.contains("\"models_verified\":false"));
        assert!(json.contains("\"attestation_available\":true"));
    }

    #[tokio::test]
    async fn test_health_tee_simulated_attestation_not_available() {
        let state = test_state_tee();
        let resp = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        let tee = health.tee.unwrap();
        // Simulated TEE does not provide real attestation
        assert!(!tee.attestation_available);
    }
}
