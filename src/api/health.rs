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

/// Non-secret speculative-decoding configuration used by this server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeStatus {
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_max: Option<u32>,
    #[serde(default = "crate::config::default_spec_mtp_recurrent_snapshots")]
    pub mtp_recurrent_snapshots: u32,
    #[serde(default = "crate::config::default_spec_mtp_recurrent_chain")]
    pub mtp_recurrent_chain: bool,
    #[serde(default = "crate::config::default_spec_mtp_adaptive")]
    pub mtp_adaptive: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_fr_vocab_size: Option<u32>,
    pub draft_min: u32,
    pub draft_p_min: f32,
}

/// Non-secret inference settings that materially affect benchmark results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceStatus {
    pub gpu_layers: i32,
    pub main_gpu: i32,
    pub tensor_split: Vec<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cpu_tensors: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_tensors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_thread: Option<u32>,
    pub flash_attention: bool,
    pub num_parallel: usize,
    pub use_mlock: bool,
    #[serde(default = "default_use_mmap_status")]
    pub use_mmap: bool,
    pub tee_mode: bool,
    pub suppress_token_metrics: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_padding_ms: Option<u64>,
}

/// Public, non-secret prompt-prefix cache contract and backend availability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptCacheStatus {
    pub enabled: bool,
    pub request_field: String,
    pub max_entries_per_model: usize,
    pub ttl_seconds: u64,
    pub authenticated_namespace: bool,
    pub supported_backends: Vec<String>,
}

impl Default for PromptCacheStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            request_field: "prompt_cache_key".to_string(),
            max_entries_per_model: crate::config::DEFAULT_PROMPT_CACHE_MAX_ENTRIES,
            ttl_seconds: crate::config::DEFAULT_PROMPT_CACHE_TTL_SECONDS,
            authenticated_namespace: true,
            supported_backends: Vec::new(),
        }
    }
}

const fn default_use_mmap_status() -> bool {
    true
}

/// Response body for GET /health.
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub loaded_models: usize,
    pub speculative: SpeculativeStatus,
    pub inference: InferenceStatus,
    #[serde(default)]
    pub prompt_cache: PromptCacheStatus,
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

    let prompt_cache_backends = state
        .backends
        .prompt_cache_backend_names()
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let resp = HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: state.uptime().as_secs(),
        loaded_models: state.loaded_model_count(),
        speculative: SpeculativeStatus {
            mode: state.config.spec_mode.clone(),
            draft_max: state.config.spec_draft_max,
            mtp_recurrent_snapshots: state.config.spec_mtp_recurrent_snapshots,
            mtp_recurrent_chain: state.config.spec_mtp_recurrent_chain,
            mtp_adaptive: state.config.spec_mtp_adaptive,
            mtp_fr_vocab_size: state.config.spec_mtp_fr_vocab_size,
            draft_min: state.config.spec_draft_min,
            draft_p_min: state.config.spec_draft_p_min,
        },
        inference: InferenceStatus {
            gpu_layers: state.config.gpu.gpu_layers,
            main_gpu: state.config.gpu.main_gpu,
            tensor_split: state.config.gpu.tensor_split.clone(),
            cpu_tensors: state.config.gpu.cpu_tensors.clone(),
            gpu_tensors: state.config.gpu.gpu_tensors.clone(),
            num_thread: state.config.num_thread,
            flash_attention: state.config.flash_attention,
            num_parallel: state.config.num_parallel,
            use_mlock: state.config.use_mlock,
            use_mmap: state.config.use_mmap,
            tee_mode: state.config.tee_mode,
            suppress_token_metrics: state.suppress_token_metrics(),
            timing_padding_ms: state.config.timing_padding_ms,
        },
        prompt_cache: PromptCacheStatus {
            enabled: !prompt_cache_backends.is_empty(),
            request_field: "prompt_cache_key".to_string(),
            max_entries_per_model: state.config.prompt_cache_max_entries,
            ttl_seconds: state.config.prompt_cache_ttl_seconds,
            authenticated_namespace: true,
            supported_backends: prompt_cache_backends,
        },
        tee,
    };
    (StatusCode::OK, Json(resp))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::MockBackend;
    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::model::registry::ModelRegistry;
    use crate::tee::attestation::DefaultTeeProvider;
    use crate::tee::privacy::DefaultPrivacyProvider;
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

    fn inference_status() -> InferenceStatus {
        InferenceStatus {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: Vec::new(),
            cpu_tensors: Vec::new(),
            gpu_tensors: Vec::new(),
            num_thread: Some(16),
            flash_attention: true,
            num_parallel: 1,
            use_mlock: false,
            use_mmap: true,
            tee_mode: false,
            suppress_token_metrics: false,
            timing_padding_ms: None,
        }
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
        assert_eq!(health.speculative.mode, "auto");
        assert_eq!(health.speculative.mtp_recurrent_snapshots, 7);
        assert!(health.speculative.mtp_recurrent_chain);
        assert!(!health.inference.suppress_token_metrics);
        assert!(!health.prompt_cache.enabled);
        assert_eq!(health.prompt_cache.request_field, "prompt_cache_key");
        assert_eq!(health.prompt_cache.max_entries_per_model, 1);
    }

    #[tokio::test]
    async fn test_health_reports_prompt_cache_backend_and_bounds() {
        let mut backends = BackendRegistry::new();
        backends.register(Arc::new(MockBackend::success().with_prompt_cache_support()));
        let config = Arc::new(PowerConfig {
            prompt_cache_max_entries: 4,
            prompt_cache_ttl_seconds: 900,
            ..Default::default()
        });
        let state = AppState::new(Arc::new(ModelRegistry::new()), Arc::new(backends), config);

        let response = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert!(health.prompt_cache.enabled);
        assert_eq!(health.prompt_cache.supported_backends, ["mock"]);
        assert_eq!(health.prompt_cache.max_entries_per_model, 4);
        assert_eq!(health.prompt_cache.ttl_seconds, 900);
        assert!(health.prompt_cache.authenticated_namespace);
    }

    #[tokio::test]
    async fn test_health_reports_effective_privacy_metric_suppression() {
        let config = Arc::new(PowerConfig {
            redact_logs: true,
            suppress_token_metrics: false,
            ..Default::default()
        });
        let privacy = DefaultPrivacyProvider::new(true).with_suppress_token_metrics(true);
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            config,
        )
        .with_privacy(Arc::new(privacy));

        let response = handler(State(state)).await.into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let health: HealthResponse = serde_json::from_slice(&body).unwrap();
        assert!(health.inference.suppress_token_metrics);
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
            speculative: SpeculativeStatus {
                mode: "mtp".to_string(),
                draft_max: Some(3),
                mtp_recurrent_snapshots: 7,
                mtp_recurrent_chain: true,
                mtp_adaptive: true,
                mtp_fr_vocab_size: Some(8192),
                draft_min: 1,
                draft_p_min: 0.25,
            },
            inference: inference_status(),
            prompt_cache: PromptCacheStatus::default(),
            tee: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"status\":\"ok\""));
        assert!(json.contains("\"uptime_seconds\":42"));
        assert!(json.contains("\"loaded_models\":3"));
        assert!(json.contains("\"mode\":\"mtp\""));
        assert!(json.contains("\"mtp_fr_vocab_size\":8192"));
        assert!(json.contains("\"gpu_layers\":-1"));
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
            speculative: SpeculativeStatus {
                mode: "off".to_string(),
                draft_max: None,
                mtp_recurrent_snapshots: 7,
                mtp_recurrent_chain: false,
                mtp_adaptive: true,
                mtp_fr_vocab_size: None,
                draft_min: 0,
                draft_p_min: 0.0,
            },
            inference: inference_status(),
            prompt_cache: PromptCacheStatus::default(),
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
