use super::*;
use crate::{
    backend::{xolotl::XolotlBackend, BackendRegistry},
    config::{PowerConfig, XolotlManagedConfig},
    model::registry::ModelRegistry,
};
use std::{path::PathBuf, sync::Arc};
#[tokio::test]
async fn managed_xolotl_without_a_live_engine_cannot_report_http_health_ok() {
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.managed = Some(XolotlManagedConfig {
        model_sha256: "1".repeat(64),
        model_parts: Vec::new(),
        gpu_uuids: vec!["GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".into()],
        model_name: "bound-model".into(),
        python: PathBuf::from("/abs/python"),
        bridge: PathBuf::from("/abs/bridge.py"),
        server: PathBuf::from("/abs/server.py"),
        tokenizer: PathBuf::from("/abs/tokenizer"),
        startup_timeout_seconds: 10,
        shutdown_timeout_seconds: 1,
    });
    let config = Arc::new(config);
    let mut backends = BackendRegistry::new();
    backends.register(Arc::new(XolotlBackend::new(config.clone())));
    let state = AppState::new(Arc::new(ModelRegistry::new()), Arc::new(backends), config);
    let response = handler(State(state)).await.into_response();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let health: HealthResponse = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(health.status, "not_ready");
    assert_eq!(health.loaded_models, 0);
}
