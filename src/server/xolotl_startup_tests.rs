use super::*;
use crate::{backend::BackendRegistry, config::XolotlManagedConfig};
fn config() -> PowerConfig {
    let mut config = PowerConfig::default();
    config.xolotl.enabled = true;
    config.xolotl.managed = Some(XolotlManagedConfig {
        model_sha256: "1".repeat(64),
        model_parts: Vec::new(),
        gpu_uuids: vec!["GPU-511088e2-3e4c-2a27-42c4-fd6659c7fbfc".into()],
        model_name: "exact-model".into(),
        python: "/abs/python".into(),
        bridge: "/abs/bridge.py".into(),
        server: "/abs/server.py".into(),
        tokenizer: "/abs/tokenizer".into(),
        startup_timeout_seconds: 10,
        shutdown_timeout_seconds: 1,
    });
    config
}
#[tokio::test]
async fn managed_xolotl_startup_refuses_disabled_or_unregistered_model() {
    for enabled in [false, true] {
        let mut config = config();
        config.xolotl.enabled = enabled;
        let state = state::AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        assert!(load_managed_xolotl(&state).await.is_err());
        assert_eq!(state.loaded_model_count(), 0);
    }
}
#[tokio::test]
async fn managed_xolotl_startup_cannot_route_a_foreign_manifest_into_a_ready_worker() {
    let registry = Arc::new(ModelRegistry::new());
    registry
        .register_transient(crate::model::manifest::ModelManifest::remote("exact-model"))
        .unwrap();
    let state = state::AppState::new(
        registry,
        Arc::new(BackendRegistry::new()),
        Arc::new(config()),
    );
    let error = load_managed_xolotl(&state).await.unwrap_err().to_string();
    assert!(error.contains("exact XQ manifest"));
    assert_eq!(state.loaded_model_count(), 0);
}
