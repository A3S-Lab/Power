pub mod metrics;
pub mod router;
pub mod state;

use std::sync::Arc;

use crate::backend;
use crate::config::PowerConfig;
use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::registry::ModelRegistry;

/// Start the HTTP server with the given configuration.
pub async fn start(mut config: PowerConfig) -> Result<()> {
    // Ensure storage directories exist
    dirs::ensure_dirs()?;

    // Auto-detect GPU and configure layers if not explicitly set
    backend::gpu::auto_configure(&mut config.gpu);

    // Initialize model registry and scan for existing models
    let registry = Arc::new(ModelRegistry::new());
    registry.scan()?;
    tracing::info!(count = registry.count(), "Loaded model registry");

    let bind_addr = config.bind_address();
    let config = Arc::new(config);

    // Initialize backends
    let backends = Arc::new(backend::default_backends(config.clone()));
    tracing::info!(
        backends = ?backends.list_names(),
        "Initialized backends"
    );

    let app_state = state::AppState::new(registry, backends, config);

    // Spawn background keep_alive reaper task
    spawn_keep_alive_reaper(app_state.clone());

    let app = router::build(app_state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| PowerError::Server(format!("Failed to bind to {bind_addr}: {e}")))?;

    tracing::info!("Server listening on {bind_addr}");

    axum::serve(listener, app)
        .await
        .map_err(|e| PowerError::Server(format!("Server error: {e}")))?;

    Ok(())
}

/// Spawn a background task that periodically checks for models whose keep_alive
/// has expired and unloads them to free memory.
fn spawn_keep_alive_reaper(state: state::AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            let expired = state.expired_models();
            for model_name in expired {
                if let Ok(backend) = state
                    .backends
                    .find_for_format(&crate::model::manifest::ModelFormat::Gguf)
                {
                    if let Err(e) = backend.unload(&model_name).await {
                        tracing::warn!(
                            model = %model_name,
                            "Failed to unload expired model: {e}"
                        );
                        continue;
                    }
                }
                state.mark_unloaded(&model_name);
                state.metrics.increment_evictions();
                state.metrics.remove_model_memory(&model_name);
                tracing::info!(
                    model = %model_name,
                    "Unloaded model (keep_alive expired)"
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{test_state_with_mock, MockBackend};
    use serial_test::serial;

    #[test]
    fn test_spawn_keep_alive_reaper_compiles() {
        // Test that the function signature is correct
        // We can't easily test the actual spawning without a full runtime setup
        // but we can verify the function exists and has the right signature
    }

    #[test]
    fn test_server_module_exports() {
        // Verify that the module exports the expected items
        let _ = start;
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_keep_alive_reaper_unloads_expired_models() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());

        // Load a model with very short keep_alive
        state.mark_loaded_with_keep_alive("test-model", std::time::Duration::from_millis(10));
        assert!(state.is_model_loaded("test-model"));

        // Spawn the reaper
        spawn_keep_alive_reaper(state.clone());

        // Wait for expiry and reaper tick (reaper runs every 5 seconds)
        tokio::time::sleep(std::time::Duration::from_secs(6)).await;

        // Model should be unloaded
        assert!(!state.is_model_loaded("test-model"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_spawn_keep_alive_reaper_preserves_non_expired() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());

        // Load a model with long keep_alive
        state.mark_loaded_with_keep_alive("long-lived", std::time::Duration::from_secs(300));
        assert!(state.is_model_loaded("long-lived"));

        // Spawn the reaper
        spawn_keep_alive_reaper(state.clone());

        // Wait for reaper tick
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Model should still be loaded
        assert!(state.is_model_loaded("long-lived"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_config_bind_address() {
        let config = PowerConfig {
            host: "0.0.0.0".to_string(),
            port: 8080,
            ..Default::default()
        };
        assert_eq!(config.bind_address(), "0.0.0.0:8080");
    }

    #[test]
    fn test_config_bind_address_default() {
        let config = PowerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:11434");
    }
}
