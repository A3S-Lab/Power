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
