use std::collections::HashSet;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::backend::BackendRegistry;
use crate::config::PowerConfig;
use crate::model::registry::ModelRegistry;

/// Shared application state accessible to all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub backends: Arc<BackendRegistry>,
    pub config: Arc<PowerConfig>,
    start_time: Instant,
    loaded_models: Arc<RwLock<HashSet<String>>>,
}

impl AppState {
    pub fn new(
        registry: Arc<ModelRegistry>,
        backends: Arc<BackendRegistry>,
        config: Arc<PowerConfig>,
    ) -> Self {
        Self {
            registry,
            backends,
            config,
            start_time: Instant::now(),
            loaded_models: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// How long this server has been running.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Whether the given model is currently loaded.
    pub fn is_model_loaded(&self, name: &str) -> bool {
        self.loaded_models.read().unwrap().contains(name)
    }

    /// Number of models currently loaded.
    pub fn loaded_model_count(&self) -> usize {
        self.loaded_models.read().unwrap().len()
    }

    /// Record that a model has been loaded.
    pub fn mark_loaded(&self, name: &str) {
        self.loaded_models.write().unwrap().insert(name.to_string());
    }

    /// Record that a model has been unloaded.
    pub fn mark_unloaded(&self, name: &str) {
        self.loaded_models.write().unwrap().remove(name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_new() {
        let registry = Arc::new(ModelRegistry::new());
        let backends = Arc::new(BackendRegistry::new());
        let config = Arc::new(PowerConfig::default());

        let state = AppState::new(registry.clone(), backends, config);
        assert_eq!(state.registry.count(), 0);
        assert_eq!(state.config.port, 11435);
    }

    #[test]
    fn test_app_state_clone() {
        let registry = Arc::new(ModelRegistry::new());
        let backends = Arc::new(BackendRegistry::new());
        let config = Arc::new(PowerConfig::default());

        let state = AppState::new(registry, backends, config);
        let cloned = state.clone();
        assert_eq!(cloned.config.host, "127.0.0.1");
    }

    #[test]
    fn test_app_state_tracks_start_time() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        let uptime = state.uptime();
        assert!(uptime.as_secs() < 1);
    }

    #[test]
    fn test_app_state_loaded_models_initially_empty() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert_eq!(state.loaded_model_count(), 0);
        assert!(!state.is_model_loaded("any-model"));
    }

    #[test]
    fn test_app_state_mark_model_loaded() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("llama-7b");
        assert!(state.is_model_loaded("llama-7b"));
        assert_eq!(state.loaded_model_count(), 1);

        // Marking the same model again doesn't duplicate.
        state.mark_loaded("llama-7b");
        assert_eq!(state.loaded_model_count(), 1);
    }

    #[test]
    fn test_app_state_mark_model_unloaded() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("llama-7b");
        assert!(state.is_model_loaded("llama-7b"));

        state.mark_unloaded("llama-7b");
        assert!(!state.is_model_loaded("llama-7b"));
        assert_eq!(state.loaded_model_count(), 0);
    }
}
