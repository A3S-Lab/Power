use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use crate::backend::BackendRegistry;
use crate::config::PowerConfig;
use crate::model::registry::ModelRegistry;
use crate::server::metrics::Metrics;

/// Tracks a loaded model's last-used time and keep-alive duration for LRU eviction.
#[derive(Debug, Clone)]
struct LoadedModelEntry {
    last_used: Instant,
    keep_alive: Duration,
}

/// Shared application state accessible to all HTTP handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    pub backends: Arc<BackendRegistry>,
    pub config: Arc<PowerConfig>,
    pub metrics: Arc<Metrics>,
    start_time: Instant,
    loaded_models: Arc<RwLock<HashMap<String, LoadedModelEntry>>>,
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
            metrics: Arc::new(Metrics::new()),
            start_time: Instant::now(),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// How long this server has been running.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Whether the given model is currently loaded.
    pub fn is_model_loaded(&self, name: &str) -> bool {
        self.loaded_models.read().unwrap().contains_key(name)
    }

    /// Number of models currently loaded.
    pub fn loaded_model_count(&self) -> usize {
        self.loaded_models.read().unwrap().len()
    }

    /// Record that a model has been loaded with the default keep-alive duration from config.
    pub fn mark_loaded(&self, name: &str) {
        let keep_alive = crate::config::parse_keep_alive(&self.config.keep_alive);
        self.loaded_models.write().unwrap().insert(
            name.to_string(),
            LoadedModelEntry {
                last_used: Instant::now(),
                keep_alive,
            },
        );
    }

    /// Record that a model has been loaded with a specific keep-alive duration.
    pub fn mark_loaded_with_keep_alive(&self, name: &str, keep_alive: Duration) {
        self.loaded_models.write().unwrap().insert(
            name.to_string(),
            LoadedModelEntry {
                last_used: Instant::now(),
                keep_alive,
            },
        );
    }

    /// Record that a model has been unloaded.
    pub fn mark_unloaded(&self, name: &str) {
        self.loaded_models.write().unwrap().remove(name);
    }

    /// Update the last-used time for a loaded model.
    pub fn touch_model(&self, name: &str) {
        if let Some(entry) = self.loaded_models.write().unwrap().get_mut(name) {
            entry.last_used = Instant::now();
        }
    }

    /// Return the name of the least-recently-used loaded model, if any.
    pub fn lru_model(&self) -> Option<String> {
        self.loaded_models
            .read()
            .unwrap()
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(name, _)| name.clone())
    }

    /// Return the name of the least-recently-used model whose keep-alive has expired.
    /// Models with `keep_alive == Duration::MAX` are never evicted.
    pub fn evictable_lru_model(&self) -> Option<String> {
        self.loaded_models
            .read()
            .unwrap()
            .iter()
            .filter(|(_, entry)| {
                entry.keep_alive != Duration::MAX && entry.last_used.elapsed() >= entry.keep_alive
            })
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(name, _)| name.clone())
    }

    /// Whether the number of loaded models has reached the configured maximum.
    pub fn needs_eviction(&self) -> bool {
        self.loaded_model_count() >= self.config.max_loaded_models
    }

    /// Return the names of all currently loaded models.
    pub fn loaded_model_names(&self) -> Vec<String> {
        self.loaded_models.read().unwrap().keys().cloned().collect()
    }
    /// Return all models whose keep-alive has expired (eligible for background unloading).
    /// Models with `keep_alive == Duration::MAX` are never expired.
    pub fn expired_models(&self) -> Vec<String> {
        self.loaded_models
            .read()
            .unwrap()
            .iter()
            .filter(|(_, entry)| {
                entry.keep_alive != Duration::MAX
                    && entry.keep_alive != Duration::ZERO
                    && entry.last_used.elapsed() >= entry.keep_alive
            })
            .map(|(name, _)| name.clone())
            .collect()
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

    #[test]
    fn test_app_state_loaded_model_names() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("model-a");
        state.mark_loaded("model-b");
        let mut names = state.loaded_model_names();
        names.sort();
        assert_eq!(names, vec!["model-a", "model-b"]);
    }

    #[test]
    fn test_lru_returns_oldest_model() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("model-a");
        // Small sleep to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.mark_loaded("model-b");

        assert_eq!(state.lru_model(), Some("model-a".to_string()));
    }

    #[test]
    fn test_touch_updates_lru_order() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("model-a");
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.mark_loaded("model-b");

        // Touch model-a so it becomes most recently used
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.touch_model("model-a");

        assert_eq!(state.lru_model(), Some("model-b".to_string()));
    }

    #[test]
    fn test_needs_eviction_at_capacity() {
        let config = PowerConfig {
            max_loaded_models: 2,
            ..Default::default()
        };
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        state.mark_loaded("model-a");
        assert!(!state.needs_eviction());

        state.mark_loaded("model-b");
        assert!(state.needs_eviction());
    }

    #[test]
    fn test_lru_empty_returns_none() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert_eq!(state.lru_model(), None);
    }

    #[test]
    fn test_expired_models_empty_when_no_models() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert!(state.expired_models().is_empty());
    }

    #[test]
    fn test_expired_models_not_expired_yet() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        // Default keep_alive is "5m" — model was just loaded, not expired
        state.mark_loaded("model-a");
        assert!(state.expired_models().is_empty());
    }

    #[test]
    fn test_expired_models_with_tiny_keep_alive() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        // Set a 1ms keep_alive so it expires immediately
        state.mark_loaded_with_keep_alive("model-a", Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(5));
        let expired = state.expired_models();
        assert_eq!(expired, vec!["model-a"]);
    }

    #[test]
    fn test_expired_models_never_expires_with_max_duration() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        // Duration::MAX means never unload
        state.mark_loaded_with_keep_alive("model-a", Duration::MAX);
        assert!(state.expired_models().is_empty());
    }

    #[test]
    fn test_expired_models_zero_duration_not_reaped() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        // Duration::ZERO means "unload immediately after request" — handled in autoload,
        // not by the background reaper.
        state.mark_loaded_with_keep_alive("model-a", Duration::ZERO);
        assert!(state.expired_models().is_empty());
    }

    #[test]
    fn test_touch_resets_expiry() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded_with_keep_alive("model-a", Duration::from_millis(50));
        std::thread::sleep(Duration::from_millis(30));
        // Touch resets the timer
        state.touch_model("model-a");
        std::thread::sleep(Duration::from_millis(30));
        // 30ms after touch, 50ms keep_alive — should NOT be expired
        assert!(state.expired_models().is_empty());
        // Wait for full expiry
        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(state.expired_models(), vec!["model-a"]);
    }
}
