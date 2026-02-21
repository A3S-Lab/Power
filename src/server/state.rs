use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::backend::BackendRegistry;
use crate::config::PowerConfig;
use crate::model::registry::ModelRegistry;
use crate::server::audit::AuditLogger;
use crate::server::auth::AuthProvider;
use crate::server::lock::{read_lock, write_lock};
use crate::server::metrics::Metrics;
use crate::tee::attestation::TeeProvider;
use crate::tee::encrypted_model::{
    DecryptedModel, LayerStreamingDecryptedModel, MemoryDecryptedModel,
};
use crate::tee::key_provider::KeyProvider;
use crate::tee::privacy::PrivacyProvider;

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
    pub tee_provider: Option<Arc<dyn TeeProvider>>,
    pub privacy: Option<Arc<dyn PrivacyProvider>>,
    pub auth: Option<Arc<dyn AuthProvider>>,
    pub audit: Option<Arc<dyn AuditLogger>>,
    pub key_provider: Option<Arc<dyn KeyProvider>>,
    start_time: Instant,
    loaded_models: Arc<RwLock<HashMap<String, LoadedModelEntry>>>,
    /// RAII handles for decrypted model files. Dropping triggers secure wipe + delete.
    decrypted_models: Arc<RwLock<HashMap<String, DecryptedModel>>>,
    /// RAII handles for in-memory decrypted models. Dropping triggers zeroize + munlock.
    memory_decrypted_models: Arc<RwLock<HashMap<String, MemoryDecryptedModel>>>,
    /// RAII handles for layer-streaming decrypted models. Dropping triggers zeroize + munlock.
    streaming_decrypted_models: Arc<RwLock<HashMap<String, LayerStreamingDecryptedModel>>>,
    /// Models currently being pulled (downloading). Prevents duplicate concurrent pulls.
    pulling_models: Arc<Mutex<HashSet<String>>>,
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
            tee_provider: None,
            privacy: None,
            auth: None,
            audit: None,
            key_provider: None,
            start_time: Instant::now(),
            loaded_models: Arc::new(RwLock::new(HashMap::new())),
            decrypted_models: Arc::new(RwLock::new(HashMap::new())),
            memory_decrypted_models: Arc::new(RwLock::new(HashMap::new())),
            streaming_decrypted_models: Arc::new(RwLock::new(HashMap::new())),
            pulling_models: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Set the TEE provider for attestation.
    pub fn with_tee_provider(mut self, provider: Arc<dyn TeeProvider>) -> Self {
        self.tee_provider = Some(provider);
        self
    }

    /// Set the privacy provider for log redaction and memory zeroing.
    pub fn with_privacy(mut self, provider: Arc<dyn PrivacyProvider>) -> Self {
        self.privacy = Some(provider);
        self
    }

    /// Set the authentication provider for API key validation.
    pub fn with_auth(mut self, provider: Arc<dyn AuthProvider>) -> Self {
        self.auth = Some(provider);
        self
    }

    /// Set the audit logger for structured audit events.
    pub fn with_audit(mut self, logger: Arc<dyn AuditLogger>) -> Self {
        self.audit = Some(logger);
        self
    }

    /// Set the key provider for model decryption key management.
    pub fn with_key_provider(mut self, provider: Arc<dyn KeyProvider>) -> Self {
        self.key_provider = Some(provider);
        self
    }

    /// Whether privacy redaction is active.
    pub fn should_redact(&self) -> bool {
        self.privacy.as_ref().is_some_and(|p| p.should_redact())
    }

    /// Sanitize a log message through the privacy provider.
    pub fn sanitize_log(&self, msg: &str) -> String {
        match &self.privacy {
            Some(p) => {
                self.metrics.increment_tee_redaction();
                p.sanitize_log(msg)
            }
            None => msg.to_string(),
        }
    }

    /// How long this server has been running.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Whether the given model is currently loaded.
    pub fn is_model_loaded(&self, name: &str) -> bool {
        read_lock(&self.loaded_models).contains_key(name)
    }

    /// Number of models currently loaded.
    pub fn loaded_model_count(&self) -> usize {
        read_lock(&self.loaded_models).len()
    }

    /// Record that a model has been loaded with the default keep-alive duration from config.
    pub fn mark_loaded(&self, name: &str) {
        let keep_alive = crate::config::parse_keep_alive(&self.config.keep_alive);
        let was_absent = write_lock(&self.loaded_models)
            .insert(
                name.to_string(),
                LoadedModelEntry {
                    last_used: Instant::now(),
                    keep_alive,
                },
            )
            .is_none();
        if was_absent {
            self.metrics.models_loaded.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record that a model has been loaded with a specific keep-alive duration.
    pub fn mark_loaded_with_keep_alive(&self, name: &str, keep_alive: Duration) {
        let was_absent = write_lock(&self.loaded_models)
            .insert(
                name.to_string(),
                LoadedModelEntry {
                    last_used: Instant::now(),
                    keep_alive,
                },
            )
            .is_none();
        if was_absent {
            self.metrics.models_loaded.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record that a model has been unloaded.
    /// Also removes any decrypted model handle, triggering secure wipe.
    pub fn mark_unloaded(&self, name: &str) {
        let was_present = write_lock(&self.loaded_models).remove(name).is_some();
        if was_present {
            self.metrics.models_loaded.fetch_sub(1, Ordering::Relaxed);
        }
        if let Some(dec) = write_lock(&self.decrypted_models).remove(name) {
            tracing::info!(model = %name, "Cleaning up decrypted model file");
            drop(dec); // triggers DecryptedModel::drop → zero-fill + delete
        }
        if let Some(mem) = write_lock(&self.memory_decrypted_models).remove(name) {
            tracing::info!(model = %name, "Cleaning up in-memory decrypted model");
            drop(mem); // triggers MemoryDecryptedModel::drop → zeroize + munlock
        }
        if let Some(stream) = write_lock(&self.streaming_decrypted_models).remove(name) {
            tracing::info!(model = %name, "Cleaning up layer-streaming decrypted model");
            drop(stream); // triggers LayerStreamingDecryptedModel::drop → zeroize + munlock
        }
    }

    /// Store a decrypted model handle for RAII cleanup on unload.
    pub fn store_decrypted(&self, name: &str, handle: DecryptedModel) {
        write_lock(&self.decrypted_models).insert(name.to_string(), handle);
    }

    /// Store an in-memory decrypted model handle for RAII cleanup on unload.
    pub fn store_memory_decrypted(&self, name: &str, handle: MemoryDecryptedModel) {
        write_lock(&self.memory_decrypted_models).insert(name.to_string(), handle);
    }

    /// Store a layer-streaming decrypted model handle for RAII cleanup on unload.
    pub fn store_streaming_decrypted(&self, name: &str, handle: LayerStreamingDecryptedModel) {
        write_lock(&self.streaming_decrypted_models).insert(name.to_string(), handle);
    }

    /// Sanitize an error message through the privacy provider (strips paths, internal state).
    pub fn sanitize_error(&self, err: &str) -> String {
        match &self.privacy {
            Some(p) => p.sanitize_error(err),
            None => err.to_string(),
        }
    }

    /// Find the best backend for a model, using TEE-aware routing when in TEE mode.
    ///
    /// In TEE mode, delegates to `BackendRegistry::find_for_tee()` which prefers
    /// picolm (layer-streaming) when the model exceeds 75% of available EPC memory.
    /// Outside TEE mode, uses standard priority-based format matching.
    pub fn find_backend(
        &self,
        format: &crate::model::manifest::ModelFormat,
        model_size_bytes: u64,
    ) -> crate::error::Result<std::sync::Arc<dyn crate::backend::Backend>> {
        if self.config.tee_mode {
            self.backends.find_for_tee(format, model_size_bytes)
        } else {
            self.backends.find_for_format(format)
        }
    }

    /// Whether token counts in responses should be rounded (side-channel mitigation).
    pub fn suppress_token_metrics(&self) -> bool {
        self.privacy
            .as_ref()
            .is_some_and(|p| p.should_suppress_token_metrics())
    }

    /// Compute the timing padding duration for this request, with ±20% jitter.
    ///
    /// Returns `None` when `timing_padding_ms` is not configured.
    /// The jitter prevents statistical timing attacks by varying the actual
    /// delay within [padding * 0.8, padding * 1.2].
    pub fn timing_padding(&self) -> Option<std::time::Duration> {
        let base_ms = self.config.timing_padding_ms?;
        if base_ms == 0 {
            return None;
        }
        // Apply ±20% jitter using subsecond nanos as a cheap entropy source.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        // Map nanos % 1000 → jitter_factor in [0.8, 1.2]
        let jitter_factor = 0.8 + (nanos % 1000) as f64 / 2500.0;
        let padded_ms = (base_ms as f64 * jitter_factor) as u64;
        Some(std::time::Duration::from_millis(padded_ms))
    }

    /// Update the last-used time for a loaded model.
    pub fn touch_model(&self, name: &str) {
        if let Some(entry) = write_lock(&self.loaded_models).get_mut(name) {
            entry.last_used = Instant::now();
        }
    }

    /// Return the name of the least-recently-used loaded model, if any.
    pub fn lru_model(&self) -> Option<String> {
        read_lock(&self.loaded_models)
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(name, _)| name.clone())
    }

    /// Return the name of the least-recently-used model whose keep-alive has expired.
    /// Models with `keep_alive == Duration::MAX` are never evicted.
    pub fn evictable_lru_model(&self) -> Option<String> {
        read_lock(&self.loaded_models)
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
        read_lock(&self.loaded_models).keys().cloned().collect()
    }
    /// Return all models whose keep-alive has expired (eligible for background unloading).
    /// Models with `keep_alive == Duration::MAX` are never expired.
    pub fn expired_models(&self) -> Vec<String> {
        read_lock(&self.loaded_models)
            .iter()
            .filter(|(_, entry)| {
                entry.keep_alive != Duration::MAX
                    && entry.keep_alive != Duration::ZERO
                    && entry.last_used.elapsed() >= entry.keep_alive
            })
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Return the expiry timestamp for a loaded model.
    /// Returns `None` if the model is not loaded or has infinite keep-alive.
    pub fn model_expires_at(&self, name: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        let models = read_lock(&self.loaded_models);
        models.get(name).and_then(|entry| {
            if entry.keep_alive == Duration::MAX {
                None
            } else {
                let elapsed = entry.last_used.elapsed();
                let remaining = entry.keep_alive.saturating_sub(elapsed);
                Some(chrono::Utc::now() + chrono::Duration::from_std(remaining).unwrap_or_default())
            }
        })
    }

    /// Mark a model as currently being pulled. Returns `false` if already pulling.
    pub fn start_pull(&self, name: &str) -> bool {
        self.pulling_models
            .lock()
            .map(|mut s| s.insert(name.to_string()))
            .unwrap_or(false)
    }

    /// Mark a model pull as finished (success or failure).
    pub fn finish_pull(&self, name: &str) {
        if let Ok(mut s) = self.pulling_models.lock() {
            s.remove(name);
        }
    }

    /// Whether a model is currently being pulled.
    pub fn is_pulling(&self, name: &str) -> bool {
        self.pulling_models
            .lock()
            .map(|s| s.contains(name))
            .unwrap_or(false)
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
        assert_eq!(state.config.port, 11434);
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

    #[test]
    fn test_store_decrypted_and_cleanup_on_unload() {
        use crate::tee::encrypted_model::{encrypt_model_file, DecryptedModel};

        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"test data for decrypted cleanup").unwrap();

        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let decrypted = DecryptedModel::decrypt(&enc_path, &key).unwrap();
        let dec_path = decrypted.path.clone();

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("enc-model");
        state.store_decrypted("enc-model", decrypted);
        assert!(dec_path.exists());

        // Unloading should trigger DecryptedModel drop → secure wipe
        state.mark_unloaded("enc-model");
        assert!(
            !dec_path.exists(),
            "Decrypted file should be wiped on unload"
        );
    }

    #[test]
    fn test_models_loaded_gauge_increments_on_load() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        state.mark_loaded("model-a");
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        state.mark_loaded("model-b");
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        // Re-inserting the same model does not double-count
        state.mark_loaded("model-a");
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn test_models_loaded_gauge_decrements_on_unload() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("model-a");
        state.mark_loaded("model-b");
        state.mark_unloaded("model-a");
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        // Unloading a model that was never loaded should not wrap around
        state.mark_unloaded("never-loaded");
        assert_eq!(
            state
                .metrics
                .models_loaded
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_suppress_token_metrics_false_by_default() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert!(!state.suppress_token_metrics());
    }

    #[test]
    fn test_timing_padding_none_by_default() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        assert!(state.timing_padding().is_none());
    }

    #[test]
    fn test_timing_padding_zero_returns_none() {
        let mut config = PowerConfig::default();
        config.timing_padding_ms = Some(0);
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        assert!(state.timing_padding().is_none());
    }

    #[test]
    fn test_timing_padding_returns_duration_within_jitter_range() {
        let mut config = PowerConfig::default();
        config.timing_padding_ms = Some(100);
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        let pad = state.timing_padding().unwrap();
        // Must be within [80ms, 120ms] (±20% jitter)
        assert!(
            pad.as_millis() >= 80,
            "padding too short: {}ms",
            pad.as_millis()
        );
        assert!(
            pad.as_millis() <= 120,
            "padding too long: {}ms",
            pad.as_millis()
        );
    }

    #[test]
    fn test_suppress_token_metrics_true_when_privacy_redacts() {
        use crate::tee::privacy::DefaultPrivacyProvider;
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
        .with_privacy(Arc::new(DefaultPrivacyProvider::new(true)));
        assert!(state.suppress_token_metrics());
    }

    #[test]
    fn test_store_memory_decrypted_and_cleanup_on_unload() {
        use crate::tee::encrypted_model::{encrypt_model_file, MemoryDecryptedModel};

        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"test data for memory decryption").unwrap();

        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let mem_model = MemoryDecryptedModel::decrypt(&enc_path, &key).unwrap();
        assert!(!mem_model.is_empty());

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("mem-model");
        state.store_memory_decrypted("mem-model", mem_model);

        // Unloading should trigger MemoryDecryptedModel drop → zeroize + munlock
        state.mark_unloaded("mem-model");
        assert!(!state.is_model_loaded("mem-model"));
    }

    #[test]
    fn test_sanitize_error_without_privacy() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        // Without privacy provider, error is returned as-is
        let err = "Failed to load /home/user/models/secret.gguf: permission denied";
        assert_eq!(state.sanitize_error(err), err);
    }

    #[test]
    fn test_sanitize_error_with_privacy() {
        let privacy = crate::tee::privacy::DefaultPrivacyProvider::new(true);
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
        .with_privacy(Arc::new(privacy));
        // With privacy provider, content after sensitive prefixes is redacted
        let err = "content: user said something secret";
        let sanitized = state.sanitize_error(err);
        assert!(sanitized.contains("[REDACTED]"));
        assert!(!sanitized.contains("secret"));
    }

    #[test]
    fn test_find_backend_non_tee_mode() {
        use crate::model::manifest::ModelFormat;
        let config = Arc::new(PowerConfig::default());
        let backends = Arc::new(crate::backend::default_backends(config.clone()));
        let state = AppState::new(Arc::new(ModelRegistry::new()), backends, config);
        // Non-TEE mode: should use find_for_format (standard priority)
        let result = state.find_backend(&ModelFormat::Gguf, 1_000_000);
        // Should succeed if any backend is registered
        #[cfg(feature = "mistralrs")]
        assert!(result.is_ok());
    }

    #[test]
    fn test_find_backend_tee_mode() {
        use crate::model::manifest::ModelFormat;
        let config = Arc::new(PowerConfig {
            tee_mode: true,
            ..Default::default()
        });
        let backends = Arc::new(crate::backend::default_backends(config.clone()));
        let state = AppState::new(Arc::new(ModelRegistry::new()), backends, config);
        // TEE mode: should use find_for_tee (EPC-aware routing)
        let result = state.find_backend(&ModelFormat::Gguf, 1_000_000);
        #[cfg(feature = "mistralrs")]
        assert!(result.is_ok());
    }

    #[test]
    fn test_store_streaming_decrypted_and_cleanup_on_unload() {
        use crate::tee::encrypted_model::{encrypt_model_file, LayerStreamingDecryptedModel};

        let dir = tempfile::tempdir().unwrap();
        let plain_path = dir.path().join("model.gguf");
        std::fs::write(&plain_path, b"test data for streaming decryption").unwrap();

        let mut key = [0u8; 32];
        for (i, b) in key.iter_mut().enumerate() {
            *b = i as u8;
        }
        let enc_path = encrypt_model_file(&plain_path, &key).unwrap();
        let stream_model = LayerStreamingDecryptedModel::decrypt(&enc_path, &key).unwrap();

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        state.mark_loaded("stream-model");
        state.store_streaming_decrypted("stream-model", stream_model);

        // Unloading should trigger LayerStreamingDecryptedModel drop → zeroize + munlock
        state.mark_unloaded("stream-model");
        assert!(!state.is_model_loaded("stream-model"));
    }
}
