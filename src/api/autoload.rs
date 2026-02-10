use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::backend::Backend;
use crate::config::parse_keep_alive;
use crate::error::Result;
use crate::model::manifest::ModelManifest;
use crate::server::state::AppState;

/// Result of an ensure_loaded call, including load timing.
pub struct LoadResult {
    /// Time spent loading the model. Zero if the model was already loaded (cache hit).
    pub load_duration: Duration,
}

/// Ensure a model is loaded before inference.
///
/// If the model is already loaded, updates its last-used time.
/// If not loaded, evicts the LRU model when at capacity, then loads.
/// An optional `keep_alive` string from the request overrides the config default.
pub async fn ensure_loaded(
    state: &AppState,
    model_name: &str,
    manifest: &ModelManifest,
    backend: &Arc<dyn Backend>,
) -> Result<LoadResult> {
    ensure_loaded_with_keep_alive(state, model_name, manifest, backend, None).await
}

/// Ensure a model is loaded with an optional per-request keep-alive override.
pub async fn ensure_loaded_with_keep_alive(
    state: &AppState,
    model_name: &str,
    manifest: &ModelManifest,
    backend: &Arc<dyn Backend>,
    keep_alive: Option<&str>,
) -> Result<LoadResult> {
    if state.is_model_loaded(model_name) {
        state.touch_model(model_name);
        return Ok(LoadResult {
            load_duration: Duration::ZERO,
        });
    }

    // Evict models: prefer evicting those whose keep_alive has expired first,
    // then fall back to LRU eviction
    while state.needs_eviction() {
        if let Some(evictable) = state.evictable_lru_model() {
            backend.unload(&evictable).await?;
            state.mark_unloaded(&evictable);
            state.metrics.increment_evictions();
            state.metrics.remove_model_memory(&evictable);
        } else if let Some(lru_name) = state.lru_model() {
            backend.unload(&lru_name).await?;
            state.mark_unloaded(&lru_name);
            state.metrics.increment_evictions();
            state.metrics.remove_model_memory(&lru_name);
        } else {
            break;
        }
    }

    let load_start = Instant::now();

    // Log memory estimate before loading (GGUF models only)
    if manifest.format == crate::model::manifest::ModelFormat::Gguf && manifest.path.exists() {
        match crate::model::gguf::estimate_memory(&manifest.path, 2048) {
            Ok(estimate) => {
                tracing::info!(
                    model = %model_name,
                    model_size = %format_bytes(estimate.model_size),
                    kv_cache = %format_bytes(estimate.kv_cache_size),
                    total_estimate = %estimate.total_display(),
                    ctx_size = estimate.context_size,
                    "Memory estimate before loading"
                );
            }
            Err(e) => {
                tracing::debug!(
                    model = %model_name,
                    error = %e,
                    "Could not estimate memory requirements"
                );
            }
        }
    }

    backend.load(manifest).await?;
    let load_duration = load_start.elapsed();

    // Record model load duration and estimated memory (file size as proxy)
    state
        .metrics
        .record_model_load(model_name, load_duration.as_secs_f64());
    state.metrics.set_model_memory(model_name, manifest.size);

    match keep_alive {
        Some(ka) => {
            let duration = parse_keep_alive(ka);
            state.mark_loaded_with_keep_alive(model_name, duration);

            // If keep_alive is "0", immediately schedule unload after request
            if duration == Duration::ZERO {
                backend.unload(model_name).await?;
                state.mark_unloaded(model_name);
            }
        }
        None => {
            state.mark_loaded(model_name);
        }
    }

    Ok(LoadResult { load_duration })
}

/// Format bytes as a human-readable string for log messages.
fn format_bytes(bytes: u64) -> String {
    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    if bytes >= GB {
        format!("{:.1} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MiB", bytes as f64 / MB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::model::manifest::{ModelFormat, ModelManifest};
    use crate::model::registry::ModelRegistry;

    fn test_state() -> AppState {
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
    }

    fn dummy_manifest() -> ModelManifest {
        ModelManifest {
            name: "test-model".to_string(),
            format: ModelFormat::Gguf,
            size: 0,
            sha256: "abc".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from("/tmp/fake.gguf"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        }
    }

    #[tokio::test]
    async fn test_ensure_loaded_skips_when_already_loaded() {
        let state = test_state();
        let manifest = dummy_manifest();
        let config = Arc::new(PowerConfig::default());
        let backend = crate::backend::default_backends(config)
            .find_for_format(&ModelFormat::Gguf)
            .unwrap();

        // Pre-mark the model as loaded — ensure_loaded should return Ok
        // without calling backend.load().
        state.mark_loaded("test-model");
        let result = ensure_loaded(&state, "test-model", &manifest, &backend).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_ensure_loaded_attempts_load_when_not_loaded() {
        let state = test_state();
        let manifest = dummy_manifest();
        let config = Arc::new(PowerConfig::default());
        let backend = crate::backend::default_backends(config)
            .find_for_format(&ModelFormat::Gguf)
            .unwrap();

        // Model is not marked loaded, so ensure_loaded will call backend.load()
        // which fails because there is no real model file — that's expected.
        let result = ensure_loaded(&state, "test-model", &manifest, &backend).await;
        assert!(result.is_err());
        // Model should NOT be marked loaded on failure.
        assert!(!state.is_model_loaded("test-model"));
    }

    #[tokio::test]
    async fn test_ensure_loaded_evicts_lru_when_at_capacity() {
        let state = test_state_with_mock(MockBackend::success());
        // Default max_loaded_models is 1, so loading a second model should evict the first.
        let manifest_a = sample_manifest("model-a");
        let manifest_b = sample_manifest("model-b");
        let backend = state.backends.find_for_format(&ModelFormat::Gguf).unwrap();

        // Load model-a
        ensure_loaded(&state, "model-a", &manifest_a, &backend)
            .await
            .unwrap();
        assert!(state.is_model_loaded("model-a"));
        assert_eq!(state.loaded_model_count(), 1);

        // Load model-b — should evict model-a
        ensure_loaded(&state, "model-b", &manifest_b, &backend)
            .await
            .unwrap();
        assert!(state.is_model_loaded("model-b"));
        assert!(!state.is_model_loaded("model-a"));
        assert_eq!(state.loaded_model_count(), 1);
    }

    #[tokio::test]
    async fn test_ensure_loaded_touches_on_cache_hit() {
        let config = Arc::new(PowerConfig {
            max_loaded_models: 3,
            ..Default::default()
        });
        let mut backends = BackendRegistry::new();
        backends.register(Arc::new(MockBackend::success()));
        let state = AppState::new(Arc::new(ModelRegistry::new()), Arc::new(backends), config);

        let manifest_a = sample_manifest("model-a");
        let manifest_b = sample_manifest("model-b");
        let backend = state.backends.find_for_format(&ModelFormat::Gguf).unwrap();

        // Load both models
        ensure_loaded(&state, "model-a", &manifest_a, &backend)
            .await
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        ensure_loaded(&state, "model-b", &manifest_b, &backend)
            .await
            .unwrap();

        // model-a is LRU
        assert_eq!(state.lru_model(), Some("model-a".to_string()));

        // Touch model-a via ensure_loaded (cache hit)
        std::thread::sleep(std::time::Duration::from_millis(10));
        ensure_loaded(&state, "model-a", &manifest_a, &backend)
            .await
            .unwrap();

        // Now model-b should be LRU
        assert_eq!(state.lru_model(), Some("model-b".to_string()));
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(2_147_483_648), "2.0 GiB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GiB");
        assert_eq!(format_bytes(5_368_709_120), "5.0 GiB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1_048_576), "1 MiB");
        assert_eq!(format_bytes(524_288_000), "500 MiB");
    }

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1024), "1024 B");
        assert_eq!(format_bytes(999_999), "999999 B");
    }

    #[tokio::test]
    async fn test_ensure_loaded_with_keep_alive_marks_loaded() {
        let state = test_state_with_mock(MockBackend::success());
        let manifest = sample_manifest("ka-model");
        let backend = state.backends.find_for_format(&ModelFormat::Gguf).unwrap();

        let result = ensure_loaded_with_keep_alive(
            &state, "ka-model", &manifest, &backend, Some("5m"),
        ).await;
        assert!(result.is_ok());
        assert!(state.is_model_loaded("ka-model"));
    }

    #[tokio::test]
    async fn test_ensure_loaded_with_keep_alive_zero_unloads_immediately() {
        let state = test_state_with_mock(MockBackend::success());
        let manifest = sample_manifest("zero-model");
        let backend = state.backends.find_for_format(&ModelFormat::Gguf).unwrap();

        let result = ensure_loaded_with_keep_alive(
            &state, "zero-model", &manifest, &backend, Some("0"),
        ).await;
        assert!(result.is_ok());
        // keep_alive=0 should immediately unload
        assert!(!state.is_model_loaded("zero-model"));
    }

    #[tokio::test]
    async fn test_ensure_loaded_cache_hit_returns_zero_duration() {
        let state = test_state_with_mock(MockBackend::success());
        let manifest = sample_manifest("cached");
        let backend = state.backends.find_for_format(&ModelFormat::Gguf).unwrap();

        // First load
        ensure_loaded(&state, "cached", &manifest, &backend).await.unwrap();
        assert!(state.is_model_loaded("cached"));

        // Second call should be cache hit with zero duration
        let result = ensure_loaded(&state, "cached", &manifest, &backend).await.unwrap();
        assert_eq!(result.load_duration, Duration::ZERO);
    }
}
