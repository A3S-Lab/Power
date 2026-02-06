use std::sync::Arc;

use crate::backend::Backend;
use crate::error::Result;
use crate::model::manifest::ModelManifest;
use crate::server::state::AppState;

/// Ensure a model is loaded before inference. Skips if already loaded.
pub async fn ensure_loaded(
    state: &AppState,
    model_name: &str,
    manifest: &ModelManifest,
    backend: &Arc<dyn Backend>,
) -> Result<()> {
    if state.is_model_loaded(model_name) {
        return Ok(());
    }
    backend.load(manifest).await?;
    state.mark_loaded(model_name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        }
    }

    #[tokio::test]
    async fn test_ensure_loaded_skips_when_already_loaded() {
        let state = test_state();
        let manifest = dummy_manifest();
        let backend = crate::backend::default_backends()
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
        let backend = crate::backend::default_backends()
            .find_for_format(&ModelFormat::Gguf)
            .unwrap();

        // Model is not marked loaded, so ensure_loaded will call backend.load()
        // which fails because there is no real model file — that's expected.
        let result = ensure_loaded(&state, "test-model", &manifest, &backend).await;
        assert!(result.is_err());
        // Model should NOT be marked loaded on failure.
        assert!(!state.is_model_loaded("test-model"));
    }
}
