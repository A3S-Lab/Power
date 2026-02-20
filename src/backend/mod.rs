pub mod chat_template;
pub mod gpu;
pub mod json_schema;
pub mod llamacpp;
pub mod mistralrs_backend;
/// Test utilities for integration tests. Not part of the public API.
#[doc(hidden)]
pub mod test_utils;
pub mod think_parser;
pub mod tool_parser;
pub mod types;

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};
use crate::server::request_context::RequestContext;

use types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};

/// Trait for inference backends that can load models and run inference.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Human-readable name of this backend.
    fn name(&self) -> &str;

    /// Check if this backend can serve the given model format.
    fn supports(&self, format: &ModelFormat) -> bool;

    /// Load a model into memory, ready for inference.
    async fn load(&self, manifest: &ModelManifest) -> Result<()>;

    /// Unload a model from memory.
    async fn unload(&self, model_name: &str) -> Result<()>;

    /// Run chat completion inference, returning a stream of token chunks.
    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>>;

    /// Run text completion inference, returning a stream of token chunks.
    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>>;

    /// Generate embeddings for the given input texts.
    async fn embed(&self, model_name: &str, request: EmbeddingRequest)
        -> Result<EmbeddingResponse>;

    /// Clean up request-scoped resources after inference completes.
    ///
    /// Default: no-op. Backends that support KV cache isolation should
    /// override this to clear the request's cache slot, zeroize
    /// intermediate buffers, etc.
    async fn cleanup_request(&self, _model_name: &str, _ctx: &RequestContext) -> Result<()> {
        Ok(())
    }
}

/// Registry of available inference backends.
///
/// Backends are stored in priority order — the first registered backend that
/// supports a given format wins. This allows callers to control preference
/// (e.g., register mistral.rs before llama.cpp for pure-Rust priority).
pub struct BackendRegistry {
    backends: Vec<Arc<dyn Backend>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a new backend. Backends registered first have higher priority.
    pub fn register(&mut self, backend: Arc<dyn Backend>) {
        self.backends.push(backend);
    }

    /// Find the highest-priority backend that supports the given model format.
    pub fn find_for_format(&self, format: &ModelFormat) -> Result<Arc<dyn Backend>> {
        self.backends
            .iter()
            .find(|b| b.supports(format))
            .cloned()
            .ok_or_else(|| {
                PowerError::BackendNotAvailable(format!(
                    "No backend available for format: {format}"
                ))
            })
    }

    /// Find a backend by name.
    pub fn find_by_name(&self, name: &str) -> Result<Arc<dyn Backend>> {
        self.backends
            .iter()
            .find(|b| b.name() == name)
            .cloned()
            .ok_or_else(|| PowerError::BackendNotAvailable(format!("No backend with name: {name}")))
    }

    /// List all registered backend names.
    pub fn list_names(&self) -> Vec<&str> {
        self.backends.iter().map(|b| b.name()).collect()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a `BackendRegistry` with all available backends pre-registered.
///
/// Backends are registered in priority order: mistral.rs first (pure Rust),
/// then llama.cpp. When both features are enabled, both backends are available
/// and `find_for_format` returns the first match (mistral.rs).
/// Use `find_by_name("llama.cpp")` to explicitly select the llama.cpp backend.
pub fn default_backends(#[allow(unused)] config: Arc<PowerConfig>) -> BackendRegistry {
    #[allow(unused_mut)]
    let mut registry = BackendRegistry::new();

    // Register mistral.rs backend first (pure Rust, higher priority)
    #[cfg(feature = "mistralrs")]
    registry.register(Arc::new(mistralrs_backend::MistralRsBackend::new(
        config.clone(),
    )));

    // Register llama.cpp backend (available as fallback or explicit selection)
    #[cfg(feature = "llamacpp")]
    registry.register(Arc::new(llamacpp::LlamaCppBackend::new(config.clone())));

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Arc<PowerConfig> {
        Arc::new(PowerConfig::default())
    }

    #[cfg(any(feature = "mistralrs", feature = "llamacpp"))]
    #[test]
    fn test_backend_registry_find() {
        let registry = default_backends(test_config());
        let backend = registry.find_for_format(&ModelFormat::Gguf);
        assert!(backend.is_ok());
        let backend = backend.unwrap();
        let name = backend.name();
        assert!(name == "mistral.rs" || name == "llama.cpp");
    }

    #[cfg(feature = "mistralrs")]
    #[test]
    fn test_mistralrs_has_priority() {
        // When mistralrs is enabled, it should be the first match for GGUF
        let registry = default_backends(test_config());
        let backend = registry.find_for_format(&ModelFormat::Gguf).unwrap();
        assert_eq!(backend.name(), "mistral.rs");
    }

    #[cfg(any(feature = "mistralrs", feature = "llamacpp"))]
    #[test]
    fn test_backend_registry_list() {
        let registry = default_backends(test_config());
        let names = registry.list_names();
        assert!(!names.is_empty());
    }

    #[cfg(all(feature = "mistralrs", feature = "llamacpp"))]
    #[test]
    fn test_both_backends_registered() {
        let registry = default_backends(test_config());
        let names = registry.list_names();
        assert!(names.contains(&"mistral.rs"));
        assert!(names.contains(&"llama.cpp"));
        assert_eq!(names.len(), 2);
    }

    #[cfg(all(feature = "mistralrs", feature = "llamacpp"))]
    #[test]
    fn test_find_by_name_selects_specific_backend() {
        let registry = default_backends(test_config());
        let llama = registry.find_by_name("llama.cpp");
        assert!(llama.is_ok());
        assert_eq!(llama.unwrap().name(), "llama.cpp");

        let mistral = registry.find_by_name("mistral.rs");
        assert!(mistral.is_ok());
        assert_eq!(mistral.unwrap().name(), "mistral.rs");
    }

    #[test]
    fn test_find_by_name_not_found() {
        let registry = BackendRegistry::new();
        let result = registry.find_by_name("nonexistent");
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("No backend with name")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[cfg(feature = "mistralrs")]
    #[test]
    fn test_mistralrs_backend_supports() {
        let backend = mistralrs_backend::MistralRsBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::Gguf));
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_llamacpp_backend_supports() {
        let backend = llamacpp::LlamaCppBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::Gguf));
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_find_for_format_unsupported() {
        let registry = BackendRegistry::new();
        let result = registry.find_for_format(&ModelFormat::SafeTensors);
        assert!(result.is_err());
        match result {
            Err(e) => assert!(e.to_string().contains("No backend available")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn test_backend_registry_default() {
        let registry = BackendRegistry::default();
        assert!(registry.list_names().is_empty());
    }

    #[cfg(any(feature = "mistralrs", feature = "llamacpp"))]
    #[test]
    fn test_registry_priority_order() {
        // First registered backend wins for find_for_format
        let mut registry = BackendRegistry::new();
        let config = test_config();
        #[cfg(feature = "mistralrs")]
        {
            registry.register(Arc::new(mistralrs_backend::MistralRsBackend::new(
                config.clone(),
            )));
        }
        #[cfg(feature = "llamacpp")]
        {
            registry.register(Arc::new(llamacpp::LlamaCppBackend::new(config.clone())));
        }
        let first = registry.find_for_format(&ModelFormat::Gguf).unwrap();
        // First registered backend should be returned
        assert_eq!(first.name(), registry.list_names()[0]);
    }
}
