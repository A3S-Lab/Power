pub mod chat_template;
pub mod gguf_stream;
pub mod gpu;
pub mod json_schema;
pub mod llamacpp;
pub mod mistralrs_backend;
pub mod picolm;
pub mod picolm_ops;
pub mod proxy;
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
use crate::tee::encrypted_model::{LayerStreamingDecryptedModel, MemoryDecryptedModel};

use types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk,
    EffectivePromptDigest, EmbeddingRequest, EmbeddingResponse,
};

/// Trait for inference backends that can load models and run inference.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Human-readable name of this backend.
    fn name(&self) -> &str;

    /// Check if this backend can serve the given model format.
    fn supports(&self, format: &ModelFormat) -> bool;

    /// Check whether this backend can serve one exact model manifest.
    ///
    /// Format-only backends inherit the existing behavior. Architecture-aware
    /// backends should override this method so multiple implementations of the
    /// same container format can coexist without relying on raw backend names.
    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        self.supports(&manifest.format)
    }

    /// Load a model into memory, ready for inference.
    async fn load(&self, manifest: &ModelManifest) -> Result<()>;

    /// Return true when this backend can load a decrypted model without writing
    /// plaintext back to disk.
    fn supports_memory_load(&self, _format: &ModelFormat) -> bool {
        false
    }

    /// Load a model from locked plaintext memory.
    ///
    /// Backends must take ownership of `plaintext` or otherwise keep the bytes
    /// alive for as long as inference can access them. The default fails closed
    /// so encrypted in-memory loading is never silently downgraded to a file path.
    async fn load_from_memory(
        &self,
        manifest: &ModelManifest,
        plaintext: MemoryDecryptedModel,
    ) -> Result<()> {
        drop(plaintext);
        Err(PowerError::BackendNotAvailable(format!(
            "backend '{}' does not support locked in-memory model loading for format {} \
             (model '{}')",
            self.name(),
            manifest.format,
            manifest.name
        )))
    }

    /// Return true when this backend can load a decrypted model from the
    /// layer-streaming plaintext source without writing plaintext back to disk.
    fn supports_streaming_decrypt_load(&self, _format: &ModelFormat) -> bool {
        false
    }

    /// Load a model from the layer-streaming decrypted plaintext source.
    ///
    /// Backends must take ownership of `plaintext` or otherwise keep the bytes
    /// alive for as long as inference can access them. The default fails closed
    /// so encrypted streaming decrypt loading is never silently downgraded to a
    /// file path.
    async fn load_from_streaming_decrypt(
        &self,
        manifest: &ModelManifest,
        plaintext: LayerStreamingDecryptedModel,
    ) -> Result<()> {
        drop(plaintext);
        Err(PowerError::BackendNotAvailable(format!(
            "backend '{}' does not support layer-streaming decrypted model loading for format {} \
             (model '{}')",
            self.name(),
            manifest.format,
            manifest.name
        )))
    }

    /// Unload a model from memory.
    async fn unload(&self, model_name: &str) -> Result<()>;

    /// Run chat completion inference, returning a stream of token chunks.
    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>>;

    /// Return the digest of the exact rendered chat prompt used for inference.
    ///
    /// Backends should return `None` unless they locally render the prompt and can
    /// guarantee the same bytes are passed to the model. This keeps receipts from
    /// overclaiming for delegated renderers such as upstream proxy servers.
    async fn effective_chat_prompt_digest(
        &self,
        _model_name: &str,
        _request: &ChatRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        Ok(None)
    }

    /// Run text completion inference, returning a stream of token chunks.
    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>>;

    /// Return the digest of the exact text-completion prompt used for inference.
    ///
    /// Backends should return `None` unless they can guarantee the same prompt
    /// bytes or token IDs are passed to the model. Delegating/proxy backends
    /// should stay absent unless the upstream exposes an explicit digest.
    async fn effective_completion_prompt_digest(
        &self,
        _model_name: &str,
        _request: &CompletionRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        Ok(None)
    }

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

    /// Pull (download) a model by name from a remote registry.
    ///
    /// Default: not supported. Backends that can fetch models should override this.
    async fn pull(&self, name: &str) -> Result<ModelManifest> {
        Err(PowerError::BackendNotAvailable(format!(
            "backend '{}' does not support model pull (requested: '{name}')",
            self.name()
        )))
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

    /// Find the highest-priority backend that supports the exact manifest.
    pub fn find_for_manifest(&self, manifest: &ModelManifest) -> Result<Arc<dyn Backend>> {
        self.backends
            .iter()
            .find(|backend| backend.supports_manifest(manifest))
            .cloned()
            .ok_or_else(|| {
                PowerError::BackendNotAvailable(format!(
                    "No backend available for model '{}' with format {}",
                    manifest.name, manifest.format
                ))
            })
    }

    /// Append another registry while preserving both registries' priority order.
    pub fn extend(&mut self, other: BackendRegistry) {
        self.backends.extend(other.backends);
    }

    /// Find a backend for TEE-constrained inference.
    ///
    /// When `model_size_bytes` exceeds 75% of available EPC memory, prefers the
    /// picolm backend (layer-streaming, O(layer_size) RAM) over mistralrs/llamacpp
    /// (which load the full model into RAM). Falls back to normal priority order
    /// when EPC info is unavailable or the model fits comfortably.
    pub fn find_for_tee(
        &self,
        format: &ModelFormat,
        model_size_bytes: u64,
    ) -> Result<Arc<dyn Backend>> {
        use crate::tee::epc::model_exceeds_epc;

        if model_exceeds_epc(model_size_bytes) {
            // Prefer picolm for layer-streaming when model won't fit in EPC
            if let Ok(b) = self.find_by_name("picolm") {
                if b.supports(format) {
                    tracing::info!(
                        model_size_mb = model_size_bytes / 1_048_576,
                        "TEE EPC pressure: routing to picolm layer-streaming backend"
                    );
                    return Ok(b);
                }
            }
            // picolm not available — warn and fall through to normal selection
            tracing::warn!(
                model_size_mb = model_size_bytes / 1_048_576,
                "Model may exceed TEE EPC budget. Enable --features picolm for layer-streaming."
            );
        }

        self.find_for_format(format)
    }

    /// Find a backend for one exact manifest under TEE memory pressure.
    ///
    /// The EPC-aware picolm preference is retained, but the candidate must also
    /// accept the complete manifest. This prevents a format-compatible backend
    /// from intercepting a model family it cannot execute.
    pub fn find_for_tee_manifest(&self, manifest: &ModelManifest) -> Result<Arc<dyn Backend>> {
        use crate::tee::epc::model_exceeds_epc;

        if model_exceeds_epc(manifest.size) {
            if let Ok(backend) = self.find_by_name("picolm") {
                if backend.supports_manifest(manifest) {
                    tracing::info!(
                        model_size_mb = manifest.size / 1_048_576,
                        "TEE EPC pressure: routing to picolm layer-streaming backend"
                    );
                    return Ok(backend);
                }
            }
            tracing::warn!(
                model_size_mb = manifest.size / 1_048_576,
                "Model may exceed TEE EPC budget. Enable --features picolm for layer-streaming."
            );
        }

        self.find_for_manifest(manifest)
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

    // Register picolm backend (pure Rust, layer-streaming, TEE memory-efficient)
    // Lower priority than mistralrs/llamacpp for normal use; selected automatically
    // by memory-aware routing when model_size > EPC * 0.75.
    #[cfg(feature = "picolm")]
    registry.register(Arc::new(picolm::PicolmBackend::new(config.clone())));

    // Register the proxy backend when any upstream is configured. Serves
    // `ModelFormat::Remote` models by forwarding to an OpenAI-compatible server.
    if !config.proxy_upstreams.is_empty() {
        registry.register(Arc::new(proxy::ProxyBackend::new(config.clone())));
    }

    registry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, MockBackend};

    #[cfg(any(feature = "mistralrs", feature = "llamacpp"))]
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
        #[cfg(feature = "picolm")]
        assert!(names.contains(&"picolm"));
        assert_eq!(names.len(), 2 + usize::from(cfg!(feature = "picolm")));
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
        assert!(backend.supports(&ModelFormat::SafeTensors));
        assert!(backend.supports(&ModelFormat::HuggingFace));
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

    #[test]
    fn manifest_matching_skips_an_incompatible_specialized_backend() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::success()
                .with_name("olmoe")
                .with_family("olmoe"),
        ));
        registry.register(Arc::new(MockBackend::success().with_name("generic")));

        let mut manifest = sample_manifest("other-model");
        manifest.family = Some("other".to_string());

        let backend = registry.find_for_manifest(&manifest).unwrap();
        assert_eq!(backend.name(), "generic");
    }

    #[test]
    fn manifest_matching_prefers_a_compatible_specialized_backend() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::success()
                .with_name("olmoe")
                .with_family("olmoe"),
        ));
        registry.register(Arc::new(MockBackend::success().with_name("generic")));

        let mut manifest = sample_manifest("olmoe-model");
        manifest.family = Some("olmoe".to_string());

        let backend = registry.find_for_manifest(&manifest).unwrap();
        assert_eq!(backend.name(), "olmoe");
    }

    #[test]
    fn tee_manifest_matching_preserves_manifest_filtering() {
        let mut registry = BackendRegistry::new();
        registry.register(Arc::new(
            MockBackend::success()
                .with_name("olmoe")
                .with_family("olmoe"),
        ));
        registry.register(Arc::new(MockBackend::success().with_name("generic")));

        let mut manifest = sample_manifest("other-model");
        manifest.family = Some("other".to_string());
        manifest.size = 0;

        let backend = registry.find_for_tee_manifest(&manifest).unwrap();
        assert_eq!(backend.name(), "generic");
    }

    #[test]
    fn extending_a_registry_preserves_custom_backend_priority() {
        let mut custom = BackendRegistry::new();
        custom.register(Arc::new(MockBackend::success().with_name("custom")));
        let mut defaults = BackendRegistry::new();
        defaults.register(Arc::new(MockBackend::success().with_name("default")));

        custom.extend(defaults);

        assert_eq!(custom.list_names(), ["custom", "default"]);
        assert_eq!(
            custom.find_for_format(&ModelFormat::Gguf).unwrap().name(),
            "custom"
        );
    }
}
