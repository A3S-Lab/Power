// llama.cpp backend implementation.
//
// When the `llamacpp` feature is enabled, this uses `llama-cpp-2` Rust bindings
// to load GGUF models and run inference (chat, completion, embeddings).
// Without the feature, it returns `BackendNotAvailable` errors.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
#[cfg(feature = "llamacpp")]
use std::collections::HashMap;
#[cfg(feature = "llamacpp")]
use tokio::sync::RwLock;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

#[cfg(feature = "llamacpp")]
use super::chat_template::{self, ChatTemplateKind};
use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use super::Backend;

/// Whether a model was loaded for inference or embedding.
#[cfg(feature = "llamacpp")]
#[derive(Debug, Clone, Copy, PartialEq)]
enum LoadMode {
    Inference,
    Embedding,
}

/// Tracks a loaded model's path, name, template, and the loaded LlamaModel handle.
#[cfg(feature = "llamacpp")]
struct LoadedModel {
    #[allow(dead_code)]
    name: String,
    path: std::path::PathBuf,
    model: Arc<llama_cpp_2::model::LlamaModel>,
    chat_template: ChatTemplateKind,
    load_mode: LoadMode,
}

/// llama.cpp backend for GGUF model inference.
pub struct LlamaCppBackend {
    #[cfg(feature = "llamacpp")]
    models: RwLock<HashMap<String, LoadedModel>>,
    #[cfg(feature = "llamacpp")]
    llama_backend: std::sync::OnceLock<llama_cpp_2::llama_backend::LlamaBackend>,
    #[allow(dead_code)]
    config: Arc<PowerConfig>,
}

impl LlamaCppBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            #[cfg(feature = "llamacpp")]
            models: RwLock::new(HashMap::new()),
            #[cfg(feature = "llamacpp")]
            llama_backend: std::sync::OnceLock::new(),
            config,
        }
    }
}

// ============================================================================
// Feature-gated implementation using llama-cpp-2
// ============================================================================

#[cfg(feature = "llamacpp")]
#[async_trait]
impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        use llama_cpp_2::llama_backend::LlamaBackend;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::LlamaModel;

        tracing::info!(model = %manifest.name, path = %manifest.path.display(), "Loading model");

        let backend = self
            .llama_backend
            .get_or_init(|| LlamaBackend::init().expect("Failed to initialize llama.cpp backend"));

        let gpu_layers = self.config.gpu.gpu_layers;
        let params = if gpu_layers != 0 {
            LlamaModelParams::default().with_n_gpu_layers(gpu_layers)
        } else {
            LlamaModelParams::default()
        };

        let path = manifest.path.clone();
        let model_name = manifest.name.clone();

        // Load model in a blocking task since it's CPU-intensive
        let model = tokio::task::spawn_blocking(move || {
            LlamaModel::load_from_file(backend, &path, &params)
                .map_err(|e| PowerError::InferenceFailed(format!("Failed to load model: {e}")))
        })
        .await
        .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

        let model_arc = Arc::new(model);

        // Detect chat template from GGUF metadata
        let chat_template = model_arc
            .metadata()
            .and_then(|meta| meta.get("tokenizer.chat_template"))
            .and_then(|v| v.as_str())
            .map(chat_template::detect)
            .unwrap_or(ChatTemplateKind::Phi);

        self.models.write().await.insert(
            model_name.clone(),
            LoadedModel {
                name: model_name.clone(),
                path: manifest.path.clone(),
                model: model_arc,
                chat_template,
                load_mode: LoadMode::Inference,
            },
        );

        tracing::info!(model = %manifest.name, "Model loaded successfully");
        Ok(())
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        if self.models.write().await.remove(model_name).is_some() {
            tracing::info!(model = model_name, "Model unloaded");
        }
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        // Look up the chat template for this model
        let template = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| m.chat_template.clone())
                .unwrap_or(ChatTemplateKind::Phi)
        };

        let prompt = chat_template::format_chat_prompt(&request.messages, &template);

        let completion_req = CompletionRequest {
            prompt,
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stop: request.stop,
            stream: request.stream,
        };

        let stream = self.complete(model_name, completion_req).await?;

        // Map CompletionResponseChunk -> ChatResponseChunk
        use futures::StreamExt;
        let chat_stream = stream.map(|chunk_result| {
            chunk_result.map(|chunk| ChatResponseChunk {
                content: chunk.text,
                done: chunk.done,
            })
        });

        Ok(Box::pin(chat_stream))
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::context::LlamaContext;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;
        use llama_cpp_2::token::LlamaToken;

        let model_arc = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| m.model.clone())
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
                })?
        };

        let max_tokens = request.max_tokens.unwrap_or(512) as usize;
        let temperature = request.temperature.unwrap_or(0.8);
        let top_p = request.top_p.unwrap_or(0.95);

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<CompletionResponseChunk>>(32);

        // Run inference in a blocking task
        tokio::task::spawn_blocking(move || {
            let ctx_params =
                LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(2048).unwrap());
            let mut ctx = match LlamaContext::with_model(&model_arc, ctx_params) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                        "Failed to create context: {e}"
                    ))));
                    return;
                }
            };

            // Tokenize the prompt
            let tokens =
                match model_arc.str_to_token(&request.prompt, llama_cpp_2::model::AddBos::Always) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                            "Tokenization failed: {e}"
                        ))));
                        return;
                    }
                };

            // Create batch and add prompt tokens
            let mut batch = LlamaBatch::new(2048, 1);
            for (i, &token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                if batch.add(token, i as i32, &[0], is_last).is_err() {
                    let _ = tx.blocking_send(Err(PowerError::InferenceFailed(
                        "Failed to add token to batch".to_string(),
                    )));
                    return;
                }
            }

            // Decode prompt
            if let Err(e) = ctx.decode(&mut batch) {
                let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                    "Decode failed: {e}"
                ))));
                return;
            }

            // Set up sampler
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(temperature),
                LlamaSampler::top_p(top_p, 1),
                LlamaSampler::dist(0),
            ]);

            let mut n_cur = tokens.len();
            let eos_token = model_arc.token_eos();

            // Generate tokens
            for _ in 0..max_tokens {
                let new_token = sampler.sample(&ctx, -1);

                if new_token == eos_token {
                    let _ = tx.blocking_send(Ok(CompletionResponseChunk {
                        text: String::new(),
                        done: true,
                    }));
                    return;
                }

                let text = model_arc
                    .token_to_str(new_token, llama_cpp_2::token::data::LlamaTokenAttr::all())
                    .unwrap_or_default()
                    .to_string();

                if tx
                    .blocking_send(Ok(CompletionResponseChunk { text, done: false }))
                    .is_err()
                {
                    return; // receiver dropped
                }

                // Prepare next batch
                batch.clear();
                if batch.add(new_token, n_cur as i32, &[0], true).is_err() {
                    return;
                }
                n_cur += 1;

                if let Err(e) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                        "Decode failed: {e}"
                    ))));
                    return;
                }
            }

            // Max tokens reached
            let _ = tx.blocking_send(Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
            }));
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn embed(
        &self,
        model_name: &str,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::context::LlamaContext;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::LlamaModel;

        // Check if model needs to be reloaded with embedding mode
        let needs_reload = {
            let models = self.models.read().await;
            match models.get(model_name) {
                Some(m) => m.load_mode != LoadMode::Embedding,
                None => {
                    return Err(PowerError::InferenceFailed(format!(
                        "Model '{model_name}' not loaded"
                    )));
                }
            }
        };

        if needs_reload {
            let (path, chat_template) = {
                let models = self.models.read().await;
                let m = models.get(model_name).unwrap();
                (m.path.clone(), m.chat_template.clone())
            };

            tracing::info!(model = model_name, "Reloading model with embedding mode");

            let backend = self.llama_backend.get().ok_or_else(|| {
                PowerError::InferenceFailed("llama.cpp backend not initialized".to_string())
            })?;

            let gpu_layers = self.config.gpu.gpu_layers;
            let params = if gpu_layers != 0 {
                LlamaModelParams::default()
                    .with_n_gpu_layers(gpu_layers)
                    .with_embedding(true)
            } else {
                LlamaModelParams::default().with_embedding(true)
            };

            let path_clone = path.clone();
            let model = tokio::task::spawn_blocking(move || {
                LlamaModel::load_from_file(backend, &path_clone, &params).map_err(|e| {
                    PowerError::InferenceFailed(format!(
                        "Failed to reload model for embedding: {e}"
                    ))
                })
            })
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

            let model_arc = Arc::new(model);
            let name = model_name.to_string();
            self.models.write().await.insert(
                name.clone(),
                LoadedModel {
                    name,
                    path,
                    model: model_arc,
                    chat_template,
                    load_mode: LoadMode::Embedding,
                },
            );
        }

        let model_arc = {
            let models = self.models.read().await;
            models.get(model_name).unwrap().model.clone()
        };

        let input = request.input.clone();

        tokio::task::spawn_blocking(move || {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(2048))
                .with_embeddings(true);
            let mut ctx = LlamaContext::with_model(&model_arc, ctx_params).map_err(|e| {
                PowerError::InferenceFailed(format!("Failed to create context: {e}"))
            })?;

            let mut embeddings = Vec::with_capacity(input.len());

            for text in &input {
                let tokens = model_arc
                    .str_to_token(text, llama_cpp_2::model::AddBos::Always)
                    .map_err(|e| {
                        PowerError::InferenceFailed(format!("Tokenization failed: {e}"))
                    })?;

                let mut batch = LlamaBatch::new(2048, 1);
                for (i, &token) in tokens.iter().enumerate() {
                    let is_last = i == tokens.len() - 1;
                    batch.add(token, i as i32, &[0], is_last).map_err(|_| {
                        PowerError::InferenceFailed("Failed to add token to batch".to_string())
                    })?;
                }

                ctx.decode(&mut batch)
                    .map_err(|e| PowerError::InferenceFailed(format!("Decode failed: {e}")))?;

                let emb = ctx.embeddings_seq_ith(0).map_err(|e| {
                    PowerError::InferenceFailed(format!("Failed to get embeddings: {e}"))
                })?;
                embeddings.push(emb.to_vec());

                ctx.clear_kv_cache();
            }

            Ok(EmbeddingResponse { embeddings })
        })
        .await
        .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))?
    }
}

// ============================================================================
// Stub implementation when llamacpp feature is disabled
// ============================================================================

#[cfg(not(feature = "llamacpp"))]
#[async_trait]
impl Backend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        tracing::warn!(
            model = %manifest.name,
            "llama.cpp backend compiled without `llamacpp` feature"
        );
        Err(PowerError::BackendNotAvailable(
            "llama.cpp backend requires the `llamacpp` feature flag. \
             Rebuild with: cargo build --features llamacpp"
                .to_string(),
        ))
    }

    async fn unload(&self, _model_name: &str) -> Result<()> {
        Ok(())
    }

    async fn chat(
        &self,
        _model_name: &str,
        _request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        Err(PowerError::BackendNotAvailable(
            "llama.cpp backend requires the `llamacpp` feature flag".to_string(),
        ))
    }

    async fn complete(
        &self,
        _model_name: &str,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        Err(PowerError::BackendNotAvailable(
            "llama.cpp backend requires the `llamacpp` feature flag".to_string(),
        ))
    }

    async fn embed(
        &self,
        _model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(
            "llama.cpp backend requires the `llamacpp` feature flag".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::model::manifest::ModelFormat;

    fn test_config() -> Arc<PowerConfig> {
        Arc::new(PowerConfig::default())
    }

    #[test]
    fn test_new_creates_backend() {
        let backend = LlamaCppBackend::new(test_config());
        assert_eq!(backend.name(), "llama.cpp");
    }

    #[test]
    fn test_supports_gguf() {
        let backend = LlamaCppBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::Gguf));
    }

    #[test]
    fn test_does_not_support_safetensors() {
        let backend = LlamaCppBackend::new(test_config());
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_backend_stores_config() {
        let mut config = PowerConfig::default();
        config.gpu.gpu_layers = -1;
        let config = Arc::new(config);
        let backend = LlamaCppBackend::new(config.clone());
        assert_eq!(backend.config.gpu.gpu_layers, -1);
    }

    #[cfg(not(feature = "llamacpp"))]
    #[tokio::test]
    async fn test_stub_load_returns_error() {
        use crate::model::manifest::ModelManifest;
        use std::path::PathBuf;

        let backend = LlamaCppBackend::new(test_config());
        let manifest = ModelManifest {
            name: "test".to_string(),
            format: ModelFormat::Gguf,
            size: 0,
            sha256: "abc".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/test"),
        };
        let result = backend.load(&manifest).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("llamacpp"));
    }

    #[cfg(not(feature = "llamacpp"))]
    #[tokio::test]
    async fn test_stub_chat_returns_error() {
        let backend = LlamaCppBackend::new(test_config());
        let request = ChatRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: false,
        };
        let result = backend.chat("test", request).await;
        assert!(result.is_err());
    }

    #[cfg(not(feature = "llamacpp"))]
    #[tokio::test]
    async fn test_stub_complete_returns_error() {
        let backend = LlamaCppBackend::new(test_config());
        let request = CompletionRequest {
            prompt: "test".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: false,
        };
        let result = backend.complete("test", request).await;
        assert!(result.is_err());
    }

    #[cfg(not(feature = "llamacpp"))]
    #[tokio::test]
    async fn test_stub_unload_succeeds() {
        let backend = LlamaCppBackend::new(test_config());
        let result = backend.unload("test").await;
        assert!(result.is_ok());
    }

    #[cfg(not(feature = "llamacpp"))]
    #[tokio::test]
    async fn test_stub_embed_returns_error() {
        let backend = LlamaCppBackend::new(test_config());
        let request = EmbeddingRequest {
            input: vec!["test".to_string()],
        };
        let result = backend.embed("test", request).await;
        assert!(result.is_err());
    }
}
