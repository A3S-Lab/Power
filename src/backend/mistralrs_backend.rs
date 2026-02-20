// mistral.rs backend implementation.
//
// When the `mistralrs` feature is enabled, this uses the pure Rust `mistralrs`
// crate (built on candle) to load GGUF models and run inference (chat,
// completion, embeddings). Without the feature, it returns `BackendNotAvailable`
// errors.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use super::Backend;

/// mistral.rs backend for GGUF model inference (pure Rust, no C++ dependency).
pub struct MistralRsBackend {
    #[cfg(feature = "mistralrs")]
    models: tokio::sync::RwLock<std::collections::HashMap<String, LoadedModel>>,
    #[allow(dead_code)]
    config: Arc<PowerConfig>,
}

#[cfg(feature = "mistralrs")]
struct LoadedModel {
    /// The mistralrs Model handle for inference (wrapped in Arc since Model is not Clone).
    model: Arc<mistralrs::Model>,
    /// Raw Jinja2 template string from GGUF metadata (for chat template rendering).
    #[allow(dead_code)]
    raw_template: Option<String>,
}

impl MistralRsBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            #[cfg(feature = "mistralrs")]
            models: tokio::sync::RwLock::new(std::collections::HashMap::new()),
            config,
        }
    }
}

// ============================================================================
// Feature-gated implementation using mistralrs (pure Rust)
// ============================================================================

#[cfg(feature = "mistralrs")]
#[async_trait]
impl Backend for MistralRsBackend {
    fn name(&self) -> &str {
        "mistral.rs"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        tracing::info!(model = %manifest.name, path = %manifest.path.display(), "Loading model via mistral.rs");

        // Extract the directory and filename from the manifest path
        let model_dir = manifest
            .path
            .parent()
            .ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "Invalid model path (no parent dir): {}",
                    manifest.path.display()
                ))
            })?
            .to_string_lossy()
            .to_string();

        let model_filename = manifest
            .path
            .file_name()
            .ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "Invalid model path (no filename): {}",
                    manifest.path.display()
                ))
            })?
            .to_string_lossy()
            .to_string();

        // Build the GGUF model using mistralrs
        let mut builder = mistralrs::GgufModelBuilder::new(model_dir, vec![model_filename]);

        // Warn about unsupported config fields for this backend
        if self.config.num_parallel > 1 {
            tracing::warn!(
                num_parallel = self.config.num_parallel,
                "num_parallel > 1 is not supported by the mistral.rs backend; ignored"
            );
        }
        if self.config.num_thread.is_some() {
            tracing::debug!("num_thread config is not applied at mistral.rs load time");
        }

        // Apply chat template override if available
        if let Some(ref template) = manifest.template_override {
            builder = builder.with_chat_template(template.clone());
        }

        // Force CPU if no GPU layers configured
        if self.config.gpu.gpu_layers == 0 {
            builder = builder.with_force_cpu();
        }

        let model = builder.build().await.map_err(|e| {
            PowerError::InferenceFailed(format!("Failed to load model via mistral.rs: {e}"))
        })?;

        let raw_template = manifest.template_override.clone();

        self.models.write().await.insert(
            manifest.name.clone(),
            LoadedModel {
                model: Arc::new(model),
                raw_template,
            },
        );

        tracing::info!(model = %manifest.name, "Model loaded successfully via mistral.rs");
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
        use mistralrs::{RequestBuilder, TextMessageRole};

        let model = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| Arc::clone(&m.model))
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
                })?
        };

        // Build the request with messages and sampling parameters
        let mut req_builder = RequestBuilder::new();

        if let Some(temp) = request.temperature {
            req_builder = req_builder.set_sampler_temperature(temp as f64);
        }
        if let Some(top_p) = request.top_p {
            req_builder = req_builder.set_sampler_topp(top_p as f64);
        }
        if let Some(top_k) = request.top_k {
            req_builder = req_builder.set_sampler_topk(top_k as usize);
        }
        if let Some(min_p) = request.min_p {
            req_builder = req_builder.set_sampler_minp(min_p as f64);
        }
        if let Some(max_tokens) = request.max_tokens {
            req_builder = req_builder.set_sampler_max_len(max_tokens as usize);
        }
        if let Some(freq_pen) = request.frequency_penalty {
            req_builder = req_builder.set_sampler_frequency_penalty(freq_pen);
        }
        if let Some(pres_pen) = request.presence_penalty {
            req_builder = req_builder.set_sampler_presence_penalty(pres_pen);
        }
        if let Some(ref stops) = request.stop {
            if !stops.is_empty() {
                req_builder =
                    req_builder.set_sampler_stop_toks(mistralrs::StopTokens::Seqs(stops.clone()));
            }
        }

        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "system" => TextMessageRole::System,
                "user" => TextMessageRole::User,
                "assistant" => TextMessageRole::Assistant,
                "tool" => TextMessageRole::Tool,
                _ => TextMessageRole::User,
            };
            req_builder = req_builder.add_message(role, msg.content.text());
        }

        let has_tools = request.tools.is_some();

        // Use non-streaming API and convert to a stream.
        // mistralrs's Stream<'_> borrows &Model which prevents moving into
        // tokio::spawn. We use send_chat_request (non-streaming) and emit
        // the full response as chunks through a channel.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ChatResponseChunk>>(32);

        tokio::spawn(async move {
            let response = model.send_chat_request(req_builder).await;

            match response {
                Ok(resp) => {
                    for choice in &resp.choices {
                        let raw_content = choice.message.content.as_deref().unwrap_or("");

                        // Parse think blocks from the full response
                        let mut think_parser = super::think_parser::ThinkBlockParser::new();
                        let (content_part, thinking_part) = think_parser.feed(raw_content);
                        let (flush_c, flush_t) = think_parser.flush();
                        let content = content_part + &flush_c;
                        let thinking = thinking_part + &flush_t;

                        let thinking_content = if thinking.is_empty() {
                            None
                        } else {
                            Some(thinking)
                        };

                        // Parse tool calls from the full response text
                        let tool_calls = if has_tools {
                            super::tool_parser::parse_tool_calls(raw_content)
                        } else {
                            None
                        };

                        let done_reason = if tool_calls.is_some() {
                            Some("tool_calls".to_string())
                        } else {
                            Some(choice.finish_reason.clone())
                        };

                        let chat_chunk = ChatResponseChunk {
                            content,
                            thinking_content,
                            done: true,
                            prompt_tokens: Some(resp.usage.prompt_tokens as u32),
                            done_reason,
                            prompt_eval_duration_ns: None,
                            tool_calls,
                        };

                        let _ = tx.send(Ok(chat_chunk)).await;
                    }

                    // If no choices, send an empty done chunk
                    if resp.choices.is_empty() {
                        let _ = tx
                            .send(Ok(ChatResponseChunk {
                                content: String::new(),
                                thinking_content: None,
                                done: true,
                                prompt_tokens: Some(resp.usage.prompt_tokens as u32),
                                done_reason: Some("stop".to_string()),
                                prompt_eval_duration_ns: None,
                                tool_calls: None,
                            }))
                            .await;
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Err(PowerError::InferenceFailed(format!(
                            "mistral.rs inference failed: {e}"
                        ))))
                        .await;
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        // Build a chat request from the completion request and delegate
        // mistralrs uses chat-based API; we wrap the prompt as a user message
        let chat_request = ChatRequest {
            messages: vec![super::types::ChatMessage {
                role: "user".to_string(),
                content: super::types::MessageContent::Text(request.prompt.clone()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stop: request.stop.clone(),
            stream: request.stream,
            top_k: request.top_k,
            min_p: request.min_p,
            repeat_penalty: request.repeat_penalty,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            seed: request.seed,
            num_ctx: request.num_ctx,
            mirostat: request.mirostat,
            mirostat_tau: request.mirostat_tau,
            mirostat_eta: request.mirostat_eta,
            tfs_z: request.tfs_z,
            typical_p: request.typical_p,
            response_format: request.response_format.clone(),
            tools: None,
            tool_choice: None,
            repeat_last_n: request.repeat_last_n,
            penalize_newline: request.penalize_newline,
            num_batch: request.num_batch,
            num_thread: request.num_thread,
            num_thread_batch: request.num_thread_batch,
            flash_attention: request.flash_attention,
            num_gpu: request.num_gpu,
            main_gpu: request.main_gpu,
            use_mmap: request.use_mmap,
            use_mlock: request.use_mlock,
            num_parallel: request.num_parallel,
        };

        // Get the chat stream and map ChatResponseChunk -> CompletionResponseChunk
        let chat_stream = self.chat(model_name, chat_request).await?;

        use futures::StreamExt;
        let completion_stream = chat_stream.map(|chunk_result| {
            chunk_result.map(|chat_chunk| CompletionResponseChunk {
                text: chat_chunk.content,
                done: chat_chunk.done,
                prompt_tokens: chat_chunk.prompt_tokens,
                done_reason: chat_chunk.done_reason,
                prompt_eval_duration_ns: chat_chunk.prompt_eval_duration_ns,
                token_id: None,
            })
        });

        Ok(Box::pin(completion_stream))
    }

    async fn embed(
        &self,
        model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        // For embeddings, mistralrs uses a separate EmbeddingModelBuilder.
        // Since our Backend trait loads models via `load()` which uses GgufModelBuilder,
        // we need to send an embedding request through the loaded model's runner.
        // However, mistralrs's high-level Model API doesn't directly expose embedding
        // for GGUF text models. We'll use the lower-level request API.
        let _model = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| Arc::clone(&m.model))
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
                })?
        };

        // For now, embeddings through GGUF text models are not directly supported
        // by mistralrs's high-level API. Users should use a dedicated embedding model.
        // This matches the reality that most GGUF models are text generation models.
        Err(PowerError::InferenceFailed(
            "Embedding generation via GGUF models is not yet supported with the mistral.rs backend. \
             Use a dedicated embedding model or enable the `llamacpp` feature for embedding support."
                .to_string(),
        ))
    }
}

// ============================================================================
// Stub implementation when mistralrs feature is disabled
// ============================================================================

#[cfg(not(feature = "mistralrs"))]
#[async_trait]
impl Backend for MistralRsBackend {
    fn name(&self) -> &str {
        "mistral.rs"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        tracing::warn!(
            model = %manifest.name,
            "mistral.rs backend compiled without `mistralrs` feature"
        );
        Err(PowerError::BackendNotAvailable(
            "mistral.rs backend requires the `mistralrs` feature flag. \
             Rebuild with: cargo build --features mistralrs"
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
            "mistral.rs backend requires the `mistralrs` feature flag".to_string(),
        ))
    }

    async fn complete(
        &self,
        _model_name: &str,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        Err(PowerError::BackendNotAvailable(
            "mistral.rs backend requires the `mistralrs` feature flag".to_string(),
        ))
    }

    async fn embed(
        &self,
        _model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(
            "mistral.rs backend requires the `mistralrs` feature flag".to_string(),
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

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
        let backend = MistralRsBackend::new(test_config());
        assert_eq!(backend.name(), "mistral.rs");
    }

    #[test]
    fn test_supports_gguf() {
        let backend = MistralRsBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::Gguf));
    }

    #[test]
    fn test_does_not_support_safetensors() {
        let backend = MistralRsBackend::new(test_config());
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_backend_stores_config() {
        let mut config = PowerConfig::default();
        config.gpu.gpu_layers = -1;
        let config = Arc::new(config);
        let backend = MistralRsBackend::new(config.clone());
        assert_eq!(backend.config.gpu.gpu_layers, -1);
    }

    #[test]
    fn test_backend_name() {
        let backend = MistralRsBackend::new(test_config());
        assert_eq!(backend.name(), "mistral.rs");
    }

    #[test]
    fn test_backend_does_not_support_unknown_format() {
        let backend = MistralRsBackend::new(test_config());
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_backend_config_gpu_layers_default() {
        let config = PowerConfig::default();
        let backend = MistralRsBackend::new(Arc::new(config));
        assert_eq!(backend.config.gpu.gpu_layers, 0);
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn test_stub_load_returns_error() {
        use crate::model::manifest::ModelManifest;
        use std::path::PathBuf;

        let backend = MistralRsBackend::new(test_config());
        let manifest = ModelManifest {
            name: "test".to_string(),
            format: ModelFormat::Gguf,
            size: 0,
            sha256: "abc".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: PathBuf::from("/tmp/test"),
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
        };
        let result = backend.load(&manifest).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mistralrs"));
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn test_stub_chat_returns_error() {
        let backend = MistralRsBackend::new(test_config());
        let request = ChatRequest {
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: false,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_batch: None,
            num_thread: None,
            num_thread_batch: None,
            flash_attention: None,
            num_gpu: None,
            main_gpu: None,
            use_mmap: None,
            use_mlock: None,
        };
        let result = backend.chat("test", request).await;
        assert!(result.is_err());
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn test_stub_complete_returns_error() {
        let backend = MistralRsBackend::new(test_config());
        let request = CompletionRequest {
            prompt: "test".to_string(),
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: false,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            num_ctx: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            tfs_z: None,
            typical_p: None,
            response_format: None,
            images: None,
            projector_path: None,
            repeat_last_n: None,
            penalize_newline: None,
            num_batch: None,
            num_thread: None,
            num_thread_batch: None,
            flash_attention: None,
            num_gpu: None,
            main_gpu: None,
            use_mmap: None,
            use_mlock: None,
            suffix: None,
            context: None,
        };
        let result = backend.complete("test", request).await;
        assert!(result.is_err());
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn test_stub_unload_succeeds() {
        let backend = MistralRsBackend::new(test_config());
        let result = backend.unload("test").await;
        assert!(result.is_ok());
    }

    #[cfg(not(feature = "mistralrs"))]
    #[tokio::test]
    async fn test_stub_embed_returns_error() {
        let backend = MistralRsBackend::new(test_config());
        let request = EmbeddingRequest {
            input: vec!["test".to_string()],
        };
        let result = backend.embed("test", request).await;
        assert!(result.is_err());
    }
}
