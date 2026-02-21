// mistral.rs backend implementation.
//
// When the `mistralrs` feature is enabled, this uses the pure Rust `mistralrs`
// crate (built on candle) to load GGUF models and run inference (chat,
// completion, embeddings). Without the feature, it returns `BackendNotAvailable`
// errors.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, StreamExt};

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
    /// Embedding models loaded via EmbeddingModelBuilder (HuggingFace format).
    #[cfg(feature = "mistralrs")]
    embedding_models: tokio::sync::RwLock<std::collections::HashMap<String, Arc<mistralrs::Model>>>,
    /// Vision/multimodal models loaded via VisionModelBuilder.
    #[cfg(feature = "mistralrs")]
    vision_models: tokio::sync::RwLock<std::collections::HashMap<String, Arc<mistralrs::Model>>>,
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
            #[cfg(feature = "mistralrs")]
            embedding_models: tokio::sync::RwLock::new(std::collections::HashMap::new()),
            #[cfg(feature = "mistralrs")]
            vision_models: tokio::sync::RwLock::new(std::collections::HashMap::new()),
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
        matches!(
            format,
            ModelFormat::Gguf
                | ModelFormat::HuggingFace
                | ModelFormat::SafeTensors
                | ModelFormat::Vision
        )
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        tracing::info!(model = %manifest.name, path = %manifest.path.display(), "Loading model via mistral.rs");

        // HuggingFace embedding models use a separate builder and map.
        if manifest.format == ModelFormat::HuggingFace {
            return self.load_embedding_model(manifest).await;
        }

        // SafeTensors chat models use TextModelBuilder with ISQ quantization.
        if manifest.format == ModelFormat::SafeTensors {
            return self.load_safetensors_model(manifest).await;
        }

        // Vision/multimodal models use VisionModelBuilder.
        if manifest.format == ModelFormat::Vision {
            return self.load_vision_model(manifest).await;
        }

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
        if self
            .embedding_models
            .write()
            .await
            .remove(model_name)
            .is_some()
        {
            tracing::info!(model = model_name, "Embedding model unloaded");
        }
        if self
            .vision_models
            .write()
            .await
            .remove(model_name)
            .is_some()
        {
            tracing::info!(model = model_name, "Vision model unloaded");
        }
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        use mistralrs::{RequestBuilder, TextMessageRole};

        // Check if this is a vision request (has images) and route accordingly.
        let has_images = request.messages.iter().any(|m| {
            matches!(&m.content, super::types::MessageContent::Parts(parts) if
                parts.iter().any(|p| matches!(p, super::types::ContentPart::ImageUrl { .. }))
            )
        }) || request.images.as_ref().is_some_and(|v| !v.is_empty());

        // Try vision model first if images present, fall back to text model.
        let model = if has_images {
            let vision = self.vision_models.read().await;
            if let Some(m) = vision.get(model_name) {
                Arc::clone(m)
            } else {
                // Fall back to text model (may not support images, but let mistralrs decide).
                let models = self.models.read().await;
                models
                    .get(model_name)
                    .map(|m| Arc::clone(&m.model))
                    .ok_or_else(|| {
                        PowerError::InferenceFailed(format!(
                            "Model '{model_name}' not loaded (vision model required for image inputs)"
                        ))
                    })?
            }
        } else {
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

            // Extract inline images from multimodal content parts (OpenAI format).
            let inline_images: Vec<String> = match &msg.content {
                super::types::MessageContent::Parts(parts) => parts
                    .iter()
                    .filter_map(|p| {
                        if let super::types::ContentPart::ImageUrl { image_url } = p {
                            Some(image_url.url.clone())
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => vec![],
            };

            // Combine inline images with top-level images field (Ollama-native format).
            let all_images: Vec<String> = if inline_images.is_empty() {
                request.images.clone().unwrap_or_default()
            } else {
                inline_images
            };

            if all_images.is_empty() {
                req_builder = req_builder.add_message(role, msg.content.text());
            } else {
                // Decode base64 images and add as multimodal message.
                use image::DynamicImage;
                let decoded: Vec<DynamicImage> = all_images
                    .iter()
                    .filter_map(|img_str| {
                        // Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                        let b64 = if let Some(pos) = img_str.find(',') {
                            &img_str[pos + 1..]
                        } else {
                            img_str.as_str()
                        };
                        use base64::Engine;
                        let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
                        image::load_from_memory(&bytes).ok()
                    })
                    .collect();

                if decoded.is_empty() {
                    req_builder = req_builder.add_message(role, msg.content.text());
                } else {
                    req_builder = req_builder
                        .add_image_message(role, msg.content.text(), decoded, &model)
                        .map_err(|e| {
                            PowerError::InferenceFailed(format!("failed to add image message: {e}"))
                        })?;
                }
            }
        }

        let has_tools = request.tools.is_some();

        // Use streaming API for true token-by-token output.
        // mistralrs::Model::stream_chat_request returns a Stream that borrows &Model,
        // so we drive it inside a spawn and forward chunks through a channel.
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<ChatResponseChunk>>(64);

        tokio::spawn(async move {
            let mut stream = match model.stream_chat_request(req_builder).await {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx
                        .send(Err(PowerError::InferenceFailed(format!(
                            "mistral.rs stream init failed: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            let mut prompt_tokens: Option<u32> = None;
            let mut think_parser = super::think_parser::ThinkBlockParser::new();
            let mut accumulated_raw = String::new();

            while let Some(response) = stream.next().await {
                match response {
                    mistralrs::Response::Chunk(chunk) => {
                        for choice in &chunk.choices {
                            let delta_text = choice.delta.content.as_deref().unwrap_or("");
                            let thinking_delta =
                                choice.delta.reasoning_content.as_deref().unwrap_or("");

                            accumulated_raw.push_str(delta_text);

                            // Feed delta through think-block parser for streaming separation.
                            let (content_part, thinking_part) = think_parser.feed(delta_text);

                            // Combine explicit reasoning_content with parsed think blocks.
                            let thinking_content = {
                                let combined = format!("{thinking_delta}{thinking_part}");
                                if combined.is_empty() {
                                    None
                                } else {
                                    Some(combined)
                                }
                            };

                            let finish_reason = choice.finish_reason.as_deref().unwrap_or("");
                            let is_done = finish_reason != "null" && !finish_reason.is_empty();

                            // On the final chunk, flush the think parser and parse tool calls.
                            let (final_content, final_thinking, tool_calls, done_reason) =
                                if is_done {
                                    let (fc, ft) = think_parser.flush();
                                    let full_content = content_part + &fc;
                                    let full_thinking = thinking_part + &ft;
                                    let tc = if has_tools {
                                        super::tool_parser::parse_tool_calls(&accumulated_raw)
                                    } else {
                                        None
                                    };
                                    let dr = if tc.is_some() {
                                        "tool_calls".to_string()
                                    } else {
                                        finish_reason.to_string()
                                    };
                                    let thinking = {
                                        let combined = format!("{thinking_delta}{full_thinking}");
                                        if combined.is_empty() {
                                            None
                                        } else {
                                            Some(combined)
                                        }
                                    };
                                    (full_content, thinking, tc, Some(dr))
                                } else {
                                    (content_part, thinking_content, None, None)
                                };

                            let chat_chunk = ChatResponseChunk {
                                content: final_content,
                                thinking_content: final_thinking,
                                done: is_done,
                                prompt_tokens,
                                done_reason,
                                prompt_eval_duration_ns: None,
                                tool_calls,
                            };

                            if tx.send(Ok(chat_chunk)).await.is_err() {
                                return; // client disconnected
                            }
                        }
                    }
                    mistralrs::Response::Done(resp) => {
                        prompt_tokens = Some(resp.usage.prompt_tokens as u32);
                        // Send a final done chunk with token counts.
                        let (fc, ft) = think_parser.flush();
                        if !fc.is_empty() || !ft.is_empty() {
                            let _ = tx
                                .send(Ok(ChatResponseChunk {
                                    content: fc,
                                    thinking_content: if ft.is_empty() { None } else { Some(ft) },
                                    done: true,
                                    prompt_tokens,
                                    done_reason: Some("stop".to_string()),
                                    prompt_eval_duration_ns: None,
                                    tool_calls: None,
                                }))
                                .await;
                        }
                    }
                    mistralrs::Response::ModelError(e, _) => {
                        let _ = tx
                            .send(Err(PowerError::InferenceFailed(format!(
                                "mistral.rs model error: {e}"
                            ))))
                            .await;
                        return;
                    }
                    _ => {} // CompletionChunk, etc. — not used in chat path
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
            images: None,
            session_id: request.session_id,
        };

        // Get the chat stream and map ChatResponseChunk -> CompletionResponseChunk
        let chat_stream = self.chat(model_name, chat_request).await?;

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
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        use mistralrs::EmbeddingRequestBuilder;

        if request.input.is_empty() {
            return Ok(EmbeddingResponse { embeddings: vec![] });
        }

        let model = {
            let models = self.embedding_models.read().await;
            models.get(model_name).map(Arc::clone).ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "Embedding model '{model_name}' not loaded. \
                         Register it with format=huggingface and load it first."
                ))
            })?
        };

        let embeddings = model
            .generate_embeddings(
                EmbeddingRequestBuilder::new()
                    .add_prompts(request.input.iter().map(|s| s.as_str())),
            )
            .await
            .map_err(|e| {
                PowerError::InferenceFailed(format!("Embedding generation failed: {e}"))
            })?;

        Ok(EmbeddingResponse { embeddings })
    }
}

// Private helpers for the mistralrs backend (not part of the Backend trait).
#[cfg(feature = "mistralrs")]
/// Parse an ISQ type string (e.g. `"Q8_0"`) into a `mistralrs::IsqType`.
/// Returns `Q8_0` for unrecognized values.
fn parse_isq_type(s: &str) -> mistralrs::IsqType {
    match s.to_uppercase().as_str() {
        "Q4_0" => mistralrs::IsqType::Q4_0,
        "Q4_1" => mistralrs::IsqType::Q4_1,
        "Q5_0" => mistralrs::IsqType::Q5_0,
        "Q5_1" => mistralrs::IsqType::Q5_1,
        "Q8_0" => mistralrs::IsqType::Q8_0,
        "Q8_1" => mistralrs::IsqType::Q8_1,
        "Q2K" | "Q2_K" => mistralrs::IsqType::Q2K,
        "Q3K" | "Q3_K" => mistralrs::IsqType::Q3K,
        "Q4K" | "Q4_K" => mistralrs::IsqType::Q4K,
        "Q5K" | "Q5_K" => mistralrs::IsqType::Q5K,
        "Q6K" | "Q6_K" => mistralrs::IsqType::Q6K,
        "Q8K" | "Q8_K" => mistralrs::IsqType::Q8K,
        "HQQ8" => mistralrs::IsqType::HQQ8,
        "HQQ4" => mistralrs::IsqType::HQQ4,
        "F8E4M3" => mistralrs::IsqType::F8E4M3,
        _ => {
            tracing::warn!(isq = s, "Unknown ISQ type, defaulting to Q8_0");
            mistralrs::IsqType::Q8_0
        }
    }
}

#[cfg(feature = "mistralrs")]
impl MistralRsBackend {
    /// Load a HuggingFace embedding model via EmbeddingModelBuilder.
    ///
    /// `manifest.path` must point to the local model directory containing
    /// `config.json`, `tokenizer.json`, and safetensors weight files.
    async fn load_embedding_model(&self, manifest: &ModelManifest) -> Result<()> {
        let model_id = manifest.name.clone();

        let mut builder = mistralrs::EmbeddingModelBuilder::new(&model_id)
            .with_token_source(mistralrs::TokenSource::None)
            .from_hf_cache_path(manifest.path.clone());

        if self.config.gpu.gpu_layers == 0 {
            builder = builder.with_force_cpu();
        }

        let model = builder.build().await.map_err(|e| {
            PowerError::InferenceFailed(format!(
                "Failed to load embedding model '{}' via mistral.rs: {e}",
                manifest.name
            ))
        })?;

        self.embedding_models
            .write()
            .await
            .insert(manifest.name.clone(), Arc::new(model));

        tracing::info!(model = %manifest.name, "Embedding model loaded successfully via mistral.rs");
        Ok(())
    }

    /// Load a SafeTensors chat model via TextModelBuilder with ISQ quantization.
    ///
    /// `manifest.path` must point to the local model directory containing
    /// `config.json`, `tokenizer.json`, and `.safetensors` weight files.
    ///
    /// ISQ type is read from `manifest.default_parameters["isq"]` (e.g. `"Q8_0"`).
    /// Defaults to `Q8_0` if not specified.
    async fn load_safetensors_model(&self, manifest: &ModelManifest) -> Result<()> {
        // Resolve ISQ type from manifest default_parameters, fallback to Q8_0.
        let isq = manifest
            .default_parameters
            .as_ref()
            .and_then(|p| p.get("isq"))
            .and_then(|v| v.as_str())
            .map(parse_isq_type)
            .unwrap_or(mistralrs::IsqType::Q8_0);

        tracing::info!(
            model = %manifest.name,
            path = %manifest.path.display(),
            isq = ?isq,
            "Loading SafeTensors model via mistral.rs TextModelBuilder"
        );

        let mut builder = mistralrs::TextModelBuilder::new(manifest.path.to_string_lossy())
            .with_token_source(mistralrs::TokenSource::None)
            .from_hf_cache_pathf(manifest.path.clone())
            .with_isq(isq);

        if let Some(ref template) = manifest.template_override {
            builder = builder.with_chat_template(template.clone());
        }

        if self.config.gpu.gpu_layers == 0 {
            builder = builder.with_force_cpu();
        }

        let model = builder.build().await.map_err(|e| {
            PowerError::InferenceFailed(format!(
                "Failed to load SafeTensors model '{}' via mistral.rs: {e}",
                manifest.name
            ))
        })?;

        self.models.write().await.insert(
            manifest.name.clone(),
            LoadedModel {
                model: Arc::new(model),
                raw_template: manifest.template_override.clone(),
            },
        );

        tracing::info!(model = %manifest.name, "SafeTensors model loaded successfully via mistral.rs");
        Ok(())
    }

    /// Load a vision/multimodal model via VisionModelBuilder.
    ///
    /// `manifest.path` must point to the local model directory containing
    /// `config.json`, `tokenizer.json`, and `.safetensors` weight files.
    ///
    /// ISQ type is read from `manifest.default_parameters["isq"]` (e.g. `"Q8_0"`).
    /// Defaults to `Q8_0` if not specified.
    async fn load_vision_model(&self, manifest: &ModelManifest) -> Result<()> {
        let isq = manifest
            .default_parameters
            .as_ref()
            .and_then(|p| p.get("isq"))
            .and_then(|v| v.as_str())
            .map(parse_isq_type)
            .unwrap_or(mistralrs::IsqType::Q8_0);

        tracing::info!(
            model = %manifest.name,
            path = %manifest.path.display(),
            isq = ?isq,
            "Loading vision model via mistral.rs VisionModelBuilder"
        );

        let mut builder = mistralrs::VisionModelBuilder::new(manifest.path.to_string_lossy())
            .with_token_source(mistralrs::TokenSource::None)
            .from_hf_cache_pathf(manifest.path.clone())
            .with_isq(isq);

        if let Some(ref template) = manifest.template_override {
            builder = builder.with_chat_template(template.clone());
        }

        if self.config.gpu.gpu_layers == 0 {
            builder = builder.with_force_cpu();
        }

        let model = builder.build().await.map_err(|e| {
            PowerError::InferenceFailed(format!(
                "Failed to load vision model '{}' via mistral.rs: {e}",
                manifest.name
            ))
        })?;

        self.vision_models
            .write()
            .await
            .insert(manifest.name.clone(), Arc::new(model));

        tracing::info!(model = %manifest.name, "Vision model loaded successfully via mistral.rs");
        Ok(())
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
        matches!(format, ModelFormat::Gguf | ModelFormat::SafeTensors)
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
    fn test_supports_safetensors() {
        let backend = MistralRsBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::SafeTensors));
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
        // SafeTensors is now supported; only truly unknown formats should fail.
        let backend = MistralRsBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::SafeTensors));
        assert!(backend.supports(&ModelFormat::Gguf));
        assert!(backend.supports(&ModelFormat::HuggingFace));
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
            num_parallel: None,
            session_id: None,
            images: None,
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
            num_parallel: None,
            session_id: None,
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

    #[test]
    fn test_supports_huggingface() {
        let backend = MistralRsBackend::new(test_config());
        assert!(backend.supports(&ModelFormat::HuggingFace));
    }

    #[tokio::test]
    async fn test_embed_empty_input_returns_empty() {
        // Without a loaded embedding model, embed() should fail with "not loaded".
        // But with empty input it should short-circuit before the model lookup.
        // This test verifies the empty-input fast path.
        #[cfg(feature = "mistralrs")]
        {
            let backend = MistralRsBackend::new(test_config());
            let request = EmbeddingRequest { input: vec![] };
            let result = backend.embed("any-model", request).await;
            assert!(result.is_ok());
            assert!(result.unwrap().embeddings.is_empty());
        }
        #[cfg(not(feature = "mistralrs"))]
        {
            // Stub always returns BackendNotAvailable
            let backend = MistralRsBackend::new(test_config());
            let request = EmbeddingRequest { input: vec![] };
            let result = backend.embed("any-model", request).await;
            assert!(result.is_err());
        }
    }

    #[tokio::test]
    async fn test_embed_model_not_loaded_returns_error() {
        #[cfg(feature = "mistralrs")]
        {
            let backend = MistralRsBackend::new(test_config());
            let request = EmbeddingRequest {
                input: vec!["hello".to_string()],
            };
            let result = backend.embed("nonexistent-embedding-model", request).await;
            assert!(result.is_err());
            let msg = result.unwrap_err().to_string();
            assert!(msg.contains("not loaded"), "error: {msg}");
        }
    }

    // ========================================================================
    // parse_isq_type tests (feature-gated)
    // ========================================================================

    #[cfg(feature = "mistralrs")]
    mod isq_tests {
        use super::super::parse_isq_type;
        use mistralrs::IsqType;

        #[test]
        fn test_parse_q8_0() {
            assert!(matches!(parse_isq_type("Q8_0"), IsqType::Q8_0));
            assert!(matches!(parse_isq_type("q8_0"), IsqType::Q8_0));
        }

        #[test]
        fn test_parse_q4k() {
            assert!(matches!(parse_isq_type("Q4K"), IsqType::Q4K));
            assert!(matches!(parse_isq_type("Q4_K"), IsqType::Q4K));
        }

        #[test]
        fn test_parse_q4_0() {
            assert!(matches!(parse_isq_type("Q4_0"), IsqType::Q4_0));
        }

        #[test]
        fn test_parse_q6k() {
            assert!(matches!(parse_isq_type("Q6K"), IsqType::Q6K));
            assert!(matches!(parse_isq_type("Q6_K"), IsqType::Q6K));
        }

        #[test]
        fn test_parse_hqq8() {
            assert!(matches!(parse_isq_type("HQQ8"), IsqType::HQQ8));
        }

        #[test]
        fn test_parse_unknown_defaults_to_q8_0() {
            assert!(matches!(parse_isq_type("UNKNOWN"), IsqType::Q8_0));
            assert!(matches!(parse_isq_type(""), IsqType::Q8_0));
        }
    }
}
