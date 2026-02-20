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

/// Default context size when `num_ctx` is not specified by the user.
///
/// Matches Ollama's default. Using the model's full `n_ctx_train` (e.g. 128K for
/// llama3.2) would allocate a massive KV cache that can OOM on machines with
/// limited memory. Users can override with `--num-ctx` or the `num_ctx` API field.
const DEFAULT_CTX_SIZE: u32 = 2048;

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
    /// Raw Jinja2 template string from GGUF metadata (for minijinja rendering).
    raw_template: Option<String>,
    load_mode: LoadMode,
    /// Trained context length from the model's GGUF metadata.
    n_ctx_train: u32,
    /// Cached context for KV cache reuse across requests.
    /// Holds (context, evaluated_tokens) — taken for each request and returned after.
    cached_ctx: Arc<std::sync::Mutex<Option<CachedContext>>>,
    /// LoRA adapter loaded from manifest.adapter_path (if any).
    lora_adapter: Option<Arc<std::sync::Mutex<SendableLoraAdapter>>>,
    /// Path to multimodal projector file (for vision models).
    projector_path: Option<String>,
}

/// A cached llama.cpp context with the tokens already evaluated in its KV cache.
#[cfg(feature = "llamacpp")]
struct CachedContext {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    /// Tokens that have been evaluated and are in the KV cache.
    evaluated_tokens: Vec<llama_cpp_2::token::LlamaToken>,
    /// Context size this was created with.
    ctx_size: u32,
}

/// Safety: LlamaContext wraps a C pointer that is safe to send between threads
/// when accessed sequentially (protected by Mutex). The llama.cpp library is
/// thread-safe for sequential access to a single context.
#[cfg(feature = "llamacpp")]
unsafe impl Send for CachedContext {}

/// Newtype wrapper around LlamaLoraAdapter to implement Send.
///
/// Safety: LlamaLoraAdapter wraps a C pointer that is safe to send between threads
/// when accessed sequentially (protected by Mutex). The adapter is only used during
/// context setup via `lora_adapter_set`, which is serialized by the Mutex.
#[cfg(feature = "llamacpp")]
struct SendableLoraAdapter(llama_cpp_2::model::LlamaLoraAdapter);

#[cfg(feature = "llamacpp")]
unsafe impl Send for SendableLoraAdapter {}

// NOTE: MtmdContext requires the `mtmd` feature on llama-cpp-2.
// Vision/multimodal support is not yet wired up; the projector_path field
// on LoadedModel is reserved for future use.

/// Create a dummy `LlamaBackend` reference for `new_context()`.
///
/// Safety: `LlamaBackend` is a zero-sized type and the `new_context` method
/// accepts `_: &LlamaBackend` as an unused proof-of-initialization parameter.
/// The actual backend is initialized once via `OnceLock` in `LlamaCppBackend::load`.
/// This helper avoids lifetime issues when calling `new_context` inside `spawn_blocking`.
#[cfg(feature = "llamacpp")]
fn backend_ref() -> &'static llama_cpp_2::llama_backend::LlamaBackend {
    // LlamaBackend is a ZST — this creates a valid reference without allocation.
    // Safety: ZSTs have no data to read/write; the reference is only used as a
    // type-level proof that the backend was initialized.
    unsafe { &*(std::ptr::NonNull::dangling().as_ptr()) }
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

        // Ensure the backend is initialized (first call wins, subsequent calls are no-ops).
        // We pre-check initialization to avoid panicking inside get_or_init.
        if self.llama_backend.get().is_none() {
            let backend = LlamaBackend::init().map_err(|e| {
                PowerError::InferenceFailed(format!("Failed to initialize llama.cpp backend: {e}"))
            })?;
            let _ = self.llama_backend.set(backend); // Ignore if another thread won the race
        }

        let gpu_layers = self.config.gpu.gpu_layers;
        let main_gpu = self.config.gpu.main_gpu;
        let use_mlock = self.config.use_mlock;
        let has_tensor_split = !self.config.gpu.tensor_split.is_empty();

        let path = manifest.path.clone();
        let model_name = manifest.name.clone();

        // Load model in a blocking task since it's CPU-intensive.
        // LlamaModelParams contains raw pointers (not Send), so we build everything
        // inside the blocking task and use backend_ref() for the ZST marker.
        let model = tokio::task::spawn_blocking(move || {
            let mut p = LlamaModelParams::default();
            if gpu_layers != 0 {
                p = p.with_n_gpu_layers(gpu_layers.max(0) as u32);
            }
            if main_gpu != 0 {
                p = p.with_main_gpu(main_gpu);
            }
            if use_mlock {
                p = p.with_use_mlock(true);
            }
            if has_tensor_split {
                use llama_cpp_2::model::params::LlamaSplitMode;
                p = p.with_split_mode(LlamaSplitMode::Layer);
                tracing::info!("Multi-GPU layer splitting enabled");
            }

            LlamaModel::load_from_file(backend_ref(), &path, &p)
                .map_err(|e| PowerError::InferenceFailed(format!("Failed to load model: {e}")))
        })
        .await
        .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

        let model_arc = Arc::new(model);

        // Detect chat template: prefer manifest.template_override (from Ollama registry),
        // then GGUF metadata, then fallback to Phi.
        let gguf_template = model_arc.meta_val_str("tokenizer.chat_template").ok();

        let raw_template_str = manifest.template_override.clone().or(gguf_template);

        let chat_template = raw_template_str
            .as_deref()
            .map(chat_template::detect)
            .unwrap_or(ChatTemplateKind::Phi);

        // Read trained context length from model metadata
        let n_ctx_train = model_arc.n_ctx_train();
        tracing::info!(model = %manifest.name, n_ctx_train = n_ctx_train, "Model context window detected");

        // Load LoRA adapter if specified in manifest
        let lora_adapter = if let Some(ref adapter_path) = manifest.adapter_path {
            let adapter_path_buf = std::path::PathBuf::from(adapter_path);
            if adapter_path_buf.exists() {
                let model_ref = model_arc.clone();
                let path = adapter_path_buf.clone();
                let adapter = tokio::task::spawn_blocking(move || {
                    let adapter = model_ref.lora_adapter_init(&path).map_err(|e| {
                        PowerError::InferenceFailed(format!(
                            "Failed to load LoRA adapter from {}: {e}",
                            path.display()
                        ))
                    })?;
                    // Wrap immediately inside spawn_blocking so we never send
                    // the raw LlamaLoraAdapter across threads.
                    Ok::<_, PowerError>(SendableLoraAdapter(adapter))
                })
                .await
                .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

                tracing::info!(
                    model = %manifest.name,
                    adapter = %adapter_path,
                    "LoRA adapter loaded"
                );
                Some(Arc::new(std::sync::Mutex::new(adapter)))
            } else {
                tracing::warn!(
                    model = %manifest.name,
                    adapter = %adapter_path,
                    "LoRA adapter file not found, skipping"
                );
                None
            }
        } else {
            None
        };

        self.models.write().await.insert(
            model_name.clone(),
            LoadedModel {
                name: model_name.clone(),
                path: manifest.path.clone(),
                model: model_arc,
                chat_template,
                raw_template: raw_template_str,
                load_mode: LoadMode::Inference,
                n_ctx_train,
                cached_ctx: Arc::new(std::sync::Mutex::new(None)),
                lora_adapter,
                projector_path: manifest.projector_path.clone(),
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
        // Look up the chat template and projector path for this model
        let (template, raw_template, projector_path, _model_n_ctx_train) = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| {
                    (
                        m.chat_template.clone(),
                        m.raw_template.clone(),
                        m.projector_path.clone(),
                        m.n_ctx_train,
                    )
                })
                .unwrap_or((ChatTemplateKind::Phi, None, None, 2048))
        };

        // Render chat template in a blocking task to avoid blocking the async executor.
        // Some GGUF models carry complex Jinja2 templates that can be slow to render.
        let messages_clone = request.messages.clone();
        let raw_template_clone = raw_template.clone();
        let template_clone = template.clone();
        let prompt = tokio::task::spawn_blocking(move || {
            chat_template::format_chat_prompt(
                &messages_clone,
                &template_clone,
                raw_template_clone.as_deref(),
            )
        })
        .await
        .map_err(|e| {
            PowerError::InferenceFailed(format!("Chat template rendering task failed: {e}"))
        })?;

        // Check if images are present in the request
        let has_images = request.messages.iter().any(|m| {
            m.images.as_ref().is_some_and(|imgs| !imgs.is_empty())
                || matches!(&m.content, super::types::MessageContent::Parts(parts)
                    if parts.iter().any(|p| matches!(p, super::types::ContentPart::ImageUrl { .. })))
        });

        if has_images && projector_path.is_none() {
            tracing::warn!(
                "Vision/multimodal images detected but no projector file available; \
                 images will be ignored and only text content will be processed. \
                 Pull a vision model (e.g. llava) to enable image processing."
            );
        } else if has_images {
            tracing::info!("Vision inference with multimodal projector");
        }

        // Extract base64 images from messages for vision inference
        let images: Vec<String> = if has_images {
            request
                .messages
                .iter()
                .flat_map(|m| {
                    // Collect from Ollama-native `images` field
                    let ollama_imgs = m.images.clone().unwrap_or_default();
                    // Collect from OpenAI image_url content parts (extract base64 data URIs)
                    let openai_imgs: Vec<String> = match &m.content {
                        super::types::MessageContent::Parts(parts) => parts
                            .iter()
                            .filter_map(|p| match p {
                                super::types::ContentPart::ImageUrl { image_url } => {
                                    // Handle data:image/...;base64,<data> format
                                    if let Some(data) = image_url.url.strip_prefix("data:") {
                                        data.split_once(",").map(|(_, b64)| b64.to_string())
                                    } else {
                                        None // URL-based images not supported yet
                                    }
                                }
                                _ => None,
                            })
                            .collect(),
                        _ => vec![],
                    };
                    ollama_imgs.into_iter().chain(openai_imgs)
                })
                .collect()
        } else {
            vec![]
        };

        let completion_req = CompletionRequest {
            prompt,
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stop: request.stop,
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
            response_format: request.response_format,
            images: if images.is_empty() {
                None
            } else {
                Some(images)
            },
            projector_path,
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
            suffix: None,
            context: None,
        };

        // Get completion stream from the underlying complete() method
        let stream = self.complete(model_name, completion_req).await?;

        // Map CompletionResponseChunk -> ChatResponseChunk with tool call and think block detection
        use futures::StreamExt;
        let collected_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let text_clone = collected_text.clone();
        let has_tools = request.tools.is_some();
        let mut think_parser = super::think_parser::ThinkBlockParser::new();
        let chat_stream = stream.map(move |chunk_result| {
            chunk_result.map(|chunk| {
                // Accumulate text for tool call detection
                if has_tools && !chunk.text.is_empty() {
                    if let Ok(mut t) = text_clone.lock() {
                        t.push_str(&chunk.text);
                    }
                }

                // Parse think blocks from the token stream
                let (content, thinking) = if chunk.done {
                    let (mut c, mut t) = think_parser.flush();
                    // Prepend any remaining text from the final chunk
                    if !chunk.text.is_empty() {
                        let (fc, ft) = think_parser.feed(&chunk.text);
                        c = fc + &c;
                        t = ft + &t;
                    }
                    (c, t)
                } else {
                    think_parser.feed(&chunk.text)
                };

                let thinking_content = if thinking.is_empty() {
                    None
                } else {
                    Some(thinking)
                };

                // On the final chunk, try to parse tool calls from accumulated text
                let tool_calls = if chunk.done && has_tools {
                    let full_text = text_clone.lock().ok();
                    full_text.and_then(|t| super::tool_parser::parse_tool_calls(&t))
                } else {
                    None
                };

                ChatResponseChunk {
                    content,
                    thinking_content,
                    done: chunk.done,
                    prompt_tokens: chunk.prompt_tokens,
                    done_reason: if tool_calls.is_some() && chunk.done {
                        Some("tool_calls".to_string())
                    } else {
                        chunk.done_reason
                    },
                    prompt_eval_duration_ns: chunk.prompt_eval_duration_ns,
                    tool_calls,
                }
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
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;

        let (model_arc, cached_ctx_mutex, lora_adapter, model_n_ctx_train) = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| {
                    (
                        m.model.clone(),
                        m.cached_ctx.clone(),
                        m.lora_adapter.clone(),
                        m.n_ctx_train,
                    )
                })
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
                })?
        };

        let max_tokens = request.max_tokens.unwrap_or(512) as usize;
        let temperature = request.temperature.unwrap_or(0.8);
        let top_p = request.top_p.unwrap_or(0.95);
        let top_k = request.top_k;
        let min_p = request.min_p;
        let repeat_penalty = request.repeat_penalty;
        let frequency_penalty = request.frequency_penalty;
        let presence_penalty = request.presence_penalty;
        let repeat_last_n = request.repeat_last_n.unwrap_or(64);
        let _penalize_newline = request.penalize_newline.unwrap_or(true);
        let seed = request.seed.unwrap_or(0).max(0) as u32;
        let ctx_size = match request.num_ctx {
            Some(requested) => {
                if requested > model_n_ctx_train {
                    tracing::warn!(
                        requested = requested,
                        trained = model_n_ctx_train,
                        "Requested context size exceeds model's trained context length, quality may degrade"
                    );
                }
                requested
            }
            None => {
                let effective = DEFAULT_CTX_SIZE.min(model_n_ctx_train);
                tracing::info!(
                    default = effective,
                    trained = model_n_ctx_train,
                    "Using default context size (override with num_ctx or --num-ctx)"
                );
                effective
            }
        };
        let num_batch = request.num_batch;
        let num_thread = request.num_thread;
        let num_thread_batch = request.num_thread_batch;
        let flash_attention = request.flash_attention.unwrap_or(false);
        let mirostat = request.mirostat;
        let mirostat_tau = request.mirostat_tau;
        let mirostat_eta = request.mirostat_eta;
        let _tfs_z = request.tfs_z; // tail_free sampling removed in llama-cpp-2 v0.1.133
        let typical_p = request.typical_p;
        let response_format = request.response_format.clone();
        let stop_sequences = request.stop.clone().unwrap_or_default();

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<CompletionResponseChunk>>(32);

        // Run inference in a blocking task
        let cached_ctx_for_return = cached_ctx_mutex.clone();
        tokio::task::spawn_blocking(move || {
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

            let prompt_token_count = tokens.len() as u32;

            // Try to reuse cached context with KV cache prefix matching
            let cached = cached_ctx_mutex.lock().unwrap().take();
            let (mut ctx, skip_tokens) = match cached {
                Some(mut cached) if cached.ctx_size == ctx_size => {
                    // Find common prefix between cached tokens and new tokens
                    let common_len = cached
                        .evaluated_tokens
                        .iter()
                        .zip(tokens.iter())
                        .take_while(|(a, b)| a == b)
                        .count();

                    if common_len > 0 && common_len <= tokens.len() {
                        // Remove KV cache entries after the common prefix
                        if common_len < cached.evaluated_tokens.len() {
                            let _ = cached.ctx.clear_kv_cache_seq(
                                Some(0),
                                Some(common_len as u32),
                                None,
                            );
                        }
                        tracing::debug!(
                            common = common_len,
                            total = tokens.len(),
                            "Reusing KV cache prefix"
                        );
                        (cached.ctx, common_len)
                    } else {
                        // No useful prefix — clear and reuse the context
                        cached.ctx.clear_kv_cache();
                        (cached.ctx, 0)
                    }
                }
                _ => {
                    // No cached context or size mismatch — create new
                    let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(
                        std::num::NonZeroU32::new(ctx_size)
                            .unwrap_or(std::num::NonZeroU32::new(2048).unwrap()),
                    ));
                    if let Some(batch) = num_batch {
                        ctx_params = ctx_params.with_n_batch(batch);
                    }
                    if let Some(threads) = num_thread {
                        ctx_params = ctx_params.with_n_threads(threads as i32);
                    }
                    if let Some(threads_batch) = num_thread_batch {
                        ctx_params = ctx_params.with_n_threads_batch(threads_batch as i32);
                    }
                    if flash_attention {
                        // LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
                        ctx_params = ctx_params.with_flash_attention_policy(1);
                    }
                    match model_arc.new_context(backend_ref(), ctx_params) {
                        Ok(c) => {
                            // Safety: model_arc is an Arc kept alive in LoadedModel for the
                            // entire duration the context exists. The context is returned to
                            // CachedContext (which stores LlamaContext<'static>) and is always
                            // dropped before the model.
                            let c: llama_cpp_2::context::LlamaContext<'static> =
                                unsafe { std::mem::transmute(c) };
                            (c, 0)
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                                "Failed to create context: {e}"
                            ))));
                            return;
                        }
                    }
                }
            };

            // Only evaluate tokens not already in the KV cache
            let tokens_to_eval = &tokens[skip_tokens..];
            let prompt_eval_start = std::time::Instant::now();

            // Apply LoRA adapter to context if available
            if let Some(ref adapter_arc) = lora_adapter {
                let mut wrapper = adapter_arc.lock().unwrap();
                if let Err(e) = ctx.lora_adapter_set(&mut wrapper.0, 1.0) {
                    let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                        "Failed to apply LoRA adapter: {e}"
                    ))));
                    return;
                }
            }

            if !tokens_to_eval.is_empty() {
                // Allocate batch only for the tokens we need to evaluate, not the full context.
                let batch_size = tokens_to_eval.len().max(1);
                let mut batch = LlamaBatch::new(batch_size, 1);
                for (i, &token) in tokens_to_eval.iter().enumerate() {
                    let pos = (skip_tokens + i) as i32;
                    let is_last = i == tokens_to_eval.len() - 1;
                    if batch.add(token, pos, &[0], is_last).is_err() {
                        let _ = tx.blocking_send(Err(PowerError::InferenceFailed(
                            "Failed to add token to batch".to_string(),
                        )));
                        return;
                    }
                }

                if let Err(e) = ctx.decode(&mut batch) {
                    let _ = tx.blocking_send(Err(PowerError::InferenceFailed(format!(
                        "Decode failed: {e}"
                    ))));
                    return;
                }
            }
            let prompt_eval_duration_ns = prompt_eval_start.elapsed().as_nanos() as u64;

            // Build sampler chain based on request parameters
            let mut samplers: Vec<LlamaSampler> = Vec::new();

            // JSON grammar constraint (supports "json" string or JSON Schema object)
            if let Some(ref fmt) = response_format {
                if let Some(grammar) = super::json_schema::format_to_gbnf(fmt) {
                    match LlamaSampler::grammar(&model_arc, &grammar, "root") {
                        Ok(s) => samplers.push(s),
                        Err(e) => {
                            tracing::warn!("Failed to create grammar sampler: {e}, ignoring");
                        }
                    }
                }
            }

            // Repetition penalties (must come before other samplers)
            if repeat_penalty.is_some() || frequency_penalty.is_some() || presence_penalty.is_some()
            {
                samplers.push(LlamaSampler::penalties(
                    repeat_last_n,
                    repeat_penalty.unwrap_or(1.0),
                    frequency_penalty.unwrap_or(0.0),
                    presence_penalty.unwrap_or(0.0),
                ));
            }

            // Mirostat replaces the standard sampling chain
            match mirostat {
                Some(1) => {
                    let tau = mirostat_tau.unwrap_or(5.0);
                    let eta = mirostat_eta.unwrap_or(0.1);
                    samplers.push(LlamaSampler::temp(temperature));
                    samplers.push(LlamaSampler::mirostat(
                        model_arc.n_vocab(),
                        seed,
                        tau,
                        eta,
                        100,
                    ));
                }
                Some(2) => {
                    let tau = mirostat_tau.unwrap_or(5.0);
                    let eta = mirostat_eta.unwrap_or(0.1);
                    samplers.push(LlamaSampler::temp(temperature));
                    samplers.push(LlamaSampler::mirostat_v2(seed, tau, eta));
                }
                _ => {
                    // Standard sampling chain
                    if let Some(k) = top_k {
                        samplers.push(LlamaSampler::top_k(k));
                    }

                    // NOTE: tail_free sampling (tfs_z) was removed in llama-cpp-2 v0.1.133

                    if let Some(p) = typical_p {
                        samplers.push(LlamaSampler::typical(p, 1));
                    }

                    samplers.push(LlamaSampler::top_p(top_p, 1));

                    if let Some(p) = min_p {
                        samplers.push(LlamaSampler::min_p(p, 1));
                    }

                    samplers.push(LlamaSampler::temp(temperature));
                    samplers.push(LlamaSampler::dist(seed));
                }
            }

            let mut sampler = LlamaSampler::chain_simple(samplers);

            let mut n_cur = tokens.len();
            let eos_token = model_arc.token_eos();
            let mut generated_text = String::new();
            let mut all_tokens = tokens.clone(); // Track all tokens for cache

            // Generate tokens
            for _i in 0..max_tokens {
                let new_token = sampler.sample(&ctx, -1);

                if new_token == eos_token {
                    let _ = tx.blocking_send(Ok(CompletionResponseChunk {
                        text: String::new(),
                        done: true,
                        prompt_tokens: Some(prompt_token_count),
                        done_reason: Some("stop".to_string()),
                        prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                        token_id: None,
                    }));
                    // Return context to cache (with prompt + generated tokens)
                    all_tokens.push(new_token);
                    *cached_ctx_for_return.lock().unwrap() = Some(CachedContext {
                        ctx,
                        evaluated_tokens: all_tokens,
                        ctx_size,
                    });
                    return;
                }

                all_tokens.push(new_token);

                let text = {
                    let mut decoder = encoding_rs::UTF_8.new_decoder();
                    model_arc
                        .token_to_piece(new_token, &mut decoder, true, None)
                        .unwrap_or_default()
                };

                generated_text.push_str(&text);

                // Check stop sequences
                let mut should_stop = false;
                for stop in &stop_sequences {
                    if generated_text.ends_with(stop) {
                        should_stop = true;
                        break;
                    }
                }

                if tx
                    .blocking_send(Ok(CompletionResponseChunk {
                        text,
                        done: should_stop,
                        prompt_tokens: if should_stop {
                            Some(prompt_token_count)
                        } else {
                            None
                        },
                        done_reason: if should_stop {
                            Some("stop".to_string())
                        } else {
                            None
                        },
                        prompt_eval_duration_ns: if should_stop {
                            Some(prompt_eval_duration_ns)
                        } else {
                            None
                        },
                        token_id: Some(new_token.0 as u32),
                    }))
                    .is_err()
                {
                    // Receiver dropped — still cache the context
                    *cached_ctx_for_return.lock().unwrap() = Some(CachedContext {
                        ctx,
                        evaluated_tokens: all_tokens,
                        ctx_size,
                    });
                    return;
                }

                if should_stop {
                    // Return context to cache
                    *cached_ctx_for_return.lock().unwrap() = Some(CachedContext {
                        ctx,
                        evaluated_tokens: all_tokens,
                        ctx_size,
                    });
                    return;
                }

                // Prepare next batch
                let mut batch = LlamaBatch::new(1, 1);
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

            // Max tokens reached — return context to cache
            let _ = tx.blocking_send(Ok(CompletionResponseChunk {
                text: String::new(),
                done: true,
                prompt_tokens: Some(prompt_token_count),
                done_reason: Some("length".to_string()),
                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                token_id: None,
            }));
            *cached_ctx_for_return.lock().unwrap() = Some(CachedContext {
                ctx,
                evaluated_tokens: all_tokens,
                ctx_size,
            });
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
            let (path, chat_template, raw_template, lora_adapter, projector_path) = {
                let models = self.models.read().await;
                let m = models.get(model_name).ok_or_else(|| {
                    PowerError::InferenceFailed(format!(
                        "Model '{model_name}' was unloaded during embed reload"
                    ))
                })?;
                (
                    m.path.clone(),
                    m.chat_template.clone(),
                    m.raw_template.clone(),
                    m.lora_adapter.clone(),
                    m.projector_path.clone(),
                )
            };

            tracing::info!(model = model_name, "Reloading model with embedding mode");

            let gpu_layers = self.config.gpu.gpu_layers;

            let path_clone = path.clone();
            let model = tokio::task::spawn_blocking(move || {
                let params = if gpu_layers != 0 {
                    LlamaModelParams::default().with_n_gpu_layers(gpu_layers.max(0) as u32)
                } else {
                    LlamaModelParams::default()
                };
                LlamaModel::load_from_file(backend_ref(), &path_clone, &params).map_err(|e| {
                    PowerError::InferenceFailed(format!(
                        "Failed to reload model for embedding: {e}"
                    ))
                })
            })
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("Task join error: {e}")))??;

            let model_arc = Arc::new(model);
            let n_ctx_train = model_arc.n_ctx_train();
            let name = model_name.to_string();
            self.models.write().await.insert(
                name.clone(),
                LoadedModel {
                    name,
                    path,
                    model: model_arc,
                    chat_template,
                    raw_template,
                    load_mode: LoadMode::Embedding,
                    n_ctx_train,
                    cached_ctx: Arc::new(std::sync::Mutex::new(None)),
                    lora_adapter,
                    projector_path,
                },
            );
        }

        let model_arc = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .ok_or_else(|| {
                    PowerError::InferenceFailed(format!(
                        "Model '{model_name}' was unloaded during embed"
                    ))
                })?
                .model
                .clone()
        };

        let input = request.input.clone();

        tokio::task::spawn_blocking(move || {
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(2048))
                .with_embeddings(true);
            let mut ctx = model_arc
                .new_context(backend_ref(), ctx_params)
                .map_err(|e| {
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

    #[test]
    fn test_backend_name() {
        let backend = LlamaCppBackend::new(test_config());
        assert_eq!(backend.name(), "llama.cpp");
    }

    #[test]
    fn test_backend_does_not_support_unknown_format() {
        let backend = LlamaCppBackend::new(test_config());
        assert!(!backend.supports(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_backend_config_gpu_layers_default() {
        let config = PowerConfig::default();
        let backend = LlamaCppBackend::new(Arc::new(config));
        assert_eq!(backend.config.gpu.gpu_layers, 0);
    }

    #[test]
    fn test_default_ctx_size_is_2048() {
        // Matches Ollama's default to prevent OOM on resource-constrained machines.
        assert_eq!(DEFAULT_CTX_SIZE, 2048);
    }

    #[test]
    fn test_default_ctx_size_less_than_large_model_ctx() {
        // Models like llama3.2 have n_ctx_train = 131072 (128K).
        // DEFAULT_CTX_SIZE must be much smaller to avoid OOM.
        const { assert!(DEFAULT_CTX_SIZE < 131072) };
        const { assert!(DEFAULT_CTX_SIZE <= 8192) };
    }
}
