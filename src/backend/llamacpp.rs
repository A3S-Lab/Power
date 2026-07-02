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
use std::num::NonZeroU32;
#[cfg(feature = "llamacpp")]
use std::sync::{Mutex, MutexGuard};
#[cfg(feature = "llamacpp")]
use tokio::sync::RwLock;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

#[cfg(feature = "llamacpp")]
use super::chat_template::{self, ChatTemplateKind};
#[cfg(feature = "llamacpp")]
use super::types::EffectivePromptDigest;
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
#[allow(dead_code)]
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
    /// Per-session KV cache map: session_id → CachedContext.
    ///
    /// Anonymous requests (session_id = None) never touch this map — they always
    /// create a fresh context and discard it after use, preventing cross-request
    /// cache leakage in multi-tenant deployments.
    session_cache: SessionCache,
    /// LoRA adapter loaded from manifest.adapter_path (if any).
    lora_adapter: Option<Arc<Mutex<SendableLoraAdapter>>>,
    /// Path to multimodal projector file (for vision models).
    projector_path: Option<String>,
    /// Multimodal context for vision/audio inference (initialized from projector_path).
    mtmd_ctx: Option<Arc<Mutex<SendableMtmdContext>>>,
}

/// Newtype wrapper around MtmdContext to implement Send.
///
/// Safety: MtmdContext wraps a C pointer that is safe to send between threads
/// when accessed sequentially (protected by Mutex). The MTMD context is only
/// used during inference inside spawn_blocking, serialized by the Mutex.
#[cfg(feature = "llamacpp")]
struct SendableMtmdContext(llama_cpp_2::mtmd::MtmdContext);

#[cfg(feature = "llamacpp")]
unsafe impl Send for SendableMtmdContext {}

/// A cached llama.cpp context with the tokens already evaluated in its KV cache.
#[cfg(feature = "llamacpp")]
struct CachedContext {
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    /// Tokens that have been evaluated and are in the KV cache.
    evaluated_tokens: Vec<llama_cpp_2::token::LlamaToken>,
    /// Context size this was created with.
    ctx_size: u32,
    /// Last time this cache entry was used (for TTL eviction).
    last_used: std::time::Instant,
}

#[cfg(feature = "llamacpp")]
type SessionCache = Arc<Mutex<HashMap<String, CachedContext>>>;

/// TTL for idle session KV caches. Sessions not used within this duration are evicted.
#[cfg(feature = "llamacpp")]
const SESSION_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(300);

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

#[cfg(feature = "llamacpp")]
fn lock_session_cache(
    cache: &SessionCache,
) -> Result<MutexGuard<'_, HashMap<String, CachedContext>>> {
    cache.lock().map_err(|_| {
        PowerError::InferenceFailed("llama.cpp: session cache lock poisoned".to_string())
    })
}

#[cfg(feature = "llamacpp")]
fn cache_session_context(cache: &SessionCache, session_id: &str, context: CachedContext) {
    match lock_session_cache(cache) {
        Ok(mut cache) => {
            cache.insert(session_id.to_string(), context);
        }
        Err(e) => {
            tracing::warn!(
                session = %session_id,
                error = %e,
                "llama.cpp: failed to return context to session cache"
            );
        }
    }
}

#[cfg(feature = "llamacpp")]
fn lock_lora_adapter(
    adapter: &Arc<Mutex<SendableLoraAdapter>>,
) -> Result<MutexGuard<'_, SendableLoraAdapter>> {
    adapter.lock().map_err(|_| {
        PowerError::InferenceFailed("llama.cpp: LoRA adapter lock poisoned".to_string())
    })
}

#[cfg(feature = "llamacpp")]
fn lock_mtmd_context(
    ctx: &Arc<Mutex<SendableMtmdContext>>,
) -> Result<MutexGuard<'_, SendableMtmdContext>> {
    ctx.lock().map_err(|_| {
        PowerError::InferenceFailed("llama.cpp: MTMD context lock poisoned".to_string())
    })
}

#[cfg(feature = "llamacpp")]
fn lock_collected_text(text: &Mutex<String>) -> MutexGuard<'_, String> {
    match text.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("llama.cpp: collected tool-call text lock poisoned, recovering");
            poisoned.into_inner()
        }
    }
}

#[cfg(feature = "llamacpp")]
fn nonzero_context_size(ctx_size: u32) -> NonZeroU32 {
    if let Some(size) = NonZeroU32::new(ctx_size) {
        return size;
    }

    tracing::warn!(
        fallback = DEFAULT_CTX_SIZE,
        "llama.cpp: context size was zero; using default context size"
    );
    NonZeroU32::new(DEFAULT_CTX_SIZE).unwrap_or(NonZeroU32::MIN)
}

#[cfg(any(feature = "llamacpp", test))]
fn collect_llamacpp_openai_images(
    message_index: usize,
    parts: &[super::types::ContentPart],
) -> Result<Vec<String>> {
    parts
        .iter()
        .enumerate()
        .filter_map(|(part_index, part)| match part {
            super::types::ContentPart::ImageUrl { image_url } => Some(
                normalize_llamacpp_image_url(message_index, part_index, &image_url.url),
            ),
            super::types::ContentPart::Text { .. } => None,
        })
        .collect()
}

#[cfg(any(feature = "llamacpp", test))]
fn collect_llamacpp_chat_images(request: &ChatRequest) -> Result<Vec<String>> {
    let mut images = Vec::new();

    for (message_index, message) in request.messages.iter().enumerate() {
        if let Some(ollama_images) = &message.images {
            images.extend(ollama_images.iter().cloned());
        }

        if let super::types::MessageContent::Parts(parts) = &message.content {
            images.extend(collect_llamacpp_openai_images(message_index, parts)?);
        }
    }

    if let Some(request_images) = &request.images {
        images.extend(request_images.iter().cloned());
    }

    Ok(images)
}

#[cfg(any(feature = "llamacpp", test))]
fn normalize_llamacpp_image_url(
    message_index: usize,
    part_index: usize,
    image_url: &str,
) -> Result<String> {
    let image_url = image_url.trim();
    if image_url.starts_with("http://") || image_url.starts_with("https://") {
        return Err(PowerError::InvalidFormat(format!(
            "Unsupported image input at message {message_index}, part {part_index}: \
             remote image URLs are not supported by llama.cpp; provide base64 image data or a data URI"
        )));
    }

    let image_data = image_url
        .split_once(',')
        .map_or(image_url, |(_, data)| data)
        .trim();
    if image_data.is_empty() {
        return Err(PowerError::InvalidFormat(format!(
            "Invalid image input at message {message_index}, part {part_index}: empty image data"
        )));
    }

    Ok(image_data.to_string())
}

#[cfg(feature = "llamacpp")]
fn send_completion_result(
    tx: &tokio::sync::mpsc::Sender<Result<CompletionResponseChunk>>,
    result: Result<CompletionResponseChunk>,
) -> bool {
    match tx.blocking_send(result) {
        Ok(()) => true,
        Err(e) => {
            tracing::debug!(
                error = %e,
                "llama.cpp completion receiver dropped; stopping inference"
            );
            false
        }
    }
}

#[cfg(any(feature = "llamacpp", test))]
fn ensure_llamacpp_images_supported(
    model_name: &str,
    has_images: bool,
    has_projector: bool,
) -> Result<()> {
    if has_images && !has_projector {
        return Err(PowerError::InvalidFormat(format!(
            "llama.cpp model '{model_name}' was not loaded with a multimodal projector; \
             image inputs cannot be processed"
        )));
    }

    Ok(())
}

// NOTE: MtmdContext requires the `mtmd` feature on llama-cpp-2.
// Requests with images are only accepted when the model has an initialized
// multimodal projector; otherwise they fail instead of falling back to text-only.

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
                Some(Arc::new(Mutex::new(adapter)))
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

        // Initialize multimodal context if projector_path is set.
        // MtmdContext::init_from_file is blocking (loads the projector weights).
        let mtmd_ctx = if let Some(ref proj_path) = manifest.projector_path {
            let proj_path_str = proj_path.clone();
            let model_ref = model_arc.clone();
            let model_name_for_log = manifest.name.clone();
            match tokio::task::spawn_blocking(move || {
                use llama_cpp_2::mtmd::{MtmdContext, MtmdContextParams};
                let params = MtmdContextParams::default();
                MtmdContext::init_from_file(&proj_path_str, &model_ref, &params)
                    .map(SendableMtmdContext)
                    .map_err(|e| {
                        PowerError::InferenceFailed(format!(
                            "Failed to initialize MTMD context from {proj_path_str}: {e}"
                        ))
                    })
            })
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("MTMD init task failed: {e}")))
            {
                Ok(Ok(ctx)) => {
                    tracing::info!(
                        model = %model_name_for_log,
                        projector = %proj_path,
                        "Multimodal projector loaded"
                    );
                    Some(Arc::new(Mutex::new(ctx)))
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        model = %manifest.name,
                        projector = %proj_path,
                        error = %e,
                        "Failed to load multimodal projector, vision inference disabled"
                    );
                    None
                }
                Err(e) => {
                    tracing::warn!(error = %e, "MTMD init task panicked");
                    None
                }
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
                session_cache: Arc::new(Mutex::new(HashMap::new())),
                lora_adapter,
                projector_path: manifest.projector_path.clone(),
                mtmd_ctx,
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
            let model = models.get(model_name).ok_or_else(|| {
                PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
            })?;
            (
                model.chat_template.clone(),
                model.raw_template.clone(),
                model.projector_path.clone(),
                model.n_ctx_train,
            )
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

        let has_images = request.has_image_inputs();
        ensure_llamacpp_images_supported(model_name, has_images, projector_path.is_some())?;
        if has_images {
            tracing::info!("Vision inference with multimodal projector");
        }

        let images = if has_images {
            collect_llamacpp_chat_images(&request)?
        } else {
            Vec::new()
        };

        let completion_req = CompletionRequest {
            prompt,
            session_id: request.session_id,
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
            num_parallel: request.num_parallel,
            suffix: None,
            context: None,
        };

        // Get completion stream from the underlying complete() method
        let stream = self.complete(model_name, completion_req).await?;

        // Map CompletionResponseChunk -> ChatResponseChunk with tool call and think block detection
        use futures::StreamExt;
        let collected_text = Arc::new(Mutex::new(String::new()));
        let text_clone = collected_text.clone();
        let has_tools = request.tools.is_some();
        let mut think_parser = super::think_parser::ThinkBlockParser::new();
        let chat_stream = stream.map(move |chunk_result| {
            chunk_result.map(|chunk| {
                // Accumulate text for tool call detection
                if has_tools && !chunk.text.is_empty() {
                    let mut text = lock_collected_text(text_clone.as_ref());
                    text.push_str(&chunk.text);
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
                    let full_text = lock_collected_text(text_clone.as_ref());
                    super::tool_parser::parse_tool_calls(&full_text)
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

    async fn effective_chat_prompt_digest(
        &self,
        model_name: &str,
        request: &ChatRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        if request.has_image_inputs() {
            return Ok(None);
        }

        let (template, raw_template) = {
            let models = self.models.read().await;
            let model = models.get(model_name).ok_or_else(|| {
                PowerError::InferenceFailed(format!("Model '{model_name}' not loaded"))
            })?;
            (model.chat_template.clone(), model.raw_template.clone())
        };

        let messages = request.messages.clone();
        let prompt = tokio::task::spawn_blocking(move || {
            chat_template::format_chat_prompt(&messages, &template, raw_template.as_deref())
        })
        .await
        .map_err(|e| {
            PowerError::InferenceFailed(format!("Chat template rendering task failed: {e}"))
        })?;

        Ok(Some(EffectivePromptDigest::chat_rendered_prompt(
            "llama.cpp",
            &prompt,
        )))
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::sampling::LlamaSampler;

        let (model_arc, session_cache, lora_adapter, model_n_ctx_train, mtmd_ctx) = {
            let models = self.models.read().await;
            models
                .get(model_name)
                .map(|m| {
                    (
                        m.model.clone(),
                        m.session_cache.clone(),
                        m.lora_adapter.clone(),
                        m.n_ctx_train,
                        m.mtmd_ctx.clone(),
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
        // Per-request num_thread overrides config default; fall back to config if not set
        let num_thread = request.num_thread.or(self.config.num_thread);
        let num_thread_batch = request.num_thread_batch;
        // Per-request flash_attention overrides config default
        let flash_attention = request
            .flash_attention
            .unwrap_or(self.config.flash_attention);
        let mirostat = request.mirostat;
        let mirostat_tau = request.mirostat_tau;
        let mirostat_eta = request.mirostat_eta;
        let _tfs_z = request.tfs_z; // tail_free sampling removed in llama-cpp-2 v0.1.133
        let typical_p = request.typical_p;
        let response_format = request.response_format.clone();
        let stop_sequences = request.stop.clone().unwrap_or_default();
        let has_images = request.images.as_ref().is_some_and(|v| !v.is_empty());
        ensure_llamacpp_images_supported(model_name, has_images, mtmd_ctx.is_some())?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<CompletionResponseChunk>>(32);

        // Evict stale session caches before starting inference.
        {
            let mut cache = lock_session_cache(&session_cache)?;
            cache.retain(|_, v| v.last_used.elapsed() < SESSION_CACHE_TTL);
        }

        let session_id = request.session_id.clone();
        let session_cache_for_return = session_cache.clone();

        // Run inference in a blocking task
        tokio::task::spawn_blocking(move || {
            let prompt_eval_start = std::time::Instant::now();

            // Determine whether to use the MTMD (multimodal) path.
            // Conditions: images present in request AND mtmd_ctx loaded for this model.
            let use_mtmd = has_images && mtmd_ctx.is_some();

            // ----------------------------------------------------------------
            // MTMD path: vision/multimodal inference
            // ----------------------------------------------------------------
            if use_mtmd {
                use llama_cpp_2::mtmd::{mtmd_default_marker, MtmdBitmap, MtmdInputText};

                let mtmd_guard = match mtmd_ctx.as_ref() {
                    Some(ctx) => match lock_mtmd_context(ctx) {
                        Ok(guard) => guard,
                        Err(e) => {
                            send_completion_result(&tx, Err(e));
                            return;
                        }
                    },
                    None => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(
                                "llama.cpp: MTMD context missing for multimodal request"
                                    .to_string(),
                            )),
                        );
                        return;
                    }
                };
                let mtmd = &mtmd_guard.0;

                // Build bitmaps from base64-encoded images.
                // Images are never logged — they pass through the privacy boundary here.
                let mut bitmaps: Vec<MtmdBitmap> = Vec::new();
                for b64 in request.images.as_deref().unwrap_or(&[]) {
                    // Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                    let b64_data = b64.find(',').map(|i| &b64[i + 1..]).unwrap_or(b64.as_str());
                    let raw = match base64::Engine::decode(
                        &base64::engine::general_purpose::STANDARD,
                        b64_data,
                    ) {
                        Ok(b) => b,
                        Err(e) => {
                            send_completion_result(
                                &tx,
                                Err(PowerError::InferenceFailed(format!(
                                    "Failed to decode base64 image: {e}"
                                ))),
                            );
                            return;
                        }
                    };
                    match MtmdBitmap::from_buffer(mtmd, &raw, false) {
                        Ok(bm) => bitmaps.push(bm),
                        Err(e) => {
                            send_completion_result(
                                &tx,
                                Err(PowerError::InferenceFailed(format!(
                                    "Failed to create bitmap from image data: {e}"
                                ))),
                            );
                            return;
                        }
                    }
                }

                // Insert media markers into the prompt — one per image.
                let marker = mtmd_default_marker();
                let markers: String = std::iter::repeat_n(marker, bitmaps.len())
                    .collect::<Vec<_>>()
                    .join("\n");
                let prompt_with_markers = format!("{markers}\n{}", request.prompt);

                let input_text = MtmdInputText {
                    text: prompt_with_markers,
                    add_special: true,
                    parse_special: true,
                };

                let bitmap_refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
                let chunks = match mtmd.tokenize(input_text, &bitmap_refs) {
                    Ok(c) => c,
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "MTMD tokenization failed: {e}"
                            ))),
                        );
                        return;
                    }
                };

                let prompt_token_count = chunks.total_tokens() as u32;

                // Create a fresh context for multimodal inference (no KV cache reuse —
                // image embeddings are request-specific and must not leak across sessions).
                let ctx_params =
                    LlamaContextParams::default().with_n_ctx(Some(nonzero_context_size(ctx_size)));
                let mut ctx = match model_arc.new_context(backend_ref(), ctx_params) {
                    Ok(c) => {
                        let c: llama_cpp_2::context::LlamaContext<'static> =
                            unsafe { std::mem::transmute(c) };
                        c
                    }
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "Failed to create MTMD context: {e}"
                            ))),
                        );
                        return;
                    }
                };

                // Evaluate all chunks (text + image embeddings) via the MTMD helper.
                let n_batch = num_batch.unwrap_or(512) as i32;
                let n_past = match chunks.eval_chunks(mtmd, &ctx, 0, 0, n_batch, true) {
                    Ok(n) => n,
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "MTMD eval_chunks failed: {e}"
                            ))),
                        );
                        return;
                    }
                };

                let prompt_eval_duration_ns = prompt_eval_start.elapsed().as_nanos() as u64;

                // Build sampler and generate tokens (same as text path below)
                let mut samplers: Vec<llama_cpp_2::sampling::LlamaSampler> = Vec::new();
                if let Some(temp) = request.temperature {
                    if temp > 0.0 {
                        samplers.push(llama_cpp_2::sampling::LlamaSampler::temp(temp));
                    }
                }
                samplers.push(llama_cpp_2::sampling::LlamaSampler::greedy());
                let mut sampler = llama_cpp_2::sampling::LlamaSampler::chain(samplers, false);

                let eos_token = model_arc.token_eos();
                let mut n_cur = n_past;
                let mut generated_text = String::new();

                for _i in 0..max_tokens {
                    let new_token = sampler.sample(&ctx, -1);
                    if new_token == eos_token {
                        send_completion_result(
                            &tx,
                            Ok(CompletionResponseChunk {
                                text: String::new(),
                                done: true,
                                prompt_tokens: Some(prompt_token_count),
                                done_reason: Some("stop".to_string()),
                                prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                                token_id: None,
                            }),
                        );
                        return;
                    }

                    let text = {
                        let mut decoder = encoding_rs::UTF_8.new_decoder();
                        model_arc
                            .token_to_piece(new_token, &mut decoder, true, None)
                            .unwrap_or_default()
                    };
                    generated_text.push_str(&text);

                    let should_stop = stop_sequences.iter().any(|s| generated_text.ends_with(s));

                    if !send_completion_result(
                        &tx,
                        Ok(CompletionResponseChunk {
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
                        }),
                    ) || should_stop
                    {
                        return;
                    }

                    let mut batch = LlamaBatch::new(1, 1);
                    if batch.add(new_token, n_cur, &[0], true).is_err() {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(
                                "Failed to add generated token to MTMD batch".to_string(),
                            )),
                        );
                        return;
                    }
                    n_cur += 1;
                    if let Err(e) = ctx.decode(&mut batch) {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "MTMD decode failed: {e}"
                            ))),
                        );
                        return;
                    }
                }

                send_completion_result(
                    &tx,
                    Ok(CompletionResponseChunk {
                        text: String::new(),
                        done: true,
                        prompt_tokens: Some(prompt_token_count),
                        done_reason: Some("length".to_string()),
                        prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                        token_id: None,
                    }),
                );
                return;
            }

            // ----------------------------------------------------------------
            // Text-only path (original implementation)
            // ----------------------------------------------------------------

            // Tokenize the prompt
            let tokens =
                match model_arc.str_to_token(&request.prompt, llama_cpp_2::model::AddBos::Always) {
                    Ok(t) => t,
                    Err(e) => {
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(format!(
                                "Tokenization failed: {e}"
                            ))),
                        );
                        return;
                    }
                };

            let prompt_token_count = tokens.len() as u32;

            // Try to reuse cached context with KV cache prefix matching.
            // Only reuse if the request carries a session_id — anonymous requests
            // always get a fresh context to prevent cross-request cache leakage.
            let cached = match session_id.as_deref() {
                Some(sid) => match lock_session_cache(&session_cache) {
                    Ok(mut cache) => cache.remove(sid),
                    Err(e) => {
                        send_completion_result(&tx, Err(e));
                        return;
                    }
                },
                None => None,
            };
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
                    let mut ctx_params = LlamaContextParams::default()
                        .with_n_ctx(Some(nonzero_context_size(ctx_size)));
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
                            send_completion_result(
                                &tx,
                                Err(PowerError::InferenceFailed(format!(
                                    "Failed to create context: {e}"
                                ))),
                            );
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
                let mut wrapper = match lock_lora_adapter(adapter_arc) {
                    Ok(wrapper) => wrapper,
                    Err(e) => {
                        send_completion_result(&tx, Err(e));
                        return;
                    }
                };
                if let Err(e) = ctx.lora_adapter_set(&mut wrapper.0, 1.0) {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!(
                            "Failed to apply LoRA adapter: {e}"
                        ))),
                    );
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
                        send_completion_result(
                            &tx,
                            Err(PowerError::InferenceFailed(
                                "Failed to add token to batch".to_string(),
                            )),
                        );
                        return;
                    }
                }

                if let Err(e) = ctx.decode(&mut batch) {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!("Decode failed: {e}"))),
                    );
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
                    send_completion_result(
                        &tx,
                        Ok(CompletionResponseChunk {
                            text: String::new(),
                            done: true,
                            prompt_tokens: Some(prompt_token_count),
                            done_reason: Some("stop".to_string()),
                            prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                            token_id: None,
                        }),
                    );
                    // Return context to session cache (only when session_id is set).
                    all_tokens.push(new_token);
                    if let Some(ref sid) = session_id {
                        cache_session_context(
                            &session_cache_for_return,
                            sid,
                            CachedContext {
                                ctx,
                                evaluated_tokens: all_tokens,
                                ctx_size,
                                last_used: std::time::Instant::now(),
                            },
                        );
                    }
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

                if !send_completion_result(
                    &tx,
                    Ok(CompletionResponseChunk {
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
                    }),
                ) {
                    // Receiver dropped — cache context if session is set
                    if let Some(ref sid) = session_id {
                        cache_session_context(
                            &session_cache_for_return,
                            sid,
                            CachedContext {
                                ctx,
                                evaluated_tokens: all_tokens,
                                ctx_size,
                                last_used: std::time::Instant::now(),
                            },
                        );
                    }
                    return;
                }

                if should_stop {
                    // Return context to session cache
                    if let Some(ref sid) = session_id {
                        cache_session_context(
                            &session_cache_for_return,
                            sid,
                            CachedContext {
                                ctx,
                                evaluated_tokens: all_tokens,
                                ctx_size,
                                last_used: std::time::Instant::now(),
                            },
                        );
                    }
                    return;
                }

                // Prepare next batch
                let mut batch = LlamaBatch::new(1, 1);
                if batch.add(new_token, n_cur as i32, &[0], true).is_err() {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(
                            "Failed to add generated token to batch".to_string(),
                        )),
                    );
                    return;
                }
                n_cur += 1;

                if let Err(e) = ctx.decode(&mut batch) {
                    send_completion_result(
                        &tx,
                        Err(PowerError::InferenceFailed(format!("Decode failed: {e}"))),
                    );
                    return;
                }
            }

            // Max tokens reached — cache context if session is set
            send_completion_result(
                &tx,
                Ok(CompletionResponseChunk {
                    text: String::new(),
                    done: true,
                    prompt_tokens: Some(prompt_token_count),
                    done_reason: Some("length".to_string()),
                    prompt_eval_duration_ns: Some(prompt_eval_duration_ns),
                    token_id: None,
                }),
            );
            if let Some(ref sid) = session_id {
                cache_session_context(
                    &session_cache_for_return,
                    sid,
                    CachedContext {
                        ctx,
                        evaluated_tokens: all_tokens,
                        ctx_size,
                        last_used: std::time::Instant::now(),
                    },
                );
            }
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
                    session_cache: Arc::new(Mutex::new(HashMap::new())),
                    lora_adapter,
                    projector_path,
                    mtmd_ctx: None, // Embedding models don't use multimodal projectors
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
    use crate::backend::types::{ChatMessage, ContentPart, ImageUrl, MessageContent};
    use crate::backend::Backend;
    use crate::model::manifest::ModelFormat;

    fn test_config() -> Arc<PowerConfig> {
        Arc::new(PowerConfig::default())
    }

    fn test_chat_request() -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("describe this".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
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
            parallel_tool_calls: None,
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
            images: None,
            session_id: None,
        }
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

    #[test]
    fn test_collect_llamacpp_openai_images_accepts_data_uri() {
        let parts = vec![
            ContentPart::Text {
                text: "describe this".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: None,
                },
            },
        ];

        let images = collect_llamacpp_openai_images(0, &parts).unwrap();

        assert_eq!(images, vec!["aGVsbG8=".to_string()]);
    }

    #[test]
    fn test_collect_llamacpp_openai_images_accepts_base64_data() {
        let parts = vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: " aGVsbG8= ".to_string(),
                detail: None,
            },
        }];

        let images = collect_llamacpp_openai_images(1, &parts).unwrap();

        assert_eq!(images, vec!["aGVsbG8=".to_string()]);
    }

    #[test]
    fn test_collect_llamacpp_openai_images_rejects_remote_urls() {
        let parts = vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/image.png".to_string(),
                detail: None,
            },
        }];

        let err = collect_llamacpp_openai_images(2, &parts).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("message 2"), "error: {msg}");
        assert!(msg.contains("part 0"), "error: {msg}");
        assert!(msg.contains("remote image URLs"), "error: {msg}");
    }

    #[test]
    fn test_collect_llamacpp_openai_images_rejects_empty_data() {
        let parts = vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "data:image/png;base64,".to_string(),
                detail: None,
            },
        }];

        let err = collect_llamacpp_openai_images(3, &parts).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("message 3"), "error: {msg}");
        assert!(msg.contains("empty image data"), "error: {msg}");
    }

    #[test]
    fn test_collect_llamacpp_chat_images_combines_supported_sources() {
        let mut request = test_chat_request();
        request.messages[0].images = Some(vec!["message-base64-image".to_string()]);
        request.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe this".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,part-base64-image".to_string(),
                    detail: None,
                },
            },
        ]);
        request.images = Some(vec!["request-base64-image".to_string()]);

        let images = collect_llamacpp_chat_images(&request).unwrap();

        assert_eq!(
            images,
            vec![
                "message-base64-image".to_string(),
                "part-base64-image".to_string(),
                "request-base64-image".to_string(),
            ]
        );
    }

    #[cfg(feature = "llamacpp")]
    #[tokio::test]
    async fn test_effective_prompt_digest_absent_for_llamacpp_images() {
        let backend = LlamaCppBackend::new(test_config());
        let mut request = test_chat_request();
        request.images = Some(vec!["request-base64-image".to_string()]);

        let digest = backend
            .effective_chat_prompt_digest("not-loaded", &request)
            .await
            .unwrap();

        assert!(digest.is_none());
    }

    #[test]
    fn test_ensure_llamacpp_images_supported_allows_text_only_without_projector() {
        assert!(ensure_llamacpp_images_supported("llama3", false, false).is_ok());
    }

    #[test]
    fn test_ensure_llamacpp_images_supported_allows_images_with_projector() {
        assert!(ensure_llamacpp_images_supported("llava", true, true).is_ok());
    }

    #[test]
    fn test_ensure_llamacpp_images_supported_rejects_images_without_projector() {
        let err = ensure_llamacpp_images_supported("llama3", true, false).unwrap_err();

        let msg = err.to_string();
        assert!(msg.contains("llama3"), "error: {msg}");
        assert!(msg.contains("multimodal projector"), "error: {msg}");
        assert!(
            msg.contains("image inputs cannot be processed"),
            "error: {msg}"
        );
    }

    #[cfg(feature = "llamacpp")]
    fn test_completion_chunk(done: bool) -> CompletionResponseChunk {
        CompletionResponseChunk {
            text: String::new(),
            done,
            prompt_tokens: None,
            done_reason: None,
            prompt_eval_duration_ns: None,
            token_id: None,
        }
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_send_completion_result_sends_when_receiver_open() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);

        assert!(send_completion_result(&tx, Ok(test_completion_chunk(true))));

        let sent = rx.blocking_recv().unwrap().unwrap();
        assert!(sent.done);
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_send_completion_result_reports_closed_receiver() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(rx);

        assert!(!send_completion_result(
            &tx,
            Ok(test_completion_chunk(false))
        ));
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_session_cache_lock_poison_returns_error() {
        let cache: SessionCache = Arc::new(Mutex::new(HashMap::new()));
        let poison_cache = Arc::clone(&cache);
        let _ = std::panic::catch_unwind(move || {
            let _guard = poison_cache.lock().unwrap();
            panic!("poison session cache");
        });

        let err = match lock_session_cache(&cache) {
            Ok(_) => panic!("expected poisoned session cache error"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("session cache lock poisoned"));
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_collected_text_lock_recovers_from_poison() {
        let text = Arc::new(Mutex::new(String::from("prefix")));
        let poison_text = Arc::clone(&text);
        let _ = std::panic::catch_unwind(move || {
            let mut guard = poison_text.lock().unwrap();
            guard.push_str("-poisoned");
            panic!("poison collected text");
        });

        {
            let mut guard = lock_collected_text(text.as_ref());
            guard.push_str("-recovered");
        }

        assert_eq!(
            lock_collected_text(text.as_ref()).as_str(),
            "prefix-poisoned-recovered"
        );
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_nonzero_context_size_preserves_valid_value() {
        assert_eq!(nonzero_context_size(4096).get(), 4096);
    }

    #[cfg(feature = "llamacpp")]
    #[test]
    fn test_nonzero_context_size_falls_back_for_zero() {
        assert_eq!(nonzero_context_size(0).get(), DEFAULT_CTX_SIZE);
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
            parallel_tool_calls: None,
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
            images: None,
            session_id: None,
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
            num_parallel: None,
            suffix: None,
            context: None,
            session_id: None,
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
