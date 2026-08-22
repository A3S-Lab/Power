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
use std::ptr::NonNull;
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
use super::prompt_cache::{
    BoundedPromptCache, PromptCacheMetricsSnapshot, PromptCacheSupport, PromptCacheTelemetry,
};
#[cfg(feature = "llamacpp")]
use super::types::EffectivePromptDigest;
use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use super::Backend;

#[cfg(feature = "llamacpp")]
mod chat_runtime;
#[cfg(feature = "llamacpp")]
mod completion;
#[cfg(feature = "llamacpp")]
mod embedding;
#[cfg(feature = "llamacpp")]
mod external_draft;
#[cfg(feature = "llamacpp")]
mod model_loading;
#[cfg(feature = "llamacpp")]
mod speculative_runtime;
#[cfg(feature = "llamacpp")]
use external_draft::{
    external_draft_strategy, loads_mtp_weights as llamacpp_loads_mtp_weights,
    selected_external_draft, LoadedExternalDraft,
};
#[cfg(feature = "llamacpp")]
use speculative_runtime::{
    build_llamacpp_sampler, ensure_mtp_fr_available, llamacpp_speculative_capabilities,
    run_external_draft_completion, run_mtp_completion, sample_target_token, use_backend_greedy,
    LlamaContextSettings, LlamaSamplingSettings, MtpCompletionSettings, SpeculativeTelemetry,
};

/// Default context size when `num_ctx` is not specified by the user.
///
/// Matches Ollama's default. Using the model's full `n_ctx_train` (e.g. 128K for
/// llama3.2) would allocate a massive KV cache that can OOM on machines with
/// limited memory. Users can override with `--num-ctx` or the `num_ctx` API field.
#[allow(dead_code)]
const DEFAULT_CTX_SIZE: u32 = 2048;

/// Translate Power's GPU layer convention to llama-cpp-2's unsigned setter.
///
/// Power exposes `-1` as "offload every layer" while llama-cpp-2 represents
/// that request by saturating `u32::MAX` to `i32::MAX`. Passing zero explicitly
/// is important because llama.cpp's default model parameters offload all layers.
#[cfg(feature = "llamacpp")]
fn llamacpp_gpu_layers(gpu_layers: i32) -> Result<u32> {
    match gpu_layers {
        -1 => Ok(u32::MAX),
        0.. => Ok(gpu_layers as u32),
        _ => Err(PowerError::Config(format!(
            "gpu.gpu_layers must be -1 or a non-negative integer, got {gpu_layers}"
        ))),
    }
}

#[cfg(feature = "llamacpp")]
fn validated_llamacpp_batch(num_batch: Option<u32>, ctx_size: u32) -> Result<Option<u32>> {
    if let Some(batch) = num_batch {
        if batch == 0 {
            return Err(PowerError::InvalidRequest(
                "llama.cpp num_batch must be greater than zero".to_string(),
            ));
        }
        if batch > ctx_size {
            return Err(PowerError::InvalidRequest(format!(
                "llama.cpp num_batch ({batch}) must not exceed the effective context size ({ctx_size})"
            )));
        }
    }
    Ok(num_batch)
}

#[cfg(feature = "llamacpp")]
fn llamacpp_model_params(
    gpu_layers: u32,
    main_gpu: i32,
    use_mmap: bool,
    use_mlock: bool,
    has_tensor_split: bool,
    load_mtp: bool,
) -> llama_cpp_sys_2::llama_model_params {
    let mut params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
    params.n_gpu_layers = i32::try_from(gpu_layers).unwrap_or(i32::MAX);
    params.main_gpu = main_gpu;
    params.load_mode = match (use_mmap, use_mlock) {
        (false, false) => llama_cpp_sys_2::LLAMA_LOAD_MODE_NONE,
        (true, false) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP,
        (false, true) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MLOCK,
        (true, true) => llama_cpp_sys_2::LLAMA_LOAD_MODE_MMAP_MLOCK,
    };
    if has_tensor_split {
        params.split_mode = llama_cpp_sys_2::LLAMA_SPLIT_MODE_LAYER;
    }
    params.load_mtp = load_mtp;
    params
}

/// Build an anchored C++ regular expression for one exact GGUF tensor name.
#[cfg(feature = "llamacpp")]
fn exact_tensor_pattern(name: &str) -> String {
    let mut pattern = String::with_capacity(name.len().saturating_mul(2).saturating_add(2));
    pattern.push('^');
    for character in name.chars() {
        if matches!(
            character,
            '\\' | '.' | '^' | '$' | '|' | '?' | '*' | '+' | '(' | ')' | '[' | ']' | '{' | '}'
        ) {
            pattern.push('\\');
        }
        pattern.push(character);
    }
    pattern.push('$');
    pattern
}

/// Own the C strings and null-terminated raw override array for one model load.
///
/// llama.cpp only borrows these pointers while `llama_load_model_from_file`
/// runs, so this owner is kept in the same blocking closure until load returns.
#[cfg(feature = "llamacpp")]
struct LlamaTensorOverrides {
    _patterns: Vec<std::ffi::CString>,
    entries: Vec<llama_cpp_sys_2::llama_model_tensor_buft_override>,
}

#[cfg(feature = "llamacpp")]
impl LlamaTensorOverrides {
    fn new(cpu_names: &[String], gpu_names: &[String], main_gpu: i32) -> Result<Self> {
        let cpu_buffer_type = if cpu_names.is_empty() {
            None
        } else {
            let buffer_type = unsafe { llama_cpp_sys_2::ggml_backend_cpu_buffer_type() };
            if buffer_type.is_null() {
                return Err(PowerError::InferenceFailed(
                    "llama.cpp CPU buffer type is unavailable".to_string(),
                ));
            }
            Some(buffer_type)
        };
        let gpu_buffer_type = if gpu_names.is_empty() {
            None
        } else {
            Some(llamacpp_primary_gpu_buffer_type(main_gpu)?)
        };

        let mut names_and_buffers = Vec::with_capacity(cpu_names.len() + gpu_names.len());
        if let Some(buffer_type) = cpu_buffer_type {
            names_and_buffers.extend(
                cpu_names
                    .iter()
                    .map(|name| (name, buffer_type, "gpu.cpu_tensors")),
            );
        }
        if let Some(buffer_type) = gpu_buffer_type {
            names_and_buffers.extend(
                gpu_names
                    .iter()
                    .map(|name| (name, buffer_type, "gpu.gpu_tensors")),
            );
        }

        let patterns = names_and_buffers
            .iter()
            .map(|(name, _, field)| {
                std::ffi::CString::new(exact_tensor_pattern(name)).map_err(|error| {
                    PowerError::Config(format!(
                        "{field} contains an invalid tensor name {name:?}: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut entries = patterns
            .iter()
            .zip(names_and_buffers.iter())
            .map(|(pattern, (_, buffer_type, _))| {
                llama_cpp_sys_2::llama_model_tensor_buft_override {
                    pattern: pattern.as_ptr(),
                    buft: *buffer_type,
                }
            })
            .collect::<Vec<_>>();
        entries.push(llama_cpp_sys_2::llama_model_tensor_buft_override {
            pattern: std::ptr::null(),
            buft: std::ptr::null_mut(),
        });

        Ok(Self {
            _patterns: patterns,
            entries,
        })
    }

    fn apply(&self, params: &mut llama_cpp_sys_2::llama_model_params) {
        params.tensor_buft_overrides = self.entries.as_ptr();
    }
}

#[cfg(feature = "llamacpp")]
fn llamacpp_primary_gpu_buffer_type(
    main_gpu: i32,
) -> Result<llama_cpp_sys_2::ggml_backend_buffer_type_t> {
    let main_gpu = usize::try_from(main_gpu).map_err(|_| {
        PowerError::Config("gpu.main_gpu must be non-negative for gpu_tensors".to_string())
    })?;
    let mut discrete = Vec::new();
    let mut integrated = Vec::new();
    let count = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
    for index in 0..count {
        let device = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(index) };
        if device.is_null() {
            continue;
        }
        match unsafe { llama_cpp_sys_2::ggml_backend_dev_type(device) } {
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU => discrete.push(device),
            llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_IGPU => integrated.push(device),
            _ => {}
        }
    }
    let devices = if discrete.is_empty() {
        &integrated
    } else {
        &discrete
    };
    let device = devices.get(main_gpu).copied().ok_or_else(|| {
        PowerError::Config(format!(
            "gpu.main_gpu index {main_gpu} is unavailable for gpu_tensors (detected {} compatible GPU device(s))",
            devices.len()
        ))
    })?;
    let buffer_type = unsafe { llama_cpp_sys_2::ggml_backend_dev_buffer_type(device) };
    if buffer_type.is_null() {
        return Err(PowerError::InferenceFailed(format!(
            "llama.cpp GPU device {main_gpu} has no default buffer type"
        )));
    }
    Ok(buffer_type)
}

/// Load a model with the reviewed raw `load_mtp` flag that llama-cpp-2 does
/// not currently expose in its safe parameter builder.
#[cfg(feature = "llamacpp")]
fn load_llamacpp_model(
    path: &std::path::Path,
    params: llama_cpp_sys_2::llama_model_params,
) -> Result<llama_cpp_2::model::LlamaModel> {
    let path_str = path.to_str().ok_or_else(|| {
        PowerError::InferenceFailed(format!("Model path is not valid UTF-8: {}", path.display()))
    })?;
    let path_c = std::ffi::CString::new(path_str).map_err(|error| {
        PowerError::InferenceFailed(format!(
            "Model path contains an interior NUL byte ({}): {error}",
            path.display()
        ))
    })?;
    let raw = unsafe { llama_cpp_sys_2::llama_load_model_from_file(path_c.as_ptr(), params) };
    let raw = NonNull::new(raw).ok_or_else(|| {
        PowerError::InferenceFailed(format!("Failed to load model: {}", path.display()))
    })?;

    // Safety: at the pinned llama-cpp-rs revision, LlamaModel is
    // `#[repr(transparent)]` over this exact NonNull<llama_model>. Ownership of
    // the successful raw load transfers to LlamaModel, whose Drop calls the
    // matching llama_model_free function from the same sys crate revision.
    Ok(unsafe {
        std::mem::transmute::<NonNull<llama_cpp_sys_2::llama_model>, llama_cpp_2::model::LlamaModel>(
            raw,
        )
    })
}

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
    /// Speculative strategies supported by this loaded model and backend.
    speculative_capabilities: crate::speculative::SpeculativeCapabilities,
    /// Verified external draft model, loaded only when selected by policy.
    external_draft: Option<LoadedExternalDraft>,
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
}

#[cfg(feature = "llamacpp")]
type SessionCache = Arc<Mutex<BoundedPromptCache<CachedContext>>>;

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
) -> Result<MutexGuard<'_, BoundedPromptCache<CachedContext>>> {
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
                error = %e,
                "llama.cpp: failed to return context to session cache"
            );
        }
    }
}

#[cfg(feature = "llamacpp")]
fn cache_prompt_boundary_context(
    cache: &SessionCache,
    session_id: &str,
    mut ctx: llama_cpp_2::context::LlamaContext<'static>,
    mut prompt_tokens: Vec<llama_cpp_2::token::LlamaToken>,
    ctx_size: u32,
    checkpoint: Option<&llama_cpp_2::SeqState>,
) {
    let restored = match checkpoint {
        Some(checkpoint) => match ctx.state_seq_set(checkpoint, 0) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "llama.cpp: failed to restore prompt-boundary recurrent state; clearing cached context"
                );
                false
            }
        },
        None => true,
    };
    let truncated = u32::try_from(prompt_tokens.len())
        .ok()
        .filter(|_| restored)
        .map(|prompt_len| ctx.clear_kv_cache_seq(Some(0), Some(prompt_len), None));
    if !matches!(&truncated, Some(Ok(true))) {
        tracing::debug!(
            result = ?truncated,
            "llama.cpp: prompt-boundary normalization unavailable; retaining an allocated cache miss"
        );
        ctx.clear_kv_cache();
        prompt_tokens.clear();
    }
    cache_session_context(
        cache,
        session_id,
        CachedContext {
            ctx,
            evaluated_tokens: prompt_tokens,
            ctx_size,
        },
    );
}

/// Return the prefix that may stay resident while still producing fresh logits.
///
/// If the entire new prompt matches, llama.cpp must decode its final token
/// again. KV truncation alone does not restore the logits row for that token;
/// the context may still expose logits from a generated suffix.
#[cfg(feature = "llamacpp")]
fn matched_and_reusable_prompt_prefix_len<T: PartialEq>(
    cached: &[T],
    prompt: &[T],
) -> (usize, usize) {
    let common_len = cached
        .iter()
        .zip(prompt.iter())
        .take_while(|(left, right)| left == right)
        .count();
    let reusable_len = if common_len == prompt.len() {
        common_len.saturating_sub(1)
    } else {
        common_len
    };
    (common_len, reusable_len)
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
            super::types::ContentPart::ImageUrl { image_url, .. } => Some(
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
    #[cfg(feature = "llamacpp")]
    prompt_cache_telemetry: Arc<PromptCacheTelemetry>,
    #[cfg(feature = "llamacpp")]
    speculative_telemetry: Arc<SpeculativeTelemetry>,
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
            #[cfg(feature = "llamacpp")]
            prompt_cache_telemetry: Arc::new(PromptCacheTelemetry::default()),
            #[cfg(feature = "llamacpp")]
            speculative_telemetry: Arc::new(SpeculativeTelemetry::default()),
            config,
        }
    }

    #[cfg(feature = "llamacpp")]
    fn new_session_cache(&self) -> SessionCache {
        Arc::new(Mutex::new(BoundedPromptCache::new(
            self.config.prompt_cache_max_entries,
            std::time::Duration::from_secs(self.config.prompt_cache_ttl_seconds),
            self.prompt_cache_telemetry.clone(),
        )))
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

    fn prompt_cache_support(&self) -> PromptCacheSupport {
        PromptCacheSupport::PrefixMatch
    }

    fn prompt_cache_metrics(&self) -> Option<PromptCacheMetricsSnapshot> {
        Some(self.prompt_cache_telemetry.snapshot())
    }

    fn speculative_metrics(&self) -> Vec<crate::backend::SpeculativeMetricsSnapshot> {
        self.speculative_telemetry.snapshot()
    }

    async fn loaded_speculative_artifacts(&self) -> Vec<crate::backend::LoadedSpeculativeArtifact> {
        self.models
            .read()
            .await
            .iter()
            .filter_map(|(model_name, loaded)| {
                loaded.external_draft.as_ref().map(|draft| {
                    crate::backend::LoadedSpeculativeArtifact {
                        backend: self.name().to_string(),
                        model: model_name.clone(),
                        strategy: draft.identity.kind.as_str().to_string(),
                        artifact_size: draft.identity.size,
                        artifact_sha256: draft.identity.sha256.clone(),
                        target_sha256: draft.identity.target_sha256.clone(),
                    }
                })
            })
            .collect()
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        model_loading::load(self, manifest).await
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        model_loading::unload(self, model_name).await
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        chat_runtime::chat(self, model_name, request).await
    }

    async fn effective_chat_prompt_digest(
        &self,
        model_name: &str,
        request: &ChatRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        chat_runtime::effective_chat_prompt_digest(self, model_name, request).await
    }

    async fn effective_completion_prompt_digest(
        &self,
        model_name: &str,
        request: &CompletionRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        chat_runtime::effective_completion_prompt_digest(self, model_name, request).await
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        completion::complete(self, model_name, request).await
    }

    async fn embed(
        &self,
        model_name: &str,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        embedding::embed(self, model_name, request).await
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
mod tests;
