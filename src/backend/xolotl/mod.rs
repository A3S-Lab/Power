//! Xolotl backend for the DeepSeek-V4.1-Flash-1017 XQ pack.
//!
//! Two placements are admitted. Eight H200s use
//! `--layer-split 5,10,15,20,25,30,35 --split-device 0,1,2,3,4,5,6,7`.
//! One GeForce RTX 4090 uses the single-card expert cache (`--experts gpu`
//! plus `--rank`) and emits no layer split. `ds41_generate` rejects `--gpus`.
//! Omitting `--split-device` on a split keeps at most two visible CUDA devices,
//! and `--layer-split auto` is one cut at `n_layers / 2`.
//!
//! With an explicit ACL `managed` block, Power owns the packaged adapter and
//! real resident engine as one process group, proves READY and model identity,
//! and reports failed readiness if either process is lost. The external
//! upstream mode remains available when the block is absent.

mod detect;
mod managed;
mod plan;
mod prepare;
pub(crate) use prepare::prepare_managed_model;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use futures::{Stream, TryStreamExt};

use crate::config::{PowerConfig, XolotlConfig};
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
};
use super::{ensure_non_speculative_mode, Backend};

pub use detect::{is_xolotl_required_manifest, is_xolotl_xq_path};
pub use plan::{
    parse_nvidia_smi_csv, plan_from_nvidia_smi_csv, query_nvidia_smi, split_log_is_ready,
    CudaDevice, XolotlLaunchPlan,
};

/// Front for one XQ pack, with optional ownership of its resident process.
pub struct XolotlBackend {
    config: Arc<PowerConfig>,
    http: reqwest::Client,
    /// Injected inventory. Production load calls `nvidia-smi` when this is empty.
    devices: Option<Vec<CudaDevice>>,
    loaded: tokio::sync::RwLock<HashMap<String, XolotlLaunchPlan>>,
    engine: tokio::sync::Mutex<Option<managed::ManagedEngine>>,
}

/// Bounded deadline for cold expert-cache inference and million-token prefill.
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(2 * 60 * 60);

fn upstream_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(UPSTREAM_TIMEOUT)
        .build()
        .expect("xolotl upstream HTTP client")
}

impl XolotlBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            config,
            http: upstream_client(),
            devices: None,
            loaded: tokio::sync::RwLock::new(HashMap::new()),
            engine: tokio::sync::Mutex::new(None),
        }
    }

    /// Use a fixture inventory so tests never call `nvidia-smi`.
    pub fn with_devices(mut self, devices: Vec<CudaDevice>) -> Self {
        self.devices = Some(devices);
        self
    }

    async fn loaded_plan(&self, model_name: &str) -> Result<XolotlLaunchPlan> {
        let plan = self
            .loaded
            .read()
            .await
            .get(model_name)
            .cloned()
            .ok_or_else(|| {
                PowerError::ModelNotFound(format!("xolotl model '{model_name}' is not loaded"))
            })?;
        if self.config.xolotl.managed.is_some() {
            self.engine
                .lock()
                .await
                .as_mut()
                .ok_or_else(|| {
                    PowerError::InferenceFailed("managed Xolotl engine is absent".into())
                })?
                .ensure_alive()?;
        }
        Ok(plan)
    }

    async fn post_json(&self, url: &str, body: &serde_json::Value) -> Result<serde_json::Value> {
        let response = self
            .http
            .post(url)
            .json(body)
            .send()
            .await
            .map_err(|error| {
                PowerError::InferenceFailed(format!("xolotl request to {url} failed: {error}"))
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| {
            PowerError::InferenceFailed(format!(
                "xolotl response from {url} could not be read: {error}"
            ))
        })?;
        if !status.is_success() {
            return Err(PowerError::InferenceFailed(format!(
                "xolotl upstream {url} returned {status}: {}",
                bounded_text(&text)
            )));
        }
        serde_json::from_str(&text).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "xolotl upstream {url} returned non-JSON: {error}: {}",
                bounded_text(&text)
            ))
        })
    }
}

fn require_upstream(config: &XolotlConfig) -> Result<String> {
    let Some(raw) = config
        .upstream
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Err(PowerError::Config("xolotl.upstream is unset".into()));
    };
    let url = raw.trim_end_matches('/').to_string();
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(PowerError::Config(format!(
            "xolotl.upstream must be an http(s) origin for serve_ds41.py, got {url}"
        )));
    }
    if url.ends_with("/v1") {
        return Err(PowerError::Config(
            "xolotl.upstream is the serve_ds41.py origin, for example http://127.0.0.1:8150, without a /v1 suffix"
                .into(),
        ));
    }
    Ok(url)
}

fn upstream_error(plan: &XolotlLaunchPlan, error: PowerError) -> PowerError {
    if plan.argv.iter().any(|arg| arg == "--layer-split") {
        return PowerError::InferenceFailed(format!(
            "{error}. Power does not start ds41_generate. Start `{}`, wait until the log contains `{}`, then set xolotl.upstream to the serve_ds41.py origin.",
            plan.command_line(),
            plan.ready_log_line
        ));
    }
    PowerError::InferenceFailed(format!(
        "{error}. Power does not start ds41_generate. Start `{}`. The RTX 4090 placement has no layer split. Set xolotl.upstream to the serve_ds41.py origin.",
        plan.command_line()
    ))
}

fn chat_body(model_name: &str, request: &ChatRequest) -> Result<serde_json::Value> {
    if request.has_image_inputs() {
        return Err(PowerError::InferenceFailed(
            "xolotl chat refuses image inputs".into(),
        ));
    }
    crate::backend::openai_compat::chat_request_body(model_name, request, true)
}

fn completion_body(model_name: &str, request: &CompletionRequest) -> Result<serde_json::Value> {
    if request
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty())
    {
        return Err(PowerError::InferenceFailed(
            "xolotl completion refuses image inputs".into(),
        ));
    }
    let mut body = serde_json::json!({
        "model": model_name,
        "prompt": request.prompt,
        "stream": false,
    });
    copy_sampling(
        &mut body,
        request.temperature,
        request.top_p,
        request.max_tokens,
        request.stop.as_ref(),
    );
    Ok(body)
}

fn copy_sampling(
    body: &mut serde_json::Value,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stop: Option<&Vec<String>>,
) {
    if let Some(temperature) = temperature {
        body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = top_p {
        body["top_p"] = serde_json::json!(top_p);
    }
    if let Some(max_tokens) = max_tokens {
        body["max_tokens"] = serde_json::json!(max_tokens);
    }
    if let Some(stop) = stop {
        body["stop"] = serde_json::json!(stop);
    }
}

fn string_field(json: &serde_json::Value, pointer: &str, label: &str) -> Result<String> {
    match json.pointer(pointer) {
        Some(serde_json::Value::String(text)) => Ok(text.clone()),
        Some(serde_json::Value::Null) => Ok(String::new()),
        Some(_) => Err(PowerError::InferenceFailed(format!(
            "xolotl upstream {label} is not a string"
        ))),
        None => Err(PowerError::InferenceFailed(format!(
            "xolotl upstream response has no {label}"
        ))),
    }
}

fn finish_reason(json: &serde_json::Value) -> String {
    json.pointer("/choices/0/finish_reason")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("stop")
        .to_string()
}

fn prompt_tokens(json: &serde_json::Value) -> Option<u32> {
    json.pointer("/usage/prompt_tokens")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn completion_stream(
    text: String,
    prompt_tokens: Option<u32>,
    done_reason: String,
) -> Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>> {
    let mut chunks = Vec::new();
    if !text.is_empty() {
        chunks.push(Ok(CompletionResponseChunk {
            text,
            done: false,
            prompt_tokens: None,
            done_reason: None,
            prompt_eval_duration_ns: None,
            token_id: None,
        }));
    }
    chunks.push(Ok(CompletionResponseChunk {
        text: String::new(),
        done: true,
        prompt_tokens,
        done_reason: Some(done_reason),
        prompt_eval_duration_ns: None,
        token_id: None,
    }));
    Box::pin(stream::iter(chunks))
}

fn bounded_text(text: &str) -> &str {
    let mut end = text.len().min(512);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[async_trait]
impl Backend for XolotlBackend {
    fn supports_openai_tool_selection(&self) -> bool {
        true
    }

    fn name(&self) -> &str {
        "xolotl"
    }

    fn reports_exact_chat_usage(&self) -> bool {
        true
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        let _ = format;
        false
    }

    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        is_xolotl_required_manifest(manifest)
    }

    async fn runtime_readiness(&self) -> Result<()> {
        if self.config.xolotl.managed.is_none() {
            return Ok(());
        }
        // Health probes share this owner with each other and with lifecycle
        // operations. Contention alone does not mean a resident model is lost.
        // Bound the complete wait and HTTP proof so startup/shutdown still fail
        // closed without leaving health requests queued for the load deadline.
        tokio::time::timeout(Duration::from_secs(2), async {
            let mut owner = self.engine.lock().await;
            owner
                .as_mut()
                .ok_or_else(|| {
                    PowerError::InferenceFailed("managed Xolotl model is not loaded".into())
                })?
                .probe(&self.http)
                .await
        })
        .await
        .map_err(|_| {
            PowerError::InferenceFailed("managed Xolotl readiness deadline exceeded".into())
        })?
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name())?;
        if !is_xolotl_required_manifest(manifest) {
            return Err(PowerError::Config(format!(
                "xolotl backend refuses non-XQ pack '{}'",
                manifest.path.display()
            )));
        }
        let devices = match &self.devices {
            Some(devices) => devices.clone(),
            None => tokio::task::spawn_blocking(query_nvidia_smi)
                .await
                .map_err(|error| {
                    PowerError::Config(format!("nvidia-smi task failed: {error}"))
                })??,
        };
        let launch = plan::plan(&self.config, manifest, &devices)?;
        let launch = tokio::task::spawn_blocking(move || {
            plan::ensure_artifacts_exist(&launch)?;
            Ok::<_, PowerError>(launch)
        })
        .await
        .map_err(|error| PowerError::Config(format!("xolotl artifact check failed: {error}")))??;
        let mut loaded = self.loaded.write().await;
        if let Some(managed) = &self.config.xolotl.managed {
            let mut owner = self.engine.lock().await;
            if !loaded.is_empty() {
                if !loaded.contains_key(&manifest.name) {
                    return Err(PowerError::InferenceFailed(
                        "managed Xolotl already owns another model".into(),
                    ));
                }
                return owner
                    .as_mut()
                    .ok_or_else(|| {
                        PowerError::InferenceFailed("managed Xolotl lost its owner".into())
                    })?
                    .probe(&self.http)
                    .await;
            }
            if owner.is_some() {
                return Err(PowerError::InferenceFailed(
                    "managed Xolotl retained an unregistered owner".into(),
                ));
            }
            let origin = require_upstream(&self.config.xolotl)?;
            *owner = Some(
                managed::ManagedEngine::start(
                    managed,
                    &launch,
                    &origin,
                    &manifest.name,
                    &self.http,
                )
                .await?,
            );
        }
        loaded.insert(manifest.name.clone(), launch);
        Ok(())
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        let mut loaded = self.loaded.write().await;
        if self.config.xolotl.managed.is_some() && loaded.contains_key(model_name) {
            let mut owner = self.engine.lock().await;
            if let Some(engine) = owner.as_mut() {
                engine.stop().await?;
            }
            *owner = None;
        }
        loaded.remove(model_name);
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name())?;
        let launch = self.loaded_plan(model_name).await?;
        let tool_policy = crate::backend::openai_compat::ChatToolPolicy::from_request(&request)?;
        let body = chat_body(model_name, &request)?;
        let upstream = require_upstream(&self.config.xolotl)
            .map_err(|error| upstream_error(&launch, error))?;
        let http = self.http.clone();
        let url = format!("{upstream}/v1/chat/completions");
        // The stream owns startup and the response: dropping it cancels reads
        // during header waits, million-token prefill, and token generation.
        let startup = async move {
            let response = http
                .post(&url)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|e| PowerError::InferenceFailed(format!("xolotl stream failed: {e}")))?;
            let response = crate::backend::openai_compat::require_chat_stream(response)?;
            Ok::<_, PowerError>(crate::backend::openai_compat::chat_response_stream(
                response,
                true,
                tool_policy,
            ))
        };
        Ok(Box::pin(stream::once(startup).try_flatten()))
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name())?;
        let launch = self.loaded_plan(model_name).await?;
        let body = completion_body(model_name, &request)?;
        let upstream = require_upstream(&self.config.xolotl)
            .map_err(|error| upstream_error(&launch, error))?;
        let json = self
            .post_json(&format!("{upstream}/v1/completions"), &body)
            .await?;
        let text = string_field(&json, "/choices/0/text", "completion text")?;
        Ok(completion_stream(
            text,
            prompt_tokens(&json),
            finish_reason(&json),
        ))
    }

    async fn embed(
        &self,
        model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(format!(
            "xolotl backend does not support embeddings (model '{model_name}')"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use axum::extract::{Json, State};
    use axum::routing::post;
    use axum::Router;
    use futures::StreamExt;
    use serde_json::{json, Value};

    use crate::backend::test_utils::sample_manifest;

    fn h200(index: u32) -> CudaDevice {
        CudaDevice {
            index,
            name: "NVIDIA H200".into(),
            compute_major: 9,
            compute_minor: 0,
            memory_bytes: 143_771 * 1024 * 1024,
        }
    }

    fn eight_h200() -> Vec<CudaDevice> {
        (0..8).map(h200).collect()
    }

    fn binary_name() -> &'static str {
        if cfg!(windows) {
            "ds41_generate.exe"
        } else {
            "ds41_generate"
        }
    }

    fn engram_list(root: &Path) -> String {
        let left = root.join("engram-a");
        let right = root.join("engram-b");
        if cfg!(windows) {
            format!("{};{}", left.display(), right.display())
        } else {
            format!("{}:{}", left.display(), right.display())
        }
    }

    fn enabled_config(root: &Path) -> PowerConfig {
        let mut config = PowerConfig::default();
        config.xolotl.enabled = true;
        config.xolotl.bin = Some(root.join(binary_name()).display().to_string());
        config.xolotl.model = Some(
            root.join("DeepSeek-V4.1-Flash-1017-XQ.gguf")
                .display()
                .to_string(),
        );
        config.xolotl.engram = Some(engram_list(root));
        config.xolotl.rank = Some(root.join("rank-1017-085.bin").display().to_string());
        config
    }

    fn write_artifacts(root: &Path) {
        std::fs::write(root.join(binary_name()), b"not-a-gpu-binary").unwrap();
        std::fs::write(root.join("DeepSeek-V4.1-Flash-1017-XQ.gguf"), b"gguf").unwrap();
        std::fs::create_dir(root.join("engram-a")).unwrap();
        std::fs::create_dir(root.join("engram-b")).unwrap();
        std::fs::write(root.join("rank-1017-085.bin"), b"rank").unwrap();
    }

    fn manifest_for(config: &PowerConfig) -> ModelManifest {
        let mut manifest = sample_manifest("xq-1017");
        manifest.path = PathBuf::from(config.xolotl.model.clone().unwrap());
        manifest
    }

    fn error_text<T>(result: Result<T>) -> String {
        match result {
            Err(error) => error.to_string(),
            Ok(_) => panic!("expected the xolotl call to fail"),
        }
    }

    fn chat_request() -> ChatRequest {
        serde_json::from_value(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 256
        }))
        .unwrap()
    }

    #[test]
    fn claims_xq_manifests_without_claiming_every_gguf() {
        let backend = XolotlBackend::new(Arc::new(PowerConfig::default()));
        assert!(!backend.supports(&ModelFormat::Gguf));
        let mut xq = sample_manifest("xq");
        xq.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-XQ-Uncensored.gguf");
        assert!(backend.supports_manifest(&xq));
        let ordinary = sample_manifest("qwen");
        assert!(!backend.supports_manifest(&ordinary));

        let mut unsloth = sample_manifest("ud");
        unsloth.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-UD-IQ1_M.gguf");
        assert!(!backend.supports_manifest(&unsloth));
    }

    #[test]
    fn llama_mistral_and_picolm_refuse_the_xq_manifest() {
        let config = Arc::new(PowerConfig::default());
        let mut xq = sample_manifest("xq-1017");
        xq.format = crate::model::manifest::ModelFormat::Gguf;
        xq.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-XQ-Uncensored.gguf");
        let llama = crate::backend::llamacpp::LlamaCppBackend::new(config.clone());
        let mistral = crate::backend::mistralrs_backend::MistralRsBackend::new(config.clone());
        let picolm = crate::backend::picolm::PicolmBackend::new(config);
        assert!(!llama.supports_manifest(&xq));
        assert!(!mistral.supports_manifest(&xq));
        assert!(!picolm.supports_manifest(&xq));
    }

    #[tokio::test]
    async fn disabled_and_speculative_loads_fail_closed() {
        let manifest = {
            let mut manifest = sample_manifest("xq-1017");
            manifest.path = PathBuf::from("DeepSeek-V4.1-Flash-1017-XQ.gguf");
            manifest
        };
        let disabled =
            XolotlBackend::new(Arc::new(PowerConfig::default())).with_devices(eight_h200());
        let message = disabled.load(&manifest).await.unwrap_err().to_string();
        assert!(message.contains("xolotl.enabled"));

        let mut speculative = PowerConfig::default();
        speculative.spec_mode = "mtp".into();
        speculative.xolotl.enabled = true;
        let backend = XolotlBackend::new(Arc::new(speculative)).with_devices(eight_h200());
        let message = backend.load(&manifest).await.unwrap_err().to_string();
        assert!(message.contains("mtp"));
    }

    #[tokio::test]
    async fn chat_without_upstream_returns_the_eight_card_command() {
        let root = tempfile::tempdir().unwrap();
        write_artifacts(root.path());
        let config = enabled_config(root.path());
        let manifest = manifest_for(&config);
        let backend = XolotlBackend::new(Arc::new(config)).with_devices(eight_h200());
        backend.load(&manifest).await.unwrap();

        let digest = backend
            .effective_chat_prompt_digest("xq-1017", &chat_request())
            .await
            .unwrap();
        assert!(digest.is_none());

        let message = error_text(backend.chat("xq-1017", chat_request()).await);
        assert!(message.contains("CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"));
        assert!(message.contains("--layer-split 5,10,15,20,25,30,35"));
        assert!(message.contains("--split-device 0,1,2,3,4,5,6,7"));
        assert!(message.contains(
            "ds41: 7-way layer split at 5 10 15 20 25 30 35 across GPUs 0 1 2 3 4 5 6 7"
        ));
        assert!(message.contains("serve_ds41.py"));
        assert!(!message.contains("--gpus"));
        assert!(!message.contains("--split-skip-if-fits"));
        assert!(!message.contains("STRATA_DSPARK"));

        backend.unload("xq-1017").await.unwrap();
        let missing = error_text(backend.chat("xq-1017", chat_request()).await);
        assert!(missing.contains("not loaded"));
    }

    #[tokio::test]
    async fn rtx_4090_load_records_the_single_card_command() {
        let root = tempfile::tempdir().unwrap();
        write_artifacts(root.path());
        let mut config = enabled_config(root.path());
        config.xolotl.gpus = vec![0];
        config.xolotl.cuda_architecture = "8.9".into();
        config.xolotl.layer_split = "none".into();
        let manifest = manifest_for(&config);
        let device = CudaDevice {
            index: 0,
            name: "NVIDIA GeForce RTX 4090".into(),
            compute_major: 8,
            compute_minor: 9,
            memory_bytes: 24_564 * 1024 * 1024,
        };
        let backend = XolotlBackend::new(Arc::new(config)).with_devices(vec![device]);
        backend.load(&manifest).await.unwrap();
        let message = error_text(backend.chat("xq-1017", chat_request()).await);
        assert!(message.contains("CUDA_VISIBLE_DEVICES=0 "));
        assert!(message.contains("--backend cuda"));
        assert!(message.contains("--experts gpu"));
        assert!(message.contains("--engram"));
        assert!(message.contains("rank-1017"));
        assert!(message.contains("RTX 4090 placement has no layer split"));
        assert!(!message.contains("--layer-split"));
        assert!(!message.contains("--split-device"));
        assert!(!message.contains("--gpus"));
        assert!(!message.contains("STRATA_DSPARK"));
    }

    #[derive(Clone)]
    struct UpstreamState {
        chat: Arc<Mutex<Option<Value>>>,
        completion: Arc<Mutex<Option<Value>>>,
    }

    async fn chat_handler(
        State(state): State<UpstreamState>,
        Json(body): Json<Value>,
    ) -> axum::response::Response {
        use axum::response::IntoResponse;
        *state.chat.lock().expect("chat body lock") = Some(body);
        let events = [
            json!({"choices":[{"index":0,"delta":{"content":"ready","reasoning_content":"measured reasoning"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":4,"completion_tokens":17,"total_tokens":21},
                "timings":{"prompt_n":4,"prompt_ms":2.0,"predicted_n":17,"predicted_ms":200.0,"predicted_per_second":85.0}}),
        ];
        let mut wire = events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect::<String>();
        wire.push_str("data: [DONE]\n\n");
        ([("content-type", "text/event-stream")], wire).into_response()
    }

    async fn completion_handler(
        State(state): State<UpstreamState>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        *state.completion.lock().expect("completion body lock") = Some(body);
        Json(json!({
            "choices": [{"text": "pong", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2}
        }))
    }

    #[tokio::test]
    async fn chat_streams_and_completion_forwards_json_response() {
        let chat = Arc::new(Mutex::new(None));
        let completion_body = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_handler))
            .route("/v1/completions", post(completion_handler))
            .with_state(UpstreamState {
                chat: Arc::clone(&chat),
                completion: Arc::clone(&completion_body),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let root = tempfile::tempdir().unwrap();
        write_artifacts(root.path());
        let mut config = enabled_config(root.path());
        config.xolotl.gpus = vec![0];
        config.xolotl.cuda_architecture = "8.9".into();
        config.xolotl.layer_split = "none".into();
        config.xolotl.upstream = Some(origin);
        let manifest = manifest_for(&config);
        let device = CudaDevice {
            index: 0,
            name: "NVIDIA GeForce RTX 4090".into(),
            compute_major: 8,
            compute_minor: 9,
            memory_bytes: 24_564 * 1024 * 1024,
        };
        let backend = XolotlBackend::new(Arc::new(config)).with_devices(vec![device]);
        backend.load(&manifest).await.unwrap();

        let mut request = chat_request();
        request.stream = true;
        let mut stream = backend.chat("xq-1017", request).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.content, "ready");
        assert!(!first.done);
        let done = stream.next().await.unwrap().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("stop"));
        assert_eq!(done.prompt_tokens, Some(4));
        assert_eq!(
            first.thinking_content.as_deref(),
            Some("measured reasoning")
        );
        assert!(backend.reports_exact_chat_usage());
        let observation = done.upstream_timings.as_ref().unwrap();
        assert_eq!(observation["usage"]["completion_tokens"], 17);
        assert_eq!(observation["timings"]["predicted_per_second"], 85.0);
        assert_eq!(done.prompt_eval_duration_ns, Some(2_000_000));
        assert!(stream.next().await.is_none());

        let body = chat.lock().expect("chat body lock").clone().unwrap();
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["messages"][0]["content"], json!("hello"));
        assert_eq!(body["temperature"], json!(1.0));
        assert_eq!(body["top_p"], json!(1.0));
        assert_eq!(body["max_tokens"], json!(256));

        let completion: CompletionRequest = serde_json::from_value(json!({
            "prompt": "ping",
            "max_tokens": 16,
            "stream": true
        }))
        .unwrap();
        let mut stream = backend.complete("xq-1017", completion).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.text, "pong");
        let done = stream.next().await.unwrap().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("length"));
        assert_eq!(done.prompt_tokens, Some(2));
        let posted = completion_body
            .lock()
            .expect("completion body lock")
            .clone()
            .unwrap();
        assert_eq!(posted["stream"], json!(false));
        assert_eq!(posted["prompt"], json!("ping"));

        server.abort();
    }

    #[tokio::test]
    async fn cold_upstream_outlasts_the_reqwest_default() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 2048];
            use std::io::{Read, Write};
            let _ = stream.read(&mut buf);
            std::thread::sleep(Duration::from_secs(35));
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
                .unwrap();
            stream.flush().unwrap();
        });
        let started = std::time::Instant::now();
        let result = upstream_client()
            .get(format!("http://{addr}/"))
            .send()
            .await;
        assert!(
            started.elapsed() >= Duration::from_secs(31),
            "client returned in {:?}",
            started.elapsed()
        );
        assert!(result.is_ok(), "{result:?}");
        server.join().unwrap();
    }
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod streaming_tests;
