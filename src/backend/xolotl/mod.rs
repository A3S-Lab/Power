//! Xolotl backend for the DeepSeek-V4.1-Flash-1017 XQ pack on eight H200s.
//!
//! `ds41_generate` (Indras-Mirror/Xolotl `tools/v41/engine/ds41_generate.cpp`)
//! rejects `--gpus`. Omitting `--split-device` keeps at most two visible CUDA
//! devices, and `--layer-split auto` is one cut at `n_layers / 2`. The admitted
//! launch is `--layer-split 5,10,15,20,25,30,35 --split-device 0,1,2,3,4,5,6,7`
//! with `CUDA_VISIBLE_DEVICES` set to the eight nvidia-smi indices.
//!
//! Power does not link Xolotl kernels and does not spawn `ds41_generate`.
//! `--serve` speaks the engine line protocol. Chat posts one non-streaming
//! OpenAI request to `serve_ds41.py` (`tools/v41/serve/serve_ds41.py`, port 8150).

mod detect;
mod plan;

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use futures::Stream;

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
    parse_nvidia_smi_csv, query_nvidia_smi, split_log_is_ready, CudaDevice, XolotlLaunchPlan,
};

/// Front for one XQ pack. The GPU process stays outside this type.
pub struct XolotlBackend {
    config: Arc<PowerConfig>,
    http: reqwest::Client,
    /// Injected inventory. Production load calls `nvidia-smi` when this is empty.
    devices: Option<Vec<CudaDevice>>,
    loaded: tokio::sync::RwLock<HashMap<String, XolotlLaunchPlan>>,
}

impl XolotlBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
            devices: None,
            loaded: tokio::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Use a fixture inventory so tests never call `nvidia-smi`.
    pub fn with_devices(mut self, devices: Vec<CudaDevice>) -> Self {
        self.devices = Some(devices);
        self
    }

    async fn loaded_plan(&self, model_name: &str) -> Result<XolotlLaunchPlan> {
        self.loaded
            .read()
            .await
            .get(model_name)
            .cloned()
            .ok_or_else(|| {
                PowerError::ModelNotFound(format!("xolotl model '{model_name}' is not loaded"))
            })
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
    PowerError::InferenceFailed(format!(
        "{error}. Power does not start ds41_generate. Start `{}`, wait until the log contains `{}`, then set xolotl.upstream to the serve_ds41.py origin.",
        plan.command_line(),
        plan.ready_log_line
    ))
}

fn chat_body(model_name: &str, request: &ChatRequest) -> Result<serde_json::Value> {
    if request.has_image_inputs() {
        return Err(PowerError::InferenceFailed(
            "xolotl chat refuses image inputs".into(),
        ));
    }
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(PowerError::InferenceFailed(
            "xolotl chat refuses tool definitions on this non-streaming upstream path".into(),
        ));
    }
    let messages = serde_json::to_value(&request.messages).map_err(|error| {
        PowerError::InferenceFailed(format!(
            "xolotl chat messages could not be encoded: {error}"
        ))
    })?;
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
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

fn chat_stream(
    content: String,
    prompt_tokens: Option<u32>,
    done_reason: String,
) -> Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>> {
    let mut chunks = Vec::new();
    if !content.is_empty() {
        chunks.push(Ok(ChatResponseChunk {
            content,
            thinking_content: None,
            done: false,
            prompt_tokens: None,
            done_reason: None,
            prompt_eval_duration_ns: None,
            tool_calls: None,
            upstream_timings: None,
        }));
    }
    chunks.push(Ok(ChatResponseChunk {
        content: String::new(),
        thinking_content: None,
        done: true,
        prompt_tokens,
        done_reason: Some(done_reason),
        prompt_eval_duration_ns: None,
        tool_calls: None,
        upstream_timings: None,
    }));
    Box::pin(stream::iter(chunks))
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
    fn name(&self) -> &str {
        "xolotl"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        let _ = format;
        false
    }

    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        is_xolotl_required_manifest(manifest)
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
        self.loaded
            .write()
            .await
            .insert(manifest.name.clone(), launch);
        Ok(())
    }

    async fn unload(&self, model_name: &str) -> Result<()> {
        self.loaded.write().await.remove(model_name);
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name())?;
        let launch = self.loaded_plan(model_name).await?;
        let body = chat_body(model_name, &request)?;
        let upstream = require_upstream(&self.config.xolotl)
            .map_err(|error| upstream_error(&launch, error))?;
        let json = self
            .post_json(&format!("{upstream}/v1/chat/completions"), &body)
            .await?;
        let content = string_field(&json, "/choices/0/message/content", "chat content")?;
        Ok(chat_stream(
            content,
            prompt_tokens(&json),
            finish_reason(&json),
        ))
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

    #[derive(Clone)]
    struct UpstreamState {
        chat: Arc<Mutex<Option<Value>>>,
    }

    async fn chat_handler(
        State(state): State<UpstreamState>,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        *state.chat.lock().expect("chat body lock") = Some(body);
        Json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "ready"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4}
        }))
    }

    async fn completion_handler(Json(_body): Json<Value>) -> Json<Value> {
        Json(json!({
            "choices": [{"text": "pong", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 2}
        }))
    }

    #[tokio::test]
    async fn chat_and_completion_forward_one_json_response() {
        let chat = Arc::new(Mutex::new(None));
        let app = Router::new()
            .route("/v1/chat/completions", post(chat_handler))
            .route("/v1/completions", post(completion_handler))
            .with_state(UpstreamState {
                chat: Arc::clone(&chat),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let root = tempfile::tempdir().unwrap();
        write_artifacts(root.path());
        let mut config = enabled_config(root.path());
        config.xolotl.upstream = Some(origin);
        let manifest = manifest_for(&config);
        let backend = XolotlBackend::new(Arc::new(config)).with_devices(eight_h200());
        backend.load(&manifest).await.unwrap();

        let mut stream = backend.chat("xq-1017", chat_request()).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.content, "ready");
        assert!(!first.done);
        let done = stream.next().await.unwrap().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("stop"));
        assert_eq!(done.prompt_tokens, Some(4));
        assert!(stream.next().await.is_none());

        let body = chat.lock().expect("chat body lock").clone().unwrap();
        assert_eq!(body["stream"], json!(false));
        assert_eq!(body["messages"][0]["content"], json!("hello"));
        assert_eq!(body["temperature"], json!(1.0));
        assert_eq!(body["top_p"], json!(1.0));
        assert_eq!(body["max_tokens"], json!(256));

        let completion: CompletionRequest = serde_json::from_value(json!({
            "prompt": "ping",
            "max_tokens": 16
        }))
        .unwrap();
        let mut stream = backend.complete("xq-1017", completion).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.text, "pong");
        let done = stream.next().await.unwrap().unwrap();
        assert!(done.done);
        assert_eq!(done.done_reason.as_deref(), Some("length"));
        assert_eq!(done.prompt_tokens, Some(2));

        server.abort();
    }
}
