//! Prism backend — serve PrismML / Bonsai GGUF packs through a pinned Prism runtime.
//!
//! Power does **not** link Prism kernels. This backend:
//! 1. Claims only Prism-required GGUF manifests (`PTQ1_0` / `PQ2_0` / Bonsai 2 markers)
//! 2. Forwards OpenAI chat/completions to a Prism `llama-server` upstream
//! 3. Leaves Power's pinned `llamacpp` speculative stack untouched
//! 4. Selects acceleration via `prism_profile` (baseline|dspark|mtp|dflash|kv4), never via
//!    Power `spec_mode = dspark|mtp|dflash`
//!
//! Configure with `prism_upstream = "http://127.0.0.1:8080"` (or `A3S_PRISM_UPSTREAM`).
//! See `docs/prism-acceleration-plan.md`.

mod detect;
mod profile;
mod stream;
mod timings;

#[cfg(test)]
mod streaming_tests;

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::{Stream, TryStreamExt};
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_stream::wrappers::ReceiverStream;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

use super::prompt_cache::PromptCacheSupport;
use super::types::{
    ChatMessage, ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk,
    EmbeddingRequest, EmbeddingResponse,
};
use super::{ensure_non_speculative_mode, Backend, SpeculativeMetricsSnapshot};

pub use detect::{is_prism_required_manifest, is_prism_required_path};
pub use profile::{resolve_profile, PrismProfile};
pub use timings::{UpstreamSpecMetrics, UpstreamTimings};

/// How long a successful `/health` probe may be reused before the next probe.
const HEALTH_CACHE_TTL: Duration = Duration::from_secs(5);

/// PrismML / Bonsai runtime fronted by Power.
pub struct PrismBackend {
    config: Arc<PowerConfig>,
    profile: PrismProfile,
    http: reqwest::Client,
    /// Models that passed Prism load checks (name → absolute weight path).
    loaded: RwLock<HashMap<String, String>>,
    /// Cumulative upstream-reported speculative counters (not Power-verified).
    spec_metrics: Arc<Mutex<UpstreamSpecMetrics>>,
    /// Cached healthy upstream base URL + probe time (avoids per-request RTT).
    healthy_upstream: Arc<Mutex<Option<(Instant, String)>>>,
}

impl PrismBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        let profile = resolve_profile(config.prism_profile.as_deref()).unwrap_or_else(|error| {
            tracing::error!(%error, "invalid prism_profile; falling back to baseline");
            PrismProfile::Baseline
        });
        Self {
            config,
            profile,
            http: reqwest::Client::new(),
            loaded: RwLock::new(HashMap::new()),
            spec_metrics: Arc::new(Mutex::new(UpstreamSpecMetrics::default())),
            healthy_upstream: Arc::new(Mutex::new(None)),
        }
    }

    pub fn profile(&self) -> PrismProfile {
        self.profile
    }

    fn upstream_base(&self) -> Result<String> {
        if let Some(url) = self
            .config
            .prism_upstream
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            return Ok(url.trim_end_matches('/').to_string());
        }
        if let Ok(url) = std::env::var("A3S_PRISM_UPSTREAM") {
            let url = url.trim();
            if !url.is_empty() {
                return Ok(url.trim_end_matches('/').to_string());
            }
        }
        Err(PowerError::Config(
            "prism backend requires prism_upstream in config.acl \
             (or A3S_PRISM_UPSTREAM) pointing at a Prism llama-server"
                .into(),
        ))
    }

    fn validate_profile_artifacts(&self) -> Result<()> {
        // Re-resolve so ACL typos fail closed at load time, not only at construction.
        let profile = resolve_profile(self.config.prism_profile.as_deref())?;
        if profile != self.profile {
            return Err(PowerError::Config(
                "prism_profile changed after backend construction; restart Power to apply".into(),
            ));
        }
        if !profile.requires_drafter() {
            return Ok(());
        }
        let Some(path) = self
            .config
            .prism_drafter
            .as_ref()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return Err(PowerError::Config(format!(
                "prism_profile={} requires prism_drafter pointing at a target-matched \
                 speculative sidecar (DSpark: *dspark-dflash*; DFlash: *DFlash*.gguf). \
                 Bonsai-2 has no official DSpark pin — prefer prism_profile=mtp with an \
                 in-file MTP graft, or prism_profile=dflash with a community DFlash2 \
                 GGUF. See docs/prism-acceleration-plan.md",
                profile.as_str()
            )));
        };
        let path = Path::new(path);
        if !path.is_file() {
            return Err(PowerError::Config(format!(
                "prism_drafter is not a file: {}",
                path.display()
            )));
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let looks_ok = match profile {
            PrismProfile::Dspark => name.contains("dspark") || name.contains("dflash"),
            PrismProfile::Dflash => name.contains("dflash") || name.contains("draft"),
            _ => name.contains("dspark") || name.contains("dflash") || name.contains("draft"),
        };
        if !looks_ok {
            tracing::warn!(
                drafter = %path.display(),
                profile = profile.as_str(),
                "prism_drafter filename does not look like a matched speculative sidecar"
            );
        }
        Ok(())
    }

    async fn ensure_upstream_healthy(&self) -> Result<String> {
        let base = self.upstream_base()?;
        ensure_healthy_upstream(&self.http, &self.healthy_upstream, &base).await
    }

    async fn invalidate_health_cache(&self) {
        *self.healthy_upstream.lock().await = None;
    }

    async fn post_json(&self, path: &[&str], body: serde_json::Value) -> Result<serde_json::Value> {
        let base = self.ensure_upstream_healthy().await?;
        let mut url = base;
        for segment in path {
            url.push('/');
            url.push_str(segment);
        }
        let resp = self.http.post(&url).json(&body).send().await.map_err(|e| {
            PowerError::InferenceFailed(format!("prism request to {url} failed: {e}"))
        })?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("prism response body error: {e}")))?;
        if !status.is_success() {
            self.invalidate_health_cache().await;
            return Err(PowerError::InferenceFailed(format!(
                "prism upstream returned {status}: {text}"
            )));
        }
        serde_json::from_str(&text).map_err(|e| {
            PowerError::InferenceFailed(format!("prism upstream returned invalid JSON: {e}"))
        })
    }

    async fn chat_streaming(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        let body = build_prism_chat_body(model_name, &request, true);
        let base = self.upstream_base()?;
        let http = self.http.clone();
        let healthy = Arc::clone(&self.healthy_upstream);
        let spec_metrics = Arc::clone(&self.spec_metrics);
        let profile = self.profile;
        let model = model_name.to_string();

        // The returned stream owns both startup and response. Dropping it cancels
        // health checks, response-header waits, prefill, and decode HTTP reads
        // without leaving a detached producer or a buffered generation behind.
        let startup = async move {
            let base = ensure_healthy_upstream(&http, &healthy, &base).await?;
            let url = format!("{base}/v1/chat/completions");
            let response = http
                .post(&url)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .json(&body)
                .send()
                .await
                .map_err(|error| {
                    PowerError::InferenceFailed(format!("prism stream to {url} failed: {error}"))
                })?;
            if !response.status().is_success() {
                *healthy.lock().await = None;
                return Err(PowerError::InferenceFailed(format!(
                    "prism upstream stream returned {}",
                    response.status()
                )));
            }
            let content_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .map(str::trim);
            if content_type != Some("text/event-stream") {
                return Err(PowerError::InferenceFailed(
                    "prism upstream did not return text/event-stream".into(),
                ));
            }
            Ok(stream::chat_response_stream(
                response,
                model,
                profile,
                spec_metrics,
            ))
        };
        Ok(Box::pin(futures::stream::once(startup).try_flatten()))
    }

    async fn record_timings(&self, model_name: &str, timings: &UpstreamTimings) {
        tracing::info!(
            model = %model_name,
            profile = %self.profile.as_str(),
            source = "upstream-reported",
            strategy = self.profile.upstream_source_label(),
            predicted_per_second = ?timings.predicted_per_second,
            draft_n = ?timings.draft_n,
            draft_n_accepted = ?timings.draft_n_accepted,
            prompt_per_second = ?timings.prompt_per_second,
            "Prism upstream timings"
        );
        if self.profile.expects_speculation_timings() && !timings.speculation_engaged() {
            tracing::warn!(
                model = %model_name,
                profile = self.profile.as_str(),
                "prism speculation profile selected but upstream timings have no draft_n; \
                 start Prism with matching --spec-type (draft-dspark|draft-mtp)"
            );
        }
        let mut guard = self.spec_metrics.lock().await;
        guard.observe(timings);
    }

    fn strategy_label(&self) -> &'static str {
        self.profile.upstream_source_label()
    }
}

async fn ensure_healthy_upstream(
    http: &reqwest::Client,
    healthy: &Mutex<Option<(Instant, String)>>,
    base: &str,
) -> Result<String> {
    {
        let guard = healthy.lock().await;
        if let Some((probed_at, cached_base)) = guard.as_ref() {
            if cached_base == base && probed_at.elapsed() < HEALTH_CACHE_TTL {
                return Ok(cached_base.clone());
            }
        }
    }
    let health = format!("{base}/health");
    let response = http.get(&health).send().await.map_err(|error| {
        PowerError::InferenceFailed(format!(
            "prism upstream health check failed ({health}): {error}"
        ))
    })?;
    if !response.status().is_success() {
        *healthy.lock().await = None;
        return Err(PowerError::InferenceFailed(format!(
            "prism upstream health check returned {} ({health})",
            response.status()
        )));
    }
    *healthy.lock().await = Some((Instant::now(), base.to_string()));
    Ok(base.to_string())
}

/// Build the OpenAI-compatible chat body forwarded to Prism `llama-server`.
fn build_prism_chat_body(
    model_name: &str,
    request: &ChatRequest,
    stream: bool,
) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(prism_chat_message_json)
        .collect();

    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
        "stream": stream,
    });
    if stream {
        // Usage arrives in a separate final event on OpenAI-compatible servers.
        body["stream_options"] = serde_json::json!({ "include_usage": true });
    }
    if let Some(temp) = request.temperature {
        body["temperature"] = temp.into();
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = top_p.into();
    }
    if let Some(max_tokens) = request.max_tokens {
        body["max_tokens"] = max_tokens.into();
    }
    if let Some(ref tools) = request.tools {
        if !tools.is_empty() {
            body["tools"] = serde_json::to_value(tools).unwrap_or(serde_json::json!([]));
        }
    }
    if let Some(ref tool_choice) = request.tool_choice {
        body["tool_choice"] =
            serde_json::to_value(tool_choice).unwrap_or(serde_json::json!("auto"));
    }
    if let Some(ref stop) = request.stop {
        body["stop"] = serde_json::to_value(stop).unwrap_or(serde_json::Value::Null);
    }
    body
}

fn prism_chat_message_json(message: &ChatMessage) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("role".into(), serde_json::json!(message.role));
    obj.insert(
        "content".into(),
        serde_json::to_value(&message.content).unwrap_or(serde_json::json!("")),
    );
    if let Some(ref name) = message.name {
        obj.insert("name".into(), serde_json::json!(name));
    }
    if let Some(ref tool_calls) = message.tool_calls {
        if !tool_calls.is_empty() {
            obj.insert(
                "tool_calls".into(),
                serde_json::to_value(tool_calls).unwrap_or(serde_json::json!([])),
            );
        }
    }
    if let Some(ref tool_call_id) = message.tool_call_id {
        obj.insert("tool_call_id".into(), serde_json::json!(tool_call_id));
    }
    serde_json::Value::Object(obj)
}

#[async_trait]
impl Backend for PrismBackend {
    fn name(&self) -> &str {
        "prism"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        // Format-only routing must not claim all GGUF — Prism packs are selected
        // exclusively through `supports_manifest`.
        let _ = format;
        false
    }

    fn supports_manifest(&self, manifest: &ModelManifest) -> bool {
        is_prism_required_manifest(manifest)
    }

    fn prompt_cache_support(&self) -> PromptCacheSupport {
        // Upstream may cache prefixes, but Power cannot guarantee the keyed
        // `prompt_cache_key` contract through the HTTP Prism path. DSpark
        // additionally withdraws cross-request reuse on the Prism side.
        let _ = self.profile.allows_cross_request_prompt_cache();
        PromptCacheSupport::Unsupported
    }

    fn speculative_metrics(&self) -> Vec<SpeculativeMetricsSnapshot> {
        match self.spec_metrics.try_lock() {
            Ok(guard) => {
                if guard.requests == 0 {
                    Vec::new()
                } else {
                    vec![guard.snapshot("*", self.strategy_label())]
                }
            }
            Err(_) => Vec::new(),
        }
    }

    async fn load(&self, manifest: &ModelManifest) -> Result<()> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name()).map_err(|error| {
            PowerError::Config(format!(
                "{error}. For Prism acceleration use prism_profile=baseline|dspark|mtp|dflash|kv4 \
                 (not Power spec_mode=mtp|dflash|dspark); see docs/prism-acceleration-plan.md"
            ))
        })?;
        if !is_prism_required_manifest(manifest) {
            return Err(PowerError::Config(format!(
                "prism backend refuses non-Prism pack '{}'",
                manifest.path.display()
            )));
        }
        if !manifest.path.is_file() {
            return Err(PowerError::Config(format!(
                "prism model path is not a file: {}",
                manifest.path.display()
            )));
        }
        self.validate_profile_artifacts()?;
        let _ = self.ensure_upstream_healthy().await?;
        self.loaded
            .write()
            .await
            .insert(manifest.name.clone(), manifest.path.display().to_string());
        tracing::info!(
            model = %manifest.name,
            path = %manifest.path.display(),
            profile = %self.profile.as_str(),
            "Prism model admitted (weights served by Prism upstream)"
        );
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
        if !self.loaded.read().await.contains_key(model_name) {
            return Err(PowerError::ModelNotFound(format!(
                "prism model '{model_name}' is not loaded"
            )));
        }

        // Both client response modes consume the same owned upstream SSE path.
        self.chat_streaming(model_name, request).await
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        ensure_non_speculative_mode(&self.config.spec_mode, self.name())?;
        if !self.loaded.read().await.contains_key(model_name) {
            return Err(PowerError::ModelNotFound(format!(
                "prism model '{model_name}' is not loaded"
            )));
        }

        let mut body = serde_json::json!({
            "model": model_name,
            "prompt": request.prompt,
            "stream": false,
        });
        if let Some(temp) = request.temperature {
            body["temperature"] = temp.into();
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = max_tokens.into();
        }

        let json = self.post_json(&["v1", "completions"], body).await?;
        let text = json
            .pointer("/choices/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(timings) = UpstreamTimings::from_response(&json) {
            self.record_timings(model_name, &timings).await;
        }

        let (tx, rx) = mpsc::channel(4);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(CompletionResponseChunk {
                    text,
                    done: false,
                    prompt_tokens: None,
                    done_reason: None,
                    prompt_eval_duration_ns: None,
                    token_id: None,
                }))
                .await;
            let _ = tx
                .send(Ok(CompletionResponseChunk {
                    text: String::new(),
                    done: true,
                    prompt_tokens: None,
                    done_reason: Some("stop".into()),
                    prompt_eval_duration_ns: None,
                    token_id: None,
                }))
                .await;
        });
        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn embed(
        &self,
        model_name: &str,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        Err(PowerError::BackendNotAvailable(format!(
            "prism backend does not support embeddings (model '{model_name}')"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::backend::test_utils::sample_manifest;

    #[test]
    fn claims_only_prism_gguf_manifests() {
        let backend = PrismBackend::new(Arc::new(PowerConfig::default()));
        let mut prism = sample_manifest("bonsai2");
        prism.format = ModelFormat::Gguf;
        prism.path = PathBuf::from("Ternary-Bonsai-2-27B-PTQ1_0.gguf");
        assert!(backend.supports_manifest(&prism));

        let mut ordinary = sample_manifest("qwen");
        ordinary.format = ModelFormat::Gguf;
        ordinary.path = PathBuf::from("Qwen3.8-27B-Q6_K.gguf");
        assert!(!backend.supports_manifest(&ordinary));
    }

    #[test]
    fn detect_helpers_match_bonsai2_names() {
        assert!(is_prism_required_path(std::path::Path::new(
            "D:/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
        )));
        assert!(is_prism_required_path(std::path::Path::new(
            "Ternary-Bonsai-2-27B-PQ2_0.gguf"
        )));
        assert!(!is_prism_required_path(std::path::Path::new(
            "Qwen3.8-27B-Q6_K.gguf"
        )));
    }

    #[test]
    fn dspark_profile_fails_closed_without_drafter() {
        let mut config = PowerConfig::default();
        config.prism_profile = Some("dspark".into());
        config.prism_upstream = Some("http://127.0.0.1:8080".into());
        let backend = PrismBackend::new(Arc::new(config));
        assert_eq!(backend.profile(), PrismProfile::Dspark);
        let err = backend.validate_profile_artifacts().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("prism_drafter"),
            "expected drafter fail-closed, got: {msg}"
        );
    }

    #[test]
    fn baseline_profile_does_not_require_drafter() {
        let backend = PrismBackend::new(Arc::new(PowerConfig::default()));
        assert_eq!(backend.profile(), PrismProfile::Baseline);
        backend.validate_profile_artifacts().unwrap();
        assert!(!backend.prompt_cache_support().is_supported());
    }

    #[test]
    fn power_spec_modes_fail_closed_on_prism_load() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("Ternary-Bonsai-2-27B-PTQ1_0.gguf");
        std::fs::write(&model_path, b"fake-prism-weights").unwrap();

        for mode in ["mtp", "dflash", "dflash2", "dspark", "draft-model"] {
            let mut config = PowerConfig::default();
            config.spec_mode = mode.into();
            config.prism_upstream = Some("http://127.0.0.1:9".into());
            config.prism_profile = Some("baseline".into());
            let backend = PrismBackend::new(Arc::new(config));
            let mut manifest = sample_manifest("bonsai2-ptq1");
            manifest.format = ModelFormat::Gguf;
            manifest.path = model_path.clone();
            let err = futures::executor::block_on(backend.load(&manifest)).unwrap_err();
            let msg = err.to_string();
            assert!(
                msg.contains("prism_profile") || msg.contains("spec_mode") || msg.contains("prism"),
                "mode={mode} expected fail-closed hint, got: {msg}"
            );
        }
    }

    #[test]
    fn dspark_profile_rejects_missing_drafter_file() {
        let mut config = PowerConfig::default();
        config.prism_profile = Some("dspark".into());
        config.prism_upstream = Some("http://127.0.0.1:8080".into());
        config.prism_drafter = Some("D:/missing/Ternary-Bonsai-2-27B-dspark-dflash.gguf".into());
        let backend = PrismBackend::new(Arc::new(config));
        let err = backend.validate_profile_artifacts().unwrap_err();
        assert!(
            err.to_string().contains("prism_drafter"),
            "got: {}",
            err.to_string()
        );
    }

    #[test]
    fn dspark_profile_accepts_existing_drafter_path() {
        let dir = tempfile::tempdir().unwrap();
        let drafter = dir.path().join("fake-dspark-dflash.gguf");
        std::fs::write(&drafter, b"fake-drafter").unwrap();
        let mut config = PowerConfig::default();
        config.prism_profile = Some("dspark".into());
        config.prism_drafter = Some(drafter.display().to_string());
        let backend = PrismBackend::new(Arc::new(config));
        backend.validate_profile_artifacts().unwrap();
        assert!(!backend.profile().allows_cross_request_prompt_cache());
    }

    #[test]
    fn chat_body_forwards_tools_and_tool_choice() {
        use crate::backend::types::{
            ChatMessage, FunctionDefinition, MessageContent, Tool, ToolChoice,
        };

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: MessageContent::Text("write a file".into()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            temperature: Some(0.2),
            top_p: None,
            max_tokens: Some(128),
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
            stream_options: None,
            tools: Some(vec![Tool {
                tool_type: "function".into(),
                function: FunctionDefinition {
                    name: "write".into(),
                    description: Some("Write a file".into()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "file_path": { "type": "string" },
                            "content": { "type": "string" }
                        },
                        "required": ["file_path", "content"]
                    }),
                    strict: None,
                    unsupported: Default::default(),
                },
                unsupported: Default::default(),
            }]),
            tool_choice: Some(ToolChoice::String("auto".into())),
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

        let body = build_prism_chat_body("bonsai2-ptq1", &request, false);
        assert_eq!(body["model"], "bonsai2-ptq1");
        assert_eq!(body["max_tokens"], 128);
        assert_eq!(body["stream"], false);
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["function"]["name"], "write");
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn chat_body_enables_upstream_stream_for_tool_free_requests() {
        use crate::backend::types::{ChatMessage, MessageContent};

        let request = ChatRequest {
            messages: vec![ChatMessage {
                role: "user".into(),
                content: MessageContent::Text("hi".into()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            temperature: None,
            top_p: None,
            max_tokens: Some(32),
            stop: None,
            stream: true,
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
            stream_options: None,
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
        let body = build_prism_chat_body("bonsai2-ptq1", &request, true);
        assert_eq!(body["stream"], true);
    }
}
