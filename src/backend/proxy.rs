//! Proxy backend — forward inference to an upstream OpenAI-compatible server.
//!
//! Lets a3s-power act as the verifiable serving front-door for an existing
//! accelerated engine (vLLM, TGI, SGLang, OpenAI, ...). Clients talk to Power;
//! Power applies its routing, auth, rate-limiting and log-redaction layers and
//! proxies the request to the upstream named in
//! [`PowerConfig::proxy_upstreams`](crate::config::PowerConfig::proxy_upstreams).
//!
//! This is how Power *replaces vLLM in the stack* without reimplementing its
//! CUDA kernels or PagedAttention: it absorbs vLLM as a swappable backend.
//!
//! # Trust boundary
//!
//! Proxied inference runs on the **upstream**, outside any TEE. This is the
//! non-confidential fast path — no hardware attestation covers proxied prompts
//! or responses. Use the in-process backends (mistral.rs / picolm) when content
//! must stay inside the enclave.

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::types::{
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk,
    EffectivePromptDigest, EmbeddingRequest, EmbeddingResponse,
};
use super::Backend;
use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::model::manifest::{ModelFormat, ModelManifest};

/// Forwards inference to upstream OpenAI-compatible servers.
pub struct ProxyBackend {
    config: Arc<PowerConfig>,
    http: reqwest::Client,
}

impl ProxyBackend {
    pub fn new(config: Arc<PowerConfig>) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    /// Resolve the upstream base URL for a model, trimming any trailing slash.
    fn upstream(&self, model_name: &str) -> Result<String> {
        self.config
            .proxy_upstreams
            .get(model_name)
            .map(|u| u.trim_end_matches('/').to_string())
            .ok_or_else(|| {
                PowerError::ModelNotFound(format!(
                    "no proxy upstream configured for '{model_name}'"
                ))
            })
    }

    fn effective_prompt_digest_url(&self, model_name: &str) -> Result<String> {
        let upstream = self.upstream(model_name)?;
        let path = self.config.proxy_effective_prompt_digest_path.trim();
        let segments = configured_proxy_path_segments(path)?;
        proxy_endpoint_url(&upstream, &segments)
    }
}

#[async_trait]
impl Backend for ProxyBackend {
    fn name(&self) -> &str {
        "proxy"
    }

    fn supports(&self, format: &ModelFormat) -> bool {
        matches!(format, ModelFormat::Remote)
    }

    async fn load(&self, _manifest: &ModelManifest) -> Result<()> {
        // Nothing to load — the upstream owns the weights.
        Ok(())
    }

    async fn unload(&self, _model_name: &str) -> Result<()> {
        Ok(())
    }

    async fn chat(
        &self,
        model_name: &str,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatResponseChunk>> + Send>>> {
        let url = proxy_endpoint_url(&self.upstream(model_name)?, &["v1", "chat", "completions"])?;
        let body = build_chat_body(model_name, &request);
        let resp = send_stream(&self.http, &url, body).await?;

        let (tx, rx) = mpsc::channel::<Result<ChatResponseChunk>>(64);
        tokio::spawn(async move {
            let mut stream = Box::pin(resp.bytes_stream());
            let mut buf = String::new();
            while let Some(event) = next_sse_event(&mut stream, &mut buf).await {
                match event {
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                    Ok(None) => break, // [DONE]
                    Ok(Some(json)) => {
                        let delta = json["choices"][0]["delta"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        let done = !json["choices"][0]["finish_reason"].is_null();
                        if !delta.is_empty()
                            && tx
                                .send(Ok(ChatResponseChunk {
                                    content: delta,
                                    thinking_content: None,
                                    done: false,
                                    prompt_tokens: None,
                                    done_reason: None,
                                    prompt_eval_duration_ns: None,
                                    tool_calls: None,
                                }))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        if done {
                            break;
                        }
                    }
                }
            }
            let _ = tx
                .send(Ok(ChatResponseChunk {
                    content: String::new(),
                    thinking_content: None,
                    done: true,
                    prompt_tokens: None,
                    done_reason: Some("stop".to_string()),
                    prompt_eval_duration_ns: None,
                    tool_calls: None,
                }))
                .await;
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn effective_chat_prompt_digest(
        &self,
        model_name: &str,
        request: &ChatRequest,
    ) -> Result<Option<EffectivePromptDigest>> {
        if !self.config.proxy_effective_prompt_digest
            && !self.config.proxy_effective_prompt_digest_required
        {
            return Ok(None);
        }

        if request.has_image_inputs() {
            if self.config.proxy_effective_prompt_digest_required {
                return Err(PowerError::InferenceFailed(
                    "proxy effective prompt digest is required, but image-bearing chat requests must leave effective_prompt absent unless the exact multimodal prompt representation is exposed".to_string(),
                ));
            }
            return Ok(None);
        }

        let url = self.effective_prompt_digest_url(model_name)?;
        let mut body = build_chat_body(model_name, request);
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".to_string(), false.into());
        }

        let resp = self.http.post(&url).json(&body).send().await.map_err(|e| {
            PowerError::InferenceFailed(format!(
                "proxy effective prompt digest request to {url} failed: {e}"
            ))
        })?;

        let status = resp.status();
        if matches!(
            status,
            reqwest::StatusCode::NOT_FOUND
                | reqwest::StatusCode::METHOD_NOT_ALLOWED
                | reqwest::StatusCode::NOT_IMPLEMENTED
        ) && !self.config.proxy_effective_prompt_digest_required
        {
            return Ok(None);
        }

        if !status.is_success() {
            return Err(PowerError::InferenceFailed(format!(
                "proxy upstream returned {status} for effective prompt digest"
            )));
        }

        let json: serde_json::Value = resp.json().await.map_err(|e| {
            PowerError::InferenceFailed(format!("proxy effective prompt digest decode failed: {e}"))
        })?;
        parse_effective_prompt_digest_response(&json).map(Some)
    }

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        let url = proxy_endpoint_url(&self.upstream(model_name)?, &["v1", "completions"])?;
        let body = build_completion_body(model_name, &request);
        let resp = send_stream(&self.http, &url, body).await?;

        let (tx, rx) = mpsc::channel::<Result<CompletionResponseChunk>>(64);
        tokio::spawn(async move {
            let mut stream = Box::pin(resp.bytes_stream());
            let mut buf = String::new();
            while let Some(event) = next_sse_event(&mut stream, &mut buf).await {
                match event {
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                    Ok(None) => break,
                    Ok(Some(json)) => {
                        let text = json["choices"][0]["text"]
                            .as_str()
                            .unwrap_or("")
                            .to_string();
                        let done = !json["choices"][0]["finish_reason"].is_null();
                        if !text.is_empty()
                            && tx
                                .send(Ok(CompletionResponseChunk {
                                    text,
                                    done: false,
                                    prompt_tokens: None,
                                    done_reason: None,
                                    prompt_eval_duration_ns: None,
                                    token_id: None,
                                }))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        if done {
                            break;
                        }
                    }
                }
            }
            let _ = tx
                .send(Ok(CompletionResponseChunk {
                    text: String::new(),
                    done: true,
                    prompt_tokens: None,
                    done_reason: Some("stop".to_string()),
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
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        let url = proxy_endpoint_url(&self.upstream(model_name)?, &["v1", "embeddings"])?;
        let body = serde_json::json!({ "model": model_name, "input": request.input });
        let resp =
            self.http.post(&url).json(&body).send().await.map_err(|e| {
                PowerError::InferenceFailed(format!("proxy embed request failed: {e}"))
            })?;
        if !resp.status().is_success() {
            return Err(PowerError::InferenceFailed(format!(
                "proxy upstream returned {} for embeddings",
                resp.status()
            )));
        }
        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| PowerError::InferenceFailed(format!("proxy embed decode failed: {e}")))?;
        let embeddings = json["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|item| {
                        item["embedding"]
                            .as_array()
                            .map(|v| {
                                v.iter()
                                    .filter_map(|x| x.as_f64().map(|f| f as f32))
                                    .collect()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(EmbeddingResponse { embeddings })
    }
}

fn proxy_endpoint_url(upstream: &str, segments: &[&str]) -> Result<String> {
    let trimmed = upstream.trim();
    if trimmed.is_empty() {
        return Err(PowerError::Config(
            "proxy upstream URL cannot be empty".to_string(),
        ));
    }

    let mut url = reqwest::Url::parse(trimmed)
        .map_err(|e| PowerError::Config(format!("invalid proxy upstream URL {trimmed:?}: {e}")))?;
    url.set_query(None);
    url.set_fragment(None);
    {
        let mut path = url.path_segments_mut().map_err(|_| {
            PowerError::Config(format!(
                "proxy upstream URL {trimmed:?} cannot be used as a base URL"
            ))
        })?;
        path.pop_if_empty();
        for segment in segments {
            path.push(segment);
        }
    }

    Ok(url.to_string())
}

fn configured_proxy_path_segments(path: &str) -> Result<Vec<&str>> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err(PowerError::Config(
            "proxy_effective_prompt_digest_path cannot be empty".to_string(),
        ));
    }
    if trimmed.contains('?') || trimmed.contains('#') {
        return Err(PowerError::Config(
            "proxy_effective_prompt_digest_path must be a path without query or fragment"
                .to_string(),
        ));
    }
    if reqwest::Url::parse(trimmed).is_ok() {
        return Err(PowerError::Config(
            "proxy_effective_prompt_digest_path must be a path, not an absolute URL".to_string(),
        ));
    }

    let segments: Vec<&str> = trimmed
        .trim_matches('/')
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    if segments.is_empty() {
        return Err(PowerError::Config(
            "proxy_effective_prompt_digest_path cannot be empty".to_string(),
        ));
    }
    if segments
        .iter()
        .any(|segment| matches!(*segment, "." | ".."))
    {
        return Err(PowerError::Config(
            "proxy_effective_prompt_digest_path must not contain dot path segments".to_string(),
        ));
    }

    Ok(segments)
}

fn parse_effective_prompt_digest_response(
    json: &serde_json::Value,
) -> Result<EffectivePromptDigest> {
    let claim = json.get("effective_prompt").unwrap_or(json);
    let sha256 = claim
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .ok_or_else(|| {
            PowerError::InferenceFailed(
                "proxy effective prompt digest response missing sha256".to_string(),
            )
        })?;
    if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(PowerError::InferenceFailed(
            "proxy effective prompt digest sha256 must be 64 hex characters".to_string(),
        ));
    }

    let kind = claim
        .get("kind")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("chat.rendered-prompt");
    if kind != "chat.rendered-prompt" {
        return Err(PowerError::InferenceFailed(format!(
            "proxy effective prompt digest kind must be chat.rendered-prompt, got {kind}"
        )));
    }

    let backend = claim
        .get("backend")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("proxy-upstream");

    Ok(EffectivePromptDigest {
        backend: backend.to_string(),
        kind: kind.to_string(),
        sha256: sha256.to_ascii_lowercase(),
    })
}

/// POST a streaming request body and return the response, erroring on non-2xx.
async fn send_stream(
    http: &reqwest::Client,
    url: &str,
    body: serde_json::Value,
) -> Result<reqwest::Response> {
    let resp =
        http.post(url).json(&body).send().await.map_err(|e| {
            PowerError::InferenceFailed(format!("proxy request to {url} failed: {e}"))
        })?;
    if !resp.status().is_success() {
        return Err(PowerError::InferenceFailed(format!(
            "proxy upstream returned {}",
            resp.status()
        )));
    }
    Ok(resp)
}

/// Pull the next SSE `data:` event from a byte stream, buffering partial lines.
///
/// Returns `Ok(Some(json))` for a data line, `Ok(None)` for the `[DONE]`
/// sentinel, `None` when the byte stream ends, `Err` on a transport failure.
/// Generic over the chunk type so the concrete `bytes::Bytes` is never named
/// (it is not a direct dependency).
async fn next_sse_event<S, T>(
    stream: &mut S,
    buf: &mut String,
) -> Option<Result<Option<serde_json::Value>>>
where
    S: Stream<Item = reqwest::Result<T>> + Unpin,
    T: AsRef<[u8]>,
{
    loop {
        // Emit any complete `data:` line already buffered.
        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            let line = line.trim();
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                return Some(Ok(None));
            }
            if data.is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(v) => return Some(Ok(Some(v))),
                // Skip unparseable keep-alive/comment lines rather than abort.
                Err(_) => continue,
            }
        }
        // Need more bytes.
        match stream.next().await {
            Some(Ok(bytes)) => buf.push_str(&String::from_utf8_lossy(bytes.as_ref())),
            Some(Err(e)) => {
                return Some(Err(PowerError::InferenceFailed(format!(
                    "proxy stream error: {e}"
                ))))
            }
            None => return None,
        }
    }
}

fn build_chat_body(model_name: &str, request: &ChatRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = request
        .messages
        .iter()
        .map(|m| {
            let mut msg = serde_json::json!({
                "role": m.role,
                "content": m.content,
            });
            let obj = msg.as_object_mut().expect("message is a json object");
            if let Some(name) = &m.name {
                obj.insert("name".to_string(), serde_json::json!(name));
            }
            if let Some(tool_calls) = &m.tool_calls {
                obj.insert("tool_calls".to_string(), serde_json::json!(tool_calls));
            }
            if let Some(tool_call_id) = &m.tool_call_id {
                obj.insert("tool_call_id".to_string(), serde_json::json!(tool_call_id));
            }
            if let Some(images) = &m.images {
                if !images.is_empty() {
                    obj.insert("images".to_string(), serde_json::json!(images));
                }
            }
            msg
        })
        .collect();
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
        "stream": true,
    });
    if let Some(response_format) = &request.response_format {
        body.as_object_mut()
            .expect("body is a json object")
            .insert("response_format".into(), response_format.clone());
    }
    if let Some(tools) = &request.tools {
        body.as_object_mut()
            .expect("body is a json object")
            .insert("tools".into(), serde_json::json!(tools));
    }
    if let Some(tool_choice) = &request.tool_choice {
        body.as_object_mut()
            .expect("body is a json object")
            .insert("tool_choice".into(), serde_json::json!(tool_choice));
    }
    if let Some(parallel_tool_calls) = request.parallel_tool_calls {
        body.as_object_mut()
            .expect("body is a json object")
            .insert("parallel_tool_calls".into(), parallel_tool_calls.into());
    }
    if let Some(images) = &request.images {
        if !images.is_empty() {
            body.as_object_mut()
                .expect("body is a json object")
                .insert("images".into(), serde_json::json!(images));
        }
    }
    set_common(
        &mut body,
        request.temperature,
        request.top_p,
        request.max_tokens,
        &request.stop,
        request.top_k,
        request.min_p,
        request.repeat_penalty,
        request.repeat_last_n,
        request.penalize_newline,
        request.num_ctx,
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.tfs_z,
        request.typical_p,
        request.frequency_penalty,
        request.presence_penalty,
        request.seed,
    );
    body
}

fn build_completion_body(model_name: &str, request: &CompletionRequest) -> serde_json::Value {
    let mut body = serde_json::json!({
        "model": model_name,
        "prompt": request.prompt,
        "stream": true,
    });
    set_common(
        &mut body,
        request.temperature,
        request.top_p,
        request.max_tokens,
        &request.stop,
        request.top_k,
        request.min_p,
        request.repeat_penalty,
        request.repeat_last_n,
        request.penalize_newline,
        request.num_ctx,
        request.mirostat,
        request.mirostat_tau,
        request.mirostat_eta,
        request.tfs_z,
        request.typical_p,
        request.frequency_penalty,
        request.presence_penalty,
        request.seed,
    );
    body
}

/// Copy the common OpenAI sampling params onto a request body when present.
#[allow(clippy::too_many_arguments)]
fn set_common(
    body: &mut serde_json::Value,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    stop: &Option<Vec<String>>,
    top_k: Option<i32>,
    min_p: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    penalize_newline: Option<bool>,
    num_ctx: Option<u32>,
    mirostat: Option<u32>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    tfs_z: Option<f32>,
    typical_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<i64>,
) {
    let obj = body.as_object_mut().expect("body is a json object");
    if let Some(t) = temperature {
        obj.insert("temperature".into(), t.into());
    }
    if let Some(p) = top_p {
        obj.insert("top_p".into(), p.into());
    }
    if let Some(m) = max_tokens {
        obj.insert("max_tokens".into(), m.into());
    }
    if let Some(s) = stop {
        if !s.is_empty() {
            obj.insert("stop".into(), serde_json::json!(s));
        }
    }
    if let Some(k) = top_k {
        obj.insert("top_k".into(), k.into());
    }
    if let Some(p) = min_p {
        obj.insert("min_p".into(), p.into());
    }
    if let Some(p) = repeat_penalty {
        obj.insert("repeat_penalty".into(), p.into());
    }
    if let Some(n) = repeat_last_n {
        obj.insert("repeat_last_n".into(), n.into());
    }
    if let Some(p) = penalize_newline {
        obj.insert("penalize_newline".into(), p.into());
    }
    if let Some(n) = num_ctx {
        obj.insert("num_ctx".into(), n.into());
    }
    if let Some(m) = mirostat {
        obj.insert("mirostat".into(), m.into());
    }
    if let Some(t) = mirostat_tau {
        obj.insert("mirostat_tau".into(), t.into());
    }
    if let Some(e) = mirostat_eta {
        obj.insert("mirostat_eta".into(), e.into());
    }
    if let Some(z) = tfs_z {
        obj.insert("tfs_z".into(), z.into());
    }
    if let Some(p) = typical_p {
        obj.insert("typical_p".into(), p.into());
    }
    if let Some(f) = frequency_penalty {
        obj.insert("frequency_penalty".into(), f.into());
    }
    if let Some(p) = presence_penalty {
        obj.insert("presence_penalty".into(), p.into());
    }
    if let Some(s) = seed {
        obj.insert("seed".into(), s.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::types::{
        ChatMessage, ContentPart, FunctionDefinition, ImageUrl, MessageContent, Tool, ToolChoice,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    fn chat_req() -> ChatRequest {
        ChatRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hi".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            }],
            session_id: None,
            temperature: Some(0.5),
            top_p: None,
            max_tokens: Some(16),
            stop: Some(vec!["END".to_string()]),
            stream: true,
            top_k: None,
            min_p: None,
            repeat_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: Some(7),
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
        }
    }

    fn image_chat_req() -> ChatRequest {
        let mut request = chat_req();
        request.messages[0].images = Some(vec!["aGVsbG8=".to_string()]);
        request
    }

    #[test]
    fn build_chat_body_maps_params() {
        let body = build_chat_body("llama-70b", &chat_req());
        assert_eq!(body["model"], "llama-70b");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_tokens"], 16);
        assert_eq!(body["stop"][0], "END");
        assert_eq!(body["seed"], 7);
        // Unset params must be omitted, not null.
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn build_chat_body_preserves_multimodal_tools_and_extended_sampling() {
        let mut request = chat_req();
        request.messages[0].content = MessageContent::Parts(vec![
            ContentPart::Text {
                text: "describe this".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,aGVsbG8=".to_string(),
                    detail: Some("low".to_string()),
                },
            },
        ]);
        request.messages[0].images = Some(vec!["message-base64-image".to_string()]);
        request.response_format = Some(serde_json::json!({"type":"json_object"}));
        request.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "lookup".to_string(),
                description: Some("Look up a value".to_string()),
                parameters: serde_json::json!({"type":"object"}),
            },
        }]);
        request.tool_choice = Some(ToolChoice::String("auto".to_string()));
        request.parallel_tool_calls = Some(false);
        request.images = Some(vec!["request-base64-image".to_string()]);
        request.top_k = Some(40);
        request.min_p = Some(0.5);
        request.repeat_penalty = Some(1.25);
        request.repeat_last_n = Some(64);
        request.penalize_newline = Some(true);
        request.num_ctx = Some(4096);
        request.mirostat = Some(2);
        request.mirostat_tau = Some(5.0);
        request.mirostat_eta = Some(0.25);
        request.tfs_z = Some(0.75);
        request.typical_p = Some(0.5);

        let body = build_chat_body("llama-70b", &request);

        assert_eq!(body["messages"][0]["content"][0]["text"], "describe this");
        assert_eq!(
            body["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,aGVsbG8="
        );
        assert_eq!(body["messages"][0]["images"][0], "message-base64-image");
        assert_eq!(body["images"][0], "request-base64-image");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["tools"][0]["function"]["name"], "lookup");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["min_p"], 0.5);
        assert_eq!(body["repeat_penalty"], 1.25);
        assert_eq!(body["repeat_last_n"], 64);
        assert_eq!(body["penalize_newline"], true);
        assert_eq!(body["num_ctx"], 4096);
        assert_eq!(body["mirostat"], 2);
        assert_eq!(body["mirostat_tau"], 5.0);
        assert_eq!(body["mirostat_eta"], 0.25);
        assert_eq!(body["tfs_z"], 0.75);
        assert_eq!(body["typical_p"], 0.5);
    }

    #[test]
    fn build_completion_body_preserves_extended_sampling() {
        let request = CompletionRequest {
            prompt: "hello".to_string(),
            session_id: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop: None,
            stream: true,
            top_k: Some(40),
            min_p: Some(0.5),
            repeat_penalty: Some(1.25),
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            num_ctx: Some(2048),
            mirostat: Some(1),
            mirostat_tau: Some(4.0),
            mirostat_eta: Some(0.25),
            tfs_z: Some(0.75),
            typical_p: Some(0.5),
            response_format: None,
            images: None,
            projector_path: None,
            repeat_last_n: Some(32),
            penalize_newline: Some(false),
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
        };

        let body = build_completion_body("llama-70b", &request);

        assert_eq!(body["prompt"], "hello");
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["min_p"], 0.5);
        assert_eq!(body["repeat_penalty"], 1.25);
        assert_eq!(body["repeat_last_n"], 32);
        assert_eq!(body["penalize_newline"], false);
        assert_eq!(body["num_ctx"], 2048);
        assert_eq!(body["mirostat"], 1);
        assert_eq!(body["mirostat_tau"], 4.0);
        assert_eq!(body["mirostat_eta"], 0.25);
        assert_eq!(body["tfs_z"], 0.75);
        assert_eq!(body["typical_p"], 0.5);
    }

    #[test]
    fn supports_only_remote() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig::default()));
        assert!(backend.supports(&ModelFormat::Remote));
        assert!(!backend.supports(&ModelFormat::Gguf));
    }

    #[test]
    fn upstream_missing_is_model_not_found() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig::default()));
        let err = backend.upstream("nope").unwrap_err();
        assert!(matches!(err, PowerError::ModelNotFound(_)));
    }

    #[test]
    fn proxy_endpoint_url_preserves_base_path_and_drops_query_fragment() {
        let url = proxy_endpoint_url(
            "http://upstream.local/proxy/base?stale=1#section",
            &["v1", "chat", "completions"],
        )
        .unwrap();

        assert_eq!(url, "http://upstream.local/proxy/base/v1/chat/completions");
    }

    #[test]
    fn effective_prompt_digest_url_encodes_configured_path_segments() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest_path: "/v1/chat/rendered prompt".to_string(),
            ..proxy_config("http://upstream.local/proxy?stale=1#section".to_string())
        }));

        let url = backend.effective_prompt_digest_url("llama-70b").unwrap();

        assert_eq!(url, "http://upstream.local/proxy/v1/chat/rendered%20prompt");
    }

    #[test]
    fn effective_prompt_digest_url_rejects_query_in_configured_path() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest_path: "/v1/chat/effective-prompt-digest?debug=1"
                .to_string(),
            ..proxy_config("http://upstream.local".to_string())
        }));

        let err = backend
            .effective_prompt_digest_url("llama-70b")
            .unwrap_err();
        assert!(err.to_string().contains("query or fragment"));
    }

    fn proxy_config(upstream: String) -> PowerConfig {
        let mut proxy_upstreams = HashMap::new();
        proxy_upstreams.insert("llama-70b".to_string(), upstream);
        PowerConfig {
            proxy_upstreams,
            ..Default::default()
        }
    }

    async fn spawn_test_server(app: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), server)
    }

    #[tokio::test]
    async fn effective_prompt_digest_default_is_absent_without_upstream_lookup() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig::default()));
        let digest = backend
            .effective_chat_prompt_digest("llama-70b", &chat_req())
            .await
            .unwrap();
        assert!(digest.is_none());
    }

    #[tokio::test]
    async fn effective_prompt_digest_optional_image_request_is_absent_without_upstream_lookup() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest: true,
            ..Default::default()
        }));

        let digest = backend
            .effective_chat_prompt_digest("llama-70b", &image_chat_req())
            .await
            .unwrap();
        assert!(digest.is_none());
    }

    #[tokio::test]
    async fn effective_prompt_digest_required_image_request_fails_closed() {
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest_required: true,
            ..Default::default()
        }));

        let err = backend
            .effective_chat_prompt_digest("llama-70b", &image_chat_req())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("image-bearing"));
    }

    #[tokio::test]
    async fn effective_prompt_digest_optional_unsupported_endpoint_is_absent() {
        let (upstream, server) = spawn_test_server(axum::Router::new()).await;
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest: true,
            ..proxy_config(upstream)
        }));

        let digest = backend
            .effective_chat_prompt_digest("llama-70b", &chat_req())
            .await
            .unwrap();
        assert!(digest.is_none());
        server.abort();
    }

    #[tokio::test]
    async fn effective_prompt_digest_required_unsupported_endpoint_fails() {
        let (upstream, server) = spawn_test_server(axum::Router::new()).await;
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest_required: true,
            ..proxy_config(upstream)
        }));

        let err = backend
            .effective_chat_prompt_digest("llama-70b", &chat_req())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("effective prompt digest"));
        server.abort();
    }

    #[tokio::test]
    async fn effective_prompt_digest_success_uses_upstream_claim() {
        let received = Arc::new(Mutex::new(None::<serde_json::Value>));
        let handler_received = received.clone();
        let app = axum::Router::new().route(
            "/v1/chat/effective-prompt-digest",
            axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
                let handler_received = handler_received.clone();
                async move {
                    {
                        let mut received = handler_received.lock().unwrap();
                        *received = Some(body);
                    }
                    axum::Json(serde_json::json!({
                        "effective_prompt": {
                            "backend": "vllm",
                            "kind": "chat.rendered-prompt",
                            "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789ABCDEF"
                        }
                    }))
                }
            }),
        );
        let (upstream, server) = spawn_test_server(app).await;
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest: true,
            ..proxy_config(upstream)
        }));

        let digest = backend
            .effective_chat_prompt_digest("llama-70b", &chat_req())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(digest.backend, "vllm");
        assert_eq!(digest.kind, "chat.rendered-prompt");
        assert_eq!(
            digest.sha256,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        );
        let body = received.lock().unwrap().clone().unwrap();
        assert_eq!(body["model"], "llama-70b");
        assert_eq!(body["stream"], false);
        assert_eq!(body["messages"][0]["content"], "hi");
        server.abort();
    }

    #[tokio::test]
    async fn effective_prompt_digest_malformed_sha_fails() {
        let app = axum::Router::new().route(
            "/v1/chat/effective-prompt-digest",
            axum::routing::post(|| async {
                axum::Json(serde_json::json!({
                    "sha256": "not-a-sha"
                }))
            }),
        );
        let (upstream, server) = spawn_test_server(app).await;
        let backend = ProxyBackend::new(Arc::new(PowerConfig {
            proxy_effective_prompt_digest: true,
            ..proxy_config(upstream)
        }));

        let err = backend
            .effective_chat_prompt_digest("llama-70b", &chat_req())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("64 hex"));
        server.abort();
    }

    // ── SSE parser (`next_sse_event`) ─────────────────────────────────────────

    /// Build a byte stream of `reqwest::Result<&[u8]>` chunks for the parser.
    fn byte_stream(
        chunks: Vec<&'static [u8]>,
    ) -> impl Stream<Item = reqwest::Result<&'static [u8]>> + Unpin {
        futures::stream::iter(chunks.into_iter().map(Ok::<&[u8], reqwest::Error>))
    }

    #[tokio::test]
    async fn sse_reassembles_line_split_across_chunks() {
        // A single data line arrives split across three byte chunks.
        let mut s = byte_stream(vec![
            b"data: {\"choices\":[{\"delta\":{\"con",
            b"tent\":\"Hi\"},\"finish_reason\":null}]}\n",
            b"\ndata: [DONE]\n\n",
        ]);
        let mut buf = String::new();
        match next_sse_event(&mut s, &mut buf).await {
            Some(Ok(Some(json))) => {
                assert_eq!(json["choices"][0]["delta"]["content"], "Hi");
            }
            other => panic!("expected data event, got {other:?}"),
        }
        assert!(
            matches!(next_sse_event(&mut s, &mut buf).await, Some(Ok(None))),
            "expected [DONE] sentinel"
        );
    }

    #[tokio::test]
    async fn sse_skips_comments_blanks_and_malformed() {
        // Keep-alive comment, blank line, malformed data, then a real event.
        let mut s = byte_stream(vec![
            b": keep-alive ping\n\n",
            b"\n",
            b"data: not-json-at-all\n\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        ]);
        let mut buf = String::new();
        match next_sse_event(&mut s, &mut buf).await {
            Some(Ok(Some(json))) => assert_eq!(json["choices"][0]["delta"]["content"], "ok"),
            other => panic!("expected the valid event, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sse_handles_crlf_line_endings() {
        let mut s = byte_stream(vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\r\n\r\n",
        ]);
        let mut buf = String::new();
        match next_sse_event(&mut s, &mut buf).await {
            Some(Ok(Some(json))) => assert_eq!(json["choices"][0]["delta"]["content"], "x"),
            other => panic!("CRLF event not parsed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sse_stream_end_without_done_returns_none() {
        let mut s = byte_stream(vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        ]);
        let mut buf = String::new();
        assert!(matches!(
            next_sse_event(&mut s, &mut buf).await,
            Some(Ok(Some(_)))
        ));
        // Underlying byte stream is now exhausted with no [DONE].
        assert!(
            next_sse_event(&mut s, &mut buf).await.is_none(),
            "exhausted stream must yield None"
        );
    }

    #[tokio::test]
    async fn sse_multiple_events_in_one_chunk() {
        // Two data events delivered in a single byte chunk.
        let mut s = byte_stream(vec![
            b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        ]);
        let mut buf = String::new();
        let mut got = String::new();
        while let Some(Ok(Some(json))) = next_sse_event(&mut s, &mut buf).await {
            got.push_str(
                json["choices"][0]["delta"]["content"]
                    .as_str()
                    .unwrap_or(""),
            );
        }
        assert_eq!(got, "ab");
    }
}
