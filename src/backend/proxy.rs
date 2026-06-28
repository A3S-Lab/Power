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
    ChatRequest, ChatResponseChunk, CompletionRequest, CompletionResponseChunk, EmbeddingRequest,
    EmbeddingResponse,
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
        let url = format!("{}/v1/chat/completions", self.upstream(model_name)?);
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

    async fn complete(
        &self,
        model_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionResponseChunk>> + Send>>> {
        let url = format!("{}/v1/completions", self.upstream(model_name)?);
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
        let url = format!("{}/v1/embeddings", self.upstream(model_name)?);
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
        .map(|m| serde_json::json!({ "role": m.role, "content": m.content.text() }))
        .collect();
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": messages,
        "stream": true,
    });
    set_common(
        &mut body,
        request.temperature,
        request.top_p,
        request.max_tokens,
        &request.stop,
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
    use crate::backend::types::{ChatMessage, MessageContent};

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
