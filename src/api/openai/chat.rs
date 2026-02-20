use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;
use std::time::Instant;
use zeroize::Zeroize;

use super::openai_error;
use crate::api::types::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, Usage,
};
use crate::backend::types::{ChatMessage, ChatRequest, MessageContent};
use crate::server::audit::AuditEvent;
use crate::server::auth::AuthId;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// POST /v1/chat/completions - OpenAI-compatible chat completion.

pub async fn handler(
    State(state): State<AppState>,
    auth_id: Option<axum::Extension<AuthId>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Build request context for isolation and audit tracking
    let ctx = RequestContext::new(auth_id.map(|a| a.0 .0.clone()));
    state.metrics.increment_active_requests();

    // Privacy: redact inference content from logs
    if state.should_redact() {
        let sanitized = state.sanitize_log(&format!("chat request model={model_name}"));
        tracing::debug!("{sanitized}");
    } else {
        tracing::debug!(model = %model_name, "Chat completion request");
    }

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            state.metrics.decrement_active_requests();
            return openai_error(
                "model_not_found",
                &format!("model '{model_name}' not found"),
            )
            .into_response();
        }
    };

    let backend = match state.backends.find_for_format(&manifest.format) {
        Ok(b) => b,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error("backend_unavailable", &e.to_string()).into_response();
        }
    };

    let load_result = match crate::api::autoload::ensure_loaded_with_keep_alive(
        &state,
        &model_name,
        &manifest,
        &backend,
        request.keep_alive.as_deref(),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error("model_load_failed", &e.to_string()).into_response();
        }
    };
    let unload_after_use = load_result.unload_after_use;

    let response_format = request.response_format.as_ref().map(|f| {
        if f.r#type == "json_schema" {
            // Extract the actual JSON Schema and pass it directly to the backend
            // so it can generate a GBNF grammar for constrained output.
            if let Some(ref spec) = f.json_schema {
                if let Some(ref schema) = spec.schema {
                    return schema.clone();
                }
            }
            // Fallback to generic JSON if schema is missing
            serde_json::json!("json")
        } else {
            // "json_object" or "text" — pass type as-is
            serde_json::json!({"type": f.r#type})
        }
    });
    let backend_request = ChatRequest {
        messages: request
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
                name: m.name.clone(),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
                images: None,
            })
            .collect(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: request.stream.unwrap_or(false),
        top_k: None,
        min_p: None,
        repeat_penalty: None,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        seed: request.seed,
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format,
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone(),
        repeat_last_n: None,
        penalize_newline: None,
        num_batch: None,
        num_thread: state.config.num_thread,
        num_thread_batch: None,
        flash_attention: if state.config.flash_attention {
            Some(true)
        } else {
            None
        },
        num_gpu: None,
        main_gpu: None,
        use_mmap: None,
        use_mlock: if state.config.use_mlock {
            Some(true)
        } else {
            None
        },
        num_parallel: Some(state.config.num_parallel as u32),
    };

    let is_stream = request.stream.unwrap_or(false);
    let include_usage_chunk = state.suppress_token_metrics()
        || request
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let id = request_id.clone();
                let model = model_name.clone();
                let created = chrono::Utc::now().timestamp();
                let start = Instant::now();

                // First chunk: send role
                let first_chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![ChatChunkChoice {
                        index: 0,
                        delta: ChatDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            reasoning_content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };

                let first_event = futures::stream::once(async move {
                    let data = serde_json::to_string(&first_chunk).unwrap_or_default();
                    Ok::<_, Infallible>(Event::default().data(data))
                });

                let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let counter_clone = eval_counter.clone();
                let eval_counter2 = eval_counter.clone();
                let prompt_tokens_shared =
                    std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let prompt_tokens_clone = prompt_tokens_shared.clone();
                let prompt_tokens_shared2 = prompt_tokens_shared.clone();
                let ttft_recorded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let ttft_clone = ttft_recorded.clone();
                let metrics = state.metrics.clone();
                let metrics_done = state.metrics.clone();
                let metrics_cleanup = state.metrics.clone();
                let model_for_metrics = model_name.clone();
                let model_for_done = model_name.clone();
                let model_for_cleanup = model_name.clone();
                let backend_cleanup = backend.clone();
                let ctx_cleanup = ctx.clone();
                let audit_stream = state.audit.clone();
                let ctx_audit = ctx.clone();
                let state_cleanup = state.clone();
                let model_for_unload = model_name.clone();

                let id_for_done = id.clone();
                let model_for_done2 = model.clone();
                let content_stream = stream.map(move |chunk| {
                    let event_data = match chunk {
                        Ok(c) => {
                            if !c.done {
                                counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                // Record TTFT on first content chunk
                                if !ttft_clone.swap(true, std::sync::atomic::Ordering::Relaxed) {
                                    metrics.record_ttft(
                                        &model_for_metrics,
                                        start.elapsed().as_secs_f64(),
                                    );
                                }
                            }
                            if let Some(pt) = c.prompt_tokens {
                                prompt_tokens_clone.store(pt, std::sync::atomic::Ordering::Relaxed);
                            }
                            let finish_reason = if c.done {
                                Some(c.done_reason.clone().unwrap_or_else(|| "stop".to_string()))
                            } else {
                                None
                            };
                            let chunk_resp = ChatCompletionChunk {
                                id: id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![ChatChunkChoice {
                                    index: 0,
                                    delta: ChatDelta {
                                        role: None,
                                        content: if c.done { None } else { Some(c.content) },
                                        reasoning_content: if c.done {
                                            None
                                        } else {
                                            c.thinking_content
                                        },
                                        tool_calls: c.tool_calls,
                                    },
                                    finish_reason,
                                }],
                            };
                            serde_json::to_string(&chunk_resp).unwrap_or_default()
                        }
                        Err(e) => serde_json::to_string(&serde_json::json!({
                            "error": { "message": e.to_string() }
                        }))
                        .unwrap_or_default(),
                    };
                    Ok::<_, Infallible>(Event::default().data(event_data))
                });

                // Emit a final usage chunk before [DONE] when either suppress_token_metrics is
                // active (TEE privacy mode) or the client set stream_options.include_usage.
                // Reads are deferred into an async closure so they execute after the content
                // stream is fully consumed (counters have final values at that point).
                let usage_event = if include_usage_chunk {
                    futures::stream::once(async move {
                        let eval_count2 = eval_counter2.load(std::sync::atomic::Ordering::Relaxed);
                        let prompt_tokens2 =
                            prompt_tokens_shared2.load(std::sync::atomic::Ordering::Relaxed);
                        let rp = super::round_tokens(prompt_tokens2);
                        let rc = super::round_tokens(eval_count2);
                        let usage_chunk = ChatCompletionChunk {
                            id: id_for_done,
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_for_done2,
                            choices: vec![],
                        };
                        let mut val = serde_json::to_value(&usage_chunk).unwrap_or_default();
                        val["usage"] = serde_json::json!({
                            "prompt_tokens": rp,
                            "completion_tokens": rc,
                            "total_tokens": rp + rc
                        });
                        let data = serde_json::to_string(&val).unwrap_or_default();
                        Ok::<_, Infallible>(Event::default().data(data))
                    })
                    .left_stream()
                } else {
                    futures::stream::empty().right_stream()
                };

                let done_event = futures::stream::once(async move {
                    // Record final metrics when stream completes
                    let duration = start.elapsed().as_secs_f64();
                    let eval_count = eval_counter.load(std::sync::atomic::Ordering::Relaxed);
                    let prompt_tokens =
                        prompt_tokens_shared.load(std::sync::atomic::Ordering::Relaxed);
                    metrics_done.record_inference_duration(&model_for_done, duration);
                    metrics_done.record_tokens(&model_for_done, "input", prompt_tokens as u64);
                    metrics_done.record_tokens(&model_for_done, "output", eval_count as u64);

                    // Request isolation: clean up backend resources
                    backend_cleanup
                        .cleanup_request(&model_for_cleanup, &ctx_cleanup)
                        .await
                        .ok();
                    metrics_cleanup.decrement_active_requests();

                    // Unload model if keep_alive=0 (after inference, not before)
                    if unload_after_use {
                        let _ = backend_cleanup.unload(&model_for_unload).await;
                        state_cleanup.mark_unloaded(&model_for_unload);
                    }

                    // Audit: log successful streaming inference
                    if let Some(ref audit) = audit_stream {
                        audit.log(&AuditEvent::success(
                            &ctx_audit.request_id,
                            ctx_audit.auth_id.clone(),
                            "chat",
                            Some(model_for_cleanup.clone()),
                            Some(ctx_audit.elapsed().as_millis() as u64),
                            Some(eval_count as u64),
                        ));
                    }

                    Ok::<_, Infallible>(Event::default().data("[DONE]"))
                });

                let full_stream = first_event
                    .chain(content_stream)
                    .chain(usage_event)
                    .chain(done_event);

                Sse::new(full_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                // Non-streaming: collect full response
                let start = Instant::now();
                let mut full_content = String::new();
                let mut full_thinking = String::new();
                let mut completion_tokens: u32 = 0;
                let mut prompt_tokens: u32 = 0;
                let mut finish_reason = "stop".to_string();
                let mut ttft_recorded = false;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_content.push_str(&c.content);
                            if let Some(ref t) = c.thinking_content {
                                full_thinking.push_str(t);
                            }
                            if !c.done {
                                completion_tokens += 1;
                                if !ttft_recorded {
                                    state
                                        .metrics
                                        .record_ttft(&model_name, start.elapsed().as_secs_f64());
                                    ttft_recorded = true;
                                }
                            }
                            if let Some(pt) = c.prompt_tokens {
                                prompt_tokens = pt;
                            }
                            if let Some(reason) = c.done_reason {
                                finish_reason = reason;
                            }
                        }
                        Err(e) => {
                            return openai_error("server_error", &e.to_string()).into_response();
                        }
                    }
                }

                let total_duration_secs = start.elapsed().as_secs_f64();

                state
                    .metrics
                    .record_inference_duration(&model_name, total_duration_secs);
                state
                    .metrics
                    .record_tokens(&model_name, "input", prompt_tokens as u64);
                state
                    .metrics
                    .record_tokens(&model_name, "output", completion_tokens as u64);

                // Round token counts to nearest 10 when suppress_token_metrics is enabled
                // to prevent exact token-count side-channel inference.
                let (reported_prompt, reported_completion) = if state.suppress_token_metrics() {
                    (
                        super::round_tokens(prompt_tokens),
                        super::round_tokens(completion_tokens),
                    )
                } else {
                    (prompt_tokens, completion_tokens)
                };

                let response = ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatCompletionMessage {
                            role: "assistant".to_string(),
                            content: MessageContent::Text(full_content.clone()),
                            name: None,
                            tool_calls: None,
                            tool_call_id: None,
                            images: None,
                            thinking: if full_thinking.is_empty() {
                                None
                            } else {
                                Some(full_thinking.clone())
                            },
                        },
                        finish_reason: Some(finish_reason),
                    }],
                    usage: Usage {
                        prompt_tokens: reported_prompt,
                        completion_tokens: reported_completion,
                        total_tokens: reported_prompt + reported_completion,
                    },
                    system_fingerprint: Some("fp_a3s_power".to_string()),
                };

                // Privacy: zeroize inference buffers in TEE mode
                if state.should_redact() {
                    full_content.zeroize();
                    full_thinking.zeroize();
                }

                // Request isolation: clean up backend resources
                backend.cleanup_request(&model_name, &ctx).await.ok();
                state.metrics.decrement_active_requests();

                // Unload model if keep_alive=0 (after inference, not before)
                if unload_after_use {
                    let _ = backend.unload(&model_name).await;
                    state.mark_unloaded(&model_name);
                }

                // Audit: log successful inference
                if let Some(ref audit) = state.audit {
                    audit.log(&AuditEvent::success(
                        &ctx.request_id,
                        ctx.auth_id.clone(),
                        "chat",
                        Some(model_name.clone()),
                        Some(ctx.elapsed().as_millis() as u64),
                        Some(completion_tokens as u64),
                    ));
                }

                Json(response).into_response()
            }
        }
        Err(e) => {
            state.metrics.decrement_active_requests();
            if let Some(ref audit) = state.audit {
                audit.log(&AuditEvent::failure(
                    &ctx.request_id,
                    ctx.auth_id.clone(),
                    "chat",
                    Some(model_name.clone()),
                    e.to_string(),
                ));
            }
            openai_error("server_error", &e.to_string()).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::model::manifest::ModelFormat;
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_openai_chat_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not found"));
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("st-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"st-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_non_streaming_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["model"], "test");
        assert!(json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Hello"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_streaming_returns_sse() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(content_type.contains("text/event-stream"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mock load failure"));
        // active_requests must return to zero after load failure
        assert_eq!(state.metrics.active_requests(), 0);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_with_tools() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                "model":"test",
                "messages":[{"role":"user","content":"weather in SF?"}],
                "tools":[{
                    "type":"function",
                    "function":{
                        "name":"get_weather",
                        "description":"Get weather",
                        "parameters":{"type":"object","properties":{"location":{"type":"string"}}}
                    }
                }],
                "tool_choice":"auto"
            }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"][0]["finish_reason"].as_str().is_some());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_with_vision() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                "model":"test",
                "messages":[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":"What is this?"},
                        {"type":"image_url","image_url":{"url":"https://example.com/img.jpg"}}
                    ]
                }]
            }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_with_tool_result() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "model":"test",
                "messages":[
                    {"role":"user","content":"weather?"},
                    {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{}"}}]},
                    {"role":"tool","content":"72F","tool_call_id":"call_1"}
                ]
            }"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert!(json["choices"][0]["message"]["content"].as_str().is_some());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_streaming_with_tools() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"test","description":"test","parameters":{"type":"object"}}}],
                "stream":true
            }"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(content_type.contains("text/event-stream"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_has_usage_field() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["usage"].is_object());
        assert!(json["usage"]["completion_tokens"].is_number());
        assert!(json["usage"]["total_tokens"].is_number());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_has_choices() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let choices = json["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0]["index"], 0);
        assert!(choices[0]["message"]["role"].is_string());
        assert!(choices[0]["message"]["content"].is_string());
        assert!(choices[0]["finish_reason"].is_string());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_with_temperature() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"temperature":0.5,"max_tokens":100,"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    async fn test_openai_chat_default_stream_false() {
        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        // OpenAI default is stream=false
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        // Non-streaming should return JSON, not SSE
        assert!(content_type.contains("application/json"));
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_streaming_with_include_usage() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        // The SSE stream should contain a usage chunk before [DONE]
        assert!(body_str.contains("\"usage\""), "expected usage chunk in SSE stream");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_round_tokens_rounds_to_nearest_10() {
        assert_eq!(super::super::round_tokens(0), 0);
        assert_eq!(super::super::round_tokens(1), 0);
        assert_eq!(super::super::round_tokens(4), 0);
        assert_eq!(super::super::round_tokens(5), 10);
        assert_eq!(super::super::round_tokens(9), 10);
        assert_eq!(super::super::round_tokens(10), 10);
        assert_eq!(super::super::round_tokens(14), 10);
        assert_eq!(super::super::round_tokens(15), 20);
        assert_eq!(super::super::round_tokens(99), 100);
        assert_eq!(super::super::round_tokens(100), 100);
        assert_eq!(super::super::round_tokens(1234), 1230);
        assert_eq!(super::super::round_tokens(1235), 1240);
    }

    #[tokio::test]
    async fn test_active_requests_decremented_on_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"nonexistent","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let _ = app.oneshot(req).await.unwrap();
        assert_eq!(state.metrics.active_requests(), 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_keep_alive_zero_unloads_after_inference() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false,"keep_alive":"0"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Model should be unloaded after inference when keep_alive=0
        assert!(!state.is_model_loaded("test"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_keep_alive_zero_streaming_unloads_after_inference() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true,"keep_alive":"0"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        // Consume the full SSE stream so the done_event fires
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        // Model should be unloaded after the stream is fully consumed
        assert!(!state.is_model_loaded("test"));

        std::env::remove_var("A3S_POWER_HOME");
    }
}
