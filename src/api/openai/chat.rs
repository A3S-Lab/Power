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

    let runtime_policy = match crate::api::prompt_policy::runtime_policy_claim_with_gpu_config(
        &manifest,
        Some(&state.config.gpu),
    ) {
        Ok(policy) => policy,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error(
                "receipt_failed",
                &format!("failed to build runtime policy receipt claim: {e}"),
            )
            .into_response();
        }
    };

    let backend = match state.find_backend(&manifest.format, manifest.size) {
        Ok(b) => b,
        Err(e) => {
            state.metrics.decrement_active_requests();
            return openai_error("backend_unavailable", &state.sanitize_error(&e.to_string()))
                .into_response();
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
            return openai_error("model_load_failed", &state.sanitize_error(&e.to_string()))
                .into_response();
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
                images: m.images.clone(),
            })
            .collect(),
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: request.stream.unwrap_or(false),
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
        response_format,
        tools: request.tools.clone(),
        tool_choice: request.tool_choice.clone(),
        parallel_tool_calls: request.parallel_tool_calls,
        repeat_last_n: request.repeat_last_n,
        penalize_newline: request.penalize_newline,
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
        images: None,
        session_id: None,
    };

    let effective_prompt = match backend
        .effective_chat_prompt_digest(&model_name, &backend_request)
        .await
    {
        Ok(digest) => digest,
        Err(e) => {
            if unload_after_use {
                crate::api::autoload::unload_after_request(&state, &model_name, &backend).await;
            }
            state.metrics.decrement_active_requests();
            return openai_error(
                "receipt_failed",
                &format!("failed to build effective prompt receipt claim: {e}"),
            )
            .into_response();
        }
    };
    let attestation_receipt =
        match crate::api::receipt::chat_receipt_with_runtime_policy_and_effective_prompt(
            &request,
            runtime_policy,
            effective_prompt,
        ) {
            Ok(receipt) => receipt,
            Err(e) => {
                if unload_after_use {
                    crate::api::autoload::unload_after_request(&state, &model_name, &backend).await;
                }
                state.metrics.decrement_active_requests();
                return openai_error(
                    "receipt_failed",
                    &format!("failed to build attestation receipt: {e}"),
                )
                .into_response();
            }
        };
    let attestation_receipt_sha256 = match crate::api::receipt::receipt_digest(&attestation_receipt)
    {
        Ok(digest) => digest,
        Err(e) => {
            if unload_after_use {
                crate::api::autoload::unload_after_request(&state, &model_name, &backend).await;
            }
            state.metrics.decrement_active_requests();
            return openai_error(
                "receipt_failed",
                &format!("failed to digest attestation receipt: {e}"),
            )
            .into_response();
        }
    };

    let is_stream = request.stream.unwrap_or(false);
    let include_usage_chunk = state.suppress_token_metrics()
        || request
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

    // Admission control: hold a permit for the whole request (including the
    // streamed body). Releases on completion or early client disconnect.
    let permit = state.limiter.acquire().await;

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let id = request_id.clone();
                let model = model_name.clone();
                let created = chrono::Utc::now().timestamp();
                let start = Instant::now();

                // Timing padding: delay before sending first token to prevent
                // prompt-length inference from response latency.
                if let Some(pad) = state.timing_padding() {
                    tokio::time::sleep(pad).await;
                }

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
                    Ok::<_, Infallible>(super::sse_json_event(&first_chunk))
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
                let receipt_for_usage = attestation_receipt.clone();
                let receipt_digest_for_usage = attestation_receipt_sha256.clone();
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
                            super::sse_json_data(&chunk_resp)
                        }
                        Err(e) => super::sse_json_data(&serde_json::json!({
                            "error": { "message": e.to_string() }
                        })),
                    };
                    Ok::<_, Infallible>(Event::default().data(event_data))
                });

                // Emit a final usage chunk before [DONE] when either suppress_token_metrics is
                // active (TEE privacy mode) or the client set stream_options.include_usage.
                // Reads are deferred into an async closure so they execute after the content
                // stream is fully consumed (counters have final values at that point).
                let usage_event = futures::stream::once(async move {
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
                    let mut val = match serde_json::to_value(&usage_chunk) {
                        Ok(val) => val,
                        Err(e) => {
                            return Ok::<_, Infallible>(super::sse_json_event(
                                &serde_json::json!({
                                    "error": { "message": format!("failed to serialize usage chunk: {e}") }
                                }),
                            ));
                        }
                    };
                    if include_usage_chunk {
                        val["usage"] = serde_json::json!({
                            "prompt_tokens": rp,
                            "completion_tokens": rc,
                            "total_tokens": rp + rc
                        });
                    }
                    val["attestation_receipt"] = serde_json::json!(receipt_for_usage);
                    val["attestation_receipt_sha256"] = serde_json::json!(receipt_digest_for_usage);
                    Ok::<_, Infallible>(super::sse_json_event(&val))
                });

                let done_event = futures::stream::once(async move {
                    // Hold the admission permit until the stream finishes, then
                    // release it (also released if the client disconnects early).
                    let _permit = permit;
                    // Record final metrics when stream completes
                    let duration = start.elapsed().as_secs_f64();
                    let eval_count = eval_counter.load(std::sync::atomic::Ordering::Relaxed);
                    let prompt_tokens =
                        prompt_tokens_shared.load(std::sync::atomic::Ordering::Relaxed);
                    metrics_done.record_inference_duration(&model_for_done, duration);
                    metrics_done.record_tokens(&model_for_done, "input", prompt_tokens as u64);
                    metrics_done.record_tokens(&model_for_done, "output", eval_count as u64);

                    // Request isolation: clean up backend resources
                    crate::api::autoload::cleanup_after_request(
                        &model_for_cleanup,
                        &ctx_cleanup,
                        &backend_cleanup,
                    )
                    .await;
                    metrics_cleanup.decrement_active_requests();

                    // Unload model if keep_alive=0 (after inference, not before)
                    if unload_after_use {
                        crate::api::autoload::unload_after_request(
                            &state_cleanup,
                            &model_for_unload,
                            &backend_cleanup,
                        )
                        .await;
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
                // Timing padding: delay before processing to prevent
                // prompt-length inference from response latency.
                if let Some(pad) = state.timing_padding() {
                    tokio::time::sleep(pad).await;
                }
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
                            return openai_error(
                                "server_error",
                                &state.sanitize_error(&e.to_string()),
                            )
                            .into_response();
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
                    attestation_receipt: Some(attestation_receipt),
                    attestation_receipt_sha256: Some(attestation_receipt_sha256),
                };

                // Privacy: zeroize inference buffers in TEE mode
                if state.should_redact() {
                    full_content.zeroize();
                    full_thinking.zeroize();
                }

                // Request isolation: clean up backend resources
                crate::api::autoload::cleanup_after_request(&model_name, &ctx, &backend).await;
                state.metrics.decrement_active_requests();

                // Unload model if keep_alive=0 (after inference, not before)
                if unload_after_use {
                    crate::api::autoload::unload_after_request(&state, &model_name, &backend).await;
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
            openai_error("server_error", &state.sanitize_error(&e.to_string())).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::backend::types::EffectivePromptDigest;
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
    async fn test_openai_chat_receipt_includes_effective_prompt_digest() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let prompt_digest = EffectivePromptDigest::chat_rendered_prompt("mock", "rendered prompt");
        let state = test_state_with_mock(
            MockBackend::success().with_effective_prompt(prompt_digest.clone()),
        );
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
        assert_eq!(
            json["attestation_receipt"]["effective_prompt"]["kind"],
            "chat.rendered-prompt"
        );
        assert_eq!(
            json["attestation_receipt"]["effective_prompt"]["sha256"],
            prompt_digest.sha256
        );

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
        assert_eq!(
            json["attestation_receipt"]["schema"],
            "a3s.power.inference-receipt.v2"
        );
        assert_eq!(
            json["attestation_receipt"]["request_type"],
            "chat-completion"
        );
        assert_eq!(
            json["attestation_receipt_sha256"].as_str().unwrap().len(),
            64
        );

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
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(
            body_str.contains("\"attestation_receipt_sha256\""),
            "expected attestation receipt digest in SSE stream"
        );

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
        assert!(json["attestation_receipt"]["effective_prompt"].is_null());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_forwards_message_images_to_backend() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success();
        let chat_request_capture = mock.chat_request_capture();
        let state = test_state_with_mock(mock);
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"What is this?","images":["aGVsbG8="]}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let captured = chat_request_capture
            .lock()
            .expect("chat request lock poisoned")
            .clone()
            .expect("expected backend chat request to be captured");
        assert_eq!(captured.messages.len(), 1);
        assert_eq!(
            captured.messages[0].images.as_deref(),
            Some(&["aGVsbG8=".to_string()][..])
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_forwards_extended_sampling_controls() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success();
        let chat_request_capture = mock.chat_request_capture();
        let state = test_state_with_mock(mock);
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
                    "messages":[{"role":"user","content":"hi"}],
                    "stream":false,
                    "top_k":40,
                    "min_p":0.5,
                    "repeat_penalty":1.25,
                    "repeat_last_n":64,
                    "penalize_newline":true,
                    "num_ctx":4096,
                    "mirostat":2,
                    "mirostat_tau":5.0,
                    "mirostat_eta":0.25,
                    "tfs_z":0.75,
                    "typical_p":0.5,
                    "parallel_tool_calls":false
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let captured = chat_request_capture
            .lock()
            .expect("chat request lock poisoned")
            .clone()
            .expect("expected backend chat request to be captured");
        assert_eq!(captured.top_k, Some(40));
        assert_eq!(captured.min_p, Some(0.5));
        assert_eq!(captured.repeat_penalty, Some(1.25));
        assert_eq!(captured.repeat_last_n, Some(64));
        assert_eq!(captured.penalize_newline, Some(true));
        assert_eq!(captured.num_ctx, Some(4096));
        assert_eq!(captured.mirostat, Some(2));
        assert_eq!(captured.mirostat_tau, Some(5.0));
        assert_eq!(captured.mirostat_eta, Some(0.25));
        assert_eq!(captured.tfs_z, Some(0.75));
        assert_eq!(captured.typical_p, Some(0.5));
        assert_eq!(captured.parallel_tool_calls, Some(false));
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["top_k"],
            serde_json::json!(40)
        );
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["penalize_newline"],
            serde_json::json!(true)
        );
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["parallel_tool_calls"],
            serde_json::json!(false)
        );

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
        assert!(
            body_str.contains("\"usage\""),
            "expected usage chunk in SSE stream"
        );
        assert!(
            body_str.contains("\"attestation_receipt_sha256\""),
            "expected attestation receipt digest in SSE stream"
        );

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
