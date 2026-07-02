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
use crate::backend::types::{ChatMessage, ChatRequest, MessageContent, ToolCall, ToolChoice};
use crate::server::audit::AuditEvent;
use crate::server::auth::AuthId;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

fn has_effective_tools(request: &ChatCompletionRequest) -> bool {
    request
        .effective_tools()
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
}

fn unsupported_local_tool_choice_message(request: &ChatCompletionRequest) -> Option<String> {
    let tool_choice = request.effective_tool_choice()?;
    let has_tools = has_effective_tools(request);

    match tool_choice {
        ToolChoice::String(value) if value == "auto" => None,
        ToolChoice::String(value) if value == "none" && !has_tools => None,
        ToolChoice::String(value) if value == "none" => Some(
            "local chat backends cannot enforce tool_choice=\"none\" while tools are provided; omit tools or use a remote model"
                .to_string(),
        ),
        ToolChoice::String(value) if value == "required" => Some(
            "local chat backends cannot enforce tool_choice=\"required\"; use tool_choice=\"auto\" or a remote model"
                .to_string(),
        ),
        ToolChoice::String(value) => Some(format!(
            "unsupported tool_choice value '{value}'; supported values are auto, none, and required"
        )),
        ToolChoice::Specific(_) => Some(
            "local chat backends cannot enforce named tool_choice or legacy function_call; use tool_choice=\"auto\" or a remote model"
                .to_string(),
        ),
    }
}

fn unsupported_local_parallel_tool_calls_message(
    request: &ChatCompletionRequest,
) -> Option<String> {
    if request.parallel_tool_calls == Some(false) && has_effective_tools(request) {
        Some(
            "local chat backends cannot enforce parallel_tool_calls=false while tools are provided; omit tools or use a remote model"
                .to_string(),
        )
    } else {
        None
    }
}

/// POST /v1/chat/completions - OpenAI-compatible chat completion.
pub async fn handler(
    State(state): State<AppState>,
    auth_id: Option<axum::Extension<AuthId>>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    if let Some(choice_count) = request.n {
        if choice_count != 1 {
            return openai_error(
                "unsupported_choice_count",
                &format!(
                    "unsupported chat completion choice count n={choice_count}; only n=1 is supported"
                ),
            )
            .into_response();
        }
    }
    if request.has_conflicting_max_token_limits() {
        return openai_error(
            "conflicting_max_token_limits",
            "max_tokens and max_completion_tokens must match when both are provided",
        )
        .into_response();
    }
    if request.has_stream_options_without_stream() {
        return openai_error(
            "unsupported_stream_options",
            "chat completion stream_options are only supported when stream=true",
        )
        .into_response();
    }
    if let Some(message) = request
        .stream_options
        .as_ref()
        .and_then(|options| options.unsupported_fields_message())
    {
        return openai_error("unsupported_stream_options", &message).into_response();
    }
    if request.logprobs.unwrap_or(false) || request.top_logprobs.is_some() {
        return openai_error(
            "unsupported_logprobs",
            "chat completion logprobs are not supported; omit logprobs and top_logprobs",
        )
        .into_response();
    }
    if request.logit_bias.is_some() {
        return openai_error(
            "unsupported_logit_bias",
            "chat completion logit_bias is not supported; omit logit_bias",
        )
        .into_response();
    }
    if request.has_unsupported_modalities() {
        return openai_error(
            "unsupported_chat_modalities",
            r#"chat completion modalities are not supported except ["text"]"#,
        )
        .into_response();
    }
    if request.audio.is_some() {
        return openai_error(
            "unsupported_chat_audio",
            "chat completion audio output is not supported; omit audio",
        )
        .into_response();
    }
    if request.prediction.is_some() {
        return openai_error(
            "unsupported_prediction",
            "chat completion prediction hints are not supported; omit prediction",
        )
        .into_response();
    }
    if request.reasoning_effort.is_some() {
        return openai_error(
            "unsupported_reasoning_effort",
            "chat completion reasoning_effort is not supported; omit reasoning_effort",
        )
        .into_response();
    }
    if let Some((code, message)) = request
        .response_format
        .as_ref()
        .and_then(|format| format.validation_error())
    {
        return openai_error(code, &message).into_response();
    }
    if request.has_thinking_inputs() {
        return openai_error(
            "unsupported_message_thinking",
            "chat message thinking input is not supported; omit message.thinking",
        )
        .into_response();
    }
    if request.has_conflicting_tool_definitions() {
        return openai_error(
            "conflicting_tool_definitions",
            "tools and legacy functions cannot both be provided",
        )
        .into_response();
    }
    if request.has_conflicting_tool_choice() {
        return openai_error(
            "conflicting_tool_choice",
            "tool_choice and legacy function_call cannot both be provided",
        )
        .into_response();
    }
    if let Some(message) = request.unsupported_tool_fields_message() {
        return openai_error("unsupported_tool_definition", &message).into_response();
    }

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
    if !matches!(manifest.format, crate::model::manifest::ModelFormat::Remote) {
        if let Some(message) = unsupported_local_tool_choice_message(&request) {
            state.metrics.decrement_active_requests();
            return openai_error("unsupported_tool_choice", &message).into_response();
        }
        if let Some(message) = unsupported_local_parallel_tool_calls_message(&request) {
            state.metrics.decrement_active_requests();
            return openai_error("unsupported_parallel_tool_calls", &message).into_response();
        }
    }

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

    let effective_tools = request.effective_tools();
    let effective_tool_choice = request.effective_tool_choice();
    let response_format =
        super::backend_response_format(&manifest.format, request.response_format.as_ref());
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
        max_tokens: request.effective_max_tokens(),
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
        stream_options: request
            .stream_options
            .as_ref()
            .map(|options| serde_json::json!(options)),
        tools: effective_tools,
        tool_choice: effective_tool_choice,
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
                let mut tool_calls = Vec::<ToolCall>::new();
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
                            if let Some(calls) = c.tool_calls {
                                tool_calls.extend(calls);
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
                            tool_calls: if tool_calls.is_empty() {
                                None
                            } else {
                                Some(tool_calls)
                            },
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
    use crate::api::types::ChatCompletionRequest;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::backend::types::EffectivePromptDigest;
    use crate::model::manifest::ModelFormat;
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[test]
    fn backend_response_format_preserves_remote_json_schema_wire_shape() {
        let format: crate::api::types::ResponseFormat = serde_json::from_value(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "Weather",
                "description": "Weather answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    },
                    "required": ["city"]
                },
                "strict": true
            }
        }))
        .unwrap();

        let value =
            super::super::backend_response_format(&ModelFormat::Remote, Some(&format)).unwrap();

        assert_eq!(value["type"], "json_schema");
        assert_eq!(value["json_schema"]["name"], "Weather");
        assert_eq!(value["json_schema"]["description"], "Weather answer");
        assert_eq!(value["json_schema"]["strict"], true);
        assert_eq!(
            value["json_schema"]["schema"]["properties"]["city"]["type"],
            "string"
        );
    }

    #[test]
    fn backend_response_format_extracts_schema_for_local_backends() {
        let format: crate::api::types::ResponseFormat = serde_json::from_value(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "Weather",
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    }
                }
            }
        }))
        .unwrap();

        let value =
            super::super::backend_response_format(&ModelFormat::Gguf, Some(&format)).unwrap();

        assert_eq!(value["type"], "object");
        assert_eq!(value["properties"]["city"]["type"], "string");
        assert!(value.get("json_schema").is_none());
    }

    #[test]
    fn local_tool_choice_validation_rejects_non_default_policy() {
        let auto: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "tool_choice":"auto"
            }"#,
        )
        .unwrap();
        assert!(super::unsupported_local_tool_choice_message(&auto).is_none());

        let none_without_tools: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tool_choice":"none"
            }"#,
        )
        .unwrap();
        assert!(super::unsupported_local_tool_choice_message(&none_without_tools).is_none());

        let none_with_tools: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "tool_choice":"none"
            }"#,
        )
        .unwrap();
        assert!(
            super::unsupported_local_tool_choice_message(&none_with_tools)
                .unwrap()
                .contains("cannot enforce tool_choice=\"none\"")
        );

        let required: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "tool_choice":"required"
            }"#,
        )
        .unwrap();
        assert!(super::unsupported_local_tool_choice_message(&required)
            .unwrap()
            .contains("cannot enforce tool_choice=\"required\""));

        let specific: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "functions":[{"name":"lookup","parameters":{"type":"object"}}],
                "function_call":{"name":"lookup"}
            }"#,
        )
        .unwrap();
        assert!(super::unsupported_local_tool_choice_message(&specific)
            .unwrap()
            .contains("named tool_choice"));
    }

    #[test]
    fn local_parallel_tool_calls_validation_rejects_false_with_tools() {
        let false_without_tools: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "parallel_tool_calls":false
            }"#,
        )
        .unwrap();
        assert!(
            super::unsupported_local_parallel_tool_calls_message(&false_without_tools).is_none()
        );

        let true_with_tools: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "parallel_tool_calls":true
            }"#,
        )
        .unwrap();
        assert!(super::unsupported_local_parallel_tool_calls_message(&true_with_tools).is_none());

        let false_with_tools: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                "parallel_tool_calls":false
            }"#,
        )
        .unwrap();
        assert!(
            super::unsupported_local_parallel_tool_calls_message(&false_with_tools)
                .unwrap()
                .contains("cannot enforce parallel_tool_calls=false")
        );

        let false_with_legacy_functions: ChatCompletionRequest = serde_json::from_str(
            r#"{
                "model":"test",
                "messages":[{"role":"user","content":"hi"}],
                "functions":[{"name":"lookup","parameters":{"type":"object"}}],
                "parallel_tool_calls":false
            }"#,
        )
        .unwrap();
        assert!(
            super::unsupported_local_parallel_tool_calls_message(&false_with_legacy_functions)
                .unwrap()
                .contains("cannot enforce parallel_tool_calls=false")
        );
    }

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
    async fn test_openai_chat_rejects_multi_choice_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"n":2}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_choice_count");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("only n=1 is supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_conflicting_max_completion_tokens() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"max_completion_tokens":9}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "conflicting_max_token_limits");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("must match"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_stream_options_without_stream() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"stream_options":{"include_usage":true}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_stream_options");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stream=true"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_stream_options_fields() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true,"include_obfuscation":true}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_stream_options");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("include_obfuscation"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_logprobs_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"logprobs":true}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_logprobs");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("logprobs are not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_logit_bias_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"logit_bias":{"123":-100}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_logit_bias");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("logit_bias is not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_allows_text_modalities_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"modalities":["text"]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "model_not_found");
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_modalities_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"modalities":["text","audio"]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_chat_modalities");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("modalities are not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_audio_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"audio":{"format":"wav","voice":"alloy"}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_chat_audio");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("audio output is not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_prediction_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"prediction":{"type":"content","content":"hi"}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_prediction");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("prediction hints are not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_reasoning_effort_request() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"reasoning_effort":"high"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_reasoning_effort");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reasoning_effort is not supported"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_response_format_type() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_response_format");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("supported values"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_response_format_field() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object","future_policy":true}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_response_format");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unsupported response_format field(s): future_policy"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_incomplete_json_schema_response_format() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"Answer"}}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "invalid_response_format");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("json_schema.schema"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_message_thinking_input() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"missing","messages":[{"role":"assistant","content":"answer","thinking":"hidden reasoning"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_message_thinking");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("message.thinking"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_conflicting_tool_definition_fields() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"missing",
                    "messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                    "functions":[{"name":"lookup","parameters":{"type":"object"}}]
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "conflicting_tool_definitions");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot both be provided"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_conflicting_tool_choice_fields() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"missing",
                    "messages":[{"role":"user","content":"hi"}],
                    "tool_choice":"auto",
                    "function_call":"auto"
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "conflicting_tool_choice");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot both be provided"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_tool_definition_fields() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"missing",
                    "messages":[{"role":"user","content":"hi"}],
                    "tools":[{
                        "type":"function",
                        "cache_control":true,
                        "function":{
                            "name":"lookup",
                            "parameters":{"type":"object"}
                        }
                    }]
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_tool_definition");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cache_control"));
    }

    #[tokio::test]
    async fn test_openai_chat_rejects_unsupported_tool_function_fields() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"missing",
                    "messages":[{"role":"user","content":"hi"}],
                    "tools":[{
                        "type":"function",
                        "function":{
                            "name":"lookup",
                            "parameters":{"type":"object"},
                            "x-strict-mode":true
                        }
                    }]
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_tool_definition");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("x-strict-mode"));
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_rejects_local_tool_choice_none_with_tools() {
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
                r#"{
                    "model":"test",
                    "messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                    "tool_choice":"none"
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_tool_choice");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot enforce tool_choice=\"none\""));
        assert_eq!(state.metrics.active_requests(), 0);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_rejects_local_named_legacy_function_call() {
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
                    "messages":[{"role":"user","content":"hi"}],
                    "functions":[{"name":"lookup","parameters":{"type":"object"}}],
                    "function_call":{"name":"lookup"}
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_tool_choice");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("named tool_choice"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_rejects_local_parallel_tool_calls_false_with_tools() {
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
                r#"{
                    "model":"test",
                    "messages":[{"role":"user","content":"hi"}],
                    "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
                    "parallel_tool_calls":false
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_parallel_tool_calls");
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot enforce parallel_tool_calls=false"));
        assert_eq!(state.metrics.active_requests(), 0);

        std::env::remove_var("A3S_POWER_HOME");
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
            json["attestation_receipt"]["decoding"]["parameters"]["n"],
            serde_json::Value::Null
        );
        assert_eq!(
            json["attestation_receipt_sha256"].as_str().unwrap().len(),
            64
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_accepts_single_choice_request() {
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
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"n":1}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["choices"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["n"],
            serde_json::json!(1)
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_forwards_max_completion_tokens() {
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
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false,"max_completion_tokens":12}"#,
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
        assert_eq!(captured.max_tokens, Some(12));
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["max_tokens"],
            serde_json::Value::Null
        );
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["max_completion_tokens"],
            serde_json::json!(12)
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
    async fn test_openai_chat_maps_legacy_functions_to_tools() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success().with_remote_support();
        let chat_request_capture = mock.chat_request_capture();
        let state = test_state_with_mock(mock);
        let mut manifest = sample_manifest("test");
        manifest.format = ModelFormat::Remote;
        state.registry.register(manifest).unwrap();
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
                    "functions":[{
                        "name":"get_weather",
                        "description":"Get weather",
                        "parameters":{"type":"object","properties":{"location":{"type":"string"}}},
                        "strict":true
                    }],
                    "function_call":{"name":"get_weather"}
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
        let tools = captured.tools.expect("expected legacy functions as tools");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].tool_type, "function");
        assert_eq!(tools[0].function.name, "get_weather");
        assert_eq!(tools[0].function.strict, Some(true));
        assert_eq!(
            serde_json::to_value(captured.tool_choice.expect("expected mapped tool choice"))
                .unwrap(),
            serde_json::json!({
                "type": "function",
                "function": {"name": "get_weather"}
            })
        );
        assert!(json["attestation_receipt"]["decoding"]["tools_sha256"].is_string());
        assert!(json["attestation_receipt"]["decoding"]["tool_choice_sha256"].is_string());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_non_streaming_preserves_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::with_tool_calls());
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
                    "messages":[{"role":"user","content":"weather in Paris?"}],
                    "tools":[{
                        "type":"function",
                        "function":{
                            "name":"get_weather",
                            "description":"Get weather",
                            "parameters":{"type":"object"}
                        }
                    }],
                    "stream":false
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "tool_calls");
        let calls = choice["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");

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
                    "typical_p":0.5
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
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["top_k"],
            serde_json::json!(40)
        );
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["penalize_newline"],
            serde_json::json!(true)
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_preserves_remote_parallel_tool_calls() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success().with_remote_support();
        let chat_request_capture = mock.chat_request_capture();
        let state = test_state_with_mock(mock);
        let mut manifest = sample_manifest("test");
        manifest.format = ModelFormat::Remote;
        state.registry.register(manifest).unwrap();
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
                    "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
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
        assert_eq!(captured.parallel_tool_calls, Some(false));
        assert!(captured
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty()));
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["parallel_tool_calls"],
            serde_json::json!(false)
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_chat_preserves_remote_json_schema_response_format() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success().with_remote_support();
        let chat_request_capture = mock.chat_request_capture();
        let state = test_state_with_mock(mock);
        let mut manifest = sample_manifest("test");
        manifest.format = ModelFormat::Remote;
        state.registry.register(manifest).unwrap();
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
                    "response_format":{
                        "type":"json_schema",
                        "json_schema":{
                            "name":"Weather",
                            "description":"Weather answer",
                            "schema":{
                                "type":"object",
                                "properties":{"city":{"type":"string"}},
                                "required":["city"]
                            },
                            "strict":true
                        }
                    }
                }"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let captured = chat_request_capture
            .lock()
            .expect("chat request lock poisoned")
            .clone()
            .expect("expected backend chat request to be captured");
        let response_format = captured
            .response_format
            .expect("expected response_format to be forwarded");
        assert_eq!(response_format["type"], "json_schema");
        assert_eq!(response_format["json_schema"]["name"], "Weather");
        assert_eq!(
            response_format["json_schema"]["description"],
            "Weather answer"
        );
        assert_eq!(response_format["json_schema"]["strict"], true);
        assert_eq!(
            response_format["json_schema"]["schema"]["properties"]["city"]["type"],
            "string"
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
        let captured = chat_request_capture
            .lock()
            .expect("chat request lock poisoned")
            .clone()
            .expect("expected backend chat request to be captured");
        assert_eq!(
            captured.stream_options,
            Some(serde_json::json!({"include_usage": true}))
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
