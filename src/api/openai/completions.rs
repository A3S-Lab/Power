use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;
use std::time::Instant;
use zeroize::Zeroize;

use super::openai_error;
use crate::api::types::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};
use crate::server::audit::AuditEvent;
use crate::server::auth::AuthId;
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// POST /v1/completions - OpenAI-compatible text completion.
pub async fn handler(
    State(state): State<AppState>,
    auth_id: Option<axum::Extension<AuthId>>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let is_stream = request.stream.unwrap_or(false);
    let include_usage_chunk = state.suppress_token_metrics()
        || request
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);

    // Build request context for isolation and audit tracking
    let ctx = RequestContext::new(auth_id.map(|a| a.0 .0.clone()));
    state.metrics.increment_active_requests();

    // Privacy: redact inference content from logs
    if state.should_redact() {
        tracing::debug!("completion request model={model_name} [content redacted]");
    } else {
        tracing::debug!(model = %model_name, "Completion request");
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
    let attestation_receipt =
        match crate::api::receipt::completion_receipt_with_runtime_policy(&request, runtime_policy)
        {
            Ok(receipt) => receipt,
            Err(e) => {
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
            state.metrics.decrement_active_requests();
            return openai_error(
                "receipt_failed",
                &format!("failed to digest attestation receipt: {e}"),
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

    let backend_request = crate::backend::types::CompletionRequest {
        prompt: request.prompt,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: is_stream,
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
        response_format: None,
        images: None,
        projector_path: None,
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
        suffix: None,
        context: None,
        session_id: None,
    };

    // Admission control: hold a permit for the whole request (including the
    // streamed body). Releases on completion or early client disconnect.
    let permit = state.limiter.acquire().await;

    match backend.complete(&model_name, backend_request).await {
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
                let id_for_receipt = request_id.clone();
                let model_for_receipt = model_name.clone();
                let receipt_for_usage = attestation_receipt.clone();
                let receipt_digest_for_usage = attestation_receipt_sha256.clone();

                let sse_stream = stream.map(move |chunk| {
                    let data = match chunk {
                        Ok(c) => {
                            if !c.done {
                                counter_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
                                Some(c.done_reason.unwrap_or_else(|| "stop".to_string()))
                            } else {
                                None
                            };
                            let resp = CompletionResponse {
                                id: id.clone(),
                                object: "text_completion".to_string(),
                                created,
                                model: model.clone(),
                                choices: vec![CompletionChoice {
                                    index: 0,
                                    text: c.text,
                                    finish_reason,
                                }],
                                usage: Usage {
                                    prompt_tokens: c.prompt_tokens.unwrap_or(0),
                                    completion_tokens: 0,
                                    total_tokens: 0,
                                },
                                system_fingerprint: None,
                                attestation_receipt: None,
                                attestation_receipt_sha256: None,
                            };
                            super::sse_json_data(&resp)
                        }
                        Err(e) => super::sse_json_data(&serde_json::json!({
                            "error": { "message": e.to_string() }
                        })),
                    };
                    Ok::<_, Infallible>(Event::default().data(data))
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
                            "completion",
                            Some(model_for_cleanup.clone()),
                            Some(ctx_audit.elapsed().as_millis() as u64),
                            Some(eval_count as u64),
                        ));
                    }

                    Ok(Event::default().data("[DONE]"))
                });

                // Emit a final receipt chunk before [DONE]. It also carries rounded token
                // counts when suppress_token_metrics is active or stream_options.include_usage is set.
                // Reads are deferred into an async closure so they execute after sse_stream
                // is fully consumed (counters have final values at that point).
                let receipt_event = futures::stream::once(async move {
                    let eval_count2 = eval_counter2.load(std::sync::atomic::Ordering::Relaxed);
                    let pt2 = prompt_tokens_shared2.load(std::sync::atomic::Ordering::Relaxed);
                    let rp = super::round_tokens(pt2);
                    let rc = super::round_tokens(eval_count2);
                    let mut val = serde_json::json!({
                        "id": id_for_receipt,
                        "object": "text_completion",
                        "created": created,
                        "model": model_for_receipt,
                        "choices": [],
                        "attestation_receipt": receipt_for_usage,
                        "attestation_receipt_sha256": receipt_digest_for_usage
                    });
                    if include_usage_chunk {
                        val["usage"] = serde_json::json!({
                            "prompt_tokens": rp,
                            "completion_tokens": rc,
                            "total_tokens": rp + rc
                        });
                    }
                    Ok::<_, Infallible>(super::sse_json_event(&val))
                });

                Sse::new(sse_stream.chain(receipt_event).chain(done_event))
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                let start = Instant::now();
                let mut full_text = String::new();
                let mut completion_tokens: u32 = 0;
                let mut prompt_tokens: u32 = 0;
                let mut finish_reason = "stop".to_string();
                let mut ttft_recorded = false;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
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

                let (reported_prompt, reported_completion) = if state.suppress_token_metrics() {
                    (
                        super::round_tokens(prompt_tokens),
                        super::round_tokens(completion_tokens),
                    )
                } else {
                    (prompt_tokens, completion_tokens)
                };

                let response = CompletionResponse {
                    id: request_id,
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name.clone(),
                    choices: vec![CompletionChoice {
                        index: 0,
                        text: full_text.clone(),
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
                    full_text.zeroize();
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
                        "completion",
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
                    "completion",
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
    use crate::model::manifest::ModelFormat;
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_openai_completions_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"nonexistent","prompt":"hi"}"#))
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
    async fn test_openai_completions_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("st-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"st-model","prompt":"hi"}"#))
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
    async fn test_openai_completions_non_streaming_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["model"], "test");
        assert!(json["choices"][0]["text"]
            .as_str()
            .unwrap()
            .contains("World"));
        assert_eq!(
            json["attestation_receipt"]["schema"],
            "a3s.power.inference-receipt.v2"
        );
        assert_eq!(
            json["attestation_receipt"]["request_type"],
            "text-completion"
        );
        assert_eq!(
            json["attestation_receipt_sha256"].as_str().unwrap().len(),
            64
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_completions_streaming_returns_sse() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":true}"#,
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
    async fn test_openai_completions_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","prompt":"hi"}"#))
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
    async fn test_openai_completions_with_options() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":false,"temperature":0.5,"max_tokens":100}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "text_completion");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_completions_forwards_extended_sampling_controls() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let mock = MockBackend::success();
        let completion_request_capture = mock.completion_request_capture();
        let state = test_state_with_mock(mock);
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                    "model":"test",
                    "prompt":"hi",
                    "stream":false,
                    "top_k":40,
                    "min_p":0.5,
                    "repeat_penalty":1.25,
                    "repeat_last_n":64,
                    "penalize_newline":false,
                    "num_ctx":2048,
                    "mirostat":1,
                    "mirostat_tau":4.0,
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

        let captured = completion_request_capture
            .lock()
            .expect("completion request lock poisoned")
            .clone()
            .expect("expected backend completion request to be captured");
        assert_eq!(captured.top_k, Some(40));
        assert_eq!(captured.min_p, Some(0.5));
        assert_eq!(captured.repeat_penalty, Some(1.25));
        assert_eq!(captured.repeat_last_n, Some(64));
        assert_eq!(captured.penalize_newline, Some(false));
        assert_eq!(captured.num_ctx, Some(2048));
        assert_eq!(captured.mirostat, Some(1));
        assert_eq!(captured.mirostat_tau, Some(4.0));
        assert_eq!(captured.mirostat_eta, Some(0.25));
        assert_eq!(captured.tfs_z, Some(0.75));
        assert_eq!(captured.typical_p, Some(0.5));
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["top_k"],
            serde_json::json!(40)
        );
        assert_eq!(
            json["attestation_receipt"]["decoding"]["parameters"]["penalize_newline"],
            serde_json::json!(false)
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_completions_has_usage_field() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":false}"#,
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
    async fn test_openai_completions_has_choices() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":false}"#,
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
        assert!(choices[0]["text"].is_string());
        assert!(choices[0]["finish_reason"].is_string());

        std::env::remove_var("A3S_POWER_HOME");
    }
}
