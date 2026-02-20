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
use crate::server::request_context::RequestContext;
use crate::server::state::AppState;

/// POST /v1/completions - OpenAI-compatible text completion.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let is_stream = request.stream.unwrap_or(false);

    // Build request context for isolation and audit tracking
    let ctx = RequestContext::new(None);
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
            return openai_error("server_error", &e.to_string()).into_response();
        }
    };

    if let Err(e) =
        crate::api::autoload::ensure_loaded(&state, &model_name, &manifest, &backend).await
    {
        return openai_error("server_error", &e.to_string()).into_response();
    }

    let backend_request = crate::backend::types::CompletionRequest {
        prompt: request.prompt,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: is_stream,
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
        response_format: None,
        images: None,
        projector_path: None,
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
        suffix: None,
        context: None,
    };

    match backend.complete(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let id = request_id.clone();
                let model = model_name.clone();
                let created = chrono::Utc::now().timestamp();
                let start = Instant::now();

                let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let counter_clone = eval_counter.clone();
                let prompt_tokens_shared =
                    std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let prompt_tokens_clone = prompt_tokens_shared.clone();
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

                let sse_stream = stream
                    .map(move |chunk| {
                        let data = match chunk {
                            Ok(c) => {
                                if !c.done {
                                    counter_clone
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    if !ttft_clone.swap(true, std::sync::atomic::Ordering::Relaxed)
                                    {
                                        metrics.record_ttft(
                                            &model_for_metrics,
                                            start.elapsed().as_secs_f64(),
                                        );
                                    }
                                }
                                if let Some(pt) = c.prompt_tokens {
                                    prompt_tokens_clone
                                        .store(pt, std::sync::atomic::Ordering::Relaxed);
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
                                };
                                serde_json::to_string(&resp).unwrap_or_default()
                            }
                            Err(e) => serde_json::to_string(&serde_json::json!({
                                "error": { "message": e.to_string() }
                            }))
                            .unwrap_or_default(),
                        };
                        Ok::<_, Infallible>(Event::default().data(data))
                    })
                    .chain(futures::stream::once(async move {
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

                        Ok(Event::default().data("[DONE]"))
                    }));

                Sse::new(sse_stream)
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
                            return openai_error("server_error", &e.to_string()).into_response();
                        }
                    }
                }

                let total_duration_secs = start.elapsed().as_secs_f64();

                // Record Phase 6 metrics
                state
                    .metrics
                    .record_inference_duration(&model_name, total_duration_secs);
                state
                    .metrics
                    .record_tokens(&model_name, "input", prompt_tokens as u64);
                state
                    .metrics
                    .record_tokens(&model_name, "output", completion_tokens as u64);

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
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };

                // Privacy: zeroize inference buffers in TEE mode
                if state.should_redact() {
                    full_text.zeroize();
                }

                // Request isolation: clean up backend resources
                backend.cleanup_request(&model_name, &ctx).await.ok();
                state.metrics.decrement_active_requests();

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

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_openai_completions_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","prompt":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mock load failure"));

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
