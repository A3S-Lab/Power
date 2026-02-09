use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::time::Instant;

use crate::api::types::{GenerateRequest, GenerateResponse};
use crate::backend::types::CompletionRequest;
use crate::server::state::AppState;

/// Apply manifest default_parameters as fallback for unset options.
fn apply_defaults<T: serde::de::DeserializeOwned>(
    val: Option<T>,
    defaults: &Option<std::collections::HashMap<String, serde_json::Value>>,
    key: &str,
) -> Option<T> {
    if val.is_some() {
        return val;
    }
    defaults
        .as_ref()
        .and_then(|m| m.get(key))
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

/// POST /api/generate - Text generation (Ollama-compatible).
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            return Json(serde_json::json!({
                "error": format!("model '{}' not found", model_name)
            }))
            .into_response();
        }
    };

    let backend = match state.backends.find_for_format(&manifest.format) {
        Ok(b) => b,
        Err(e) => {
            return Json(serde_json::json!({ "error": e.to_string() })).into_response();
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
            return Json(serde_json::json!({ "error": e.to_string() })).into_response();
        }
    };
    let load_duration_ns = load_result.load_duration.as_nanos() as u64;

    let opts = request.options.as_ref();
    let defaults = &manifest.default_parameters;
    let response_format = request.format.clone();

    // Warn about unsupported images
    if request.images.is_some() {
        tracing::warn!("images field in /api/generate not yet supported; images will be ignored");
    }

    // Build the prompt: if system is provided and raw is not set, prepend it
    let prompt = if let Some(ref system) = request.system {
        if request.raw.unwrap_or(false) {
            request.prompt.clone()
        } else {
            format!("{}\n\n{}", system, request.prompt)
        }
    } else if let Some(ref sys) = manifest.system_prompt {
        if request.raw.unwrap_or(false) {
            request.prompt.clone()
        } else {
            format!("{}\n\n{}", sys, request.prompt)
        }
    } else {
        request.prompt.clone()
    };

    let backend_request = CompletionRequest {
        prompt,
        temperature: apply_defaults(opts.and_then(|o| o.temperature), defaults, "temperature"),
        top_p: apply_defaults(opts.and_then(|o| o.top_p), defaults, "top_p"),
        max_tokens: apply_defaults(opts.and_then(|o| o.num_predict), defaults, "num_predict"),
        stop: opts.and_then(|o| o.stop.clone()),
        stream: request.stream.unwrap_or(false),
        top_k: apply_defaults(opts.and_then(|o| o.top_k), defaults, "top_k"),
        min_p: apply_defaults(opts.and_then(|o| o.min_p), defaults, "min_p"),
        repeat_penalty: apply_defaults(
            opts.and_then(|o| o.repeat_penalty),
            defaults,
            "repeat_penalty",
        ),
        frequency_penalty: apply_defaults(
            opts.and_then(|o| o.frequency_penalty),
            defaults,
            "frequency_penalty",
        ),
        presence_penalty: apply_defaults(
            opts.and_then(|o| o.presence_penalty),
            defaults,
            "presence_penalty",
        ),
        seed: apply_defaults(opts.and_then(|o| o.seed), defaults, "seed"),
        num_ctx: apply_defaults(opts.and_then(|o| o.num_ctx), defaults, "num_ctx"),
        mirostat: apply_defaults(opts.and_then(|o| o.mirostat), defaults, "mirostat"),
        mirostat_tau: apply_defaults(opts.and_then(|o| o.mirostat_tau), defaults, "mirostat_tau"),
        mirostat_eta: apply_defaults(opts.and_then(|o| o.mirostat_eta), defaults, "mirostat_eta"),
        tfs_z: apply_defaults(opts.and_then(|o| o.tfs_z), defaults, "tfs_z"),
        typical_p: apply_defaults(opts.and_then(|o| o.typical_p), defaults, "typical_p"),
        response_format,
    };

    let is_stream = request.stream.unwrap_or(false);

    match backend.complete(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                // Streaming: return newline-delimited JSON (Ollama wire format)
                let model_name_owned = model_name.clone();
                let start = Instant::now();
                let load_dur = load_duration_ns;
                let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let counter_clone = eval_counter.clone();
                let prompt_tokens_shared =
                    std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let prompt_tokens_clone = prompt_tokens_shared.clone();
                let context_tokens = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
                let context_clone = context_tokens.clone();
                let ttft_recorded = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let ttft_clone = ttft_recorded.clone();
                let metrics = state.metrics.clone();
                let metrics_done = state.metrics.clone();
                let model_for_metrics = model_name.clone();
                let model_for_done = model_name.clone();
                let ndjson_stream = stream
                    .map(move |chunk| {
                        match chunk {
                            Ok(c) => {
                                if !c.done {
                                    counter_clone
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    // Record TTFT on first content chunk
                                    if !ttft_clone.swap(true, std::sync::atomic::Ordering::Relaxed)
                                    {
                                        metrics.record_ttft(
                                            &model_for_metrics,
                                            start.elapsed().as_secs_f64(),
                                        );
                                    }
                                    // Collect token IDs for context return
                                    if let Some(tid) = c.token_id {
                                        context_clone.lock().unwrap().push(tid);
                                    }
                                }
                                if let Some(pt) = c.prompt_tokens {
                                    prompt_tokens_clone
                                        .store(pt, std::sync::atomic::Ordering::Relaxed);
                                }
                                let eval_count_val =
                                    counter_clone.load(std::sync::atomic::Ordering::Relaxed);
                                GenerateResponse {
                                    model: model_name_owned.clone(),
                                    response: c.text,
                                    done: c.done,
                                    done_reason: c.done_reason,
                                    total_duration: if c.done {
                                        Some(start.elapsed().as_nanos() as u64)
                                    } else {
                                        None
                                    },
                                    load_duration: if c.done { Some(load_dur) } else { None },
                                    prompt_eval_count: c.prompt_tokens,
                                    prompt_eval_duration: c.prompt_eval_duration_ns,
                                    eval_count: if c.done { Some(eval_count_val) } else { None },
                                    eval_duration: if c.done {
                                        Some(start.elapsed().as_nanos() as u64)
                                    } else {
                                        None
                                    },
                                    context: if c.done {
                                        let ctx = context_clone.lock().unwrap();
                                        if ctx.is_empty() {
                                            None
                                        } else {
                                            Some(ctx.clone())
                                        }
                                    } else {
                                        None
                                    },
                                }
                            }
                            Err(e) => GenerateResponse {
                                model: model_name_owned.clone(),
                                response: format!("Error: {e}"),
                                done: true,
                                done_reason: None,
                                total_duration: None,
                                load_duration: None,
                                prompt_eval_count: None,
                                prompt_eval_duration: None,
                                eval_count: None,
                                eval_duration: None,
                                context: None,
                            },
                        }
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
                        metrics_done.record_usage(crate::server::metrics::UsageRecord {
                            timestamp: chrono::Utc::now(),
                            model: model_for_done.clone(),
                            prompt_tokens,
                            completion_tokens: eval_count,
                            total_tokens: prompt_tokens + eval_count,
                            duration_secs: duration,
                            cost_dollars: 0.0,
                        });
                        // Sentinel: empty response to flush metrics (not sent to client)
                        GenerateResponse {
                            model: model_for_done,
                            response: String::new(),
                            done: true,
                            done_reason: None,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: None,
                            prompt_eval_duration: None,
                            eval_count: None,
                            eval_duration: None,
                            context: None,
                        }
                    }))
                    // Skip the sentinel metrics-only chunk
                    .filter(|resp| {
                        let dominated = resp.response.is_empty()
                            && resp.done
                            && resp.total_duration.is_none()
                            && resp.eval_count.is_none();
                        futures::future::ready(!dominated)
                    });
                crate::api::sse::ndjson_response(ndjson_stream)
            } else {
                // Non-streaming: collect all chunks into one response
                let start = Instant::now();
                let mut full_text = String::new();
                let mut eval_count: u32 = 0;
                let mut prompt_eval_count: Option<u32> = None;
                let mut prompt_eval_duration: Option<u64> = None;
                let mut done_reason: Option<String> = None;
                let mut ttft_recorded = false;
                let mut context_tokens = Vec::<u32>::new();
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
                            if !c.done {
                                eval_count += 1;
                                if !ttft_recorded {
                                    state
                                        .metrics
                                        .record_ttft(&model_name, start.elapsed().as_secs_f64());
                                    ttft_recorded = true;
                                }
                                if let Some(tid) = c.token_id {
                                    context_tokens.push(tid);
                                }
                            }
                            if c.prompt_tokens.is_some() {
                                prompt_eval_count = c.prompt_tokens;
                            }
                            if c.prompt_eval_duration_ns.is_some() {
                                prompt_eval_duration = c.prompt_eval_duration_ns;
                            }
                            if c.done_reason.is_some() {
                                done_reason = c.done_reason;
                            }
                        }
                        Err(e) => {
                            return Json(serde_json::json!({ "error": e.to_string() }))
                                .into_response();
                        }
                    }
                }
                let total_duration = start.elapsed().as_nanos() as u64;
                let total_duration_secs = start.elapsed().as_secs_f64();
                let pt = prompt_eval_count.unwrap_or(0);

                // Record Phase 6 metrics
                state
                    .metrics
                    .record_inference_duration(&model_name, total_duration_secs);
                state.metrics.record_tokens(&model_name, "input", pt as u64);
                state
                    .metrics
                    .record_tokens(&model_name, "output", eval_count as u64);
                state
                    .metrics
                    .record_usage(crate::server::metrics::UsageRecord {
                        timestamp: chrono::Utc::now(),
                        model: model_name.clone(),
                        prompt_tokens: pt,
                        completion_tokens: eval_count,
                        total_tokens: pt + eval_count,
                        duration_secs: total_duration_secs,
                        cost_dollars: 0.0,
                    });

                Json(GenerateResponse {
                    model: model_name,
                    response: full_text,
                    done: true,
                    done_reason,
                    total_duration: Some(total_duration),
                    load_duration: Some(load_duration_ns),
                    prompt_eval_count,
                    prompt_eval_duration,
                    eval_count: Some(eval_count),
                    eval_duration: Some(total_duration),
                    context: if context_tokens.is_empty() {
                        None
                    } else {
                        Some(context_tokens)
                    },
                })
                .into_response()
            }
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
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
    async fn test_generate_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"nonexistent","prompt":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("st-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"st-model","prompt":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_non_streaming_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
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
        assert_eq!(json["model"], "test");
        assert_eq!(json["done"], true);
        assert!(json["response"].as_str().unwrap().contains("World"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_streaming_returns_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
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
        assert!(
            content_type.contains("application/x-ndjson"),
            "expected NDJSON content-type, got: {content_type}"
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"test","prompt":"hi"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"]
            .as_str()
            .unwrap()
            .contains("mock load failure"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_with_system_prompt() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","system":"You are a pirate.","stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_with_raw_mode() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"raw prompt","raw":true,"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_with_images_accepted() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"describe","images":["base64data=="],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_with_context_field() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"continue","context":[1,2,3],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_with_suffix() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"fill","suffix":"END","stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_system_with_raw_skips_prepend() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        // When raw=true, system prompt should NOT be prepended
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"raw","system":"ignored","raw":true,"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_non_streaming_returns_context_tokens() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
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
        assert_eq!(json["done"], true);
        // MockBackend emits token_id: Some(42) for the non-done chunk
        let context = json["context"].as_array();
        assert!(context.is_some(), "context field should be present");
        let ctx = context.unwrap();
        assert!(!ctx.is_empty(), "context should contain token IDs");
        assert_eq!(ctx[0], 42);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_generate_streaming_body_is_valid_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/generate")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","prompt":"hi","stream":true}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();

        // Each non-empty line should be valid JSON
        let lines: Vec<&str> = text.trim().split('\n').filter(|l| !l.is_empty()).collect();
        assert!(!lines.is_empty(), "should have at least one NDJSON line");
        for line in &lines {
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(line);
            assert!(
                parsed.is_ok(),
                "each line should be valid JSON, got: {line}"
            );
            let json = parsed.unwrap();
            assert!(json.get("model").is_some(), "each line should have 'model'");
            assert!(
                json.get("response").is_some(),
                "each line should have 'response'"
            );
        }

        // Last line should have done=true
        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }
}
