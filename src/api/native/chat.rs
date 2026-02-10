use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::time::Instant;

use crate::api::types::{ChatCompletionMessage, NativeChatRequest, NativeChatResponse};
use crate::backend::types::{ChatMessage, ChatRequest, MessageContent, ToolCall};
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

/// POST /api/chat - Chat completion (Ollama-compatible).
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<NativeChatRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();

    let manifest = match state.registry.get(&model_name) {
        Ok(m) => m,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({
                    "error": format!("model '{}' not found", model_name)
                })),
            )
                .into_response();
        }
    };

    let backend = match state.backends.find_for_format(&manifest.format) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
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
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response();
        }
    };
    let load_duration_ns = load_result.load_duration.as_nanos() as u64;

    let opts = request.options.as_ref();
    let defaults = &manifest.default_parameters;
    let response_format = request.format.clone();

    // Build messages, prepending system_prompt from manifest if set
    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(ref sys) = manifest.system_prompt {
        // Only prepend if the user hasn't already provided a system message
        let has_system = request.messages.iter().any(|m| m.role == "system");
        if !has_system {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: MessageContent::Text(sys.clone()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                images: None,
            });
        }
    }
    // Prepend pre-seeded messages from Modelfile MESSAGE directives
    for msg in &manifest.messages {
        messages.push(ChatMessage {
            role: msg.role.clone(),
            content: MessageContent::Text(msg.content.clone()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            images: None,
        });
    }
    messages.extend(request.messages.iter().map(|m| ChatMessage {
        role: m.role.clone(),
        content: m.content.clone(),
        name: m.name.clone(),
        tool_calls: m.tool_calls.clone(),
        tool_call_id: m.tool_call_id.clone(),
        images: m.images.clone(),
    }));

    let backend_request = ChatRequest {
        messages,
        temperature: apply_defaults(opts.and_then(|o| o.temperature), defaults, "temperature"),
        top_p: apply_defaults(opts.and_then(|o| o.top_p), defaults, "top_p"),
        max_tokens: apply_defaults(opts.and_then(|o| o.num_predict), defaults, "num_predict"),
        stop: opts.and_then(|o| o.stop.clone()),
        stream: request.stream.unwrap_or(true),
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
        tools: request.tools.clone(),
        tool_choice: None,
        repeat_last_n: apply_defaults(
            opts.and_then(|o| o.repeat_last_n),
            defaults,
            "repeat_last_n",
        ),
        penalize_newline: apply_defaults(
            opts.and_then(|o| o.penalize_newline),
            defaults,
            "penalize_newline",
        ),
        num_batch: apply_defaults(opts.and_then(|o| o.num_batch), defaults, "num_batch"),
        num_thread: apply_defaults(opts.and_then(|o| o.num_thread), defaults, "num_thread"),
        num_thread_batch: apply_defaults(
            opts.and_then(|o| o.num_thread_batch),
            defaults,
            "num_thread_batch",
        ),
        flash_attention: apply_defaults(
            opts.and_then(|o| o.flash_attention),
            defaults,
            "flash_attention",
        ),
        num_gpu: apply_defaults(opts.and_then(|o| o.num_gpu), defaults, "num_gpu"),
        main_gpu: apply_defaults(opts.and_then(|o| o.main_gpu), defaults, "main_gpu"),
        use_mmap: apply_defaults(opts.and_then(|o| o.use_mmap), defaults, "use_mmap"),
        use_mlock: apply_defaults(opts.and_then(|o| o.use_mlock), defaults, "use_mlock"),
    };

    let is_stream = request.stream.unwrap_or(true);

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let model_name_owned = model_name.clone();
                let start = Instant::now();
                let load_dur = load_duration_ns;
                // Use shared counter for streaming eval_count
                let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let counter_clone = eval_counter.clone();
                let prompt_tokens_shared =
                    std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let prompt_tokens_clone = prompt_tokens_shared.clone();
                // Track time-to-first-token
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
                                }
                                if let Some(pt) = c.prompt_tokens {
                                    prompt_tokens_clone
                                        .store(pt, std::sync::atomic::Ordering::Relaxed);
                                }
                                let eval_count_val =
                                    counter_clone.load(std::sync::atomic::Ordering::Relaxed);
                                NativeChatResponse {
                                    model: model_name_owned.clone(),
                                    created_at: crate::api::ollama_timestamp(),
                                    message: ChatCompletionMessage {
                                        role: "assistant".to_string(),
                                        content: MessageContent::Text(c.content),
                                        name: None,
                                        tool_calls: c.tool_calls.clone(),
                                        tool_call_id: None,
                                        images: None,
                                    },
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
                                }
                            }
                            Err(e) => NativeChatResponse {
                                model: model_name_owned.clone(),
                                created_at: crate::api::ollama_timestamp(),
                                message: ChatCompletionMessage {
                                    role: "assistant".to_string(),
                                    content: MessageContent::Text(format!("Error: {e}")),
                                    name: None,
                                    tool_calls: None,
                                    tool_call_id: None,
                                    images: None,
                                },
                                done: true,
                                done_reason: None,
                                total_duration: None,
                                load_duration: None,
                                prompt_eval_count: None,
                                prompt_eval_duration: None,
                                eval_count: None,
                                eval_duration: None,
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
                        // Sentinel for metrics flush
                        NativeChatResponse {
                            model: model_for_done,
                            created_at: crate::api::ollama_timestamp(),
                            message: ChatCompletionMessage {
                                role: "assistant".to_string(),
                                content: MessageContent::Text(String::new()),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                                images: None,
                            },
                            done: true,
                            done_reason: None,
                            total_duration: None,
                            load_duration: None,
                            prompt_eval_count: None,
                            prompt_eval_duration: None,
                            eval_count: None,
                            eval_duration: None,
                        }
                    }))
                    // Skip the sentinel metrics-only chunk
                    .filter(|resp| {
                        let is_sentinel = resp.message.content.text().is_empty()
                            && resp.done
                            && resp.total_duration.is_none()
                            && resp.eval_count.is_none();
                        futures::future::ready(!is_sentinel)
                    });
                crate::api::sse::ndjson_response(ndjson_stream)
            } else {
                // Collect full response
                let start = Instant::now();
                let mut full_content = String::new();
                let mut eval_count: u32 = 0;
                let mut prompt_eval_count: Option<u32> = None;
                let mut prompt_eval_duration: Option<u64> = None;
                let mut done_reason: Option<String> = None;
                let mut last_tool_calls: Option<Vec<ToolCall>> = None;
                let mut ttft_recorded = false;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_content.push_str(&c.content);
                            if !c.done {
                                eval_count += 1;
                                if !ttft_recorded {
                                    state
                                        .metrics
                                        .record_ttft(&model_name, start.elapsed().as_secs_f64());
                                    ttft_recorded = true;
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
                            if c.tool_calls.is_some() {
                                last_tool_calls = c.tool_calls;
                            }
                        }
                        Err(e) => {
                            return (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                Json(serde_json::json!({ "error": e.to_string() })),
                            )
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

                Json(NativeChatResponse {
                    model: model_name,
                    created_at: crate::api::ollama_timestamp(),
                    message: ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: MessageContent::Text(full_content),
                        name: None,
                        tool_calls: last_tool_calls,
                        tool_call_id: None,
                        images: None,
                    },
                    done: true,
                    done_reason,
                    total_duration: Some(total_duration),
                    load_duration: Some(load_duration_ns),
                    prompt_eval_count,
                    prompt_eval_duration,
                    eval_count: Some(eval_count),
                    eval_duration: Some(total_duration),
                })
                .into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
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
    async fn test_chat_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_backend_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("st-model");
        manifest.format = ModelFormat::SafeTensors;
        state.registry.register(manifest).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"st-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_non_streaming_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert_eq!(json["model"], "test");
        assert_eq!(json["done"], true);
        assert!(json["message"]["content"]
            .as_str()
            .unwrap()
            .contains("Hello"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_streaming_returns_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert!(
            content_type.contains("application/x-ndjson"),
            "expected NDJSON content-type, got: {content_type}"
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_load_failure() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::load_fails());
        state.registry.register(sample_manifest("test")).unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
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
    async fn test_chat_with_tools() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert_eq!(json["model"], "test");
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_with_multimodal_content() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_with_images_field() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{
                "model":"test",
                "messages":[{
                    "role":"user",
                    "content":"describe this image",
                    "images":["iVBORw0KGgo="]
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
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_with_tool_result_message() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(r#"{
                "model":"test",
                "messages":[
                    {"role":"user","content":"weather?"},
                    {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"SF\"}"}}]},
                    {"role":"tool","content":"72F sunny","tool_call_id":"call_1"}
                ],
                "stream":false
            }"#))
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
    async fn test_chat_response_has_no_images_field() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8(body.to_vec()).unwrap();
        // Response message should not contain images field (it's None, skipped)
        assert!(!body_str.contains("\"images\""));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_streaming_body_is_valid_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
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
                json.get("message").is_some(),
                "each line should have 'message'"
            );
        }

        // Last line should have done=true
        let last: serde_json::Value = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_non_streaming_has_timing_fields() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
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
        assert_eq!(json["done"], true);
        // Timing fields should be present on final response
        assert!(json["total_duration"].is_number());
        assert!(json["load_duration"].is_number());
        assert!(json["eval_count"].is_number());
        assert!(json["eval_duration"].is_number());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_apply_defaults_returns_explicit_value() {
        let defaults = Some(std::collections::HashMap::from([
            ("temperature".to_string(), serde_json::json!(0.5)),
        ]));
        let result: Option<f32> = super::apply_defaults(Some(0.9), &defaults, "temperature");
        assert_eq!(result, Some(0.9));
    }

    #[test]
    fn test_apply_defaults_uses_default_when_none() {
        let defaults = Some(std::collections::HashMap::from([
            ("temperature".to_string(), serde_json::json!(0.5)),
        ]));
        let result: Option<f32> = super::apply_defaults(None, &defaults, "temperature");
        assert_eq!(result, Some(0.5));
    }

    #[test]
    fn test_apply_defaults_returns_none_when_no_defaults() {
        let defaults: Option<std::collections::HashMap<String, serde_json::Value>> = None;
        let result: Option<f32> = super::apply_defaults(None, &defaults, "temperature");
        assert!(result.is_none());
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_with_options() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"temperature":0.5,"top_p":0.9}}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_default_stream_true() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state.registry.register(sample_manifest("test")).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let content_type = resp.headers().get("content-type").unwrap().to_str().unwrap().to_string();
        assert!(content_type.contains("application/x-ndjson"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_with_system_prompt_from_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("test");
        manifest.system_prompt = Some("You are a pirate.".to_string());
        state.registry.register(manifest).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["done"], true);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_chat_user_system_overrides_manifest() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let mut manifest = sample_manifest("test");
        manifest.system_prompt = Some("You are a pirate.".to_string());
        state.registry.register(manifest).unwrap();
        state.mark_loaded("test");

        let app = router::build(state);
        // User provides their own system message — manifest system should NOT be prepended
        let req = Request::builder()
            .method("POST")
            .uri("/api/chat")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"test","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"hi"}],"stream":false}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var("A3S_POWER_HOME");
    }
}
