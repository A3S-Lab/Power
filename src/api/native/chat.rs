use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;
use std::time::Instant;

use crate::api::types::{ChatCompletionMessage, NativeChatRequest, NativeChatResponse};
use crate::backend::types::{ChatMessage, ChatRequest, MessageContent};
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
            });
        }
    }
    messages.extend(request.messages.iter().map(|m| ChatMessage {
        role: m.role.clone(),
        content: m.content.clone(),
        name: m.name.clone(),
        tool_calls: m.tool_calls.clone(),
        tool_call_id: m.tool_call_id.clone(),
    }));

    let backend_request = ChatRequest {
        messages,
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
        tools: request.tools.clone(),
        tool_choice: None,
    };

    let is_stream = request.stream.unwrap_or(false);

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let model_name_owned = model_name.clone();
                let start = Instant::now();
                let load_dur = load_duration_ns;
                // Use shared counter for streaming eval_count
                let eval_counter = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
                let counter_clone = eval_counter.clone();
                let sse_stream = stream
                    .map(move |chunk| {
                        let resp = match chunk {
                            Ok(c) => {
                                if !c.done {
                                    counter_clone
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                                let eval_count_val =
                                    counter_clone.load(std::sync::atomic::Ordering::Relaxed);
                                NativeChatResponse {
                                    model: model_name_owned.clone(),
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
                        };
                        let data = serde_json::to_string(&resp).unwrap_or_default();
                        Ok::<_, Infallible>(Event::default().data(data))
                    })
                    .chain(futures::stream::once(async {
                        Ok(Event::default().data("[DONE]"))
                    }));
                Sse::new(sse_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                // Collect full response
                let start = Instant::now();
                let mut full_content = String::new();
                let mut eval_count: u32 = 0;
                let mut prompt_eval_count: Option<u32> = None;
                let mut prompt_eval_duration: Option<u64> = None;
                let mut done_reason: Option<String> = None;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_content.push_str(&c.content);
                            if !c.done {
                                eval_count += 1;
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
                Json(NativeChatResponse {
                    model: model_name,
                    message: ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: MessageContent::Text(full_content),
                        name: None,
                        tool_calls: None,
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
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
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
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["error"].as_str().unwrap().contains("No backend"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
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
    async fn test_chat_streaming_returns_sse() {
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
        assert!(content_type.contains("text/event-stream"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
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
}
