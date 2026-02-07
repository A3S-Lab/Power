use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;
use std::time::Instant;

use crate::api::types::{GenerateRequest, GenerateResponse};
use crate::server::state::AppState;

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

    if let Err(e) = crate::api::autoload::ensure_loaded_with_keep_alive(
        &state,
        &model_name,
        &manifest,
        &backend,
        request.keep_alive.as_deref(),
    )
    .await
    {
        return Json(serde_json::json!({ "error": e.to_string() })).into_response();
    }

    let opts = request.options.as_ref();
    let response_format = request.format.clone();
    let backend_request = crate::backend::types::CompletionRequest {
        prompt: request.prompt,
        temperature: opts.and_then(|o| o.temperature),
        top_p: opts.and_then(|o| o.top_p),
        max_tokens: opts.and_then(|o| o.num_predict),
        stop: opts.and_then(|o| o.stop.clone()),
        stream: request.stream.unwrap_or(false),
        top_k: opts.and_then(|o| o.top_k),
        min_p: opts.and_then(|o| o.min_p),
        repeat_penalty: opts.and_then(|o| o.repeat_penalty),
        frequency_penalty: opts.and_then(|o| o.frequency_penalty),
        presence_penalty: opts.and_then(|o| o.presence_penalty),
        seed: opts.and_then(|o| o.seed),
        num_ctx: opts.and_then(|o| o.num_ctx),
        mirostat: opts.and_then(|o| o.mirostat),
        mirostat_tau: opts.and_then(|o| o.mirostat_tau),
        mirostat_eta: opts.and_then(|o| o.mirostat_eta),
        tfs_z: opts.and_then(|o| o.tfs_z),
        typical_p: opts.and_then(|o| o.typical_p),
        response_format,
    };

    let is_stream = request.stream.unwrap_or(false);

    match backend.complete(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                // Streaming: return newline-delimited JSON
                let model_name_owned = model_name.clone();
                let start = Instant::now();
                let sse_stream = stream
                    .map(move |chunk| {
                        let resp = match chunk {
                            Ok(c) => GenerateResponse {
                                model: model_name_owned.clone(),
                                response: c.text,
                                done: c.done,
                                done_reason: c.done_reason,
                                total_duration: if c.done {
                                    Some(start.elapsed().as_nanos() as u64)
                                } else {
                                    None
                                },
                                load_duration: None,
                                prompt_eval_count: c.prompt_tokens,
                                prompt_eval_duration: None,
                                eval_count: None,
                                eval_duration: None,
                            },
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
                // Non-streaming: collect all chunks into one response
                let start = Instant::now();
                let mut full_text = String::new();
                let mut eval_count: u32 = 0;
                let mut prompt_eval_count: Option<u32> = None;
                let mut done_reason: Option<String> = None;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
                            if !c.done {
                                eval_count += 1;
                            }
                            if c.prompt_tokens.is_some() {
                                prompt_eval_count = c.prompt_tokens;
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
                Json(GenerateResponse {
                    model: model_name,
                    response: full_text,
                    done: true,
                    done_reason,
                    total_duration: Some(total_duration),
                    load_duration: None,
                    prompt_eval_count,
                    prompt_eval_duration: None,
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
    async fn test_generate_streaming_returns_sse() {
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
        assert!(content_type.contains("text/event-stream"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
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
}
