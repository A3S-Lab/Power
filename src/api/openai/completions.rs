use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;

use super::openai_error;
use crate::api::types::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};
use crate::server::state::AppState;

/// POST /v1/completions - OpenAI-compatible text completion.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<CompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let is_stream = request.stream.unwrap_or(false);

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
        seed: request.seed.map(|s| s as u32),
        num_ctx: None,
        mirostat: None,
        mirostat_tau: None,
        mirostat_eta: None,
        tfs_z: None,
        typical_p: None,
        response_format: None,
    };

    match backend.complete(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let id = request_id.clone();
                let model = model_name.clone();
                let created = chrono::Utc::now().timestamp();

                let sse_stream = stream
                    .map(move |chunk| {
                        let data = match chunk {
                            Ok(c) => {
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
                    .chain(futures::stream::once(async {
                        Ok(Event::default().data("[DONE]"))
                    }));

                Sse::new(sse_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                let mut full_text = String::new();
                let mut completion_tokens: u32 = 0;
                let mut prompt_tokens: u32 = 0;
                let mut finish_reason = "stop".to_string();
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
                            if !c.done {
                                completion_tokens += 1;
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

                Json(CompletionResponse {
                    id: request_id,
                    object: "text_completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name,
                    choices: vec![CompletionChoice {
                        index: 0,
                        text: full_text,
                        finish_reason: Some(finish_reason),
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                })
                .into_response()
            }
        }
        Err(e) => openai_error("server_error", &e.to_string()).into_response(),
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
}
