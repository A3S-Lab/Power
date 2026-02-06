use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;

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

    let backend_request = crate::backend::types::CompletionRequest {
        prompt: request.prompt,
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: is_stream,
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
                                let resp = CompletionResponse {
                                    id: id.clone(),
                                    object: "text_completion".to_string(),
                                    created,
                                    model: model.clone(),
                                    choices: vec![CompletionChoice {
                                        index: 0,
                                        text: c.text,
                                        finish_reason: if c.done {
                                            Some("stop".to_string())
                                        } else {
                                            None
                                        },
                                    }],
                                    usage: Usage {
                                        prompt_tokens: 0,
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
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
                            if !c.done {
                                completion_tokens += 1;
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
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Usage {
                        prompt_tokens: 0,
                        completion_tokens,
                        total_tokens: completion_tokens,
                    },
                })
                .into_response()
            }
        }
        Err(e) => openai_error("server_error", &e.to_string()).into_response(),
    }
}

fn openai_error(code: &str, message: &str) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": code
        }
    }))
}
