use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;

use crate::api::types::{ChatCompletionMessage, NativeChatRequest, NativeChatResponse};
use crate::backend::types::{ChatMessage, ChatRequest};
use crate::server::state::AppState;

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

    if let Err(e) = crate::api::autoload::ensure_loaded(&state, &model_name, &manifest, &backend).await {
        return Json(serde_json::json!({ "error": e.to_string() })).into_response();
    }

    let backend_request = ChatRequest {
        messages: request
            .messages
            .iter()
            .map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect(),
        temperature: request.options.as_ref().and_then(|o| o.temperature),
        top_p: request.options.as_ref().and_then(|o| o.top_p),
        max_tokens: request.options.as_ref().and_then(|o| o.num_predict),
        stop: request.options.as_ref().and_then(|o| o.stop.clone()),
        stream: request.stream.unwrap_or(false),
    };

    let is_stream = request.stream.unwrap_or(false);

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let model_name_owned = model_name.clone();
                let sse_stream = stream
                    .map(move |chunk| {
                        let resp = match chunk {
                            Ok(c) => NativeChatResponse {
                                model: model_name_owned.clone(),
                                message: ChatCompletionMessage {
                                    role: "assistant".to_string(),
                                    content: c.content,
                                },
                                done: c.done,
                                total_duration: None,
                                eval_count: None,
                            },
                            Err(e) => NativeChatResponse {
                                model: model_name_owned.clone(),
                                message: ChatCompletionMessage {
                                    role: "assistant".to_string(),
                                    content: format!("Error: {e}"),
                                },
                                done: true,
                                total_duration: None,
                                eval_count: None,
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
                let mut full_content = String::new();
                let mut eval_count: u32 = 0;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_content.push_str(&c.content);
                            if !c.done {
                                eval_count += 1;
                            }
                        }
                        Err(e) => {
                            return Json(serde_json::json!({ "error": e.to_string() }))
                                .into_response();
                        }
                    }
                }
                Json(NativeChatResponse {
                    model: model_name,
                    message: ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: full_content,
                    },
                    done: true,
                    total_duration: None,
                    eval_count: Some(eval_count),
                })
                .into_response()
            }
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}
