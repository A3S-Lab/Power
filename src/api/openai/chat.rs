use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use std::convert::Infallible;

use super::openai_error;
use crate::api::types::{
    ChatChoice, ChatChunkChoice, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatDelta, Usage,
};
use crate::backend::types::{ChatMessage, ChatRequest};
use crate::server::state::AppState;

/// POST /v1/chat/completions - OpenAI-compatible chat completion.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    let model_name = request.model.clone();
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

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

    if let Err(e) = crate::api::autoload::ensure_loaded(&state, &model_name, &manifest, &backend).await {
        return openai_error("server_error", &e.to_string()).into_response();
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
        temperature: request.temperature,
        top_p: request.top_p,
        max_tokens: request.max_tokens,
        stop: request.stop.clone(),
        stream: request.stream.unwrap_or(false),
    };

    let is_stream = request.stream.unwrap_or(false);

    match backend.chat(&model_name, backend_request).await {
        Ok(stream) => {
            if is_stream {
                let id = request_id.clone();
                let model = model_name.clone();
                let created = chrono::Utc::now().timestamp();

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
                        },
                        finish_reason: None,
                    }],
                };

                let first_event = futures::stream::once(async move {
                    let data = serde_json::to_string(&first_chunk).unwrap_or_default();
                    Ok::<_, Infallible>(Event::default().data(data))
                });

                let content_stream = stream.map(move |chunk| {
                    let event_data = match chunk {
                        Ok(c) => {
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
                                    },
                                    finish_reason: if c.done {
                                        Some("stop".to_string())
                                    } else {
                                        None
                                    },
                                }],
                            };
                            serde_json::to_string(&chunk_resp).unwrap_or_default()
                        }
                        Err(e) => serde_json::to_string(&serde_json::json!({
                            "error": { "message": e.to_string() }
                        }))
                        .unwrap_or_default(),
                    };
                    Ok::<_, Infallible>(Event::default().data(event_data))
                });

                let done_event = futures::stream::once(async {
                    Ok::<_, Infallible>(Event::default().data("[DONE]"))
                });

                let full_stream = first_event.chain(content_stream).chain(done_event);

                Sse::new(full_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            } else {
                // Non-streaming: collect full response
                let mut full_content = String::new();
                let mut completion_tokens: u32 = 0;
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_content.push_str(&c.content);
                            if !c.done {
                                completion_tokens += 1;
                            }
                        }
                        Err(e) => {
                            return openai_error("server_error", &e.to_string()).into_response();
                        }
                    }
                }

                Json(ChatCompletionResponse {
                    id: request_id,
                    object: "chat.completion".to_string(),
                    created: chrono::Utc::now().timestamp(),
                    model: model_name,
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatCompletionMessage {
                            role: "assistant".to_string(),
                            content: full_content,
                        },
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
