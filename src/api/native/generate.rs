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

    if let Err(e) = crate::api::autoload::ensure_loaded(&state, &model_name, &manifest, &backend).await {
        return Json(serde_json::json!({ "error": e.to_string() })).into_response();
    }

    let backend_request = crate::backend::types::CompletionRequest {
        prompt: request.prompt,
        temperature: request.options.as_ref().and_then(|o| o.temperature),
        top_p: request.options.as_ref().and_then(|o| o.top_p),
        max_tokens: request.options.as_ref().and_then(|o| o.num_predict),
        stop: request.options.as_ref().and_then(|o| o.stop.clone()),
        stream: request.stream.unwrap_or(false),
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
                                total_duration: if c.done {
                                    Some(start.elapsed().as_nanos() as u64)
                                } else {
                                    None
                                },
                                load_duration: None,
                                eval_count: None,
                                eval_duration: None,
                            },
                            Err(e) => GenerateResponse {
                                model: model_name_owned.clone(),
                                response: format!("Error: {e}"),
                                done: true,
                                total_duration: None,
                                load_duration: None,
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
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(c) => {
                            full_text.push_str(&c.text);
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
                let total_duration = start.elapsed().as_nanos() as u64;
                Json(GenerateResponse {
                    model: model_name,
                    response: full_text,
                    done: true,
                    total_duration: Some(total_duration),
                    load_duration: None,
                    eval_count: Some(eval_count),
                    eval_duration: Some(total_duration),
                })
                .into_response()
            }
        }
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })).into_response(),
    }
}
