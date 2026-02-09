use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::types::{PullRequest, PullResponse};
use crate::server::state::AppState;

/// POST /api/pull - Pull/download a model (Ollama-compatible).
///
/// Supports streaming progress updates via SSE.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<PullRequest>,
) -> impl IntoResponse {
    let model_name = request.name.clone();
    let is_stream = request.stream.unwrap_or(false);

    if state.registry.exists(&model_name) {
        return Json(PullResponse {
            status: "success".to_string(),
            digest: None,
            total: None,
            completed: None,
        })
        .into_response();
    }

    if is_stream {
        // Streaming progress via SSE
        let registry = state.registry.clone();
        let (tx, rx) = mpsc::channel::<PullResponse>(32);
        let name_or_url = model_name.clone();

        tokio::spawn(async move {
            let _ = tx
                .send(PullResponse {
                    status: "pulling".to_string(),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .await;

            let progress_tx = tx.clone();
            let progress = Box::new(move |downloaded: u64, total: u64| {
                let _ = progress_tx.try_send(PullResponse {
                    status: "downloading".to_string(),
                    digest: None,
                    total: Some(total),
                    completed: Some(downloaded),
                });
            });

            match crate::model::pull::pull_model(&name_or_url, None, Some(progress)).await {
                Ok(manifest) => {
                    let digest = format!("sha256:{}", &manifest.sha256);
                    let size = manifest.size;
                    match registry.register(manifest) {
                        Ok(()) => {
                            let _ = tx
                                .send(PullResponse {
                                    status: "success".to_string(),
                                    digest: Some(digest),
                                    total: Some(size),
                                    completed: Some(size),
                                })
                                .await;
                        }
                        Err(e) => {
                            let _ = tx
                                .send(PullResponse {
                                    status: format!("error: {e}"),
                                    digest: None,
                                    total: None,
                                    completed: None,
                                })
                                .await;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(PullResponse {
                            status: format!("error: {e}"),
                            digest: None,
                            total: None,
                            completed: None,
                        })
                        .await;
                }
            }
        });

        let event_stream = ReceiverStream::new(rx);

        crate::api::sse::ndjson_response(event_stream)
    } else {
        // Non-streaming: download and return final status
        match crate::model::pull::pull_model(&model_name, None, None).await {
            Ok(manifest) => {
                let digest = format!("sha256:{}", &manifest.sha256);
                let size = manifest.size;
                if let Err(e) = state.registry.register(manifest) {
                    return Json(PullResponse {
                        status: format!("error: {e}"),
                        digest: None,
                        total: None,
                        completed: None,
                    })
                    .into_response();
                }
                Json(PullResponse {
                    status: "success".to_string(),
                    digest: Some(digest),
                    total: Some(size),
                    completed: Some(size),
                })
                .into_response()
            }
            Err(e) => Json(PullResponse {
                status: format!("error: {e}"),
                digest: None,
                total: None,
                completed: None,
            })
            .into_response(),
        }
    }
}
