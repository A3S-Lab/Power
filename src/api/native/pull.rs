use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::types::{PullRequest, PullResponse};
use crate::model::resolve;
use crate::server::state::AppState;

/// POST /api/pull - Pull/download a model (Ollama-compatible).
///
/// Supports streaming progress updates via NDJSON.
/// Shows per-layer progress with digest identifiers, matching Ollama's output:
///   pulling manifest
///   pulling sha256:abc123... (model weights)
///   pulling sha256:def456... (projector, if present)
///   verifying sha256:...
///   writing manifest
///   success
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<PullRequest>,
) -> impl IntoResponse {
    let model_name = request.name.clone();
    let is_stream = request.stream.unwrap_or(true);

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
        // Streaming progress via NDJSON
        let registry = state.registry.clone();
        let (tx, rx) = mpsc::channel::<PullResponse>(32);
        let name_or_url = model_name.clone();

        tokio::spawn(async move {
            // Step 1: Pulling manifest
            let _ = tx
                .send(PullResponse {
                    status: "pulling manifest".to_string(),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .await;

            // Resolve the model to get layer digests for per-layer progress
            let resolved = if !resolve::is_url(&name_or_url) {
                match resolve::resolve(&name_or_url).await {
                    Ok(r) => Some(r),
                    Err(e) => {
                        let _ = tx
                            .send(PullResponse {
                                status: format!("error: {e}"),
                                digest: None,
                                total: None,
                                completed: None,
                            })
                            .await;
                        return;
                    }
                }
            } else {
                None
            };

            // Extract layer digest for per-layer progress display
            let layer_digest = resolved.as_ref().and_then(|r| match &r.source {
                resolve::ModelSource::OllamaRegistry(reg) => {
                    Some(reg.model_digest.clone())
                }
                _ => None,
            });

            // Step 2: Show per-layer pulling status
            if let Some(ref digest) = layer_digest {
                let short = truncate_digest(digest);
                let _ = tx
                    .send(PullResponse {
                        status: format!("pulling {short}"),
                        digest: Some(digest.clone()),
                        total: None,
                        completed: None,
                    })
                    .await;
            }

            // Step 3: Download with progress
            let progress_tx = tx.clone();
            let progress_digest = layer_digest.clone();
            let progress = Box::new(move |downloaded: u64, total: u64| {
                let _ = progress_tx.try_send(PullResponse {
                    status: "downloading".to_string(),
                    digest: progress_digest.clone(),
                    total: Some(total),
                    completed: Some(downloaded),
                });
            });

            let pull_result = if let Some(resolved) = resolved {
                // Use the already-resolved URL
                crate::model::pull::pull_model(
                    &name_or_url,
                    Some(&resolved.url),
                    Some(progress),
                )
                .await
            } else {
                crate::model::pull::pull_model(&name_or_url, None, Some(progress)).await
            };

            match pull_result {
                Ok(manifest) => {
                    let digest = format!("sha256:{}", &manifest.sha256);
                    let size = manifest.size;

                    // Step 4: Verifying
                    let _ = tx
                        .send(PullResponse {
                            status: format!("verifying {}", truncate_digest(&digest)),
                            digest: Some(digest.clone()),
                            total: Some(size),
                            completed: Some(size),
                        })
                        .await;

                    // Step 5: Writing manifest
                    let _ = tx
                        .send(PullResponse {
                            status: "writing manifest".to_string(),
                            digest: Some(digest.clone()),
                            total: Some(size),
                            completed: Some(size),
                        })
                        .await;

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

/// Truncate a digest for display (e.g. "sha256:abc123def456..." → "sha256:abc123def4...").
fn truncate_digest(digest: &str) -> String {
    if digest.len() > 19 {
        format!("{}...", &digest[..19])
    } else {
        digest.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    #[serial]
    async fn test_pull_already_exists_returns_success() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("existing-model"))
            .unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/pull")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"existing-model"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "success");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_pull_streaming_returns_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/pull")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name":"nonexistent-model","stream":true}"#))
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
    async fn test_pull_request_deserializes_stream_default() {
        // When stream is not specified, it defaults to false
        let json = r#"{"name":"test-model"}"#;
        let req: crate::api::types::PullRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "test-model");
        assert_eq!(req.stream, None);
    }

    #[test]
    fn test_truncate_digest_long() {
        let digest = "sha256:abc123def456789012345";
        let truncated = truncate_digest(digest);
        assert_eq!(truncated, "sha256:abc123def456...");
    }

    #[test]
    fn test_truncate_digest_short() {
        let digest = "sha256:abc";
        let truncated = truncate_digest(digest);
        assert_eq!(truncated, "sha256:abc");
    }
}
