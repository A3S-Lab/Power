use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::types::{PushRequest, PushResponse};
use crate::server::state::AppState;

/// POST /api/push - Push a model to a remote registry.
pub async fn handler(
    State(state): State<AppState>,
    Json(request): Json<PushRequest>,
) -> impl IntoResponse {
    let manifest = match state.registry.get(&request.name) {
        Ok(m) => m,
        Err(_) => {
            return Json(PushResponse {
                status: format!("error: model '{}' not found", request.name),
                digest: None,
                total: None,
                completed: None,
            })
            .into_response();
        }
    };

    let is_stream = request.stream.unwrap_or(false);

    if is_stream {
        // Streaming progress via SSE
        let (tx, rx) = mpsc::channel::<PushResponse>(32);
        let destination = request.destination.clone();

        tokio::spawn(async move {
            let _ = tx
                .send(PushResponse {
                    status: "pushing".to_string(),
                    digest: None,
                    total: Some(manifest.size),
                    completed: Some(0),
                })
                .await;

            let progress_tx = tx.clone();
            let progress = Box::new(move |completed: u64, total: u64| {
                let _ = progress_tx.try_send(PushResponse {
                    status: "uploading".to_string(),
                    digest: None,
                    total: Some(total),
                    completed: Some(completed),
                });
            });

            match crate::model::push::push_model(&manifest, &destination, Some(progress)).await {
                Ok(digest) => {
                    let _ = tx
                        .send(PushResponse {
                            status: "success".to_string(),
                            digest: Some(digest),
                            total: Some(manifest.size),
                            completed: Some(manifest.size),
                        })
                        .await;
                }
                Err(e) => {
                    let _ = tx
                        .send(PushResponse {
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
        // Non-streaming: push and return final status
        match crate::model::push::push_model(&manifest, &request.destination, None).await {
            Ok(digest) => Json(PushResponse {
                status: "success".to_string(),
                digest: Some(digest),
                total: Some(manifest.size),
                completed: Some(manifest.size),
            })
            .into_response(),
            Err(e) => Json(PushResponse {
                status: format!("error: {e}"),
                digest: None,
                total: None,
                completed: None,
            })
            .into_response(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::{sample_manifest, test_state_with_mock, MockBackend};
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_push_model_not_found() {
        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/push")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"name":"nonexistent","destination":"http://localhost:9999"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["status"].as_str().unwrap().contains("not found"));
    }

    #[tokio::test]
    #[serial]
    async fn test_push_connection_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("test-push"))
            .unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/push")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"name":"test-push","destination":"http://localhost:1"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["status"].as_str().unwrap().contains("error"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_push_streaming_returns_ndjson() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        state
            .registry
            .register(sample_manifest("test-push-stream"))
            .unwrap();

        let app = router::build(state);
        let req = Request::builder()
            .method("POST")
            .uri("/api/push")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"name":"test-push-stream","destination":"http://localhost:1","stream":true}"#,
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
}
