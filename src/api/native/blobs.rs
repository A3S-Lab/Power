use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;

use crate::dirs;
use crate::model::storage;
use crate::server::state::AppState;

/// HEAD /api/blobs/:digest - Check if a blob exists.
pub async fn check_handler(
    State(_state): State<AppState>,
    Path(digest): Path<String>,
) -> impl IntoResponse {
    let hash = digest.strip_prefix("sha256:").unwrap_or(&digest);
    let blob_path = dirs::blobs_dir().join(format!("sha256-{}", hash));

    if blob_path.exists() {
        match tokio::fs::metadata(&blob_path).await {
            Ok(meta) => (
                StatusCode::OK,
                [(header::CONTENT_LENGTH, meta.len().to_string())],
            )
                .into_response(),
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        }
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

/// POST /api/blobs/:digest - Upload a blob with digest verification.
pub async fn upload_handler(
    State(_state): State<AppState>,
    Path(digest): Path<String>,
    body: Bytes,
) -> impl IntoResponse {
    let hash = digest.strip_prefix("sha256:").unwrap_or(&digest);

    // Verify the hash matches the uploaded data
    let computed_hash = storage::compute_sha256(&body);
    if computed_hash != hash {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": format!(
                    "Digest mismatch: expected sha256:{}, got sha256:{}",
                    hash, computed_hash
                )
            })),
        )
            .into_response();
    }

    match storage::store_blob(&body) {
        Ok((_path, _stored_hash)) => StatusCode::CREATED.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to store blob: {e}") })),
        )
            .into_response(),
    }
}

/// GET /api/blobs/:digest - Download a blob.
pub async fn download_handler(
    State(_state): State<AppState>,
    Path(digest): Path<String>,
) -> impl IntoResponse {
    let hash = digest.strip_prefix("sha256:").unwrap_or(&digest);
    let blob_path = dirs::blobs_dir().join(format!("sha256-{}", hash));

    match tokio::fs::read(&blob_path).await {
        Ok(data) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/octet-stream".to_string())],
            data,
        )
            .into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

/// DELETE /api/blobs/:digest - Delete a blob.
pub async fn delete_handler(
    State(_state): State<AppState>,
    Path(digest): Path<String>,
) -> impl IntoResponse {
    let hash = digest.strip_prefix("sha256:").unwrap_or(&digest);
    let blob_path = dirs::blobs_dir().join(format!("sha256-{}", hash));

    if !blob_path.exists() {
        return StatusCode::NOT_FOUND.into_response();
    }

    match tokio::fs::remove_file(&blob_path).await {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": format!("Failed to delete blob: {e}") })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::test_state_with_mock;
    use crate::backend::test_utils::MockBackend;
    use crate::model::storage;
    use crate::server::router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use serial_test::serial;
    use tower::util::ServiceExt;

    #[tokio::test]
    #[serial]
    async fn test_check_blob_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);
        let req = Request::builder()
            .method("HEAD")
            .uri("/api/blobs/sha256:0000000000000000000000000000000000000000000000000000000000000000")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_upload_and_check_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"test blob data for upload";
        let hash = storage::compute_sha256(data);

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state.clone());

        // Upload
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::from(data.to_vec()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Check exists
        let app = router::build(state);
        let req = Request::builder()
            .method("HEAD")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_upload_blob_digest_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);

        let req = Request::builder()
            .method("POST")
            .uri("/api/blobs/sha256:wronghash")
            .body(Body::from("some data"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_download_blob_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);

        let req = Request::builder()
            .method("GET")
            .uri("/api/blobs/sha256:0000000000000000000000000000000000000000000000000000000000000000")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_upload_and_download_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"download me later";
        let hash = storage::compute_sha256(data);

        let state = test_state_with_mock(MockBackend::success());

        // Upload
        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::from(data.to_vec()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Download
        let app = router::build(state);
        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(body.as_ref(), data);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_delete_blob_not_found() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let state = test_state_with_mock(MockBackend::success());
        let app = router::build(state);

        let req = Request::builder()
            .method("DELETE")
            .uri("/api/blobs/sha256:0000000000000000000000000000000000000000000000000000000000000000")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[tokio::test]
    #[serial]
    async fn test_upload_and_delete_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"delete me later";
        let hash = storage::compute_sha256(data);

        let state = test_state_with_mock(MockBackend::success());

        // Upload
        let app = router::build(state.clone());
        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::from(data.to_vec()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Delete
        let app = router::build(state.clone());
        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify gone
        let app = router::build(state);
        let req = Request::builder()
            .method("HEAD")
            .uri(format!("/api/blobs/sha256:{}", hash))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        std::env::remove_var("A3S_POWER_HOME");
    }
}
