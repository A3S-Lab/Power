use crate::error::{PowerError, Result};
use crate::model::manifest::ModelManifest;

/// Callback for reporting push progress: (completed_bytes, total_bytes).
pub type ProgressCallback = Box<dyn Fn(u64, u64) + Send + Sync>;

/// Push a model's blob and manifest to a remote registry.
///
/// The protocol follows the Docker Registry v2 pattern:
/// 1. Check if blob exists on remote (HEAD)
/// 2. Upload blob if missing (POST)
/// 3. Upload manifest (POST)
///
/// When `insecure` is true, TLS certificate verification is skipped.
pub async fn push_model(
    manifest: &ModelManifest,
    destination: &str,
    progress: Option<ProgressCallback>,
    insecure: bool,
) -> Result<String> {
    let mut builder = reqwest::Client::builder();
    if insecure {
        builder = builder.danger_accept_invalid_certs(true);
        tracing::warn!("TLS certificate verification disabled for push (insecure mode)");
    }
    let client = builder
        .build()
        .map_err(|e| PowerError::UploadFailed(format!("Failed to build HTTP client: {e}")))?;
    let digest = format!("sha256:{}", manifest.sha256);
    let blob_size = manifest.size;

    // 1. Check if blob already exists on remote
    let check_url = format!("{}/api/blobs/{}", destination.trim_end_matches('/'), digest);
    let check_resp =
        client.head(&check_url).send().await.map_err(|e| {
            PowerError::UploadFailed(format!("Failed to check blob on remote: {e}"))
        })?;

    if check_resp.status() == reqwest::StatusCode::NOT_FOUND {
        // 2. Read blob from disk and upload
        let blob_data = tokio::fs::read(&manifest.path).await.map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to read blob {}: {e}",
                manifest.path.display()
            )))
        })?;

        let upload_url = format!("{}/api/blobs/{}", destination.trim_end_matches('/'), digest);
        let upload_resp = client
            .post(&upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(blob_data)
            .send()
            .await
            .map_err(|e| PowerError::UploadFailed(format!("Failed to upload blob: {e}")))?;

        if !upload_resp.status().is_success() {
            return Err(PowerError::UploadFailed(format!(
                "Remote rejected blob upload: HTTP {}",
                upload_resp.status()
            )));
        }

        if let Some(ref cb) = progress {
            cb(blob_size, blob_size);
        }
    } else if let Some(ref cb) = progress {
        // Blob already exists, report as complete
        cb(blob_size, blob_size);
    }

    // 3. Upload manifest
    let manifest_url = format!(
        "{}/api/manifests/{}",
        destination.trim_end_matches('/'),
        manifest.name
    );
    let manifest_json = serde_json::to_string(manifest).map_err(PowerError::Serialization)?;
    let manifest_resp = client
        .post(&manifest_url)
        .header("Content-Type", "application/json")
        .body(manifest_json)
        .send()
        .await
        .map_err(|e| PowerError::UploadFailed(format!("Failed to upload manifest: {e}")))?;

    if !manifest_resp.status().is_success() {
        return Err(PowerError::UploadFailed(format!(
            "Remote rejected manifest upload: HTTP {}",
            manifest_resp.status()
        )));
    }

    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::test_utils::sample_manifest;

    #[test]
    fn test_progress_callback_type() {
        // Verify the callback type compiles and can be constructed
        let _cb: ProgressCallback = Box::new(|completed, total| {
            assert!(completed <= total);
        });
    }

    #[test]
    fn test_progress_callback_invocation() {
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        let cb: ProgressCallback = Box::new(move |completed, total| {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(completed, 100);
            assert_eq!(total, 100);
        });
        cb(100, 100);
        assert!(called.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_push_model_connection_refused() {
        let manifest = sample_manifest("test-push");
        let result = push_model(&manifest, "http://localhost:1", None, false).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to check blob") || err.contains("Upload failed"));
    }

    #[tokio::test]
    async fn test_push_model_invalid_destination() {
        let manifest = sample_manifest("test-push");
        let result = push_model(&manifest, "not-a-url", None, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_push_model_with_progress_callback() {
        let manifest = sample_manifest("test-push");
        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let called_clone = called.clone();
        let progress: ProgressCallback = Box::new(move |_completed, _total| {
            called_clone.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // Will fail due to connection, but progress callback type is valid
        let _result = push_model(&manifest, "http://localhost:1", Some(progress), false).await;
        // Connection fails before progress is called, so we just verify it compiles
    }

    #[test]
    fn test_progress_callback_with_different_values() {
        let cb: ProgressCallback = Box::new(|completed, total| {
            assert!(completed <= total);
        });
        cb(0, 100);
        cb(50, 100);
        cb(100, 100);
    }

    #[test]
    fn test_progress_callback_zero_total() {
        let cb: ProgressCallback = Box::new(|completed, total| {
            assert_eq!(completed, 0);
            assert_eq!(total, 0);
        });
        cb(0, 0);
    }

    #[test]
    fn test_progress_callback_large_values() {
        let cb: ProgressCallback = Box::new(|completed, total| {
            assert_eq!(completed, 4_000_000_000);
            assert_eq!(total, 4_000_000_000);
        });
        cb(4_000_000_000, 4_000_000_000);
    }

    #[tokio::test]
    async fn test_push_model_with_trailing_slash() {
        let manifest = sample_manifest("test-push");
        let result = push_model(&manifest, "http://localhost:1/", None, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_push_model_empty_destination() {
        let manifest = sample_manifest("test-push");
        let result = push_model(&manifest, "", None, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_push_model_insecure_mode() {
        let manifest = sample_manifest("test-push");
        // Insecure mode should still fail on connection, but the client builds successfully
        let result = push_model(&manifest, "http://localhost:1", None, true).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Failed to check blob") || err.contains("Upload failed"));
    }
}
