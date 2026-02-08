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
pub async fn push_model(
    manifest: &ModelManifest,
    destination: &str,
    progress: Option<ProgressCallback>,
) -> Result<String> {
    let client = reqwest::Client::new();
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

    #[test]
    fn test_progress_callback_type() {
        // Verify the callback type compiles and can be constructed
        let _cb: ProgressCallback = Box::new(|completed, total| {
            assert!(completed <= total);
        });
    }
}
