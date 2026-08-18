use std::fs::File as StdFile;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{
    remote_url_is_allowed, ArtifactBundle, ArtifactBundleError, ArtifactSource, BundleArtifact,
    BundleProvisionPolicy, ProvisionedArtifactBundle, LOCK_NAME, RECEIPT_NAME,
};

/// Provision a complete bundle, or reuse it without network access after
/// independently re-verifying every artifact.
pub async fn provision_artifact_bundle(
    bundle: &ArtifactBundle,
    policy: &BundleProvisionPolicy,
) -> Result<ProvisionedArtifactBundle, ArtifactBundleError> {
    validate_policy(bundle, policy)?;
    ensure_directory(&policy.destination).await?;
    let _lock = acquire_lock(policy.destination.join(LOCK_NAME)).await?;

    let client = if policy.allow_network {
        Some(
            reqwest::Client::builder()
                .timeout(policy.request_timeout)
                .redirect(reqwest::redirect::Policy::custom(|attempt| {
                    if attempt.previous().len() >= 5 || !remote_url_is_allowed(attempt.url()) {
                        attempt.stop()
                    } else {
                        attempt.follow()
                    }
                }))
                .build()
                .map_err(|_| ArtifactBundleError::Network {
                    artifact: "bundle".to_string(),
                    reason: "HTTP client construction failed",
                })?,
        )
    } else {
        None
    };
    let mut installed_artifacts = 0;
    let mut reused_artifacts = 0;
    let mut receipt_artifacts = Vec::with_capacity(bundle.artifacts.len());

    for artifact in &bundle.artifacts {
        let target = policy.destination.join(&artifact.name);
        let size = match verify_existing(&target, artifact).await? {
            Some(size) => {
                reused_artifacts += 1;
                size
            }
            None => {
                let size = match &artifact.source {
                    ArtifactSource::Inline(bytes) => {
                        write_inline_artifact(&target, artifact, bytes).await?
                    }
                    ArtifactSource::Remote(url) => {
                        let client =
                            client
                                .as_ref()
                                .ok_or_else(|| ArtifactBundleError::OfflineMissing {
                                    artifact: artifact.name.clone(),
                                })?;
                        download_artifact(client, &target, artifact, url).await?
                    }
                };
                installed_artifacts += 1;
                size
            }
        };
        receipt_artifacts.push(BundleReceiptArtifact {
            name: &artifact.name,
            sha256: &artifact.sha256,
            bytes: size,
        });
    }

    let receipt = BundleReceipt {
        schema: "a3s.power.artifact-bundle.v1",
        name: &bundle.name,
        revision: &bundle.revision,
        artifacts: receipt_artifacts,
    };
    let receipt = serde_json::to_vec(&receipt).map_err(|error| ArtifactBundleError::Io {
        operation: "serializing the bundle receipt",
        source: std::io::Error::other(error),
    })?;
    write_receipt(&policy.destination.join(RECEIPT_NAME), &receipt).await?;

    Ok(ProvisionedArtifactBundle {
        root: policy.destination.clone(),
        installed_artifacts,
        reused_artifacts,
    })
}

#[derive(Serialize)]
struct BundleReceipt<'a> {
    schema: &'static str,
    name: &'a str,
    revision: &'a str,
    artifacts: Vec<BundleReceiptArtifact<'a>>,
}

#[derive(Serialize)]
struct BundleReceiptArtifact<'a> {
    name: &'a str,
    sha256: &'a str,
    bytes: u64,
}

fn validate_policy(
    bundle: &ArtifactBundle,
    policy: &BundleProvisionPolicy,
) -> Result<(), ArtifactBundleError> {
    if policy.destination.as_os_str().is_empty() {
        return Err(ArtifactBundleError::Invalid(
            "bundle destination must not be empty".to_string(),
        ));
    }
    if policy.request_timeout.is_zero() {
        return Err(ArtifactBundleError::Invalid(
            "request timeout must be positive".to_string(),
        ));
    }
    let total = bundle.artifacts.iter().try_fold(0_u64, |total, artifact| {
        total.checked_add(artifact.max_bytes).ok_or_else(|| {
            ArtifactBundleError::Invalid("bundle admission size overflowed u64".to_string())
        })
    })?;
    if policy.max_total_bytes == 0 || total > policy.max_total_bytes {
        return Err(ArtifactBundleError::Invalid(format!(
            "bundle admission limit is {} bytes but artifacts allow {total} bytes",
            policy.max_total_bytes
        )));
    }
    Ok(())
}

async fn ensure_directory(path: &Path) -> Result<(), ArtifactBundleError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(ArtifactBundleError::Invalid(
                "bundle destination must be a real directory".to_string(),
            ));
        }
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(ArtifactBundleError::Io {
                operation: "inspecting the bundle destination",
                source,
            });
        }
    }
    tokio::fs::create_dir_all(path)
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "creating the bundle destination",
            source,
        })?;
    let metadata =
        tokio::fs::symlink_metadata(path)
            .await
            .map_err(|source| ArtifactBundleError::Io {
                operation: "verifying the bundle destination",
                source,
            })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ArtifactBundleError::Invalid(
            "bundle destination must be a real directory".to_string(),
        ));
    }
    Ok(())
}

async fn acquire_lock(path: PathBuf) -> Result<StdFile, ArtifactBundleError> {
    tokio::task::spawn_blocking(move || {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)
            .map_err(|source| ArtifactBundleError::Io {
                operation: "opening the bundle lock",
                source,
            })?;
        FileExt::lock_exclusive(&file).map_err(|source| ArtifactBundleError::Io {
            operation: "acquiring the bundle lock",
            source,
        })?;
        Ok(file)
    })
    .await
    .map_err(|_| ArtifactBundleError::BlockingTask)?
}

async fn verify_existing(
    path: &Path,
    artifact: &BundleArtifact,
) -> Result<Option<u64>, ArtifactBundleError> {
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(ArtifactBundleError::Io {
                operation: "inspecting an installed artifact",
                source,
            });
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ArtifactBundleError::Invalid(format!(
            "installed artifact '{}' must be a real file",
            artifact.name
        )));
    }
    if metadata.len() > artifact.max_bytes {
        return Err(ArtifactBundleError::TooLarge {
            artifact: artifact.name.clone(),
            maximum_bytes: artifact.max_bytes,
        });
    }
    let (sha256, bytes) = hash_file(path, artifact.max_bytes, &artifact.name).await?;
    if sha256 != artifact.sha256 {
        return Err(ArtifactBundleError::Integrity {
            artifact: artifact.name.clone(),
        });
    }
    Ok(Some(bytes))
}

async fn hash_file(
    path: &Path,
    maximum_bytes: u64,
    artifact: &str,
) -> Result<(String, u64), ArtifactBundleError> {
    let mut file = tokio::fs::File::open(path)
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "opening an installed artifact",
            source,
        })?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    let mut bytes = 0_u64;
    loop {
        let read = file
            .read(&mut buffer)
            .await
            .map_err(|source| ArtifactBundleError::Io {
                operation: "reading an installed artifact",
                source,
            })?;
        if read == 0 {
            break;
        }
        bytes = bytes
            .checked_add(u64::try_from(read).unwrap_or(u64::MAX))
            .ok_or_else(|| ArtifactBundleError::TooLarge {
                artifact: artifact.to_string(),
                maximum_bytes,
            })?;
        if bytes > maximum_bytes {
            return Err(ArtifactBundleError::TooLarge {
                artifact: artifact.to_string(),
                maximum_bytes,
            });
        }
        digest.update(&buffer[..read]);
    }
    Ok((format!("{:x}", digest.finalize()), bytes))
}

async fn write_inline_artifact(
    target: &Path,
    artifact: &BundleArtifact,
    bytes: &[u8],
) -> Result<u64, ArtifactBundleError> {
    let (staging_path, mut staging) = create_staging_file(target).await?;
    let result = async {
        staging
            .write_all(bytes)
            .await
            .map_err(|source| ArtifactBundleError::Io {
                operation: "writing an inline artifact",
                source,
            })?;
        sync_staging(&mut staging).await?;
        Ok(u64::try_from(bytes.len()).unwrap_or(u64::MAX))
    }
    .await;
    drop(staging);
    finish_staging(result, &staging_path, target, artifact).await
}

async fn download_artifact(
    client: &reqwest::Client,
    target: &Path,
    artifact: &BundleArtifact,
    url: &str,
) -> Result<u64, ArtifactBundleError> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| ArtifactBundleError::Network {
            artifact: artifact.name.clone(),
            reason: network_error_reason(&error),
        })?;
    if !response.status().is_success() {
        return Err(ArtifactBundleError::HttpStatus {
            artifact: artifact.name.clone(),
            status: response.status().as_u16(),
        });
    }
    if response
        .content_length()
        .is_some_and(|length| length > artifact.max_bytes)
    {
        return Err(ArtifactBundleError::TooLarge {
            artifact: artifact.name.clone(),
            maximum_bytes: artifact.max_bytes,
        });
    }

    let (staging_path, mut staging) = create_staging_file(target).await?;
    let result = download_response(response, &mut staging, artifact).await;
    drop(staging);
    finish_staging(result, &staging_path, target, artifact).await
}

async fn download_response(
    mut response: reqwest::Response,
    staging: &mut tokio::fs::File,
    artifact: &BundleArtifact,
) -> Result<u64, ArtifactBundleError> {
    let mut digest = Sha256::new();
    let mut bytes = 0_u64;
    while let Some(chunk) =
        response
            .chunk()
            .await
            .map_err(|error| ArtifactBundleError::Network {
                artifact: artifact.name.clone(),
                reason: network_error_reason(&error),
            })?
    {
        bytes = bytes
            .checked_add(u64::try_from(chunk.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| ArtifactBundleError::TooLarge {
                artifact: artifact.name.clone(),
                maximum_bytes: artifact.max_bytes,
            })?;
        if bytes > artifact.max_bytes {
            return Err(ArtifactBundleError::TooLarge {
                artifact: artifact.name.clone(),
                maximum_bytes: artifact.max_bytes,
            });
        }
        staging
            .write_all(&chunk)
            .await
            .map_err(|source| ArtifactBundleError::Io {
                operation: "writing a downloaded artifact",
                source,
            })?;
        digest.update(&chunk);
    }
    if format!("{:x}", digest.finalize()) != artifact.sha256 {
        return Err(ArtifactBundleError::Integrity {
            artifact: artifact.name.clone(),
        });
    }
    sync_staging(staging).await?;
    Ok(bytes)
}

async fn create_staging_file(
    target: &Path,
) -> Result<(PathBuf, tokio::fs::File), ArtifactBundleError> {
    let parent = target.parent().ok_or_else(|| {
        ArtifactBundleError::Invalid("artifact target has no parent directory".to_string())
    })?;
    let name = target
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| ArtifactBundleError::Invalid("artifact name must be UTF-8".to_string()))?;
    for _ in 0..16 {
        let path = parent.join(format!(".{name}.{:016x}.partial", rand::random::<u64>()));
        match tokio::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .await
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(ArtifactBundleError::Io {
                    operation: "creating an artifact staging file",
                    source,
                });
            }
        }
    }
    Err(ArtifactBundleError::Io {
        operation: "allocating a unique artifact staging file",
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "staging namespace exhausted",
        ),
    })
}

async fn sync_staging(staging: &mut tokio::fs::File) -> Result<(), ArtifactBundleError> {
    staging
        .flush()
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "synchronizing an artifact staging file",
            source,
        })?;
    staging
        .sync_all()
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "synchronizing an artifact staging file",
            source,
        })
}

async fn finish_staging(
    result: Result<u64, ArtifactBundleError>,
    staging_path: &Path,
    target: &Path,
    artifact: &BundleArtifact,
) -> Result<u64, ArtifactBundleError> {
    let bytes = match result {
        Ok(bytes) => bytes,
        Err(error) => {
            let _ = tokio::fs::remove_file(staging_path).await;
            return Err(error);
        }
    };
    if tokio::fs::symlink_metadata(target).await.is_ok() {
        let _ = tokio::fs::remove_file(staging_path).await;
        return Err(ArtifactBundleError::Invalid(format!(
            "artifact '{}' appeared while provisioning was locked",
            artifact.name
        )));
    }
    tokio::fs::rename(staging_path, target)
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "committing a verified artifact",
            source,
        })?;
    Ok(bytes)
}

async fn write_receipt(path: &Path, bytes: &[u8]) -> Result<(), ArtifactBundleError> {
    if tokio::fs::read(path)
        .await
        .is_ok_and(|existing| existing == bytes)
    {
        return Ok(());
    }
    let (staging_path, mut staging) = create_staging_file(path).await?;
    let result = async {
        staging
            .write_all(bytes)
            .await
            .map_err(|source| ArtifactBundleError::Io {
                operation: "writing the bundle receipt",
                source,
            })?;
        sync_staging(&mut staging).await
    }
    .await;
    drop(staging);
    if let Err(error) = result {
        let _ = tokio::fs::remove_file(&staging_path).await;
        return Err(error);
    }
    match tokio::fs::remove_file(path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(source) => {
            let _ = tokio::fs::remove_file(&staging_path).await;
            return Err(ArtifactBundleError::Io {
                operation: "replacing the bundle receipt",
                source,
            });
        }
    }
    tokio::fs::rename(&staging_path, path)
        .await
        .map_err(|source| ArtifactBundleError::Io {
            operation: "committing the bundle receipt",
            source,
        })
}

fn network_error_reason(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "request timed out"
    } else if error.is_connect() {
        "connection failed"
    } else if error.is_body() {
        "response body failed"
    } else {
        "request failed"
    }
}
