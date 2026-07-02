use std::collections::HashSet;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::manifest::ModelManifest;

#[derive(Serialize)]
struct DirectoryManifestDigest<'a> {
    schema: &'static str,
    entries: &'a [DirectoryDigestEntry],
}

#[derive(Serialize)]
struct DirectoryDigestEntry {
    path: String,
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

/// Store a model file in the content-addressed blob store.
///
/// Returns the blob path and SHA-256 hash of the stored file.
pub fn store_blob(data: &[u8]) -> Result<(PathBuf, String)> {
    let blob_dir = dirs::blobs_dir();
    std::fs::create_dir_all(&blob_dir)?;

    let hash = compute_sha256(data);
    let blob_name = format!("sha256-{hash}");
    let blob_path = blob_dir.join(&blob_name);

    if !blob_path.exists() {
        std::fs::write(&blob_path, data).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to write blob {}: {e}",
                blob_path.display()
            )))
        })?;
    }

    Ok((blob_path, hash))
}

/// Delete the blob file associated with a model manifest.
pub fn delete_blob(manifest: &ModelManifest) -> Result<()> {
    if manifest.path.exists() {
        std::fs::remove_file(&manifest.path).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to delete blob {}: {e}",
                manifest.path.display()
            )))
        })?;
    }
    Ok(())
}

/// Verify the integrity of a blob file against its expected SHA-256 hash.
pub fn verify_blob(path: &std::path::Path, expected_sha256: &str) -> Result<bool> {
    let data = std::fs::read(path).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to read blob for verification {}: {e}",
            path.display()
        )))
    })?;
    let actual = compute_sha256(&data);
    Ok(actual == expected_sha256)
}

/// Compute SHA-256 hash of the given data, returned as a hex string.
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{result:x}")
}

/// Compute SHA-256 hash of a file on disk (streaming, memory-efficient).
pub fn compute_sha256_file(path: &std::path::Path) -> Result<String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to open file for hashing {}: {e}",
            path.display()
        )))
    })?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = file.read(&mut buf).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to read file for hashing {}: {e}",
                path.display()
            )))
        })?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Compute a SHA-256 digest for either a file or a deterministic directory manifest.
pub fn compute_sha256_path(path: &std::path::Path) -> Result<String> {
    if path.is_file() {
        return compute_sha256_file(path);
    }
    if path.is_dir() {
        return compute_sha256_directory(path);
    }
    Err(PowerError::Io(std::io::Error::other(format!(
        "Path is neither a regular file nor a directory: {}",
        path.display()
    ))))
}

/// Compute SHA-256 over a canonical manifest of all files in a directory.
pub fn compute_sha256_directory(path: &std::path::Path) -> Result<String> {
    let mut entries = Vec::new();
    let metadata = std::fs::symlink_metadata(path).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to inspect directory {}: {e}",
            path.display()
        )))
    })?;
    if !metadata.is_dir() {
        return Err(PowerError::Io(std::io::Error::other(format!(
            "Path is not a directory: {}",
            path.display()
        ))));
    }

    entries.push(DirectoryDigestEntry {
        path: ".".to_string(),
        kind: "directory",
        mode: permission_mode(&metadata),
        size: None,
        sha256: None,
    });
    collect_directory_digest_entries(path, path, &mut entries)?;

    let manifest = DirectoryManifestDigest {
        schema: "a3s.power.directory-manifest.v1",
        entries: &entries,
    };
    let bytes = serde_json::to_vec(&manifest).map_err(|e| {
        PowerError::Config(format!(
            "Failed to serialize directory digest manifest: {e}"
        ))
    })?;
    Ok(compute_sha256(&bytes))
}

fn collect_directory_digest_entries(
    root: &std::path::Path,
    dir: &std::path::Path,
    entries: &mut Vec<DirectoryDigestEntry>,
) -> Result<()> {
    let mut children = std::fs::read_dir(dir)
        .map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to read directory {}: {e}",
                dir.display()
            )))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to read directory entry in {}: {e}",
                dir.display()
            )))
        })?;
    children.sort_by_key(|entry| entry.file_name());

    for child in children {
        let path = child.path();
        let metadata = std::fs::symlink_metadata(&path).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to inspect directory entry {}: {e}",
                path.display()
            )))
        })?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(PowerError::Config(format!(
                "Directory model digest does not support symlinks: {}",
                path.display()
            )));
        }

        let relative_path = canonical_relative_path(root, &path)?;
        if file_type.is_dir() {
            entries.push(DirectoryDigestEntry {
                path: relative_path,
                kind: "directory",
                mode: permission_mode(&metadata),
                size: None,
                sha256: None,
            });
            collect_directory_digest_entries(root, &path, entries)?;
        } else if file_type.is_file() {
            entries.push(DirectoryDigestEntry {
                path: relative_path,
                kind: "file",
                mode: permission_mode(&metadata),
                size: Some(metadata.len()),
                sha256: Some(compute_sha256_file(&path)?),
            });
        } else {
            return Err(PowerError::Config(format!(
                "Directory model digest does not support special files: {}",
                path.display()
            )));
        }
    }

    Ok(())
}

fn canonical_relative_path(root: &std::path::Path, path: &std::path::Path) -> Result<String> {
    let relative = path.strip_prefix(root).map_err(|e| {
        PowerError::Config(format!(
            "Failed to build relative path for {} under {}: {e}",
            path.display(),
            root.display()
        ))
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        match component {
            std::path::Component::Normal(part) => {
                let Some(part) = part.to_str() else {
                    return Err(PowerError::Config(format!(
                        "Directory model digest requires UTF-8 paths: {}",
                        path.display()
                    )));
                };
                parts.push(part.to_string());
            }
            _ => {
                return Err(PowerError::Config(format!(
                    "Directory model digest encountered unsupported path component: {}",
                    path.display()
                )));
            }
        }
    }
    Ok(parts.join("/"))
}

#[cfg(unix)]
fn permission_mode(metadata: &std::fs::Metadata) -> Option<u32> {
    use std::os::unix::fs::PermissionsExt;
    Some(metadata.permissions().mode() & 0o7777)
}

#[cfg(not(unix))]
fn permission_mode(_metadata: &std::fs::Metadata) -> Option<u32> {
    None
}

/// Store a local file into the content-addressed blob store by copying it.
///
/// Returns the blob path and SHA-256 hash. Uses streaming hash computation
/// so it works with arbitrarily large files without loading them into memory.
/// The source file is NOT modified or deleted.
pub fn store_blob_from_path(source: &std::path::Path) -> Result<(PathBuf, String)> {
    let blob_dir = dirs::blobs_dir();
    std::fs::create_dir_all(&blob_dir)?;

    let hash = compute_sha256_file(source)?;
    let blob_name = format!("sha256-{hash}");
    let blob_path = blob_dir.join(&blob_name);

    if !blob_path.exists() {
        std::fs::copy(source, &blob_path).map_err(|e| {
            PowerError::Io(std::io::Error::other(format!(
                "Failed to copy '{}' to blob store: {e}",
                source.display()
            )))
        })?;
    }

    Ok((blob_path, hash))
}

fn cleanup_temp_source(source: &std::path::Path, reason: &str) {
    match std::fs::remove_file(source) {
        Ok(()) => tracing::debug!(
            path = %source.display(),
            reason,
            "Removed temporary blob source"
        ),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => tracing::debug!(
            path = %source.display(),
            reason,
            "Temporary blob source was already removed"
        ),
        Err(e) => tracing::warn!(
            path = %source.display(),
            reason,
            error = %e,
            "Failed to remove temporary blob source"
        ),
    }
}

/// Move a temporary file into the content-addressed blob store.
///
/// Like `store_blob_from_path`, but tries to rename (move) the source file
/// instead of copying, which is much faster for large files on the same
/// filesystem. The source file is removed after a successful store.
pub fn store_blob_from_temp(source: &std::path::Path) -> Result<(PathBuf, String)> {
    let blob_dir = dirs::blobs_dir();
    std::fs::create_dir_all(&blob_dir)?;

    let hash = compute_sha256_file(source)?;
    let blob_name = format!("sha256-{hash}");
    let blob_path = blob_dir.join(&blob_name);

    if !blob_path.exists() {
        // Try rename first (fast, same filesystem), fall back to copy
        if let Err(rename_err) = std::fs::rename(source, &blob_path) {
            tracing::debug!(
                source = %source.display(),
                destination = %blob_path.display(),
                error = %rename_err,
                "Blob rename failed, falling back to copy"
            );
            std::fs::copy(source, &blob_path).map_err(|e| {
                PowerError::Io(std::io::Error::other(format!(
                    "Failed to copy '{}' to blob store: {e}",
                    source.display()
                )))
            })?;
            cleanup_temp_source(source, "copied temp file into blob store");
        }
    } else {
        // Blob already exists, just clean up the temp source
        cleanup_temp_source(source, "blob already existed");
    }

    Ok((blob_path, hash))
}

/// Remove blob files that are not referenced by any model manifest.
///
/// Scans the blobs directory and compares against the set of blob paths
/// referenced by registered manifests. Any blob file not referenced is deleted.
///
/// Returns the number of blobs removed and total bytes freed.
pub fn prune_unused_blobs(manifests: &[ModelManifest]) -> Result<(usize, u64)> {
    let blob_dir = dirs::blobs_dir();
    if !blob_dir.exists() {
        return Ok((0, 0));
    }

    // Collect all referenced blob paths (model file + adapter + projector)
    let mut referenced: HashSet<PathBuf> = HashSet::new();
    for m in manifests {
        referenced.insert(m.path.clone());
        if let Some(ref adapter) = m.adapter_path {
            referenced.insert(PathBuf::from(adapter));
        }
        if let Some(ref projector) = m.projector_path {
            referenced.insert(PathBuf::from(projector));
        }
    }

    let mut removed = 0usize;
    let mut freed = 0u64;

    let entries = std::fs::read_dir(&blob_dir).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to read blobs directory {}: {e}",
            blob_dir.display()
        )))
    })?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if !referenced.contains(&path) {
            let size = blob_file_size(&path)?;
            match std::fs::remove_file(&path) {
                Ok(()) => {
                    tracing::info!(
                        path = %path.display(),
                        size,
                        "Pruned unused blob"
                    );
                    removed += 1;
                    freed += size;
                }
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Failed to prune blob"
                    );
                }
            }
        }
    }

    Ok((removed, freed))
}

fn blob_file_size(path: &std::path::Path) -> Result<u64> {
    path.metadata().map(|metadata| metadata.len()).map_err(|e| {
        PowerError::Io(std::io::Error::other(format!(
            "Failed to inspect blob {} before pruning: {e}",
            path.display()
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_compute_sha256() {
        let hash = compute_sha256(b"hello world");
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    #[serial]
    fn test_store_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"test model data";
        let (path, hash) = store_blob(data).unwrap();

        assert!(path.exists());
        assert!(path.to_string_lossy().contains(&format!("sha256-{hash}")));

        let stored = std::fs::read(&path).unwrap();
        assert_eq!(stored, data);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_deduplication() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"identical data";
        let (path1, hash1) = store_blob(data).unwrap();
        let (path2, hash2) = store_blob(data).unwrap();

        assert_eq!(path1, path2);
        assert_eq!(hash1, hash2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_verify_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"verify me";
        let (path, hash) = store_blob(data).unwrap();

        assert!(verify_blob(&path, &hash).unwrap());
        assert!(!verify_blob(&path, "wrong-hash").unwrap());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_delete_blob() {
        let dir = tempfile::tempdir().unwrap();

        // Write a blob file directly to avoid env var races through store_blob
        let blob_path = dir.path().join("blob-to-delete");
        std::fs::write(&blob_path, b"to be deleted").unwrap();
        assert!(blob_path.exists());

        let manifest = crate::model::manifest::ModelManifest {
            name: "test".to_string(),
            format: crate::model::manifest::ModelFormat::Gguf,
            size: 13,
            sha256: "test".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: blob_path.clone(),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        delete_blob(&manifest).unwrap();
        assert!(!blob_path.exists());
    }

    #[test]
    #[serial]
    fn test_delete_blob_nonexistent_path() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Manifest pointing to a nonexistent file — should succeed (no-op)
        let manifest = crate::model::manifest::ModelManifest {
            name: "ghost".to_string(),
            format: crate::model::manifest::ModelFormat::Gguf,
            size: 0,
            sha256: "none".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: std::path::PathBuf::from("/tmp/nonexistent-blob-file"),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        // Should not error — file doesn't exist, so nothing to delete
        delete_blob(&manifest).unwrap();

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_compute_sha256_empty() {
        let hash = compute_sha256(b"");
        assert_eq!(
            hash,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_compute_sha256_directory_is_order_stable() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();

        std::fs::create_dir(first.path().join("nested")).unwrap();
        std::fs::write(first.path().join("config.json"), br#"{"model":"a"}"#).unwrap();
        std::fs::write(
            first.path().join("nested").join("weights.safetensors"),
            b"weights",
        )
        .unwrap();

        std::fs::create_dir(second.path().join("nested")).unwrap();
        std::fs::write(
            second.path().join("nested").join("weights.safetensors"),
            b"weights",
        )
        .unwrap();
        std::fs::write(second.path().join("config.json"), br#"{"model":"a"}"#).unwrap();

        assert_eq!(
            compute_sha256_directory(first.path()).unwrap(),
            compute_sha256_directory(second.path()).unwrap()
        );
    }

    #[test]
    fn test_compute_sha256_directory_changes_when_file_changes() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.safetensors");
        std::fs::write(&model_path, b"weights-v1").unwrap();
        let first = compute_sha256_directory(dir.path()).unwrap();

        std::fs::write(&model_path, b"weights-v2").unwrap();
        let second = compute_sha256_directory(dir.path()).unwrap();

        assert_ne!(first, second);
    }

    #[test]
    #[serial]
    fn test_verify_blob_nonexistent_file() {
        let result = verify_blob(std::path::Path::new("/tmp/nonexistent-verify-test"), "abc");
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_prune_unused_blobs_removes_orphans() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Store two blobs
        let (path_a, _) = store_blob(b"model-a-data").unwrap();
        let (_, _) = store_blob(b"orphan-data").unwrap();

        // Only reference path_a in manifests
        let manifest = crate::model::manifest::ModelManifest {
            name: "model-a".to_string(),
            format: crate::model::manifest::ModelFormat::Gguf,
            size: 12,
            sha256: "test".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: path_a.clone(),
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        let (removed, freed) = prune_unused_blobs(&[manifest]).unwrap();
        assert_eq!(removed, 1);
        assert!(freed > 0);
        // Referenced blob should still exist
        assert!(path_a.exists());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_prune_unused_blobs_no_orphans() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let (path_a, _) = store_blob(b"data-a").unwrap();

        let manifest = crate::model::manifest::ModelManifest {
            name: "a".to_string(),
            format: crate::model::manifest::ModelFormat::Gguf,
            size: 6,
            sha256: "test".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: path_a,
            system_prompt: None,
            template_override: None,
            default_parameters: None,
            modelfile_content: None,
            license: None,
            adapter_path: None,
            projector_path: None,
            messages: vec![],
            family: None,
            families: None,
        };

        let (removed, freed) = prune_unused_blobs(&[manifest]).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(freed, 0);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_prune_unused_blobs_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Create blobs dir but leave it empty
        std::fs::create_dir_all(dirs::blobs_dir()).unwrap();

        let (removed, freed) = prune_unused_blobs(&[]).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(freed, 0);

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    fn test_prune_unused_blobs_nonexistent_dir() {
        // When blobs dir doesn't exist, should return (0, 0)
        let _dir = tempfile::tempdir().unwrap();
        // prune_unused_blobs checks dirs::blobs_dir() which may or may not exist;
        // the function handles missing dirs gracefully by returning (0, 0).
        let result = prune_unused_blobs(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_blob_file_size_reports_metadata_errors() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing-blob");

        let err = blob_file_size(&missing).unwrap_err();

        assert!(
            err.to_string().contains("Failed to inspect blob"),
            "error: {err}"
        );
        assert!(err.to_string().contains("missing-blob"), "error: {err}");
    }

    #[test]
    fn test_compute_sha256_file() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").unwrap();

        let hash = compute_sha256_file(&file_path).unwrap();
        let expected = compute_sha256(b"hello world");
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_compute_sha256_file_nonexistent() {
        let result = compute_sha256_file(std::path::Path::new("/nonexistent/file.bin"));
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_store_blob_from_path() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("model.gguf");
        std::fs::write(&source_path, b"fake gguf data").unwrap();

        let (blob_path, _hash) = store_blob_from_path(&source_path).unwrap();
        assert!(blob_path.exists());

        // Verify content matches
        let stored = std::fs::read(&blob_path).unwrap();
        assert_eq!(stored, b"fake gguf data");

        // Verify blob name contains sha256
        let filename = blob_path.file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("sha256-"));

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_path_dedup() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("model.gguf");
        std::fs::write(&source_path, b"same content").unwrap();

        let (path1, _) = store_blob_from_path(&source_path).unwrap();
        let (path2, _) = store_blob_from_path(&source_path).unwrap();
        assert_eq!(path1, path2);

        std::env::remove_var("A3S_POWER_HOME");
    }

    // ========================================================================
    // store_blob_from_temp integration tests
    // ========================================================================

    #[test]
    #[serial]
    fn test_store_blob_from_temp_moves_file() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("partial-abc123");
        std::fs::write(&source_path, b"large model data").unwrap();
        assert!(source_path.exists());

        let (blob_path, hash) = store_blob_from_temp(&source_path).unwrap();

        // Blob should exist with correct content
        assert!(blob_path.exists());
        let stored = std::fs::read(&blob_path).unwrap();
        assert_eq!(stored, b"large model data");

        // Hash should be valid hex
        assert!(!hash.is_empty());
        let filename = blob_path.file_name().unwrap().to_str().unwrap();
        assert_eq!(filename, format!("sha256-{hash}"));

        // Source temp file should be gone (renamed or deleted)
        assert!(!source_path.exists());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_temp_dedup_cleans_source() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // First: store via normal path to create the blob
        let source_dir = tempfile::tempdir().unwrap();
        let source1 = source_dir.path().join("original.bin");
        std::fs::write(&source1, b"dedup content").unwrap();
        let (blob_path, _) = store_blob_from_path(&source1).unwrap();
        assert!(blob_path.exists());

        // Second: store_blob_from_temp with same content — blob already exists
        let source2 = source_dir.path().join("partial-duplicate");
        std::fs::write(&source2, b"dedup content").unwrap();
        let (blob_path2, hash2) = store_blob_from_temp(&source2).unwrap();

        // Should return same blob path
        assert_eq!(blob_path, blob_path2);
        assert!(!hash2.is_empty());

        // Temp source should be cleaned up even though blob already existed
        assert!(!source2.exists());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_temp_same_dir_uses_rename() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        // Create blobs dir and put the temp file directly in it (same filesystem)
        let blobs_dir = dirs::blobs_dir();
        std::fs::create_dir_all(&blobs_dir).unwrap();
        let source_path = blobs_dir.join("partial-inplace");
        std::fs::write(&source_path, b"rename me").unwrap();

        let (blob_path, hash) = store_blob_from_temp(&source_path).unwrap();

        assert!(blob_path.exists());
        assert!(!source_path.exists()); // renamed away
        let stored = std::fs::read(&blob_path).unwrap();
        assert_eq!(stored, b"rename me");
        assert!(blob_path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("sha256-"));
        assert!(!hash.is_empty());

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_path_preserves_source() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("user-model.gguf");
        std::fs::write(&source_path, b"user data").unwrap();

        let (blob_path, hash) = store_blob_from_path(&source_path).unwrap();

        // Blob should exist
        assert!(blob_path.exists());
        assert!(!hash.is_empty());

        // Source file should still exist (not moved/deleted)
        assert!(source_path.exists());
        let original = std::fs::read(&source_path).unwrap();
        assert_eq!(original, b"user data");

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_path_returns_correct_hash() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("hashtest.bin");
        std::fs::write(&source_path, b"hash me").unwrap();

        let (blob_path, hash) = store_blob_from_path(&source_path).unwrap();

        // Hash from store_blob_from_path should match compute_sha256
        let expected_hash = compute_sha256(b"hash me");
        assert_eq!(hash, expected_hash);
        assert_eq!(
            blob_path.file_name().unwrap().to_str().unwrap(),
            format!("sha256-{expected_hash}")
        );

        std::env::remove_var("A3S_POWER_HOME");
    }

    #[test]
    #[serial]
    fn test_store_blob_from_temp_nonexistent_source_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let result = store_blob_from_temp(std::path::Path::new("/nonexistent/partial-xyz"));
        assert!(result.is_err());

        std::env::remove_var("A3S_POWER_HOME");
    }
}
