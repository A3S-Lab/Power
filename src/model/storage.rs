use std::collections::HashSet;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::manifest::ModelManifest;

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

/// Store a local file into the content-addressed blob store by copying it.
///
/// Returns the blob path in the store.
pub fn store_blob_from_path(source: &std::path::Path) -> Result<PathBuf> {
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

    Ok(blob_path)
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
            let size = path.metadata().map(|m| m.len()).unwrap_or(0);
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
    #[serial]
    fn test_delete_blob() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let data = b"to be deleted";
        let (path, _hash) = store_blob(data).unwrap();
        assert!(path.exists());

        let manifest = crate::model::manifest::ModelManifest {
            name: "test".to_string(),
            format: crate::model::manifest::ModelFormat::Gguf,
            size: data.len() as u64,
            sha256: "test".to_string(),
            parameters: None,
            created_at: chrono::Utc::now(),
            path: path.clone(),
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
        assert!(!path.exists());

        std::env::remove_var("A3S_POWER_HOME");
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
    fn test_store_blob_from_path() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("model.gguf");
        std::fs::write(&source_path, b"fake gguf data").unwrap();

        let blob_path = store_blob_from_path(&source_path).unwrap();
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
    fn test_store_blob_from_path_dedup() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("A3S_POWER_HOME", dir.path());

        let source_dir = tempfile::tempdir().unwrap();
        let source_path = source_dir.path().join("model.gguf");
        std::fs::write(&source_path, b"same content").unwrap();

        let path1 = store_blob_from_path(&source_path).unwrap();
        let path2 = store_blob_from_path(&source_path).unwrap();
        assert_eq!(path1, path2);

        std::env::remove_var("A3S_POWER_HOME");
    }
}
