use std::collections::HashSet;
use std::path::PathBuf;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dirs;
use crate::error::{PowerError, Result};
use crate::model::manifest::ModelManifest;
use crate::weight_collection::{
    canonicalize_collection_root, discover_safetensors, hash_safetensors,
};

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
    let actual = compute_sha256_file(path)?;
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

/// Compute the canonical digest used by the embedded SafeTensors weight store.
///
/// Unlike a generic directory digest, this identity covers only recursively
/// discovered `.safetensors` files and binds each stable relative name, length,
/// and complete byte sequence. Attestation and accelerator declarations must
/// use this same identity or a real report can never bind the executed weights.
pub fn compute_sha256_safetensors_collection(path: &std::path::Path) -> Result<String> {
    let root = canonicalize_collection_root(path)?;
    let files = discover_safetensors(&[root], usize::MAX)?;
    Ok(hash_safetensors(&files, u64::MAX, |file| {
        std::fs::File::open(file).map_err(Into::into)
    })?
    .sha256)
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

    // Collect all referenced blob paths (model, adapters, draft heads, and projectors).
    let mut referenced: HashSet<PathBuf> = HashSet::new();
    for m in manifests {
        referenced.insert(m.path.clone());
        if let Some(ref adapter) = m.adapter_path {
            referenced.insert(PathBuf::from(adapter));
        }
        if let Some(ref adapter) = m.adapter_artifact {
            referenced.insert(adapter.path.clone());
        }
        if let Some(ref draft) = m.external_draft {
            referenced.insert(draft.path.clone());
        }
        if let Some(ref projector) = m.projector_path {
            referenced.insert(PathBuf::from(projector));
        }
        if let Some(ref projector) = m.projector_artifact {
            referenced.insert(projector.path.clone());
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
mod tests;
