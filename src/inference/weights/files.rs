use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

use super::range_io::open_cache_bypass;
use super::{WeightFileDescriptor, WeightReadStrategy};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WeightCollectionFile {
    pub(super) path: PathBuf,
    pub(super) relative_path: String,
}

/// Resolves one logical source's physical roots and rejects layouts where a
/// file could be discovered through more than one root.
pub(super) fn resolve_weight_roots(root: &Path, shard_roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::with_capacity(shard_roots.len().saturating_add(1));
    for configured in std::iter::once(root).chain(shard_roots.iter().map(PathBuf::as_path)) {
        let canonical = std::fs::canonicalize(configured).map_err(|error| {
            PowerError::InvalidFormat(format!(
                "failed to resolve model directory '{}': {error}",
                configured.display()
            ))
        })?;
        if !std::fs::metadata(&canonical)?.is_dir() {
            return Err(PowerError::InvalidFormat(format!(
                "model root '{}' is not a directory",
                canonical.display()
            )));
        }
        if roots.iter().any(|existing: &PathBuf| {
            canonical == *existing
                || canonical.starts_with(existing)
                || existing.starts_with(&canonical)
        }) {
            return Err(PowerError::Config(format!(
                "model root '{}' duplicates or overlaps another root in the same logical source",
                canonical.display()
            )));
        }
        roots.push(canonical);
    }
    Ok(roots)
}

pub(super) fn discover_safetensors(
    roots: &[PathBuf],
    max_files: usize,
) -> Result<Vec<WeightCollectionFile>> {
    let mut files = BTreeMap::<String, PathBuf>::new();
    for root in roots {
        let files_before_root = files.len();
        let mut pending = vec![root.clone()];
        while let Some(directory) = pending.pop() {
            for entry in std::fs::read_dir(&directory)? {
                let entry = entry?;
                let file_type = entry.file_type()?;
                if file_type.is_symlink() {
                    return Err(PowerError::InvalidFormat(format!(
                        "model path '{}' is a symbolic link",
                        entry.path().display()
                    )));
                }
                if file_type.is_dir() {
                    pending.push(entry.path());
                    continue;
                }
                if !file_type.is_file() {
                    continue;
                }
                if entry.path().extension().and_then(|value| value.to_str()) != Some("safetensors")
                {
                    continue;
                }
                let canonical = std::fs::canonicalize(entry.path())?;
                let relative = canonical.strip_prefix(root).map_err(|_| {
                    PowerError::InvalidFormat(format!(
                        "model file '{}' escapes its model root",
                        canonical.display()
                    ))
                })?;
                let relative = relative.to_str().ok_or_else(|| {
                    PowerError::InvalidFormat("model file names must be valid UTF-8".to_string())
                })?;
                let relative = relative.to_string();
                if files.insert(relative.clone(), canonical).is_some() {
                    return Err(PowerError::InvalidFormat(format!(
                        "logical weight collection contains duplicate file '{relative}'"
                    )));
                }
                if files.len() > max_files {
                    return Err(PowerError::InvalidFormat(format!(
                        "model contains more than {max_files} SafeTensors files"
                    )));
                }
            }
        }
        if files.len() == files_before_root {
            return Err(PowerError::InvalidFormat(format!(
                "model root '{}' contains no SafeTensors files",
                root.display()
            )));
        }
    }
    Ok(files
        .into_iter()
        .map(|(relative_path, path)| WeightCollectionFile {
            path,
            relative_path,
        })
        .collect())
}

pub(super) fn hash_files(
    files: &[WeightCollectionFile],
    max_bytes: u64,
    read_strategy: WeightReadStrategy,
) -> Result<(String, u64, Vec<WeightFileDescriptor>)> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut descriptors = Vec::with_capacity(files.len());
    let mut buffer = Zeroizing::new(vec![0_u8; HASH_BUFFER_BYTES]);
    for discovered in files {
        let path = &discovered.path;
        let relative = &discovered.relative_path;
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(PowerError::InvalidFormat(format!(
                "model file '{}' must be a non-empty regular file",
                path.display()
            )));
        }
        total = total
            .checked_add(metadata.len())
            .ok_or_else(|| PowerError::InvalidFormat("model byte length overflowed".to_string()))?;
        if total > max_bytes {
            return Err(PowerError::InvalidFormat(format!(
                "model contains {total} bytes, exceeding the {max_bytes} byte limit"
            )));
        }
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(metadata.len().to_le_bytes());
        let mut file_hasher = Sha256::new();
        let mut file = if read_strategy == WeightReadStrategy::PositionalCacheBypass {
            open_cache_bypass(path)?
        } else {
            File::open(path)?
        };
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            file_hasher.update(&buffer[..read]);
        }
        descriptors.push(WeightFileDescriptor {
            relative_path: relative.clone(),
            bytes: metadata.len(),
            sha256: format!("{:x}", file_hasher.finalize()),
        });
    }
    Ok((format!("{:x}", hasher.finalize()), total, descriptors))
}
