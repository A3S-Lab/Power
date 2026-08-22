use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{PowerError, Result};
use crate::weight_collection::{
    canonicalize_collection_root, discover_safetensors as discover_canonical_safetensors,
    hash_safetensors,
};

use super::range_io::open_cache_bypass;
use super::{WeightFileDescriptor, WeightReadStrategy};

pub(super) use crate::weight_collection::CanonicalWeightFile as WeightCollectionFile;

/// Resolves one logical source's physical roots and rejects layouts where a
/// file could be discovered through more than one root.
pub(super) fn resolve_weight_roots(root: &Path, shard_roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut roots = Vec::with_capacity(shard_roots.len().saturating_add(1));
    for configured in std::iter::once(root).chain(shard_roots.iter().map(PathBuf::as_path)) {
        let canonical = canonicalize_collection_root(configured)?;
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
    discover_canonical_safetensors(roots, max_files)
}

pub(super) fn hash_files(
    files: &[WeightCollectionFile],
    max_bytes: u64,
    read_strategy: WeightReadStrategy,
) -> Result<(String, u64, Vec<WeightFileDescriptor>)> {
    let digest = hash_safetensors(files, max_bytes, |path| {
        if read_strategy == WeightReadStrategy::PositionalCacheBypass {
            open_cache_bypass(path)
        } else {
            File::open(path).map_err(Into::into)
        }
    })?;
    let files = digest
        .files
        .into_iter()
        .map(|file| WeightFileDescriptor {
            relative_path: file.relative_path,
            bytes: file.bytes,
            sha256: file.sha256,
        })
        .collect();
    Ok((digest.sha256, digest.bytes, files))
}
