use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalWeightFile {
    pub(crate) path: PathBuf,
    pub(crate) relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalWeightFileDigest {
    pub(crate) relative_path: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalWeightCollectionDigest {
    pub(crate) sha256: String,
    pub(crate) bytes: u64,
    pub(crate) files: Vec<CanonicalWeightFileDigest>,
}

/// Discovers one canonical SafeTensors collection beneath already resolved roots.
///
/// Relative names are the collection identity. Duplicate names across shard
/// roots, symlinks, non-UTF-8 names, and empty roots fail before any bytes are
/// trusted.
pub(crate) fn discover_safetensors(
    roots: &[PathBuf],
    max_files: usize,
) -> Result<Vec<CanonicalWeightFile>> {
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
                if !file_type.is_file()
                    || entry.path().extension().and_then(|value| value.to_str())
                        != Some("safetensors")
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
                let relative = portable_relative_path(relative)?;
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
        .map(|(relative_path, path)| CanonicalWeightFile {
            path,
            relative_path,
        })
        .collect())
}

/// Hashes stable relative names, lengths, and complete bytes in lexical order.
///
/// The opener is supplied by the caller so the embedded runtime can retain its
/// cache-bypass policy while server-side attestation uses ordinary read-only
/// files. Both paths consume this single canonical digest implementation.
pub(crate) fn hash_safetensors<F>(
    files: &[CanonicalWeightFile],
    max_bytes: u64,
    mut open: F,
) -> Result<CanonicalWeightCollectionDigest>
where
    F: FnMut(&Path) -> Result<File>,
{
    if files.is_empty() {
        return Err(PowerError::InvalidFormat(
            "weight collection contains no SafeTensors files".to_string(),
        ));
    }
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
        let relative_bytes = u64::try_from(relative.len()).map_err(|_| {
            PowerError::InvalidFormat("model file name length cannot be represented".to_string())
        })?;
        hasher.update(relative_bytes.to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(metadata.len().to_le_bytes());
        let mut file = open(path)?;
        let opened_metadata = file.metadata()?;
        if !opened_metadata.is_file() || opened_metadata.len() != metadata.len() {
            return Err(PowerError::InvalidFormat(format!(
                "model file '{}' changed while its weight digest was being prepared",
                path.display()
            )));
        }
        let mut file_hasher = Sha256::new();
        let mut read_bytes = 0_u64;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let read_bytes_this_round = u64::try_from(read).map_err(|_| {
                PowerError::InvalidFormat("model read length cannot be represented".to_string())
            })?;
            read_bytes = read_bytes
                .checked_add(read_bytes_this_round)
                .ok_or_else(|| PowerError::InvalidFormat("model byte length overflowed".into()))?;
            hasher.update(&buffer[..read]);
            file_hasher.update(&buffer[..read]);
        }
        if read_bytes != metadata.len() {
            return Err(PowerError::InvalidFormat(format!(
                "model file '{}' changed while its weight digest was being computed",
                path.display()
            )));
        }
        descriptors.push(CanonicalWeightFileDigest {
            relative_path: relative.clone(),
            bytes: metadata.len(),
            sha256: format!("{:x}", file_hasher.finalize()),
        });
    }
    Ok(CanonicalWeightCollectionDigest {
        sha256: format!("{:x}", hasher.finalize()),
        bytes: total,
        files: descriptors,
    })
}

pub(crate) fn portable_relative_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(part) = component else {
            return Err(PowerError::InvalidFormat(
                "model file paths must be canonical relative paths".to_string(),
            ));
        };
        parts.push(
            part.to_str()
                .ok_or_else(|| {
                    PowerError::InvalidFormat("model file names must be valid UTF-8".to_string())
                })?
                .to_string(),
        );
    }
    if parts.is_empty() {
        return Err(PowerError::InvalidFormat(
            "model file path cannot be empty".to_string(),
        ));
    }
    Ok(parts.join("/"))
}

pub(crate) fn canonicalize_collection_root(root: &Path) -> Result<PathBuf> {
    let canonical = std::fs::canonicalize(root).map_err(|error| {
        PowerError::InvalidFormat(format!(
            "failed to resolve model directory '{}': {error}",
            root.display()
        ))
    })?;
    if !std::fs::metadata(&canonical)?.is_dir() {
        return Err(PowerError::InvalidFormat(format!(
            "model root '{}' is not a directory",
            canonical.display()
        )));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_uses_portable_relative_names() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        std::fs::write(nested.join("model.safetensors"), b"fixture").unwrap();
        let root = canonicalize_collection_root(directory.path()).unwrap();
        let files = discover_safetensors(&[root], 1).unwrap();
        assert_eq!(files[0].relative_path, "nested/model.safetensors");
    }

    #[test]
    fn hashing_rejects_a_file_changed_between_metadata_and_open() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.safetensors");
        std::fs::write(&path, b"fixture").unwrap();
        let root = canonicalize_collection_root(directory.path()).unwrap();
        let files = discover_safetensors(&[root], 1).unwrap();
        let error = hash_safetensors(&files, 1024, |file| {
            std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(file)?;
            Ok(File::open(file)?)
        })
        .unwrap_err();
        assert!(error.to_string().contains("changed"));
    }

    #[test]
    fn hashing_rejects_an_empty_collection() {
        let error = hash_safetensors(&[], 1024, |file| Ok(File::open(file)?)).unwrap_err();
        assert!(error.to_string().contains("no SafeTensors files"));
    }
}
