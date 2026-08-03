use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

use super::range_io::open_cache_bypass;
use super::{WeightFileDescriptor, WeightReadStrategy};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

pub(super) fn discover_safetensors(root: &Path, max_files: usize) -> Result<Vec<PathBuf>> {
    let mut pending = vec![root.to_path_buf()];
    let mut paths = Vec::new();
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
            if entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors") {
                let canonical = std::fs::canonicalize(entry.path())?;
                if !canonical.starts_with(root) {
                    return Err(PowerError::InvalidFormat(format!(
                        "model file '{}' escapes its model root",
                        canonical.display()
                    )));
                }
                paths.push(canonical);
                if paths.len() > max_files {
                    return Err(PowerError::InvalidFormat(format!(
                        "model contains more than {max_files} SafeTensors files"
                    )));
                }
            }
        }
    }
    Ok(paths)
}

pub(super) fn hash_files(
    root: &Path,
    paths: &[PathBuf],
    max_bytes: u64,
    read_strategy: WeightReadStrategy,
) -> Result<(String, u64, Vec<WeightFileDescriptor>)> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut descriptors = Vec::with_capacity(paths.len());
    let mut buffer = Zeroizing::new(vec![0_u8; HASH_BUFFER_BYTES]);
    for path in paths {
        let relative = path.strip_prefix(root).map_err(|_| {
            PowerError::InvalidFormat(format!(
                "model file '{}' is outside its root",
                path.display()
            ))
        })?;
        let relative = relative.to_str().ok_or_else(|| {
            PowerError::InvalidFormat("model file names must be valid UTF-8".to_string())
        })?;
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
            relative_path: relative.to_string(),
            bytes: metadata.len(),
            sha256: format!("{:x}", file_hasher.finalize()),
        });
    }
    Ok((format!("{:x}", hasher.finalize()), total, descriptors))
}
