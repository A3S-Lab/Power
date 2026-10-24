//! Format-neutral, hash-bound raw weights. Model owners interpret the ranges.

use std::fs::File;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::weight_collection::{canonicalize_collection_root, portable_relative_path};

use super::range_io::{read_buffered_into, RANGE_READ_CHUNK_BYTES};
use super::{InferenceLimits, WeightFileDescriptor};

/// A verified regular file under an explicitly supplied trusted model root.
///
/// Opening streams the complete file through SHA256 using bounded scratch,
/// retaining a digest for each 1 MiB chunk. Later positional reads authenticate
/// every touched chunk against that table on the same held file handle. Path
/// replacement cannot redirect a read, and in-place payload changes cannot be
/// returned under the original identity. No model format or I/O fallback is
/// inferred. This source uses buffered positional reads only.
///
/// The host owns the trusted root and its directory entries during opening,
/// as for WeightStore. This is not a sandbox for an attacker-controlled root.
/// The index, scratch and maximum output are bounded together by
/// `InferenceLimits::max_state_bytes`; file size uses `max_model_bytes`.
pub struct VerifiedRawWeightFile {
    file: File,
    descriptor: WeightFileDescriptor,
    chunk_hashes: Vec<[u8; 32]>,
    max_read_bytes: u64,
}

/// Zeroizing, authenticated bytes with their explicit source and range.
pub struct RawWeightRangeRead {
    bytes: Zeroizing<Vec<u8>>,
    descriptor: WeightFileDescriptor,
    offset: u64,
}

impl VerifiedRawWeightFile {
    pub fn open(
        trusted_root: impl AsRef<Path>,
        descriptor: WeightFileDescriptor,
        limits: &InferenceLimits,
        max_read_bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        check_cancelled(cancellation)?;
        limits.validate()?;
        validate_descriptor(&descriptor)?;
        if descriptor.bytes > limits.max_model_bytes {
            return Err(invalid("raw weight file exceeds the model byte limit"));
        }
        if max_read_bytes == 0 {
            return Err(invalid("raw weight read budget must be positive"));
        }
        let chunks = descriptor.bytes.div_ceil(RANGE_READ_CHUNK_BYTES as u64);
        let scratch_bytes = descriptor.bytes.min(RANGE_READ_CHUNK_BYTES as u64);
        let memory = chunks
            .checked_mul(32)
            .and_then(|index| index.checked_add(scratch_bytes))
            .and_then(|value| value.checked_add(max_read_bytes))
            .ok_or_else(|| invalid("raw weight reader memory budget overflowed"))?;
        limits.checked_state_bytes(memory, "raw weight reader index, scratch and output")?;
        usize::try_from(max_read_bytes)
            .map_err(|_| invalid("raw weight read budget exceeds the host address range"))?;
        let path = resolve_file(trusted_root.as_ref(), &descriptor.relative_path)?;
        let file = File::open(path)?;
        check_file(&file, descriptor.bytes)?;
        let mut scratch = zeroed(scratch_bytes)?;
        let mut chunk_hashes = Vec::new();
        chunk_hashes
            .try_reserve_exact(
                usize::try_from(chunks).map_err(|_| {
                    invalid("raw weight chunk count exceeds the host address range")
                })?,
            )
            .map_err(|_| invalid("raw weight digest index allocation failed"))?;
        let mut full_hash = Sha256::new();
        let mut offset = 0;
        while offset < descriptor.bytes {
            let bytes = (descriptor.bytes - offset).min(scratch_bytes) as usize;
            let chunk = &mut scratch[..bytes];
            read_buffered_into(&file, chunk, offset, cancellation)?;
            full_hash.update(&*chunk);
            chunk_hashes.push(Sha256::digest(&*chunk).into());
            offset += bytes as u64;
        }
        check_cancelled(cancellation)?;
        check_file(&file, descriptor.bytes)?;
        let actual = format!("{:x}", full_hash.finalize());
        if actual != descriptor.sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: "raw-weight-file".into(),
                expected: descriptor.sha256,
                actual,
            });
        }
        Ok(Self {
            file,
            descriptor,
            chunk_hashes,
            max_read_bytes,
        })
    }

    pub fn descriptor(&self) -> &WeightFileDescriptor {
        &self.descriptor
    }

    pub fn max_read_bytes(&self) -> u64 {
        self.max_read_bytes
    }

    /// Reads a non-empty range, authenticating whole chunks before exposing
    /// any bytes. Scratch is at most 1 MiB, including for unaligned ranges.
    /// Concurrent reads need separate output/scratch budgets from their owner.
    pub fn read_range(
        &self,
        offset: u64,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<RawWeightRangeRead> {
        check_cancelled(cancellation)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("raw weight range overflowed"))?;
        if bytes == 0 || bytes > self.max_read_bytes || end > self.descriptor.bytes {
            return Err(invalid(
                "raw weight range is empty or exceeds its file/read budget",
            ));
        }
        check_file(&self.file, self.descriptor.bytes)?;
        let mut output = zeroed(bytes)?;
        let mut scratch = zeroed(self.descriptor.bytes.min(RANGE_READ_CHUNK_BYTES as u64))?;
        let first = offset / RANGE_READ_CHUNK_BYTES as u64;
        let last = (end - 1) / RANGE_READ_CHUNK_BYTES as u64;
        let mut copied = 0;
        for index in first..=last {
            let chunk_offset = index * RANGE_READ_CHUNK_BYTES as u64;
            let chunk_bytes =
                (self.descriptor.bytes - chunk_offset).min(RANGE_READ_CHUNK_BYTES as u64) as usize;
            let chunk = &mut scratch[..chunk_bytes];
            read_buffered_into(&self.file, chunk, chunk_offset, cancellation)?;
            let digest: [u8; 32] = Sha256::digest(&*chunk).into();
            let expected = self
                .chunk_hashes
                .get(index as usize)
                .ok_or_else(|| invalid("raw weight range references an unknown chunk"))?;
            if &digest != expected {
                return Err(invalid(
                    "raw weight chunk changed after full SHA256 verification",
                ));
            }
            let start = offset.saturating_sub(chunk_offset) as usize;
            let stop = (end - chunk_offset).min(chunk_bytes as u64) as usize;
            let count = stop - start;
            output[copied..copied + count].copy_from_slice(&chunk[start..stop]);
            copied += count;
        }
        check_file(&self.file, self.descriptor.bytes)?;
        check_cancelled(cancellation)?;
        Ok(RawWeightRangeRead {
            bytes: output,
            descriptor: self.descriptor.clone(),
            offset,
        })
    }
}

impl RawWeightRangeRead {
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub fn descriptor(&self) -> &WeightFileDescriptor {
        &self.descriptor
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }
}

impl std::fmt::Debug for VerifiedRawWeightFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedRawWeightFile")
            .field("bytes", &self.descriptor.bytes)
            .field("chunks", &self.chunk_hashes.len())
            .field("max_read_bytes", &self.max_read_bytes)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for RawWeightRangeRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawWeightRangeRead")
            .field("bytes", &self.bytes.len())
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

fn validate_descriptor(descriptor: &WeightFileDescriptor) -> Result<()> {
    let path = &descriptor.relative_path;
    if path.is_empty()
        || path.len() > 4096
        || path
            .chars()
            .any(|c| c.is_control() || matches!(c, '\\' | ':'))
        || path.split('/').any(|part| matches!(part, "" | "." | ".."))
    {
        return Err(invalid(
            "raw weight path must be a canonical portable relative path",
        ));
    }
    if descriptor.bytes == 0
        || descriptor.sha256.len() != 64
        || !descriptor
            .sha256
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    {
        return Err(invalid(
            "raw weight descriptor requires positive bytes and lowercase SHA256",
        ));
    }
    Ok(())
}

fn resolve_file(trusted_root: &Path, relative_path: &str) -> Result<PathBuf> {
    let root = canonicalize_collection_root(trusted_root)?;
    let mut path = root.clone();
    let parts: Vec<_> = relative_path.split('/').collect();
    for (index, part) in parts.iter().enumerate() {
        path.push(part);
        let metadata = std::fs::symlink_metadata(&path)?;
        if is_link(&metadata)
            || (index + 1 < parts.len() && !metadata.is_dir())
            || (index + 1 == parts.len() && !metadata.is_file())
        {
            return Err(invalid(
                "raw weight path contains a link or non-regular component",
            ));
        }
    }
    let resolved = std::fs::canonicalize(&path)?;
    let relative = resolved
        .strip_prefix(&root)
        .map_err(|_| invalid("raw weight path escapes its trusted root"))?;
    if portable_relative_path(relative)? != relative_path {
        return Err(invalid(
            "raw weight path resolves through a non-canonical alias",
        ));
    }
    Ok(resolved)
}

fn is_link(metadata: &std::fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // FILE_ATTRIBUTE_REPARSE_POINT also covers directory junctions.
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn check_file(file: &File, bytes: u64) -> Result<()> {
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != bytes {
        return Err(invalid(
            "raw weight file size or kind differs from its verified descriptor",
        ));
    }
    Ok(())
}

fn zeroed(bytes: u64) -> Result<Zeroizing<Vec<u8>>> {
    let length = usize::try_from(bytes)
        .map_err(|_| invalid("raw weight buffer exceeds the host address range"))?;
    let mut output = Zeroizing::new(Vec::new());
    output
        .try_reserve_exact(length)
        .map_err(|_| invalid("raw weight buffer allocation failed"))?;
    output.resize(length, 0);
    Ok(output)
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceCancelled);
    }
    Ok(())
}

fn invalid(message: &str) -> PowerError {
    PowerError::InvalidFormat(message.into())
}

#[cfg(test)]
mod tests;
