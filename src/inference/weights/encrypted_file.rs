use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

use super::encrypted_format::{
    chunk_aad, encode_header, parse_header, plaintext_chunk_bytes, validate_chunk_bytes,
    SeekableEncryptedFileDescriptor, HEADER_BYTES, NONCE_BYTES, TAG_BYTES,
};
use super::range_io::read_exact_loop;

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

/// A zeroizing AES-256 key for seekable encrypted weight sources.
#[derive(Clone)]
pub struct SeekableWeightKey(Zeroizing<[u8; 32]>);

impl SeekableWeightKey {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// Parses a 32-byte key without retaining the decoded temporary buffer.
    pub fn from_hex(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(PowerError::Config(
                "seekable weight key must contain exactly 64 hexadecimal characters".to_string(),
            ));
        }
        let mut bytes = Zeroizing::new([0_u8; 32]);
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let pair = std::str::from_utf8(pair).map_err(|_| {
                PowerError::Config("seekable weight key contains invalid hexadecimal".to_string())
            })?;
            bytes[index] = u8::from_str_radix(pair, 16).map_err(|_| {
                PowerError::Config("seekable weight key contains invalid hexadecimal".to_string())
            })?;
        }
        Ok(Self(Zeroizing::new(*bytes)))
    }

    /// Loads a hex key from an environment variable into a zeroizing owner.
    pub fn from_env(variable: &str) -> Result<Self> {
        if variable.is_empty() {
            return Err(PowerError::Config(
                "seekable weight key environment variable must not be empty".to_string(),
            ));
        }
        let value = Zeroizing::new(std::env::var(variable).map_err(|_| {
            PowerError::Config(format!(
                "seekable weight key environment variable '{variable}' is not set"
            ))
        })?);
        Self::from_hex(&value)
    }

    fn cipher(&self) -> Result<Aes256Gcm> {
        Aes256Gcm::new_from_slice(self.0.as_slice())
            .map_err(|error| PowerError::Config(format!("invalid AES-256 weight key: {error}")))
    }
}

impl std::fmt::Debug for SeekableWeightKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SeekableWeightKey([REDACTED])")
    }
}

impl From<[u8; 32]> for SeekableWeightKey {
    fn from(value: [u8; 32]) -> Self {
        Self::new(value)
    }
}

/// Result of a complete bounded-memory authentication pass.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeekableEncryptedFileVerification {
    pub descriptor: SeekableEncryptedFileDescriptor,
    pub ciphertext_sha256: String,
    pub peak_plaintext_chunk_bytes: u64,
}

/// A positional reader for independently authenticated AES-256-GCM chunks.
pub struct SeekableEncryptedFile {
    file: File,
    cipher: Aes256Gcm,
    header: [u8; HEADER_BYTES],
    descriptor: SeekableEncryptedFileDescriptor,
}

impl SeekableEncryptedFile {
    pub fn open(path: impl AsRef<Path>, key: &SeekableWeightKey) -> Result<Self> {
        let path = path.as_ref();
        let metadata = std::fs::symlink_metadata(path)?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(PowerError::InvalidFormat(
                "encrypted weight source must be a regular non-symlink file".to_string(),
            ));
        }
        let mut file = OpenOptions::new().read(true).open(path)?;
        let mut header = [0_u8; HEADER_BYTES];
        file.read_exact(&mut header).map_err(|error| {
            PowerError::InvalidFormat(format!("encrypted weight header is truncated: {error}"))
        })?;
        let descriptor = parse_header(&header, metadata.len())?;
        Ok(Self {
            file,
            cipher: key.cipher()?,
            header,
            descriptor,
        })
    }

    pub fn descriptor(&self) -> &SeekableEncryptedFileDescriptor {
        &self.descriptor
    }

    pub fn read_range(
        &self,
        offset: u64,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<Zeroizing<Vec<u8>>> {
        check_cancelled(cancellation)?;
        let end = offset.checked_add(bytes).ok_or_else(|| {
            PowerError::InvalidFormat("encrypted plaintext range overflowed".to_string())
        })?;
        if end > self.descriptor.plaintext_bytes {
            return Err(PowerError::InvalidFormat(
                "encrypted plaintext range exceeds the authenticated file".to_string(),
            ));
        }
        let output_len = usize::try_from(bytes).map_err(|_| {
            PowerError::InvalidFormat(
                "encrypted plaintext range exceeds the host address space".to_string(),
            )
        })?;
        let mut output = Zeroizing::new(vec![0_u8; output_len]);
        if bytes == 0 {
            return Ok(output);
        }

        let chunk_bytes = u64::from(self.descriptor.chunk_bytes);
        let first = offset / chunk_bytes;
        let last = (end - 1) / chunk_bytes;
        let mut copied = 0_usize;
        for index in first..=last {
            check_cancelled(cancellation)?;
            let chunk = self.decrypt_chunk(index, cancellation)?;
            let chunk_start = index.checked_mul(chunk_bytes).ok_or_else(|| {
                PowerError::InvalidFormat("encrypted chunk offset overflowed".to_string())
            })?;
            let copy_start = offset.max(chunk_start) - chunk_start;
            let chunk_end = chunk_start
                .checked_add(u64::try_from(chunk.len()).map_err(|_| {
                    PowerError::InvalidFormat("encrypted chunk length overflowed".to_string())
                })?)
                .ok_or_else(|| {
                    PowerError::InvalidFormat("encrypted chunk range overflowed".to_string())
                })?;
            let copy_end = end.min(chunk_end) - chunk_start;
            let source_start = usize::try_from(copy_start).map_err(|_| {
                PowerError::InvalidFormat("encrypted range offset overflowed".to_string())
            })?;
            let source_end = usize::try_from(copy_end).map_err(|_| {
                PowerError::InvalidFormat("encrypted range offset overflowed".to_string())
            })?;
            let copied_now = source_end.saturating_sub(source_start);
            output[copied..copied + copied_now].copy_from_slice(&chunk[source_start..source_end]);
            copied = copied.saturating_add(copied_now);
        }
        check_cancelled(cancellation)?;
        if copied != output.len() {
            return Err(PowerError::InvalidFormat(
                "encrypted range reader produced an incomplete plaintext buffer".to_string(),
            ));
        }
        Ok(output)
    }

    pub fn verify(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<SeekableEncryptedFileVerification> {
        self.verify_into(&mut Sha256::new(), cancellation)
    }

    pub(super) fn verify_into(
        &self,
        collection_plaintext: &mut Sha256,
        cancellation: &CancellationToken,
    ) -> Result<SeekableEncryptedFileVerification> {
        let mut plaintext = Sha256::new();
        let mut ciphertext = Sha256::new();
        ciphertext.update(self.header);
        let mut peak = 0_u64;
        for index in 0..self.descriptor.chunks {
            check_cancelled(cancellation)?;
            let chunk =
                self.decrypt_chunk_with_hasher(index, cancellation, Some(&mut ciphertext))?;
            peak = peak.max(u64::try_from(chunk.len()).unwrap_or(u64::MAX));
            plaintext.update(chunk.as_slice());
            collection_plaintext.update(chunk.as_slice());
        }
        check_cancelled(cancellation)?;
        let plaintext_sha256 = format!("{:x}", plaintext.finalize());
        if plaintext_sha256 != self.descriptor.plaintext_sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: "seekable encrypted weight file".to_string(),
                expected: self.descriptor.plaintext_sha256.clone(),
                actual: plaintext_sha256,
            });
        }
        Ok(SeekableEncryptedFileVerification {
            descriptor: self.descriptor.clone(),
            ciphertext_sha256: format!("{:x}", ciphertext.finalize()),
            peak_plaintext_chunk_bytes: peak,
        })
    }

    fn decrypt_chunk(
        &self,
        index: u64,
        cancellation: &CancellationToken,
    ) -> Result<Zeroizing<Vec<u8>>> {
        self.decrypt_chunk_with_hasher(index, cancellation, None)
    }

    fn decrypt_chunk_with_hasher(
        &self,
        index: u64,
        cancellation: &CancellationToken,
        ciphertext_hasher: Option<&mut Sha256>,
    ) -> Result<Zeroizing<Vec<u8>>> {
        if index >= self.descriptor.chunks {
            return Err(PowerError::InvalidFormat(
                "encrypted chunk index exceeds the authenticated file".to_string(),
            ));
        }
        let plaintext_bytes = plaintext_chunk_bytes(&self.descriptor, index)?;
        let encrypted_bytes = plaintext_bytes.checked_add(TAG_BYTES).ok_or_else(|| {
            PowerError::InvalidFormat("encrypted chunk length overflowed".to_string())
        })?;
        let stride = u64::from(self.descriptor.chunk_bytes)
            .checked_add(u64::try_from(NONCE_BYTES + TAG_BYTES).unwrap_or(28))
            .ok_or_else(|| {
                PowerError::InvalidFormat("encrypted chunk stride overflowed".to_string())
            })?;
        let physical_offset = u64::try_from(HEADER_BYTES)
            .unwrap_or(64)
            .checked_add(index.checked_mul(stride).ok_or_else(|| {
                PowerError::InvalidFormat("encrypted chunk offset overflowed".to_string())
            })?)
            .ok_or_else(|| {
                PowerError::InvalidFormat("encrypted chunk offset overflowed".to_string())
            })?;
        let mut nonce_bytes = [0_u8; NONCE_BYTES];
        read_exact_loop(
            &mut nonce_bytes,
            physical_offset,
            cancellation,
            |buffer, position| read_at(&self.file, buffer, position),
        )?;
        let ciphertext_offset = physical_offset
            .checked_add(u64::try_from(NONCE_BYTES).unwrap_or(12))
            .ok_or_else(|| {
                PowerError::InvalidFormat("encrypted chunk offset overflowed".to_string())
            })?;
        let mut encrypted = Zeroizing::new(vec![0_u8; encrypted_bytes]);
        read_exact_loop(
            encrypted.as_mut_slice(),
            ciphertext_offset,
            cancellation,
            |buffer, position| read_at(&self.file, buffer, position),
        )?;
        if let Some(hasher) = ciphertext_hasher {
            hasher.update(nonce_bytes);
            hasher.update(encrypted.as_slice());
        }
        let aad = chunk_aad(&self.header, index);
        #[allow(deprecated)]
        self.cipher
            .decrypt_in_place(Nonce::from_slice(&nonce_bytes), &aad, &mut *encrypted)
            .map_err(|_| PowerError::IntegrityCheckFailed {
                model: "seekable encrypted weight file".to_string(),
                expected: "valid AES-256-GCM chunk authentication".to_string(),
                actual: "wrong key or modified ciphertext/header".to_string(),
            })?;
        Ok(encrypted)
    }
}

impl std::fmt::Debug for SeekableEncryptedFile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SeekableEncryptedFile")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

pub fn encrypt_seekable_weight_file(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    key: &SeekableWeightKey,
    chunk_bytes: u32,
) -> Result<SeekableEncryptedFileVerification> {
    validate_chunk_bytes(chunk_bytes)?;
    let source = source.as_ref();
    let destination = destination.as_ref();
    let metadata = std::fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.len() == 0 {
        return Err(PowerError::InvalidFormat(
            "plaintext weight source must be a non-empty regular non-symlink file".to_string(),
        ));
    }
    if destination.exists() {
        return Err(PowerError::Config(
            "encrypted weight destination already exists".to_string(),
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        PowerError::Config("encrypted weight destination has no parent".to_string())
    })?;
    std::fs::create_dir_all(parent)?;

    let mut source_file = File::open(source)?;
    let mut hash_buffer = Zeroizing::new(vec![0_u8; HASH_BUFFER_BYTES]);
    let mut plaintext_hasher = Sha256::new();
    loop {
        let read = source_file.read(hash_buffer.as_mut_slice())?;
        if read == 0 {
            break;
        }
        plaintext_hasher.update(&hash_buffer[..read]);
    }
    let plaintext_sha = plaintext_hasher.finalize();
    source_file.seek(SeekFrom::Start(0))?;

    let chunks = metadata.len().div_ceil(u64::from(chunk_bytes));
    if chunks > u64::from(u32::MAX) {
        return Err(PowerError::InvalidFormat(
            "plaintext weight file requires too many authenticated chunks".to_string(),
        ));
    }
    let header = encode_header(chunk_bytes, metadata.len(), &plaintext_sha);
    let cipher = key.cipher()?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&header)?;
    let mut ciphertext_hasher = Sha256::new();
    ciphertext_hasher.update(header);
    let mut peak = 0_u64;
    for index in 0..chunks {
        let remaining = metadata
            .len()
            .saturating_sub(index.saturating_mul(u64::from(chunk_bytes)));
        let length = usize::try_from(remaining.min(u64::from(chunk_bytes))).map_err(|_| {
            PowerError::InvalidFormat("plaintext chunk length exceeds the host range".to_string())
        })?;
        let mut chunk = Zeroizing::new(vec![0_u8; length]);
        source_file.read_exact(chunk.as_mut_slice())?;
        peak = peak.max(u64::try_from(length).unwrap_or(u64::MAX));
        let mut nonce_bytes = [0_u8; NONCE_BYTES];
        rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);
        let aad = chunk_aad(&header, index);
        #[allow(deprecated)]
        cipher
            .encrypt_in_place(Nonce::from_slice(&nonce_bytes), &aad, &mut *chunk)
            .map_err(|_| {
                PowerError::InferenceFailed(
                    "failed to encrypt an authenticated weight chunk".to_string(),
                )
            })?;
        temporary.write_all(&nonce_bytes)?;
        temporary.write_all(chunk.as_slice())?;
        ciphertext_hasher.update(nonce_bytes);
        ciphertext_hasher.update(chunk.as_slice());
    }
    temporary.flush()?;
    temporary.as_file().sync_all()?;
    let ciphertext_bytes = temporary.as_file().metadata()?.len();
    temporary
        .persist_noclobber(destination)
        .map_err(|error| PowerError::Io(error.error))?;

    Ok(SeekableEncryptedFileVerification {
        descriptor: SeekableEncryptedFileDescriptor {
            plaintext_bytes: metadata.len(),
            plaintext_sha256: format!("{plaintext_sha:x}"),
            ciphertext_bytes,
            chunk_bytes,
            chunks,
        },
        ciphertext_sha256: format!("{:x}", ciphertext_hasher.finalize()),
        peak_plaintext_chunk_bytes: peak,
    })
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "encrypted weight read was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    file.seek_read(buffer, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_file: &File, _buffer: &mut [u8], _offset: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "positional encrypted weight reads are unsupported on this platform",
    ))
}
