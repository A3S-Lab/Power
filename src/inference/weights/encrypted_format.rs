use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

const MAGIC: &[u8; 8] = b"A3SWGT01";
const VERSION: u32 = 1;
pub(super) const HEADER_BYTES: usize = 64;
pub(super) const TAG_BYTES: usize = 16;
pub(super) const NONCE_BYTES: usize = 12;
const MIN_CHUNK_BYTES: u32 = 4 * 1024;
const MAX_CHUNK_BYTES: u32 = 64 * 1024 * 1024;

pub const DEFAULT_ENCRYPTED_CHUNK_BYTES: u32 = 1024 * 1024;

/// Public, path-free metadata authenticated by every encrypted chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeekableEncryptedFileDescriptor {
    pub plaintext_bytes: u64,
    pub plaintext_sha256: String,
    pub ciphertext_bytes: u64,
    pub chunk_bytes: u32,
    pub chunks: u64,
}

pub(super) fn encode_header(
    chunk_bytes: u32,
    plaintext_bytes: u64,
    plaintext_sha256: &[u8],
) -> [u8; HEADER_BYTES] {
    let mut header = [0_u8; HEADER_BYTES];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&chunk_bytes.to_le_bytes());
    header[16..24].copy_from_slice(&plaintext_bytes.to_le_bytes());
    header[24..56].copy_from_slice(plaintext_sha256);
    header
}

pub(super) fn parse_header(
    header: &[u8; HEADER_BYTES],
    ciphertext_bytes: u64,
) -> Result<SeekableEncryptedFileDescriptor> {
    if &header[..8] != MAGIC {
        return Err(PowerError::InvalidFormat(
            "encrypted weight magic is unsupported".to_string(),
        ));
    }
    if read_u32(header, 8)? != VERSION {
        return Err(PowerError::InvalidFormat(
            "encrypted weight version is unsupported".to_string(),
        ));
    }
    if header[56..].iter().any(|byte| *byte != 0) {
        return Err(PowerError::InvalidFormat(
            "encrypted weight reserved header bytes must be zero".to_string(),
        ));
    }
    let chunk_bytes = read_u32(header, 12)?;
    validate_chunk_bytes(chunk_bytes)?;
    let plaintext_bytes = read_u64(header, 16)?;
    if plaintext_bytes == 0 {
        return Err(PowerError::InvalidFormat(
            "encrypted weight plaintext must not be empty".to_string(),
        ));
    }
    let chunks = plaintext_bytes.div_ceil(u64::from(chunk_bytes));
    if chunks > u64::from(u32::MAX) {
        return Err(PowerError::InvalidFormat(
            "encrypted weight file declares too many authenticated chunks".to_string(),
        ));
    }
    let expected_bytes = u64::try_from(HEADER_BYTES)
        .unwrap_or(64)
        .checked_add(plaintext_bytes)
        .and_then(|bytes| {
            chunks
                .checked_mul(u64::try_from(TAG_BYTES + NONCE_BYTES).unwrap_or(28))
                .and_then(|tags| bytes.checked_add(tags))
        })
        .ok_or_else(|| {
            PowerError::InvalidFormat("encrypted weight file length overflowed".to_string())
        })?;
    if ciphertext_bytes != expected_bytes {
        return Err(PowerError::InvalidFormat(format!(
            "encrypted weight file has {ciphertext_bytes} bytes, expected {expected_bytes}"
        )));
    }
    Ok(SeekableEncryptedFileDescriptor {
        plaintext_bytes,
        plaintext_sha256: hex_lower(&header[24..56]),
        ciphertext_bytes,
        chunk_bytes,
        chunks,
    })
}

pub(super) fn validate_chunk_bytes(chunk_bytes: u32) -> Result<()> {
    if !(MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES).contains(&chunk_bytes) || !chunk_bytes.is_power_of_two()
    {
        return Err(PowerError::Config(format!(
            "encrypted weight chunk size must be a power of two within {MIN_CHUNK_BYTES}..={MAX_CHUNK_BYTES} bytes"
        )));
    }
    Ok(())
}

pub(super) fn plaintext_chunk_bytes(
    descriptor: &SeekableEncryptedFileDescriptor,
    index: u64,
) -> Result<usize> {
    let start = index
        .checked_mul(u64::from(descriptor.chunk_bytes))
        .ok_or_else(|| {
            PowerError::InvalidFormat("encrypted chunk offset overflowed".to_string())
        })?;
    usize::try_from(
        descriptor
            .plaintext_bytes
            .saturating_sub(start)
            .min(u64::from(descriptor.chunk_bytes)),
    )
    .map_err(|_| PowerError::InvalidFormat("encrypted chunk length overflowed".to_string()))
}

pub(super) fn chunk_aad(header: &[u8; HEADER_BYTES], index: u64) -> [u8; HEADER_BYTES + 8] {
    let mut aad = [0_u8; HEADER_BYTES + 8];
    aad[..HEADER_BYTES].copy_from_slice(header);
    aad[HEADER_BYTES..].copy_from_slice(&index.to_le_bytes());
    aad
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    bytes
        .get(offset..offset + 4)
        .and_then(|value| value.try_into().ok())
        .map(u32::from_le_bytes)
        .ok_or_else(|| {
            PowerError::InvalidFormat("encrypted weight header is truncated".to_string())
        })
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    bytes
        .get(offset..offset + 8)
        .and_then(|value| value.try_into().ok())
        .map(u64::from_le_bytes)
        .ok_or_else(|| {
            PowerError::InvalidFormat("encrypted weight header is truncated".to_string())
        })
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}
