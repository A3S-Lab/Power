use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::super::sealed_state::decode_sha256;

pub(super) fn canonical_sha256(value: impl Into<String>, label: &str) -> Result<String> {
    let value = value.into().to_ascii_lowercase();
    validate_canonical_sha256(&value, label)?;
    Ok(value)
}

pub(super) fn validate_canonical_sha256(value: &str, label: &str) -> Result<()> {
    decode_sha256(value, label)?;
    if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(PowerError::InvalidFormat(format!(
            "{label} SHA-256 must use canonical lowercase hexadecimal"
        )));
    }
    Ok(())
}

pub(super) fn domain_sha256(domain: &[u8], payload: &[u8]) -> Result<String> {
    let payload_len = u64::try_from(payload.len()).map_err(|_| {
        PowerError::InvalidRequest("shape-profile canonical payload length overflowed".to_string())
    })?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(payload_len.to_le_bytes());
    digest.update(payload);
    Ok(format!("{:x}", digest.finalize()))
}
