mod authorization;
mod envelope;
mod store;
mod types;

pub use envelope::SealedStateEnvelope;
pub use store::SealedStateStore;
pub use types::{
    OpenedSealedState, RecoveredSealedState, SealedStateBinding, SealedStateExportScope,
    SealedStateKey, SealedStateRecoverySource, SealedStateRollbackPolicy, SealedStateScope,
    TeeStateExportAuthorization,
};

use crate::error::{PowerError, Result};

pub(super) fn check_cancelled(cancellation: &tokio_util::sync::CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "sealed model-state operation was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn decode_sha256(value: &str, label: &str) -> Result<[u8; 32]> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    let decoded = hex::decode(value)
        .map_err(|_| PowerError::InvalidRequest(format!("{label} SHA-256 could not be decoded")))?;
    decoded.try_into().map_err(|_| {
        PowerError::InvalidRequest(format!("{label} SHA-256 must contain exactly 32 bytes"))
    })
}

pub(super) fn encode_sha256(value: &[u8; 32]) -> String {
    hex::encode(value)
}
