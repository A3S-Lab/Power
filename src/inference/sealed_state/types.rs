use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;
use crate::tee::attestation::TeeType;

use super::decode_sha256;

/// Digest-only identity for opaque state serialized by a model crate.
///
/// Power never interprets the bytes. The exact weight collection prevents
/// cross-model replay, the layout digest versions the model-owned topology,
/// and the state identifier digest isolates sessions without storing a
/// plaintext conversation or tenant identifier.
#[derive(Clone, PartialEq, Eq)]
pub struct SealedStateBinding {
    weights_sha256: String,
    layout_sha256: String,
    state_id_sha256: String,
}

impl SealedStateBinding {
    pub fn new(
        weights_sha256: impl Into<String>,
        layout_sha256: impl Into<String>,
        state_id_sha256: impl Into<String>,
    ) -> Result<Self> {
        let binding = Self {
            weights_sha256: weights_sha256.into(),
            layout_sha256: layout_sha256.into(),
            state_id_sha256: state_id_sha256.into(),
        };
        binding.validate()?;
        Ok(binding)
    }

    /// Hashes a bounded opaque identifier into the envelope binding.
    ///
    /// This is domain separation, not anonymization. Callers should use a
    /// high-entropy random session identifier when the digest itself could be
    /// observed outside the trust boundary.
    pub fn for_identifier(
        weights_sha256: impl Into<String>,
        layout_sha256: impl Into<String>,
        state_identifier: &[u8],
        limits: &InferenceLimits,
    ) -> Result<Self> {
        if state_identifier.is_empty() || state_identifier.len() > limits.max_graph_name_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "sealed state identifier must contain between 1 and {} bytes",
                limits.max_graph_name_bytes
            )));
        }
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-sealed-state-identifier-v1\0");
        hasher.update((state_identifier.len() as u64).to_le_bytes());
        hasher.update(state_identifier);
        Self::new(
            weights_sha256,
            layout_sha256,
            format!("{:x}", hasher.finalize()),
        )
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn layout_sha256(&self) -> &str {
        &self.layout_sha256
    }

    pub fn state_id_sha256(&self) -> &str {
        &self.state_id_sha256
    }

    pub(super) fn validate(&self) -> Result<()> {
        decode_sha256(&self.weights_sha256, "sealed state weight collection")?;
        decode_sha256(&self.layout_sha256, "sealed state layout")?;
        decode_sha256(&self.state_id_sha256, "sealed state identifier")?;
        Ok(())
    }
}

impl std::fmt::Debug for SealedStateBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealedStateBinding")
            .field("weights_sha256", &self.weights_sha256)
            .field("layout_sha256", &self.layout_sha256)
            .field("state_id_sha256", &self.state_id_sha256)
            .finish()
    }
}

/// Owned state-sealing key that is zeroized on drop.
///
/// Construction consumes the caller's array so Power does not create a
/// long-lived cloneable key type. Callers should likewise clear any source key
/// material used to construct the array.
pub struct SealedStateKey {
    bytes: Zeroizing<[u8; 32]>,
}

impl SealedStateKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self {
            bytes: Zeroizing::new(bytes),
        }
    }

    pub(super) fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

impl std::fmt::Debug for SealedStateKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SealedStateKey([REDACTED])")
    }
}

/// Caller-owned rollback floor.
///
/// Power enforces this value but does not claim to provide a hardware monotonic
/// counter. A deployment that needs rollback resistance must retain the floor
/// in a trusted monotonic source outside the sealed-state file set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SealedStateRollbackPolicy {
    minimum_generation: u64,
}

impl SealedStateRollbackPolicy {
    pub const fn new(minimum_generation: u64) -> Self {
        Self { minimum_generation }
    }

    pub const fn minimum_generation(self) -> u64 {
        self.minimum_generation
    }

    pub(super) fn validate(self, generation: u64) -> Result<()> {
        if generation < self.minimum_generation {
            return Err(PowerError::PolicyViolation(format!(
                "sealed state generation {generation} is below the caller-pinned rollback floor {}",
                self.minimum_generation
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealedStateExportScope {
    TeeLocal,
    TeeAuthorized,
}

/// Access policy supplied for sealing, opening, or persistence.
///
/// Local-only envelopes never expose a public byte-export method. Moving state
/// outside the attested boundary requires an authorization derived from an
/// already hardware-verified report and bound to an explicit policy digest.
/// `TeeLocal` is a caller assertion about the selected storage/key boundary;
/// Power does not pretend that a filesystem path proves hardware isolation.
#[derive(Clone, Copy)]
pub enum SealedStateScope<'a> {
    TeeLocal,
    TeeAuthorizedExport(&'a TeeStateExportAuthorization),
}

impl std::fmt::Debug for SealedStateScope<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TeeLocal => formatter.write_str("TeeLocal"),
            Self::TeeAuthorizedExport(_) => formatter.write_str("TeeAuthorizedExport(..)"),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct TeeStateExportAuthorization {
    pub(super) authorization_sha256: String,
    pub(super) claims_sha256: String,
    pub(super) measurement_sha256: String,
    pub(super) export_policy_sha256: String,
    pub(super) weights_sha256: String,
    pub(super) tee_type: TeeType,
}

impl TeeStateExportAuthorization {
    pub fn authorization_sha256(&self) -> &str {
        &self.authorization_sha256
    }

    pub fn export_policy_sha256(&self) -> &str {
        &self.export_policy_sha256
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn tee_type(&self) -> TeeType {
        self.tee_type
    }
}

impl std::fmt::Debug for TeeStateExportAuthorization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TeeStateExportAuthorization")
            .field("authorization_sha256", &self.authorization_sha256)
            .field("claims_sha256", &self.claims_sha256)
            .field("measurement_sha256", &self.measurement_sha256)
            .field("export_policy_sha256", &self.export_policy_sha256)
            .field("weights_sha256", &self.weights_sha256)
            .field("tee_type", &self.tee_type)
            .finish()
    }
}

pub struct OpenedSealedState {
    pub(super) generation: u64,
    pub(super) bytes: Zeroizing<Vec<u8>>,
}

impl OpenedSealedState {
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Zeroizing<Vec<u8>> {
        self.bytes
    }
}

impl std::fmt::Debug for OpenedSealedState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenedSealedState")
            .field("generation", &self.generation)
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealedStateRecoverySource {
    Primary,
    Backup,
}

pub struct RecoveredSealedState {
    pub(super) source: SealedStateRecoverySource,
    pub(super) state: OpenedSealedState,
}

impl RecoveredSealedState {
    pub fn source(&self) -> SealedStateRecoverySource {
        self.source
    }

    pub fn generation(&self) -> u64 {
        self.state.generation()
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.state.as_bytes()
    }

    pub fn into_state(self) -> OpenedSealedState {
        self.state
    }
}

impl std::fmt::Debug for RecoveredSealedState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecoveredSealedState")
            .field("source", &self.source)
            .field("state", &self.state)
            .finish()
    }
}
