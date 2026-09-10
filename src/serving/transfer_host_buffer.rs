//! Host-buffered transfer state reuses [`SealedStateEnvelope`].
//!
//! Available when the `embedded-inference` feature is enabled. Wire tickets
//! remain opaque adapter connection metadata and must not carry a second sealed
//! persistence format (always enforced). Tickets are intentionally **not**
//! force-sealed with [`SealedStateEnvelope`]: the host buffers this helper seals
//! are the persistence boundary; tickets only carry connection metadata.
//! When a buffered-host (or local staging) adapter needs to persist opaque
//! transfer bytes, it seals them through this helper so Power does not invent a
//! peer-tier ciphertext schema beside `a3s.power.sealed-model-state.v1`.
//!
//! Identity v2 binds the serving privacy mode, `privacy_policy_sha256`, and
//! optional `attestation_policy_sha256` into the sealed state-id digest so a
//! peer (or reopen) with matching model/layout/execution bindings but a
//! mismatched privacy or attestation policy fails closed — parallel to
//! buffered-host loopback transfer AAD v2.
//!
//! This is not high-speed transport, production-adapter, attested-fabric, or
//! TEE-export evidence. Tickets stay adapter-owned; receipts stay content-free
//! proofs.

use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::error::{PowerError, Result};
use crate::inference::{
    InferenceLimits, OpenedSealedState, SealedStateBinding, SealedStateEnvelope, SealedStateKey,
    SealedStateRollbackPolicy, SealedStateScope,
};

use super::{ServingPrivacyMode, StateKind, StateTransferBinding};

/// Domain separation for transfer host-buffer state identifiers (v2 binds
/// privacy/attestation policy digests into the sealed state-id).
const TRANSFER_HOST_BUFFER_DOMAIN: &[u8] = b"a3s.power.transfer-host-buffer.v2\0";

/// Exact identity required to seal or open one host-buffered transfer payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferHostBufferIdentity {
    pub transfer_id: Uuid,
    pub source_worker_epoch: Uuid,
    pub destination_worker_epoch: Uuid,
    pub binding: StateTransferBinding,
    /// Immutable serving-profile generation for this process configuration.
    pub generation: u64,
    /// Profile privacy mode cryptographically bound into the sealed state-id.
    pub privacy: ServingPrivacyMode,
    /// Profile privacy-policy digest bound into the sealed state-id.
    pub privacy_policy_sha256: String,
    /// Optional attestation-policy digest; required when privacy is
    /// [`ServingPrivacyMode::AttestedPrivateFabric`].
    pub attestation_policy_sha256: Option<String>,
}

impl TransferHostBufferIdentity {
    pub fn validate(&self) -> Result<()> {
        self.binding.validate()?;
        if self.transfer_id.is_nil()
            || self.source_worker_epoch.is_nil()
            || self.destination_worker_epoch.is_nil()
        {
            return Err(PowerError::InvalidRequest(
                "transfer host-buffer identity requires non-nil transfer and worker epochs"
                    .to_string(),
            ));
        }
        if self.generation == 0 {
            return Err(PowerError::InvalidRequest(
                "transfer host-buffer generation must be greater than zero".to_string(),
            ));
        }
        validate_sha256(
            &self.privacy_policy_sha256,
            "transfer host-buffer privacy policy",
        )?;
        if let Some(policy) = &self.attestation_policy_sha256 {
            validate_sha256(policy, "transfer host-buffer attestation policy")?;
        }
        if matches!(self.privacy, ServingPrivacyMode::AttestedPrivateFabric)
            && self.attestation_policy_sha256.is_none()
        {
            return Err(PowerError::InvalidRequest(
                "transfer host-buffer attested-private-fabric identity requires attestation_policy_sha256"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

/// Derive the sealed-state binding for one transfer host buffer.
///
/// Model and layout digests reuse the transfer binding. Execution identity,
/// kind, sizes, epochs, transfer id, generation, and privacy/attestation
/// policy digests are folded into the sealed state-id digest so a stale,
/// swapped, or privacy-mismatched transfer cannot open.
pub fn sealed_binding_for_transfer_host_buffer(
    identity: &TransferHostBufferIdentity,
    limits: &InferenceLimits,
) -> Result<SealedStateBinding> {
    identity.validate()?;
    limits.checked_state_bytes(identity.binding.state_bytes, "transfer host-buffer state")?;
    let state_id = transfer_host_buffer_state_id(identity);
    SealedStateBinding::new(
        identity.binding.model_sha256.clone(),
        identity.binding.layout_sha256.clone(),
        state_id,
    )
}

/// Seal opaque transfer bytes with the existing sealed-model-state envelope.
pub fn seal_transfer_host_buffer(
    identity: &TransferHostBufferIdentity,
    state: &[u8],
    key: &SealedStateKey,
    limits: &InferenceLimits,
    cancellation: &CancellationToken,
) -> Result<SealedStateEnvelope> {
    identity.validate()?;
    if u64::try_from(state.len()).ok() != Some(identity.binding.state_bytes) {
        return Err(PowerError::InvalidRequest(
            "transfer host-buffer plaintext length must match the binding state_bytes".to_string(),
        ));
    }
    let binding = sealed_binding_for_transfer_host_buffer(identity, limits)?;
    SealedStateEnvelope::seal(
        &binding,
        identity.generation,
        state,
        key,
        SealedStateScope::TeeLocal,
        limits,
        cancellation,
    )
}

/// Open a host-buffered transfer envelope under the exact transfer identity.
///
/// Stale profile generation, swapped epochs, binding drift, or mismatched
/// privacy/attestation policy digests fail closed.
pub fn open_transfer_host_buffer(
    envelope: &SealedStateEnvelope,
    identity: &TransferHostBufferIdentity,
    key: &SealedStateKey,
    limits: &InferenceLimits,
    cancellation: &CancellationToken,
) -> Result<OpenedSealedState> {
    identity.validate()?;
    let binding = sealed_binding_for_transfer_host_buffer(identity, limits)?;
    let opened = envelope.open(
        &binding,
        key,
        SealedStateScope::TeeLocal,
        SealedStateRollbackPolicy::new(identity.generation),
        limits,
        cancellation,
    )?;
    if opened.generation() != identity.generation {
        return Err(PowerError::InvalidRequest(
            "transfer host-buffer envelope generation does not match the serving profile"
                .to_string(),
        ));
    }
    if u64::try_from(opened.as_bytes().len()).ok() != Some(identity.binding.state_bytes) {
        return Err(PowerError::InvalidRequest(
            "opened transfer host-buffer length does not match the binding state_bytes".to_string(),
        ));
    }
    Ok(opened)
}

fn transfer_host_buffer_state_id(identity: &TransferHostBufferIdentity) -> String {
    let mut hasher = Sha256::new();
    hasher.update(TRANSFER_HOST_BUFFER_DOMAIN);
    hasher.update(identity.transfer_id.as_bytes());
    hasher.update(identity.source_worker_epoch.as_bytes());
    hasher.update(identity.destination_worker_epoch.as_bytes());
    hasher.update(identity.binding.model_sha256.as_bytes());
    hasher.update(identity.binding.execution_sha256.as_bytes());
    hasher.update(identity.binding.layout_sha256.as_bytes());
    hasher.update(state_kind_tag(identity.binding.state_kind));
    hasher.update(identity.binding.token_count.to_le_bytes());
    hasher.update(identity.binding.state_bytes.to_le_bytes());
    hasher.update(identity.generation.to_le_bytes());
    hasher.update(privacy_mode_tag(identity.privacy));
    hasher.update(identity.privacy_policy_sha256.as_bytes());
    hasher.update(b"\0");
    match &identity.attestation_policy_sha256 {
        Some(policy) => {
            hasher.update(b"attestation\0");
            hasher.update(policy.as_bytes());
        }
        None => hasher.update(b"attestation-absent"),
    }
    hex::encode(hasher.finalize())
}

fn state_kind_tag(kind: StateKind) -> &'static [u8] {
    match kind {
        StateKind::KvCache => b"kv-cache",
        StateKind::Recurrent => b"recurrent",
    }
}

fn privacy_mode_tag(privacy: ServingPrivacyMode) -> &'static [u8] {
    match privacy {
        ServingPrivacyMode::AuthenticatedEncryptedTransport => b"authenticated-encrypted-transport",
        ServingPrivacyMode::AttestedPrivateFabric => b"attested-private-fabric",
    }
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{SealedStateExportScope, SealedStateStore};

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn identity(generation: u64) -> TransferHostBufferIdentity {
        TransferHostBufferIdentity {
            transfer_id: Uuid::from_u128(11),
            source_worker_epoch: Uuid::from_u128(22),
            destination_worker_epoch: Uuid::from_u128(33),
            binding: StateTransferBinding {
                model_sha256: digest('1'),
                execution_sha256: digest('2'),
                layout_sha256: digest('3'),
                state_kind: StateKind::KvCache,
                token_count: 16,
                state_bytes: 32,
            },
            generation,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: digest('7'),
            attestation_policy_sha256: None,
        }
    }

    fn limits() -> InferenceLimits {
        InferenceLimits {
            max_state_bytes: 1_024,
            ..InferenceLimits::default()
        }
    }

    #[test]
    fn host_buffer_reuses_sealed_model_state_envelope_and_store() {
        let identity = identity(7);
        let key = SealedStateKey::from_bytes([0x51; 32]);
        let state = vec![0xab; 32];
        let cancellation = CancellationToken::new();
        let envelope =
            seal_transfer_host_buffer(&identity, &state, &key, &limits(), &cancellation).unwrap();
        assert_eq!(envelope.export_scope(), SealedStateExportScope::TeeLocal);
        assert_eq!(envelope.generation(), 7);

        let opened =
            open_transfer_host_buffer(&envelope, &identity, &key, &limits(), &cancellation)
                .unwrap();
        assert_eq!(opened.generation(), 7);
        assert_eq!(opened.as_bytes(), state.as_slice());

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("transfer-host-buffer.bin");
        let store = SealedStateStore::new(&target).unwrap();
        let binding = sealed_binding_for_transfer_host_buffer(&identity, &limits()).unwrap();
        store
            .commit(
                &envelope,
                &binding,
                &key,
                SealedStateScope::TeeLocal,
                SealedStateRollbackPolicy::new(7),
                &limits(),
                &cancellation,
            )
            .unwrap();
        let recovered = store
            .load(
                &binding,
                &key,
                SealedStateScope::TeeLocal,
                SealedStateRollbackPolicy::new(7),
                &limits(),
                &cancellation,
            )
            .unwrap()
            .expect("committed transfer host buffer must recover");
        assert_eq!(recovered.generation(), 7);
        assert_eq!(recovered.as_bytes(), state.as_slice());
    }

    #[test]
    fn stale_generation_and_swapped_binding_fail_closed() {
        let identity = identity(7);
        let key = SealedStateKey::from_bytes([0x52; 32]);
        let state = vec![0xcd; 32];
        let cancellation = CancellationToken::new();
        let envelope =
            seal_transfer_host_buffer(&identity, &state, &key, &limits(), &cancellation).unwrap();

        let mut stale = identity.clone();
        stale.generation = 8;
        assert!(
            open_transfer_host_buffer(&envelope, &stale, &key, &limits(), &cancellation).is_err()
        );

        let mut swapped_epoch = identity.clone();
        swapped_epoch.source_worker_epoch = Uuid::from_u128(99);
        assert!(open_transfer_host_buffer(
            &envelope,
            &swapped_epoch,
            &key,
            &limits(),
            &cancellation,
        )
        .is_err());

        let mut wrong_len = identity.clone();
        wrong_len.binding.state_bytes = 31;
        assert!(
            seal_transfer_host_buffer(&wrong_len, &state, &key, &limits(), &cancellation).is_err()
        );
    }

    #[test]
    fn mismatched_privacy_or_attestation_policy_fail_closed_on_open() {
        let identity = identity(7);
        let key = SealedStateKey::from_bytes([0x53; 32]);
        let state = vec![0xef; 32];
        let cancellation = CancellationToken::new();
        let envelope =
            seal_transfer_host_buffer(&identity, &state, &key, &limits(), &cancellation).unwrap();

        let mut privacy_mismatch = identity.clone();
        privacy_mismatch.privacy_policy_sha256 = digest('a');
        assert!(
            open_transfer_host_buffer(
                &envelope,
                &privacy_mismatch,
                &key,
                &limits(),
                &cancellation,
            )
            .is_err(),
            "privacy policy mismatch must fail closed on sealed host-buffer open"
        );

        let mut mode_mismatch = identity.clone();
        mode_mismatch.privacy = ServingPrivacyMode::AttestedPrivateFabric;
        mode_mismatch.attestation_policy_sha256 = Some(digest('8'));
        assert!(
            open_transfer_host_buffer(&envelope, &mode_mismatch, &key, &limits(), &cancellation)
                .is_err(),
            "privacy mode mismatch must fail closed on sealed host-buffer open"
        );

        let sealed_with_attestation = TransferHostBufferIdentity {
            attestation_policy_sha256: Some(digest('8')),
            ..identity.clone()
        };
        let attested_envelope = seal_transfer_host_buffer(
            &sealed_with_attestation,
            &state,
            &key,
            &limits(),
            &cancellation,
        )
        .unwrap();
        let mut attestation_mismatch = sealed_with_attestation.clone();
        attestation_mismatch.attestation_policy_sha256 = Some(digest('9'));
        assert!(
            open_transfer_host_buffer(
                &attested_envelope,
                &attestation_mismatch,
                &key,
                &limits(),
                &cancellation,
            )
            .is_err(),
            "attestation policy mismatch must fail closed on sealed host-buffer open"
        );

        // Matching privacy + attestation still opens.
        open_transfer_host_buffer(
            &attested_envelope,
            &sealed_with_attestation,
            &key,
            &limits(),
            &cancellation,
        )
        .expect("matching privacy/attestation identity must open");
    }

    #[test]
    fn attested_private_fabric_without_attestation_policy_fails_closed() {
        let mut identity = identity(7);
        identity.privacy = ServingPrivacyMode::AttestedPrivateFabric;
        identity.attestation_policy_sha256 = None;
        assert!(identity.validate().is_err());
    }

    #[test]
    fn sealed_binding_is_stable_and_domain_separated() {
        let first = sealed_binding_for_transfer_host_buffer(&identity(7), &limits()).unwrap();
        let second = sealed_binding_for_transfer_host_buffer(&identity(7), &limits()).unwrap();
        assert_eq!(first.weights_sha256(), digest('1'));
        assert_eq!(first.layout_sha256(), digest('3'));
        assert_eq!(first.state_id_sha256(), second.state_id_sha256());
        assert_ne!(
            first.state_id_sha256(),
            sealed_binding_for_transfer_host_buffer(&identity(8), &limits())
                .unwrap()
                .state_id_sha256()
        );
        let mut privacy_drift = identity(7);
        privacy_drift.privacy_policy_sha256 = digest('b');
        assert_ne!(
            first.state_id_sha256(),
            sealed_binding_for_transfer_host_buffer(&privacy_drift, &limits())
                .unwrap()
                .state_id_sha256(),
            "privacy policy must change the sealed state-id digest"
        );
    }
}
