use aes_gcm::aead::{consts::U12, Aead, AeadCore, KeyInit, OsRng, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::types::{
    OpenedSealedState, SealedStateBinding, SealedStateExportScope, SealedStateKey,
    SealedStateRollbackPolicy, SealedStateScope, TeeStateExportAuthorization,
};
use super::{check_cancelled, decode_sha256, encode_sha256};

const MAGIC: [u8; 8] = *b"A3SPST1\0";
const VERSION: u16 = 1;
pub(super) const SEALED_STATE_HEADER_BYTES: usize = 180;
const AEAD_TAG_BYTES: u64 = 16;
// NIST SP 800-38D permits at most 2^39 - 256 plaintext bits per GCM invocation.
const MAX_AES_GCM_PLAINTEXT_BYTES: u64 = (1_u64 << 36) - 32;
const LOCAL_SCOPE: u8 = 0;
const AUTHORIZED_SCOPE: u8 = 1;

#[derive(Clone, PartialEq, Eq)]
struct EnvelopeHeader {
    scope: SealedStateExportScope,
    generation: u64,
    plaintext_bytes: u64,
    ciphertext_bytes: u64,
    weights_sha256: [u8; 32],
    layout_sha256: [u8; 32],
    state_id_sha256: [u8; 32],
    export_authorization_sha256: [u8; 32],
    nonce: [u8; 12],
}

/// Authenticated, bounded ciphertext for opaque model-owned warm state.
///
/// The binary header contains digests and lengths only. It is authenticated as
/// AES-256-GCM associated data, while the model-owned state remains encrypted.
/// Local-only envelopes intentionally expose no raw-byte accessor.
pub struct SealedStateEnvelope {
    header: EnvelopeHeader,
    encoded: Zeroizing<Vec<u8>>,
}

impl SealedStateEnvelope {
    pub const SCHEMA: &'static str = "a3s.power.sealed-model-state.v1";

    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        binding: &SealedStateBinding,
        generation: u64,
        state: &[u8],
        key: &SealedStateKey,
        scope: SealedStateScope<'_>,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        check_cancelled(cancellation)?;
        binding.validate()?;
        if generation == 0 {
            return Err(PowerError::InvalidRequest(
                "sealed state generation must be greater than zero".to_string(),
            ));
        }
        if state.is_empty() {
            return Err(PowerError::InvalidRequest(
                "sealed model state must not be empty".to_string(),
            ));
        }
        let plaintext_bytes = u64::try_from(state.len()).map_err(|_| {
            PowerError::InvalidRequest("sealed model state length exceeds u64".to_string())
        })?;
        limits.checked_state_bytes(plaintext_bytes, "sealed model state")?;
        validate_cipher_plaintext_bound(plaintext_bytes)?;
        let ciphertext_bytes = plaintext_bytes.checked_add(AEAD_TAG_BYTES).ok_or_else(|| {
            PowerError::InvalidRequest("sealed state ciphertext length overflowed".to_string())
        })?;
        let (export_scope, export_authorization_sha256) = scope_fields(scope, binding)?;
        let generated_nonce = Aes256Gcm::generate_nonce(&mut OsRng);
        let nonce: [u8; 12] = generated_nonce.into();
        let header = EnvelopeHeader {
            scope: export_scope,
            generation,
            plaintext_bytes,
            ciphertext_bytes,
            weights_sha256: decode_sha256(binding.weights_sha256(), "sealed state weights")?,
            layout_sha256: decode_sha256(binding.layout_sha256(), "sealed state layout")?,
            state_id_sha256: decode_sha256(binding.state_id_sha256(), "sealed state identifier")?,
            export_authorization_sha256,
            nonce,
        };
        let aad = Zeroizing::new(header.encode());
        let cipher = Aes256Gcm::new_from_slice(key.as_bytes()).map_err(|_| {
            PowerError::InvalidRequest("sealed state key must contain 32 bytes".to_string())
        })?;
        let ciphertext = cipher
            .encrypt(
                &Nonce::<U12>::from(header.nonce),
                Payload {
                    msg: state,
                    aad: &aad,
                },
            )
            .map_err(|_| PowerError::InferenceFailed("failed to seal model state".to_string()))?;
        check_cancelled(cancellation)?;
        if u64::try_from(ciphertext.len()).ok() != Some(ciphertext_bytes) {
            return Err(PowerError::InferenceFailed(
                "sealed state cipher returned an unexpected ciphertext length".to_string(),
            ));
        }
        let capacity = SEALED_STATE_HEADER_BYTES
            .checked_add(ciphertext.len())
            .ok_or_else(|| PowerError::InvalidRequest("sealed envelope size overflowed".into()))?;
        let mut encoded = Zeroizing::new(Vec::with_capacity(capacity));
        encoded.extend_from_slice(&aad);
        encoded.extend_from_slice(&ciphertext);
        Ok(Self { header, encoded })
    }

    /// Imports an authorized or local sealed envelope from bounded bytes.
    /// Authentication and model/scope matching happen in [`Self::open`].
    pub fn import(encoded: &[u8], limits: &InferenceLimits) -> Result<Self> {
        // Reject before copying so an untrusted oversized slice cannot cause a
        // second allocation outside the configured state bound.
        validate_encoded_bound(encoded.len(), limits)?;
        Self::import_owned(Zeroizing::new(encoded.to_vec()), limits)
    }

    pub fn generation(&self) -> u64 {
        self.header.generation
    }

    pub fn export_scope(&self) -> SealedStateExportScope {
        self.header.scope
    }

    /// Returns ciphertext bytes only for an envelope carrying this exact
    /// hardware-TEE export authorization.
    pub fn export(
        &self,
        authorization: &TeeStateExportAuthorization,
    ) -> Result<Zeroizing<Vec<u8>>> {
        if self.header.scope != SealedStateExportScope::TeeAuthorized {
            return Err(PowerError::PolicyViolation(
                "TEE-local sealed state cannot be exported".to_string(),
            ));
        }
        let authorization_digest = decode_sha256(
            authorization.authorization_sha256(),
            "state export authorization",
        )?;
        let weights = decode_sha256(authorization.weights_sha256(), "state export model")?;
        if authorization_digest != self.header.export_authorization_sha256
            || weights != self.header.weights_sha256
        {
            return Err(PowerError::PolicyViolation(
                "state export authorization does not match this sealed envelope".to_string(),
            ));
        }
        Ok(Zeroizing::new(self.encoded.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &self,
        binding: &SealedStateBinding,
        key: &SealedStateKey,
        scope: SealedStateScope<'_>,
        rollback: SealedStateRollbackPolicy,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<OpenedSealedState> {
        check_cancelled(cancellation)?;
        validate_encoded_bound(self.encoded.len(), limits)?;
        limits.checked_state_bytes(self.header.plaintext_bytes, "sealed model state")?;
        validate_cipher_plaintext_bound(self.header.plaintext_bytes)?;
        rollback.validate(self.header.generation)?;
        validate_binding(&self.header, binding)?;
        validate_scope(&self.header, binding, scope)?;
        let aad = self
            .encoded
            .get(..SEALED_STATE_HEADER_BYTES)
            .ok_or_else(|| {
                PowerError::InvalidFormat("sealed state envelope is missing its header".to_string())
            })?;
        let ciphertext = self
            .encoded
            .get(SEALED_STATE_HEADER_BYTES..)
            .ok_or_else(|| {
                PowerError::InvalidFormat("sealed state envelope is missing ciphertext".to_string())
            })?;
        let cipher = Aes256Gcm::new_from_slice(key.as_bytes()).map_err(|_| {
            PowerError::InvalidRequest("sealed state key must contain 32 bytes".to_string())
        })?;
        let plaintext = cipher
            .decrypt(
                &Nonce::<U12>::from(self.header.nonce),
                Payload {
                    msg: ciphertext,
                    aad,
                },
            )
            .map_err(|_| PowerError::IntegrityCheckFailed {
                model: "sealed model state".to_string(),
                expected: "authenticated AES-256-GCM envelope".to_string(),
                actual: "wrong key or tampered envelope".to_string(),
            })?;
        let plaintext = Zeroizing::new(plaintext);
        check_cancelled(cancellation)?;
        if u64::try_from(plaintext.len()).ok() != Some(self.header.plaintext_bytes) {
            return Err(PowerError::InvalidFormat(
                "opened sealed state length does not match its authenticated header".to_string(),
            ));
        }
        Ok(OpenedSealedState {
            generation: self.header.generation,
            bytes: plaintext,
        })
    }

    pub(super) fn encoded(&self) -> &[u8] {
        &self.encoded
    }

    pub(super) fn import_owned(
        encoded: Zeroizing<Vec<u8>>,
        limits: &InferenceLimits,
    ) -> Result<Self> {
        validate_encoded_bound(encoded.len(), limits)?;
        let total_bytes = u64::try_from(encoded.len()).map_err(|_| {
            PowerError::InvalidRequest("sealed state envelope length exceeds u64".to_string())
        })?;
        let header = EnvelopeHeader::parse(&encoded, total_bytes, limits)?;
        Ok(Self { header, encoded })
    }

    pub(super) fn inspect_generation(
        encoded_header: &[u8],
        total_bytes: u64,
        limits: &InferenceLimits,
    ) -> Result<u64> {
        validate_encoded_bound_u64(total_bytes, limits)?;
        Ok(EnvelopeHeader::parse(encoded_header, total_bytes, limits)?.generation)
    }
}

impl std::fmt::Debug for SealedStateEnvelope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SealedStateEnvelope")
            .field("schema", &Self::SCHEMA)
            .field("generation", &self.header.generation)
            .field("scope", &self.header.scope)
            .field("plaintext_bytes", &self.header.plaintext_bytes)
            .field(
                "weights_sha256",
                &encode_sha256(&self.header.weights_sha256),
            )
            .field("layout_sha256", &encode_sha256(&self.header.layout_sha256))
            .field(
                "state_id_sha256",
                &encode_sha256(&self.header.state_id_sha256),
            )
            .finish_non_exhaustive()
    }
}

impl EnvelopeHeader {
    fn encode(&self) -> Vec<u8> {
        let mut encoded = Vec::with_capacity(SEALED_STATE_HEADER_BYTES);
        encoded.extend_from_slice(&MAGIC);
        encoded.extend_from_slice(&VERSION.to_le_bytes());
        encoded.push(match self.scope {
            SealedStateExportScope::TeeLocal => LOCAL_SCOPE,
            SealedStateExportScope::TeeAuthorized => AUTHORIZED_SCOPE,
        });
        encoded.push(0);
        encoded.extend_from_slice(&(SEALED_STATE_HEADER_BYTES as u32).to_le_bytes());
        encoded.extend_from_slice(&self.generation.to_le_bytes());
        encoded.extend_from_slice(&self.plaintext_bytes.to_le_bytes());
        encoded.extend_from_slice(&self.ciphertext_bytes.to_le_bytes());
        encoded.extend_from_slice(&self.weights_sha256);
        encoded.extend_from_slice(&self.layout_sha256);
        encoded.extend_from_slice(&self.state_id_sha256);
        encoded.extend_from_slice(&self.export_authorization_sha256);
        encoded.extend_from_slice(&self.nonce);
        encoded
    }

    fn parse(encoded: &[u8], total_bytes: u64, limits: &InferenceLimits) -> Result<Self> {
        if encoded.len() < SEALED_STATE_HEADER_BYTES {
            return Err(PowerError::InvalidFormat(
                "sealed state envelope is truncated before its fixed header".to_string(),
            ));
        }
        let mut reader = HeaderReader::new(encoded);
        if reader.take::<8>()? != MAGIC || u16::from_le_bytes(reader.take::<2>()?) != VERSION {
            return Err(PowerError::InvalidFormat(
                "sealed state envelope has an unsupported magic or version".to_string(),
            ));
        }
        let scope = match reader.take::<1>()?[0] {
            LOCAL_SCOPE => SealedStateExportScope::TeeLocal,
            AUTHORIZED_SCOPE => SealedStateExportScope::TeeAuthorized,
            _ => {
                return Err(PowerError::InvalidFormat(
                    "sealed state envelope has an unsupported export scope".to_string(),
                ))
            }
        };
        if reader.take::<1>()? != [0]
            || u32::from_le_bytes(reader.take::<4>()?) != SEALED_STATE_HEADER_BYTES as u32
        {
            return Err(PowerError::InvalidFormat(
                "sealed state envelope has non-zero reserved data or a wrong header length"
                    .to_string(),
            ));
        }
        let generation = u64::from_le_bytes(reader.take::<8>()?);
        let plaintext_bytes = u64::from_le_bytes(reader.take::<8>()?);
        let ciphertext_bytes = u64::from_le_bytes(reader.take::<8>()?);
        let header = Self {
            scope,
            generation,
            plaintext_bytes,
            ciphertext_bytes,
            weights_sha256: reader.take::<32>()?,
            layout_sha256: reader.take::<32>()?,
            state_id_sha256: reader.take::<32>()?,
            export_authorization_sha256: reader.take::<32>()?,
            nonce: reader.take::<12>()?,
        };
        if generation == 0 || plaintext_bytes == 0 {
            return Err(PowerError::InvalidFormat(
                "sealed state generation and plaintext length must be non-zero".to_string(),
            ));
        }
        limits.checked_state_bytes(plaintext_bytes, "sealed model state")?;
        validate_cipher_plaintext_bound(plaintext_bytes)?;
        if ciphertext_bytes
            != plaintext_bytes.checked_add(AEAD_TAG_BYTES).ok_or_else(|| {
                PowerError::InvalidFormat("sealed state ciphertext length overflowed".to_string())
            })?
        {
            return Err(PowerError::InvalidFormat(
                "sealed state ciphertext length is inconsistent".to_string(),
            ));
        }
        let expected = u64::try_from(SEALED_STATE_HEADER_BYTES)
            .ok()
            .and_then(|header_bytes| header_bytes.checked_add(ciphertext_bytes))
            .ok_or_else(|| {
                PowerError::InvalidFormat("sealed envelope length overflowed".to_string())
            })?;
        if total_bytes != expected {
            return Err(PowerError::InvalidFormat(
                "sealed state envelope is truncated or has trailing bytes".to_string(),
            ));
        }
        let authorization_is_zero = header.export_authorization_sha256 == [0_u8; 32];
        if (scope == SealedStateExportScope::TeeLocal && !authorization_is_zero)
            || (scope == SealedStateExportScope::TeeAuthorized && authorization_is_zero)
        {
            return Err(PowerError::InvalidFormat(
                "sealed state export scope and authorization digest are inconsistent".to_string(),
            ));
        }
        Ok(header)
    }
}

fn scope_fields(
    scope: SealedStateScope<'_>,
    binding: &SealedStateBinding,
) -> Result<(SealedStateExportScope, [u8; 32])> {
    match scope {
        SealedStateScope::TeeLocal => Ok((SealedStateExportScope::TeeLocal, [0_u8; 32])),
        SealedStateScope::TeeAuthorizedExport(authorization) => {
            authorization.validate_for(binding)?;
            Ok((
                SealedStateExportScope::TeeAuthorized,
                decode_sha256(
                    authorization.authorization_sha256(),
                    "state export authorization",
                )?,
            ))
        }
    }
}

fn validate_binding(header: &EnvelopeHeader, binding: &SealedStateBinding) -> Result<()> {
    binding.validate()?;
    if header.weights_sha256 != decode_sha256(binding.weights_sha256(), "sealed state weights")?
        || header.layout_sha256 != decode_sha256(binding.layout_sha256(), "sealed state layout")?
        || header.state_id_sha256
            != decode_sha256(binding.state_id_sha256(), "sealed state identifier")?
    {
        return Err(PowerError::PolicyViolation(
            "sealed state belongs to a different model, layout, or state identifier".to_string(),
        ));
    }
    Ok(())
}

fn validate_scope(
    header: &EnvelopeHeader,
    binding: &SealedStateBinding,
    scope: SealedStateScope<'_>,
) -> Result<()> {
    match (header.scope, scope) {
        (SealedStateExportScope::TeeLocal, SealedStateScope::TeeLocal) => Ok(()),
        (
            SealedStateExportScope::TeeAuthorized,
            SealedStateScope::TeeAuthorizedExport(authorization),
        ) => {
            authorization.validate_for(binding)?;
            if header.export_authorization_sha256
                != decode_sha256(
                    authorization.authorization_sha256(),
                    "state export authorization",
                )?
            {
                return Err(PowerError::PolicyViolation(
                    "sealed state requires a different TEE export authorization".to_string(),
                ));
            }
            Ok(())
        }
        _ => Err(PowerError::PolicyViolation(
            "sealed state local/export scope does not match the requested access".to_string(),
        )),
    }
}

fn validate_encoded_bound(encoded_bytes: usize, limits: &InferenceLimits) -> Result<()> {
    let encoded_bytes = u64::try_from(encoded_bytes).map_err(|_| {
        PowerError::InvalidRequest("sealed state envelope length exceeds u64".to_string())
    })?;
    validate_encoded_bound_u64(encoded_bytes, limits)
}

fn validate_encoded_bound_u64(encoded_bytes: u64, limits: &InferenceLimits) -> Result<()> {
    let maximum = maximum_encoded_bytes(limits)?;
    if encoded_bytes > maximum {
        return Err(PowerError::InvalidRequest(format!(
            "sealed state envelope exceeds the {maximum} byte authenticated bound"
        )));
    }
    Ok(())
}

pub(super) fn maximum_encoded_bytes(limits: &InferenceLimits) -> Result<u64> {
    limits
        .max_state_bytes
        .min(MAX_AES_GCM_PLAINTEXT_BYTES)
        .checked_add(AEAD_TAG_BYTES)
        .and_then(|bytes| bytes.checked_add(SEALED_STATE_HEADER_BYTES as u64))
        .ok_or_else(|| PowerError::Config("sealed state envelope limit overflowed".to_string()))
}

fn validate_cipher_plaintext_bound(plaintext_bytes: u64) -> Result<()> {
    if plaintext_bytes > MAX_AES_GCM_PLAINTEXT_BYTES {
        return Err(PowerError::InvalidRequest(format!(
            "sealed model state exceeds the {MAX_AES_GCM_PLAINTEXT_BYTES} byte AES-GCM invocation bound"
        )));
    }
    Ok(())
}

struct HeaderReader<'a> {
    encoded: &'a [u8],
    cursor: usize,
}

impl<'a> HeaderReader<'a> {
    fn new(encoded: &'a [u8]) -> Self {
        Self { encoded, cursor: 0 }
    }

    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self.cursor.checked_add(N).ok_or_else(|| {
            PowerError::InvalidFormat("sealed state header offset overflowed".to_string())
        })?;
        let bytes = self.encoded.get(self.cursor..end).ok_or_else(|| {
            PowerError::InvalidFormat("sealed state header is truncated".to_string())
        })?;
        self.cursor = end;
        bytes.try_into().map_err(|_| {
            PowerError::InvalidFormat("sealed state header field has a wrong length".to_string())
        })
    }
}
