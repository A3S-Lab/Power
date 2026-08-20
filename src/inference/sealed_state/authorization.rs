#[cfg(any(feature = "server", test))]
use sha2::{Digest, Sha256};
#[cfg(any(feature = "server", test))]
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::tee::attestation::TeeType;
#[cfg(any(feature = "server", test))]
use crate::tee::attestation::{
    canonical_claims_bytes, require_verified_hardware_claims, AttestationReport, ModelDigestKind,
};
#[cfg(feature = "server")]
use crate::verify::VerifiedHardwareAttestation;

use super::decode_sha256;
#[cfg(any(feature = "server", test))]
use super::encode_sha256;
use super::types::{SealedStateBinding, TeeStateExportAuthorization};

impl TeeStateExportAuthorization {
    /// Creates an explicit export token from an opaque strict-verification
    /// proof and caller-approved policy digest.
    ///
    /// The report carried by the proof must bind the exact plaintext/directory
    /// model digest so an authorization cannot be replayed across models.
    #[cfg(feature = "server")]
    pub fn from_verified_attestation(
        proof: &VerifiedHardwareAttestation<'_>,
        export_policy_sha256: &str,
    ) -> Result<Self> {
        Self::from_attestation_report(proof.report(), export_policy_sha256)
    }

    #[cfg(test)]
    pub(crate) fn from_attestation_report_for_test(
        report: &AttestationReport,
        export_policy_sha256: &str,
    ) -> Result<Self> {
        Self::from_attestation_report(report, export_policy_sha256)
    }

    #[cfg(any(feature = "server", test))]
    fn from_attestation_report(
        report: &AttestationReport,
        export_policy_sha256: &str,
    ) -> Result<Self> {
        let export_policy = decode_sha256(export_policy_sha256, "state export policy")?;
        let claims = require_verified_hardware_claims(report)?;
        let model = claims.model.as_ref().ok_or_else(|| {
            PowerError::PolicyViolation(
                "sealed-state export authorization requires a model-bound attestation claim"
                    .to_string(),
            )
        })?;
        if model.kind == ModelDigestKind::CiphertextArtifactSha256 || model.digest.len() != 32 {
            return Err(PowerError::PolicyViolation(
                "sealed-state export authorization requires the canonical plaintext or directory weight digest"
                    .to_string(),
            ));
        }
        let weights: [u8; 32] = model.digest.as_slice().try_into().map_err(|_| {
            PowerError::PolicyViolation(
                "attested model digest must contain exactly 32 bytes".to_string(),
            )
        })?;
        let canonical_claims = Zeroizing::new(canonical_claims_bytes(claims)?);
        let claims_digest: [u8; 32] = Sha256::digest(canonical_claims.as_slice()).into();
        let measurement_digest: [u8; 32] = Sha256::digest(&report.measurement).into();

        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-tee-state-export-authorization-v1\0");
        hasher.update(match report.tee_type {
            TeeType::SevSnp => [1_u8],
            TeeType::Tdx => [2_u8],
            TeeType::Simulated | TeeType::None => {
                return Err(PowerError::PolicyViolation(
                    "sealed-state export requires a hardware TEE".to_string(),
                ))
            }
        });
        hasher.update(claims_digest);
        hasher.update(measurement_digest);
        hasher.update(export_policy);
        hasher.update(weights);
        let authorization: [u8; 32] = hasher.finalize().into();

        Ok(Self {
            authorization_sha256: encode_sha256(&authorization),
            claims_sha256: encode_sha256(&claims_digest),
            measurement_sha256: encode_sha256(&measurement_digest),
            export_policy_sha256: export_policy_sha256.to_string(),
            weights_sha256: encode_sha256(&weights),
            tee_type: report.tee_type,
        })
    }

    pub(super) fn validate_for(&self, binding: &SealedStateBinding) -> Result<()> {
        decode_sha256(&self.authorization_sha256, "state export authorization")?;
        decode_sha256(&self.claims_sha256, "state export claims")?;
        decode_sha256(&self.measurement_sha256, "state export measurement")?;
        decode_sha256(&self.export_policy_sha256, "state export policy")?;
        decode_sha256(&self.weights_sha256, "state export model")?;
        if self.tee_type != TeeType::SevSnp && self.tee_type != TeeType::Tdx {
            return Err(PowerError::PolicyViolation(
                "state export authorization is not hardware TEE bound".to_string(),
            ));
        }
        if self.weights_sha256 != binding.weights_sha256() {
            return Err(PowerError::PolicyViolation(
                "state export authorization belongs to a different model".to_string(),
            ));
        }
        Ok(())
    }
}
