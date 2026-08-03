use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};
use crate::inference::RuntimeDeviceIdentity;
use crate::tee::attestation::{
    canonical_claims_bytes, require_verified_hardware_claims, AttestationReport, ModelDigestKind,
};

use super::types::{
    validate_sha256, AcceleratorResidencyDeclaration, AcceleratorSecurityRequirement,
};

/// Digest-only binding from an already verified GPU-confidential attestation
/// report to one accelerator declaration.
///
/// This constructor deliberately reuses Power's existing canonical v2 claims
/// and CPU-TEE `report_data` binding. It performs structural and digest
/// matching, not a second hardware-signature verification; callers must pass a
/// report accepted by their existing strict verifier or produced inside the
/// current attested runtime.
#[derive(Clone, PartialEq, Eq)]
pub struct ConfidentialGpuBinding {
    claims_sha256: String,
    declaration_sha256: String,
    weights_sha256: String,
    execution_policy_sha256: String,
    runtime_device: RuntimeDeviceIdentity,
}

impl ConfidentialGpuBinding {
    pub fn from_verified_attestation_report(
        report: &AttestationReport,
        declaration: &AcceleratorResidencyDeclaration,
    ) -> Result<Self> {
        declaration.validate()?;
        if declaration.security != AcceleratorSecurityRequirement::ConfidentialGpu {
            return Err(PowerError::PolicyViolation(
                "a confidential GPU binding cannot be attached to a local accelerator declaration"
                    .to_string(),
            ));
        }
        let claims = require_verified_hardware_claims(report)?;

        let expected_weights = hex::decode(&declaration.weights_sha256).map_err(|_| {
            PowerError::InvalidFormat(
                "accelerator declaration contains an invalid weight digest".to_string(),
            )
        })?;
        let model = claims.model.as_ref().ok_or_else(|| {
            PowerError::PolicyViolation(
                "confidential GPU claims are not bound to the model weights".to_string(),
            )
        })?;
        if model.kind == ModelDigestKind::CiphertextArtifactSha256
            || model.digest != expected_weights
        {
            return Err(PowerError::PolicyViolation(
                "confidential GPU model claim does not match the canonical weight collection"
                    .to_string(),
            ));
        }

        let gpu = claims.gpu.as_ref().ok_or_else(|| {
            PowerError::PolicyViolation(
                "confidential GPU declaration has no GPU evidence claim".to_string(),
            )
        })?;
        if gpu.provider != "nvidia-nras"
            || gpu.evidence_digest.len() != 32
            || gpu
                .verdict_digest
                .as_ref()
                .is_none_or(|digest| digest.len() != 32)
            || !gpu.devices.iter().any(|device| device.device_type == "gpu")
            || gpu.nonce.as_ref().is_some_and(|nonce| {
                claims
                    .nonce
                    .as_ref()
                    .is_none_or(|claim_nonce| claim_nonce != nonce)
            })
        {
            return Err(PowerError::PolicyViolation(
                "confidential GPU evidence is missing the pinned NVIDIA NRAS evidence or verdict"
                    .to_string(),
            ));
        }

        let execution = claims
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.execution.as_ref())
            .ok_or_else(|| {
                PowerError::PolicyViolation(
                    "confidential GPU claims do not bind an execution policy".to_string(),
                )
            })?;
        let expected_policy = hex::decode(&declaration.execution_policy_sha256).map_err(|_| {
            PowerError::InvalidFormat(
                "accelerator declaration contains an invalid execution-policy digest".to_string(),
            )
        })?;
        if execution.gpu_sha256 != expected_policy {
            return Err(PowerError::PolicyViolation(
                "confidential GPU execution-policy claim does not match the accelerator declaration"
                    .to_string(),
            ));
        }

        let mut hasher = Sha256::new();
        hasher.update(canonical_claims_bytes(claims)?);
        let claims_sha256 = format!("{:x}", hasher.finalize());
        validate_sha256(&claims_sha256, "confidential GPU claims")?;
        Ok(Self {
            claims_sha256,
            declaration_sha256: declaration.declaration_sha256.clone(),
            weights_sha256: declaration.weights_sha256.clone(),
            execution_policy_sha256: declaration.execution_policy_sha256.clone(),
            runtime_device: declaration.runtime_device,
        })
    }

    pub fn claims_sha256(&self) -> &str {
        &self.claims_sha256
    }

    pub(super) fn validate_for(&self, declaration: &AcceleratorResidencyDeclaration) -> Result<()> {
        if self.declaration_sha256 != declaration.declaration_sha256
            || self.weights_sha256 != declaration.weights_sha256
            || self.execution_policy_sha256 != declaration.execution_policy_sha256
            || self.runtime_device != declaration.runtime_device
        {
            return Err(PowerError::PolicyViolation(
                "confidential GPU binding belongs to a different accelerator declaration"
                    .to_string(),
            ));
        }
        validate_sha256(&self.claims_sha256, "confidential GPU claims")
    }
}

impl std::fmt::Debug for ConfidentialGpuBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfidentialGpuBinding")
            .field("claims_sha256", &self.claims_sha256)
            .field("declaration_sha256", &self.declaration_sha256)
            .field("runtime_device", &self.runtime_device)
            .finish_non_exhaustive()
    }
}
