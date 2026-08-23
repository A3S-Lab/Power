#[cfg(any(feature = "server", test))]
use std::collections::BTreeSet;

#[cfg(any(feature = "server", test))]
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};
use crate::inference::RuntimeDeviceIdentity;
use crate::tee::attestation::TeeType;
#[cfg(any(feature = "server", test))]
use crate::tee::attestation::{
    canonical_claims_bytes, require_verified_hardware_claims, AttestationReport, ModelDigestKind,
};
#[cfg(feature = "server")]
use crate::verify::VerifiedConfidentialGpuAttestation;

#[cfg(any(feature = "server", test))]
use super::types::AcceleratorSecurityRequirement;
use super::types::{validate_sha256, AcceleratorResidencyDeclaration};

/// Digest-only binding from a strictly verified GPU-confidential attestation to
/// one accelerator declaration.
#[derive(Clone, PartialEq, Eq)]
pub struct ConfidentialGpuBinding {
    tee_type: TeeType,
    claims_sha256: String,
    declaration_sha256: String,
    weights_sha256: String,
    execution_policy_sha256: String,
    inference_execution_policy_sha256: String,
    auxiliary_artifacts_sha256: Option<String>,
    runtime_device: RuntimeDeviceIdentity,
    device_mesh_sha256: Option<String>,
}

impl ConfidentialGpuBinding {
    /// Bind a declaration to the exact report carried by an opaque proof from
    /// [`crate::verify::verify_confidential_gpu_attestation`].
    #[cfg(feature = "server")]
    pub fn from_verified_attestation(
        proof: &VerifiedConfidentialGpuAttestation<'_>,
        declaration: &AcceleratorResidencyDeclaration,
    ) -> Result<Self> {
        Self::from_attestation_report(proof.report(), declaration)
    }

    #[cfg(test)]
    pub(crate) fn from_attestation_report_for_test(
        report: &AttestationReport,
        declaration: &AcceleratorResidencyDeclaration,
    ) -> Result<Self> {
        Self::from_attestation_report(report, declaration)
    }

    #[cfg(any(feature = "server", test))]
    fn from_attestation_report(
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
        if let Some(mesh) = &declaration.device_mesh {
            validate_attested_mesh(mesh, &gpu.devices)?;
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
        let inference_execution_policy_sha256 = execution
            .inference_sha256
            .as_deref()
            .ok_or_else(|| {
                PowerError::PolicyViolation(
                    "confidential GPU claims do not bind an inference execution policy".to_string(),
                )
            })
            .and_then(|digest| {
                encode_attested_sha256(digest, "confidential GPU inference execution policy")
            })?;
        let auxiliary_artifacts_sha256 = execution
            .auxiliary_artifacts_sha256
            .as_deref()
            .map(|digest| encode_attested_sha256(digest, "confidential GPU auxiliary artifacts"))
            .transpose()?;

        let mut hasher = Sha256::new();
        hasher.update(canonical_claims_bytes(claims)?);
        let claims_sha256 = format!("{:x}", hasher.finalize());
        validate_sha256(&claims_sha256, "confidential GPU claims")?;
        Ok(Self {
            tee_type: claims.tee_type,
            claims_sha256,
            declaration_sha256: declaration.declaration_sha256.clone(),
            weights_sha256: declaration.weights_sha256.clone(),
            execution_policy_sha256: declaration.execution_policy_sha256.clone(),
            inference_execution_policy_sha256,
            auxiliary_artifacts_sha256,
            runtime_device: declaration.runtime_device,
            device_mesh_sha256: declaration
                .device_mesh
                .as_ref()
                .map(|mesh| mesh.mesh_sha256.clone()),
        })
    }

    pub fn tee_type(&self) -> TeeType {
        self.tee_type
    }

    pub fn claims_sha256(&self) -> &str {
        &self.claims_sha256
    }

    pub fn declaration_sha256(&self) -> &str {
        &self.declaration_sha256
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn execution_policy_sha256(&self) -> &str {
        &self.execution_policy_sha256
    }

    pub fn inference_execution_policy_sha256(&self) -> &str {
        &self.inference_execution_policy_sha256
    }

    pub fn auxiliary_artifacts_sha256(&self) -> Option<&str> {
        self.auxiliary_artifacts_sha256.as_deref()
    }

    pub fn runtime_device(&self) -> RuntimeDeviceIdentity {
        self.runtime_device
    }

    pub fn device_mesh_sha256(&self) -> Option<&str> {
        self.device_mesh_sha256.as_deref()
    }

    pub(super) fn validate_for(&self, declaration: &AcceleratorResidencyDeclaration) -> Result<()> {
        if !matches!(self.tee_type, TeeType::SevSnp | TeeType::Tdx) {
            return Err(PowerError::PolicyViolation(
                "confidential GPU binding requires a verified hardware TEE".to_string(),
            ));
        }
        if self.declaration_sha256 != declaration.declaration_sha256
            || self.weights_sha256 != declaration.weights_sha256
            || self.execution_policy_sha256 != declaration.execution_policy_sha256
            || self.runtime_device != declaration.runtime_device
            || self.device_mesh_sha256
                != declaration
                    .device_mesh
                    .as_ref()
                    .map(|mesh| mesh.mesh_sha256.clone())
        {
            return Err(PowerError::PolicyViolation(
                "confidential GPU binding belongs to a different accelerator declaration"
                    .to_string(),
            ));
        }
        validate_sha256(&self.claims_sha256, "confidential GPU claims")?;
        validate_sha256(
            &self.inference_execution_policy_sha256,
            "confidential GPU inference execution policy",
        )?;
        if let Some(auxiliary_artifacts_sha256) = &self.auxiliary_artifacts_sha256 {
            validate_sha256(
                auxiliary_artifacts_sha256,
                "confidential GPU auxiliary artifacts",
            )?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for ConfidentialGpuBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfidentialGpuBinding")
            .field("tee_type", &self.tee_type)
            .field("claims_sha256", &self.claims_sha256)
            .field("declaration_sha256", &self.declaration_sha256)
            .field("inference_execution_policy", &"sha256")
            .field(
                "auxiliary_artifacts",
                &self.auxiliary_artifacts_sha256.as_ref().map(|_| "sha256"),
            )
            .field("runtime_device", &self.runtime_device)
            .field("device_mesh_sha256", &self.device_mesh_sha256)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "server", test))]
fn encode_attested_sha256(digest: &[u8], label: &str) -> Result<String> {
    if digest.len() != 32 {
        return Err(PowerError::PolicyViolation(format!(
            "{label} must be a 32-byte SHA-256 digest"
        )));
    }
    let digest = hex::encode(digest);
    validate_sha256(&digest, label)?;
    Ok(digest)
}

#[cfg(any(feature = "server", test))]
fn validate_attested_mesh(
    mesh: &super::AcceleratorDeviceMeshDeclaration,
    devices: &[crate::tee::attestation::GpuDeviceClaim],
) -> Result<()> {
    let expected_gpus = mesh
        .nodes
        .iter()
        .filter_map(|node| node.attestation_gpu_claim_index)
        .collect::<BTreeSet<_>>();
    let expected_fabrics = mesh
        .attestation_fabric_claim_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut actual_gpus = BTreeSet::new();
    let mut actual_fabrics = BTreeSet::new();
    let mut all_indices = BTreeSet::new();
    for device in devices {
        if !all_indices.insert(device.index) {
            return Err(PowerError::PolicyViolation(
                "confidential GPU evidence contains duplicate device claim indices".to_string(),
            ));
        }
        match device.device_type.as_str() {
            "gpu" => {
                actual_gpus.insert(device.index);
            }
            "nvswitch" => {
                actual_fabrics.insert(device.index);
            }
            _ => {
                return Err(PowerError::PolicyViolation(
                    "confidential mesh evidence contains an unsupported NVIDIA device type"
                        .to_string(),
                ))
            }
        }
    }
    if expected_gpus != actual_gpus || expected_fabrics != actual_fabrics {
        return Err(PowerError::PolicyViolation(
            "confidential accelerator mesh does not bind the exact attested GPU/NVSwitch claim topology"
                .to_string(),
        ));
    }
    Ok(())
}
