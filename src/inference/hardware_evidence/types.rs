use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::super::{
    ExecutionDigest, RuntimeDeviceIdentity, StorageBenchmarkComparison, StorageBenchmarkReport,
    StorageBenchmarkSystem, TuningProfileBinding, TuningProfileDecision, TuningProfileEvidence,
    TuningProfilePolicy,
};

/// Canonical model, runtime, device, and named-hardware binding shared by one
/// reviewable evidence bundle.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HardwareEvidenceBinding {
    pub power_version: String,
    pub power_commit: String,
    pub weights_sha256: String,
    pub graph_source_sha256: String,
    pub runtime_device: RuntimeDeviceIdentity,
    pub runtime_sha256: String,
    pub device_sha256: String,
    pub environment_sha256: String,
}

impl HardwareEvidenceBinding {
    pub fn new(
        power_version: impl Into<String>,
        power_commit: impl Into<String>,
        weights_sha256: impl Into<String>,
        graph_source_sha256: impl Into<String>,
        runtime_device: RuntimeDeviceIdentity,
        system: &StorageBenchmarkSystem,
    ) -> Result<Self> {
        super::binding(
            power_version.into(),
            power_commit.into(),
            weights_sha256.into(),
            graph_source_sha256.into(),
            runtime_device,
            system,
        )
    }

    /// Produces the exact canonical binding expected by Power's existing
    /// lossless tuning evaluator for this platform.
    pub fn tuning_binding(
        &self,
        calibration_workload: ExecutionDigest,
    ) -> Result<TuningProfileBinding> {
        super::tuning_binding(self, calibration_workload)
    }
}

impl std::fmt::Debug for HardwareEvidenceBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HardwareEvidenceBinding")
            .field("power_version", &self.power_version)
            .field("power_commit", &"revision")
            .field("weights", &"sha256")
            .field("graph_source", &"sha256")
            .field("runtime_device", &self.runtime_device)
            .field("runtime", &"sha256")
            .field("device", &"sha256")
            .field("environment", &"sha256")
            .finish()
    }
}

/// Digest-only parity evidence owned by a model integration.
///
/// `artifact_sha256` pins the model-owned detailed artifact without placing its
/// path, logs, tensors, prompts, or state bytes in Power's evidence bundle.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelParityArtifact {
    pub schema: String,
    pub binding: HardwareEvidenceBinding,
    pub case_sha256: String,
    pub artifact_sha256: String,
    pub reference_implementation_sha256: String,
    pub tested_configuration_sha256: String,
    pub workload: ExecutionDigest,
    pub reference_output: ExecutionDigest,
    pub tested_output: ExecutionDigest,
}

impl ModelParityArtifact {
    pub const SCHEMA: &'static str = "a3s.power.model-parity-artifact.v1";

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        binding: HardwareEvidenceBinding,
        case_sha256: impl Into<String>,
        artifact_sha256: impl Into<String>,
        reference_implementation_sha256: impl Into<String>,
        tested_configuration_sha256: impl Into<String>,
        workload: ExecutionDigest,
        reference_output: ExecutionDigest,
        tested_output: ExecutionDigest,
    ) -> Result<Self> {
        let artifact = Self {
            schema: Self::SCHEMA.to_string(),
            binding,
            case_sha256: case_sha256.into(),
            artifact_sha256: artifact_sha256.into(),
            reference_implementation_sha256: reference_implementation_sha256.into(),
            tested_configuration_sha256: tested_configuration_sha256.into(),
            workload,
            reference_output,
            tested_output,
        };
        super::validate_parity_artifact(&artifact)?;
        Ok(artifact)
    }
}

impl std::fmt::Debug for ModelParityArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelParityArtifact")
            .field("binding", &self.binding)
            .field("case", &"redacted-sha256")
            .field("artifact", &"redacted-sha256")
            .field("reference_implementation", &"sha256")
            .field("tested_configuration", &"sha256")
            .field(
                "output_parity",
                &(self.reference_output == self.tested_output),
            )
            .finish()
    }
}

/// Self-contained aggregate evidence for one model, implementation, runtime,
/// device, and named hardware environment.
///
/// Construction and verification do not upload, persist, or log this value.
/// The SHA-256 detects mutation; authenticity still requires the digest to be
/// pinned by a signed release, attestation, or another caller-owned trust root.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HardwareEvidenceBundle {
    pub schema: String,
    pub binding: HardwareEvidenceBinding,
    pub storage_reports: Vec<StorageBenchmarkReport>,
    pub storage_comparison: StorageBenchmarkComparison,
    pub tuning_evidence: TuningProfileEvidence,
    pub tuning_policy: TuningProfilePolicy,
    pub tuning_decision: TuningProfileDecision,
    pub parity_artifacts: Vec<ModelParityArtifact>,
    pub sha256: String,
}

impl HardwareEvidenceBundle {
    pub const SCHEMA: &'static str = "a3s.power.hardware-evidence-bundle.v1";

    pub fn build(
        binding: HardwareEvidenceBinding,
        storage_reports: Vec<StorageBenchmarkReport>,
        tuning_evidence: TuningProfileEvidence,
        tuning_policy: TuningProfilePolicy,
        parity_artifacts: Vec<ModelParityArtifact>,
    ) -> Result<Self> {
        super::build_bundle(
            binding,
            storage_reports,
            tuning_evidence,
            tuning_policy,
            parity_artifacts,
        )
    }

    /// Replays every derivation and verifies the embedded canonical digest.
    pub fn verify(&self) -> Result<()> {
        super::verify_bundle(self)
    }

    /// Verifies the bundle and a digest supplied by a caller-owned trust root.
    pub fn verify_pinned(&self, expected_sha256: &str) -> Result<()> {
        super::verify_pinned_bundle(self, expected_sha256)
    }
}

impl std::fmt::Debug for HardwareEvidenceBundle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HardwareEvidenceBundle")
            .field("schema", &self.schema)
            .field("binding", &self.binding)
            .field("storage_report_count", &self.storage_reports.len())
            .field(
                "tuning_candidate_count",
                &self.tuning_evidence.candidates.len(),
            )
            .field("parity_artifact_count", &self.parity_artifacts.len())
            .field("sha256", &self.sha256)
            .finish()
    }
}
