use serde::{Deserialize, Serialize};

use crate::admission::AdmissionSnapshot;
use crate::error::Result;

use super::super::{
    ConfidentialGpuBinding, ExecutionBatchLifecycleEvidence, ExecutionDigest,
    ModelSessionPoolSnapshot, ResidentTensorSnapshot, RuntimeDeviceIdentity, ShapeProfileBinding,
    ShapeProfileExecutionEvidence, TensorBatchBenchmarkReport,
};

/// Hardware and confidentiality class covered by one release capture.
///
/// The class intentionally says nothing about a model family, container format,
/// tokenizer, decoder, or operator set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReleasePlatform {
    Cpu,
    Cuda,
    Metal,
    ConfidentialGpu,
}

/// Immutable source and workload identities shared by every required capture.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseRevisionBinding {
    pub power_version: String,
    pub power_commit: String,
    pub weights_sha256: String,
    pub graph_source_sha256: String,
    pub graph_declaration_sha256: String,
    pub shape_profile_declaration_sha256: String,
    pub tee_policy_sha256: String,
}

impl ReleaseRevisionBinding {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        power_version: impl Into<String>,
        power_commit: impl Into<String>,
        weights_sha256: impl Into<String>,
        graph_source_sha256: impl Into<String>,
        graph_declaration_sha256: impl Into<String>,
        shape_profile_declaration_sha256: impl Into<String>,
        tee_policy_sha256: impl Into<String>,
    ) -> Result<Self> {
        let binding = Self {
            power_version: power_version.into(),
            power_commit: power_commit.into(),
            weights_sha256: weights_sha256.into(),
            graph_source_sha256: graph_source_sha256.into(),
            graph_declaration_sha256: graph_declaration_sha256.into(),
            shape_profile_declaration_sha256: shape_profile_declaration_sha256.into(),
            tee_policy_sha256: tee_policy_sha256.into(),
        };
        super::validation::validate_revision_binding(&binding)?;
        Ok(binding)
    }
}

impl std::fmt::Debug for ReleaseRevisionBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReleaseRevisionBinding")
            .field("power_version", &self.power_version)
            .field("power_commit", &"revision")
            .field("weights", &"sha256")
            .field("graph_source", &"sha256")
            .field("graph_declaration", &"sha256")
            .field("shape_profile_declaration", &"sha256")
            .field("tee_policy", &"sha256")
            .finish()
    }
}

/// Explicit release policy. Verification accepts exactly these platform
/// classes; missing, duplicate, or undeclared captures fail closed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseEvidencePolicy {
    pub schema: String,
    pub revision: ReleaseRevisionBinding,
    pub required_platforms: Vec<ReleasePlatform>,
}

impl ReleaseEvidencePolicy {
    pub const SCHEMA: &'static str = "a3s.power.release-evidence-policy.v1";

    pub fn new(
        revision: ReleaseRevisionBinding,
        required_platforms: Vec<ReleasePlatform>,
    ) -> Result<Self> {
        let policy = Self {
            schema: Self::SCHEMA.to_string(),
            revision,
            required_platforms,
        };
        super::validation::validate_policy(&policy)?;
        Ok(policy)
    }

    /// Production v1 coverage: ordinary CPU, CUDA, and Metal plus a separately
    /// attested confidential-GPU capture.
    pub fn strict_v1(revision: ReleaseRevisionBinding) -> Result<Self> {
        Self::new(
            revision,
            vec![
                ReleasePlatform::Cpu,
                ReleasePlatform::Cuda,
                ReleasePlatform::Metal,
                ReleasePlatform::ConfidentialGpu,
            ],
        )
    }

    pub fn policy_sha256(&self) -> Result<String> {
        super::validation::validate_policy(self)?;
        super::digest::policy_sha256(self)
    }
}

/// How a peak-used-memory observation was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum PeakMemoryMethod {
    /// Process-global allocator accounting, including an atomic live-byte peak.
    HostAllocator,
    /// Sampled process resident set. The interval is part of the evidence.
    ProcessResidentSet { sample_interval_nanos: u64 },
    /// Sampled accelerator pool usage derived from total minus available bytes.
    DevicePoolAvailability { sample_interval_nanos: u64 },
}

/// One bounded, content-free peak-used-memory observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BoundedMemoryEvidence {
    pub method: PeakMemoryMethod,
    pub baseline_used_bytes: u64,
    pub peak_used_bytes: u64,
    pub final_used_bytes: u64,
    pub sample_count: u64,
}

impl BoundedMemoryEvidence {
    pub fn host_allocator(
        baseline_used_bytes: u64,
        peak_used_bytes: u64,
        final_used_bytes: u64,
    ) -> Result<Self> {
        let evidence = Self {
            method: PeakMemoryMethod::HostAllocator,
            baseline_used_bytes,
            peak_used_bytes,
            final_used_bytes,
            sample_count: 1,
        };
        super::validation::validate_memory_observation(&evidence, "host")?;
        Ok(evidence)
    }

    pub fn sampled(
        method: PeakMemoryMethod,
        baseline_used_bytes: u64,
        peak_used_bytes: u64,
        final_used_bytes: u64,
        sample_count: u64,
    ) -> Result<Self> {
        let evidence = Self {
            method,
            baseline_used_bytes,
            peak_used_bytes,
            final_used_bytes,
            sample_count,
        };
        super::validation::validate_memory_observation(&evidence, "sampled")?;
        Ok(evidence)
    }

    pub fn additional_peak_bytes(&self) -> u64 {
        self.peak_used_bytes
            .saturating_sub(self.baseline_used_bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PeakMemoryEvidence {
    pub host: BoundedMemoryEvidence,
    pub device: Option<BoundedMemoryEvidence>,
}

/// Active-work cancellation plus post-cancellation admission and resident-state
/// cleanup. All fields are counters or digests; request content is absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CancellationContractEvidence {
    pub lifecycle: ExecutionBatchLifecycleEvidence,
    pub admission_after: AdmissionSnapshot,
    pub resident_after: ResidentTensorSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct QueueExpiryEvidence {
    pub before: AdmissionSnapshot,
    pub after: AdmissionSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplicaRecoveryEvidence {
    pub before: ModelSessionPoolSnapshot,
    pub retired: ModelSessionPoolSnapshot,
    pub recovered: ModelSessionPoolSnapshot,
}

/// Explicit dynamic-fallback selection with exact typed output parity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExactFallbackEvidence {
    pub selection: ShapeProfileExecutionEvidence,
    pub reference_output: ExecutionDigest,
    pub fallback_output: ExecutionDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseContractEvidence {
    pub peak_memory: PeakMemoryEvidence,
    pub cancellation: CancellationContractEvidence,
    pub queue_expiry: QueueExpiryEvidence,
    pub replica_recovery: ReplicaRecoveryEvidence,
    pub exact_fallback: ExactFallbackEvidence,
}

/// Digest-only projection of an already verified confidential-GPU binding.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConfidentialReleaseBinding {
    pub(super) verified_claims_sha256: String,
    pub(super) accelerator_declaration_sha256: String,
    pub(super) weights_sha256: String,
    pub(super) execution_policy_sha256: String,
    pub(super) runtime_device: RuntimeDeviceIdentity,
    pub(super) device_mesh_sha256: Option<String>,
}

impl ConfidentialReleaseBinding {
    fn from_verified(binding: &ConfidentialGpuBinding) -> Self {
        Self {
            verified_claims_sha256: binding.claims_sha256().to_string(),
            accelerator_declaration_sha256: binding.declaration_sha256().to_string(),
            weights_sha256: binding.weights_sha256().to_string(),
            execution_policy_sha256: binding.execution_policy_sha256().to_string(),
            runtime_device: binding.runtime_device(),
            device_mesh_sha256: binding.device_mesh_sha256().map(str::to_string),
        }
    }

    pub fn verified_claims_sha256(&self) -> &str {
        &self.verified_claims_sha256
    }

    pub fn accelerator_declaration_sha256(&self) -> &str {
        &self.accelerator_declaration_sha256
    }

    pub fn weights_sha256(&self) -> &str {
        &self.weights_sha256
    }

    pub fn execution_policy_sha256(&self) -> &str {
        &self.execution_policy_sha256
    }

    pub fn runtime_device(&self) -> RuntimeDeviceIdentity {
        self.runtime_device
    }

    pub fn device_mesh_sha256(&self) -> Option<&str> {
        self.device_mesh_sha256.as_deref()
    }
}

impl std::fmt::Debug for ConfidentialReleaseBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfidentialReleaseBinding")
            .field("verified_claims", &"sha256")
            .field("accelerator_declaration", &"sha256")
            .field("weights", &"sha256")
            .field("execution_policy", &"sha256")
            .field("runtime_device", &self.runtime_device)
            .field(
                "device_mesh",
                &self.device_mesh_sha256.as_ref().map(|_| "sha256"),
            )
            .finish()
    }
}

/// Local execution or an externally authenticated confidential-GPU binding.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum ReleaseCaptureSecurity {
    Local,
    ConfidentialGpu { binding: ConfidentialReleaseBinding },
}

impl ReleaseCaptureSecurity {
    /// Projects only digest-bound fields from a binding that Power created
    /// after strict attestation-claim validation.
    pub fn from_verified_confidential_gpu(binding: &ConfidentialGpuBinding) -> Result<Self> {
        let security = Self::ConfidentialGpu {
            binding: ConfidentialReleaseBinding::from_verified(binding),
        };
        super::validation::validate_security(&security)?;
        Ok(security)
    }
}

/// Self-contained evidence for one platform class on one named system.
///
/// The tensor benchmark supplies raw scalar/batch samples and exact output
/// parity. The shape binding and runtime contracts add graph, TEE, memory, and
/// failure-lifecycle coverage without interpreting the workload architecture.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseCapture {
    pub schema: String,
    pub security: ReleaseCaptureSecurity,
    pub shape_binding: ShapeProfileBinding,
    pub tensor_batch: TensorBatchBenchmarkReport,
    pub contracts: ReleaseContractEvidence,
    pub sha256: String,
}

impl ReleaseCapture {
    pub const SCHEMA: &'static str = "a3s.power.release-capture.v1";

    pub fn build(
        security: ReleaseCaptureSecurity,
        shape_binding: ShapeProfileBinding,
        tensor_batch: TensorBatchBenchmarkReport,
        contracts: ReleaseContractEvidence,
    ) -> Result<Self> {
        let mut capture = Self {
            schema: Self::SCHEMA.to_string(),
            security,
            shape_binding,
            tensor_batch,
            contracts,
            sha256: String::new(),
        };
        super::validation::validate_capture_structure(&capture)?;
        capture.sha256 = super::digest::capture_sha256(&capture)?;
        capture.verify()?;
        Ok(capture)
    }

    pub fn platform(&self) -> Result<ReleasePlatform> {
        super::validation::capture_platform(self)
    }

    pub fn verify(&self) -> Result<()> {
        super::validation::verify_capture(self)
    }
}

impl std::fmt::Debug for ReleaseCapture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReleaseCapture")
            .field("schema", &self.schema)
            .field("security", &self.security)
            .field("shape_binding", &self.shape_binding)
            .field("tensor_batch", &self.tensor_batch)
            .field("contracts", &"verified")
            .field("sha256", &self.sha256)
            .finish()
    }
}

/// Canonical fail-closed release artifact covering exactly one policy.
///
/// Its digest detects mutation. Authenticity still requires the digest to be
/// pinned by a signed release, attestation, or another caller-owned trust root.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseEvidenceBundle {
    pub schema: String,
    pub policy: ReleaseEvidencePolicy,
    pub captures: Vec<ReleaseCapture>,
    pub sha256: String,
}

impl ReleaseEvidenceBundle {
    pub const SCHEMA: &'static str = "a3s.power.release-evidence-bundle.v1";

    pub fn build(policy: ReleaseEvidencePolicy, captures: Vec<ReleaseCapture>) -> Result<Self> {
        super::build_bundle(policy, captures)
    }

    pub fn verify(&self) -> Result<()> {
        super::validation::verify_bundle(self)
    }

    pub fn verify_pinned(&self, expected_sha256: &str) -> Result<()> {
        super::validation::verify_pinned_bundle(self, expected_sha256)
    }
}

impl std::fmt::Debug for ReleaseEvidenceBundle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReleaseEvidenceBundle")
            .field("schema", &self.schema)
            .field("policy", &self.policy)
            .field("capture_count", &self.captures.len())
            .field("sha256", &self.sha256)
            .finish()
    }
}
