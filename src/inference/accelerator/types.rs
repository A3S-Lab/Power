use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};
use crate::inference::{
    ExecutionDigest, ExecutionPermit, RuntimeDeviceIdentity, RuntimeDeviceKind, WeightKey,
};

use super::mesh::AcceleratorDeviceMeshDeclaration;
use super::mesh_execution::AcceleratorMeshExecutionSummary;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorFallbackMode {
    #[default]
    Deny,
    AllowExact,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorFallbackTarget {
    #[default]
    Cpu,
    RuntimeDevice,
}

impl AcceleratorFallbackTarget {
    pub(super) fn identity(self, runtime_device: RuntimeDeviceIdentity) -> RuntimeDeviceIdentity {
        match self {
            Self::Cpu => RuntimeDeviceIdentity {
                kind: RuntimeDeviceKind::Cpu,
                ordinal: None,
            },
            Self::RuntimeDevice => runtime_device,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorSecurityRequirement {
    #[default]
    Local,
    ConfidentialGpu,
}

/// Model-owned fused implementation and exact fallback identities plus the
/// canonical order of active residency-plan groups used by one fused batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorFusedBatchSpec {
    pub fused_kernel_sha256: String,
    pub exact_fallback_sha256: String,
    pub residency_group_ids: Vec<String>,
    #[serde(default)]
    pub fallback_mode: AcceleratorFallbackMode,
    #[serde(default)]
    pub fallback_target: AcceleratorFallbackTarget,
    #[serde(default)]
    pub security: AcceleratorSecurityRequirement,
}

impl AcceleratorFusedBatchSpec {
    pub fn new(
        fused_kernel_sha256: impl Into<String>,
        exact_fallback_sha256: impl Into<String>,
        residency_group_ids: Vec<String>,
    ) -> Self {
        Self {
            fused_kernel_sha256: fused_kernel_sha256.into(),
            exact_fallback_sha256: exact_fallback_sha256.into(),
            residency_group_ids,
            fallback_mode: AcceleratorFallbackMode::Deny,
            fallback_target: AcceleratorFallbackTarget::Cpu,
            security: AcceleratorSecurityRequirement::Local,
        }
    }

    pub fn with_fallback_mode(mut self, mode: AcceleratorFallbackMode) -> Self {
        self.fallback_mode = mode;
        self
    }

    pub fn with_fallback_target(mut self, target: AcceleratorFallbackTarget) -> Self {
        self.fallback_target = target;
        self
    }

    pub fn with_security(mut self, security: AcceleratorSecurityRequirement) -> Self {
        self.security = security;
        self
    }

    pub(super) fn validate(&self, max_groups: usize, max_name_bytes: usize) -> Result<()> {
        validate_sha256(&self.fused_kernel_sha256, "fused kernel")?;
        validate_sha256(&self.exact_fallback_sha256, "exact fallback")?;
        if self.fused_kernel_sha256 == self.exact_fallback_sha256 {
            return Err(PowerError::InvalidRequest(
                "fused kernel and exact fallback identities must be distinct".to_string(),
            ));
        }
        if self.residency_group_ids.is_empty() || self.residency_group_ids.len() > max_groups {
            return Err(PowerError::InvalidRequest(format!(
                "accelerator fused batch must contain between 1 and {max_groups} residency groups"
            )));
        }
        let mut ids = BTreeSet::new();
        for id in &self.residency_group_ids {
            validate_identifier(id, max_name_bytes, "residency group")?;
            if !ids.insert(id) {
                return Err(PowerError::InvalidRequest(format!(
                    "accelerator fused batch declares residency group '{id}' more than once"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorResidencyGroup {
    pub canonical_index: usize,
    pub residency_group_id: String,
    pub bytes: u64,
    pub weights: Vec<WeightKey>,
}

/// Digest-bound declaration for weights that must already be wholly resident
/// on one resolved accelerator before a model-owned fused operation starts.
///
/// The declaration can reveal model topology and must not be logged or
/// persisted automatically. Receipts contain only its digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorResidencyDeclaration {
    pub schema: String,
    pub weights_sha256: String,
    pub active_plan_sha256: String,
    pub runtime_device: RuntimeDeviceIdentity,
    pub fused_kernel_sha256: String,
    pub exact_fallback_sha256: String,
    pub fallback_mode: AcceleratorFallbackMode,
    pub fallback_target: AcceleratorFallbackTarget,
    pub security: AcceleratorSecurityRequirement,
    pub max_input_bytes: usize,
    pub max_tensor_elements: usize,
    pub groups: Vec<AcceleratorResidencyGroup>,
    pub total_weights: usize,
    pub total_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_mesh: Option<AcceleratorDeviceMeshDeclaration>,
    pub declaration_sha256: String,
    pub execution_policy_sha256: String,
}

impl AcceleratorResidencyDeclaration {
    pub const SCHEMA: &'static str = "a3s.power.accelerator-residency-declaration.v1";
    pub const MESH_SCHEMA: &'static str = "a3s.power.accelerator-residency-declaration.v2";

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build(
        weights_sha256: String,
        active_plan_sha256: String,
        runtime_device: RuntimeDeviceIdentity,
        spec: &AcceleratorFusedBatchSpec,
        max_input_bytes: usize,
        max_tensor_elements: usize,
        groups: Vec<AcceleratorResidencyGroup>,
        total_weights: usize,
        total_bytes: u64,
        device_mesh: Option<AcceleratorDeviceMeshDeclaration>,
    ) -> Result<Self> {
        let schema = if device_mesh.is_some() {
            Self::MESH_SCHEMA
        } else {
            Self::SCHEMA
        };
        let mut declaration = Self {
            schema: schema.to_string(),
            weights_sha256,
            active_plan_sha256,
            runtime_device,
            fused_kernel_sha256: spec.fused_kernel_sha256.clone(),
            exact_fallback_sha256: spec.exact_fallback_sha256.clone(),
            fallback_mode: spec.fallback_mode,
            fallback_target: spec.fallback_target,
            security: spec.security,
            max_input_bytes,
            max_tensor_elements,
            groups,
            total_weights,
            total_bytes,
            device_mesh,
            declaration_sha256: String::new(),
            execution_policy_sha256: String::new(),
        };
        let digest = declaration.recompute_sha256()?;
        declaration.declaration_sha256 = digest.clone();
        declaration.execution_policy_sha256 = digest;
        declaration.validate()?;
        Ok(declaration)
    }

    #[cfg(all(test, feature = "server"))]
    pub(crate) fn confidential_release_fixture(
        weights_sha256: String,
        runtime_device: RuntimeDeviceIdentity,
    ) -> Self {
        let group_id = "release-fixture".to_string();
        let spec = AcceleratorFusedBatchSpec::new(
            "11".repeat(32),
            "22".repeat(32),
            vec![group_id.clone()],
        )
        .with_fallback_mode(AcceleratorFallbackMode::AllowExact)
        .with_security(AcceleratorSecurityRequirement::ConfidentialGpu);
        Self::build(
            weights_sha256,
            "33".repeat(32),
            runtime_device,
            &spec,
            1_024,
            1_024,
            vec![AcceleratorResidencyGroup {
                canonical_index: 0,
                residency_group_id: group_id,
                bytes: 4,
                weights: vec![WeightKey::new(0, "release.weight")],
            }],
            1,
            4,
            None,
        )
        .expect("confidential release fixture declaration should be valid")
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA && self.schema != Self::MESH_SCHEMA {
            return Err(PowerError::InvalidFormat(
                "accelerator residency declaration has an unsupported schema".to_string(),
            ));
        }
        self.runtime_device.validate()?;
        match (&self.schema[..], &self.device_mesh) {
            (Self::SCHEMA, None) => {}
            (Self::MESH_SCHEMA, Some(mesh)) => {
                mesh.validate()?;
                mesh.validate_security(self.security)?;
                if mesh.primary_runtime_device() != self.runtime_device {
                    return Err(PowerError::InvalidFormat(
                        "accelerator device mesh primary does not match the runtime device"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(PowerError::InvalidFormat(
                    "accelerator residency schema and device mesh shape do not match".to_string(),
                ))
            }
        }
        validate_sha256(&self.weights_sha256, "weight collection")?;
        validate_sha256(&self.active_plan_sha256, "active residency plan")?;
        validate_sha256(&self.fused_kernel_sha256, "fused kernel")?;
        validate_sha256(&self.exact_fallback_sha256, "exact fallback")?;
        validate_sha256(&self.declaration_sha256, "accelerator declaration")?;
        validate_sha256(
            &self.execution_policy_sha256,
            "accelerator execution policy",
        )?;
        if self.fused_kernel_sha256 == self.exact_fallback_sha256
            || self.max_input_bytes == 0
            || self.max_tensor_elements == 0
            || self.groups.is_empty()
        {
            return Err(PowerError::InvalidFormat(
                "accelerator residency declaration contains invalid execution bounds or identities"
                    .to_string(),
            ));
        }

        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut total_weights = 0_usize;
        let mut total_bytes = 0_u64;
        for (index, group) in self.groups.iter().enumerate() {
            if group.canonical_index != index
                || group.residency_group_id.is_empty()
                || group.weights.is_empty()
                || group.bytes == 0
                || !ids.insert(&group.residency_group_id)
            {
                return Err(PowerError::InvalidFormat(
                    "accelerator residency declaration contains an invalid group".to_string(),
                ));
            }
            for key in &group.weights {
                if !keys.insert(key) {
                    return Err(PowerError::InvalidFormat(
                        "accelerator residency declaration references a weight more than once"
                            .to_string(),
                    ));
                }
            }
            total_weights = total_weights
                .checked_add(group.weights.len())
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "accelerator declaration weight count overflowed".to_string(),
                    )
                })?;
            total_bytes = total_bytes.checked_add(group.bytes).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "accelerator declaration resident bytes overflowed".to_string(),
                )
            })?;
        }
        if total_weights != self.total_weights || total_bytes != self.total_bytes {
            return Err(PowerError::InvalidFormat(
                "accelerator residency declaration totals are inconsistent".to_string(),
            ));
        }
        let recomputed = self.recompute_sha256()?;
        if self.declaration_sha256 != recomputed || self.execution_policy_sha256 != recomputed {
            return Err(PowerError::InvalidFormat(
                "accelerator residency declaration digest does not match its canonical payload"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn recompute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Payload<'a> {
            schema: &'a str,
            weights_sha256: &'a str,
            active_plan_sha256: &'a str,
            runtime_device: RuntimeDeviceIdentity,
            fused_kernel_sha256: &'a str,
            exact_fallback_sha256: &'a str,
            fallback_mode: AcceleratorFallbackMode,
            fallback_target: AcceleratorFallbackTarget,
            security: AcceleratorSecurityRequirement,
            max_input_bytes: usize,
            max_tensor_elements: usize,
            groups: &'a [AcceleratorResidencyGroup],
            total_weights: usize,
            total_bytes: u64,
            #[serde(skip_serializing_if = "Option::is_none")]
            device_mesh: Option<&'a AcceleratorDeviceMeshDeclaration>,
        }
        let payload = Payload {
            schema: &self.schema,
            weights_sha256: &self.weights_sha256,
            active_plan_sha256: &self.active_plan_sha256,
            runtime_device: self.runtime_device,
            fused_kernel_sha256: &self.fused_kernel_sha256,
            exact_fallback_sha256: &self.exact_fallback_sha256,
            fallback_mode: self.fallback_mode,
            fallback_target: self.fallback_target,
            security: self.security,
            max_input_bytes: self.max_input_bytes,
            max_tensor_elements: self.max_tensor_elements,
            groups: &self.groups,
            total_weights: self.total_weights,
            total_bytes: self.total_bytes,
            device_mesh: self.device_mesh.as_ref(),
        };
        let encoded = serde_json::to_vec(&payload)?;
        let mut hasher = Sha256::new();
        if self.schema == Self::MESH_SCHEMA {
            hasher.update(b"a3s-power-accelerator-residency-declaration-v2\0");
        } else {
            hasher.update(b"a3s-power-accelerator-residency-declaration-v1\0");
        }
        hasher.update(encoded);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcceleratorFallbackReason {
    PlanChanged,
    ResidencyUnavailable,
    KernelUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum AcceleratorExecutionPath {
    Accelerator,
    Fallback { reason: AcceleratorFallbackReason },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorExecutionEvidence {
    pub schema: String,
    pub declaration_sha256: String,
    pub weights_sha256: String,
    pub runtime_device: RuntimeDeviceIdentity,
    pub execution_device: RuntimeDeviceIdentity,
    pub path: AcceleratorExecutionPath,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_target: Option<AcceleratorFallbackTarget>,
    pub implementation_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidential_claims_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_mesh_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub execution_devices: Vec<RuntimeDeviceIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_transfers_sha256: Option<String>,
    pub input_sha256: String,
    pub output_sha256: String,
}

impl AcceleratorExecutionEvidence {
    pub const SCHEMA: &'static str = "a3s.power.accelerator-execution-evidence.v1";
    pub const MESH_SCHEMA: &'static str = "a3s.power.accelerator-execution-evidence.v2";

    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA && self.schema != Self::MESH_SCHEMA {
            return Err(PowerError::InvalidFormat(
                "accelerator execution evidence has an unsupported schema".to_string(),
            ));
        }
        self.runtime_device.validate()?;
        self.execution_device.validate()?;
        match (&self.path, self.fallback_target) {
            (AcceleratorExecutionPath::Accelerator, None)
                if self.execution_device == self.runtime_device => {}
            (AcceleratorExecutionPath::Fallback { .. }, Some(target))
                if self.execution_device == target.identity(self.runtime_device) => {}
            _ => {
                return Err(PowerError::InvalidFormat(
                    "accelerator evidence execution device does not match its selected path"
                        .to_string(),
                ))
            }
        }
        validate_sha256(&self.declaration_sha256, "accelerator declaration")?;
        validate_sha256(&self.weights_sha256, "weight collection")?;
        validate_sha256(&self.implementation_sha256, "accelerator implementation")?;
        validate_sha256(&self.input_sha256, "accelerator input")?;
        validate_sha256(&self.output_sha256, "accelerator output")?;
        if let Some(digest) = &self.confidential_claims_sha256 {
            validate_sha256(digest, "confidential GPU claims")?;
        }
        match (&self.schema[..], &self.device_mesh_sha256) {
            (Self::SCHEMA, None)
                if self.execution_devices.is_empty() && self.peer_transfers_sha256.is_none() => {}
            (Self::MESH_SCHEMA, Some(mesh_sha256)) if !self.execution_devices.is_empty() => {
                validate_sha256(mesh_sha256, "accelerator device mesh")?;
                validate_sha256(
                    self.peer_transfers_sha256.as_deref().ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "mesh execution evidence is missing its peer-transfer digest"
                                .to_string(),
                        )
                    })?,
                    "accelerator peer transfers",
                )?;
                let mut canonical = self.execution_devices.clone();
                for device in &canonical {
                    device.validate()?;
                }
                canonical.sort();
                canonical.dedup();
                if canonical != self.execution_devices
                    || !canonical.contains(&self.execution_device)
                {
                    return Err(PowerError::InvalidFormat(
                        "mesh execution devices are not canonical or omit the actual output device"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(PowerError::InvalidFormat(
                    "accelerator evidence schema and mesh fields do not match".to_string(),
                ))
            }
        }
        Ok(())
    }
}

pub struct AcceleratorExecutionCompletion {
    pub(super) declaration_sha256: String,
    pub(super) weights_sha256: String,
    pub(super) runtime_device: RuntimeDeviceIdentity,
    pub(super) execution_device: RuntimeDeviceIdentity,
    pub(super) path: AcceleratorExecutionPath,
    pub(super) fallback_target: Option<AcceleratorFallbackTarget>,
    pub(super) implementation_sha256: String,
    pub(super) confidential_claims_sha256: Option<String>,
    pub(super) mesh: Option<AcceleratorMeshExecutionSummary>,
    pub(super) _permit: ExecutionPermit,
}

impl AcceleratorExecutionCompletion {
    pub fn complete(
        self,
        input: &ExecutionDigest,
        output: &ExecutionDigest,
    ) -> Result<AcceleratorExecutionEvidence> {
        validate_sha256(&input.sha256, "execution input")?;
        validate_sha256(&output.sha256, "execution output")?;
        let evidence = AcceleratorExecutionEvidence {
            schema: if self.mesh.is_some() {
                AcceleratorExecutionEvidence::MESH_SCHEMA.to_string()
            } else {
                AcceleratorExecutionEvidence::SCHEMA.to_string()
            },
            declaration_sha256: self.declaration_sha256,
            weights_sha256: self.weights_sha256,
            runtime_device: self.runtime_device,
            execution_device: self.execution_device,
            path: self.path,
            fallback_target: self.fallback_target,
            implementation_sha256: self.implementation_sha256,
            confidential_claims_sha256: self.confidential_claims_sha256,
            device_mesh_sha256: self.mesh.as_ref().map(|mesh| mesh.mesh_sha256.clone()),
            execution_devices: self
                .mesh
                .as_ref()
                .map(|mesh| mesh.execution_devices.clone())
                .unwrap_or_default(),
            peer_transfers_sha256: self
                .mesh
                .as_ref()
                .map(|mesh| mesh.peer_transfers_sha256.clone()),
            input_sha256: input.sha256.clone(),
            output_sha256: output.sha256.clone(),
        };
        evidence.validate()?;
        Ok(evidence)
    }
}

impl std::fmt::Debug for AcceleratorExecutionCompletion {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorExecutionCompletion")
            .field("declaration_sha256", &self.declaration_sha256)
            .field("runtime_device", &self.runtime_device)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidRequest(format!(
            "{label} SHA-256 must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

pub(super) fn validate_identifier(value: &str, max_bytes: usize, label: &str) -> Result<()> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(PowerError::InvalidRequest(format!(
            "{label} identity is empty, oversized, or contains control characters"
        )));
    }
    Ok(())
}
