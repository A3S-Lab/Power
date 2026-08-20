use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;

use crate::admission::AdmissionSnapshot;
use crate::error::{PowerError, Result};

use super::super::sealed_state::decode_sha256;
use super::super::{InferenceLimits, ModelIdentity, RuntimeDeviceIdentity, RuntimeDeviceKind};

const MAX_REPLICAS_PER_SESSION: usize = 256;

const fn default_max_replicas_per_session() -> usize {
    1
}

/// Exact model and model-owned execution identity for one shareable session.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionBinding {
    pub model: ModelIdentity,
    pub execution_sha256: String,
}

impl ModelSessionBinding {
    pub fn new(model: ModelIdentity, execution_sha256: impl Into<String>) -> Self {
        Self {
            model,
            execution_sha256: execution_sha256.into(),
        }
    }

    fn validate(&self, limits: &InferenceLimits) -> Result<()> {
        for (label, value) in [
            ("model session family", self.model.family.as_str()),
            ("model session revision", self.model.revision.as_str()),
        ] {
            if value.is_empty()
                || value.len() > limits.max_graph_name_bytes
                || value.chars().any(char::is_control)
            {
                return Err(PowerError::InvalidRequest(format!(
                    "{label} must be non-empty, control-free, and at most {} bytes",
                    limits.max_graph_name_bytes
                )));
            }
        }
        decode_sha256(&self.model.weights_sha256, "model session weights")?;
        decode_sha256(&self.execution_sha256, "model session execution")?;
        Ok(())
    }

    pub(super) fn key_sha256(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"a3s-power-model-session-key-v1\0");
        update_text(&mut digest, &self.model.family);
        update_text(&mut digest, &self.model.revision);
        update_text(&mut digest, &self.model.weights_sha256);
        update_text(&mut digest, &self.execution_sha256);
        format!("{:x}", digest.finalize())
    }
}

impl std::fmt::Debug for ModelSessionBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelSessionBinding")
            .field("model", &"bound")
            .field("execution", &"sha256")
            .finish()
    }
}

/// Per-replica resource declaration for one exact model session identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionSpec {
    pub(super) binding: ModelSessionBinding,
    pub(super) limits: InferenceLimits,
    pub(super) resident_bytes: u64,
}

impl ModelSessionSpec {
    pub fn new(
        binding: ModelSessionBinding,
        limits: InferenceLimits,
        resident_bytes: u64,
    ) -> Result<Self> {
        let spec = Self {
            binding,
            limits,
            resident_bytes,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn binding(&self) -> &ModelSessionBinding {
        &self.binding
    }

    pub fn limits(&self) -> &InferenceLimits {
        &self.limits
    }

    /// Resident bytes reserved for each independently mutable replica.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Canonical declaration digest, including the exact resolved device and
    /// every resource limit used to construct the shared runtime.
    pub fn declaration_sha256(&self, device: RuntimeDeviceIdentity) -> Result<String> {
        self.validate()?;
        device.validate()?;
        let mut digest = Sha256::new();
        digest.update(b"a3s-power-model-session-declaration-v1\0");
        update_text(&mut digest, &self.binding.key_sha256());
        digest.update(self.resident_bytes.to_le_bytes());
        digest.update([match device.kind {
            RuntimeDeviceKind::Cpu => 0,
            RuntimeDeviceKind::Cuda => 1,
            RuntimeDeviceKind::Metal => 2,
        }]);
        update_optional_usize(&mut digest, device.ordinal)?;
        update_limits(&mut digest, &self.limits)?;
        Ok(format!("{:x}", digest.finalize()))
    }

    pub(super) fn validate(&self) -> Result<()> {
        self.limits.validate()?;
        self.binding.validate(&self.limits)?;
        if self.resident_bytes == 0 || self.resident_bytes > self.limits.max_model_bytes {
            return Err(PowerError::InvalidRequest(format!(
                "model session resident bytes must be between 1 and {}",
                self.limits.max_model_bytes
            )));
        }
        Ok(())
    }
}

/// Hard bounds shared by every model entry on one resolved device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelSessionPoolPolicy {
    pub max_sessions: usize,
    pub max_resident_bytes: u64,
    pub max_concurrent_device_requests: usize,
    pub max_queued_device_requests: usize,
    #[serde(default = "default_max_replicas_per_session")]
    pub max_replicas_per_session: usize,
}

impl ModelSessionPoolPolicy {
    pub fn new(
        max_sessions: usize,
        max_resident_bytes: u64,
        max_concurrent_device_requests: usize,
        max_queued_device_requests: usize,
    ) -> Result<Self> {
        let policy = Self {
            max_sessions,
            max_resident_bytes,
            max_concurrent_device_requests,
            max_queued_device_requests,
            max_replicas_per_session: default_max_replicas_per_session(),
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Enables exclusive, independently initialized replicas for each exact
    /// session identity. One remains the backward-compatible default.
    pub fn with_max_replicas_per_session(mut self, maximum: usize) -> Result<Self> {
        self.max_replicas_per_session = maximum;
        self.validate()?;
        Ok(self)
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.max_sessions == 0
            || self.max_resident_bytes == 0
            || self.max_concurrent_device_requests == 0
            || self.max_replicas_per_session == 0
        {
            return Err(PowerError::Config(
                "model session pool count, replica count, resident bytes, and device concurrency must be greater than zero"
                    .to_string(),
            ));
        }
        if self.max_sessions > Semaphore::MAX_PERMITS
            || self.max_concurrent_device_requests > Semaphore::MAX_PERMITS
            || self.max_queued_device_requests > Semaphore::MAX_PERMITS
            || self.max_replicas_per_session > MAX_REPLICAS_PER_SESSION
        {
            return Err(PowerError::Config(format!(
                "model session pool count and admission bounds cannot exceed {}; replicas per session cannot exceed {MAX_REPLICAS_PER_SESSION}",
                Semaphore::MAX_PERMITS,
            )));
        }
        Ok(())
    }
}

/// Aggregate, content-free state for one device-bound session pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSessionPoolSnapshot {
    pub device: RuntimeDeviceIdentity,
    pub maximum_sessions: usize,
    pub maximum_resident_bytes: u64,
    pub registered_sessions: usize,
    pub ready_sessions: usize,
    pub maximum_replicas_per_session: usize,
    pub reserved_replicas: usize,
    pub ready_replicas: usize,
    pub leased_replicas: usize,
    pub waiting_replica_requests: usize,
    pub expired_replica_requests: u64,
    pub replicas_pending_reconstruction: usize,
    pub replica_retirements: u64,
    pub replica_reconstructions: u64,
    pub reserved_bytes: u64,
    pub device_admission: AdmissionSnapshot,
}

pub(super) fn replica_declaration_sha256(
    session_declaration_sha256: &str,
    maximum_replicas: usize,
    reserved_bytes: u64,
) -> Result<String> {
    let maximum_replicas = u64::try_from(maximum_replicas).map_err(|_| {
        PowerError::InvalidRequest("model session replica count cannot be represented".to_string())
    })?;
    let mut digest = Sha256::new();
    digest.update(b"a3s-power-model-session-replica-declaration-v1\0");
    update_text(&mut digest, session_declaration_sha256);
    digest.update(maximum_replicas.to_le_bytes());
    digest.update(reserved_bytes.to_le_bytes());
    Ok(format!("{:x}", digest.finalize()))
}

fn update_text(digest: &mut Sha256, value: &str) {
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn update_optional_usize(digest: &mut Sha256, value: Option<usize>) -> Result<()> {
    digest.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        digest.update(
            u64::try_from(value)
                .map_err(|_| {
                    PowerError::InvalidRequest(
                        "model session device ordinal cannot be represented".to_string(),
                    )
                })?
                .to_le_bytes(),
        );
    }
    Ok(())
}

fn update_limits(digest: &mut Sha256, limits: &InferenceLimits) -> Result<()> {
    for value in [
        limits.max_model_files,
        limits.max_weight_sources,
        limits.max_input_bytes,
        limits.max_tensor_elements,
        limits.max_graph_plan_bytes,
        limits.max_graph_nodes,
        limits.max_graph_initializers,
        limits.max_graph_name_bytes,
        limits.max_context_tokens,
        limits.max_generated_tokens,
        limits.max_concurrent_requests,
        limits.max_queued_requests,
    ] {
        digest.update(
            u64::try_from(value)
                .map_err(|_| {
                    PowerError::InvalidRequest(
                        "model session resource limit cannot be represented".to_string(),
                    )
                })?
                .to_le_bytes(),
        );
    }
    for value in [
        limits.max_model_bytes,
        limits.max_resident_weight_bytes,
        limits.max_state_bytes,
        limits.max_image_pixels,
    ] {
        digest.update(value.to_le_bytes());
    }
    Ok(())
}
