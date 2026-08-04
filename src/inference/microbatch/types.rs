use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::super::sealed_state::decode_sha256;
use super::super::{
    ExecutionBatchBinding, HardwareMemorySnapshot, InferenceLimits, RuntimeDeviceIdentity,
};

pub(super) const BASIS_POINTS: u64 = 10_000;

/// Model-owned peak resource declaration for one scheduler slot.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MicrobatchCandidate {
    pub slot_sha256: String,
    pub input_bytes: usize,
    pub input_elements: usize,
    pub state_bytes: u64,
    pub host_peak_bytes: u64,
    pub device_peak_bytes: u64,
}

impl MicrobatchCandidate {
    pub fn new(
        slot_sha256: impl Into<String>,
        input_bytes: usize,
        input_elements: usize,
        state_bytes: u64,
        host_peak_bytes: u64,
        device_peak_bytes: u64,
    ) -> Result<Self> {
        let candidate = Self {
            slot_sha256: slot_sha256.into(),
            input_bytes,
            input_elements,
            state_bytes,
            host_peak_bytes,
            device_peak_bytes,
        };
        candidate.validate_intrinsic()?;
        Ok(candidate)
    }

    pub(super) fn validate_intrinsic(&self) -> Result<()> {
        decode_sha256(&self.slot_sha256, "microbatch slot")?;
        if self.input_bytes == 0 || self.input_elements == 0 {
            return Err(PowerError::InvalidRequest(
                "microbatch input bytes and elements must be greater than zero".to_string(),
            ));
        }
        let declared_memory = self
            .host_peak_bytes
            .checked_add(self.device_peak_bytes)
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "microbatch slot peak memory declaration overflowed".to_string(),
                )
            })?;
        let required_memory = u64::try_from(self.input_bytes)
            .ok()
            .and_then(|bytes| bytes.checked_add(self.state_bytes))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "microbatch input and state byte declaration overflowed".to_string(),
                )
            })?;
        if declared_memory < required_memory {
            return Err(PowerError::InvalidRequest(
                "microbatch host and device peak bytes must cover input plus state bytes"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for MicrobatchCandidate {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MicrobatchCandidate")
            .field("slot", &"redacted-sha256")
            .field("input_bytes", &self.input_bytes)
            .field("input_elements", &self.input_elements)
            .field("state_bytes", &self.state_bytes)
            .field("host_peak_bytes", &self.host_peak_bytes)
            .field("device_peak_bytes", &self.device_peak_bytes)
            .finish()
    }
}

/// Caller-selected memory headroom and maximum slot count per microbatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MicrobatchPolicy {
    pub max_batch_items: usize,
    pub host_available_fraction_bps: u16,
    pub device_available_fraction_bps: u16,
    #[serde(default)]
    pub host_reserve_bytes: u64,
    #[serde(default)]
    pub device_reserve_bytes: u64,
    #[serde(default)]
    pub base_host_bytes: u64,
    #[serde(default)]
    pub base_device_bytes: u64,
}

impl MicrobatchPolicy {
    pub fn new(
        max_batch_items: usize,
        host_available_fraction_bps: u16,
        device_available_fraction_bps: u16,
    ) -> Result<Self> {
        let policy = Self {
            max_batch_items,
            host_available_fraction_bps,
            device_available_fraction_bps,
            host_reserve_bytes: 0,
            device_reserve_bytes: 0,
            base_host_bytes: 0,
            base_device_bytes: 0,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn with_host_reserve_bytes(mut self, bytes: u64) -> Self {
        self.host_reserve_bytes = bytes;
        self
    }

    pub fn with_device_reserve_bytes(mut self, bytes: u64) -> Self {
        self.device_reserve_bytes = bytes;
        self
    }

    pub fn with_base_memory(mut self, host_bytes: u64, device_bytes: u64) -> Self {
        self.base_host_bytes = host_bytes;
        self.base_device_bytes = device_bytes;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_batch_items == 0 {
            return Err(PowerError::Config(
                "microbatch maximum item count must be greater than zero".to_string(),
            ));
        }
        if self.host_available_fraction_bps == 0 && self.device_available_fraction_bps == 0 {
            return Err(PowerError::Config(
                "at least one microbatch memory fraction must be greater than zero".to_string(),
            ));
        }
        if u64::from(self.host_available_fraction_bps) > BASIS_POINTS
            || u64::from(self.device_available_fraction_bps) > BASIS_POINTS
        {
            return Err(PowerError::Config(
                "microbatch memory fractions cannot exceed 10,000 basis points".to_string(),
            ));
        }
        Ok(())
    }
}

/// Runtime limits that affect microbatch grouping and declaration identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MicrobatchLimits {
    pub max_concurrent_requests: usize,
    pub max_input_bytes: usize,
    pub max_tensor_elements: usize,
    pub max_state_bytes: u64,
    pub max_graph_nodes: usize,
}

impl From<&InferenceLimits> for MicrobatchLimits {
    fn from(limits: &InferenceLimits) -> Self {
        Self {
            max_concurrent_requests: limits.max_concurrent_requests,
            max_input_bytes: limits.max_input_bytes,
            max_tensor_elements: limits.max_tensor_elements,
            max_state_bytes: limits.max_state_bytes,
            max_graph_nodes: limits.max_graph_nodes,
        }
    }
}

/// One exact input slot retained in caller order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannedMicrobatchSlot {
    pub source_index: usize,
    pub slot_sha256: String,
    pub input_bytes: usize,
    pub input_elements: usize,
    pub state_bytes: u64,
    pub host_peak_bytes: u64,
    pub device_peak_bytes: u64,
}

impl PlannedMicrobatchSlot {
    pub(super) fn candidate(&self) -> MicrobatchCandidate {
        MicrobatchCandidate {
            slot_sha256: self.slot_sha256.clone(),
            input_bytes: self.input_bytes,
            input_elements: self.input_elements,
            state_bytes: self.state_bytes,
            host_peak_bytes: self.host_peak_bytes,
            device_peak_bytes: self.device_peak_bytes,
        }
    }
}

/// One contiguous, resource-safe microbatch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannedMicrobatch {
    pub index: usize,
    pub slots: Vec<PlannedMicrobatchSlot>,
    pub input_bytes: usize,
    pub input_elements: usize,
    pub state_bytes: u64,
    pub host_peak_bytes: u64,
    pub device_peak_bytes: u64,
}

/// Deterministic point-in-time plan. The raw memory snapshot is caller-owned
/// and is never placed in an execution receipt automatically.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MicrobatchPlan {
    pub schema: String,
    pub declaration_sha256: String,
    pub binding: ExecutionBatchBinding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_declaration_sha256: Option<String>,
    pub runtime_device: RuntimeDeviceIdentity,
    pub snapshot: HardwareMemorySnapshot,
    pub limits: MicrobatchLimits,
    pub policy: MicrobatchPolicy,
    pub host_budget_bytes: u64,
    pub device_budget_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_budget_bytes: Option<u64>,
    pub slot_count: usize,
    pub batches: Vec<PlannedMicrobatch>,
}

impl MicrobatchPlan {
    pub const SCHEMA: &'static str = "a3s.power.microbatch-plan.v1";

    pub fn validate(&self) -> Result<()> {
        super::planner::validate_plan(self)
    }

    pub fn revalidate_pressure(&self, current: &HardwareMemorySnapshot) -> Result<()> {
        super::planner::revalidate_pressure(self, current)
    }

    pub(super) fn revalidate_for_runtime(
        &self,
        device: RuntimeDeviceIdentity,
        limits: &InferenceLimits,
        current: &HardwareMemorySnapshot,
    ) -> Result<()> {
        if self.runtime_device != device || self.limits != MicrobatchLimits::from(limits) {
            return Err(PowerError::InvalidRequest(
                "microbatch plan belongs to a different runtime device or limit set".to_string(),
            ));
        }
        self.revalidate_pressure(current)
    }
}

impl std::fmt::Debug for MicrobatchPlan {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MicrobatchPlan")
            .field("declaration", &"sha256")
            .field("runtime_device", &self.runtime_device)
            .field("slot_count", &self.slot_count)
            .field("batch_count", &self.batches.len())
            .finish_non_exhaustive()
    }
}
