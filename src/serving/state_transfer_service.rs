use std::fmt;

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use crate::error::{PowerError, Result};

use super::{
    ServingPhase, StateTransferBinding, StateTransferCapabilities, StateTransferReceipt,
    StateTransferSource, StateTransferTarget, TransferHealth,
};

const MAX_LOCAL_HANDLE_BYTES: usize = 512;
const MAX_TRANSFER_LIFETIME_SECONDS: i64 = 300;

/// Process-local backend handle. It is deliberately not serializable and its
/// debug representation never reveals the adapter-owned value.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModelStateHandle(String);

impl ModelStateHandle {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_LOCAL_HANDLE_BYTES
            || value.trim() != value
            || value.chars().any(char::is_control)
        {
            return Err(PowerError::InvalidRequest(format!(
                "model-state handle must be non-empty, trimmed, control-free, and at most {MAX_LOCAL_HANDLE_BYTES} bytes"
            )));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModelStateHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ModelStateHandle([REDACTED])")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrepareStateTransfer {
    pub transfer_id: Uuid,
    pub local_worker_epoch: Uuid,
    pub binding: StateTransferBinding,
    pub destination: ModelStateHandle,
    pub expires_at: DateTime<Utc>,
}

impl PrepareStateTransfer {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        validate_command_identity(self.transfer_id, self.local_worker_epoch)?;
        validate_phase_and_binding(ServingPhase::Decode, &self.binding, capabilities)?;
        let maximum_expiry = now
            .checked_add_signed(Duration::seconds(MAX_TRANSFER_LIFETIME_SECONDS))
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "state-transfer expiry is outside the representable range".to_string(),
                )
            })?;
        if self.expires_at <= now || self.expires_at > maximum_expiry {
            return Err(PowerError::InvalidRequest(format!(
                "state-transfer expiry must be within {MAX_TRANSFER_LIFETIME_SECONDS} seconds"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishStateTransfer {
    pub local_worker_epoch: Uuid,
    pub source: ModelStateHandle,
    pub target: StateTransferTarget,
}

impl PublishStateTransfer {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        validate_command_identity(self.target.transfer_id, self.local_worker_epoch)?;
        validate_phase_and_binding(ServingPhase::Prefill, &self.target.binding, capabilities)?;
        self.target.validate_at(now, capabilities)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsumeStateTransfer {
    pub local_worker_epoch: Uuid,
    pub destination: ModelStateHandle,
    pub source: StateTransferSource,
}

impl ConsumeStateTransfer {
    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        validate_command_identity(self.source.transfer_id, self.local_worker_epoch)?;
        validate_phase_and_binding(ServingPhase::Decode, &self.source.binding, capabilities)?;
        if self.local_worker_epoch != self.source.destination_worker_epoch {
            return Err(PowerError::InvalidRequest(
                "state-transfer source belongs to a different destination process".to_string(),
            ));
        }
        self.source.validate_at(now, capabilities)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbortStateTransfer {
    pub transfer_id: Uuid,
    pub local_worker_epoch: Uuid,
}

impl AbortStateTransfer {
    pub fn validate(&self) -> Result<()> {
        validate_command_identity(self.transfer_id, self.local_worker_epoch)
    }
}

/// Infrastructure port implemented by a concrete state-transfer data path.
///
/// The adapter owns registered memory, connection metadata, timeouts, and
/// cleanup. Power passes only local opaque handles and bounded wire tickets;
/// Gateway and Cloud never receive KV bytes.
#[async_trait]
pub trait StateTransferService: Send + Sync {
    fn capabilities(&self) -> StateTransferCapabilities;
    fn health(&self) -> TransferHealth;

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget>;

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource>;

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt>;

    async fn abort(&self, command: AbortStateTransfer) -> Result<()>;
}

fn validate_command_identity(transfer_id: Uuid, worker_epoch: Uuid) -> Result<()> {
    if transfer_id.is_nil() || worker_epoch.is_nil() {
        return Err(PowerError::InvalidRequest(
            "state-transfer command identity is invalid".to_string(),
        ));
    }
    Ok(())
}

fn validate_phase_and_binding(
    phase: ServingPhase,
    binding: &StateTransferBinding,
    capabilities: &StateTransferCapabilities,
) -> Result<()> {
    capabilities.validate()?;
    binding.validate()?;
    if !capabilities.supports_phase(phase) {
        return Err(PowerError::InvalidRequest(format!(
            "state-transfer adapter does not support the {phase:?} phase"
        )));
    }
    if binding.state_bytes > capabilities.max_transfer_bytes {
        return Err(PowerError::InvalidRequest(
            "state-transfer command exceeds the adapter byte limit".to_string(),
        ));
    }
    Ok(())
}
