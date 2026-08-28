use std::pin::Pin;

use chrono::{DateTime, Utc};
use futures::Stream;
use uuid::Uuid;

use crate::backend::types::{ChatResponseChunk, CompletionResponseChunk};
use crate::error::{PowerError, Result};

use super::super::{
    AbortStateTransfer, ConsumeStateTransfer, ModelStateHandle, ServingExecutionProfile,
    ServingPhase, StateTransferBinding, StateTransferCapabilities, StateTransferReceipt,
    StateTransferService, TransferHealth,
};
use super::{
    configured_cancellation_timeout_ms, configured_transfer_timeout_ms, validate_command_identity,
    validate_expiry, validate_sha256, PhaseExecutionHandle, PhaseExecutorCapabilities,
};

/// Prefill reservation owned by the executor until execute or abort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedPrefillPhase {
    execution_id: Uuid,
    local_worker_epoch: Uuid,
    execution_profile_sha256: String,
    execution: PhaseExecutionHandle,
    expires_at: DateTime<Utc>,
}

impl PreparedPrefillPhase {
    pub fn new(
        execution_id: Uuid,
        local_worker_epoch: Uuid,
        execution_profile_sha256: String,
        execution: PhaseExecutionHandle,
        expires_at: DateTime<Utc>,
    ) -> Result<Self> {
        validate_command_identity(execution_id, local_worker_epoch)?;
        validate_sha256(
            &execution_profile_sha256,
            "prepared prefill execution profile",
        )?;
        Ok(Self {
            execution_id,
            local_worker_epoch,
            execution_profile_sha256,
            execution,
            expires_at,
        })
    }

    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &PhaseExecutorCapabilities,
        profile: &ServingExecutionProfile,
    ) -> Result<()> {
        profile.validate_phase_executor_capabilities(capabilities)?;
        if capabilities.phase != ServingPhase::Prefill
            || self.execution_profile_sha256 != capabilities.execution_profile_sha256
        {
            return Err(PowerError::InvalidRequest(
                "prepared prefill does not match the active phase executor".to_string(),
            ));
        }
        validate_command_identity(self.execution_id, self.local_worker_epoch)?;
        validate_expiry(
            now,
            self.expires_at,
            configured_transfer_timeout_ms(profile)?,
        )
    }

    pub fn execution_id(&self) -> Uuid {
        self.execution_id
    }

    pub fn local_worker_epoch(&self) -> Uuid {
        self.local_worker_epoch
    }

    pub fn execution(&self) -> &PhaseExecutionHandle {
        &self.execution
    }

    pub fn expires_at(&self) -> DateTime<Utc> {
        self.expires_at
    }
}

/// Decode reservation and the exact local destination that receives state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDecodePhase {
    execution_id: Uuid,
    local_worker_epoch: Uuid,
    execution_profile_sha256: String,
    execution: PhaseExecutionHandle,
    destination: ModelStateHandle,
    binding: StateTransferBinding,
    expires_at: DateTime<Utc>,
}

impl PreparedDecodePhase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        execution_id: Uuid,
        local_worker_epoch: Uuid,
        execution_profile_sha256: String,
        execution: PhaseExecutionHandle,
        destination: ModelStateHandle,
        binding: StateTransferBinding,
        expires_at: DateTime<Utc>,
    ) -> Result<Self> {
        validate_command_identity(execution_id, local_worker_epoch)?;
        validate_sha256(
            &execution_profile_sha256,
            "prepared decode execution profile",
        )?;
        binding.validate()?;
        Ok(Self {
            execution_id,
            local_worker_epoch,
            execution_profile_sha256,
            execution,
            destination,
            binding,
            expires_at,
        })
    }

    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        capabilities: &PhaseExecutorCapabilities,
        profile: &ServingExecutionProfile,
    ) -> Result<()> {
        profile.validate_phase_executor_capabilities(capabilities)?;
        profile.validate_state_binding(&self.binding)?;
        if capabilities.phase != ServingPhase::Decode
            || self.execution_profile_sha256 != capabilities.execution_profile_sha256
        {
            return Err(PowerError::InvalidRequest(
                "prepared decode does not match the active phase executor".to_string(),
            ));
        }
        validate_command_identity(self.execution_id, self.local_worker_epoch)?;
        validate_expiry(
            now,
            self.expires_at,
            configured_transfer_timeout_ms(profile)?,
        )
    }

    pub fn execution_id(&self) -> Uuid {
        self.execution_id
    }

    pub fn local_worker_epoch(&self) -> Uuid {
        self.local_worker_epoch
    }

    pub fn execution(&self) -> &PhaseExecutionHandle {
        &self.execution
    }

    pub fn destination(&self) -> &ModelStateHandle {
        &self.destination
    }

    pub fn binding(&self) -> &StateTransferBinding {
        &self.binding
    }

    pub fn expires_at(&self) -> DateTime<Utc> {
        self.expires_at
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreparedPhaseExecution {
    Prefill(PreparedPrefillPhase),
    Decode(PreparedDecodePhase),
}

impl PreparedPhaseExecution {
    pub fn phase(&self) -> ServingPhase {
        match self {
            Self::Prefill(_) => ServingPhase::Prefill,
            Self::Decode(_) => ServingPhase::Decode,
        }
    }
}

/// Prefill output that still requires publication through the transfer port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProducedModelState {
    execution_id: Uuid,
    local_worker_epoch: Uuid,
    execution_profile_sha256: String,
    source: ModelStateHandle,
    binding: StateTransferBinding,
}

impl ProducedModelState {
    pub fn new(
        execution_id: Uuid,
        local_worker_epoch: Uuid,
        execution_profile_sha256: String,
        source: ModelStateHandle,
        binding: StateTransferBinding,
    ) -> Result<Self> {
        validate_command_identity(execution_id, local_worker_epoch)?;
        validate_sha256(
            &execution_profile_sha256,
            "produced-state execution profile",
        )?;
        binding.validate()?;
        Ok(Self {
            execution_id,
            local_worker_epoch,
            execution_profile_sha256,
            source,
            binding,
        })
    }

    pub fn validate_for(
        &self,
        prepared: &PreparedPrefillPhase,
        profile: &ServingExecutionProfile,
    ) -> Result<()> {
        profile.validate_state_binding(&self.binding)?;
        if self.execution_id != prepared.execution_id
            || self.local_worker_epoch != prepared.local_worker_epoch
            || self.execution_profile_sha256 != prepared.execution_profile_sha256
        {
            return Err(PowerError::InvalidRequest(
                "produced model state does not match the prepared prefill execution".to_string(),
            ));
        }
        Ok(())
    }

    pub fn source(&self) -> &ModelStateHandle {
        &self.source
    }

    pub fn binding(&self) -> &StateTransferBinding {
        &self.binding
    }
}

/// Proof that one transfer receipt corresponds to the state consumed into a
/// specific decode destination handle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedModelState {
    transfer_id: Uuid,
    local_worker_epoch: Uuid,
    execution_profile_sha256: String,
    destination: ModelStateHandle,
    binding: StateTransferBinding,
}

impl ImportedModelState {
    /// Consume state through the configured adapter and return a proof tied to
    /// the exact local destination handle used for that call.
    pub async fn consume_at(
        service: &dyn StateTransferService,
        command: ConsumeStateTransfer,
        now: DateTime<Utc>,
        profile: &ServingExecutionProfile,
    ) -> Result<Self> {
        let capabilities = service.capabilities();
        profile.validate_state_transfer_capabilities(&capabilities)?;
        profile.validate_state_binding(&command.source.binding)?;
        command.validate_at(now, &capabilities)?;
        let configured_timeout_ms = configured_transfer_timeout_ms(profile)?;
        validate_expiry(now, command.source.expires_at, configured_timeout_ms)?;
        if !matches!(
            service.health(),
            TransferHealth::Ready | TransferHealth::Degraded
        ) {
            return Err(PowerError::BackendNotAvailable(
                "state-transfer adapter cannot accept imported state".to_string(),
            ));
        }
        let configured_timeout = std::time::Duration::from_millis(configured_timeout_ms);
        let remaining = (command.source.expires_at - now).to_std().map_err(|_| {
            PowerError::InvalidRequest("state-transfer source has already expired".to_string())
        })?;
        let consume_result = tokio::time::timeout(
            configured_timeout.min(remaining),
            service.consume_source(command.clone()),
        )
        .await;
        let receipt = match consume_result {
            Ok(result) => result?,
            Err(_) => {
                let cancellation_timeout =
                    std::time::Duration::from_millis(configured_cancellation_timeout_ms(profile)?);
                let cleanup = tokio::time::timeout(
                    cancellation_timeout,
                    service.abort(AbortStateTransfer {
                        transfer_id: command.source.transfer_id,
                        local_worker_epoch: command.local_worker_epoch,
                    }),
                )
                .await;
                return match cleanup {
                    Ok(Ok(())) => Err(PowerError::BackendNotAvailable(
                        "state-transfer consume exceeded the immutable profile timeout".to_string(),
                    )),
                    Ok(Err(_)) | Err(_) => Err(PowerError::BackendNotAvailable(
                        "state-transfer consume timed out and cleanup was not confirmed"
                            .to_string(),
                    )),
                };
            }
        };
        Self::verify_receipt_at(&command, receipt, now, Utc::now(), profile, &capabilities)
    }

    fn verify_receipt_at(
        command: &ConsumeStateTransfer,
        receipt: StateTransferReceipt,
        started_at: DateTime<Utc>,
        verified_at: DateTime<Utc>,
        profile: &ServingExecutionProfile,
        capabilities: &StateTransferCapabilities,
    ) -> Result<Self> {
        profile.validate_state_transfer_capabilities(capabilities)?;
        profile.validate_state_binding(&command.source.binding)?;
        command.validate_at(started_at, capabilities)?;
        receipt.validate_for(&command.source, capabilities)?;
        if receipt.completed_at < started_at || receipt.completed_at > verified_at {
            return Err(PowerError::InvalidRequest(
                "state-transfer receipt completion is outside the local consume attempt"
                    .to_string(),
            ));
        }
        Ok(Self {
            transfer_id: receipt.transfer_id,
            local_worker_epoch: command.local_worker_epoch,
            execution_profile_sha256: profile.sha256()?,
            destination: command.destination.clone(),
            binding: receipt.binding,
        })
    }

    fn validate(
        &self,
        profile: &ServingExecutionProfile,
        capabilities: &StateTransferCapabilities,
    ) -> Result<()> {
        profile.validate_state_transfer_capabilities(capabilities)?;
        profile.validate_state_binding(&self.binding)?;
        if self.transfer_id.is_nil()
            || self.local_worker_epoch.is_nil()
            || self.execution_profile_sha256 != profile.sha256()?
        {
            return Err(PowerError::InvalidRequest(
                "imported model state does not match the active serving process".to_string(),
            ));
        }
        Ok(())
    }

    pub fn transfer_id(&self) -> Uuid {
        self.transfer_id
    }

    pub fn destination(&self) -> &ModelStateHandle {
        &self.destination
    }

    pub fn binding(&self) -> &StateTransferBinding {
        &self.binding
    }
}

/// Phase-specific input accepted only after preparation and compatibility
/// validation.
pub enum ExecutePhaseExecution {
    Prefill {
        prepared: PreparedPrefillPhase,
    },
    Decode {
        prepared: Box<PreparedDecodePhase>,
        state: Box<ImportedModelState>,
    },
}

impl ExecutePhaseExecution {
    pub fn prefill(prepared: PreparedPrefillPhase) -> Self {
        Self::Prefill { prepared }
    }

    pub fn decode(prepared: PreparedDecodePhase, state: ImportedModelState) -> Result<Self> {
        if prepared.local_worker_epoch != state.local_worker_epoch
            || prepared.execution_profile_sha256 != state.execution_profile_sha256
            || prepared.destination != state.destination
            || prepared.binding != state.binding
        {
            return Err(PowerError::InvalidRequest(
                "imported model state does not match the prepared decode execution".to_string(),
            ));
        }
        Ok(Self::Decode {
            prepared: Box::new(prepared),
            state: Box::new(state),
        })
    }

    pub fn phase(&self) -> ServingPhase {
        match self {
            Self::Prefill { .. } => ServingPhase::Prefill,
            Self::Decode { .. } => ServingPhase::Decode,
        }
    }

    pub fn validate_at(
        &self,
        now: DateTime<Utc>,
        executor_capabilities: &PhaseExecutorCapabilities,
        transfer_capabilities: &StateTransferCapabilities,
        profile: &ServingExecutionProfile,
    ) -> Result<()> {
        profile.validate_phase_executor_capabilities(executor_capabilities)?;
        profile.validate_state_transfer_capabilities(transfer_capabilities)?;
        match self {
            Self::Prefill { prepared } => prepared.validate_at(now, executor_capabilities, profile),
            Self::Decode { prepared, state } => {
                prepared.validate_at(now, executor_capabilities, profile)?;
                state.validate(profile, transfer_capabilities)?;
                if prepared.local_worker_epoch != state.local_worker_epoch
                    || prepared.execution_profile_sha256 != state.execution_profile_sha256
                    || prepared.destination != state.destination
                    || prepared.binding != state.binding
                {
                    return Err(PowerError::InvalidRequest(
                        "imported model state does not match the prepared decode execution"
                            .to_string(),
                    ));
                }
                Ok(())
            }
        }
    }
}

/// One backend response item; state transfer never creates these values.
#[derive(Debug, Clone)]
pub enum PhaseResponseChunk {
    Chat(ChatResponseChunk),
    Completion(CompletionResponseChunk),
}

pub type PhaseResponseStream =
    Pin<Box<dyn Stream<Item = Result<PhaseResponseChunk>> + Send + 'static>>;

/// Successful phase execution. A decode is successful only when its backend
/// returns a response stream after consuming verified state.
pub enum PhaseExecutionOutput {
    Prefill(ProducedModelState),
    Decode(PhaseResponseStream),
}

impl PhaseExecutionOutput {
    pub fn phase(&self) -> ServingPhase {
        match self {
            Self::Prefill(_) => ServingPhase::Prefill,
            Self::Decode(_) => ServingPhase::Decode,
        }
    }
}
