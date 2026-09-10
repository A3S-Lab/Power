//! Empty / Unavailable ServingPhaseExecutor and StateTransferService placeholders.
//!
//! These types bind the immutable profile capability shape while remaining
//! Unavailable and refusing work until a concrete adapter is injected.
//! Composition treats Empty as not injected. This does **not** claim
//! high-speed-network transport or production readiness.

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::{
    AbortPhaseExecution, AbortStateTransfer, AdapterProvisionState, ConsumeStateTransfer,
    ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PhaseExecutorCapabilities,
    PhaseExecutorHealth, PhaseRequest, PreparePhaseExecution, PrepareStateTransfer,
    PreparedPhaseExecution, ProductionAdapterContract, PublishStateTransfer,
    RetryableUnavailableReason, ServingExecutionProfile, ServingPhase, ServingPhaseExecutor,
    StateTransferCapabilities, StateTransferReceipt, StateTransferService, StateTransferSource,
    StateTransferTarget, TransferHealth,
};

/// Reject Empty placeholders and non-required contracts before listeners start.
pub fn validate_injected_production_adapters(
    state_transfer: &dyn StateTransferService,
    phase_executor: &dyn ServingPhaseExecutor,
) -> Result<()> {
    if state_transfer.provision().is_empty() || phase_executor.provision().is_empty() {
        return Err(PowerError::Config(
            "prefill-decode serving requires injected state-transfer and phase-executor adapters; Empty placeholders remain Unavailable until injected"
                .to_string(),
        ));
    }
    state_transfer.production_contract().validate()?;
    phase_executor.production_contract().validate()?;
    Ok(())
}

fn empty_unavailable(port: &str) -> PowerError {
    PowerError::BackendNotAvailable(format!(
        "Empty {port} adapter is Unavailable until a concrete production adapter is injected"
    ))
}

fn profile_transfer_capabilities(
    profile: &ServingExecutionProfile,
) -> Result<StateTransferCapabilities> {
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "Empty state-transfer adapter requires a prefill-decode serving profile".to_string(),
        ));
    };
    let capabilities = StateTransferCapabilities {
        execution_profile_sha256: profile.sha256()?,
        phases: vec![ServingPhase::from(execution.role)],
        protocols: vec![execution.protocol],
        max_transfer_bytes: execution.max_state_bytes,
        max_inflight_transfers: execution.max_inflight_transfers,
    };
    profile.validate_state_transfer_capabilities(&capabilities)?;
    Ok(capabilities)
}

/// Fail-closed state-transfer placeholder until a concrete adapter is injected.
///
/// Capabilities may match the immutable profile so callers can reason about the
/// required shape, but health stays [`TransferHealth::Unavailable`] and every
/// data-path method refuses work. Composition treats Empty as not injected.
#[derive(Debug, Clone)]
pub struct EmptyStateTransferService {
    capabilities: StateTransferCapabilities,
}

impl EmptyStateTransferService {
    /// Build an Empty/Unavailable transfer placeholder bound to one profile.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        Ok(Self {
            capabilities: profile_transfer_capabilities(profile)?,
        })
    }
}

#[async_trait]
impl StateTransferService for EmptyStateTransferService {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Unavailable
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Empty
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    async fn prepare_destination(
        &self,
        _command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        Err(empty_unavailable("state-transfer"))
    }

    async fn publish_source(&self, _command: PublishStateTransfer) -> Result<StateTransferSource> {
        Err(empty_unavailable("state-transfer"))
    }

    async fn consume_source(&self, _command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        Err(empty_unavailable("state-transfer"))
    }

    async fn abort(&self, _command: AbortStateTransfer) -> Result<()> {
        Err(empty_unavailable("state-transfer"))
    }
}

/// Fail-closed phase-executor placeholder until a concrete adapter is injected.
#[derive(Debug, Clone)]
pub struct EmptyServingPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
}

impl EmptyServingPhaseExecutor {
    /// Build an Empty/Unavailable phase-executor placeholder bound to one profile.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
        })
    }
}

#[async_trait]
impl ServingPhaseExecutor for EmptyServingPhaseExecutor {
    fn capabilities(&self) -> PhaseExecutorCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> PhaseExecutorHealth {
        PhaseExecutorHealth::Unavailable
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Empty
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    async fn prepare(
        &self,
        _command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn execute(
        &self,
        _command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        Err(empty_unavailable("phase-executor"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        DisaggregatedServingRole, EmptyServingPhaseExecutor, EmptyStateTransferService,
        PhaseSessionPoolMode, PhaseWeightCacheMode, PrefillDecodeExecutionProfile,
        ServingPrivacyMode, StateKind, StateTransferProtocol,
    };

    fn profile() -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "fixture".to_string(),
            backend: "fixture".to_string(),
            model_sha256: "1".repeat(64),
            backend_sha256: "2".repeat(64),
            execution_sha256: "3".repeat(64),
            device_sha256: "4".repeat(64),
            layout_sha256: "5".repeat(64),
            peer_set_sha256: "6".repeat(64),
            generation: 1,
            state_kind: StateKind::KvCache,
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            max_state_bytes: 1024,
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: "7".repeat(64),
            attestation_policy_sha256: None,
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport: None,
            phase_executor: None,
            state_ownership: None,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn empty_state_transfer_is_unavailable_and_refuses_work() {
        let service = EmptyStateTransferService::for_profile(&profile()).unwrap();
        assert_eq!(service.health(), TransferHealth::Unavailable);
        assert!(service.provision().is_empty());
        assert_eq!(
            service.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
        let err = service
            .abort(AbortStateTransfer {
                transfer_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Unavailable until"));
    }

    #[tokio::test]
    async fn empty_phase_executor_is_unavailable_and_refuses_ready_work() {
        let executor = EmptyServingPhaseExecutor::for_profile(&profile()).unwrap();
        assert_eq!(executor.health(), PhaseExecutorHealth::Unavailable);
        assert!(executor.provision().is_empty());
        let decision = executor
            .prepare(PreparePhaseExecution {
                execution_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
                model: "fixture".to_string(),
                expires_at: chrono::Utc::now() + chrono::Duration::seconds(30),
                request: PhaseRequest::Completion(
                    serde_json::from_value(serde_json::json!({ "prompt": "unused" })).unwrap(),
                ),
            })
            .await
            .unwrap();
        match decision {
            PhaseDecision::RetryableUnavailable { reason, .. } => {
                assert_eq!(reason, RetryableUnavailableReason::ExecutorUnavailable);
            }
            PhaseDecision::Ready(_) => panic!("expected retryable unavailable, got Ready"),
            PhaseDecision::Recompute { .. } => {
                panic!("expected retryable unavailable, got Recompute")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected retryable unavailable, got TerminalFailure")
            }
        }
    }

    #[test]
    fn empty_pair_fails_injected_production_validation() {
        let profile = profile();
        let transfer = EmptyStateTransferService::for_profile(&profile).unwrap();
        let executor = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate_injected_production_adapters(&transfer, &executor).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));
    }
}
