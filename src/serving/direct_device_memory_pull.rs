//! Product-surface `DirectDeviceMemoryPullV1` high-speed transfer port.
//!
//! This pair is the named composition surface for device-memory / HSN pull.
//! It reports [`AdapterProvisionState::Injected`] under
//! [`ProductionAdapterContract::REQUIRED`] so ACL/builder can install it
//! instead of Empty placeholders, but health stays Unavailable and every
//! data-path method refuses work until a real high-speed adapter is bound.
//!
//! This is **not** high-speed-network evidence, RDMA, NIXL, llama.cpp P/D, or
//! model-semantic readiness. Worker observation must not list prefill/decode
//! in `ready_phases` while this port is the composed transport.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::{
    AbortPhaseExecution, AbortStateTransfer, AdapterProvisionState, ConsumeStateTransfer,
    ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PhaseExecutorCapabilities,
    PhaseExecutorHealth, PreparePhaseExecution, PrepareStateTransfer, PreparedPhaseExecution,
    ProductionAdapterContract, PublishStateTransfer, RetryableUnavailableReason,
    ServingExecutionProfile, ServingPhaseExecutor, StateTransferCapabilities,
    StateTransferProtocol, StateTransferReceipt, StateTransferService, StateTransferSource,
    StateTransferTarget, TransferHealth,
};

fn unavailable(port: &str) -> PowerError {
    PowerError::BackendNotAvailable(format!(
        "DirectDeviceMemoryPullV1 {port} is Unavailable until a real high-speed-network adapter is bound"
    ))
}

fn require_direct_device_memory_pull<'a>(
    profile: &'a ServingExecutionProfile,
    port: &str,
) -> Result<&'a super::PrefillDecodeExecutionProfile> {
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(format!(
            "direct-device-memory-pull {port} requires a prefill-decode serving profile"
        )));
    };
    if !matches!(
        execution.protocol,
        StateTransferProtocol::DirectDeviceMemoryPullV1
    ) {
        return Err(PowerError::Config(format!(
            "direct-device-memory-pull {port} requires DirectDeviceMemoryPullV1"
        )));
    }
    Ok(execution)
}

/// Injectable `DirectDeviceMemoryPullV1` [`StateTransferService`].
///
/// Construct with [`Self::for_profile`]. Capabilities bind the immutable
/// profile; health is always [`TransferHealth::Unavailable`].
#[derive(Debug, Clone)]
pub struct DirectDeviceMemoryPullStateTransfer {
    capabilities: StateTransferCapabilities,
}

impl DirectDeviceMemoryPullStateTransfer {
    /// Bind one process profile to the product DirectDeviceMemoryPull port.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        let execution = require_direct_device_memory_pull(profile, "adapter")?;
        let capabilities = StateTransferCapabilities {
            execution_profile_sha256: profile.sha256()?,
            phases: vec![super::ServingPhase::from(execution.role)],
            protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
            max_transfer_bytes: execution.max_state_bytes,
            max_inflight_transfers: execution.max_inflight_transfers,
        };
        profile.validate_state_transfer_capabilities(&capabilities)?;
        Ok(Self { capabilities })
    }
}

#[async_trait]
impl StateTransferService for DirectDeviceMemoryPullStateTransfer {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Unavailable
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Injected
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    async fn prepare_destination(
        &self,
        _command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        Err(unavailable("state-transfer"))
    }

    async fn publish_source(&self, _command: PublishStateTransfer) -> Result<StateTransferSource> {
        Err(unavailable("state-transfer"))
    }

    async fn consume_source(&self, _command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        Err(unavailable("state-transfer"))
    }

    async fn abort(&self, _command: AbortStateTransfer) -> Result<()> {
        Err(unavailable("state-transfer"))
    }
}

/// Injectable companion [`ServingPhaseExecutor`] for DirectDeviceMemoryPull.
///
/// Pairs with [`DirectDeviceMemoryPullStateTransfer`] so composition can install
/// both ports. It does not own KV layout or a backend session; prepare/execute
/// stay [`RetryableUnavailableReason::ExecutorUnavailable`].
#[derive(Debug, Clone)]
pub struct DirectDeviceMemoryPullPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
}

impl DirectDeviceMemoryPullPhaseExecutor {
    /// Build both product-surface ports for one DirectDeviceMemoryPull profile.
    pub fn paired_for_profile(
        profile: &ServingExecutionProfile,
    ) -> Result<(Arc<DirectDeviceMemoryPullStateTransfer>, Arc<Self>)> {
        let transfer = Arc::new(DirectDeviceMemoryPullStateTransfer::for_profile(profile)?);
        let executor = Arc::new(Self::for_profile(profile)?);
        Ok((transfer, executor))
    }

    /// Bind one phase executor to a DirectDeviceMemoryPull profile.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        require_direct_device_memory_pull(profile, "phase executor")?;
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
        })
    }
}

#[async_trait]
impl ServingPhaseExecutor for DirectDeviceMemoryPullPhaseExecutor {
    fn capabilities(&self) -> PhaseExecutorCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> PhaseExecutorHealth {
        PhaseExecutorHealth::Unavailable
    }

    fn provision(&self) -> AdapterProvisionState {
        AdapterProvisionState::Injected
    }

    fn production_contract(&self) -> ProductionAdapterContract {
        ProductionAdapterContract::REQUIRED
    }

    fn may_advertise_ready_phases(&self) -> bool {
        // Named HSN product port — never advertise without HSN evidence.
        false
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
        Err(unavailable("phase-executor"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        validate_injected_production_adapters, BoundedStateTransferService,
        DisaggregatedServingRole, DistributedServingRuntime, EmptyServingPhaseExecutor,
        EmptyStateTransferService, PhaseRequest, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, ServingCompositionTransport, ServingPhase,
        ServingPrivacyMode, StateKind,
    };

    fn profile(transport: Option<ServingCompositionTransport>) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "direct-device-memory-pull".to_string(),
            backend_sha256: "2".repeat(64),
            execution_sha256: "3".repeat(64),
            device_sha256: "4".repeat(64),
            layout_sha256: "5".repeat(64),
            peer_set_sha256: "6".repeat(64),
            generation: 7,
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            state_kind: StateKind::KvCache,
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
            transport,
            phase_executor: None,
            state_ownership: None,
            phase_execution: None,
        })
        .unwrap()
    }

    #[test]
    fn construction_requires_direct_device_memory_pull_protocol() {
        let mut buffered = profile(None);
        if let ServingExecutionProfile::PrefillDecode { execution } = &mut buffered {
            execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        }
        let err = DirectDeviceMemoryPullStateTransfer::for_profile(&buffered).unwrap_err();
        assert!(err.to_string().contains("DirectDeviceMemoryPullV1"));
        let err = DirectDeviceMemoryPullPhaseExecutor::for_profile(&buffered).unwrap_err();
        assert!(err.to_string().contains("DirectDeviceMemoryPullV1"));
    }

    #[test]
    fn construction_rejects_aggregated_profiles() {
        let err =
            DirectDeviceMemoryPullStateTransfer::for_profile(&ServingExecutionProfile::default())
                .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));
    }

    #[tokio::test]
    async fn product_pair_is_injected_required_and_unavailable() {
        let profile = profile(Some(ServingCompositionTransport::DirectDeviceMemoryPull));
        let (transfer, executor) =
            DirectDeviceMemoryPullPhaseExecutor::paired_for_profile(&profile).unwrap();

        assert!(transfer.provision().is_injected());
        assert!(executor.provision().is_injected());
        assert_eq!(
            transfer.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
        assert_eq!(
            executor.production_contract(),
            ProductionAdapterContract::REQUIRED
        );
        assert_eq!(transfer.health(), TransferHealth::Unavailable);
        assert_eq!(executor.health(), PhaseExecutorHealth::Unavailable);
        assert_eq!(
            transfer.capabilities().protocols,
            vec![StateTransferProtocol::DirectDeviceMemoryPullV1]
        );
        assert_eq!(executor.capabilities().phase, ServingPhase::Decode);

        validate_injected_production_adapters(transfer.as_ref(), executor.as_ref()).unwrap();

        let err = transfer
            .abort(AbortStateTransfer {
                transfer_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Unavailable until"));
        assert!(err.to_string().contains("high-speed-network"));

        let decision = executor
            .prepare(PreparePhaseExecution {
                execution_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
                model: "internal/model-v1".to_string(),
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
    fn product_pair_composes_but_never_advertises_prefill_decode() {
        let profile = profile(Some(ServingCompositionTransport::DirectDeviceMemoryPull));
        assert!(!profile.may_advertise_prefill_decode());
        let (transfer, executor) =
            DirectDeviceMemoryPullPhaseExecutor::paired_for_profile(&profile).unwrap();
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, executor).unwrap();
        assert!(!runtime.accepts_work());
        assert_eq!(runtime.phase(), ServingPhase::Decode);
        assert_eq!(runtime.transfer_health(), TransferHealth::Unavailable);
    }

    #[test]
    fn empty_placeholders_remain_rejected_for_the_same_profile() {
        let profile = profile(Some(ServingCompositionTransport::DirectDeviceMemoryPull));
        let transfer = EmptyStateTransferService::for_profile(&profile).unwrap();
        let executor = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate_injected_production_adapters(&transfer, &executor).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));
        assert!(transfer.provision().is_empty());
    }
}
