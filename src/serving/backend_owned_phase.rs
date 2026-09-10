//! Product-surface backend-owned [`ServingPhaseExecutor`] port.
//!
//! This is the named composition surface for a real model/backend phase
//! executor (llama.cpp / picolm layout + KV ownership). It reports
//! [`AdapterProvisionState::Injected`] under
//! [`ProductionAdapterContract::REQUIRED`] so ACL/builder can install it
//! instead of Empty placeholders, but health stays Unavailable and every
//! prepare/execute path refuses Ready work until a concrete state-layout +
//! KV ownership adapter is bound.
//!
//! Pairs only with [`BufferedHostLoopbackStateTransfer`] (or refuses wrong
//! transport). Transfer completion alone never yields cache-hit or decode
//! success. This does **not** invent model-semantic P/D.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::{
    AbortPhaseExecution, AdapterProvisionState, BufferedHostLoopbackStateTransfer,
    ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PhaseExecutorCapabilities,
    PhaseExecutorHealth, PreparePhaseExecution, PreparedPhaseExecution, ProductionAdapterContract,
    RetryableUnavailableReason, ServingExecutionProfile, ServingPhaseExecutor, ServingPrivacyMode,
    StateTransferProtocol, StateTransferService,
};

fn unavailable() -> PowerError {
    PowerError::BackendNotAvailable(
        "BackendOwnedPhaseExecutor is Unavailable until a real state-layout and KV ownership adapter is bound"
            .to_string(),
    )
}

fn require_buffered_host_loopback<'a>(
    profile: &'a ServingExecutionProfile,
) -> Result<&'a super::PrefillDecodeExecutionProfile> {
    let ServingExecutionProfile::PrefillDecode { execution } = profile else {
        return Err(PowerError::Config(
            "backend-owned phase executor requires a prefill-decode serving profile".to_string(),
        ));
    };
    if !matches!(
        execution.protocol,
        StateTransferProtocol::BufferedHostMemoryPullV1
    ) {
        return Err(PowerError::Config(
            "backend-owned phase executor requires BufferedHostMemoryPullV1 (pairs with buffered-host loopback transfer; refuses wrong transport)"
                .to_string(),
        ));
    }
    if !matches!(
        execution.privacy,
        ServingPrivacyMode::AuthenticatedEncryptedTransport
    ) {
        return Err(PowerError::Config(
            "backend-owned phase executor requires AuthenticatedEncryptedTransport privacy"
                .to_string(),
        ));
    }
    Ok(execution)
}

/// Injectable backend-owned [`ServingPhaseExecutor`] product port.
///
/// Construct with [`Self::pair_with`] against
/// [`BufferedHostLoopbackStateTransfer`], or [`Self::paired_for_profile`] to
/// build both ports together. Capabilities bind the immutable profile; health
/// is always [`PhaseExecutorHealth::Unavailable`].
#[derive(Debug, Clone)]
pub struct BackendOwnedPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
}

impl BackendOwnedPhaseExecutor {
    /// Build buffered-host loopback transfer + Unavailable backend-owned phase.
    ///
    /// The transfer may become Ready; this executor stays Unavailable and
    /// never advertises model-semantic P/D readiness.
    pub fn paired_for_profile(
        profile: &ServingExecutionProfile,
    ) -> Result<(Arc<BufferedHostLoopbackStateTransfer>, Arc<Self>)> {
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(profile)?);
        let executor = Arc::new(Self::pair_with(profile, Arc::clone(&transfer))?);
        Ok((transfer, executor))
    }

    /// Bind one backend-owned phase executor to an already-constructed loopback
    /// transfer. Refuses a transfer bound to a different profile digest.
    pub fn pair_with(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
    ) -> Result<Self> {
        require_buffered_host_loopback(profile)?;
        let transfer_caps = transfer.capabilities();
        let profile_sha256 = profile.sha256()?;
        if transfer_caps.execution_profile_sha256 != profile_sha256 {
            return Err(PowerError::Config(
                "backend-owned phase executor requires a buffered-host loopback transfer bound to the same profile digest"
                    .to_string(),
            ));
        }
        if !transfer_caps
            .protocols
            .iter()
            .any(|protocol| matches!(protocol, StateTransferProtocol::BufferedHostMemoryPullV1))
        {
            return Err(PowerError::Config(
                "backend-owned phase executor refuses transfer protocols other than BufferedHostMemoryPullV1"
                    .to_string(),
            ));
        }
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
        })
    }

    /// Bind one backend-owned phase executor to a buffered-host loopback profile
    /// without constructing the transfer (builder may inject transfer separately).
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        require_buffered_host_loopback(profile)?;
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
        })
    }
}

#[async_trait]
impl ServingPhaseExecutor for BackendOwnedPhaseExecutor {
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

    async fn prepare(
        &self,
        _command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        // Transport / transfer success must never become a Ready reservation.
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn execute(
        &self,
        _command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        // Never claim cache hit or decode success without a real KV/layout adapter.
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        Err(unavailable())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        validate_injected_production_adapters, BoundedStateTransferService,
        DirectDeviceMemoryPullPhaseExecutor, DisaggregatedServingRole, DistributedServingRuntime,
        EmptyServingPhaseExecutor, PhaseRequest, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, ServingCompositionPhaseExecutor,
        ServingCompositionTransport, ServingPhase, ServingPrivacyMode, StateKind,
        StateTransferProtocol,
    };

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn buffered_profile(
        transport: Option<ServingCompositionTransport>,
        phase_executor: Option<ServingCompositionPhaseExecutor>,
    ) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: digest('1'),
            backend: "backend-owned".to_string(),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: digest('5'),
            peer_set_sha256: digest('6'),
            generation: 7,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
            state_kind: StateKind::KvCache,
            max_state_bytes: 1024,
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: digest('7'),
            attestation_policy_sha256: None,
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport,
            phase_executor,
        })
        .unwrap()
    }

    #[test]
    fn construction_requires_buffered_host_memory_pull() {
        let mut direct = buffered_profile(None, None);
        if let ServingExecutionProfile::PrefillDecode { execution } = &mut direct {
            execution.protocol = StateTransferProtocol::DirectDeviceMemoryPullV1;
        }
        let err = BackendOwnedPhaseExecutor::for_profile(&direct).unwrap_err();
        assert!(err.to_string().contains("BufferedHostMemoryPullV1"));
        assert!(err.to_string().contains("refuses wrong transport"));
    }

    #[test]
    fn construction_rejects_aggregated_profiles() {
        let err = BackendOwnedPhaseExecutor::for_profile(&ServingExecutionProfile::default())
            .unwrap_err();
        assert!(err.to_string().contains("prefill-decode"));
    }

    #[test]
    fn refuses_pairing_with_direct_device_memory_pull_product_port() {
        let direct = ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: digest('1'),
            backend: "direct-device-memory-pull".to_string(),
            backend_sha256: digest('2'),
            execution_sha256: digest('3'),
            device_sha256: digest('4'),
            layout_sha256: digest('5'),
            peer_set_sha256: digest('6'),
            generation: 7,
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            state_kind: StateKind::KvCache,
            max_state_bytes: 1024,
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: digest('7'),
            attestation_policy_sha256: None,
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport: Some(ServingCompositionTransport::DirectDeviceMemoryPull),
            phase_executor: None,
        })
        .unwrap();
        // DirectDeviceMemoryPull pair constructs; backend-owned must not.
        let _ = DirectDeviceMemoryPullPhaseExecutor::paired_for_profile(&direct).unwrap();
        let err = BackendOwnedPhaseExecutor::for_profile(&direct).unwrap_err();
        assert!(err.to_string().contains("BufferedHostMemoryPullV1"));
    }

    #[tokio::test]
    async fn product_port_is_injected_required_and_unavailable() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();

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
        assert_eq!(transfer.health(), crate::serving::TransferHealth::Ready);
        assert_eq!(executor.health(), PhaseExecutorHealth::Unavailable);
        assert_eq!(executor.capabilities().phase, ServingPhase::Decode);
        assert!(!profile.may_advertise_prefill_decode());

        validate_injected_production_adapters(transfer.as_ref(), executor.as_ref()).unwrap();

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
            PhaseDecision::Ready(_) => panic!("transport alone must never yield Ready"),
            PhaseDecision::Recompute { .. } => {
                panic!("expected retryable unavailable, got Recompute")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected retryable unavailable, got TerminalFailure")
            }
        }

        let err = executor
            .abort(AbortPhaseExecution {
                execution_id: uuid::Uuid::new_v4(),
                local_worker_epoch: uuid::Uuid::new_v4(),
                execution: None,
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("Unavailable until"));
        assert!(err.to_string().contains("KV ownership"));
    }

    #[test]
    fn composed_runtime_never_advertises_prefill_decode() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, executor).unwrap();
        assert!(!runtime.accepts_work());
        assert_eq!(runtime.phase(), ServingPhase::Decode);
    }

    #[test]
    fn empty_placeholders_remain_rejected_for_the_same_profile() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let transfer = crate::serving::EmptyStateTransferService::for_profile(&profile).unwrap();
        let executor = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate_injected_production_adapters(&transfer, &executor).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));
    }

    #[test]
    fn pair_with_rejects_foreign_profile_digest() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let other = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            None,
        );
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&other).unwrap());
        let err = BackendOwnedPhaseExecutor::pair_with(&profile, transfer).unwrap_err();
        assert!(err.to_string().contains("same profile digest"));
    }
}
