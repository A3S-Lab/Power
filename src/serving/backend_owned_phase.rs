//! Product-surface backend-owned [`ServingPhaseExecutor`] port.
//!
//! This is the named composition surface for a real model/backend phase
//! executor (llama.cpp / picolm layout + KV ownership). It reports
//! [`AdapterProvisionState::Injected`] under
//! [`ProductionAdapterContract::REQUIRED`] so ACL/builder can install it
//! instead of Empty placeholders.
//!
//! Default [`EmptyBackendPhaseStateOwnership`] keeps health
//! [`PhaseExecutorHealth::Unavailable`]. Binding a non-Empty
//! [`BackendPhaseStateOwnership`] validates profile `layout_sha256` (via
//! `state_layout_sha256`) and related digests fail-closed, then advances
//! health to [`PhaseExecutorHealth::Eligible`]. Eligible still refuses Ready
//! prepare/execute until a real execute adapter path exists—matching layout
//! registration alone is not decode success and does not invent model-semantic
//! P/D.
//!
//! Pairs only with [`BufferedHostLoopbackStateTransfer`] (or refuses wrong
//! transport). Transfer completion alone never yields cache-hit or decode
//! success.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::backend_phase_state_ownership::{
    bind_backend_phase_state_ownership, BackendPhaseStateOwnership, EmptyBackendPhaseStateOwnership,
};
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

fn ready_blocked() -> PowerError {
    PowerError::BackendNotAvailable(
        "BackendOwnedPhaseExecutor is Eligible after state-layout binding but Ready prepare/execute remains blocked until a real execute adapter path exists"
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
/// build both ports together (Empty ownership → Unavailable). Bind a concrete
/// [`BackendPhaseStateOwnership`] via [`Self::with_state_ownership`] /
/// [`Self::pair_with_ownership`] to advance to Eligible after fail-closed
/// layout validation. Ready work stays refused until an execute adapter exists.
#[derive(Clone)]
pub struct BackendOwnedPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
    ownership: Arc<dyn BackendPhaseStateOwnership>,
    health: PhaseExecutorHealth,
}

impl fmt::Debug for BackendOwnedPhaseExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendOwnedPhaseExecutor")
            .field("capabilities", &self.capabilities)
            .field("health", &self.health)
            .field("ownership_empty", &self.ownership.is_empty())
            .finish()
    }
}

impl BackendOwnedPhaseExecutor {
    /// Build buffered-host loopback transfer + backend-owned phase with Empty
    /// ownership (Unavailable).
    ///
    /// The transfer may become Ready; this executor stays Unavailable until a
    /// non-Empty ownership surface is bound, and never advertises model-semantic
    /// P/D readiness from registration alone.
    pub fn paired_for_profile(
        profile: &ServingExecutionProfile,
    ) -> Result<(Arc<BufferedHostLoopbackStateTransfer>, Arc<Self>)> {
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(profile)?);
        let executor = Arc::new(Self::pair_with(profile, Arc::clone(&transfer))?);
        Ok((transfer, executor))
    }

    /// Bind one backend-owned phase executor to an already-constructed loopback
    /// transfer under Empty ownership. Refuses a transfer bound to a different
    /// profile digest.
    pub fn pair_with(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
    ) -> Result<Self> {
        Self::pair_with_ownership(profile, transfer, Arc::new(EmptyBackendPhaseStateOwnership))
    }

    /// Bind loopback transfer + ownership. Empty ownership → Unavailable;
    /// matching layout → Eligible; mismatch → fail closed.
    pub fn pair_with_ownership(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
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
        Self::with_state_ownership(profile, ownership)
    }

    /// Bind one backend-owned phase executor to a buffered-host loopback profile
    /// under Empty ownership without constructing the transfer.
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        Self::with_state_ownership(profile, Arc::new(EmptyBackendPhaseStateOwnership))
    }

    /// Bind ownership against the immutable profile.
    ///
    /// Validates `state_layout_sha256` (and optional related digests) fail-closed
    /// before becoming Eligible. Empty ownership stays Unavailable. Eligible
    /// still blocks Ready prepare/execute.
    pub fn with_state_ownership(
        profile: &ServingExecutionProfile,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
    ) -> Result<Self> {
        require_buffered_host_loopback(profile)?;
        let health = bind_backend_phase_state_ownership(ownership.as_ref(), profile)?;
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
            ownership,
            health,
        })
    }

    /// Opaque ownership surface currently bound to this executor.
    pub fn ownership(&self) -> &dyn BackendPhaseStateOwnership {
        self.ownership.as_ref()
    }
}

#[async_trait]
impl ServingPhaseExecutor for BackendOwnedPhaseExecutor {
    fn capabilities(&self) -> PhaseExecutorCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> PhaseExecutorHealth {
        self.health
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
        // Layout registration (Eligible) and transport success must never become
        // a Ready reservation without a real execute adapter path.
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn execute(
        &self,
        _command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        // Never claim cache hit or decode success from layout binding alone.
        PhaseDecision::retryable_unavailable(RetryableUnavailableReason::ExecutorUnavailable, None)
    }

    async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
        match self.health {
            PhaseExecutorHealth::Unavailable => Err(unavailable()),
            PhaseExecutorHealth::Eligible
            | PhaseExecutorHealth::Ready
            | PhaseExecutorHealth::Degraded => Err(ready_blocked()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::{
        validate_injected_production_adapters, BackendPhaseStateOwnership,
        BoundedStateTransferService, DirectDeviceMemoryPullPhaseExecutor, DisaggregatedServingRole,
        DistributedServingRuntime, EmptyServingPhaseExecutor, ModelStateHandle, PhaseRequest,
        PhaseSessionPoolMode, PhaseWeightCacheMode, PrefillDecodeExecutionProfile,
        ServingCompositionPhaseExecutor, ServingCompositionTransport, ServingPhase,
        ServingPrivacyMode, StateKind, StateTransferProtocol,
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

    struct MatchingOwnership {
        layout: String,
    }

    impl BackendPhaseStateOwnership for MatchingOwnership {
        fn state_layout_sha256(&self) -> Option<&str> {
            Some(&self.layout)
        }

        fn import_opaque_state(&self, _opaque: &[u8]) -> Result<ModelStateHandle> {
            ModelStateHandle::new("matching-imported")
        }

        fn export_opaque_state(&self, _handle: &ModelStateHandle) -> Result<Vec<u8>> {
            Ok(b"matching".to_vec())
        }
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
    async fn unregistered_empty_ownership_stays_unavailable() {
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
        assert!(executor.ownership().is_empty());
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
    fn mismatched_layout_registration_fails_closed() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let ownership: Arc<dyn BackendPhaseStateOwnership> = Arc::new(MatchingOwnership {
            layout: digest('9'),
        });
        let err = BackendOwnedPhaseExecutor::with_state_ownership(&profile, ownership).unwrap_err();
        assert!(err.to_string().contains("state_layout_sha256"));
        assert!(err.to_string().contains("does not match"));
    }

    #[tokio::test]
    async fn matching_layout_is_eligible_but_not_ready_decode() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let ownership: Arc<dyn BackendPhaseStateOwnership> = Arc::new(MatchingOwnership {
            layout: digest('5'),
        });
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let executor = BackendOwnedPhaseExecutor::pair_with_ownership(
            &profile,
            Arc::clone(&transfer),
            ownership,
        )
        .unwrap();

        assert_eq!(executor.health(), PhaseExecutorHealth::Eligible);
        assert!(!executor.health().accepts_work());
        assert!(!executor.ownership().is_empty());
        assert_eq!(
            executor.ownership().state_layout_sha256(),
            Some(digest('5').as_str())
        );

        // Opaque hooks may succeed; that still is not Ready decode.
        let imported = executor.ownership().import_opaque_state(b"opaque").unwrap();
        assert_eq!(imported.as_str(), "matching-imported");

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
            PhaseDecision::Ready(_) => {
                panic!("matching layout registration alone must never emit Ready decode")
            }
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
        assert!(err.to_string().contains("Eligible"));
        assert!(err.to_string().contains("execute adapter"));

        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, Arc::new(executor)).unwrap();
        assert!(!runtime.accepts_work());
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
