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
//! health to [`PhaseExecutorHealth::Eligible`]. Default
//! [`EmptyBackendPhaseExecution`] keeps Eligible from becoming Ready.
//! Binding a Ready-capable [`BackendPhaseExecution`] (ACL
//! `phase_execution = pending` ? [`PendingBackendPhaseExecution`], or a
//! concrete backend) advances Eligible to [`PhaseExecutorHealth::Ready`] and
//! delegates prepare/execute. Pending unlocks Ready health only;
//! prepare/execute still fail closed until a real backend implementor exists.
//! Matching layout registration alone is not decode success and does not
//! invent model-semantic P/D.
//!
//! [`super::ProfileBoundBackendPhaseStateOwnership`] is the honest interim
//! product surface: it mirrors closed profile digests (including the closed
//! backend artifact digest) without owning KV or inventing layout semantics.
//! Opaque import/export on that surface fail closed. It is not a real
//! llama.cpp / picolm adapter.
//!
//! Pairs only with [`BufferedHostLoopbackStateTransfer`] (or refuses wrong
//! transport). Transfer completion alone never yields cache-hit or decode
//! success.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::{PowerError, Result};

use super::backend_phase_execution::{
    bind_backend_phase_execution, BackendPhaseExecution, EmptyBackendPhaseExecution,
    PendingBackendPhaseExecution,
};
use super::backend_phase_state_ownership::{
    bind_backend_phase_state_ownership, BackendPhaseStateOwnership,
    EmptyBackendPhaseStateOwnership, ProfileBoundBackendPhaseStateOwnership,
};
use super::{
    AbortPhaseExecution, AdapterProvisionState, BufferedHostLoopbackStateTransfer,
    ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PhaseExecutorCapabilities,
    PhaseExecutorHealth, PreparePhaseExecution, PreparedPhaseExecution, ProductionAdapterContract,
    RetryableUnavailableReason, ServingCompositionPhaseExecution, ServingCompositionStateOwnership,
    ServingExecutionProfile, ServingPhaseExecutor, ServingPrivacyMode, StateTransferProtocol,
    StateTransferService,
};

fn unavailable() -> PowerError {
    PowerError::BackendNotAvailable(
        "BackendOwnedPhaseExecutor is Unavailable until a real state-layout and KV ownership adapter is bound"
            .to_string(),
    )
}

/// Resolve ACL/composition ownership for a backend-owned phase profile.
///
/// Absent `state_ownership` keeps Empty (Unavailable). `profile-bound`
/// installs [`ProfileBoundBackendPhaseStateOwnership`] (Eligible after digests
/// match).
fn composition_ownership(
    profile: &ServingExecutionProfile,
) -> Result<Arc<dyn BackendPhaseStateOwnership>> {
    match profile.composition_state_ownership() {
        None => Ok(Arc::new(EmptyBackendPhaseStateOwnership)),
        Some(ServingCompositionStateOwnership::ProfileBound) => Ok(Arc::new(
            ProfileBoundBackendPhaseStateOwnership::for_profile(profile)?,
        )),
    }
}

/// Resolve ACL/composition phase execution for a backend-owned phase profile.
///
/// Absent `phase_execution` keeps Empty (Eligible refuses Ready). `pending`
/// installs [`PendingBackendPhaseExecution`] (Eligible ? Ready health; work
/// still fail-closed).
fn composition_execution(
    profile: &ServingExecutionProfile,
) -> Result<Arc<dyn BackendPhaseExecution>> {
    match profile.composition_phase_execution() {
        None => Ok(Arc::new(EmptyBackendPhaseExecution)),
        Some(ServingCompositionPhaseExecution::Pending) => {
            Ok(Arc::new(PendingBackendPhaseExecution))
        }
    }
}

fn ready_blocked() -> PowerError {
    PowerError::BackendNotAvailable(
        "BackendOwnedPhaseExecutor is Eligible after state-layout binding but Ready prepare/execute remains blocked until a Ready-capable execute adapter is bound"
            .to_string(),
    )
}

fn require_buffered_host_loopback(
    profile: &ServingExecutionProfile,
) -> Result<&super::PrefillDecodeExecutionProfile> {
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
/// build both ports together (Empty ownership ? Unavailable). Bind a concrete
/// [`BackendPhaseStateOwnership`] via [`Self::with_state_ownership`] /
/// [`Self::pair_with_ownership`] to advance to Eligible after fail-closed
/// layout validation. Bind a Ready-capable [`BackendPhaseExecution`] (or ACL
/// `phase_execution = pending`) to advance Eligible to Ready and delegate
/// prepare/execute. Empty execution keeps Eligible refusing Ready work.
#[derive(Clone)]
pub struct BackendOwnedPhaseExecutor {
    capabilities: PhaseExecutorCapabilities,
    ownership: Arc<dyn BackendPhaseStateOwnership>,
    execution: Arc<dyn BackendPhaseExecution>,
    health: PhaseExecutorHealth,
}

impl fmt::Debug for BackendOwnedPhaseExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BackendOwnedPhaseExecutor")
            .field("capabilities", &self.capabilities)
            .field("health", &self.health)
            .field("ownership_empty", &self.ownership.is_empty())
            .field("execution_empty", &self.execution.is_empty())
            .field(
                "execution_can_produce_ready",
                &self.execution.can_produce_ready(),
            )
            .finish()
    }
}

impl BackendOwnedPhaseExecutor {
    /// Build buffered-host loopback transfer + backend-owned phase.
    ///
    /// Honors ACL `state_ownership`: absent ? Empty (Unavailable);
    /// `profile-bound` ? Eligible after fail-closed digest bind. Honors ACL
    /// `phase_execution`: absent ? Empty (Eligible refuses Ready);
    /// `pending` ? Ready health after Eligible. The transfer may become Ready;
    /// this executor never advertises model-semantic P/D readiness from
    /// registration or pending unlock alone.
    pub fn paired_for_profile(
        profile: &ServingExecutionProfile,
    ) -> Result<(Arc<BufferedHostLoopbackStateTransfer>, Arc<Self>)> {
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(profile)?);
        let ownership = composition_ownership(profile)?;
        let execution = composition_execution(profile)?;
        let executor = Arc::new(Self::pair_with_ownership_and_execution(
            profile,
            Arc::clone(&transfer),
            ownership,
            execution,
        )?);
        Ok((transfer, executor))
    }

    /// Bind one backend-owned phase executor to an already-constructed loopback
    /// transfer under Empty ownership and Empty execution. Refuses a transfer
    /// bound to a different profile digest.
    pub fn pair_with(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
    ) -> Result<Self> {
        Self::pair_with_ownership_and_execution(
            profile,
            transfer,
            Arc::new(EmptyBackendPhaseStateOwnership),
            Arc::new(EmptyBackendPhaseExecution),
        )
    }

    /// Bind loopback transfer + ownership under Empty execution.
    /// Empty ownership ? Unavailable; matching layout ? Eligible; mismatch ?
    /// fail closed. Eligible still refuses Ready without a Ready-capable
    /// execution adapter.
    pub fn pair_with_ownership(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
    ) -> Result<Self> {
        Self::pair_with_ownership_and_execution(
            profile,
            transfer,
            ownership,
            Arc::new(EmptyBackendPhaseExecution),
        )
    }

    /// Bind loopback transfer + ownership + execution.
    ///
    /// Health is Unavailable (Empty ownership), Eligible (matching ownership,
    /// Empty/non-producing execution), or Ready (Eligible ownership +
    /// `can_produce_ready` execution).
    pub fn pair_with_ownership_and_execution(
        profile: &ServingExecutionProfile,
        transfer: Arc<BufferedHostLoopbackStateTransfer>,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
        execution: Arc<dyn BackendPhaseExecution>,
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
        Self::with_state_ownership_and_execution(profile, ownership, execution)
    }

    /// Bind one backend-owned phase executor for the profile, honoring ACL
    /// `state_ownership` and `phase_execution` (Empty when absent).
    pub fn for_profile(profile: &ServingExecutionProfile) -> Result<Self> {
        Self::with_state_ownership_and_execution(
            profile,
            composition_ownership(profile)?,
            composition_execution(profile)?,
        )
    }

    /// Bind ownership against the immutable profile under Empty execution.
    ///
    /// Validates `state_layout_sha256` (and optional related digests) fail-closed
    /// before becoming Eligible. Empty ownership stays Unavailable. Eligible
    /// still blocks Ready prepare/execute without a Ready-capable execution
    /// adapter.
    pub fn with_state_ownership(
        profile: &ServingExecutionProfile,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
    ) -> Result<Self> {
        Self::with_state_ownership_and_execution(
            profile,
            ownership,
            Arc::new(EmptyBackendPhaseExecution),
        )
    }

    /// Bind ownership + execution against the immutable profile.
    pub fn with_state_ownership_and_execution(
        profile: &ServingExecutionProfile,
        ownership: Arc<dyn BackendPhaseStateOwnership>,
        execution: Arc<dyn BackendPhaseExecution>,
    ) -> Result<Self> {
        require_buffered_host_loopback(profile)?;
        let ownership_health = bind_backend_phase_state_ownership(ownership.as_ref(), profile)?;
        let health = bind_backend_phase_execution(ownership_health, execution.as_ref());
        Ok(Self {
            capabilities: PhaseExecutorCapabilities::for_profile(profile)?,
            ownership,
            execution,
            health,
        })
    }

    /// Opaque ownership surface currently bound to this executor.
    pub fn ownership(&self) -> &dyn BackendPhaseStateOwnership {
        self.ownership.as_ref()
    }

    /// Prepare/execute surface currently bound to this executor.
    pub fn execution(&self) -> &dyn BackendPhaseExecution {
        self.execution.as_ref()
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
        command: PreparePhaseExecution,
    ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
        match self.health {
            PhaseExecutorHealth::Unavailable => PhaseDecision::retryable_unavailable(
                RetryableUnavailableReason::ExecutorUnavailable,
                None,
            ),
            PhaseExecutorHealth::Eligible => {
                // Layout registration alone must never become a Ready reservation.
                PhaseDecision::retryable_unavailable(
                    RetryableUnavailableReason::ExecutorUnavailable,
                    None,
                )
            }
            PhaseExecutorHealth::Ready | PhaseExecutorHealth::Degraded => {
                self.execution.prepare(command).await
            }
        }
    }

    async fn execute(
        &self,
        command: ExecutePhaseExecution,
    ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
        match self.health {
            PhaseExecutorHealth::Unavailable => PhaseDecision::retryable_unavailable(
                RetryableUnavailableReason::ExecutorUnavailable,
                None,
            ),
            PhaseExecutorHealth::Eligible => {
                // Never claim cache hit or decode success from layout binding alone.
                PhaseDecision::retryable_unavailable(
                    RetryableUnavailableReason::ExecutorUnavailable,
                    None,
                )
            }
            PhaseExecutorHealth::Ready | PhaseExecutorHealth::Degraded => {
                self.execution.execute(command).await
            }
        }
    }

    async fn abort(&self, command: AbortPhaseExecution) -> Result<()> {
        match self.health {
            PhaseExecutorHealth::Unavailable => Err(unavailable()),
            PhaseExecutorHealth::Eligible => Err(ready_blocked()),
            PhaseExecutorHealth::Ready | PhaseExecutorHealth::Degraded => {
                self.execution.abort(command).await
            }
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
        ProfileBoundBackendPhaseStateOwnership, ServingCompositionPhaseExecutor,
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
        buffered_profile_with_options(transport, phase_executor, None, None)
    }

    fn buffered_profile_with_options(
        transport: Option<ServingCompositionTransport>,
        phase_executor: Option<ServingCompositionPhaseExecutor>,
        state_ownership: Option<crate::serving::ServingCompositionStateOwnership>,
        phase_execution: Option<ServingCompositionPhaseExecution>,
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
            state_ownership,
            phase_execution,
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
            state_ownership: None,
            phase_execution: None,
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

    #[tokio::test]
    async fn profile_bound_ownership_is_eligible_without_ready_or_kv() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let ownership: Arc<dyn BackendPhaseStateOwnership> =
            Arc::new(ProfileBoundBackendPhaseStateOwnership::for_profile(&profile).unwrap());
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let executor = BackendOwnedPhaseExecutor::pair_with_ownership(
            &profile,
            Arc::clone(&transfer),
            ownership,
        )
        .unwrap();

        assert_eq!(executor.health(), PhaseExecutorHealth::Eligible);
        assert!(!executor.health().accepts_work());
        assert_eq!(
            executor.ownership().state_layout_sha256(),
            Some(digest('5').as_str())
        );
        assert_eq!(
            executor.ownership().backend_sha256(),
            Some(digest('2').as_str())
        );
        assert!(executor
            .ownership()
            .import_opaque_state(b"opaque")
            .unwrap_err()
            .to_string()
            .contains("ProfileBoundBackendPhaseStateOwnership"));

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
                panic!("profile-bound digest ownership must never emit Ready decode")
            }
            PhaseDecision::Recompute { .. } => {
                panic!("expected retryable unavailable, got Recompute")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected retryable unavailable, got TerminalFailure")
            }
        }

        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, Arc::new(executor)).unwrap();
        assert!(!runtime.accepts_work());
    }

    #[test]
    fn paired_for_profile_honors_profile_bound_state_ownership_acl() {
        use crate::serving::ServingCompositionStateOwnership;

        let profile = buffered_profile_with_options(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
            Some(ServingCompositionStateOwnership::ProfileBound),
            None,
        );
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(transfer.health(), crate::serving::TransferHealth::Ready);
        assert_eq!(executor.health(), PhaseExecutorHealth::Eligible);
        assert!(!executor.health().accepts_work());
        assert!(!executor.ownership().is_empty());
        assert!(executor.execution().is_empty());
        assert!(!profile.may_advertise_prefill_decode());
    }

    #[test]
    fn paired_for_profile_honors_pending_phase_execution_acl() {
        use crate::serving::ServingCompositionStateOwnership;

        let profile = buffered_profile_with_options(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
            Some(ServingCompositionStateOwnership::ProfileBound),
            Some(ServingCompositionPhaseExecution::Pending),
        );
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(transfer.health(), crate::serving::TransferHealth::Ready);
        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);
        assert!(executor.health().accepts_work());
        assert!(executor.execution().can_produce_ready());
        // Backend-owned composition still suppresses worker P/D advertising.
        assert!(!profile.may_advertise_prefill_decode());
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile, bounded, executor).unwrap();
        assert!(!runtime.accepts_work());
    }

    #[tokio::test]
    async fn pending_execution_unlocks_ready_health_but_work_fails_closed() {
        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let ownership: Arc<dyn BackendPhaseStateOwnership> = Arc::new(MatchingOwnership {
            layout: digest('5'),
        });
        let execution: Arc<dyn BackendPhaseExecution> = Arc::new(PendingBackendPhaseExecution);
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let executor = BackendOwnedPhaseExecutor::pair_with_ownership_and_execution(
            &profile,
            Arc::clone(&transfer),
            ownership,
            execution,
        )
        .unwrap();

        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);
        assert!(executor.health().accepts_work());
        assert!(!executor.execution().is_empty());

        let err = match executor
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
        {
            Ok(_) => panic!("pending prepare must fail closed after Ready unlock"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("PendingBackendPhaseExecution"));
        assert!(err.to_string().contains("unlocks Ready health only"));

        // Pending without Eligible ownership stays Unavailable.
        let unavailable = BackendOwnedPhaseExecutor::with_state_ownership_and_execution(
            &profile,
            Arc::new(EmptyBackendPhaseStateOwnership),
            Arc::new(PendingBackendPhaseExecution),
        )
        .unwrap();
        assert_eq!(unavailable.health(), PhaseExecutorHealth::Unavailable);
        assert!(!unavailable.health().accepts_work());
    }

    #[tokio::test]
    async fn ready_capable_execution_can_emit_ready_prepare_decision() {
        struct ReadyProbeExecution {
            profile_sha256: String,
        }

        #[async_trait]
        impl BackendPhaseExecution for ReadyProbeExecution {
            fn can_produce_ready(&self) -> bool {
                true
            }

            async fn prepare(
                &self,
                command: PreparePhaseExecution,
            ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
                let execution = crate::serving::PhaseExecutionHandle::new(format!(
                    "ready-probe:{}",
                    command.execution_id
                ))?;
                let prepared = crate::serving::PreparedDecodePhase::new(
                    command.execution_id,
                    command.local_worker_epoch,
                    self.profile_sha256.clone(),
                    execution,
                    ModelStateHandle::new(format!("destination:{}", command.execution_id))?,
                    crate::serving::StateTransferBinding {
                        model_sha256: digest('1'),
                        execution_sha256: digest('3'),
                        layout_sha256: digest('5'),
                        state_kind: StateKind::KvCache,
                        token_count: 1,
                        state_bytes: 8,
                    },
                    command.expires_at,
                )?;
                Ok(PhaseDecision::ready(PreparedPhaseExecution::Decode(
                    prepared,
                )))
            }

            async fn execute(
                &self,
                _command: ExecutePhaseExecution,
            ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
                PhaseDecision::retryable_unavailable(
                    RetryableUnavailableReason::ExecutorUnavailable,
                    None,
                )
            }

            async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
                Ok(())
            }
        }

        let profile = buffered_profile(
            Some(ServingCompositionTransport::BufferedHostLoopback),
            Some(ServingCompositionPhaseExecutor::BackendOwned),
        );
        let ownership: Arc<dyn BackendPhaseStateOwnership> = Arc::new(MatchingOwnership {
            layout: digest('5'),
        });
        let transfer = Arc::new(BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap());
        let executor = BackendOwnedPhaseExecutor::pair_with_ownership_and_execution(
            &profile,
            transfer,
            ownership,
            Arc::new(ReadyProbeExecution {
                profile_sha256: profile.sha256().unwrap(),
            }),
        )
        .unwrap();
        assert_eq!(executor.health(), PhaseExecutorHealth::Ready);

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
            PhaseDecision::Ready(PreparedPhaseExecution::Decode(_)) => {}
            PhaseDecision::Ready(_) => {
                panic!("expected Ready decode reservation from bound execution")
            }
            PhaseDecision::Recompute { .. } => {
                panic!("expected Ready decode, got Recompute")
            }
            PhaseDecision::RetryableUnavailable { .. } => {
                panic!("expected Ready decode, got RetryableUnavailable")
            }
            PhaseDecision::TerminalFailure { .. } => {
                panic!("expected Ready decode, got TerminalFailure")
            }
        }
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
