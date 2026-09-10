//! Process-local collector for the public worker observation contract.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::serving::{
    AdmissionObservation, DistributedServingRuntime, PromptCacheObservation, ServingPhase,
    TransferHealth, WorkerCapabilities, WorkerObservation, WORKER_OBSERVATION_SCHEMA,
};

use super::state::AppState;

#[derive(Clone)]
pub(super) struct WorkerObservationSource {
    inner: Arc<WorkerObservationSourceInner>,
}

struct WorkerObservationSourceInner {
    worker_epoch: Uuid,
    observation_generation: AtomicU64,
}

impl WorkerObservationSource {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(WorkerObservationSourceInner {
                worker_epoch: Uuid::new_v4(),
                observation_generation: AtomicU64::new(0),
            }),
        }
    }

    pub(super) fn observe(
        &self,
        state: &AppState,
        observed_at: DateTime<Utc>,
    ) -> WorkerObservation {
        let generation = self
            .inner
            .observation_generation
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_add(1))
            })
            .unwrap_or(u64::MAX)
            .saturating_add(1);
        let supporting_backends =
            u64::try_from(state.backends.prompt_cache_backend_names().len()).unwrap_or(u64::MAX);
        let (entries, capacity) = state.backends.prompt_cache_metrics().into_iter().fold(
            (0_u64, 0_u64),
            |(entries, capacity), (_, snapshot)| {
                (
                    entries.saturating_add(snapshot.entries),
                    capacity.saturating_add(snapshot.capacity),
                )
            },
        );
        let pressure_basis_points = cache_pressure_basis_points(entries, capacity);
        let prompt_cache_supported = supporting_backends > 0;
        let ttl = chrono::Duration::seconds(
            i64::try_from(state.config.worker_observation_ttl_seconds).unwrap_or(i64::MAX),
        );
        let serving = serving_observation(state);

        WorkerObservation {
            schema: WORKER_OBSERVATION_SCHEMA.to_string(),
            worker_epoch: self.inner.worker_epoch,
            observation_generation: generation,
            observed_at,
            expires_at: observed_at + ttl,
            capabilities: WorkerCapabilities {
                phases: serving.phases,
                prompt_cache: prompt_cache_supported,
                state_transfer: serving.state_transfer,
            },
            ready_phases: serving.ready_phases,
            admission: admission_observation(state),
            prompt_cache: PromptCacheObservation {
                supported: prompt_cache_supported,
                entries,
                capacity,
                pressure_basis_points,
            },
            transfer_health: serving.transfer_health,
        }
    }

    pub(super) fn worker_epoch(&self) -> Uuid {
        self.inner.worker_epoch
    }
}

struct ServingObservation {
    phases: Vec<ServingPhase>,
    ready_phases: Vec<ServingPhase>,
    state_transfer: bool,
    transfer_health: TransferHealth,
}

fn serving_observation(state: &AppState) -> ServingObservation {
    let profile = &state.config.serving_execution;
    if profile.is_aggregated() {
        return ServingObservation {
            phases: vec![ServingPhase::Aggregated],
            ready_phases: vec![ServingPhase::Aggregated],
            state_transfer: false,
            transfer_health: TransferHealth::Unsupported,
        };
    }

    let Some(runtime) = matching_distributed_runtime(state) else {
        return unsupported_distributed_observation();
    };
    let transfer_health = runtime.transfer_health();
    let phase = profile.phase();
    let ready_phases = if state.auth.is_some() && runtime.accepts_work() {
        vec![phase]
    } else {
        Vec::new()
    };
    ServingObservation {
        phases: vec![phase],
        ready_phases,
        state_transfer: true,
        transfer_health,
    }
}

/// Prefer the shared fail-fast P/D AdmissionController when a matching runtime
/// is composed. Do not invent a second capacity story from the HTTP limiter.
fn admission_observation(state: &AppState) -> AdmissionObservation {
    if let Some(runtime) = matching_distributed_runtime(state) {
        let phase = runtime.admission_snapshot();
        let transfer = runtime.transfer_admission_snapshot();
        // Construction already refuses mismatched limits; observe still requires
        // the same ACL fail-fast shape so a drifted process cannot project a
        // waiting queue or alternate active bound.
        if phase.active_limit == transfer.active_limit
            && phase.waiting_limit == Some(0)
            && transfer.waiting_limit == Some(0)
        {
            return AdmissionObservation {
                active_limit: phase
                    .active_limit
                    .map(|limit| u64::try_from(limit).unwrap_or(u64::MAX)),
                active: u64::try_from(phase.active).unwrap_or(u64::MAX),
                waiting: u64::try_from(phase.waiting).unwrap_or(u64::MAX),
            };
        }
    }

    AdmissionObservation {
        active_limit: (state.config.max_concurrent_requests > 0)
            .then_some(state.config.max_concurrent_requests),
        active: state.metrics.running_requests(),
        waiting: state.metrics.waiting_requests(),
    }
}

fn matching_distributed_runtime(state: &AppState) -> Option<&Arc<DistributedServingRuntime>> {
    let profile = &state.config.serving_execution;
    if profile.is_aggregated() {
        return None;
    }
    let runtime = state.distributed_serving.as_ref()?;
    if runtime.profile() != profile
        || matches!(runtime.transfer_health(), TransferHealth::Unsupported)
    {
        return None;
    }
    Some(runtime)
}

fn unsupported_distributed_observation() -> ServingObservation {
    ServingObservation {
        phases: Vec::new(),
        ready_phases: Vec::new(),
        state_transfer: false,
        transfer_health: TransferHealth::Unsupported,
    }
}

fn cache_pressure_basis_points(entries: u64, capacity: u64) -> u16 {
    if capacity == 0 {
        return if entries == 0 { 0 } else { 10_000 };
    }
    let pressure = u128::from(entries)
        .saturating_mul(10_000)
        .checked_div(u128::from(capacity))
        .unwrap_or(10_000)
        .min(10_000);
    u16::try_from(pressure).unwrap_or(10_000)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::error::{PowerError, Result};
    use crate::model::registry::ModelRegistry;
    use crate::server::auth::ApiKeyAuth;
    use crate::serving::{
        AbortPhaseExecution, AbortStateTransfer, BoundedStateTransferService, ConsumeStateTransfer,
        DisaggregatedServingRole, DistributedServingRuntime, ExecutePhaseExecution, PhaseDecision,
        PhaseExecutionOutput, PhaseExecutorCapabilities, PhaseExecutorHealth, PhaseSessionPoolMode,
        PhaseWeightCacheMode, PrefillDecodeExecutionProfile, PreparePhaseExecution,
        PrepareStateTransfer, PreparedPhaseExecution, PublishStateTransfer,
        ServingCompositionTransport, ServingExecutionProfile, ServingPhaseExecutor,
        ServingPrivacyMode, StateKind, StateTransferCapabilities, StateTransferProtocol,
        StateTransferReceipt, StateTransferService, StateTransferSource, StateTransferTarget,
    };

    use super::{cache_pressure_basis_points, AppState, ServingPhase, TransferHealth};

    struct TestStateTransferService {
        health: TransferHealth,
        capabilities: StateTransferCapabilities,
    }

    struct TestPhaseExecutor {
        health: PhaseExecutorHealth,
        capabilities: PhaseExecutorCapabilities,
    }

    fn execution_profile() -> ServingExecutionProfile {
        execution_profile_with_generation(7)
    }

    fn execution_profile_with_generation(generation: u64) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "llama.cpp".to_string(),
            backend_sha256: "2".repeat(64),
            execution_sha256: "3".repeat(64),
            device_sha256: "4".repeat(64),
            layout_sha256: "5".repeat(64),
            peer_set_sha256: "6".repeat(64),
            generation,
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
            transport: None,
            phase_executor: None,
            state_ownership: None,
        })
        .unwrap()
    }

    #[async_trait]
    impl StateTransferService for TestStateTransferService {
        fn capabilities(&self) -> StateTransferCapabilities {
            self.capabilities.clone()
        }

        fn health(&self) -> TransferHealth {
            self.health
        }

        async fn prepare_destination(
            &self,
            _command: PrepareStateTransfer,
        ) -> Result<StateTransferTarget> {
            Err(PowerError::BackendNotAvailable("test adapter".to_string()))
        }

        async fn publish_source(
            &self,
            _command: PublishStateTransfer,
        ) -> Result<StateTransferSource> {
            Err(PowerError::BackendNotAvailable("test adapter".to_string()))
        }

        async fn consume_source(
            &self,
            _command: ConsumeStateTransfer,
        ) -> Result<StateTransferReceipt> {
            Err(PowerError::BackendNotAvailable("test adapter".to_string()))
        }

        async fn abort(&self, _command: AbortStateTransfer) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl ServingPhaseExecutor for TestPhaseExecutor {
        fn capabilities(&self) -> PhaseExecutorCapabilities {
            self.capabilities.clone()
        }

        fn health(&self) -> PhaseExecutorHealth {
            self.health
        }

        async fn prepare(
            &self,
            _command: PreparePhaseExecution,
        ) -> Result<PhaseDecision<PreparedPhaseExecution>> {
            Err(PowerError::BackendNotAvailable(
                "test phase executor".to_string(),
            ))
        }

        async fn execute(
            &self,
            _command: ExecutePhaseExecution,
        ) -> Result<PhaseDecision<PhaseExecutionOutput>> {
            Err(PowerError::BackendNotAvailable(
                "test phase executor".to_string(),
            ))
        }

        async fn abort(&self, _command: AbortPhaseExecution) -> Result<()> {
            Ok(())
        }
    }

    fn state_with_services(
        transfer_health: TransferHealth,
        executor_health: PhaseExecutorHealth,
    ) -> AppState {
        state_with_services_and_limits(transfer_health, executor_health, 2, 0)
    }

    fn state_with_services_and_limits(
        transfer_health: TransferHealth,
        executor_health: PhaseExecutorHealth,
        max_inflight_transfers: u32,
        max_concurrent_requests: u64,
    ) -> AppState {
        let profile = execution_profile_with_inflight(max_inflight_transfers);
        let service = TestStateTransferService {
            health: transfer_health,
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers,
            },
        };
        let executor = TestPhaseExecutor {
            health: executor_health,
            capabilities: PhaseExecutorCapabilities::for_profile(&profile).unwrap(),
        };
        let config = PowerConfig {
            serving_execution: profile.clone(),
            max_concurrent_requests,
            ..PowerConfig::default()
        };
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        let transfer = BoundedStateTransferService::new(
            profile.clone(),
            state.worker_epoch(),
            Arc::new(service),
        )
        .unwrap();
        let runtime =
            DistributedServingRuntime::new(profile, Arc::new(transfer), Arc::new(executor))
                .unwrap();
        state
            .with_distributed_serving(Arc::new(runtime))
            .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])))
    }

    fn execution_profile_with_inflight(max_inflight_transfers: u32) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "llama.cpp".to_string(),
            backend_sha256: "2".repeat(64),
            execution_sha256: "3".repeat(64),
            device_sha256: "4".repeat(64),
            layout_sha256: "5".repeat(64),
            peer_set_sha256: "6".repeat(64),
            generation: 7,
            protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
            state_kind: StateKind::KvCache,
            max_state_bytes: 1024,
            max_inflight_transfers,
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

    #[test]
    fn cache_pressure_is_bounded_and_handles_zero_capacity() {
        assert_eq!(cache_pressure_basis_points(0, 0), 0);
        assert_eq!(cache_pressure_basis_points(1, 0), 10_000);
        assert_eq!(cache_pressure_basis_points(1, 4), 2_500);
        assert_eq!(cache_pressure_basis_points(8, 4), 10_000);
    }

    #[test]
    fn ready_composed_services_project_the_exact_decode_phase() {
        let state = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        let observation = state.worker_observation();

        assert_eq!(observation.worker_epoch, state.worker_epoch());
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert_eq!(observation.ready_phases, [ServingPhase::Decode]);
        assert!(observation.capabilities.state_transfer);
        assert_eq!(observation.transfer_health, TransferHealth::Ready);
    }

    #[test]
    fn unavailable_transfer_service_keeps_transport_capability() {
        let observation =
            state_with_services(TransferHealth::Unavailable, PhaseExecutorHealth::Ready)
                .worker_observation();

        assert!(observation.capabilities.state_transfer);
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
        assert_eq!(observation.transfer_health, TransferHealth::Unavailable);
    }

    #[test]
    fn direct_device_memory_pull_product_port_never_lists_ready_phases() {
        let profile = ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
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
            transport: Some(ServingCompositionTransport::DirectDeviceMemoryPull),
            phase_executor: None,
            state_ownership: None,
        })
        .unwrap();
        let (transfer, executor) =
            crate::serving::DirectDeviceMemoryPullPhaseExecutor::paired_for_profile(&profile)
                .unwrap();
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        );
        let bounded =
            BoundedStateTransferService::new(profile.clone(), state.worker_epoch(), transfer)
                .unwrap();
        let runtime = DistributedServingRuntime::new(profile, Arc::new(bounded), executor).unwrap();
        let state = state
            .with_distributed_serving(Arc::new(runtime))
            .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])));
        let observation = state.worker_observation();

        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.capabilities.state_transfer);
        assert!(observation.ready_phases.is_empty());
        assert_eq!(observation.transfer_health, TransferHealth::Unavailable);
        assert!(!state.distributed_serving.as_ref().unwrap().accepts_work());
    }

    #[test]
    fn unavailable_phase_executor_keeps_capability_but_suppresses_readiness() {
        let observation =
            state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Unavailable)
                .worker_observation();

        assert!(observation.capabilities.state_transfer);
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
    }

    #[test]
    fn eligible_phase_executor_never_lists_ready_phases() {
        // Gap verification: Eligible (accepts_work false) must not advertise P/D.
        let observation = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Eligible)
            .worker_observation();

        assert!(observation.capabilities.state_transfer);
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
        assert!(!observation
            .ready_phases
            .iter()
            .any(|phase| matches!(phase, ServingPhase::Prefill | ServingPhase::Decode)));
    }

    #[test]
    fn backend_owned_profile_bound_composition_suppresses_ready_phases() {
        use crate::serving::{
            BackendOwnedPhaseExecutor, ServingCompositionPhaseExecutor,
            ServingCompositionStateOwnership, ServingCompositionTransport,
        };

        let profile = ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "backend-owned".to_string(),
            backend_sha256: "2".repeat(64),
            execution_sha256: "3".repeat(64),
            device_sha256: "4".repeat(64),
            layout_sha256: "5".repeat(64),
            peer_set_sha256: "6".repeat(64),
            generation: 7,
            protocol: StateTransferProtocol::BufferedHostMemoryPullV1,
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
            transport: Some(ServingCompositionTransport::BufferedHostLoopback),
            phase_executor: Some(ServingCompositionPhaseExecutor::BackendOwned),
            state_ownership: Some(ServingCompositionStateOwnership::ProfileBound),
        })
        .unwrap();
        let (transfer, executor) = BackendOwnedPhaseExecutor::paired_for_profile(&profile).unwrap();
        assert_eq!(executor.health(), PhaseExecutorHealth::Eligible);
        assert!(!executor.health().accepts_work());
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile.clone(), bounded, executor).unwrap();
        assert!(!runtime.accepts_work());

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                serving_execution: profile,
                api_keys: vec!["service-key".to_string()],
                ..PowerConfig::default()
            }),
        )
        .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])))
        .with_distributed_serving(Arc::new(runtime));

        let observation = state.worker_observation();
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
    }

    #[test]
    fn missing_service_authentication_suppresses_distributed_readiness() {
        let mut state = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        state.auth = None;

        let observation = state.worker_observation();
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
        assert!(observation.capabilities.state_transfer);
    }

    #[test]
    fn aggregated_profile_never_projects_an_injected_distributed_runtime() {
        let distributed = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        let runtime = distributed.distributed_serving.unwrap();
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
        .with_distributed_serving(runtime);

        let observation = state.worker_observation();
        assert!(!observation.capabilities.state_transfer);
        assert_eq!(observation.transfer_health, TransferHealth::Unsupported);
    }

    #[test]
    fn missing_or_mismatched_runtime_fails_closed_in_observation() {
        let profile = execution_profile();
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                serving_execution: profile,
                ..PowerConfig::default()
            }),
        );
        let observation = state.worker_observation();
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
        assert!(!observation.capabilities.state_transfer);

        let distributed = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        let runtime = distributed.distributed_serving.unwrap();
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                serving_execution: execution_profile_with_generation(8),
                ..PowerConfig::default()
            }),
        )
        .with_distributed_serving(runtime);
        let observation = state.worker_observation();
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
        assert!(!observation.capabilities.state_transfer);
    }

    #[test]
    fn composed_observation_reuses_fail_fast_inflight_admission_not_http_limiter() {
        let state = state_with_services_and_limits(
            TransferHealth::Ready,
            PhaseExecutorHealth::Ready,
            2,
            99,
        );
        let observation = state.worker_observation();

        assert_eq!(observation.admission.active_limit, Some(2));
        assert_eq!(observation.admission.active, 0);
        assert_eq!(observation.admission.waiting, 0);
        assert_ne!(
            observation.admission.active_limit,
            Some(99),
            "worker observation must not invent a second admission surface from max_concurrent_requests"
        );
    }

    #[test]
    fn aggregated_observation_keeps_http_limiter_admission() {
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                max_concurrent_requests: 4,
                ..PowerConfig::default()
            }),
        );
        let observation = state.worker_observation();
        assert_eq!(observation.admission.active_limit, Some(4));
        assert_eq!(observation.admission.active, 0);
        assert_eq!(observation.admission.waiting, 0);
    }

    #[tokio::test]
    async fn observation_projects_held_phase_lease_and_stays_monotonic_after_taint() {
        use crate::serving::distributed_serving_tests::support::{
            profile, request, runtime_with_behavior, Calls, FixtureBehavior, TransferHooks,
        };
        use crate::serving::DecodePhaseRequest;
        use chrono::{Duration, Utc};
        use uuid::Uuid;

        let serving_profile = profile(DisaggregatedServingRole::Decode, 200);
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                serving_execution: serving_profile.clone(),
                max_concurrent_requests: 99,
                ..PowerConfig::default()
            }),
        );
        let hooks = Arc::new(TransferHooks::default());
        hooks
            .block_prepare_destination
            .store(true, Ordering::SeqCst);
        let calls = Arc::new(Calls::default());
        let runtime = runtime_with_behavior(
            &serving_profile,
            state.worker_epoch(),
            calls.clone(),
            FixtureBehavior {
                transfer_hooks: Arc::clone(&hooks),
                ..FixtureBehavior::default()
            },
        );
        let state = state
            .with_distributed_serving(Arc::new(runtime))
            .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])));

        let before = state.worker_observation();
        assert_eq!(before.admission.active_limit, Some(2));
        assert_eq!(before.admission.active, 0);
        assert_eq!(before.ready_phases, [ServingPhase::Decode]);

        let runtime = state.distributed_serving.as_ref().unwrap().clone();
        let execution_id = Uuid::new_v4();
        let prepare = tokio::spawn({
            let runtime = runtime.clone();
            async move {
                runtime
                    .prepare_decode(DecodePhaseRequest {
                        execution_id,
                        model: "internal/model-v1".to_string(),
                        request: request(),
                        expires_at: Utc::now() + Duration::milliseconds(200),
                    })
                    .await
            }
        });
        hooks.prepare_destination_started.notified().await;

        let held = state.worker_observation();
        assert_eq!(held.worker_epoch, before.worker_epoch);
        assert!(held.observation_generation > before.observation_generation);
        assert_eq!(held.admission.active_limit, Some(2));
        assert_eq!(held.admission.active, 1);
        assert_eq!(held.admission.waiting, 0);

        runtime.abort(execution_id).await.unwrap();
        let _ = prepare.await.unwrap();

        let after_abort = state.worker_observation();
        assert!(after_abort.observation_generation > held.observation_generation);
        assert_eq!(after_abort.admission.active, 0);
        assert_eq!(after_abort.ready_phases, [ServingPhase::Decode]);

        let tainted_runtime = runtime_with_behavior(
            &serving_profile,
            state.worker_epoch(),
            Arc::new(Calls::default()),
            FixtureBehavior {
                retryable_prepare: true,
                fail_abort: true,
                ..FixtureBehavior::default()
            },
        );
        let tainted_state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig {
                serving_execution: serving_profile,
                max_concurrent_requests: 99,
                ..PowerConfig::default()
            }),
        )
        .with_distributed_serving(Arc::new(tainted_runtime))
        .with_auth(Arc::new(ApiKeyAuth::new(&["service-key".to_string()])));
        let ready = tainted_state.worker_observation();
        assert_eq!(ready.ready_phases, [ServingPhase::Decode]);

        let taint_result = tainted_state
            .distributed_serving
            .as_ref()
            .unwrap()
            .prepare_decode(DecodePhaseRequest {
                execution_id: Uuid::new_v4(),
                model: "internal/model-v1".to_string(),
                request: request(),
                expires_at: Utc::now() + Duration::milliseconds(80),
            })
            .await;
        assert!(matches!(
            taint_result,
            Err(PowerError::BackendNotAvailable(_))
        ));
        assert!(!tainted_state
            .distributed_serving
            .as_ref()
            .unwrap()
            .accepts_work());

        let after_taint = tainted_state.worker_observation();
        assert_eq!(after_taint.worker_epoch, ready.worker_epoch);
        assert!(after_taint.observation_generation > ready.observation_generation);
        assert!(after_taint.ready_phases.is_empty());
        assert_eq!(after_taint.admission.active_limit, Some(2));
        assert_eq!(after_taint.admission.waiting, 0);
        assert_eq!(after_taint.admission.active, 0);
    }

    #[tokio::test]
    async fn matching_runtime_projects_label_free_transfer_metrics_on_service_endpoint() {
        use axum::body::Body;
        use axum::http::{Request, StatusCode};
        use tower::ServiceExt;

        let state = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        let snapshot = state
            .distributed_serving
            .as_ref()
            .unwrap()
            .transfer_runtime_snapshot();
        let app = crate::server::router::build(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();

        assert!(text.contains(&format!(
            "power_distributed_transfer_inflight_limit {}\n",
            snapshot.maximum_inflight_transfers
        )));
        assert!(text.contains("power_distributed_transfer_active "));
        assert!(text.contains("power_distributed_phase_admission_active_limit "));
        assert!(!text.contains("power_distributed_transfer_active{"));
        assert!(!text.contains("transfer_id="));
        assert!(!text.contains("tenant="));
    }

    #[tokio::test]
    async fn aggregated_service_metrics_omit_distributed_series() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        );
        let app = crate::server::router::build(state);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(!text.contains("power_distributed_transfer_"));
        assert!(!text.contains("power_distributed_phase_admission_"));
    }
}
