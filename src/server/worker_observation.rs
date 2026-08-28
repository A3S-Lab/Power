//! Process-local collector for the public worker observation contract.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::serving::{
    AdmissionObservation, PromptCacheObservation, ServingPhase, TransferHealth, WorkerCapabilities,
    WorkerObservation, WORKER_OBSERVATION_SCHEMA,
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
            admission: AdmissionObservation {
                active_limit: (state.config.max_concurrent_requests > 0)
                    .then_some(state.config.max_concurrent_requests),
                active: state.metrics.running_requests(),
                waiting: state.metrics.waiting_requests(),
            },
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

    let (Some(state_transfer), Some(phase_executor)) = (
        state.state_transfer_service.as_ref(),
        state.phase_executor.as_ref(),
    ) else {
        return unsupported_distributed_observation();
    };
    let transfer_capabilities = state_transfer.capabilities();
    let executor_capabilities = phase_executor.capabilities();
    let transfer_health = state_transfer.health();
    if profile
        .validate_state_transfer_capabilities(&transfer_capabilities)
        .is_err()
        || profile
            .validate_phase_executor_capabilities(&executor_capabilities)
            .is_err()
        || matches!(transfer_health, TransferHealth::Unsupported)
    {
        return unsupported_distributed_observation();
    }

    let phase = profile.phase();
    let transfer_ready = matches!(
        transfer_health,
        TransferHealth::Ready | TransferHealth::Degraded
    );
    let ready_phases = if transfer_ready && phase_executor.health().accepts_work() {
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
    use std::sync::Arc;

    use async_trait::async_trait;

    use crate::backend::BackendRegistry;
    use crate::config::PowerConfig;
    use crate::error::{PowerError, Result};
    use crate::model::registry::ModelRegistry;
    use crate::serving::{
        AbortPhaseExecution, AbortStateTransfer, ConsumeStateTransfer, DisaggregatedServingRole,
        ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput, PhaseExecutorCapabilities,
        PhaseExecutorHealth, PrefillDecodeExecutionProfile, PreparePhaseExecution,
        PrepareStateTransfer, PreparedPhaseExecution, PublishStateTransfer,
        ServingExecutionProfile, ServingPhaseExecutor, ServingPrivacyMode, StateKind,
        StateTransferCapabilities, StateTransferProtocol, StateTransferReceipt,
        StateTransferService, StateTransferSource, StateTransferTarget,
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
            max_inflight_transfers: 2,
            transfer_timeout_ms: 30_000,
            cancellation_timeout_ms: 5_000,
            privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
            privacy_policy_sha256: "7".repeat(64),
            attestation_policy_sha256: None,
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
        let profile = execution_profile();
        let service = TestStateTransferService {
            health: transfer_health,
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 2,
            },
        };
        let executor = TestPhaseExecutor {
            health: executor_health,
            capabilities: PhaseExecutorCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phase: profile.phase(),
            },
        };
        let config = PowerConfig {
            serving_execution: profile,
            ..PowerConfig::default()
        };
        AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(config),
        )
        .with_state_transfer_service(Arc::new(service))
        .with_phase_executor(Arc::new(executor))
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
    fn unavailable_phase_executor_keeps_capability_but_suppresses_readiness() {
        let observation =
            state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Unavailable)
                .worker_observation();

        assert!(observation.capabilities.state_transfer);
        assert_eq!(observation.capabilities.phases, [ServingPhase::Decode]);
        assert!(observation.ready_phases.is_empty());
    }

    #[test]
    fn aggregated_profile_never_projects_an_injected_transport() {
        let profile = execution_profile();
        let state = AppState::new(
            Arc::new(ModelRegistry::new()),
            Arc::new(BackendRegistry::new()),
            Arc::new(PowerConfig::default()),
        )
        .with_state_transfer_service(Arc::new(TestStateTransferService {
            health: TransferHealth::Ready,
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 2,
            },
        }))
        .with_phase_executor(Arc::new(TestPhaseExecutor {
            health: PhaseExecutorHealth::Ready,
            capabilities: PhaseExecutorCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phase: ServingPhase::Decode,
            },
        }));

        let observation = state.worker_observation();
        assert!(!observation.capabilities.state_transfer);
        assert_eq!(observation.transfer_health, TransferHealth::Unsupported);
    }

    #[test]
    fn invalid_or_unsupported_adapter_fails_closed_in_observation() {
        let mut state =
            state_with_services(TransferHealth::Unsupported, PhaseExecutorHealth::Ready);
        let observation = state.worker_observation();
        assert!(!observation.capabilities.state_transfer);
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
        assert_eq!(observation.transfer_health, TransferHealth::Unsupported);

        state.state_transfer_service = Some(Arc::new(TestStateTransferService {
            health: TransferHealth::Ready,
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: state.config.serving_execution.sha256().unwrap(),
                phases: vec![ServingPhase::Decode, ServingPhase::Prefill],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 2,
            },
        }));
        let observation = state.worker_observation();
        assert!(!observation.capabilities.state_transfer);
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
    }

    #[test]
    fn missing_or_mismatched_phase_executor_fails_closed_in_observation() {
        let mut state = state_with_services(TransferHealth::Ready, PhaseExecutorHealth::Ready);
        state.phase_executor = None;
        let observation = state.worker_observation();
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
        assert!(!observation.capabilities.state_transfer);

        let profile_sha256 = state.config.serving_execution.sha256().unwrap();
        state.phase_executor = Some(Arc::new(TestPhaseExecutor {
            health: PhaseExecutorHealth::Ready,
            capabilities: PhaseExecutorCapabilities {
                execution_profile_sha256: profile_sha256,
                phase: ServingPhase::Prefill,
            },
        }));
        let observation = state.worker_observation();
        assert!(observation.capabilities.phases.is_empty());
        assert!(observation.ready_phases.is_empty());
    }
}
