use std::sync::Arc;

use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::serving::{
    validate_injected_production_adapters, BufferedHostLoopbackPhaseExecutor,
    ServingCompositionTransport, ServingPhaseExecutor, StateTransferService, TransferHealth,
};

/// Resolve ACL composition transport into injectable adapters before startup
/// validation.
///
/// `serving_execution.transport = buffered-host-loopback` is an honest opt-in
/// that installs the product loopback pair. It refuses silent auto-wire from
/// protocol alone and refuses mixing with builder-injected adapters. Incomplete
/// external pairs remain a validation error.
pub(super) fn resolve(
    config: &PowerConfig,
    state_transfer: Option<Arc<dyn StateTransferService>>,
    phase_executor: Option<Arc<dyn ServingPhaseExecutor>>,
) -> Result<(
    Option<Arc<dyn StateTransferService>>,
    Option<Arc<dyn ServingPhaseExecutor>>,
)> {
    match config.serving_execution.composition_transport() {
        None => Ok((state_transfer, phase_executor)),
        Some(ServingCompositionTransport::BufferedHostLoopback) => {
            if state_transfer.is_some() || phase_executor.is_some() {
                return Err(PowerError::Config(
                    "serving_execution.transport = buffered-host-loopback cannot combine with builder-injected distributed adapters"
                        .to_string(),
                ));
            }
            let (transfer, executor) =
                BufferedHostLoopbackPhaseExecutor::paired_for_profile(&config.serving_execution)?;
            Ok((
                Some(transfer as Arc<dyn StateTransferService>),
                Some(executor as Arc<dyn ServingPhaseExecutor>),
            ))
        }
    }
}

/// Validate the process-local serving composition before listeners or model
/// resources are created.
pub(super) fn validate(
    config: &PowerConfig,
    state_transfer: Option<&dyn StateTransferService>,
    phase_executor: Option<&dyn ServingPhaseExecutor>,
) -> Result<()> {
    let profile = &config.serving_execution;
    if profile.is_aggregated() {
        return if state_transfer.is_none() && phase_executor.is_none() {
            Ok(())
        } else {
            Err(PowerError::Config(
                "aggregated serving cannot install distributed serving services".to_string(),
            ))
        };
    }

    let state_transfer = state_transfer.ok_or_else(|| {
        PowerError::Config("prefill-decode serving requires a state-transfer adapter".to_string())
    })?;
    let phase_executor = phase_executor.ok_or_else(|| {
        PowerError::Config("prefill-decode serving requires a phase executor".to_string())
    })?;

    validate_injected_production_adapters(state_transfer, phase_executor)?;
    profile.validate_state_transfer_capabilities(&state_transfer.capabilities())?;
    profile.validate_phase_executor_capabilities(&phase_executor.capabilities())?;
    if matches!(state_transfer.health(), TransferHealth::Unsupported) {
        return Err(PowerError::Config(
            "a configured state-transfer adapter cannot report unsupported health".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use std::sync::Arc;

    use crate::serving::{
        AbortPhaseExecution, AbortStateTransfer, BoundedStateTransferService, ConsumeStateTransfer,
        DisaggregatedServingRole, DistributedServingRuntime, EmptyServingPhaseExecutor,
        EmptyStateTransferService, ExecutePhaseExecution, PhaseDecision, PhaseExecutionOutput,
        PhaseExecutorCapabilities, PhaseExecutorHealth, PhaseSessionPoolMode, PhaseWeightCacheMode,
        PrefillDecodeExecutionProfile, PreparePhaseExecution, PrepareStateTransfer,
        PreparedPhaseExecution, PublishStateTransfer, ServingCompositionTransport,
        ServingExecutionProfile, ServingPhase, ServingPhaseExecutor, ServingPrivacyMode, StateKind,
        StateTransferCapabilities, StateTransferProtocol, StateTransferReceipt,
        StateTransferSource, StateTransferTarget,
    };

    use super::*;

    struct TestStateTransferService {
        capabilities: StateTransferCapabilities,
        health: TransferHealth,
    }

    struct TestPhaseExecutor {
        capabilities: PhaseExecutorCapabilities,
        health: PhaseExecutorHealth,
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

    fn profile() -> ServingExecutionProfile {
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
            weight_cache: PhaseWeightCacheMode::SharedWeightHierarchy,
            residency_policy_sha256: None,
            session_pool: PhaseSessionPoolMode::SharedSessionPool,
            session_pool_policy_sha256: None,
            transport: None,
        })
        .unwrap()
    }

    fn service(profile: &ServingExecutionProfile) -> TestStateTransferService {
        TestStateTransferService {
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: profile.sha256().unwrap(),
                phases: vec![ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 2,
            },
            health: TransferHealth::Ready,
        }
    }

    fn executor(profile: &ServingExecutionProfile) -> TestPhaseExecutor {
        TestPhaseExecutor {
            capabilities: PhaseExecutorCapabilities::for_profile(profile).unwrap(),
            health: PhaseExecutorHealth::Ready,
        }
    }

    fn buffered_host_profile(
        transport: Option<ServingCompositionTransport>,
    ) -> ServingExecutionProfile {
        ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "buffered-host-loopback".to_string(),
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
            transport,
        })
        .unwrap()
    }

    #[test]
    fn aggregated_default_requires_no_distributed_services() {
        validate(&PowerConfig::default(), None, None).unwrap();
    }

    #[test]
    fn aggregated_profile_rejects_unused_state_transfer_service() {
        let profile = profile();
        let error = validate(
            &PowerConfig::default(),
            Some(&service(&profile)),
            Some(&executor(&profile)),
        )
        .unwrap_err();
        assert!(error.to_string().contains("aggregated serving"));
    }

    #[test]
    fn prefill_decode_requires_an_exact_adapter_and_phase_executor_pair() {
        let profile = profile();
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        assert!(validate(&config, None, None)
            .unwrap_err()
            .to_string()
            .contains("state-transfer adapter"));

        assert!(validate(&config, Some(&service(&profile)), None)
            .unwrap_err()
            .to_string()
            .contains("phase executor"));

        validate(&config, Some(&service(&profile)), Some(&executor(&profile))).unwrap();

        let mut mismatched = service(&profile);
        mismatched.capabilities.execution_profile_sha256 = "9".repeat(64);
        assert!(
            validate(&config, Some(&mismatched), Some(&executor(&profile)),)
                .unwrap_err()
                .to_string()
                .contains("immutable serving profile")
        );

        let mut mismatched = executor(&profile);
        mismatched.capabilities.execution_profile_sha256 = "9".repeat(64);
        assert!(
            validate(&config, Some(&service(&profile)), Some(&mismatched))
                .unwrap_err()
                .to_string()
                .contains("immutable serving profile")
        );
    }

    #[test]
    fn temporarily_unavailable_composed_services_do_not_invalidate_static_startup() {
        let profile = profile();
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        let mut service = service(&profile);
        service.health = TransferHealth::Unavailable;
        let mut executor = executor(&profile);
        executor.health = PhaseExecutorHealth::Unavailable;

        validate(&config, Some(&service), Some(&executor)).unwrap();
    }

    #[test]
    fn empty_placeholders_fail_closed_until_injected() {
        let profile = profile();
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        let empty_transfer = EmptyStateTransferService::for_profile(&profile).unwrap();
        let empty_executor = EmptyServingPhaseExecutor::for_profile(&profile).unwrap();
        let err = validate(&config, Some(&empty_transfer), Some(&empty_executor)).unwrap_err();
        assert!(err.to_string().contains("Empty placeholders"));

        assert!(
            validate(&config, Some(&empty_transfer), Some(&executor(&profile)))
                .unwrap_err()
                .to_string()
                .contains("Empty placeholders")
        );

        assert!(
            validate(&config, Some(&service(&profile)), Some(&empty_executor))
                .unwrap_err()
                .to_string()
                .contains("Empty placeholders")
        );
    }

    #[test]
    fn product_buffered_host_loopback_pair_satisfies_startup_gate() {
        let profile = buffered_host_profile(None);
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        let (transfer, executor) =
            crate::serving::BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile)
                .unwrap();
        validate(&config, Some(transfer.as_ref()), Some(executor.as_ref())).unwrap();
    }

    #[test]
    fn product_buffered_host_transfer_alone_still_requires_phase_executor() {
        let profile = buffered_host_profile(None);
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        let transfer =
            crate::serving::BufferedHostLoopbackStateTransfer::for_profile(&profile).unwrap();
        let err = validate(&config, Some(&transfer), None).unwrap_err();
        assert!(err.to_string().contains("phase executor"));
    }

    #[test]
    fn aggregated_default_still_rejects_product_buffered_host_injection() {
        let profile = buffered_host_profile(None);
        let (transfer, executor) =
            crate::serving::BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile)
                .unwrap();
        let err = validate(
            &PowerConfig::default(),
            Some(transfer.as_ref()),
            Some(executor.as_ref()),
        )
        .unwrap_err();
        assert!(err.to_string().contains("aggregated serving"));
    }

    #[test]
    fn incomplete_pair_without_transport_opt_in_fails_resolve_then_validate() {
        let profile = buffered_host_profile(None);
        let config = PowerConfig {
            serving_execution: profile,
            api_keys: vec!["service-key".to_string()],
            ..PowerConfig::default()
        };
        let (transfer, executor) = resolve(&config, None, None).unwrap();
        assert!(transfer.is_none());
        assert!(executor.is_none());
        let err = validate(&config, transfer.as_deref(), executor.as_deref()).unwrap_err();
        assert!(err.to_string().contains("state-transfer adapter"));
    }

    #[test]
    fn transport_opt_in_wires_complete_product_pair_and_projects_ready_phase() {
        let profile =
            buffered_host_profile(Some(ServingCompositionTransport::BufferedHostLoopback));
        let config = PowerConfig {
            serving_execution: profile.clone(),
            api_keys: vec!["service-key".to_string()],
            ..PowerConfig::default()
        };
        let (transfer, executor) = resolve(&config, None, None).unwrap();
        validate(&config, transfer.as_deref(), executor.as_deref()).unwrap();
        let transfer = transfer.expect("transport opt-in installs transfer");
        let executor = executor.expect("transport opt-in installs phase executor");
        let bounded = Arc::new(
            BoundedStateTransferService::new(profile.clone(), uuid::Uuid::new_v4(), transfer)
                .unwrap(),
        );
        let runtime = DistributedServingRuntime::new(profile.clone(), bounded, executor).unwrap();
        assert!(runtime.accepts_work());
        assert_eq!(runtime.phase(), ServingPhase::Decode);
        assert_eq!(
            profile.composition_transport(),
            Some(ServingCompositionTransport::BufferedHostLoopback)
        );
        // Honest opt-in installs buffered-host loopback only; never an HSN protocol.
        if let ServingExecutionProfile::PrefillDecode { execution } = &profile {
            assert!(matches!(
                execution.protocol,
                StateTransferProtocol::BufferedHostMemoryPullV1
            ));
        }
    }

    #[test]
    fn transport_opt_in_refuses_builder_injected_partial_or_complete_pair() {
        let profile =
            buffered_host_profile(Some(ServingCompositionTransport::BufferedHostLoopback));
        let config = PowerConfig {
            serving_execution: profile.clone(),
            api_keys: vec!["service-key".to_string()],
            ..PowerConfig::default()
        };
        let (transfer, executor) =
            crate::serving::BufferedHostLoopbackPhaseExecutor::paired_for_profile(&profile)
                .unwrap();
        let err = match resolve(
            &config,
            Some(transfer.clone() as Arc<dyn StateTransferService>),
            None,
        ) {
            Ok(_) => panic!("expected resolve to reject a partial builder injection"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("cannot combine"));

        let err = match resolve(
            &config,
            Some(transfer as Arc<dyn StateTransferService>),
            Some(executor as Arc<dyn ServingPhaseExecutor>),
        ) {
            Ok(_) => panic!("expected resolve to reject a complete builder injection mix"),
            Err(error) => error,
        };
        assert!(err.to_string().contains("cannot combine"));
    }

    #[test]
    fn transport_opt_in_with_wrong_protocol_fails_closed_at_profile_validate() {
        let err = ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
            role: DisaggregatedServingRole::Decode,
            model: "internal/model-v1".to_string(),
            model_sha256: "1".repeat(64),
            backend: "buffered-host-loopback".to_string(),
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
            transport: Some(ServingCompositionTransport::BufferedHostLoopback),
        })
        .unwrap_err();
        assert!(err.to_string().contains("buffered-host-memory-pull-v1"));
    }

    #[test]
    fn aggregated_default_resolve_does_not_advertise_prefill_decode() {
        let config = PowerConfig::default();
        assert!(config.serving_execution.is_aggregated());
        assert!(config.serving_execution.composition_transport().is_none());
        let (transfer, executor) = resolve(&config, None, None).unwrap();
        validate(&config, transfer.as_deref(), executor.as_deref()).unwrap();
        assert!(transfer.is_none());
        assert!(executor.is_none());
        assert_eq!(config.serving_execution.phase(), ServingPhase::Aggregated);
    }

    #[test]
    fn protocol_alone_does_not_auto_wire_product_pair() {
        let profile = buffered_host_profile(None);
        let config = PowerConfig {
            serving_execution: profile,
            api_keys: vec!["service-key".to_string()],
            ..PowerConfig::default()
        };
        let (transfer, executor) = resolve(&config, None, None).unwrap();
        assert!(transfer.is_none());
        assert!(executor.is_none());
    }
}
