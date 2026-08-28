use crate::config::PowerConfig;
use crate::error::{PowerError, Result};
use crate::serving::{StateTransferService, TransferHealth};

/// Validate the process-local serving composition before listeners or model
/// resources are created.
pub(super) fn validate(
    config: &PowerConfig,
    state_transfer: Option<&dyn StateTransferService>,
) -> Result<()> {
    match (&config.serving_execution, state_transfer) {
        (profile, None) if profile.is_aggregated() => Ok(()),
        (profile, Some(_)) if profile.is_aggregated() => Err(PowerError::Config(
            "aggregated serving cannot install a state-transfer adapter".to_string(),
        )),
        (_, None) => Err(PowerError::Config(
            "prefill-decode serving requires a state-transfer adapter".to_string(),
        )),
        (profile, Some(service)) => {
            profile.validate_state_transfer_capabilities(&service.capabilities())?;
            if matches!(service.health(), TransferHealth::Unsupported) {
                return Err(PowerError::Config(
                    "a configured state-transfer adapter cannot report unsupported health"
                        .to_string(),
                ));
            }
            Err(PowerError::Config(
                "prefill-decode serving requires a verified phase executor; this build has no phase-executor composition port"
                    .to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;

    use crate::serving::{
        AbortStateTransfer, ConsumeStateTransfer, DisaggregatedServingRole,
        PrefillDecodeExecutionProfile, PrepareStateTransfer, PublishStateTransfer,
        ServingExecutionProfile, ServingPhase, ServingPrivacyMode, StateKind,
        StateTransferCapabilities, StateTransferProtocol, StateTransferReceipt,
        StateTransferSource, StateTransferTarget,
    };

    use super::*;

    struct TestStateTransferService {
        capabilities: StateTransferCapabilities,
        health: TransferHealth,
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

    #[test]
    fn aggregated_default_requires_no_distributed_services() {
        validate(&PowerConfig::default(), None).unwrap();
    }

    #[test]
    fn aggregated_profile_rejects_unused_state_transfer_service() {
        let profile = profile();
        let error = validate(&PowerConfig::default(), Some(&service(&profile))).unwrap_err();
        assert!(error.to_string().contains("aggregated serving"));
    }

    #[test]
    fn prefill_decode_requires_exact_adapter_and_phase_executor() {
        let profile = profile();
        let config = PowerConfig {
            serving_execution: profile.clone(),
            ..PowerConfig::default()
        };
        assert!(validate(&config, None)
            .unwrap_err()
            .to_string()
            .contains("state-transfer adapter"));

        let error = validate(&config, Some(&service(&profile))).unwrap_err();
        assert!(error.to_string().contains("phase executor"));

        let mut mismatched = service(&profile);
        mismatched.capabilities.execution_profile_sha256 = "9".repeat(64);
        assert!(validate(&config, Some(&mismatched))
            .unwrap_err()
            .to_string()
            .contains("immutable serving profile"));
    }
}
