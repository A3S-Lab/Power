use async_trait::async_trait;
use chrono::{Duration, TimeZone, Utc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use super::*;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: "internal/model-v1".to_string(),
        model_sha256: digest('1'),
        backend: "test-backend".to_string(),
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
    })
    .unwrap()
}

fn executor_capabilities(profile: &ServingExecutionProfile) -> PhaseExecutorCapabilities {
    PhaseExecutorCapabilities {
        execution_profile_sha256: profile.sha256().unwrap(),
        phase: profile.phase(),
    }
}

fn transfer_capabilities(profile: &ServingExecutionProfile) -> StateTransferCapabilities {
    StateTransferCapabilities {
        execution_profile_sha256: profile.sha256().unwrap(),
        phases: vec![profile.phase()],
        protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
        max_transfer_bytes: 1024,
        max_inflight_transfers: 2,
    }
}

fn binding() -> StateTransferBinding {
    StateTransferBinding {
        model_sha256: digest('1'),
        execution_sha256: digest('3'),
        layout_sha256: digest('5'),
        state_kind: StateKind::KvCache,
        token_count: 16,
        state_bytes: 512,
    }
}

fn completion_request() -> PhaseRequest {
    PhaseRequest::Completion(
        serde_json::from_value(serde_json::json!({ "prompt": "private prompt" })).unwrap(),
    )
}

struct ReceiptStateTransferService {
    capabilities: StateTransferCapabilities,
    receipt: StateTransferReceipt,
    pending: bool,
    aborted: Arc<AtomicBool>,
}

#[async_trait]
impl StateTransferService for ReceiptStateTransferService {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Ready
    }

    async fn prepare_destination(
        &self,
        _command: PrepareStateTransfer,
    ) -> crate::error::Result<StateTransferTarget> {
        Err(crate::error::PowerError::BackendNotAvailable(
            "test adapter".to_string(),
        ))
    }

    async fn publish_source(
        &self,
        _command: PublishStateTransfer,
    ) -> crate::error::Result<StateTransferSource> {
        Err(crate::error::PowerError::BackendNotAvailable(
            "test adapter".to_string(),
        ))
    }

    async fn consume_source(
        &self,
        _command: ConsumeStateTransfer,
    ) -> crate::error::Result<StateTransferReceipt> {
        if self.pending {
            return std::future::pending().await;
        }
        Ok(self.receipt.clone())
    }

    async fn abort(&self, _command: AbortStateTransfer) -> crate::error::Result<()> {
        self.aborted.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[test]
fn capabilities_bind_one_non_aggregated_phase_and_exact_profile() {
    for role in [
        DisaggregatedServingRole::Prefill,
        DisaggregatedServingRole::Decode,
    ] {
        let profile = profile(role);
        let capabilities = executor_capabilities(&profile);
        capabilities.validate().unwrap();
        profile
            .validate_phase_executor_capabilities(&capabilities)
            .unwrap();
    }

    let profile = profile(DisaggregatedServingRole::Decode);
    let mut invalid = executor_capabilities(&profile);
    invalid.phase = ServingPhase::Aggregated;
    assert!(invalid.validate().is_err());

    let mut mismatched = executor_capabilities(&profile);
    mismatched.execution_profile_sha256 = digest('9');
    assert!(profile
        .validate_phase_executor_capabilities(&mismatched)
        .is_err());
}

#[test]
fn request_debug_never_exposes_prompt_or_session_content() {
    let request = completion_request();
    let rendered = format!("{request:?}");
    assert!(rendered.contains("Completion"));
    assert!(rendered.contains("REDACTED"));
    assert!(!rendered.contains("private prompt"));
}

#[test]
fn preparation_binds_model_process_identity_and_profile_timeout() {
    let now = Utc.with_ymd_and_hms(2026, 8, 28, 0, 0, 0).unwrap();
    let profile = profile(DisaggregatedServingRole::Prefill);
    let capabilities = executor_capabilities(&profile);
    PreparePhaseExecution {
        execution_id: Uuid::new_v4(),
        local_worker_epoch: Uuid::new_v4(),
        model: "internal/model-v1".to_string(),
        request: completion_request(),
        expires_at: now + Duration::seconds(30),
    }
    .validate_at(now, &capabilities, &profile)
    .unwrap();

    let too_long = PreparePhaseExecution {
        execution_id: Uuid::new_v4(),
        local_worker_epoch: Uuid::new_v4(),
        model: "internal/model-v1".to_string(),
        request: completion_request(),
        expires_at: now + Duration::seconds(31),
    };
    assert!(too_long.validate_at(now, &capabilities, &profile).is_err());

    let wrong_model = PreparePhaseExecution {
        execution_id: Uuid::new_v4(),
        local_worker_epoch: Uuid::new_v4(),
        model: "internal/other-model".to_string(),
        request: completion_request(),
        expires_at: now + Duration::seconds(30),
    };
    assert!(wrong_model
        .validate_at(now, &capabilities, &profile)
        .is_err());
}

#[test]
fn operational_decisions_are_closed_and_retry_delay_is_bounded() {
    PhaseDecision::<()>::recompute(RecomputeReason::StateStale)
        .validate()
        .unwrap();
    PhaseDecision::<()>::retryable_unavailable(
        RetryableUnavailableReason::ResourcePressure,
        Some(250),
    )
    .unwrap()
    .validate()
    .unwrap();
    PhaseDecision::<()>::terminal_failure(TerminalFailureReason::PolicyViolation)
        .validate()
        .unwrap();

    assert!(PhaseDecision::<()>::retryable_unavailable(
        RetryableUnavailableReason::ExecutorUnavailable,
        Some(0),
    )
    .is_err());
    assert!(PhaseDecision::<()>::retryable_unavailable(
        RetryableUnavailableReason::ExecutorUnavailable,
        Some(300_001),
    )
    .is_err());
}

#[test]
fn phase_abort_covers_preparation_and_redacts_prepared_handles() {
    let execution_id = Uuid::new_v4();
    let worker_epoch = Uuid::new_v4();
    let preparing = AbortPhaseExecution::preparing(execution_id, worker_epoch).unwrap();
    preparing.validate().unwrap();
    assert!(preparing.execution().is_none());

    let prepared = AbortPhaseExecution::prepared(
        execution_id,
        worker_epoch,
        PhaseExecutionHandle::new("private-adapter-handle").unwrap(),
    )
    .unwrap();
    prepared.validate().unwrap();
    assert!(prepared.execution().is_some());
    let rendered = format!("{prepared:?}");
    assert!(rendered.contains("REDACTED"));
    assert!(!rendered.contains("private-adapter-handle"));
}

#[tokio::test]
async fn decode_can_run_only_with_state_consumed_into_its_prepared_destination() {
    let now = Utc::now();
    let profile = profile(DisaggregatedServingRole::Decode);
    let executor_capabilities = executor_capabilities(&profile);
    let transfer_capabilities = transfer_capabilities(&profile);
    let execution_id = Uuid::new_v4();
    let destination_epoch = Uuid::new_v4();
    let source_epoch = Uuid::new_v4();
    let destination = ModelStateHandle::new("decode-state").unwrap();
    let prepared = PreparedDecodePhase::new(
        execution_id,
        destination_epoch,
        profile.sha256().unwrap(),
        PhaseExecutionHandle::new("decode-execution").unwrap(),
        destination.clone(),
        binding(),
        now + Duration::seconds(30),
    )
    .unwrap();
    prepared
        .validate_at(now, &executor_capabilities, &profile)
        .unwrap();

    let source = StateTransferSource {
        schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
        transfer_id: Uuid::new_v4(),
        source_worker_epoch: source_epoch,
        destination_worker_epoch: destination_epoch,
        binding: binding(),
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        published_at: now - Duration::seconds(1),
        expires_at: now + Duration::seconds(30),
        ticket: "opaque-source".to_string(),
    };
    let consume = ConsumeStateTransfer {
        local_worker_epoch: destination_epoch,
        destination: destination.clone(),
        source: source.clone(),
    };
    let receipt = StateTransferReceipt {
        schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
        transfer_id: source.transfer_id,
        source_worker_epoch: source_epoch,
        destination_worker_epoch: destination_epoch,
        binding: binding(),
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        bytes_transferred: 512,
        integrity: StateTransferIntegrity::TransportVerified,
        completed_at: now,
    };
    let mut future_receipt = receipt.clone();
    future_receipt.completed_at = now + Duration::seconds(1);
    assert!(ImportedModelState::consume_at(
        &ReceiptStateTransferService {
            capabilities: transfer_capabilities.clone(),
            receipt: future_receipt,
            pending: false,
            aborted: Arc::new(AtomicBool::new(false)),
        },
        consume.clone(),
        now,
        &profile,
    )
    .await
    .is_err());
    let mut replay_source = source.clone();
    replay_source.published_at = now - Duration::seconds(10);
    let replay_consume = ConsumeStateTransfer {
        local_worker_epoch: destination_epoch,
        destination: destination.clone(),
        source: replay_source,
    };
    let mut replay_receipt = receipt.clone();
    replay_receipt.completed_at = now - Duration::seconds(5);
    assert!(ImportedModelState::consume_at(
        &ReceiptStateTransferService {
            capabilities: transfer_capabilities.clone(),
            receipt: replay_receipt,
            pending: false,
            aborted: Arc::new(AtomicBool::new(false)),
        },
        replay_consume,
        now,
        &profile,
    )
    .await
    .is_err());
    let service = ReceiptStateTransferService {
        capabilities: transfer_capabilities.clone(),
        receipt: receipt.clone(),
        pending: false,
        aborted: Arc::new(AtomicBool::new(false)),
    };
    let imported = ImportedModelState::consume_at(&service, consume.clone(), now, &profile)
        .await
        .unwrap();
    let command = ExecutePhaseExecution::decode(prepared.clone(), imported).unwrap();
    command
        .validate_at(
            now,
            &executor_capabilities,
            &transfer_capabilities,
            &profile,
        )
        .unwrap();

    let wrong_destination = PreparedDecodePhase::new(
        execution_id,
        destination_epoch,
        profile.sha256().unwrap(),
        PhaseExecutionHandle::new("decode-execution").unwrap(),
        ModelStateHandle::new("other-state").unwrap(),
        binding(),
        now + Duration::seconds(30),
    )
    .unwrap();
    let imported = ImportedModelState::consume_at(&service, consume, now, &profile)
        .await
        .unwrap();
    assert!(ExecutePhaseExecution::decode(wrong_destination, imported).is_err());

    let mut short_profile = profile.clone();
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut short_profile {
        execution.transfer_timeout_ms = 1;
        execution.cancellation_timeout_ms = 1;
    }
    let mut expiring_source = source;
    expiring_source.published_at = now - Duration::milliseconds(1);
    expiring_source.expires_at = now + Duration::milliseconds(1);
    let expiring_consume = ConsumeStateTransfer {
        local_worker_epoch: destination_epoch,
        destination,
        source: expiring_source,
    };
    let aborted = Arc::new(AtomicBool::new(false));
    let pending_service = ReceiptStateTransferService {
        capabilities: StateTransferCapabilities {
            execution_profile_sha256: short_profile.sha256().unwrap(),
            phases: vec![ServingPhase::Decode],
            protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
            max_transfer_bytes: 1024,
            max_inflight_transfers: 2,
        },
        receipt,
        pending: true,
        aborted: aborted.clone(),
    };
    let timeout_error = tokio::time::timeout(
        std::time::Duration::from_millis(20),
        ImportedModelState::consume_at(&pending_service, expiring_consume, now, &short_profile),
    )
    .await
    .expect("Power must enforce the one millisecond profile timeout")
    .unwrap_err();
    assert!(matches!(
        timeout_error,
        crate::error::PowerError::BackendNotAvailable(_)
    ));
    assert!(aborted.load(Ordering::SeqCst));
}

#[test]
fn phase_contract_types_are_send_and_sync_where_applicable() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<PhaseExecutorCapabilities>();
    assert_send_sync::<PhaseExecutionHandle>();
    assert_send_sync::<PreparedPhaseExecution>();
    assert_send_sync::<ImportedModelState>();
}
