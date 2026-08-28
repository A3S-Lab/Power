use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{Duration, Utc};
use tokio::sync::Notify;
use uuid::Uuid;

use super::*;
use crate::error::{PowerError, Result};

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn profile(
    role: DisaggregatedServingRole,
    maximum_inflight: u32,
    timeout_ms: u64,
) -> ServingExecutionProfile {
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
        max_inflight_transfers: maximum_inflight,
        transfer_timeout_ms: timeout_ms,
        cancellation_timeout_ms: timeout_ms.min(10),
        privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
        privacy_policy_sha256: digest('7'),
        attestation_policy_sha256: None,
    })
    .unwrap()
}

fn capabilities(profile: &ServingExecutionProfile) -> StateTransferCapabilities {
    StateTransferCapabilities {
        execution_profile_sha256: profile.sha256().unwrap(),
        phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
        protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
        max_transfer_bytes: 4096,
        max_inflight_transfers: 8,
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

#[derive(Default)]
struct DriverControl {
    prepare_calls: AtomicUsize,
    publish_calls: AtomicUsize,
    consume_calls: AtomicUsize,
    abort_calls: AtomicUsize,
    block_prepare: AtomicBool,
    invalid_target: AtomicBool,
    fail_abort: AtomicBool,
    prepare_started: Notify,
}

struct TestDriver {
    capabilities: StateTransferCapabilities,
    control: Arc<DriverControl>,
}

#[async_trait]
impl StateTransferService for TestDriver {
    fn capabilities(&self) -> StateTransferCapabilities {
        self.capabilities.clone()
    }

    fn health(&self) -> TransferHealth {
        TransferHealth::Ready
    }

    async fn prepare_destination(
        &self,
        command: PrepareStateTransfer,
    ) -> Result<StateTransferTarget> {
        self.control.prepare_calls.fetch_add(1, Ordering::SeqCst);
        self.control.prepare_started.notify_one();
        if self.control.block_prepare.load(Ordering::SeqCst) {
            std::future::pending().await
        } else {
            Ok(StateTransferTarget {
                schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
                transfer_id: if self.control.invalid_target.load(Ordering::SeqCst) {
                    Uuid::new_v4()
                } else {
                    command.transfer_id
                },
                destination_worker_epoch: command.local_worker_epoch,
                binding: command.binding,
                protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
                prepared_at: Utc::now(),
                expires_at: command.expires_at,
                ticket: "target-ticket".to_string(),
            })
        }
    }

    async fn publish_source(&self, command: PublishStateTransfer) -> Result<StateTransferSource> {
        self.control.publish_calls.fetch_add(1, Ordering::SeqCst);
        Ok(StateTransferSource {
            schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
            transfer_id: command.target.transfer_id,
            source_worker_epoch: command.local_worker_epoch,
            destination_worker_epoch: command.target.destination_worker_epoch,
            binding: command.target.binding,
            protocol: command.target.protocol,
            published_at: Utc::now(),
            expires_at: command.target.expires_at,
            ticket: "source-ticket".to_string(),
        })
    }

    async fn consume_source(&self, command: ConsumeStateTransfer) -> Result<StateTransferReceipt> {
        self.control.consume_calls.fetch_add(1, Ordering::SeqCst);
        Ok(StateTransferReceipt {
            schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
            transfer_id: command.source.transfer_id,
            source_worker_epoch: command.source.source_worker_epoch,
            destination_worker_epoch: command.local_worker_epoch,
            binding: command.source.binding,
            protocol: command.source.protocol,
            bytes_transferred: 512,
            integrity: StateTransferIntegrity::TransportVerified,
            completed_at: Utc::now(),
        })
    }

    async fn abort(&self, _command: AbortStateTransfer) -> Result<()> {
        self.control.abort_calls.fetch_add(1, Ordering::SeqCst);
        if self.control.fail_abort.load(Ordering::SeqCst) {
            Err(PowerError::BackendNotAvailable(
                "test cleanup failed".to_string(),
            ))
        } else {
            Ok(())
        }
    }
}

fn service(
    profile: &ServingExecutionProfile,
    epoch: Uuid,
    control: Arc<DriverControl>,
) -> BoundedStateTransferService {
    BoundedStateTransferService::new(
        profile.clone(),
        epoch,
        Arc::new(TestDriver {
            capabilities: capabilities(profile),
            control,
        }),
    )
    .unwrap()
}

fn prepare(epoch: Uuid, lifetime_ms: i64) -> PrepareStateTransfer {
    PrepareStateTransfer {
        transfer_id: Uuid::new_v4(),
        local_worker_epoch: epoch,
        binding: binding(),
        destination: ModelStateHandle::new("destination").unwrap(),
        expires_at: Utc::now() + Duration::milliseconds(lifetime_ms),
    }
}

fn target(transfer_id: Uuid, destination_epoch: Uuid, lifetime_ms: i64) -> StateTransferTarget {
    let now = Utc::now();
    StateTransferTarget {
        schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
        transfer_id,
        destination_worker_epoch: destination_epoch,
        binding: binding(),
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        prepared_at: now,
        expires_at: now + Duration::milliseconds(lifetime_ms),
        ticket: "remote-target-ticket".to_string(),
    }
}

#[test]
fn construction_binds_epoch_and_projects_only_configured_limits() {
    let profile = profile(DisaggregatedServingRole::Decode, 2, 100);
    let epoch = Uuid::new_v4();
    let service = service(&profile, epoch, Arc::new(DriverControl::default()));

    assert_eq!(service.local_worker_epoch(), epoch);
    assert_eq!(
        service.capabilities(),
        StateTransferCapabilities {
            execution_profile_sha256: profile.sha256().unwrap(),
            phases: vec![ServingPhase::Decode],
            protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
            max_transfer_bytes: 1024,
            max_inflight_transfers: 2,
        }
    );
    assert_eq!(service.snapshot().active_transfers, 0);

    assert!(BoundedStateTransferService::new(
        profile,
        Uuid::nil(),
        Arc::new(TestDriver {
            capabilities: StateTransferCapabilities {
                execution_profile_sha256: digest('9'),
                phases: vec![ServingPhase::Decode],
                protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
                max_transfer_bytes: 1024,
                max_inflight_transfers: 2,
            },
            control: Arc::new(DriverControl::default()),
        }),
    )
    .is_err());
}

#[tokio::test]
async fn destination_lease_is_idempotent_and_holds_capacity_until_abort() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 250);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    let service = service(&profile, epoch, control.clone());
    let first = prepare(epoch, 200);

    let target = service.prepare_destination(first.clone()).await.unwrap();
    let replay = service.prepare_destination(first.clone()).await.unwrap();
    assert_eq!(target, replay);
    assert_eq!(control.prepare_calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.snapshot().active_transfers, 1);

    let second = prepare(epoch, 200);
    assert!(matches!(
        service.prepare_destination(second.clone()).await,
        Err(PowerError::BackendNotAvailable(_))
    ));
    service
        .abort(AbortStateTransfer {
            transfer_id: first.transfer_id,
            local_worker_epoch: epoch,
        })
        .await
        .unwrap();
    service.prepare_destination(second).await.unwrap();
}

#[tokio::test]
async fn immutable_binding_and_process_epoch_fail_before_driver_use() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 100);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    let service = service(&profile, epoch, control.clone());
    let mut wrong_binding = prepare(epoch, 80);
    wrong_binding.binding.model_sha256 = digest('9');

    assert!(matches!(
        service.prepare_destination(wrong_binding).await,
        Err(PowerError::InvalidRequest(_))
    ));
    assert!(matches!(
        service
            .prepare_destination(prepare(Uuid::new_v4(), 80))
            .await,
        Err(PowerError::InvalidRequest(_))
    ));
    assert_eq!(control.prepare_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn decode_consume_requires_its_exact_prepared_destination() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 250);
    let epoch = Uuid::new_v4();
    let source_epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    let service = service(&profile, epoch, control.clone());
    let command = prepare(epoch, 200);
    let target = service.prepare_destination(command.clone()).await.unwrap();
    let source = StateTransferSource {
        schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
        transfer_id: target.transfer_id,
        source_worker_epoch: source_epoch,
        destination_worker_epoch: epoch,
        binding: target.binding,
        protocol: target.protocol,
        published_at: Utc::now(),
        expires_at: target.expires_at,
        ticket: "source-ticket".to_string(),
    };

    let receipt = service
        .consume_source(ConsumeStateTransfer {
            local_worker_epoch: epoch,
            destination: command.destination.clone(),
            source: source.clone(),
        })
        .await
        .unwrap();
    assert_eq!(receipt.transfer_id, target.transfer_id);
    assert_eq!(service.snapshot().active_transfers, 0);
    assert_eq!(control.consume_calls.load(Ordering::SeqCst), 1);

    assert!(service
        .consume_source(ConsumeStateTransfer {
            local_worker_epoch: epoch,
            destination: command.destination,
            source,
        })
        .await
        .is_err());
}

#[tokio::test]
async fn published_source_is_idempotent_and_retained_until_compensating_abort() {
    let profile = profile(DisaggregatedServingRole::Prefill, 1, 250);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    let service = service(&profile, epoch, control.clone());
    let command = PublishStateTransfer {
        local_worker_epoch: epoch,
        source: ModelStateHandle::new("source").unwrap(),
        target: target(Uuid::new_v4(), Uuid::new_v4(), 200),
    };

    let source = service.publish_source(command.clone()).await.unwrap();
    assert_eq!(service.publish_source(command).await.unwrap(), source);
    assert_eq!(control.publish_calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.snapshot().active_transfers, 1);

    service
        .abort(AbortStateTransfer {
            transfer_id: source.transfer_id,
            local_worker_epoch: epoch,
        })
        .await
        .unwrap();
    assert_eq!(service.snapshot().active_transfers, 0);
}

#[tokio::test]
async fn timeout_aborts_and_cleanup_failure_taints_the_process() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 20);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    control.block_prepare.store(true, Ordering::SeqCst);
    control.fail_abort.store(true, Ordering::SeqCst);
    let service = service(&profile, epoch, control.clone());

    assert!(matches!(
        service.prepare_destination(prepare(epoch, 15)).await,
        Err(PowerError::BackendNotAvailable(_))
    ));
    assert_eq!(control.abort_calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.health(), TransferHealth::Unavailable);
    assert_eq!(service.snapshot().cleanup_failures, 1);
}

#[tokio::test]
async fn invalid_driver_output_is_rejected_and_cleaned() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 100);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    control.invalid_target.store(true, Ordering::SeqCst);
    let service = service(&profile, epoch, control.clone());

    assert!(matches!(
        service.prepare_destination(prepare(epoch, 80)).await,
        Err(PowerError::InvalidRequest(_))
    ));
    assert_eq!(control.abort_calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.snapshot().active_transfers, 0);
    assert_eq!(service.health(), TransferHealth::Ready);
}

#[tokio::test]
async fn explicit_abort_cancels_an_inflight_driver_operation() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 250);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    control.block_prepare.store(true, Ordering::SeqCst);
    let service = Arc::new(service(&profile, epoch, control.clone()));
    let command = prepare(epoch, 200);
    let transfer_id = command.transfer_id;
    let running = {
        let service = Arc::clone(&service);
        tokio::spawn(async move { service.prepare_destination(command).await })
    };
    control.prepare_started.notified().await;

    service
        .abort(AbortStateTransfer {
            transfer_id,
            local_worker_epoch: epoch,
        })
        .await
        .unwrap();
    assert!(matches!(
        running.await.unwrap(),
        Err(PowerError::BackendNotAvailable(_))
    ));
    assert_eq!(control.abort_calls.load(Ordering::SeqCst), 1);
    assert_eq!(service.snapshot().active_transfers, 0);
}

#[tokio::test]
async fn lease_expiry_reaps_registered_resources_without_another_request() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 100);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    let service = service(&profile, epoch, control.clone());

    service
        .prepare_destination(prepare(epoch, 20))
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_millis(200), async {
        while control.abort_calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expired lease must be cleaned without another request");

    assert_eq!(service.snapshot().active_transfers, 0);
    assert_eq!(service.snapshot().timeout_expirations, 1);
    assert_eq!(service.health(), TransferHealth::Ready);
}

#[tokio::test]
async fn dropping_an_operation_triggers_bounded_cleanup() {
    let profile = profile(DisaggregatedServingRole::Decode, 1, 250);
    let epoch = Uuid::new_v4();
    let control = Arc::new(DriverControl::default());
    control.block_prepare.store(true, Ordering::SeqCst);
    let service = Arc::new(service(&profile, epoch, control.clone()));
    let running = {
        let service = Arc::clone(&service);
        tokio::spawn(async move { service.prepare_destination(prepare(epoch, 200)).await })
    };
    control.prepare_started.notified().await;
    running.abort();

    tokio::time::timeout(std::time::Duration::from_millis(100), async {
        while control.abort_calls.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropped operation must trigger cleanup");
    assert_eq!(service.snapshot().active_transfers, 0);
}

#[test]
fn bounded_service_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BoundedStateTransferService>();
    assert_send_sync::<StateTransferRuntimeSnapshot>();
}
