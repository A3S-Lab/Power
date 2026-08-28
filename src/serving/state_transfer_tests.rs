use chrono::{Duration, TimeZone, Utc};
use uuid::Uuid;

use super::*;

fn now() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 8, 28, 0, 0, 0).unwrap()
}

fn binding() -> StateTransferBinding {
    StateTransferBinding {
        model_sha256: "1".repeat(64),
        execution_sha256: "2".repeat(64),
        layout_sha256: "3".repeat(64),
        state_kind: StateKind::KvCache,
        token_count: 4_096,
        state_bytes: 8 * 1024 * 1024,
    }
}

fn capabilities() -> StateTransferCapabilities {
    StateTransferCapabilities {
        execution_profile_sha256: "9".repeat(64),
        phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
        protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
        max_transfer_bytes: 16 * 1024 * 1024,
        max_inflight_transfers: 8,
    }
}

fn target() -> StateTransferTarget {
    StateTransferTarget {
        schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
        transfer_id: Uuid::from_u128(1),
        destination_worker_epoch: Uuid::from_u128(2),
        binding: binding(),
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        prepared_at: now(),
        expires_at: now() + Duration::seconds(30),
        ticket: "decode-adapter-ticket".to_string(),
    }
}

fn source() -> StateTransferSource {
    let target = target();
    StateTransferSource {
        schema: STATE_TRANSFER_SOURCE_SCHEMA.to_string(),
        transfer_id: target.transfer_id,
        source_worker_epoch: Uuid::from_u128(3),
        destination_worker_epoch: target.destination_worker_epoch,
        binding: target.binding,
        protocol: target.protocol,
        published_at: now() + Duration::seconds(1),
        expires_at: target.expires_at,
        ticket: "prefill-adapter-ticket".to_string(),
    }
}

#[test]
fn capabilities_are_canonical_and_phase_specific() {
    let capabilities = capabilities();
    capabilities.validate().unwrap();
    assert!(capabilities.supports_phase(ServingPhase::Prefill));
    assert!(capabilities.supports_phase(ServingPhase::Decode));
    assert!(!capabilities.supports_phase(ServingPhase::Aggregated));

    let mut invalid = capabilities.clone();
    invalid.phases.reverse();
    assert!(invalid.validate().is_err());
    let mut invalid = capabilities;
    invalid.phases.insert(0, ServingPhase::Aggregated);
    assert!(invalid.validate().is_err());
}

#[test]
fn binding_and_local_handle_are_bounded() {
    binding().validate().unwrap();
    let handle = ModelStateHandle::new("backend-slot-7").unwrap();
    assert_eq!(handle.as_str(), "backend-slot-7");
    assert_eq!(format!("{handle:?}"), "ModelStateHandle([REDACTED])");
    assert!(ModelStateHandle::new(" leading").is_err());
    assert!(ModelStateHandle::new("x".repeat(513)).is_err());

    let mut invalid = binding();
    invalid.layout_sha256 = "not-a-digest".to_string();
    assert!(invalid.validate().is_err());
    let mut noncanonical = binding();
    noncanonical.model_sha256 = "A".repeat(64);
    assert!(noncanonical.validate().is_err());
}

#[test]
fn target_and_source_are_closed_short_lived_and_exactly_bound() {
    let capabilities = capabilities();
    let target = target();
    target.validate_at(now(), &capabilities).unwrap();
    source()
        .validate_for(&target, now() + Duration::seconds(1), &capabilities)
        .unwrap();

    let mut value = serde_json::to_value(&target).unwrap();
    value["unknown"] = serde_json::json!(true);
    assert!(serde_json::from_value::<StateTransferTarget>(value).is_err());

    let mut expired = target.clone();
    expired.expires_at = now();
    assert!(expired.validate_at(now(), &capabilities).is_err());
    let mut too_long = target.clone();
    too_long.expires_at = now() + Duration::seconds(301);
    assert!(too_long.validate_at(now(), &capabilities).is_err());
    let mut mismatched = source();
    mismatched.binding.layout_sha256 = "4".repeat(64);
    assert!(mismatched
        .validate_for(&target, now() + Duration::seconds(1), &capabilities,)
        .is_err());
    let mut invalid_lifetime = target.clone();
    invalid_lifetime.expires_at = invalid_lifetime.prepared_at + Duration::seconds(301);
    assert!(invalid_lifetime.validate_at(now(), &capabilities).is_err());
}

#[test]
fn receipt_proves_exact_source_identity_size_and_integrity() {
    let source = source();
    let receipt = StateTransferReceipt {
        schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
        transfer_id: source.transfer_id,
        source_worker_epoch: source.source_worker_epoch,
        destination_worker_epoch: source.destination_worker_epoch,
        binding: source.binding.clone(),
        protocol: source.protocol,
        bytes_transferred: source.binding.state_bytes,
        integrity: StateTransferIntegrity::Sha256 {
            digest: "5".repeat(64),
        },
        completed_at: now() + Duration::seconds(10),
    };
    receipt.validate_for(&source, &capabilities()).unwrap();

    let mut short = receipt.clone();
    short.bytes_transferred -= 1;
    assert!(short.validate_for(&source, &capabilities()).is_err());
    let mut invalid_digest = receipt;
    invalid_digest.integrity = StateTransferIntegrity::Sha256 {
        digest: "invalid".to_string(),
    };
    assert!(invalid_digest
        .validate_for(&source, &capabilities())
        .is_err());
}

#[test]
fn commands_enforce_phase_epoch_size_and_expiry_before_adapter_use() {
    let adapter_capabilities = capabilities();
    let prepare = PrepareStateTransfer {
        transfer_id: Uuid::from_u128(1),
        local_worker_epoch: Uuid::from_u128(2),
        binding: binding(),
        destination: ModelStateHandle::new("decode-slot").unwrap(),
        expires_at: now() + Duration::seconds(30),
    };
    prepare.validate_at(now(), &adapter_capabilities).unwrap();

    let publish = PublishStateTransfer {
        local_worker_epoch: Uuid::from_u128(3),
        source: ModelStateHandle::new("prefill-slot").unwrap(),
        target: target(),
    };
    publish
        .validate_at(now() + Duration::seconds(1), &adapter_capabilities)
        .unwrap();

    let consume = ConsumeStateTransfer {
        local_worker_epoch: Uuid::from_u128(2),
        destination: ModelStateHandle::new("decode-slot").unwrap(),
        source: source(),
    };
    consume
        .validate_at(now() + Duration::seconds(2), &adapter_capabilities)
        .unwrap();
    AbortStateTransfer {
        transfer_id: Uuid::from_u128(1),
        local_worker_epoch: Uuid::from_u128(2),
    }
    .validate()
    .unwrap();

    let mut prefill_only = adapter_capabilities;
    prefill_only.phases = vec![ServingPhase::Prefill];
    assert!(prepare.validate_at(now(), &prefill_only).is_err());
    let mut wrong_destination = consume;
    wrong_destination.local_worker_epoch = Uuid::from_u128(4);
    assert!(wrong_destination
        .validate_at(now() + Duration::seconds(2), &capabilities())
        .is_err());
}

#[test]
fn public_transfer_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<StateTransferBinding>();
    assert_send_sync::<StateTransferTarget>();
    assert_send_sync::<StateTransferSource>();
    assert_send_sync::<StateTransferReceipt>();
    assert_send_sync::<ModelStateHandle>();
    assert_send_sync::<PrepareStateTransfer>();
    assert_send_sync::<PublishStateTransfer>();
    assert_send_sync::<ConsumeStateTransfer>();
}
