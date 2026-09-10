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

fn deployment() -> ServingDeploymentIdentity {
    ServingDeploymentIdentity {
        generation: 7,
        peer_set_sha256: "6".repeat(64),
    }
}

fn target() -> StateTransferTarget {
    StateTransferTarget {
        schema: STATE_TRANSFER_TARGET_SCHEMA.to_string(),
        transfer_id: Uuid::from_u128(1),
        destination_worker_epoch: Uuid::from_u128(2),
        deployment: deployment(),
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
        deployment: target.deployment.clone(),
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
fn wire_descriptor_debug_never_exposes_adapter_tickets() {
    let target_debug = format!("{:?}", target());
    let source_debug = format!("{:?}", source());

    assert!(target_debug.contains("REDACTED"));
    assert!(source_debug.contains("REDACTED"));
    assert!(!target_debug.contains("decode-adapter-ticket"));
    assert!(!source_debug.contains("prefill-adapter-ticket"));
}

#[test]
fn receipt_proves_exact_source_identity_size_and_integrity() {
    let source = source();
    let receipt = StateTransferReceipt {
        schema: STATE_TRANSFER_RECEIPT_SCHEMA.to_string(),
        transfer_id: source.transfer_id,
        source_worker_epoch: source.source_worker_epoch,
        destination_worker_epoch: source.destination_worker_epoch,
        deployment: source.deployment.clone(),
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
    let mut invalid_digest = receipt.clone();
    invalid_digest.integrity = StateTransferIntegrity::Sha256 {
        digest: "invalid".to_string(),
    };
    assert!(invalid_digest
        .validate_for(&source, &capabilities())
        .is_err());
    let mut wrong_id = receipt;
    wrong_id.transfer_id = Uuid::from_u128(99);
    assert!(wrong_id.validate_for(&source, &capabilities()).is_err());
}

#[test]
fn deployment_identity_must_agree_across_target_source_and_receipt() {
    let capabilities = capabilities();
    let target = target();
    let source = source();
    source
        .validate_for(&target, now() + Duration::seconds(1), &capabilities)
        .unwrap();

    let mut stale = source.clone();
    stale.deployment.generation = 8;
    assert!(stale
        .validate_for(&target, now() + Duration::seconds(1), &capabilities)
        .is_err());

    let mut foreign = source;
    foreign.deployment.peer_set_sha256 = "a".repeat(64);
    assert!(foreign
        .validate_for(&target, now() + Duration::seconds(1), &capabilities)
        .is_err());
}

#[test]
fn legacy_v1_transfer_schemas_fail_closed() {
    let capabilities = capabilities();
    let mut legacy_target = target();
    legacy_target.schema = "a3s.power.state-transfer-target.v1".to_string();
    assert!(legacy_target.validate_at(now(), &capabilities).is_err());

    let mut legacy_source = source();
    legacy_source.schema = "a3s.power.state-transfer-source.v1".to_string();
    assert!(legacy_source
        .validate_for(&target(), now() + Duration::seconds(1), &capabilities)
        .is_err());
}

#[test]
fn corrupt_ticket_bytes_never_validate_as_authenticated_descriptors() {
    let capabilities = capabilities();
    let mut empty = target();
    empty.ticket.clear();
    assert!(empty.validate_at(now(), &capabilities).is_err());

    let mut control = target();
    control.ticket = "ticket\nwith-control".to_string();
    assert!(control.validate_at(now(), &capabilities).is_err());

    let mut padded = target();
    padded.ticket = " leading-space".to_string();
    assert!(padded.validate_at(now(), &capabilities).is_err());

    let mut oversized = target();
    oversized.ticket = "t".repeat(16 * 1024 + 1);
    assert!(oversized.validate_at(now(), &capabilities).is_err());

    let mut source = source();
    source.ticket = "source\0ticket".to_string();
    assert!(source
        .validate_for(&target(), now() + Duration::seconds(1), &capabilities)
        .is_err());
}

#[test]
fn tickets_must_not_carry_sealed_model_state_persistence() {
    let capabilities = capabilities();
    let mut schema = target();
    schema.ticket = "prefix-a3s.power.sealed-model-state.v1-suffix".to_string();
    assert!(schema.validate_at(now(), &capabilities).is_err());

    let mut magic = target();
    magic.ticket = "adapter-A3SPST1-meta".to_string();
    assert!(magic.validate_at(now(), &capabilities).is_err());

    let mut base64_magic = target();
    base64_magic.ticket = "QTNTUFNUMQA=opaque".to_string();
    assert!(base64_magic.validate_at(now(), &capabilities).is_err());

    let mut source = source();
    source.ticket = "a3s.power.sealed-model-state".to_string();
    assert!(source
        .validate_for(&target(), now() + Duration::seconds(1), &capabilities)
        .is_err());
}

/// First-principles property: wire tickets stay opaque adapter metadata.
///
/// They are intentionally **not** force-sealed with `SealedStateEnvelope`
/// (host buffers seal; tickets do not). Every sealed-persistence marker,
/// control character, padding, and oversize encoding must fail closed on both
/// target and source descriptors, while honest connection-metadata tickets
/// continue to validate without any sealed-envelope round trip.
#[test]
fn wire_ticket_opaque_metadata_fail_closed_property() {
    const MAX_TICKET_BYTES: usize = 16 * 1024;
    let capabilities = capabilities();
    let validate_now = now() + Duration::seconds(1);

    let forbidden_markers = [
        "a3s.power.sealed-model-state",
        "a3s.power.sealed-model-state.v1",
        "A3SPST1",
        "QTNTUFNUMQA",
        "QTNTUFNUMQA=",
        "4133535053543100",
        "4133535053543100AbC",
    ];
    for marker in forbidden_markers {
        for ticket in [
            marker.to_string(),
            format!("head-{marker}"),
            format!("{marker}-tail"),
            format!("x{marker}y"),
        ] {
            let mut probe = target();
            probe.ticket = ticket.clone();
            assert!(
                probe.validate_at(now(), &capabilities).is_err(),
                "target must reject sealed-persistence marker in {ticket:?}"
            );

            let mut probe = source();
            probe.ticket = ticket.clone();
            assert!(
                probe
                    .validate_for(&target(), validate_now, &capabilities)
                    .is_err(),
                "source must reject sealed-persistence marker in {ticket:?}"
            );
        }
    }

    for control in ['\0', '\n', '\r', '\t', '\u{7f}'] {
        let ticket = format!("ticket{control}meta");
        let mut probe = target();
        probe.ticket = ticket.clone();
        assert!(
            probe.validate_at(now(), &capabilities).is_err(),
            "target must reject control {control:?}"
        );
        let mut probe = source();
        probe.ticket = ticket;
        assert!(
            probe
                .validate_for(&target(), validate_now, &capabilities)
                .is_err(),
            "source must reject control {control:?}"
        );
    }

    for ticket in [
        String::new(),
        " leading".to_string(),
        "trailing ".to_string(),
        "t".repeat(MAX_TICKET_BYTES + 1),
        // Oversized payloads must not become a KV-byte smuggling channel.
        "k".repeat(MAX_TICKET_BYTES + 64),
    ] {
        let mut probe = target();
        probe.ticket = ticket.clone();
        assert!(
            probe.validate_at(now(), &capabilities).is_err(),
            "target must reject corrupt shape {ticket:?}"
        );
        let mut probe = source();
        probe.ticket = ticket;
        assert!(
            probe
                .validate_for(&target(), validate_now, &capabilities)
                .is_err(),
            "source must reject corrupt ticket shape"
        );
    }

    // Honest opaque connection metadata — no SealedStateEnvelope required.
    for ticket in [
        "decode-adapter-ticket".to_string(),
        "prefill-adapter-ticket".to_string(),
        "buffered-host-loopback-target:00000000-0000-0000-0000-000000000001".to_string(),
        format!(
            "{{\"schema\":\"a3s.power.buffered-host-loopback-source.v1\",\"address\":\"127.0.0.1:9\"}}"
        ),
        "a".repeat(MAX_TICKET_BYTES),
        "near-miss-A3SPST-without-final-1".to_string(),
        "near-miss-QTNTUFNUMQ-truncated".to_string(),
        "near-miss-41335350535431-without-00".to_string(),
    ] {
        let mut probe = target();
        probe.ticket = ticket.clone();
        assert!(
            probe.validate_at(now(), &capabilities).is_ok(),
            "honest opaque target ticket must validate: {ticket}"
        );
        let mut probe = source();
        probe.ticket = ticket;
        assert!(
            probe
                .validate_for(&target(), validate_now, &capabilities)
                .is_ok(),
            "honest opaque source ticket must validate without force-sealing"
        );
    }
}

#[cfg(feature = "embedded-inference")]
#[test]
fn sealed_host_buffer_persistence_bytes_never_validate_as_wire_tickets() {
    use crate::inference::{
        InferenceLimits, SealedStateKey, SealedStateRollbackPolicy, SealedStateScope,
        SealedStateStore,
    };
    use crate::serving::{
        seal_transfer_host_buffer, sealed_binding_for_transfer_host_buffer, StateTransferBinding,
        TransferHostBufferIdentity,
    };
    use tokio_util::sync::CancellationToken;

    let capabilities = capabilities();
    let small_binding = StateTransferBinding {
        model_sha256: "1".repeat(64),
        execution_sha256: "2".repeat(64),
        layout_sha256: "3".repeat(64),
        state_kind: StateKind::KvCache,
        token_count: 16,
        state_bytes: 32,
    };
    let identity = TransferHostBufferIdentity {
        transfer_id: Uuid::from_u128(11),
        source_worker_epoch: Uuid::from_u128(22),
        destination_worker_epoch: Uuid::from_u128(33),
        binding: small_binding.clone(),
        generation: 7,
    };
    let limits = InferenceLimits {
        max_state_bytes: 1_024,
        ..InferenceLimits::default()
    };
    let key = SealedStateKey::from_bytes([0x71; 32]);
    let state = vec![0xab; 32];
    let cancellation = CancellationToken::new();
    let envelope =
        seal_transfer_host_buffer(&identity, &state, &key, &limits, &cancellation).unwrap();

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("must-not-become-ticket.bin");
    let store = SealedStateStore::new(&path).unwrap();
    let sealed_binding = sealed_binding_for_transfer_host_buffer(&identity, &limits).unwrap();
    store
        .commit(
            &envelope,
            &sealed_binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(7),
            &limits,
            &cancellation,
        )
        .unwrap();
    let sealed_bytes = std::fs::read(&path).expect("committed host-buffer bytes");
    assert!(
        sealed_bytes.windows(7).any(|window| window == b"A3SPST1"),
        "committed host-buffer file must carry sealed-model-state MAGIC"
    );

    // Hex-encoded sealed persistence must not validate as a wire ticket.
    let mut hex_ticket = target();
    hex_ticket.ticket = hex::encode(&sealed_bytes[..sealed_bytes.len().min(64)]);
    assert!(
        hex_ticket.validate_at(now(), &capabilities).is_err(),
        "hex-encoded sealed host-buffer bytes must fail closed as a ticket"
    );

    // Schema name alone is also rejected (tickets are not sealed envelopes).
    let mut schema_ticket = target();
    schema_ticket.ticket = "a3s.power.sealed-model-state.v1".to_string();
    assert!(schema_ticket.validate_at(now(), &capabilities).is_err());
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
