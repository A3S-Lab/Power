use std::path::{Path, PathBuf};

use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use super::*;
use crate::tee::attestation::{
    build_claims_report_data, AttestationClaimsV2, AttestationReport, ModelDigestClaim,
    ModelDigestKind, TeeType,
};

fn digest(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn test_limits(max_state_bytes: u64) -> InferenceLimits {
    InferenceLimits {
        max_state_bytes,
        ..InferenceLimits::default()
    }
}

fn binding(limits: &InferenceLimits) -> SealedStateBinding {
    SealedStateBinding::for_identifier(digest('a'), digest('b'), b"conversation-alpha", limits)
        .unwrap()
}

fn export_report(weights_sha256: &str) -> AttestationReport {
    let nonce = vec![0x11; 32];
    let claims = AttestationClaimsV2::new(TeeType::SevSnp)
        .with_nonce(Some(&nonce))
        .with_model(ModelDigestClaim {
            name: "embedded-model".to_string(),
            kind: ModelDigestKind::PlaintextWeightsSha256,
            digest: hex::decode(weights_sha256).unwrap(),
            plaintext_digest: None,
            ciphertext_digest: None,
        });
    AttestationReport {
        version: "1.0".to_string(),
        tee_type: TeeType::SevSnp,
        report_data: build_claims_report_data(&claims).unwrap(),
        measurement: vec![0x22; 48],
        raw_report: Some(vec![0x33; 64]),
        timestamp: chrono::Utc::now(),
        nonce: Some(nonce),
        claims: Some(claims),
    }
}

fn test_authorization(weights_sha256: &str, policy: char) -> TeeStateExportAuthorization {
    TeeStateExportAuthorization::from_verified_attestation_report(
        &export_report(weights_sha256),
        &digest(policy),
    )
    .unwrap()
}

fn seal_local(
    generation: u64,
    state: &[u8],
    binding: &SealedStateBinding,
    key: &SealedStateKey,
    limits: &InferenceLimits,
) -> SealedStateEnvelope {
    SealedStateEnvelope::seal(
        binding,
        generation,
        state,
        key,
        SealedStateScope::TeeLocal,
        limits,
        &CancellationToken::new(),
    )
    .unwrap()
}

fn control_path(target: &Path, suffix: &str) -> PathBuf {
    let name = target.file_name().unwrap().to_str().unwrap();
    target
        .parent()
        .unwrap()
        .join(format!(".{name}.a3s-power-sealed-state.{suffix}"))
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

#[test]
fn local_envelope_round_trips_into_zeroizing_state_and_cannot_export() {
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x44; 32]);
    let cancellation = CancellationToken::new();
    let envelope = SealedStateEnvelope::seal(
        &binding,
        7,
        b"opaque-model-owned-kv",
        &key,
        SealedStateScope::TeeLocal,
        &limits,
        &cancellation,
    )
    .unwrap();

    assert_eq!(envelope.generation(), 7);
    assert_eq!(envelope.export_scope(), SealedStateExportScope::TeeLocal);
    let opened = envelope
        .open(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(7),
            &limits,
            &cancellation,
        )
        .unwrap();
    assert_eq!(opened.as_bytes(), b"opaque-model-owned-kv");
    let state: Zeroizing<Vec<u8>> = opened.into_bytes();
    assert_eq!(state.as_slice(), b"opaque-model-owned-kv");

    let authorization = test_authorization(binding.weights_sha256(), 'd');
    assert!(envelope.export(&authorization).is_err());
}

#[test]
fn authorized_export_is_attestation_and_model_bound() {
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x45; 32]);
    let authorization = test_authorization(binding.weights_sha256(), 'd');
    let cancellation = CancellationToken::new();
    let envelope = SealedStateEnvelope::seal(
        &binding,
        9,
        b"authorized-state",
        &key,
        SealedStateScope::TeeAuthorizedExport(&authorization),
        &limits,
        &cancellation,
    )
    .unwrap();

    let encoded = envelope.export(&authorization).unwrap();
    let imported = SealedStateEnvelope::import(&encoded, &limits).unwrap();
    assert_eq!(
        imported.export_scope(),
        SealedStateExportScope::TeeAuthorized
    );
    assert_eq!(
        imported
            .open(
                &binding,
                &key,
                SealedStateScope::TeeAuthorizedExport(&authorization),
                SealedStateRollbackPolicy::new(9),
                &limits,
                &cancellation,
            )
            .unwrap()
            .as_bytes(),
        b"authorized-state"
    );
    assert!(imported
        .open(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(0),
            &limits,
            &cancellation,
        )
        .is_err());

    let wrong_authorization = test_authorization(binding.weights_sha256(), 'e');
    assert!(imported.export(&wrong_authorization).is_err());
}

#[test]
fn wrong_binding_tampering_truncation_and_oversize_fail_closed() {
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x46; 32]);
    let authorization = test_authorization(binding.weights_sha256(), 'd');
    let cancellation = CancellationToken::new();
    let envelope = SealedStateEnvelope::seal(
        &binding,
        3,
        b"sensitive-state",
        &key,
        SealedStateScope::TeeAuthorizedExport(&authorization),
        &limits,
        &cancellation,
    )
    .unwrap();

    for wrong in [
        SealedStateBinding::for_identifier(
            digest('c'),
            digest('b'),
            b"conversation-alpha",
            &limits,
        )
        .unwrap(),
        SealedStateBinding::for_identifier(
            digest('a'),
            digest('c'),
            b"conversation-alpha",
            &limits,
        )
        .unwrap(),
        SealedStateBinding::for_identifier(digest('a'), digest('b'), b"conversation-beta", &limits)
            .unwrap(),
    ] {
        assert!(envelope
            .open(
                &wrong,
                &key,
                SealedStateScope::TeeAuthorizedExport(&authorization),
                SealedStateRollbackPolicy::new(0),
                &limits,
                &cancellation,
            )
            .is_err());
    }

    let mut tampered = envelope.export(&authorization).unwrap();
    *tampered.last_mut().unwrap() ^= 0x80;
    let tampered = SealedStateEnvelope::import(&tampered, &limits).unwrap();
    assert!(tampered
        .open(
            &binding,
            &key,
            SealedStateScope::TeeAuthorizedExport(&authorization),
            SealedStateRollbackPolicy::new(0),
            &limits,
            &cancellation,
        )
        .is_err());

    let encoded = envelope.export(&authorization).unwrap();
    assert!(SealedStateEnvelope::import(&encoded[..encoded.len() - 1], &limits).is_err());
    let smaller_limits = test_limits(3);
    assert!(SealedStateEnvelope::import(&encoded, &smaller_limits).is_err());
}

#[test]
fn fresh_nonces_wrong_keys_and_authenticated_header_tampering_are_enforced() {
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x4d; 32]);
    let wrong_key = SealedStateKey::from_bytes([0x4e; 32]);
    let authorization = test_authorization(binding.weights_sha256(), 'd');
    let cancellation = CancellationToken::new();
    let seal = || {
        SealedStateEnvelope::seal(
            &binding,
            1,
            b"same-state-and-generation",
            &key,
            SealedStateScope::TeeAuthorizedExport(&authorization),
            &limits,
            &cancellation,
        )
        .unwrap()
    };
    let first = seal();
    let second = seal();
    assert_ne!(
        first.export(&authorization).unwrap().as_slice(),
        second.export(&authorization).unwrap().as_slice()
    );
    assert!(first
        .open(
            &binding,
            &wrong_key,
            SealedStateScope::TeeAuthorizedExport(&authorization),
            SealedStateRollbackPolicy::new(0),
            &limits,
            &cancellation,
        )
        .is_err());

    let mut changed_generation = first.export(&authorization).unwrap();
    // Generation starts after magic, version, scope, reserved, and header size.
    changed_generation[17] ^= 0x01;
    let changed_generation = SealedStateEnvelope::import(&changed_generation, &limits).unwrap();
    assert!(changed_generation
        .open(
            &binding,
            &key,
            SealedStateScope::TeeAuthorizedExport(&authorization),
            SealedStateRollbackPolicy::new(0),
            &limits,
            &cancellation,
        )
        .is_err());
}

#[test]
fn store_enforces_monotonic_generations_and_external_rollback_floor() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("session.state");
    let store = SealedStateStore::new(&target).unwrap();
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x47; 32]);
    let cancellation = CancellationToken::new();
    let floor = SealedStateRollbackPolicy::new(0);

    let generation_five = seal_local(5, b"generation-five", &binding, &key, &limits);
    store
        .commit(
            &generation_five,
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .unwrap();
    assert!(store
        .commit(
            &generation_five,
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .is_err());
    let generation_four = seal_local(4, b"generation-four", &binding, &key, &limits);
    assert!(store
        .commit(
            &generation_four,
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .is_err());

    assert!(store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(6),
            &limits,
            &cancellation,
        )
        .is_err());
    let loaded = store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(5),
            &limits,
            &cancellation,
        )
        .unwrap()
        .unwrap();
    assert_eq!(loaded.generation(), 5);
    assert_eq!(loaded.as_bytes(), b"generation-five");
    assert_eq!(loaded.source(), SealedStateRecoverySource::Primary);
}

#[test]
fn interrupted_publication_recovers_the_last_committed_state() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("session.state");
    let backup = control_path(&target, "backup");
    let pending = control_path(&target, "pending");
    let store = SealedStateStore::new(&target).unwrap();
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x48; 32]);
    let cancellation = CancellationToken::new();
    let floor = SealedStateRollbackPolicy::new(0);

    store
        .commit(
            &seal_local(1, b"committed-one", &binding, &key, &limits),
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .unwrap();
    std::fs::rename(&target, &backup).unwrap();
    std::fs::write(&pending, b"interrupted-new-envelope").unwrap();

    let recovered = store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .unwrap()
        .unwrap();
    assert_eq!(recovered.source(), SealedStateRecoverySource::Backup);
    assert_eq!(recovered.generation(), 1);
    assert_eq!(recovered.as_bytes(), b"committed-one");
    drop(recovered);

    store
        .commit(
            &seal_local(2, b"committed-two", &binding, &key, &limits),
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .unwrap();
    assert!(!pending.exists());
    let recovered = store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            floor,
            &limits,
            &cancellation,
        )
        .unwrap()
        .unwrap();
    assert_eq!(recovered.source(), SealedStateRecoverySource::Primary);
    assert_eq!(recovered.generation(), 2);
    assert_eq!(recovered.as_bytes(), b"committed-two");
}

#[test]
fn corrupted_primary_uses_backup_only_when_the_rollback_floor_allows_it() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("session.state");
    let store = SealedStateStore::new(&target).unwrap();
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x49; 32]);
    let cancellation = CancellationToken::new();
    let floor = SealedStateRollbackPolicy::new(0);

    for (generation, value) in [(1, b"one".as_slice()), (2, b"two".as_slice())] {
        store
            .commit(
                &seal_local(generation, value, &binding, &key, &limits),
                &binding,
                &key,
                SealedStateScope::TeeLocal,
                floor,
                &limits,
                &cancellation,
            )
            .unwrap();
    }
    std::fs::write(&target, b"corrupt-primary").unwrap();

    let recovered = store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(1),
            &limits,
            &cancellation,
        )
        .unwrap()
        .unwrap();
    assert_eq!(recovered.source(), SealedStateRecoverySource::Backup);
    assert_eq!(recovered.generation(), 1);
    drop(recovered);
    assert!(store
        .load(
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(2),
            &limits,
            &cancellation,
        )
        .is_err());
}

#[test]
fn cancellation_bounds_and_wrong_store_binding_are_rejected() {
    let limits = test_limits(8);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x4a; 32]);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(SealedStateEnvelope::seal(
        &binding,
        1,
        b"state",
        &key,
        SealedStateScope::TeeLocal,
        &limits,
        &cancelled,
    )
    .is_err());
    assert!(SealedStateEnvelope::seal(
        &binding,
        0,
        b"state",
        &key,
        SealedStateScope::TeeLocal,
        &limits,
        &CancellationToken::new(),
    )
    .is_err());
    assert!(SealedStateEnvelope::seal(
        &binding,
        1,
        b"state-too-large",
        &key,
        SealedStateScope::TeeLocal,
        &limits,
        &CancellationToken::new(),
    )
    .is_err());

    let directory = tempfile::tempdir().unwrap();
    let store = SealedStateStore::new(directory.path().join("session.state")).unwrap();
    let active = CancellationToken::new();
    store
        .commit(
            &seal_local(1, b"state", &binding, &key, &limits),
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(0),
            &limits,
            &active,
        )
        .unwrap();
    assert!(store
        .commit(
            &seal_local(2, b"next", &binding, &key, &limits),
            &binding,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(0),
            &limits,
            &cancelled,
        )
        .is_err());
    assert_eq!(
        store
            .load(
                &binding,
                &key,
                SealedStateScope::TeeLocal,
                SealedStateRollbackPolicy::new(1),
                &limits,
                &active,
            )
            .unwrap()
            .unwrap()
            .generation(),
        1
    );
    let wrong = SealedStateBinding::for_identifier(
        digest('c'),
        digest('b'),
        b"conversation-alpha",
        &limits,
    )
    .unwrap();
    assert!(store
        .load(
            &wrong,
            &key,
            SealedStateScope::TeeLocal,
            SealedStateRollbackPolicy::new(0),
            &limits,
            &active,
        )
        .is_err());
}

#[test]
fn export_authorization_rejects_simulation_and_cross_model_use() {
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let mut simulated = export_report(binding.weights_sha256());
    simulated.tee_type = TeeType::Simulated;
    simulated.claims.as_mut().unwrap().tee_type = TeeType::Simulated;
    simulated.report_data = build_claims_report_data(simulated.claims.as_ref().unwrap()).unwrap();
    assert!(
        TeeStateExportAuthorization::from_verified_attestation_report(&simulated, &digest('d'))
            .is_err()
    );

    let authorization = test_authorization(binding.weights_sha256(), 'd');
    let wrong_model = SealedStateBinding::for_identifier(
        digest('c'),
        digest('b'),
        b"conversation-alpha",
        &limits,
    )
    .unwrap();
    assert!(SealedStateEnvelope::seal(
        &wrong_model,
        1,
        b"state",
        &SealedStateKey::from_bytes([0x4b; 32]),
        SealedStateScope::TeeAuthorizedExport(&authorization),
        &limits,
        &CancellationToken::new(),
    )
    .is_err());
}

#[test]
fn exported_envelopes_are_digest_only_and_public_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SealedStateBinding>();
    assert_send_sync::<SealedStateKey>();
    assert_send_sync::<SealedStateEnvelope>();
    assert_send_sync::<OpenedSealedState>();
    assert_send_sync::<RecoveredSealedState>();
    assert_send_sync::<SealedStateStore>();
    assert_send_sync::<TeeStateExportAuthorization>();

    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("secret-session-name.state");
    let store = SealedStateStore::new(&target).unwrap();
    let limits = test_limits(1_024);
    let binding = binding(&limits);
    let key = SealedStateKey::from_bytes([0x4c; 32]);
    let authorization = test_authorization(binding.weights_sha256(), 'd');
    let plaintext = b"private conversation tokens and kv values";
    let envelope = SealedStateEnvelope::seal(
        &binding,
        1,
        plaintext,
        &key,
        SealedStateScope::TeeAuthorizedExport(&authorization),
        &limits,
        &CancellationToken::new(),
    )
    .unwrap();
    let encoded = envelope.export(&authorization).unwrap();

    assert!(!contains_subslice(&encoded, plaintext));
    assert!(!contains_subslice(&encoded, b"conversation-alpha"));
    assert!(!format!("{envelope:?}").contains("private conversation"));
    assert!(!format!("{store:?}").contains("secret-session-name"));
    assert!(!format!("{key:?}").contains("76"));
}
