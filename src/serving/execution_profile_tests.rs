use super::*;

fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

fn profile(role: DisaggregatedServingRole) -> ServingExecutionProfile {
    ServingExecutionProfile::prefill_decode(PrefillDecodeExecutionProfile {
        role,
        model: "internal/model-v1".to_string(),
        model_sha256: digest('1'),
        backend: "llama.cpp".to_string(),
        backend_sha256: digest('2'),
        execution_sha256: digest('3'),
        device_sha256: digest('4'),
        layout_sha256: digest('5'),
        peer_set_sha256: digest('6'),
        generation: 7,
        protocol: StateTransferProtocol::DirectDeviceMemoryPullV1,
        state_kind: StateKind::KvCache,
        max_state_bytes: 8 * 1024 * 1024 * 1024,
        max_inflight_transfers: 32,
        transfer_timeout_ms: 30_000,
        cancellation_timeout_ms: 5_000,
        privacy: ServingPrivacyMode::AuthenticatedEncryptedTransport,
        privacy_policy_sha256: digest('7'),
        attestation_policy_sha256: Some(digest('8')),
    })
    .unwrap()
}

fn binding() -> StateTransferBinding {
    StateTransferBinding {
        model_sha256: digest('1'),
        execution_sha256: digest('3'),
        layout_sha256: digest('5'),
        state_kind: StateKind::KvCache,
        token_count: 4_096,
        state_bytes: 1024 * 1024,
    }
}

#[test]
fn aggregated_profile_is_the_safe_canonical_default() {
    let profile = ServingExecutionProfile::default();
    profile.validate().unwrap();
    assert!(profile.is_aggregated());
    assert_eq!(profile.phase(), ServingPhase::Aggregated);
    assert_eq!(
        serde_json::to_value(&profile).unwrap(),
        serde_json::json!({ "profile": "aggregated" })
    );
    assert_eq!(profile.sha256().unwrap().len(), 64);
}

#[test]
fn prefill_decode_profile_binds_every_static_execution_invariant() {
    for role in [
        DisaggregatedServingRole::Prefill,
        DisaggregatedServingRole::Decode,
    ] {
        let profile = profile(role);
        profile.validate().unwrap();
        assert!(!profile.is_aggregated());
        assert_eq!(profile.phase(), role.into());
        profile.validate_state_binding(&binding()).unwrap();
    }

    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["unexpected"] = serde_json::json!(true);
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
}

#[test]
fn prefill_decode_profile_rejects_noncanonical_or_unbounded_values() {
    let mut invalid = profile(DisaggregatedServingRole::Prefill);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut invalid {
        execution.model_sha256 = "A".repeat(64);
    }
    assert!(invalid.validate().is_err());

    let mut invalid = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut invalid {
        execution.cancellation_timeout_ms = execution.transfer_timeout_ms + 1;
    }
    assert!(invalid.validate().is_err());

    let mut invalid = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut invalid {
        execution.generation = 0;
    }
    assert!(invalid.validate().is_err());

    let mut invalid = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut invalid {
        execution.privacy = ServingPrivacyMode::AttestedPrivateFabric;
        execution.attestation_policy_sha256 = None;
    }
    assert!(invalid.validate().is_err());
}

#[test]
fn state_binding_must_match_model_execution_layout_kind_and_byte_limit() {
    let profile = profile(DisaggregatedServingRole::Decode);
    profile.validate_state_binding(&binding()).unwrap();

    let mut mismatched = binding();
    mismatched.layout_sha256 = digest('9');
    assert!(profile.validate_state_binding(&mismatched).is_err());

    let mut oversized = binding();
    oversized.state_bytes = 9 * 1024 * 1024 * 1024;
    assert!(profile.validate_state_binding(&oversized).is_err());
}

#[test]
fn adapter_capabilities_must_be_bound_to_the_exact_profile() {
    let profile = profile(DisaggregatedServingRole::Decode);
    let capabilities = StateTransferCapabilities {
        execution_profile_sha256: profile.sha256().unwrap(),
        phases: vec![ServingPhase::Prefill, ServingPhase::Decode],
        protocols: vec![StateTransferProtocol::DirectDeviceMemoryPullV1],
        max_transfer_bytes: 16 * 1024 * 1024 * 1024,
        max_inflight_transfers: 64,
    };
    profile
        .validate_state_transfer_capabilities(&capabilities)
        .unwrap();

    let mut wrong_profile = capabilities.clone();
    wrong_profile.execution_profile_sha256 = digest('a');
    assert!(profile
        .validate_state_transfer_capabilities(&wrong_profile)
        .is_err());

    let mut insufficient = capabilities;
    insufficient.max_inflight_transfers = 1;
    assert!(profile
        .validate_state_transfer_capabilities(&insufficient)
        .is_err());
}

#[test]
fn execution_profile_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ServingExecutionProfile>();
}
