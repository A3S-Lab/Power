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
    assert!(profile.composition_transport().is_none());
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

    let mut invalid = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut invalid {
        execution.transport = Some(ServingCompositionTransport::BufferedHostLoopback);
    }
    let err = invalid.validate().unwrap_err();
    assert!(err.to_string().contains("buffered-host-memory-pull-v1"));
}

#[test]
fn direct_device_memory_pull_transport_requires_matching_protocol() {
    let mut matching = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut matching {
        execution.transport = Some(ServingCompositionTransport::DirectDeviceMemoryPull);
    }
    matching.validate().unwrap();
    assert!(!matching.may_advertise_prefill_decode());
    assert_eq!(
        matching.composition_transport(),
        Some(ServingCompositionTransport::DirectDeviceMemoryPull)
    );

    let mut mismatched = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut mismatched {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.transport = Some(ServingCompositionTransport::DirectDeviceMemoryPull);
    }
    let err = mismatched.validate().unwrap_err();
    assert!(err.to_string().contains("direct-device-memory-pull-v1"));
}

#[test]
fn unknown_composition_transport_fails_closed_at_deserialization() {
    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["transport"] = serde_json::json!("rdma-nixl");
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
}

#[test]
fn buffered_host_loopback_may_advertise_prefill_decode_when_protocol_matches() {
    let mut loopback = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut loopback {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.transport = Some(ServingCompositionTransport::BufferedHostLoopback);
    }
    loopback.validate().unwrap();
    assert!(loopback.may_advertise_prefill_decode());
    assert!(profile(DisaggregatedServingRole::Decode).may_advertise_prefill_decode());
}

#[test]
fn backend_owned_phase_executor_never_advertises_prefill_decode() {
    let mut backend_owned = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut backend_owned {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.transport = Some(ServingCompositionTransport::BufferedHostLoopback);
        execution.phase_executor = Some(ServingCompositionPhaseExecutor::BackendOwned);
    }
    backend_owned.validate().unwrap();
    assert!(!backend_owned.may_advertise_prefill_decode());
    assert_eq!(
        backend_owned.composition_phase_executor(),
        Some(ServingCompositionPhaseExecutor::BackendOwned)
    );

    let mut with_ownership = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut with_ownership {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.transport = Some(ServingCompositionTransport::BufferedHostLoopback);
        execution.phase_executor = Some(ServingCompositionPhaseExecutor::BackendOwned);
        execution.state_ownership = Some(ServingCompositionStateOwnership::ProfileBound);
    }
    with_ownership.validate().unwrap();
    assert_eq!(
        with_ownership.composition_state_ownership(),
        Some(ServingCompositionStateOwnership::ProfileBound)
    );
    assert!(!with_ownership.may_advertise_prefill_decode());

    let mut missing_transport = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut missing_transport {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.phase_executor = Some(ServingCompositionPhaseExecutor::BackendOwned);
    }
    let err = missing_transport.validate().unwrap_err();
    assert!(err.to_string().contains("buffered-host-loopback"));
}

#[test]
fn profile_bound_state_ownership_requires_backend_owned_phase() {
    let mut profile = profile(DisaggregatedServingRole::Decode);
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut profile {
        execution.protocol = StateTransferProtocol::BufferedHostMemoryPullV1;
        execution.transport = Some(ServingCompositionTransport::BufferedHostLoopback);
        execution.state_ownership = Some(ServingCompositionStateOwnership::ProfileBound);
    }
    let err = profile.validate().unwrap_err();
    assert!(err.to_string().contains("phase_executor = backend-owned"));
}

#[test]
fn unknown_composition_state_ownership_fails_closed_at_deserialization() {
    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["state_ownership"] = serde_json::json!("llama-cpp-kv");
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
}

#[test]
fn unknown_composition_phase_executor_fails_closed_at_deserialization() {
    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["phase_executor"] = serde_json::json!("llama-cpp-ready");
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
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
fn deployment_identity_is_shared_across_roles_but_rejects_stale_generation_and_peer_set() {
    let decode = profile(DisaggregatedServingRole::Decode);
    let prefill = profile(DisaggregatedServingRole::Prefill);
    let identity = decode.deployment_identity().unwrap();
    assert_eq!(identity.generation, 7);
    assert_eq!(identity.peer_set_sha256, digest('6'));
    // Roles differ, so full profile digests differ, but deployment identity matches.
    assert_ne!(decode.sha256().unwrap(), prefill.sha256().unwrap());
    prefill.validate_deployment_identity(&identity).unwrap();

    let mut stale = identity.clone();
    stale.generation = 8;
    assert!(decode.validate_deployment_identity(&stale).is_err());

    let mut foreign_peers = identity;
    foreign_peers.peer_set_sha256 = digest('a');
    assert!(decode.validate_deployment_identity(&foreign_peers).is_err());
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
fn private_weight_cache_mode_fails_closed_at_deserialization() {
    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["weight_cache"] = serde_json::json!("private-weight-cache");
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
}

#[test]
fn private_session_pool_mode_fails_closed_at_deserialization() {
    let mut document = serde_json::to_value(profile(DisaggregatedServingRole::Decode)).unwrap();
    document["session_pool"] = serde_json::json!("private-session-pool");
    assert!(serde_json::from_value::<ServingExecutionProfile>(document).is_err());
}

#[test]
fn phase_executor_must_reuse_shared_weight_hierarchy_binding() {
    let profile = profile(DisaggregatedServingRole::Prefill);
    let matching = PhaseExecutorCapabilities::for_profile(&profile).unwrap();
    profile
        .validate_phase_executor_capabilities(&matching)
        .unwrap();

    let mut mismatched = matching.clone();
    mismatched.residency_policy_sha256 = Some(digest('9'));
    let error = profile
        .validate_phase_executor_capabilities(&mismatched)
        .unwrap_err();
    assert!(error.to_string().contains("phase executor"));

    let mut pinned = profile.clone();
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut pinned {
        execution.residency_policy_sha256 = Some(digest('b'));
    }
    // Profile digest changes when residency policy is pinned, so rebuild caps.
    let pinned_caps = PhaseExecutorCapabilities::for_profile(&pinned).unwrap();
    pinned
        .validate_phase_executor_capabilities(&pinned_caps)
        .unwrap();
    let stale = PhaseExecutorCapabilities::for_profile(&profile).unwrap();
    assert!(pinned.validate_phase_executor_capabilities(&stale).is_err());
}

#[test]
fn shared_weight_cache_default_keeps_profile_digest_stable() {
    let with_defaults = profile(DisaggregatedServingRole::Decode);
    let mut explicit = serde_json::to_value(&with_defaults).unwrap();
    assert!(explicit.get("weight_cache").is_none());
    assert!(explicit.get("residency_policy_sha256").is_none());
    assert!(explicit.get("session_pool").is_none());
    assert!(explicit.get("session_pool_policy_sha256").is_none());
    explicit["weight_cache"] = serde_json::json!("shared-weight-hierarchy");
    explicit["session_pool"] = serde_json::json!("shared-session-pool");
    let round_trip: ServingExecutionProfile = serde_json::from_value(explicit).unwrap();
    assert_eq!(
        with_defaults.sha256().unwrap(),
        round_trip.sha256().unwrap()
    );
}

#[test]
fn phase_executor_must_reuse_shared_session_pool_binding() {
    let profile = profile(DisaggregatedServingRole::Prefill);
    let matching = PhaseExecutorCapabilities::for_profile(&profile).unwrap();
    profile
        .validate_phase_executor_capabilities(&matching)
        .unwrap();

    let mut mismatched = matching.clone();
    mismatched.session_pool_policy_sha256 = Some(digest('9'));
    let error = profile
        .validate_phase_executor_capabilities(&mismatched)
        .unwrap_err();
    assert!(error.to_string().contains("phase executor"));

    let mut pinned = profile.clone();
    if let ServingExecutionProfile::PrefillDecode { execution } = &mut pinned {
        execution.session_pool_policy_sha256 = Some(digest('c'));
    }
    let pinned_caps = PhaseExecutorCapabilities::for_profile(&pinned).unwrap();
    pinned
        .validate_phase_executor_capabilities(&pinned_caps)
        .unwrap();
    let stale = PhaseExecutorCapabilities::for_profile(&profile).unwrap();
    assert!(pinned.validate_phase_executor_capabilities(&stale).is_err());
}

#[test]
fn execution_profile_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ServingExecutionProfile>();
    assert_send_sync::<PhaseWeightCacheMode>();
    assert_send_sync::<PhaseSessionPoolMode>();
    assert_send_sync::<ServingCompositionTransport>();
    assert_send_sync::<ServingCompositionPhaseExecutor>();
    assert_send_sync::<ServingCompositionStateOwnership>();
}
