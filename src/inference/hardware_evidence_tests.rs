use super::{
    ExecutionDigest, HardwareEvidenceBinding, HardwareEvidenceBundle, ModelParityArtifact,
    RuntimeDeviceIdentity, RuntimeDeviceKind, StorageBenchmarkReport, StorageBenchmarkSample,
    StorageBenchmarkSource, StorageBenchmarkSystem, StorageCachePreparation, StorageCacheState,
    TuningCandidateEvidence, TuningOrderedEvidence, TuningProfileEvidence, TuningProfileOutcome,
    TuningProfilePolicy, TuningRoundEvidence, TuningRunEvidence, WeightReadStrategy,
    WeightSourceCoverage, WeightSourceRepresentation, WeightSourceRole, WeightSourceWeighting,
};

pub(super) fn digest(character: char) -> String {
    character.to_string().repeat(64)
}

pub(super) fn system() -> StorageBenchmarkSystem {
    StorageBenchmarkSystem {
        os: "linux".to_string(),
        architecture: "x86_64".to_string(),
        cpu_model: "reviewed-test-cpu".to_string(),
        logical_cpus: 16,
        ram_bytes: 64 * 1024 * 1024 * 1024,
        filesystem_class: "ext4".to_string(),
        device_class: "nvme-and-cpu".to_string(),
    }
}

pub(super) fn storage_report(strategy: WeightReadStrategy) -> StorageBenchmarkReport {
    StorageBenchmarkReport {
        schema: StorageBenchmarkReport::SCHEMA.to_string(),
        power_version: env!("CARGO_PKG_VERSION").to_string(),
        power_commit: "a".repeat(40),
        model_collection_sha256: digest('1'),
        sources: vec![StorageBenchmarkSource {
            index: 0,
            role: WeightSourceRole::Primary,
            root_count: 1,
            coverage: WeightSourceCoverage::Complete,
            read_strategy: strategy,
            representation: WeightSourceRepresentation::CanonicalSafeTensors,
            configured_read_weight: 1,
            effective_read_weight: 1,
            source_weighting: WeightSourceWeighting::Configured,
            validation_bytes_per_second: 100,
            io_block_size: 4096,
            verified_files: 1,
            verified_tensors: 1,
            verified_bytes: 4,
        }],
        system: system(),
        cache_state: StorageCacheState::Warm,
        cache_preparation: StorageCachePreparation::WarmSequence,
        cache_state_procedure:
            "one complete unmeasured tensor sequence immediately before measurement".to_string(),
        cache_state_verified: true,
        strategy,
        concurrency: 1,
        sequence_sha256: digest('2'),
        tensor_count: 1,
        requested_bytes_per_sample: 4,
        total_requested_bytes: 4,
        total_read_bytes: 4,
        integrity_open_nanos: 10,
        output_validation_nanos: 10,
        samples: vec![StorageBenchmarkSample {
            latency_nanos: 20,
            bytes_read: 4,
            bytes_per_second: 200,
            source_fallbacks: 0,
        }],
        output_sha256: digest('3'),
    }
}

pub(super) fn binding() -> HardwareEvidenceBinding {
    HardwareEvidenceBinding::new(
        env!("CARGO_PKG_VERSION"),
        "a".repeat(40),
        digest('1'),
        digest('4'),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        },
        &system(),
    )
    .unwrap()
}

fn tuning_run(
    binding: &super::TuningProfileBinding,
    configuration_sha256: &str,
    elapsed_nanos: u64,
) -> TuningRunEvidence {
    TuningRunEvidence {
        binding: binding.clone(),
        configuration_sha256: configuration_sha256.to_string(),
        output: ExecutionDigest::token_ids(&[7]),
        sample_count: 32,
        completed_units: 32,
        elapsed_nanos,
        cache_hits: 16,
        cache_requests: 32,
        p99_latency_nanos: 10,
    }
}

pub(super) fn tuning_evidence(
    platform: &HardwareEvidenceBinding,
    candidate_fast: bool,
) -> TuningProfileEvidence {
    let tuning_binding = platform
        .tuning_binding(ExecutionDigest::token_ids(&[1, 2, 3]))
        .unwrap();
    let baseline = digest('a');
    let candidate = digest('b');
    let candidate_elapsed = if candidate_fast { 100 } else { 220 };
    let rounds = [1_u32, 0]
        .into_iter()
        .map(|round| TuningRoundEvidence {
            round,
            baseline_then_candidate: TuningOrderedEvidence {
                first: tuning_run(&tuning_binding, &baseline, 200),
                second: tuning_run(&tuning_binding, &candidate, candidate_elapsed),
            },
            candidate_then_baseline: TuningOrderedEvidence {
                first: tuning_run(&tuning_binding, &candidate, candidate_elapsed),
                second: tuning_run(&tuning_binding, &baseline, 200),
            },
        })
        .collect();
    TuningProfileEvidence {
        schema: TuningProfileEvidence::SCHEMA.to_string(),
        binding: tuning_binding,
        baseline_configuration_sha256: baseline,
        candidates: vec![TuningCandidateEvidence {
            configuration_sha256: candidate,
            rounds,
        }],
    }
}

pub(super) fn parity_artifact(configuration_sha256: &str) -> ModelParityArtifact {
    ModelParityArtifact::new(
        binding(),
        digest('5'),
        digest('6'),
        digest('7'),
        configuration_sha256,
        ExecutionDigest::token_ids(&[8]),
        ExecutionDigest::token_ids(&[9]),
        ExecutionDigest::token_ids(&[9]),
    )
    .unwrap()
}

pub(super) fn bundle(candidate_fast: bool) -> HardwareEvidenceBundle {
    let binding = binding();
    let evidence = tuning_evidence(&binding, candidate_fast);
    let selected = if candidate_fast {
        digest('b')
    } else {
        digest('a')
    };
    HardwareEvidenceBundle::build(
        binding,
        vec![
            storage_report(WeightReadStrategy::PositionalBuffered),
            storage_report(WeightReadStrategy::Mmap),
        ],
        evidence,
        TuningProfilePolicy::default(),
        vec![parity_artifact(&selected)],
    )
    .unwrap()
}

#[test]
fn bundle_is_canonical_self_verifying_and_digest_pinned() {
    let bundle = bundle(true);
    assert_eq!(bundle.schema, HardwareEvidenceBundle::SCHEMA);
    assert_eq!(bundle.sha256.len(), 64);
    assert_eq!(
        bundle.tuning_decision.outcome,
        TuningProfileOutcome::CandidateSelected
    );
    bundle.verify().unwrap();
    bundle.verify_pinned(&bundle.sha256).unwrap();

    let restored: HardwareEvidenceBundle =
        serde_json::from_str(&serde_json::to_string(&bundle).unwrap()).unwrap();
    assert_eq!(restored, bundle);
    restored.verify().unwrap();
}

#[test]
fn input_order_does_not_change_the_canonical_bundle() {
    let first = bundle(true);
    let platform = binding();
    let second = HardwareEvidenceBundle::build(
        platform.clone(),
        vec![
            storage_report(WeightReadStrategy::Mmap),
            storage_report(WeightReadStrategy::PositionalBuffered),
        ],
        tuning_evidence(&platform, true),
        TuningProfilePolicy::default(),
        vec![parity_artifact(&digest('b'))],
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.sha256, second.sha256);
}

#[test]
fn negative_tuning_result_remains_reviewable_evidence() {
    let bundle = bundle(false);
    assert_eq!(
        bundle.tuning_decision.outcome,
        TuningProfileOutcome::BaselineRetainedNoEligibleCandidate
    );
    assert_eq!(
        bundle.tuning_decision.selected_configuration_sha256,
        digest('a')
    );
    bundle.verify().unwrap();
}

#[test]
fn binding_digests_are_canonical_and_domain_separated() {
    let binding = binding();
    assert_eq!(binding.runtime_sha256.len(), 64);
    assert_eq!(binding.device_sha256.len(), 64);
    assert_eq!(binding.environment_sha256.len(), 64);
    assert_ne!(binding.runtime_sha256, binding.device_sha256);
    assert_ne!(binding.device_sha256, binding.environment_sha256);
    let tuning = binding
        .tuning_binding(ExecutionDigest::token_ids(&[1]))
        .unwrap();
    assert_eq!(tuning.runtime_sha256, binding.runtime_sha256);
    assert_eq!(tuning.device_sha256, binding.device_sha256);
    assert_eq!(tuning.environment_sha256, binding.environment_sha256);
}

#[test]
fn debug_output_omits_named_hardware_and_model_owned_artifact_digests() {
    let bundle = bundle(true);
    let debug = format!("{bundle:?}");
    assert!(!debug.contains("reviewed-test-cpu"));
    assert!(!debug.contains(&digest('6')));
    assert!(!debug.contains(&digest('7')));
    assert!(debug.contains("storage_report_count"));
}

#[test]
fn public_hardware_evidence_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HardwareEvidenceBinding>();
    assert_send_sync::<ModelParityArtifact>();
    assert_send_sync::<HardwareEvidenceBundle>();
}
