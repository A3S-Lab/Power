use crate::admission::AdmissionSnapshot;

use super::{
    BoundedMemoryEvidence, CancellationContractEvidence, ExactFallbackEvidence,
    ExecutionBatchLifecycleEvidence, ExecutionDigest, ModelSessionPoolSnapshot, PeakMemoryEvidence,
    QueueExpiryEvidence, ReleaseCapture, ReleaseCaptureSecurity, ReleaseContractEvidence,
    ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleasePlatform, ReleasePlatformBinding,
    ReleaseRevisionBinding, ReplicaRecoveryEvidence, ResidentTensorSnapshot, RuntimeDeviceIdentity,
    RuntimeMemoryReservations, ShapeProfileBinding, ShapeProfileExecutionEvidence,
    ShapeProfileExecutionPath, ShapeProfileFallbackReason, TensorBatchBenchmarkReport,
};

const POWER_COMMIT: &str = "1a9504e58fc2751e016efede2fc006615a0b8cc2";
const COMPLETE_CAPTURE_COMMIT: &str = "6b7d6e5265b34c3e9e812c830ce22cc4a35940e5";

struct Fixture {
    report: TensorBatchBenchmarkReport,
    shape: ShapeProfileBinding,
    contracts: ReleaseContractEvidence,
}

fn digest(value: char) -> String {
    std::iter::repeat_n(value, 64).collect()
}

fn cpu_report() -> TensorBatchBenchmarkReport {
    serde_json::from_str(include_str!(
        "../../docs/benchmarks/release-gate-windows-20260821/cpu.json"
    ))
    .unwrap()
}

fn admission(
    peak_active: usize,
    peak_waiting: usize,
    admitted: u64,
    deadline_expirations: u64,
) -> AdmissionSnapshot {
    AdmissionSnapshot {
        active_limit: Some(1),
        waiting_limit: Some(1),
        active: 0,
        waiting: 0,
        peak_active,
        peak_waiting,
        admitted,
        queue_rejections: 0,
        cancelled_waiters: 0,
        deadline_expirations,
    }
}

fn pool(
    device: RuntimeDeviceIdentity,
    ready_replicas: usize,
    pending: usize,
    retirements: u64,
    reconstructions: u64,
) -> ModelSessionPoolSnapshot {
    ModelSessionPoolSnapshot {
        device,
        maximum_sessions: 1,
        maximum_resident_bytes: 1_024,
        registered_sessions: 1,
        ready_sessions: 1,
        maximum_replicas_per_session: 2,
        reserved_replicas: 2,
        ready_replicas,
        leased_replicas: 0,
        waiting_replica_requests: 0,
        expired_replica_requests: 0,
        replicas_pending_reconstruction: pending,
        replica_retirements: retirements,
        replica_reconstructions: reconstructions,
        reserved_bytes: 512,
        device_admission: admission(1, 0, 1, 0),
    }
}

fn fixture() -> Fixture {
    let report = cpu_report();
    report.verify().unwrap();
    let reservations = RuntimeMemoryReservations {
        host_fixed_bytes: 32,
        host_scratch_bytes: 128,
        device_fixed_bytes: 0,
        device_scratch_bytes: 0,
    };
    let shape = ShapeProfileBinding::new(
        report.binding.weights_sha256.clone(),
        digest('b'),
        report.binding.runtime_device,
        digest('c'),
        reservations,
        digest('d'),
    )
    .unwrap();
    let selection = ShapeProfileExecutionEvidence {
        schema: ShapeProfileExecutionEvidence::SCHEMA.to_string(),
        declaration_sha256: digest('e'),
        binding_sha256: shape.binding_sha256().unwrap(),
        request_sha256: digest('f'),
        weights_sha256: shape.weights_sha256.clone(),
        runtime_device: shape.runtime_device,
        input_sha256: digest('1'),
        path: ShapeProfileExecutionPath::DynamicFallback {
            reason: ShapeProfileFallbackReason::ShapeClassUnavailable,
            implementation_sha256: digest('2'),
        },
    };
    let output = ExecutionDigest::f32_tensor(&[1, 2], &[3.0, 5.0]);
    let before_pool = pool(shape.runtime_device, 2, 0, 0, 0);
    let retired_pool = pool(shape.runtime_device, 1, 1, 1, 0);
    let recovered_pool = pool(shape.runtime_device, 2, 0, 1, 1);
    let contracts = ReleaseContractEvidence {
        peak_memory: PeakMemoryEvidence {
            host: BoundedMemoryEvidence::host_allocator(1_000, 1_100, 1_032).unwrap(),
            device: None,
        },
        cancellation: CancellationContractEvidence {
            lifecycle: ExecutionBatchLifecycleEvidence {
                schema: ExecutionBatchLifecycleEvidence::SCHEMA.to_string(),
                declaration_sha256: digest('3'),
                transcript_sha256: digest('4'),
                admitted_members: 2,
                completed_members: 1,
                cancelled_members: 1,
                committed_steps: 1,
                processed_rows: 2,
                max_active_members: 2,
                peak_state_bytes: 64,
            },
            admission_after: admission(1, 0, 2, 0),
            resident_after: ResidentTensorSnapshot {
                maximum_bytes: 256,
                active_handles: 0,
                resident_bytes: 0,
                peak_resident_bytes: 64,
                rejected_reservations: 0,
            },
        },
        queue_expiry: QueueExpiryEvidence {
            before: admission(0, 0, 0, 0),
            after: admission(1, 1, 1, 1),
        },
        replica_recovery: ReplicaRecoveryEvidence {
            before: before_pool,
            retired: retired_pool,
            recovered: recovered_pool,
        },
        exact_fallback: ExactFallbackEvidence {
            selection,
            reference_output: output.clone(),
            fallback_output: output,
        },
    };
    Fixture {
        report,
        shape,
        contracts,
    }
}

fn policy(fixture: &Fixture, platforms: Vec<ReleasePlatform>) -> ReleaseEvidencePolicy {
    let required_platforms = platforms
        .into_iter()
        .map(|platform| {
            ReleasePlatformBinding::new(
                platform,
                &fixture
                    .contracts
                    .exact_fallback
                    .selection
                    .declaration_sha256,
                &fixture.shape.tee_policy_sha256,
            )
            .unwrap()
        })
        .collect();
    ReleaseEvidencePolicy::new(
        ReleaseRevisionBinding::new(
            fixture.report.binding.power_version.clone(),
            fixture.report.binding.power_commit.clone(),
            fixture.report.binding.weights_sha256.clone(),
            fixture.report.binding.graph_source_sha256.clone(),
            fixture.shape.graph_sha256.clone(),
        )
        .unwrap(),
        required_platforms,
    )
    .unwrap()
}

fn capture(fixture: &Fixture) -> ReleaseCapture {
    ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        fixture.shape.clone(),
        fixture.report.clone(),
        fixture.contracts.clone(),
    )
    .unwrap()
}

#[test]
fn model_neutral_release_bundle_round_trips_and_verifies_a_pin() {
    let fixture = fixture();
    let capture = capture(&fixture);
    assert_eq!(
        ReleaseRevisionBinding::from_capture(&capture).unwrap(),
        policy(&fixture, vec![ReleasePlatform::Cpu]).revision
    );
    assert_eq!(
        capture.platform_binding().unwrap().platform,
        ReleasePlatform::Cpu
    );
    let bundle =
        ReleaseEvidenceBundle::build(policy(&fixture, vec![ReleasePlatform::Cpu]), vec![capture])
            .unwrap();
    bundle.verify().unwrap();
    bundle.verify_pinned(&bundle.sha256).unwrap();

    let json = serde_json::to_string(&bundle).unwrap();
    for architecture_term in ["qwen", "gguf", "tokenizer", "mtp"] {
        assert!(!json.to_ascii_lowercase().contains(architecture_term));
    }
    let restored: ReleaseEvidenceBundle = serde_json::from_str(&json).unwrap();
    restored.verify().unwrap();
    assert_eq!(restored, bundle);
}

#[test]
fn strict_v1_policy_requires_and_binds_all_platform_profiles() {
    let fixture = fixture();
    let revision = policy(&fixture, vec![ReleasePlatform::Cpu]).revision;
    let bindings = [
        ReleasePlatform::Cpu,
        ReleasePlatform::Cuda,
        ReleasePlatform::Metal,
        ReleasePlatform::ConfidentialGpu,
    ]
    .into_iter()
    .enumerate()
    .map(|(index, platform)| {
        ReleasePlatformBinding::new(
            platform,
            format!("{:064x}", index + 1),
            format!("{:064x}", index + 5),
        )
        .unwrap()
    })
    .collect();
    let strict = ReleaseEvidencePolicy::strict_v1(revision.clone(), bindings).unwrap();
    assert_eq!(strict.required_platforms.len(), 4);
    strict.verify_strict_v1().unwrap();

    let partial =
        vec![ReleasePlatformBinding::new(ReleasePlatform::Cpu, digest('e'), digest('d')).unwrap()];
    let error = ReleaseEvidencePolicy::strict_v1(revision, partial).unwrap_err();

    assert!(error.to_string().contains("requires CPU"));

    let development = policy(&fixture, vec![ReleasePlatform::Cpu]);
    let error = development.verify_strict_v1().unwrap_err();
    assert!(error.to_string().contains("requires CPU"));
}

#[test]
fn contract_failures_are_rejected_before_a_capture_is_signed() {
    let mut missing_cancellation = fixture();
    missing_cancellation
        .contracts
        .cancellation
        .lifecycle
        .cancelled_members = 0;
    assert!(ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        missing_cancellation.shape,
        missing_cancellation.report,
        missing_cancellation.contracts,
    )
    .is_err());

    let mut missing_expiry = fixture();
    missing_expiry
        .contracts
        .queue_expiry
        .after
        .deadline_expirations = 0;
    assert!(ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        missing_expiry.shape,
        missing_expiry.report,
        missing_expiry.contracts,
    )
    .is_err());

    let mut missing_recovery = fixture();
    missing_recovery
        .contracts
        .replica_recovery
        .recovered
        .replica_reconstructions = 0;
    assert!(ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        missing_recovery.shape,
        missing_recovery.report,
        missing_recovery.contracts,
    )
    .is_err());

    let mut wrong_fallback = fixture();
    wrong_fallback.contracts.exact_fallback.selection.path = ShapeProfileExecutionPath::Profile {
        profile_sha256: digest('5'),
        implementation_sha256: digest('6'),
    };
    assert!(ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        wrong_fallback.shape,
        wrong_fallback.report,
        wrong_fallback.contracts,
    )
    .is_err());

    let mut unbounded_peak = fixture();
    unbounded_peak.contracts.peak_memory.host.peak_used_bytes = 2_000;
    assert!(ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        unbounded_peak.shape,
        unbounded_peak.report,
        unbounded_peak.contracts,
    )
    .is_err());
}

#[test]
fn policy_and_security_mismatches_fail_closed() {
    let fixture = fixture();
    assert!(ReleaseEvidencePolicy::new(
        policy(&fixture, vec![ReleasePlatform::Cpu]).revision,
        vec![
            ReleasePlatformBinding::new(ReleasePlatform::Cpu, digest('e'), digest('d')).unwrap(),
            ReleasePlatformBinding::new(ReleasePlatform::Cpu, digest('f'), digest('d')).unwrap(),
        ],
    )
    .is_err());

    let confidential = serde_json::from_value::<ReleaseCaptureSecurity>(serde_json::json!({
        "kind": "confidential-gpu",
        "binding": {
            "teeType": "sev-snp",
            "launchMeasurement": "5".repeat(96),
            "attestationReportSha256": digest('6'),
            "verifiedClaimsSha256": digest('7'),
            "acceleratorDeclarationSha256": digest('8'),
            "weightsSha256": fixture.shape.weights_sha256.clone(),
            "executionPolicySha256": digest('9'),
            "inferenceExecutionPolicySha256": digest('a'),
            "runtimeDevice": { "kind": "cpu", "ordinal": null },
            "deviceMeshSha256": null
        }
    }))
    .unwrap();
    assert!(ReleaseCapture::build(
        confidential,
        fixture.shape.clone(),
        fixture.report.clone(),
        fixture.contracts.clone(),
    )
    .is_err());

    let simulated = serde_json::from_value::<ReleaseCaptureSecurity>(serde_json::json!({
        "kind": "confidential-gpu",
        "binding": {
            "teeType": "simulated",
            "launchMeasurement": "5".repeat(96),
            "attestationReportSha256": digest('6'),
            "verifiedClaimsSha256": digest('7'),
            "acceleratorDeclarationSha256": digest('8'),
            "weightsSha256": fixture.shape.weights_sha256.clone(),
            "executionPolicySha256": fixture.shape.tee_policy_sha256.clone(),
            "inferenceExecutionPolicySha256": digest('a'),
            "runtimeDevice": { "kind": "cuda", "ordinal": 0 },
            "deviceMeshSha256": null
        }
    }))
    .unwrap();
    assert!(
        ReleaseCapture::build(simulated, fixture.shape, fixture.report, fixture.contracts).is_err()
    );
}

#[test]
fn bundle_requires_the_capture_specific_profile_and_tee_policy() {
    let fixture = fixture();
    let mut wrong_profile = policy(&fixture, vec![ReleasePlatform::Cpu]);
    wrong_profile.required_platforms[0].shape_profile_declaration_sha256 = digest('9');
    assert!(ReleaseEvidenceBundle::build(wrong_profile, vec![capture(&fixture)]).is_err());

    let mut wrong_tee = policy(&fixture, vec![ReleasePlatform::Cpu]);
    wrong_tee.required_platforms[0].tee_policy_sha256 = digest('8');
    assert!(ReleaseEvidenceBundle::build(wrong_tee, vec![capture(&fixture)]).is_err());
}

#[test]
fn published_clean_revision_reports_replay_successfully() {
    for source in [
        include_str!("../../docs/benchmarks/release-gate-windows-20260821/cpu.json"),
        include_str!("../../docs/benchmarks/release-gate-windows-20260821/cuda.json"),
    ] {
        let report = serde_json::from_str::<TensorBatchBenchmarkReport>(source).unwrap();
        report.verify().unwrap();
        assert_eq!(report.binding.power_commit, POWER_COMMIT);
        assert!(report.exact_output_parity);
    }
}

#[test]
fn published_complete_cpu_cuda_contracts_replay_as_one_partial_policy() {
    let captures = [
        include_str!("../../docs/benchmarks/release-contract-windows-20260821/cpu.json"),
        include_str!("../../docs/benchmarks/release-contract-windows-20260821/cuda.json"),
    ]
    .map(|source| serde_json::from_str::<ReleaseCapture>(source).unwrap());
    for capture in &captures {
        capture.verify().unwrap();
        assert_eq!(
            capture.tensor_batch.binding.power_commit,
            COMPLETE_CAPTURE_COMMIT
        );
    }
    assert_eq!(captures[0].platform().unwrap(), ReleasePlatform::Cpu);
    assert_eq!(captures[1].platform().unwrap(), ReleasePlatform::Cuda);

    let revision = ReleaseRevisionBinding::from_capture(&captures[0]).unwrap();
    assert_eq!(
        ReleaseRevisionBinding::from_capture(&captures[1]).unwrap(),
        revision
    );
    let platform_bindings = captures
        .iter()
        .map(|capture| capture.platform_binding().unwrap())
        .collect();
    let policy = ReleaseEvidencePolicy::new(revision, platform_bindings).unwrap();
    let bundle = ReleaseEvidenceBundle::build(policy, captures.into_iter().collect()).unwrap();
    bundle.verify().unwrap();
}

#[test]
fn release_evidence_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<ReleaseRevisionBinding>();
    assert_send_sync::<ReleasePlatformBinding>();
    assert_send_sync::<ReleaseEvidencePolicy>();
    assert_send_sync::<ReleaseContractEvidence>();
    assert_send_sync::<ReleaseCapture>();
    assert_send_sync::<ReleaseEvidenceBundle>();
    assert_send_sync::<AdmissionSnapshot>();
    assert_send_sync::<ModelSessionPoolSnapshot>();
}
