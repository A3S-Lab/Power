use std::sync::Arc;

use candle_core::{Device, Tensor};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::error::PowerError;
use crate::tee::attestation::{
    build_claims_report_data, AttestationClaimsV2, AttestationReport, ExecutionPolicyClaim,
    GpuDeviceClaim, GpuDeviceValidationClaim, GpuEvidenceClaim, ModelDigestClaim, ModelDigestKind,
    RuntimePolicyClaim, TeeType,
};

fn digest(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn store() -> (tempfile::TempDir, Arc<WeightStore>) {
    let directory = tempfile::tempdir().unwrap();
    let gate = [1_f32, 2_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let up = [3_f32, 4_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let cold = [7_f32, 8_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let tensors = vec![
        (
            "layer.0.expert.0.gate",
            TensorView::new(Dtype::F32, vec![2], &gate).unwrap(),
        ),
        (
            "layer.0.expert.0.up",
            TensorView::new(Dtype::F32, vec![2], &up).unwrap(),
        ),
        (
            "layer.0.expert.1.gate",
            TensorView::new(Dtype::F32, vec![2], &cold).unwrap(),
        ),
    ];
    serialize_to_file(tensors, None, &directory.path().join("model.safetensors")).unwrap();
    let store = WeightStore::open(directory.path(), &InferenceLimits::default()).unwrap();
    (directory, Arc::new(store))
}

fn runtime(limits: InferenceLimits) -> EmbeddedRuntime {
    EmbeddedRuntime::new_test_accelerator(RuntimeDeviceKind::Cuda, 0, limits).unwrap()
}

fn hierarchy_with_limits(
    limits: InferenceLimits,
) -> (tempfile::TempDir, EmbeddedRuntime, WeightHierarchy) {
    let (directory, store) = store();
    let runtime = runtime(limits);
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            device_cache_bytes: 16,
            host_cache_bytes: 16,
            max_entries_per_layer: 8,
            telemetry: TelemetryMode::Disabled,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    (directory, runtime, hierarchy)
}

fn applied_hierarchy() -> (
    tempfile::TempDir,
    EmbeddedRuntime,
    WeightHierarchy,
    ExecutionPermit,
    CancellationToken,
) {
    let (directory, runtime, hierarchy) = hierarchy_with_limits(InferenceLimits::default());
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let plan = hierarchy
        .plan_residency(&[
            ResidencyCandidate::new(
                "expert-0",
                20,
                vec![
                    WeightKey::new(0, "layer.0.expert.0.gate"),
                    WeightKey::new(0, "layer.0.expert.0.up"),
                ],
            ),
            ResidencyCandidate::new(
                "expert-1",
                1,
                vec![WeightKey::new(0, "layer.0.expert.1.gate")],
            ),
        ])
        .unwrap();
    assert_eq!(plan.groups[0].tier, WeightTier::Device);
    hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .unwrap();
    (directory, runtime, hierarchy, permit, cancellation)
}

fn local_spec() -> AcceleratorFusedBatchSpec {
    AcceleratorFusedBatchSpec::new(digest('1'), digest('2'), vec!["expert-0".to_string()])
        .with_fallback_mode(AcceleratorFallbackMode::AllowExact)
}

fn confidential_report(
    _declaration: &AcceleratorResidencyDeclaration,
    execution_digest: Vec<u8>,
    model_digest: Vec<u8>,
) -> AttestationReport {
    let claims = AttestationClaimsV2::new(TeeType::SevSnp)
        .with_model(ModelDigestClaim {
            name: "embedded-model".to_string(),
            kind: ModelDigestKind::PlaintextWeightsSha256,
            digest: model_digest,
            plaintext_digest: None,
            ciphertext_digest: None,
        })
        .with_gpu(
            GpuEvidenceClaim::new("nvidia-nras", vec![0x33; 32])
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(vec![0x44; 32])
                .with_devices(vec![GpuDeviceClaim {
                    index: 0,
                    device_type: "gpu".to_string(),
                    attestation_nonce: None,
                    hwmodel: Some("GH100".to_string()),
                    ueid: Some("gpu-0".to_string()),
                    oemid: Some("nvidia".to_string()),
                    claims_version: Some("3.0".to_string()),
                    driver_version: Some("test".to_string()),
                    firmware_version: Some("test".to_string()),
                    measurements_result: Some("success".to_string()),
                    secure_boot: Some(true),
                    debug_status: Some("disabled".to_string()),
                    validation: GpuDeviceValidationClaim::default(),
                }]),
        )
        .with_runtime(
            RuntimePolicyClaim::new().with_execution(ExecutionPolicyClaim {
                gpu_sha256: execution_digest,
            }),
        );
    AttestationReport {
        version: "1.0".to_string(),
        tee_type: TeeType::SevSnp,
        report_data: build_claims_report_data(&claims).unwrap(),
        measurement: vec![0x55; 48],
        raw_report: Some(vec![0x66; 64]),
        timestamp: chrono::Utc::now(),
        nonce: None,
        claims: Some(claims),
    }
}

#[test]
fn declaration_is_deterministic_and_bound_to_the_active_device_plan() {
    let (_directory, _runtime, hierarchy, _permit, _cancellation) = applied_hierarchy();

    let first = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    let second = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();

    assert_eq!(first, second);
    assert_eq!(first.schema, AcceleratorResidencyDeclaration::SCHEMA);
    assert_eq!(first.runtime_device.kind, RuntimeDeviceKind::Cuda);
    assert_eq!(first.runtime_device.ordinal, Some(0));
    assert_eq!(first.groups.len(), 1);
    assert_eq!(first.groups[0].canonical_index, 0);
    assert_eq!(first.groups[0].bytes, 16);
    assert_eq!(first.total_bytes, 16);
    assert_eq!(first.total_weights, 2);
    assert_eq!(first.declaration_sha256.len(), 64);
    assert_eq!(first.execution_policy_sha256, first.declaration_sha256);

    let mut changed = local_spec();
    changed.fused_kernel_sha256 = digest('3');
    assert_ne!(
        hierarchy
            .declare_accelerator_residency(&changed)
            .unwrap()
            .declaration_sha256,
        first.declaration_sha256
    );

    let target_changed =
        local_spec().with_fallback_target(AcceleratorFallbackTarget::RuntimeDevice);
    assert_ne!(
        hierarchy
            .declare_accelerator_residency(&target_changed)
            .unwrap()
            .declaration_sha256,
        first.declaration_sha256
    );
}

#[test]
fn declaration_rejects_invalid_or_non_device_groups() {
    let (_directory, _runtime, hierarchy, _permit, _cancellation) = applied_hierarchy();

    let mut invalid_digest = local_spec();
    invalid_digest.fused_kernel_sha256 = "not-a-digest".to_string();
    assert!(hierarchy
        .declare_accelerator_residency(&invalid_digest)
        .is_err());

    let duplicate = AcceleratorFusedBatchSpec::new(
        digest('1'),
        digest('2'),
        vec!["expert-0".to_string(), "expert-0".to_string()],
    );
    assert!(hierarchy.declare_accelerator_residency(&duplicate).is_err());

    let missing =
        AcceleratorFusedBatchSpec::new(digest('1'), digest('2'), vec!["missing".to_string()]);
    assert!(hierarchy.declare_accelerator_residency(&missing).is_err());

    let host_group =
        AcceleratorFusedBatchSpec::new(digest('1'), digest('2'), vec!["expert-1".to_string()]);
    assert!(hierarchy
        .declare_accelerator_residency(&host_group)
        .is_err());
}

#[test]
fn cpu_runtime_cannot_declare_accelerator_residency() {
    let (_directory, store) = store();
    let runtime = EmbeddedRuntime::new(DevicePreference::Cpu, InferenceLimits::default()).unwrap();
    let hierarchy = WeightHierarchy::new(
        store,
        runtime.clone(),
        ResidencyPolicy {
            host_cache_bytes: 16,
            ..ResidencyPolicy::default()
        },
    )
    .unwrap();
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let plan = hierarchy
        .plan_residency(&[ResidencyCandidate::new(
            "expert-0",
            1,
            vec![WeightKey::new(0, "layer.0.expert.0.gate")],
        )])
        .unwrap();
    hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .unwrap();

    assert!(matches!(
        hierarchy.declare_accelerator_residency(&local_spec()),
        Err(PowerError::BackendNotAvailable(_))
    ));
}

#[test]
fn fused_batch_matches_the_unfused_candle_reference() {
    let (_directory, runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    let resolution = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = resolution else {
        panic!("device-resident plan should resolve to a fused batch");
    };
    let input = Tensor::new(&[5_f32, 6_f32], runtime.device().tensor_device()).unwrap();
    let output = batch
        .execute(&input, &cancellation, |input, groups, _| {
            let weights = groups[0].weights();
            input
                .broadcast_add(weights[0].tensor())
                .and_then(|value| value.broadcast_mul(weights[1].tensor()))
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();
    let (tensor, completion) = output.into_parts();
    let values = tensor.to_vec1::<f32>().unwrap();
    assert_eq!(values, vec![18.0, 32.0]);

    let input_digest = ExecutionDigest::f32_tensor(&[2], &[5.0, 6.0]);
    let output_digest = ExecutionDigest::f32_tensor(&[2], &values);
    let evidence = completion.complete(&input_digest, &output_digest).unwrap();
    assert_eq!(evidence.path, AcceleratorExecutionPath::Accelerator);
    assert_eq!(evidence.execution_device, evidence.runtime_device);
    assert_eq!(evidence.implementation_sha256, digest('1'));
    assert!(evidence.confidential_claims_sha256.is_none());
}

#[test]
fn residency_pressure_returns_an_explicit_exact_fallback_identity() {
    let (_directory, runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    hierarchy.clear_residency_plan();
    hierarchy.clear_unpinned();

    let resolution = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap();
    let AcceleratorBatchResolution::Fallback(fallback) = resolution else {
        panic!("lost device residency must not be reported as accelerated");
    };
    assert_eq!(fallback.reason(), AcceleratorFallbackReason::PlanChanged);
    assert_eq!(fallback.implementation_sha256(), digest('2'));

    let input = ExecutionDigest::f32_tensor(&[1], &[1.0]);
    let output = ExecutionDigest::f32_tensor(&[1], &[2.0]);
    let evidence = fallback.complete(&input, &output).unwrap();
    assert_eq!(
        evidence.path,
        AcceleratorExecutionPath::Fallback {
            reason: AcceleratorFallbackReason::PlanChanged
        }
    );
    assert_eq!(evidence.implementation_sha256, digest('2'));
    assert_eq!(evidence.execution_device.kind, RuntimeDeviceKind::Cpu);
    assert_eq!(evidence.runtime_device.kind, RuntimeDeviceKind::Cuda);

    let receipt = runtime
        .receipt_with_accelerator(
            ModelIdentity::new("test", "v1", hierarchy.store().sha256()),
            input,
            output,
            evidence,
        )
        .unwrap();
    assert_eq!(receipt.schema, ExecutionReceipt::ACCELERATOR_SCHEMA);
    assert!(receipt.accelerator.is_some());
}

#[test]
fn typed_kernel_unavailability_runs_the_declared_bounded_fallback() {
    let (_directory, runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("batch should be ready");
    };
    let input = Tensor::new(&[2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    let AcceleratorFusedExecution::Fallback(fallback) = batch
        .execute_or_fallback(&input, &cancellation, |_input, _groups, _| {
            Ok(AcceleratorKernelOutcome::Unavailable)
        })
        .unwrap()
    else {
        panic!("typed kernel unavailability must select the exact fallback");
    };
    assert_eq!(
        fallback.reason(),
        AcceleratorFallbackReason::KernelUnavailable
    );
    assert_eq!(fallback.target(), AcceleratorFallbackTarget::Cpu);
    let fallback_input = input.to_device(fallback.tensor_device()).unwrap();
    let output = fallback
        .execute(&fallback_input, &cancellation, |input, _| {
            input
                .affine(3.0, 1.0)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();
    assert_eq!(output.tensor().to_vec1::<f32>().unwrap(), vec![7.0, 10.0]);
    let (_, completion) = output.into_parts();
    let input_digest = ExecutionDigest::f32_tensor(&[2], &[2.0, 3.0]);
    let output_digest = ExecutionDigest::f32_tensor(&[2], &[7.0, 10.0]);
    let evidence = completion.complete(&input_digest, &output_digest).unwrap();
    assert_eq!(
        evidence.path,
        AcceleratorExecutionPath::Fallback {
            reason: AcceleratorFallbackReason::KernelUnavailable
        }
    );
    assert_eq!(
        evidence.fallback_target,
        Some(AcceleratorFallbackTarget::Cpu)
    );
    assert_eq!(evidence.execution_device.kind, RuntimeDeviceKind::Cpu);
    assert_eq!(evidence.implementation_sha256, digest('2'));
}

#[test]
fn denied_fallback_fails_closed_under_residency_pressure() {
    let (_directory, _runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let spec = local_spec().with_fallback_mode(AcceleratorFallbackMode::Deny);
    let declaration = hierarchy.declare_accelerator_residency(&spec).unwrap();
    hierarchy.clear_residency_plan();
    hierarchy.clear_unpinned();

    assert!(matches!(
        hierarchy.resolve_accelerator_batch(&declaration, None, &permit, &cancellation),
        Err(PowerError::InferenceFailed(_))
    ));
}

#[test]
fn cancellation_and_foreign_permits_are_rejected() {
    let (_directory, _runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    cancellation.cancel();
    assert!(hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .is_err());

    let foreign_runtime = runtime(InferenceLimits::default());
    let foreign_cancellation = CancellationToken::new();
    let foreign_permit = foreign_runtime.begin(&foreign_cancellation).unwrap();
    assert!(hierarchy
        .resolve_accelerator_batch(&declaration, None, &foreign_permit, &foreign_cancellation,)
        .is_err());
}

#[test]
fn fused_output_is_checked_against_runtime_bounds() {
    let limits = InferenceLimits {
        max_tensor_elements: 2,
        ..InferenceLimits::default()
    };
    let (directory, runtime, hierarchy) = hierarchy_with_limits(limits);
    let _keep = directory;
    let cancellation = CancellationToken::new();
    let permit = runtime.begin(&cancellation).unwrap();
    let plan = hierarchy
        .plan_residency(&[ResidencyCandidate::new(
            "expert-0",
            1,
            vec![
                WeightKey::new(0, "layer.0.expert.0.gate"),
                WeightKey::new(0, "layer.0.expert.0.up"),
            ],
        )])
        .unwrap();
    hierarchy
        .apply_residency_plan(&plan, &permit, &cancellation)
        .unwrap();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("batch should be ready");
    };
    let input = Tensor::new(&[1_f32, 2_f32], runtime.device().tensor_device()).unwrap();

    assert!(batch
        .execute(&input, &cancellation, |_input, _groups, _| {
            Tensor::new(&[1_f32, 2_f32, 3_f32], &Device::Cpu)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))
        })
        .is_err());
}

#[test]
fn confidential_gpu_execution_requires_matching_existing_attestation_claims() {
    let (_directory, _runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let spec = local_spec().with_security(AcceleratorSecurityRequirement::ConfidentialGpu);
    let declaration = hierarchy.declare_accelerator_residency(&spec).unwrap();

    assert!(matches!(
        hierarchy.resolve_accelerator_batch(&declaration, None, &permit, &cancellation,),
        Err(PowerError::PolicyViolation(_))
    ));

    let report = confidential_report(
        &declaration,
        hex::decode(&declaration.execution_policy_sha256).unwrap(),
        hex::decode(hierarchy.store().sha256()).unwrap(),
    );
    let binding =
        ConfidentialGpuBinding::from_attestation_report_for_test(&report, &declaration).unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_batch(&declaration, Some(&binding), &permit, &cancellation)
        .unwrap()
    else {
        panic!("matching attestation binding should admit the batch");
    };
    let input = Tensor::new(
        &[1_f32, 2_f32],
        hierarchy.runtime().device().tensor_device(),
    )
    .unwrap();
    let output = batch
        .execute(&input, &cancellation, |input, _groups, _| {
            input
                .affine(2.0, 0.0)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();
    let (_, completion) = output.into_parts();
    let input_digest = ExecutionDigest::f32_tensor(&[2], &[1.0, 2.0]);
    let output_digest = ExecutionDigest::f32_tensor(&[2], &[2.0, 4.0]);
    let evidence = completion.complete(&input_digest, &output_digest).unwrap();
    assert_eq!(
        evidence.confidential_claims_sha256,
        Some(binding.claims_sha256().to_string())
    );
}

#[test]
fn confidential_binding_rejects_wrong_policy_model_and_simulation() {
    let (_directory, _runtime, hierarchy, _permit, _cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(
            &local_spec().with_security(AcceleratorSecurityRequirement::ConfidentialGpu),
        )
        .unwrap();
    let weights = hex::decode(hierarchy.store().sha256()).unwrap();

    let wrong_policy = confidential_report(&declaration, vec![0x99; 32], weights.clone());
    assert!(
        ConfidentialGpuBinding::from_attestation_report_for_test(&wrong_policy, &declaration)
            .is_err()
    );

    let wrong_model = confidential_report(
        &declaration,
        hex::decode(&declaration.execution_policy_sha256).unwrap(),
        vec![0x88; 32],
    );
    assert!(
        ConfidentialGpuBinding::from_attestation_report_for_test(&wrong_model, &declaration)
            .is_err()
    );

    let mut missing_device = confidential_report(
        &declaration,
        hex::decode(&declaration.execution_policy_sha256).unwrap(),
        hex::decode(hierarchy.store().sha256()).unwrap(),
    );
    missing_device
        .claims
        .as_mut()
        .unwrap()
        .gpu
        .as_mut()
        .unwrap()
        .devices
        .clear();
    missing_device.report_data =
        build_claims_report_data(missing_device.claims.as_ref().unwrap()).unwrap();
    assert!(ConfidentialGpuBinding::from_attestation_report_for_test(
        &missing_device,
        &declaration
    )
    .is_err());

    let mut simulated = confidential_report(
        &declaration,
        hex::decode(&declaration.execution_policy_sha256).unwrap(),
        weights,
    );
    simulated.tee_type = TeeType::Simulated;
    simulated.claims.as_mut().unwrap().tee_type = TeeType::Simulated;
    simulated.report_data = build_claims_report_data(simulated.claims.as_ref().unwrap()).unwrap();
    assert!(
        ConfidentialGpuBinding::from_attestation_report_for_test(&simulated, &declaration).is_err()
    );
}

#[test]
fn malformed_declaration_and_receipt_replay_are_rejected() {
    let (_directory, runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let mut declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    declaration.total_bytes += 1;
    assert!(hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .is_err());

    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("batch should be ready");
    };
    let input_tensor = Tensor::new(&[1_f32], runtime.device().tensor_device()).unwrap();
    let output = batch
        .execute(&input_tensor, &cancellation, |input, _groups, _| {
            Ok(input.clone())
        })
        .unwrap();
    let (_, completion) = output.into_parts();
    let input = ExecutionDigest::f32_tensor(&[1], &[1.0]);
    let output = ExecutionDigest::f32_tensor(&[1], &[1.0]);
    let evidence = completion.complete(&input, &output).unwrap();
    let different_output = ExecutionDigest::f32_tensor(&[1], &[2.0]);
    assert!(runtime
        .receipt_with_accelerator(
            ModelIdentity::new("test", "v1", hierarchy.store().sha256()),
            input,
            different_output,
            evidence,
        )
        .is_err());
}

#[test]
fn execution_evidence_is_digest_only_and_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AcceleratorResidencyDeclaration>();
    assert_send_sync::<AcceleratorFusedBatch>();
    assert_send_sync::<AcceleratorExecutionEvidence>();
    assert_send_sync::<ConfidentialGpuBinding>();

    let (_directory, _runtime, hierarchy, permit, cancellation) = applied_hierarchy();
    let declaration = hierarchy
        .declare_accelerator_residency(&local_spec())
        .unwrap();
    hierarchy.clear_residency_plan();
    hierarchy.clear_unpinned();
    let AcceleratorBatchResolution::Fallback(fallback) = hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("fallback expected");
    };
    let input = ExecutionDigest::f32_tensor(&[1], &[1.0]);
    let output = ExecutionDigest::f32_tensor(&[1], &[2.0]);
    let json = serde_json::to_string(&fallback.complete(&input, &output).unwrap()).unwrap();
    assert!(!json.contains("layer.0"));
    assert!(!json.contains("expert-0"));
    assert!(!json.contains("1.0"));
    assert!(!json.contains("2.0"));
}
