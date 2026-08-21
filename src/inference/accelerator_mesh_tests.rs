use std::sync::Arc;

use candle_core::Tensor;
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

fn hierarchy() -> (
    tempfile::TempDir,
    EmbeddedRuntime,
    WeightHierarchy,
    ExecutionPermit,
    CancellationToken,
) {
    hierarchy_with_limits(InferenceLimits::default())
}

pub(super) fn hierarchy_with_limits(
    limits: InferenceLimits,
) -> (
    tempfile::TempDir,
    EmbeddedRuntime,
    WeightHierarchy,
    ExecutionPermit,
    CancellationToken,
) {
    let directory = tempfile::tempdir().unwrap();
    let gate = [1_f32, 2_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let up = [3_f32, 4_f32]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        vec![
            (
                "layer.0.expert.0.gate",
                TensorView::new(Dtype::F32, vec![2], &gate).unwrap(),
            ),
            (
                "layer.0.expert.0.up",
                TensorView::new(Dtype::F32, vec![2], &up).unwrap(),
            ),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let store = WeightStore::open(directory.path(), &limits).unwrap();
    let runtime =
        EmbeddedRuntime::new_test_accelerator(RuntimeDeviceKind::Cuda, 0, limits).unwrap();
    let hierarchy = WeightHierarchy::new(
        Arc::new(store),
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
    (directory, runtime, hierarchy, permit, cancellation)
}

pub(super) fn spec() -> AcceleratorFusedBatchSpec {
    AcceleratorFusedBatchSpec::new(
        digest('1'),
        digest('2'),
        digest('3'),
        vec!["expert-0".to_string()],
    )
    .with_fallback_mode(AcceleratorFallbackMode::AllowExact)
}

pub(super) fn local_mesh(runtime: &EmbeddedRuntime) -> AcceleratorDeviceMesh {
    AcceleratorDeviceMesh::new(
        "home",
        vec![
            AcceleratorMeshDevice::new(
                "peer",
                RuntimeDevice::test_accelerator(RuntimeDeviceKind::Cuda, 1).unwrap(),
            ),
            AcceleratorMeshDevice::new("home", runtime.device().clone()),
        ],
        vec![
            AcceleratorPeerTransferSpec::new("peer", "home", 8, 1),
            AcceleratorPeerTransferSpec::new("home", "peer", 8, 1),
        ],
        16,
    )
    .unwrap()
}

fn device_claim(index: u32, device_type: &str) -> GpuDeviceClaim {
    GpuDeviceClaim {
        index,
        device_type: device_type.to_string(),
        attestation_nonce: None,
        hwmodel: Some(if device_type == "gpu" {
            "GH100".to_string()
        } else {
            "NVSwitch B01".to_string()
        }),
        ueid: Some(format!("{device_type}-{index}")),
        oemid: Some("nvidia".to_string()),
        claims_version: Some("3.0".to_string()),
        driver_version: Some("test".to_string()),
        firmware_version: Some("test".to_string()),
        measurements_result: Some("success".to_string()),
        secure_boot: Some(true),
        debug_status: Some("disabled".to_string()),
        validation: GpuDeviceValidationClaim::default(),
    }
}

fn confidential_report(
    hierarchy: &WeightHierarchy,
    declaration: &AcceleratorResidencyDeclaration,
    devices: Vec<GpuDeviceClaim>,
) -> AttestationReport {
    let claims = AttestationClaimsV2::new(TeeType::SevSnp)
        .with_model(ModelDigestClaim {
            name: "embedded-model".to_string(),
            kind: ModelDigestKind::PlaintextWeightsSha256,
            digest: hex::decode(hierarchy.store().sha256()).unwrap(),
            plaintext_digest: None,
            ciphertext_digest: None,
        })
        .with_gpu(
            GpuEvidenceClaim::new("nvidia-nras", vec![0x33; 32])
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(vec![0x44; 32])
                .with_devices(devices),
        )
        .with_runtime(
            RuntimePolicyClaim::new().with_execution(ExecutionPolicyClaim {
                gpu_sha256: hex::decode(&declaration.execution_policy_sha256).unwrap(),
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
fn mesh_declaration_is_canonical_and_single_device_execution_keeps_exact_parity() {
    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let first_mesh = AcceleratorDeviceMesh::new(
        "home",
        vec![AcceleratorMeshDevice::new("home", runtime.device().clone())],
        Vec::new(),
        0,
    )
    .unwrap();
    let second_mesh = AcceleratorDeviceMesh::new(
        "home",
        vec![AcceleratorMeshDevice::new("home", runtime.device().clone())],
        Vec::new(),
        0,
    )
    .unwrap();
    let first = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &first_mesh)
        .unwrap();
    let second = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &second_mesh)
        .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.schema, AcceleratorResidencyDeclaration::MESH_SCHEMA);
    assert_eq!(
        first.device_mesh.as_ref().unwrap().schema,
        AcceleratorDeviceMeshDeclaration::SCHEMA
    );

    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&first, &first_mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("single-device mesh should resolve");
    };
    let input = Tensor::new(&[5_f32, 6_f32], runtime.device().tensor_device()).unwrap();
    let result = batch
        .execute_mesh(&input, &cancellation, |input, groups, mesh, _| {
            assert!(mesh.tensor_device("home")?.same_device(input.device()));
            input
                .broadcast_add(groups[0].weights()[0].tensor())
                .and_then(|value| value.broadcast_mul(groups[0].weights()[1].tensor()))
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))
        })
        .unwrap();
    let (output, completion) = result.into_parts();
    let values = output.to_vec1::<f32>().unwrap();
    assert_eq!(values, vec![18.0, 32.0]);
    let evidence = completion
        .complete(
            &ExecutionDigest::f32_tensor(&[2], &[5.0, 6.0]),
            &ExecutionDigest::f32_tensor(&[2], &values),
        )
        .unwrap();
    assert_eq!(evidence.schema, AcceleratorExecutionEvidence::MESH_SCHEMA);
    assert_eq!(
        evidence.execution_devices,
        vec![runtime.device().identity()]
    );
    assert_eq!(
        evidence.device_mesh_sha256,
        Some(first.device_mesh.unwrap().mesh_sha256)
    );
    assert!(evidence.peer_transfers_sha256.is_some());
}

#[test]
fn peer_transfers_are_real_bounded_and_reflected_as_actual_devices() {
    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let mesh = local_mesh(&runtime);
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    let input = Tensor::new(&[2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    let result = batch
        .execute_mesh_or_fallback(&input, &cancellation, |input, _groups, mesh, _| {
            let AcceleratorPeerTransferOutcome::Transferred(peer) =
                mesh.transfer("home", "peer", input)?
            else {
                return Ok(AcceleratorKernelOutcome::Unavailable);
            };
            let peer = peer
                .affine(2.0, 0.0)
                .map_err(|error| PowerError::InferenceFailed(error.to_string()))?;
            let AcceleratorPeerTransferOutcome::Transferred(home) =
                mesh.transfer("peer", "home", &peer)?
            else {
                return Ok(AcceleratorKernelOutcome::Unavailable);
            };
            assert_eq!(mesh.transfer_count(), 2);
            assert_eq!(mesh.transferred_bytes(), 16);
            Ok(AcceleratorKernelOutcome::Output(home))
        })
        .unwrap();
    let AcceleratorFusedExecution::Output(result) = result else {
        panic!("test peer copies should succeed");
    };
    let (output, completion) = result.into_parts();
    let values = output.to_vec1::<f32>().unwrap();
    assert_eq!(values, vec![4.0, 6.0]);
    let evidence = completion
        .complete(
            &ExecutionDigest::f32_tensor(&[2], &[2.0, 3.0]),
            &ExecutionDigest::f32_tensor(&[2], &values),
        )
        .unwrap();
    assert_eq!(
        evidence.execution_devices,
        vec![
            RuntimeDeviceIdentity {
                kind: RuntimeDeviceKind::Cuda,
                ordinal: Some(0),
            },
            RuntimeDeviceIdentity {
                kind: RuntimeDeviceKind::Cuda,
                ordinal: Some(1),
            },
        ]
    );
    assert!(evidence.peer_transfers_sha256.is_some());
}

#[test]
fn transfer_and_topology_bounds_fail_closed() {
    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let mesh = local_mesh(&runtime);
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    let oversized = Tensor::new(&[1_f32, 2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    assert!(batch
        .execute_mesh_or_fallback(&oversized, &cancellation, |input, _groups, mesh, _| {
            let _ = mesh.transfer("home", "peer", input)?;
            Ok(AcceleratorKernelOutcome::Output(input.clone()))
        })
        .is_err());

    let disconnected = AcceleratorDeviceMesh::new(
        "home",
        vec![
            AcceleratorMeshDevice::new("home", runtime.device().clone()),
            AcceleratorMeshDevice::new(
                "peer",
                RuntimeDevice::test_accelerator(RuntimeDeviceKind::Cuda, 1).unwrap(),
            ),
        ],
        Vec::new(),
        0,
    );
    assert!(disconnected.is_err());

    let duplicate = AcceleratorDeviceMesh::new(
        "home",
        vec![
            AcceleratorMeshDevice::new("home", runtime.device().clone()),
            AcceleratorMeshDevice::new("same-device", runtime.device().clone()),
        ],
        vec![
            AcceleratorPeerTransferSpec::new("home", "same-device", 8, 1),
            AcceleratorPeerTransferSpec::new("same-device", "home", 8, 1),
        ],
        16,
    );
    assert!(duplicate.is_err());
}

#[test]
fn stale_mesh_and_untracked_mesh_execution_are_rejected() {
    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let mesh = local_mesh(&runtime);
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    assert!(hierarchy
        .resolve_accelerator_batch(&declaration, None, &permit, &cancellation)
        .is_err());

    let wrong_mesh = AcceleratorDeviceMesh::new(
        "home",
        vec![AcceleratorMeshDevice::new("home", runtime.device().clone())],
        Vec::new(),
        0,
    )
    .unwrap();
    assert!(hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &wrong_mesh, None, &permit, &cancellation,)
        .is_err());
}

#[test]
fn confidential_mesh_binds_exact_gpu_and_fabric_claim_indices() {
    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let mesh = AcceleratorDeviceMesh::new(
        "home",
        vec![
            AcceleratorMeshDevice::new("home", runtime.device().clone())
                .with_attestation_gpu_claim_index(7),
            AcceleratorMeshDevice::new(
                "peer",
                RuntimeDevice::test_accelerator(RuntimeDeviceKind::Cuda, 1).unwrap(),
            )
            .with_attestation_gpu_claim_index(9),
        ],
        vec![
            AcceleratorPeerTransferSpec::new("home", "peer", 8, 1),
            AcceleratorPeerTransferSpec::new("peer", "home", 8, 1),
        ],
        16,
    )
    .unwrap()
    .with_attestation_fabric_claim_indices(vec![11])
    .unwrap();
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(
            &spec().with_security(AcceleratorSecurityRequirement::ConfidentialGpu),
            &mesh,
        )
        .unwrap();
    let report = confidential_report(
        &hierarchy,
        &declaration,
        vec![
            device_claim(11, "nvswitch"),
            device_claim(7, "gpu"),
            device_claim(9, "gpu"),
        ],
    );
    let binding =
        ConfidentialGpuBinding::from_attestation_report_for_test(&report, &declaration).unwrap();
    assert!(matches!(
        hierarchy
            .resolve_accelerator_mesh_batch(
                &declaration,
                &mesh,
                Some(&binding),
                &permit,
                &cancellation,
            )
            .unwrap(),
        AcceleratorBatchResolution::Ready(_)
    ));

    let missing_peer = confidential_report(
        &hierarchy,
        &declaration,
        vec![device_claim(7, "gpu"), device_claim(11, "nvswitch")],
    );
    assert!(
        ConfidentialGpuBinding::from_attestation_report_for_test(&missing_peer, &declaration,)
            .is_err()
    );

    let extra_gpu = confidential_report(
        &hierarchy,
        &declaration,
        vec![
            device_claim(7, "gpu"),
            device_claim(9, "gpu"),
            device_claim(11, "nvswitch"),
            device_claim(13, "gpu"),
        ],
    );
    assert!(
        ConfidentialGpuBinding::from_attestation_report_for_test(&extra_gpu, &declaration,)
            .is_err()
    );
}

#[test]
fn mesh_fallback_and_receipt_expose_no_model_topology() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AcceleratorDeviceMesh>();
    assert_send_sync::<AcceleratorDeviceMeshDeclaration>();
    assert_send_sync::<AcceleratorMeshDevice>();
    assert_send_sync::<AcceleratorPeerTransferSpec>();

    let (_directory, runtime, hierarchy, permit, cancellation) = hierarchy();
    let mesh = local_mesh(&runtime);
    let declaration = hierarchy
        .declare_accelerator_mesh_residency(&spec(), &mesh)
        .unwrap();
    let AcceleratorBatchResolution::Ready(batch) = hierarchy
        .resolve_accelerator_mesh_batch(&declaration, &mesh, None, &permit, &cancellation)
        .unwrap()
    else {
        panic!("mesh should resolve");
    };
    let input_tensor = Tensor::new(&[2_f32, 3_f32], runtime.device().tensor_device()).unwrap();
    let AcceleratorFusedExecution::Fallback(fallback) = batch
        .execute_mesh_or_fallback(&input_tensor, &cancellation, |input, _groups, mesh, _| {
            let _ = mesh.transfer("home", "peer", input)?;
            Ok(AcceleratorKernelOutcome::Unavailable)
        })
        .unwrap()
    else {
        panic!("typed mesh failure should enter exact fallback");
    };
    let input = ExecutionDigest::f32_tensor(&[2], &[2.0, 3.0]);
    let output = ExecutionDigest::f32_tensor(&[2], &[4.0, 6.0]);
    let evidence = fallback.complete(&input, &output).unwrap();
    let receipt = runtime
        .receipt_with_accelerator(
            ModelIdentity::new("test", "v1", hierarchy.store().sha256()),
            input,
            output,
            evidence,
        )
        .unwrap();
    assert_eq!(receipt.schema, ExecutionReceipt::ACCELERATOR_MESH_SCHEMA);
    let json = serde_json::to_string(&receipt).unwrap();
    assert!(!json.contains("\"home\""));
    assert!(!json.contains("\"peer\""));
    assert!(!json.contains("expert-0"));
    assert!(!json.contains("layer.0"));
    assert!(!json.contains("GH100"));
}
