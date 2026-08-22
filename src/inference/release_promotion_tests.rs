use super::{
    AcceleratorResidencyDeclaration, HardwareEvidenceBinding, ReleaseCapture,
    ReleaseCaptureSecurity, ReleaseEvidenceBundle, ReleasePlatform, RuntimeDeviceIdentity,
    RuntimeDeviceKind, ShapeProfileBinding, StorageBenchmarkSystem,
};
use crate::api::prompt_policy::canonical_gpu_execution_digest;
use crate::config::GpuConfig;
use crate::error::Result;
use crate::tee::attestation::{
    build_claims_report_data, AttestationClaimsV2, AttestationReport, ExecutionPolicyClaim,
    GpuDeviceClaim, GpuDeviceValidationClaim, GpuEvidenceClaim, ModelDigestClaim, ModelDigestKind,
    RuntimePolicyClaim, TeeType,
};
use crate::verify::{
    verify_confidential_gpu_attestation, ExpectedGpuDevices, ExpectedGpuEvidence, HardwareVerifier,
    VerifyOptions,
};

const NONCE: [u8; 32] = [0x11; 32];
const EVIDENCE_DIGEST: [u8; 32] = [0x33; 32];
const VERDICT_DIGEST: [u8; 32] = [0x44; 32];

struct AcceptFixtureSignature;

impl HardwareVerifier for AcceptFixtureSignature {
    fn verify_hardware_signature(&self, _report: &AttestationReport) -> Result<()> {
        Ok(())
    }
}

fn device_claim() -> GpuDeviceClaim {
    GpuDeviceClaim {
        index: 0,
        device_type: "gpu".to_string(),
        attestation_nonce: Some(NONCE.to_vec()),
        hwmodel: Some("GH100 A01 GSP BROM".to_string()),
        ueid: Some("gpu-fixture-0".to_string()),
        oemid: Some("5703".to_string()),
        claims_version: Some("3.0".to_string()),
        driver_version: Some("590.12".to_string()),
        firmware_version: Some("96.00.A5.00.01".to_string()),
        measurements_result: Some("success".to_string()),
        secure_boot: Some(true),
        debug_status: Some("disabled".to_string()),
        validation: GpuDeviceValidationClaim {
            arch_check: Some(true),
            attestation_report_cert_chain_fwid_match: Some(true),
            attestation_report_parsed: Some(true),
            attestation_report_nonce_match: Some(true),
            attestation_report_signature_verified: Some(true),
            driver_rim_fetched: Some(true),
            driver_rim_schema_validated: Some(true),
            driver_rim_signature_verified: Some(true),
            driver_rim_version_match: Some(true),
            driver_rim_measurements_available: Some(true),
            firmware_rim_fetched: Some(true),
            firmware_rim_schema_validated: Some(true),
            firmware_rim_signature_verified: Some(true),
            firmware_rim_version_match: Some(true),
            firmware_rim_measurements_available: Some(true),
            firmware_index_no_conflict: Some(true),
        },
    }
}

fn report_for(declaration: &AcceleratorResidencyDeclaration) -> AttestationReport {
    let claims = AttestationClaimsV2::new(TeeType::SevSnp)
        .with_nonce(Some(&NONCE))
        .with_model(ModelDigestClaim {
            name: "release-workload".to_string(),
            kind: ModelDigestKind::PlaintextWeightsSha256,
            digest: hex::decode(&declaration.weights_sha256).unwrap(),
            plaintext_digest: None,
            ciphertext_digest: None,
        })
        .with_gpu(
            GpuEvidenceClaim::new("nvidia-nras", EVIDENCE_DIGEST.to_vec())
                .with_evidence_format("nvidia-nvattest-evidence-json")
                .with_evidence_count(1)
                .with_nonce(&NONCE)
                .with_verdict_format("nvidia-nvattest-attestation-json")
                .with_verdict_digest(VERDICT_DIGEST.to_vec())
                .with_devices(vec![device_claim()]),
        )
        .with_runtime(
            RuntimePolicyClaim::new().with_execution(ExecutionPolicyClaim {
                gpu_sha256: hex::decode(&declaration.execution_policy_sha256).unwrap(),
            }),
        );
    let report_data = build_claims_report_data(&claims).unwrap();
    let measurement = vec![0x55; 48];
    let mut raw_report = vec![0_u8; 0xc0];
    raw_report[0x50..0x90].copy_from_slice(&report_data);
    raw_report[0x90..0xc0].copy_from_slice(&measurement);
    AttestationReport {
        version: "1.0".to_string(),
        tee_type: TeeType::SevSnp,
        report_data,
        measurement,
        raw_report: Some(raw_report),
        timestamp: chrono::Utc::now(),
        nonce: Some(NONCE.to_vec()),
        claims: Some(claims),
    }
}

fn verify_options<'a>(
    report: &AttestationReport,
    declaration: &AcceleratorResidencyDeclaration,
    verifier: &'a dyn HardwareVerifier,
) -> VerifyOptions<'a> {
    VerifyOptions {
        nonce: Some(NONCE.to_vec()),
        expected_model_hash: Some(hex::decode(&declaration.weights_sha256).unwrap()),
        expected_measurement: Some(report.measurement.clone()),
        expected_gpu_evidence_digest: Some(EVIDENCE_DIGEST.to_vec()),
        expected_gpu_verdict_digest: Some(VERDICT_DIGEST.to_vec()),
        expected_gpu_evidence: Some(ExpectedGpuEvidence {
            provider: Some("nvidia-nras".to_string()),
            evidence_format: Some("nvidia-nvattest-evidence-json".to_string()),
            verdict_format: Some("nvidia-nvattest-attestation-json".to_string()),
            evidence_count: Some(1),
        }),
        expected_gpu_devices: Some(ExpectedGpuDevices {
            gpu_count: Some(1),
            nvswitch_count: Some(0),
            gpu_ueids: vec!["gpu-fixture-0".to_string()],
            oemids: vec!["5703".to_string()],
            claims_versions: vec!["3.0".to_string()],
            hwmodels: vec!["GH100 A01 GSP BROM".to_string()],
            driver_versions: vec!["590.12".to_string()],
            firmware_versions: vec!["96.00.A5.00.01".to_string()],
            ..ExpectedGpuDevices::default()
        }),
        expected_chat_template_digest: None,
        expected_decoding_parameters_digest: None,
        expected_gpu_execution_digest: Some(
            hex::decode(&declaration.execution_policy_sha256).unwrap(),
        ),
        hardware_verifier: Some(verifier),
    }
}

fn local_cuda_capture(
    declaration: &AcceleratorResidencyDeclaration,
    source: ReleaseCapture,
) -> ReleaseCapture {
    source.verify().unwrap();
    let shape = ShapeProfileBinding::new(
        source.shape_binding.weights_sha256.clone(),
        source.shape_binding.graph_sha256.clone(),
        source.shape_binding.runtime_device,
        source.shape_binding.device_topology_sha256.clone(),
        source.shape_binding.runtime_reservations,
        declaration.execution_policy_sha256.clone(),
    )
    .unwrap();
    let mut contracts = source.contracts;
    contracts.exact_fallback.selection.binding_sha256 = shape.binding_sha256().unwrap();
    ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        shape,
        source.tensor_batch,
        contracts,
    )
    .unwrap()
}

fn retarget_accelerator_capture(
    mut source: ReleaseCapture,
    runtime_device: RuntimeDeviceIdentity,
    system: StorageBenchmarkSystem,
    topology_sha256: &str,
) -> ReleaseCapture {
    source.verify().unwrap();
    source.tensor_batch.system = system;
    source.tensor_batch.binding = HardwareEvidenceBinding::new(
        source.tensor_batch.binding.power_version.clone(),
        source.tensor_batch.binding.power_commit.clone(),
        source.tensor_batch.binding.weights_sha256.clone(),
        source.tensor_batch.binding.graph_source_sha256.clone(),
        runtime_device,
        &source.tensor_batch.system,
    )
    .unwrap();
    for sample in &mut source.tensor_batch.samples {
        let copies = sample.execution_count as u64;
        sample.boundary.host_to_device_copy_operations = copies;
        sample.boundary.device_to_host_copy_operations = copies;
    }
    source.tensor_batch.sha256 =
        super::execution_cost::recompute_test_report_sha256(&source.tensor_batch);

    let shape = ShapeProfileBinding::new(
        source.shape_binding.weights_sha256.clone(),
        source.shape_binding.graph_sha256.clone(),
        runtime_device,
        topology_sha256,
        source.shape_binding.runtime_reservations,
        source.shape_binding.tee_policy_sha256.clone(),
    )
    .unwrap();
    let mut contracts = source.contracts;
    contracts.exact_fallback.selection.binding_sha256 = shape.binding_sha256().unwrap();
    contracts.exact_fallback.selection.runtime_device = runtime_device;
    contracts.replica_recovery.before.device = runtime_device;
    contracts.replica_recovery.retired.device = runtime_device;
    contracts.replica_recovery.recovered.device = runtime_device;
    ReleaseCapture::build(
        ReleaseCaptureSecurity::Local,
        shape,
        source.tensor_batch,
        contracts,
    )
    .unwrap()
}

#[test]
fn opaque_confidential_proof_enables_a_strict_v1_bundle() {
    let published: ReleaseCapture = serde_json::from_str(include_str!(
        "../../docs/benchmarks/release-contract-windows-20260821/cuda.json"
    ))
    .unwrap();
    let execution_policy_sha256 = hex::encode(
        canonical_gpu_execution_digest(&GpuConfig {
            gpu_layers: -1,
            main_gpu: 0,
            tensor_split: vec![1.0],
            cpu_tensors: Vec::new(),
            gpu_tensors: Vec::new(),
        })
        .unwrap(),
    );
    let declaration = AcceleratorResidencyDeclaration::confidential_release_fixture(
        published.shape_binding.weights_sha256.clone(),
        published.shape_binding.runtime_device,
        execution_policy_sha256,
    );
    let mut confidential_system = published.tensor_batch.system.clone();
    confidential_system.device_class = "NVIDIA H100 80 GB".to_string();
    let confidential_source = retarget_accelerator_capture(
        published.clone(),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cuda,
            ordinal: Some(0),
        },
        confidential_system,
        &"a".repeat(64),
    );
    let local = local_cuda_capture(&declaration, confidential_source);
    let report = report_for(&declaration);
    let verifier = AcceptFixtureSignature;
    let options = verify_options(&report, &declaration, &verifier);
    let (_, proof) = verify_confidential_gpu_attestation(&report, &options).unwrap();

    let cpu: ReleaseCapture = serde_json::from_str(include_str!(
        "../../docs/benchmarks/release-contract-windows-20260821/cpu.json"
    ))
    .unwrap();
    let wrong_platform = cpu
        .promote_confidential_gpu(&proof, &declaration)
        .unwrap_err();
    assert!(wrong_platform.to_string().contains("local CUDA capture"));

    let promoted = local
        .clone()
        .promote_confidential_gpu(&proof, &declaration)
        .unwrap();

    promoted.verify().unwrap();
    assert_eq!(local.platform().unwrap(), ReleasePlatform::Cuda);
    assert_eq!(
        promoted.platform().unwrap(),
        ReleasePlatform::ConfidentialGpu
    );
    assert_ne!(promoted.sha256, local.sha256);
    let ReleaseCaptureSecurity::ConfidentialGpu { binding } = &promoted.security else {
        panic!("proof-backed promotion must mint the confidential security class");
    };
    assert_eq!(
        binding.accelerator_declaration_sha256(),
        declaration.declaration_sha256
    );
    assert_eq!(binding.weights_sha256(), declaration.weights_sha256);
    assert_eq!(binding.runtime_device(), declaration.runtime_device);
    assert_eq!(binding.tee_type(), TeeType::SevSnp);

    let relabel_error = ReleaseCapture::build(
        promoted.security.clone(),
        local.shape_binding,
        local.tensor_batch,
        local.contracts,
    )
    .unwrap_err();
    assert!(relabel_error.to_string().contains("proof-backed promotion"));

    let metal_system = StorageBenchmarkSystem {
        os: "macOS 15.6".to_string(),
        architecture: "arm64".to_string(),
        cpu_model: "Apple M2 Pro".to_string(),
        logical_cpus: 10,
        ram_bytes: 16 * 1024 * 1024 * 1024,
        filesystem_class: "apfs".to_string(),
        device_class: "Apple M2 Pro integrated GPU".to_string(),
    };
    let metal = retarget_accelerator_capture(
        published.clone(),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Metal,
            ordinal: Some(0),
        },
        metal_system.clone(),
        &"b".repeat(64),
    );
    let cpu: ReleaseCapture = serde_json::from_str(include_str!(
        "../../docs/benchmarks/release-contract-windows-20260821/cpu.json"
    ))
    .unwrap();
    let expected_version = cpu.tensor_batch.binding.power_version.clone();
    let expected_commit = cpu.tensor_batch.binding.power_commit.clone();
    let mut virtual_metal_system = metal_system;
    virtual_metal_system.cpu_model = "Apple M1 (Virtual)".to_string();
    virtual_metal_system.device_class = "Apple Paravirtual device".to_string();
    let virtual_metal = retarget_accelerator_capture(
        published.clone(),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Metal,
            ordinal: Some(0),
        },
        virtual_metal_system,
        &"c".repeat(64),
    );
    let virtual_error = ReleaseEvidenceBundle::build_strict_v1(vec![
        promoted.clone(),
        virtual_metal,
        published.clone(),
        cpu.clone(),
    ])
    .unwrap_err();
    assert!(virtual_error
        .to_string()
        .contains("rejects virtual, emulated, translated"));

    let bundle =
        ReleaseEvidenceBundle::build_strict_v1(vec![promoted, metal, published, cpu]).unwrap();
    bundle
        .verify_strict_v1_release(&bundle.sha256, &expected_version, &expected_commit)
        .unwrap();
}
