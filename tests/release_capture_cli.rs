#![cfg(feature = "embedded-inference")]

use std::path::Path;
use std::process::Command;

use a3s_power::inference::{
    InferenceLimits, ReleaseCapture, ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleasePlatform,
    ReleaseRevisionBinding, WeightStore,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

#[test]
fn isolated_cpu_fixture_captures_the_complete_model_neutral_contract() {
    let directory = tempfile::tempdir().expect("temporary output directory should exist");
    let capture_path = directory.path().join("capture.json");
    let mut command = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"));
    command
        .args([
            "release-fixture",
            "--device",
            "cpu",
            "--power-commit",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--filesystem-class",
            "test-memory",
            "--device-class",
            "test-cpu",
            "--cpu-model",
            "test-cpu",
            "--ram-bytes",
            "1073741824",
            "--tee-policy-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--host-fixed-bytes",
            "16777216",
            "--host-scratch-bytes",
            "67108864",
            "--device-fixed-bytes",
            "0",
            "--device-scratch-bytes",
            "0",
            "--items",
            "2",
            "--width",
            "64",
            "--warmup-rounds",
            "1",
            "--measured-rounds",
            "1",
        ])
        .arg("--output")
        .arg(&capture_path);
    let output = command
        .output()
        .expect("release fixture process should start");
    assert!(
        output.status.success(),
        "release fixture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let encoded = std::fs::read(&capture_path).expect("release fixture should write JSON");
    let capture: ReleaseCapture =
        serde_json::from_slice(&encoded).expect("release fixture output should be JSON");
    capture.verify().expect("release capture should verify");
    assert_eq!(capture.platform().unwrap(), ReleasePlatform::Cpu);
    assert_eq!(
        capture.contracts.peak_memory.host.final_used_bytes,
        capture.contracts.peak_memory.host.baseline_used_bytes
    );
    assert_eq!(
        capture.contracts.cancellation.lifecycle.cancelled_members,
        1
    );
    assert!(
        capture.contracts.queue_expiry.after.deadline_expirations
            > capture.contracts.queue_expiry.before.deadline_expirations
    );
    assert_eq!(
        capture.contracts.replica_recovery.recovered.ready_replicas,
        1
    );
    assert_eq!(
        capture.contracts.exact_fallback.reference_output,
        capture.contracts.exact_fallback.fallback_output
    );

    let serialized = String::from_utf8(encoded)
        .expect("release capture should be UTF-8")
        .to_ascii_lowercase();
    for forbidden in ["qwen", "gguf", "tokenizer", "mtp"] {
        assert!(
            !serialized.contains(forbidden),
            "release capture leaked model-specific term {forbidden}"
        );
    }
}

#[test]
fn release_bundle_verifier_rejects_partial_platform_evidence() {
    let capture: ReleaseCapture = serde_json::from_str(include_str!(
        "../docs/benchmarks/release-contract-windows-20260821/cpu.json"
    ))
    .unwrap();
    let revision = ReleaseRevisionBinding::from_capture(&capture).unwrap();
    let policy =
        ReleaseEvidencePolicy::new(revision.clone(), vec![capture.platform_binding().unwrap()])
            .unwrap();
    let bundle = ReleaseEvidenceBundle::build(policy, vec![capture]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let bundle_path = directory.path().join("release-evidence.json");
    let pin_path = directory.path().join("release-evidence.sha256");
    std::fs::write(&bundle_path, serde_json::to_vec_pretty(&bundle).unwrap()).unwrap();
    std::fs::write(&pin_path, format!("{}\n", bundle.sha256)).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("verify-release-bundle")
        .arg("--bundle")
        .arg(&bundle_path)
        .arg("--expected-sha256-file")
        .arg(&pin_path)
        .arg("--power-version")
        .arg(&revision.power_version)
        .arg("--power-commit")
        .arg(&revision.power_commit)
        .output()
        .expect("release bundle verifier should start");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("requires CPU, CUDA, Metal"),
        "unexpected verifier error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn release_capture_verifier_checks_digest_revision_and_platform() {
    let capture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/benchmarks/release-contract-windows-20260821/cpu.json");
    let capture: ReleaseCapture =
        serde_json::from_slice(&std::fs::read(&capture_path).unwrap()).unwrap();
    let revision = ReleaseRevisionBinding::from_capture(&capture).unwrap();

    let verified = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("verify-release-capture")
        .arg("--capture")
        .arg(&capture_path)
        .arg("--platform")
        .arg("cpu")
        .arg("--power-version")
        .arg(&revision.power_version)
        .arg("--power-commit")
        .arg(&revision.power_commit)
        .output()
        .expect("release capture verifier should start");
    assert!(
        verified.status.success(),
        "release capture verification failed: {}",
        String::from_utf8_lossy(&verified.stderr)
    );
    let receipt: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
    assert_eq!(
        receipt["schema"],
        "a3s.power.release-capture-verification.v1"
    );
    assert_eq!(receipt["verified"], true);
    assert_eq!(receipt["scope"], "single-capture");
    assert_eq!(receipt["strictV1BundleRequired"], true);
    assert_eq!(receipt["captureSha256"], capture.sha256);
    assert_eq!(receipt["revision"]["powerVersion"], revision.power_version);
    assert_eq!(receipt["revision"]["powerCommit"], revision.power_commit);
    assert_eq!(receipt["platformBinding"]["platform"], "cpu");

    let wrong_platform = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("verify-release-capture")
        .arg("--capture")
        .arg(&capture_path)
        .arg("--platform")
        .arg("cuda")
        .arg("--power-version")
        .arg(&revision.power_version)
        .arg("--power-commit")
        .arg(&revision.power_commit)
        .output()
        .unwrap();
    assert!(!wrong_platform.status.success());
    assert!(
        String::from_utf8_lossy(&wrong_platform.stderr).contains("has platform Cpu, expected Cuda")
    );

    let wrong_revision = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("verify-release-capture")
        .arg("--capture")
        .arg(&capture_path)
        .arg("--platform")
        .arg("cpu")
        .arg("--power-version")
        .arg(&revision.power_version)
        .arg("--power-commit")
        .arg("ffffffffffffffffffffffffffffffffffffffff")
        .output()
        .unwrap();
    assert!(!wrong_revision.status.success());
    assert!(String::from_utf8_lossy(&wrong_revision.stderr)
        .contains("does not match the expected Power version and source revision"));

    let directory = tempfile::tempdir().unwrap();
    let tampered_path = directory.path().join("tampered.json");
    let mut tampered = serde_json::to_value(&capture).unwrap();
    tampered["tensorBatch"]["exactOutputParity"] = serde_json::Value::Bool(false);
    std::fs::write(
        &tampered_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();
    let tampered_output = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("verify-release-capture")
        .arg("--capture")
        .arg(&tampered_path)
        .arg("--platform")
        .arg("cpu")
        .arg("--power-version")
        .arg(&revision.power_version)
        .arg("--power-commit")
        .arg(&revision.power_commit)
        .output()
        .unwrap();
    assert!(!tampered_output.status.success());
    assert!(String::from_utf8_lossy(&tampered_output.stderr)
        .contains("tensor batch benchmark report shape is invalid"));
}

#[test]
fn release_bundle_builder_rejects_mislabeled_captures_without_partial_outputs() {
    let capture_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs/benchmarks/release-contract-windows-20260821/cpu.json");
    let capture: ReleaseCapture =
        serde_json::from_slice(&std::fs::read(&capture_path).unwrap()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let bundle_path = directory.path().join("release-evidence.json");
    let pin_path = directory.path().join("release-evidence.sha256");

    let output = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("build-release-bundle")
        .arg("--cpu-capture")
        .arg(&capture_path)
        .arg("--cuda-capture")
        .arg(&capture_path)
        .arg("--metal-capture")
        .arg(&capture_path)
        .arg("--confidential-gpu-capture")
        .arg(&capture_path)
        .arg("--power-version")
        .arg(&capture.tensor_batch.binding.power_version)
        .arg("--power-commit")
        .arg(&capture.tensor_batch.binding.power_commit)
        .arg("--output")
        .arg(&bundle_path)
        .arg("--sha256-output")
        .arg(&pin_path)
        .output()
        .expect("release bundle builder should start");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("CUDA release capture has platform Cpu, expected Cuda"),
        "unexpected builder error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!bundle_path.exists());
    assert!(!pin_path.exists());
}

#[test]
fn caller_owned_reviewed_graph_uses_the_same_release_contract() {
    let directory = tempfile::tempdir().expect("temporary graph directory should exist");
    let weights = directory.path().join("weights");
    std::fs::create_dir(&weights).expect("weight directory should exist");
    let bias = [0.5_f32; 4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let view = TensorView::new(Dtype::F32, vec![4], &bias).expect("fixture tensor should build");
    serialize_to_file(
        vec![("bias", view)],
        None,
        &weights.join("weights.safetensors"),
    )
    .expect("fixture weights should serialize");

    let source_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    let plan = directory.path().join("plan.json");
    let inputs = directory.path().join("inputs.json");
    let reference = directory.path().join("reference.json");
    std::fs::write(
        &plan,
        serde_json::to_vec(&serde_json::json!({
            "schemaVersion": 1,
            "family": "vision.encoder",
            "role": "embedding",
            "source": {
                "format": "reviewed-json",
                "sha256": source_sha256,
                "opset": 1
            },
            "inputs": [{"name": "input", "shape": ["batch", 4]}],
            "outputs": [{"name": "output", "shape": ["batch", 4]}],
            "initializers": [{"name": "bias", "dtype": "float32", "shape": [4]}],
            "nodes": [{
                "name": "add-bias",
                "op": "Add",
                "inputs": ["input", "bias"],
                "outputs": ["output"],
                "attributes": {}
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &inputs,
        br#"{"items":[{"shape":[1,4],"values":[1.0,2.0,3.0,4.0]},{"shape":[1,4],"values":[5.0,6.0,7.0,8.0]}]}"#,
    )
    .unwrap();
    std::fs::write(&reference, br#"{"shape":[1,4],"values":[1.5,2.5,3.5,4.5]}"#).unwrap();

    let mut command = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"));
    command
        .args([
            "release-run",
            "--device",
            "cpu",
            "--power-commit",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--filesystem-class",
            "test-memory",
            "--device-class",
            "test-cpu",
            "--cpu-model",
            "test-cpu",
            "--ram-bytes",
            "1073741824",
            "--family",
            "vision.encoder",
            "--role",
            "embedding",
            "--source-format",
            "reviewed-json",
            "--source-sha256",
            source_sha256,
            "--opset",
            "1",
            "--tee-policy-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--profile-implementation-sha256",
            "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "--profile-shape-class-sha256",
            "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "--fallback-implementation-sha256",
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "--fallback-request-class-sha256",
            "1111111111111111111111111111111111111111111111111111111111111111",
            "--host-fixed-bytes",
            "16777216",
            "--host-scratch-bytes",
            "67108864",
            "--device-fixed-bytes",
            "0",
            "--device-scratch-bytes",
            "0",
            "--warmup-rounds",
            "1",
            "--measured-rounds",
            "1",
        ])
        .arg("--weights")
        .arg(&weights)
        .arg("--plan")
        .arg(&plan)
        .arg("--inputs")
        .arg(&inputs)
        .arg("--reference-output")
        .arg(&reference);
    let output = command
        .output()
        .expect("reviewed release process should start");
    assert!(
        output.status.success(),
        "reviewed release capture failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let capture: ReleaseCapture =
        serde_json::from_slice(&output.stdout).expect("release-run should emit JSON");
    capture.verify().expect("release-run capture should verify");
    assert_eq!(capture.platform().unwrap(), ReleasePlatform::Cpu);
    assert_eq!(
        capture.contracts.exact_fallback.reference_output,
        capture.contracts.exact_fallback.fallback_output
    );
    let serialized = String::from_utf8(output.stdout)
        .unwrap()
        .to_ascii_lowercase();
    assert!(!serialized.contains("vision.encoder"));
    assert!(!serialized.contains("embedding"));
}

#[test]
fn fixture_weights_are_create_new_reusable_and_digest_bound() {
    let directory = tempfile::tempdir().unwrap();
    let weights = directory.path().join("release-weights");
    let receipt = directory.path().join("weights-receipt.json");
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("materialize-release-fixture-weights")
        .arg("--directory")
        .arg(&weights)
        .arg("--width")
        .arg("64")
        .arg("--output")
        .arg(&receipt)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "weight materialization failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    assert_eq!(receipt["schema"], "a3s.power.release-fixture-weights.v1");
    let store = WeightStore::open(&weights, &InferenceLimits::default()).unwrap();
    assert_eq!(receipt["weightsSha256"], store.sha256());

    let capture_path = directory.path().join("persistent-capture.json");
    let capture = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .args([
            "release-fixture",
            "--device",
            "cpu",
            "--power-commit",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--filesystem-class",
            "test-memory",
            "--device-class",
            "test-cpu",
            "--cpu-model",
            "test-cpu",
            "--ram-bytes",
            "1073741824",
            "--tee-policy-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--host-fixed-bytes",
            "16777216",
            "--host-scratch-bytes",
            "67108864",
            "--device-fixed-bytes",
            "0",
            "--device-scratch-bytes",
            "0",
            "--items",
            "2",
            "--width",
            "64",
            "--warmup-rounds",
            "1",
            "--measured-rounds",
            "1",
        ])
        .arg("--fixture-weights")
        .arg(&weights)
        .arg("--output")
        .arg(&capture_path)
        .output()
        .unwrap();
    assert!(
        capture.status.success(),
        "persistent fixture capture failed: {}",
        String::from_utf8_lossy(&capture.stderr)
    );
    let capture: ReleaseCapture =
        serde_json::from_slice(&std::fs::read(capture_path).unwrap()).unwrap();
    assert_eq!(capture.shape_binding.weights_sha256, store.sha256());
    assert_eq!(capture.tensor_batch.binding.weights_sha256, store.sha256());

    let rollback_weights = directory.path().join("rollback-weights");
    let occupied_receipt = directory.path().join("occupied.json");
    std::fs::write(&occupied_receipt, b"caller-owned").unwrap();
    let rollback = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .arg("materialize-release-fixture-weights")
        .arg("--directory")
        .arg(&rollback_weights)
        .arg("--width")
        .arg("64")
        .arg("--output")
        .arg(&occupied_receipt)
        .output()
        .unwrap();
    assert!(!rollback.status.success());
    assert!(!rollback_weights.exists());
    assert_eq!(std::fs::read(occupied_receipt).unwrap(), b"caller-owned");
}

#[test]
fn confidential_fixture_rejects_cpu_without_partial_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let capture = directory.path().join("local-cuda.json");
    let declaration = directory.path().join("accelerator.json");
    let output = Command::new(env!("CARGO_BIN_EXE_a3s-power-tensor-batch-bench"))
        .args([
            "release-confidential-fixture",
            "--device",
            "cpu",
            "--power-commit",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--filesystem-class",
            "test-memory",
            "--device-class",
            "test-cpu",
            "--cpu-model",
            "test-cpu",
            "--ram-bytes",
            "1073741824",
            "--tee-policy-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--host-fixed-bytes",
            "16777216",
            "--host-scratch-bytes",
            "67108864",
            "--device-fixed-bytes",
            "0",
            "--device-scratch-bytes",
            "0",
            "--items",
            "2",
            "--width",
            "64",
            "--warmup-rounds",
            "1",
            "--measured-rounds",
            "1",
        ])
        .arg("--fixture-weights")
        .arg(directory.path().join("unused-weights"))
        .arg("--output")
        .arg(&capture)
        .arg("--accelerator-declaration-output")
        .arg(&declaration)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("explicitly selected CUDA device"));
    assert!(!capture.exists());
    assert!(!declaration.exists());
}
