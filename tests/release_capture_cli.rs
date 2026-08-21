#![cfg(feature = "embedded-inference")]

use std::process::Command;

use a3s_power::inference::{
    ReleaseCapture, ReleaseEvidenceBundle, ReleaseEvidencePolicy, ReleasePlatform,
    ReleaseRevisionBinding,
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
