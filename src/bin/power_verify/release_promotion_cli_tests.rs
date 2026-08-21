use super::{parse_args, run};

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn complete_promotion_args() -> Vec<String> {
    args(&[
        "--file",
        "report.json",
        "--promote-capture",
        "cuda.json",
        "--accelerator-declaration",
        "accelerator.json",
        "--promoted-output",
        "confidential.json",
        "--gpu-confidential",
        "--model-hash",
        "11".repeat(32).as_str(),
        "--gpu-evidence-digest",
        "33".repeat(32).as_str(),
        "--gpu-execution-digest",
        "22".repeat(32).as_str(),
    ])
}

#[test]
fn parser_accepts_the_complete_release_promotion_triplet() {
    let parsed = parse_args(&args(&[
        "--file",
        "report.json",
        "--promote-capture",
        "cuda.json",
        "--accelerator-declaration",
        "accelerator.json",
        "--promoted-output",
        "confidential.json",
    ]))
    .unwrap();

    assert_eq!(parsed.promote_capture.as_deref(), Some("cuda.json"));
    assert_eq!(
        parsed.accelerator_declaration.as_deref(),
        Some("accelerator.json")
    );
    assert_eq!(parsed.promoted_output.as_deref(), Some("confidential.json"));
}

#[test]
fn promotion_requires_all_three_artifact_paths() {
    let error = run(&args(&[
        "--file",
        "report.json",
        "--promote-capture",
        "cuda.json",
    ]))
    .unwrap_err();

    assert!(error.to_string().contains(
        "requires --promote-capture, --accelerator-declaration, and --promoted-output together"
    ));
}

#[test]
fn promotion_rejects_offline_verification() {
    let mut values = complete_promotion_args();
    values.push("--allow-offline".to_string());
    let error = run(&values).unwrap_err();

    assert!(error.to_string().contains("rejects --allow-offline"));
}

#[test]
fn promotion_requires_the_confidential_gpu_profile() {
    let values = complete_promotion_args()
        .into_iter()
        .filter(|value| value != "--gpu-confidential")
        .collect::<Vec<_>>();
    let error = run(&values).unwrap_err();

    assert!(error.to_string().contains("requires --gpu-confidential"));
}

#[test]
fn promotion_rejects_live_report_input() {
    let mut values = complete_promotion_args();
    values.extend(["--url".to_string(), "http://127.0.0.1:11434".to_string()]);
    let error = run(&values).unwrap_err();

    assert!(error.to_string().contains("rejects live --url input"));
}

#[test]
fn promotion_requires_independent_weight_and_execution_pins() {
    let without_model = complete_promotion_args()
        .into_iter()
        .skip_while(|value| value != "--model-hash")
        .skip(2)
        .collect::<Vec<_>>();
    let mut prefix = args(&[
        "--file",
        "report.json",
        "--promote-capture",
        "cuda.json",
        "--accelerator-declaration",
        "accelerator.json",
        "--promoted-output",
        "confidential.json",
        "--gpu-confidential",
    ]);
    prefix.extend(without_model);
    let model_error = run(&prefix).unwrap_err();
    assert!(model_error.to_string().contains("requires --model-hash"));

    let values = args(&[
        "--file",
        "report.json",
        "--promote-capture",
        "cuda.json",
        "--accelerator-declaration",
        "accelerator.json",
        "--promoted-output",
        "confidential.json",
        "--gpu-confidential",
        "--model-hash",
        "11".repeat(32).as_str(),
        "--gpu-evidence-digest",
        "33".repeat(32).as_str(),
    ]);
    let execution_error = run(&values).unwrap_err();
    assert!(execution_error
        .to_string()
        .contains("requires --gpu-execution-digest"));
}

#[test]
fn promotion_requires_a_pin_for_the_preserved_vendor_evidence() {
    let mut values = complete_promotion_args();
    let position = values
        .iter()
        .position(|value| value == "--gpu-evidence-digest")
        .unwrap();
    values.drain(position..=position + 1);
    let error = run(&values).unwrap_err();

    assert!(error.to_string().contains("requires --gpu-evidence-digest"));
}

#[cfg(not(feature = "embedded-inference"))]
#[test]
fn promotion_requires_a_verifier_build_with_the_runtime_contracts() {
    let error = run(&complete_promotion_args()).unwrap_err();

    assert!(error.to_string().contains("embedded-inference feature"));
}

#[cfg(all(feature = "embedded-inference", not(feature = "hw-verify")))]
#[test]
fn promotion_requires_a_verifier_build_with_hardware_verification() {
    let error = run(&complete_promotion_args()).unwrap_err();

    assert!(error.to_string().contains("hw-verify feature"));
}
