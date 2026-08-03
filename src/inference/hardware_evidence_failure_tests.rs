use super::hardware_evidence_tests::{
    binding, bundle, digest, parity_artifact, storage_report, system, tuning_evidence,
};
use super::{
    ExecutionDigest, HardwareEvidenceBinding, HardwareEvidenceBundle, ModelParityArtifact,
    RuntimeDeviceIdentity, RuntimeDeviceKind, StorageBenchmarkSample, TuningProfilePolicy,
    WeightReadStrategy,
};

#[test]
fn nested_or_digest_tampering_is_rejected() {
    let original = bundle(true);

    let mut storage = original.clone();
    storage.storage_comparison.output_byte_parity = false;
    assert!(storage.verify().is_err());

    let mut tuning = original.clone();
    tuning.tuning_decision.selected_configuration_sha256 = digest('c');
    assert!(tuning.verify().is_err());

    let mut parity = original.clone();
    parity.parity_artifacts[0].tested_output = ExecutionDigest::token_ids(&[10]);
    assert!(parity.verify().is_err());

    let mut platform = original.clone();
    platform.binding.environment_sha256 = digest('d');
    assert!(platform.verify().is_err());

    let mut digest_tamper = original.clone();
    digest_tamper.sha256 = digest('e');
    assert!(digest_tamper.verify().is_err());
    assert!(original.verify_pinned(&digest('f')).is_err());
}

#[test]
fn storage_parity_and_mixed_hardware_fail_closed() {
    let platform = binding();
    let evidence = tuning_evidence(&platform, true);
    let mut output_mismatch = storage_report(WeightReadStrategy::PositionalBuffered);
    output_mismatch.output_sha256 = digest('8');
    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        vec![storage_report(WeightReadStrategy::Mmap), output_mismatch],
        evidence.clone(),
        TuningProfilePolicy::default(),
        vec![parity_artifact(&digest('b'))],
    )
    .is_err());

    let mut other_hardware = storage_report(WeightReadStrategy::PositionalBuffered);
    other_hardware.system.cpu_model = "other-cpu".to_string();
    assert!(HardwareEvidenceBundle::build(
        platform,
        vec![storage_report(WeightReadStrategy::Mmap), other_hardware],
        evidence,
        TuningProfilePolicy::default(),
        vec![parity_artifact(&digest('b'))],
    )
    .is_err());
}

#[test]
fn storage_model_and_tuning_platform_mismatches_fail_closed() {
    for mutate in 0..5 {
        let mut platform = binding();
        match mutate {
            0 => platform.weights_sha256 = digest('8'),
            1 => platform.graph_source_sha256 = digest('8'),
            2 => platform.runtime_sha256 = digest('8'),
            3 => platform.device_sha256 = digest('8'),
            4 => platform.environment_sha256 = digest('8'),
            _ => unreachable!(),
        }
        assert!(HardwareEvidenceBundle::build(
            platform.clone(),
            vec![
                storage_report(WeightReadStrategy::Mmap),
                storage_report(WeightReadStrategy::PositionalBuffered),
            ],
            tuning_evidence(&binding(), true),
            TuningProfilePolicy::default(),
            vec![parity_artifact(&digest('b'))],
        )
        .is_err());
    }
}

#[test]
fn parity_artifacts_must_be_exact_unique_and_cover_the_selected_configuration() {
    let platform = binding();
    let evidence = tuning_evidence(&platform, true);
    let reports = || {
        vec![
            storage_report(WeightReadStrategy::Mmap),
            storage_report(WeightReadStrategy::PositionalBuffered),
        ]
    };

    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        reports(),
        evidence.clone(),
        TuningProfilePolicy::default(),
        vec![parity_artifact(&digest('a'))],
    )
    .is_err());

    let mut mismatch = parity_artifact(&digest('b'));
    mismatch.tested_output = ExecutionDigest::token_ids(&[99]);
    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        reports(),
        evidence.clone(),
        TuningProfilePolicy::default(),
        vec![mismatch],
    )
    .is_err());

    let mut other_platform = parity_artifact(&digest('b'));
    other_platform.binding.environment_sha256 = digest('8');
    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        reports(),
        evidence.clone(),
        TuningProfilePolicy::default(),
        vec![other_platform],
    )
    .is_err());

    let artifact = parity_artifact(&digest('b'));
    assert!(HardwareEvidenceBundle::build(
        platform,
        reports(),
        evidence,
        TuningProfilePolicy::default(),
        vec![artifact.clone(), artifact],
    )
    .is_err());
}

#[test]
fn evidence_collection_and_nested_report_bounds_are_enforced() {
    let platform = binding();
    let evidence = tuning_evidence(&platform, true);
    let parity = vec![parity_artifact(&digest('b'))];
    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        vec![storage_report(WeightReadStrategy::Mmap); 129],
        evidence.clone(),
        TuningProfilePolicy::default(),
        parity.clone(),
    )
    .is_err());

    let mut excessive_samples = storage_report(WeightReadStrategy::Mmap);
    excessive_samples.samples = vec![
        StorageBenchmarkSample {
            latency_nanos: 1,
            bytes_read: 4,
            bytes_per_second: 1,
            source_fallbacks: 0,
        };
        1_001
    ];
    excessive_samples.total_requested_bytes = 4_004;
    excessive_samples.total_read_bytes = 4_004;
    assert!(HardwareEvidenceBundle::build(
        platform.clone(),
        vec![
            excessive_samples,
            storage_report(WeightReadStrategy::PositionalBuffered),
        ],
        evidence.clone(),
        TuningProfilePolicy::default(),
        parity,
    )
    .is_err());

    let too_many_parity = vec![parity_artifact(&digest('b')); 257];
    assert!(HardwareEvidenceBundle::build(
        platform,
        vec![
            storage_report(WeightReadStrategy::Mmap),
            storage_report(WeightReadStrategy::PositionalBuffered),
        ],
        evidence,
        TuningProfilePolicy::default(),
        too_many_parity,
    )
    .is_err());
}

#[test]
fn malformed_platform_artifact_and_unknown_fields_are_rejected() {
    let mut malformed_system = system();
    malformed_system.cpu_model = "x".repeat(513);
    assert!(HardwareEvidenceBinding::new(
        env!("CARGO_PKG_VERSION"),
        "a".repeat(40),
        digest('1'),
        digest('4'),
        RuntimeDeviceIdentity {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
        },
        &malformed_system,
    )
    .is_err());

    assert!(ModelParityArtifact::new(
        binding(),
        "bad",
        digest('6'),
        digest('7'),
        digest('b'),
        ExecutionDigest::token_ids(&[1]),
        ExecutionDigest::token_ids(&[2]),
        ExecutionDigest::token_ids(&[2]),
    )
    .is_err());

    let value = serde_json::to_value(bundle(true)).unwrap();
    let mut object = value.as_object().unwrap().clone();
    object.insert("unexpected".to_string(), serde_json::Value::Bool(true));
    assert!(
        serde_json::from_value::<HardwareEvidenceBundle>(serde_json::Value::Object(object))
            .is_err()
    );
}
