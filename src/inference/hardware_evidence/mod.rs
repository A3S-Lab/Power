mod digest;
mod types;
mod validation;

pub use types::{HardwareEvidenceBinding, HardwareEvidenceBundle, ModelParityArtifact};

use crate::error::{PowerError, Result};

use super::{
    compare_storage_benchmarks, evaluate_tuning_profile, ExecutionDigest, RuntimeDeviceIdentity,
    StorageBenchmarkReport, StorageBenchmarkSystem, TuningProfileBinding, TuningProfileEvidence,
    TuningProfilePolicy,
};

pub(super) fn validate_parity_artifact(artifact: &ModelParityArtifact) -> Result<()> {
    validation::validate_parity_artifact(artifact)
}

pub(super) fn binding(
    power_version: String,
    power_commit: String,
    weights_sha256: String,
    graph_source_sha256: String,
    runtime_device: RuntimeDeviceIdentity,
    system: &StorageBenchmarkSystem,
) -> Result<HardwareEvidenceBinding> {
    validation::binding(
        power_version,
        power_commit,
        weights_sha256,
        graph_source_sha256,
        runtime_device,
        system,
    )
}

pub(super) fn tuning_binding(
    binding: &HardwareEvidenceBinding,
    calibration_workload: ExecutionDigest,
) -> Result<TuningProfileBinding> {
    validation::tuning_binding(binding, calibration_workload)
}

pub(super) fn build_bundle(
    binding: HardwareEvidenceBinding,
    storage_reports: Vec<StorageBenchmarkReport>,
    tuning_evidence: TuningProfileEvidence,
    tuning_policy: TuningProfilePolicy,
    parity_artifacts: Vec<ModelParityArtifact>,
) -> Result<HardwareEvidenceBundle> {
    validation::validate_storage_reports(&storage_reports)?;
    validation::validate_tuning_preflight(&tuning_evidence)?;
    validation::validate_parity_preflight(&parity_artifacts)?;
    let storage_reports = canonical_storage_reports(storage_reports)?;
    let tuning_evidence = canonical_tuning_evidence(tuning_evidence);
    let parity_artifacts = canonical_parity_artifacts(parity_artifacts);
    let storage_comparison = compare_storage_benchmarks(&storage_reports)?;
    let tuning_decision = evaluate_tuning_profile(&tuning_evidence, &tuning_policy)?;
    validate_derived_evidence(
        &binding,
        &storage_comparison,
        &tuning_evidence,
        &tuning_decision,
        &parity_artifacts,
    )?;
    let mut bundle = HardwareEvidenceBundle {
        schema: HardwareEvidenceBundle::SCHEMA.to_string(),
        binding,
        storage_reports,
        storage_comparison,
        tuning_evidence,
        tuning_policy,
        tuning_decision,
        parity_artifacts,
        sha256: String::new(),
    };
    bundle.sha256 = digest::bundle_sha256(&bundle)?;
    verify_bundle(&bundle)?;
    Ok(bundle)
}

pub(super) fn verify_bundle(bundle: &HardwareEvidenceBundle) -> Result<()> {
    if bundle.schema != HardwareEvidenceBundle::SCHEMA {
        return Err(PowerError::InvalidFormat(
            "hardware evidence bundle schema is unsupported".to_string(),
        ));
    }
    validation::validate_sha256(&bundle.sha256, "hardware evidence bundle SHA-256")?;
    validation::validate_storage_reports(&bundle.storage_reports)?;
    validation::validate_tuning_preflight(&bundle.tuning_evidence)?;
    validation::validate_parity_preflight(&bundle.parity_artifacts)?;

    if canonical_storage_reports(bundle.storage_reports.clone())? != bundle.storage_reports {
        return Err(PowerError::InvalidFormat(
            "hardware evidence storage reports are not canonically ordered".to_string(),
        ));
    }
    if canonical_tuning_evidence(bundle.tuning_evidence.clone()) != bundle.tuning_evidence {
        return Err(PowerError::InvalidFormat(
            "hardware evidence tuning candidates or rounds are not canonically ordered".to_string(),
        ));
    }
    if canonical_parity_artifacts(bundle.parity_artifacts.clone()) != bundle.parity_artifacts {
        return Err(PowerError::InvalidFormat(
            "hardware evidence model parity artifacts are not canonically ordered".to_string(),
        ));
    }

    let comparison = compare_storage_benchmarks(&bundle.storage_reports)?;
    if comparison != bundle.storage_comparison {
        return Err(PowerError::InvalidFormat(
            "hardware evidence storage comparison does not match its source reports".to_string(),
        ));
    }
    let decision = evaluate_tuning_profile(&bundle.tuning_evidence, &bundle.tuning_policy)?;
    if decision != bundle.tuning_decision {
        return Err(PowerError::InvalidFormat(
            "hardware evidence tuning decision does not match its source evidence and policy"
                .to_string(),
        ));
    }
    validate_derived_evidence(
        &bundle.binding,
        &bundle.storage_comparison,
        &bundle.tuning_evidence,
        &bundle.tuning_decision,
        &bundle.parity_artifacts,
    )?;

    let actual = digest::bundle_sha256(bundle)?;
    if actual != bundle.sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "hardware evidence bundle".to_string(),
            expected: bundle.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

pub(super) fn verify_pinned_bundle(
    bundle: &HardwareEvidenceBundle,
    expected_sha256: &str,
) -> Result<()> {
    validation::validate_sha256(expected_sha256, "pinned hardware evidence bundle SHA-256")?;
    verify_bundle(bundle)?;
    if bundle.sha256 != expected_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "hardware evidence bundle pin".to_string(),
            expected: expected_sha256.to_string(),
            actual: bundle.sha256.clone(),
        });
    }
    Ok(())
}

fn validate_derived_evidence(
    binding: &HardwareEvidenceBinding,
    comparison: &super::StorageBenchmarkComparison,
    tuning_evidence: &TuningProfileEvidence,
    tuning_decision: &super::TuningProfileDecision,
    parity_artifacts: &[ModelParityArtifact],
) -> Result<()> {
    if comparison.schema != super::StorageBenchmarkComparison::SCHEMA
        || !comparison.output_byte_parity
        || comparison.output_sha256s.len() != 1
        || comparison.groups.len() < 2
    {
        return Err(PowerError::InvalidFormat(
            "hardware evidence requires a multi-group storage comparison with exact output parity"
                .to_string(),
        ));
    }
    validation::validate_binding_for_system(binding, &comparison.system)?;
    if binding.power_version != comparison.power_version
        || binding.power_commit != comparison.power_commit
        || binding.weights_sha256 != comparison.model_collection_sha256
    {
        return Err(PowerError::InvalidFormat(
            "hardware evidence binding does not match the storage revision or model".to_string(),
        ));
    }
    let expected_tuning =
        binding.tuning_binding(tuning_evidence.binding.calibration_workload.clone())?;
    if tuning_evidence.binding != expected_tuning || tuning_decision.binding != expected_tuning {
        return Err(PowerError::InvalidFormat(
            "hardware evidence tuning data does not share the canonical model, runtime, device, and environment binding"
                .to_string(),
        ));
    }
    validation::validate_parity_artifacts(
        parity_artifacts,
        binding,
        &tuning_decision.selected_configuration_sha256,
    )
}

fn canonical_storage_reports(
    reports: Vec<StorageBenchmarkReport>,
) -> Result<Vec<StorageBenchmarkReport>> {
    validation::validate_storage_reports(&reports)?;
    let mut keyed = reports
        .into_iter()
        .map(|report| Ok((digest::report_sort_key(&report)?, report)))
        .collect::<Result<Vec<_>>>()?;
    keyed.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(keyed.into_iter().map(|(_, report)| report).collect())
}

fn canonical_tuning_evidence(mut evidence: TuningProfileEvidence) -> TuningProfileEvidence {
    for candidate in &mut evidence.candidates {
        candidate.rounds.sort_by_key(|round| round.round);
    }
    evidence
        .candidates
        .sort_by(|left, right| left.configuration_sha256.cmp(&right.configuration_sha256));
    evidence
}

fn canonical_parity_artifacts(mut artifacts: Vec<ModelParityArtifact>) -> Vec<ModelParityArtifact> {
    artifacts.sort_by(|left, right| {
        left.case_sha256
            .cmp(&right.case_sha256)
            .then_with(|| left.artifact_sha256.cmp(&right.artifact_sha256))
    });
    artifacts
}
