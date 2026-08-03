use std::collections::BTreeSet;

use crate::error::{PowerError, Result};

use super::super::storage_benchmark::{MAX_BENCHMARK_CONCURRENCY, MAX_BENCHMARK_SAMPLES};
use super::super::tuning::MAX_CANDIDATES;
use super::super::tuning_types::MAX_ROUNDS_PER_CANDIDATE;
use super::super::{
    ExecutionDigest, RuntimeDeviceIdentity, StorageBenchmarkReport, StorageBenchmarkSystem,
    TuningProfileBinding, WeightSourceCoverage, WeightSourceRepresentation, WeightSourceRole,
};
use super::types::{HardwareEvidenceBinding, ModelParityArtifact};

pub(super) const MAX_STORAGE_REPORTS: usize = 128;
pub(super) const MAX_STORAGE_SOURCES: usize = 64;
pub(super) const MAX_PARITY_ARTIFACTS: usize = 256;
const MAX_LABEL_BYTES: usize = 512;

pub(super) fn binding(
    power_version: String,
    power_commit: String,
    weights_sha256: String,
    graph_source_sha256: String,
    runtime_device: RuntimeDeviceIdentity,
    system: &StorageBenchmarkSystem,
) -> Result<HardwareEvidenceBinding> {
    validate_label("Power version", &power_version)?;
    validate_revision(&power_commit, "Power commit")?;
    validate_sha256(&weights_sha256, "hardware evidence weights SHA-256")?;
    validate_sha256(
        &graph_source_sha256,
        "hardware evidence graph/source SHA-256",
    )?;
    runtime_device.validate()?;
    validate_system(system)?;
    let runtime_sha256 = super::digest::runtime_sha256(&power_version, &power_commit)?;
    let environment_sha256 = super::digest::environment_sha256(system)?;
    let device_sha256 = super::digest::device_sha256(runtime_device, &environment_sha256)?;
    Ok(HardwareEvidenceBinding {
        power_version,
        power_commit,
        weights_sha256,
        graph_source_sha256,
        runtime_device,
        runtime_sha256,
        device_sha256,
        environment_sha256,
    })
}

pub(super) fn tuning_binding(
    binding: &HardwareEvidenceBinding,
    calibration_workload: ExecutionDigest,
) -> Result<TuningProfileBinding> {
    validate_binding_shape(binding)?;
    validate_execution_digest(&calibration_workload, "calibration workload")?;
    Ok(TuningProfileBinding {
        weights_sha256: binding.weights_sha256.clone(),
        graph_source_sha256: binding.graph_source_sha256.clone(),
        calibration_workload,
        runtime_sha256: binding.runtime_sha256.clone(),
        device_sha256: binding.device_sha256.clone(),
        environment_sha256: binding.environment_sha256.clone(),
    })
}

pub(super) fn validate_binding_for_system(
    binding: &HardwareEvidenceBinding,
    system: &StorageBenchmarkSystem,
) -> Result<()> {
    validate_binding_shape(binding)?;
    let expected = self::binding(
        binding.power_version.clone(),
        binding.power_commit.clone(),
        binding.weights_sha256.clone(),
        binding.graph_source_sha256.clone(),
        binding.runtime_device,
        system,
    )?;
    if binding != &expected {
        return Err(PowerError::InvalidFormat(
            "hardware evidence binding does not match its canonical runtime, device, and environment digests"
                .to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_storage_reports(reports: &[StorageBenchmarkReport]) -> Result<()> {
    if reports.len() < 2 || reports.len() > MAX_STORAGE_REPORTS {
        return Err(PowerError::InvalidRequest(format!(
            "hardware evidence requires between 2 and {MAX_STORAGE_REPORTS} storage reports"
        )));
    }
    for report in reports {
        validate_storage_report(report)?;
    }
    Ok(())
}

pub(super) fn validate_parity_artifacts(
    artifacts: &[ModelParityArtifact],
    binding: &HardwareEvidenceBinding,
    selected_configuration_sha256: &str,
) -> Result<()> {
    if artifacts.is_empty() || artifacts.len() > MAX_PARITY_ARTIFACTS {
        return Err(PowerError::InvalidRequest(format!(
            "hardware evidence requires between 1 and {MAX_PARITY_ARTIFACTS} model parity artifacts"
        )));
    }
    let mut cases = BTreeSet::new();
    let mut artifact_digests = BTreeSet::new();
    for artifact in artifacts {
        validate_parity_artifact(artifact)?;
        if &artifact.binding != binding {
            return Err(PowerError::InvalidFormat(
                "model parity artifact does not share the bundle's model, runtime, device, and environment binding"
                    .to_string(),
            ));
        }
        if artifact.tested_configuration_sha256 != selected_configuration_sha256 {
            return Err(PowerError::InvalidFormat(
                "model parity artifact does not cover the selected tuning configuration"
                    .to_string(),
            ));
        }
        if !cases.insert(artifact.case_sha256.as_str()) {
            return Err(PowerError::InvalidFormat(
                "hardware evidence contains a duplicate model parity case".to_string(),
            ));
        }
        if !artifact_digests.insert(artifact.artifact_sha256.as_str()) {
            return Err(PowerError::InvalidFormat(
                "hardware evidence contains a duplicate model parity artifact".to_string(),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_parity_preflight(artifacts: &[ModelParityArtifact]) -> Result<()> {
    if artifacts.is_empty() || artifacts.len() > MAX_PARITY_ARTIFACTS {
        return Err(PowerError::InvalidRequest(format!(
            "hardware evidence requires between 1 and {MAX_PARITY_ARTIFACTS} model parity artifacts"
        )));
    }
    for artifact in artifacts {
        validate_parity_artifact(artifact)?;
    }
    Ok(())
}

pub(super) fn validate_tuning_preflight(
    evidence: &super::super::TuningProfileEvidence,
) -> Result<()> {
    if evidence.schema != super::super::TuningProfileEvidence::SCHEMA
        || evidence.candidates.is_empty()
        || evidence.candidates.len() > MAX_CANDIDATES
    {
        return Err(PowerError::InvalidFormat(format!(
            "hardware evidence tuning input must use the supported schema and contain between 1 and {MAX_CANDIDATES} candidates"
        )));
    }
    validate_tuning_binding_strings(&evidence.binding)?;
    validate_sha256(
        &evidence.baseline_configuration_sha256,
        "hardware evidence tuning baseline configuration SHA-256",
    )?;
    for candidate in &evidence.candidates {
        validate_sha256(
            &candidate.configuration_sha256,
            "hardware evidence tuning candidate configuration SHA-256",
        )?;
        if candidate.rounds.is_empty() || candidate.rounds.len() > MAX_ROUNDS_PER_CANDIDATE {
            return Err(PowerError::InvalidFormat(format!(
                "hardware evidence tuning candidate must contain between 1 and {MAX_ROUNDS_PER_CANDIDATE} rounds before policy evaluation"
            )));
        }
        for round in &candidate.rounds {
            for run in [
                &round.baseline_then_candidate.first,
                &round.baseline_then_candidate.second,
                &round.candidate_then_baseline.first,
                &round.candidate_then_baseline.second,
            ] {
                validate_tuning_binding_strings(&run.binding)?;
                validate_sha256(
                    &run.configuration_sha256,
                    "hardware evidence tuning run configuration SHA-256",
                )?;
                validate_execution_digest(&run.output, "hardware evidence tuning run output")?;
            }
        }
    }
    Ok(())
}

pub(super) fn validate_parity_artifact(artifact: &ModelParityArtifact) -> Result<()> {
    if artifact.schema != ModelParityArtifact::SCHEMA {
        return Err(PowerError::InvalidFormat(
            "model parity artifact schema is unsupported".to_string(),
        ));
    }
    validate_binding_shape(&artifact.binding)?;
    for (value, label) in [
        (&artifact.case_sha256, "model parity case SHA-256"),
        (&artifact.artifact_sha256, "model parity artifact SHA-256"),
        (
            &artifact.reference_implementation_sha256,
            "model parity reference implementation SHA-256",
        ),
        (
            &artifact.tested_configuration_sha256,
            "model parity tested configuration SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    validate_execution_digest(&artifact.workload, "model parity workload")?;
    validate_execution_digest(&artifact.reference_output, "model parity reference output")?;
    validate_execution_digest(&artifact.tested_output, "model parity tested output")?;
    if artifact.reference_output != artifact.tested_output {
        return Err(PowerError::InvalidFormat(
            "model parity artifact does not have exact typed output parity".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidFormat(format!(
            "{label} must contain 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_binding_shape(binding: &HardwareEvidenceBinding) -> Result<()> {
    validate_label("Power version", &binding.power_version)?;
    validate_revision(&binding.power_commit, "Power commit")?;
    for (value, label) in [
        (&binding.weights_sha256, "hardware evidence weights SHA-256"),
        (
            &binding.graph_source_sha256,
            "hardware evidence graph/source SHA-256",
        ),
        (&binding.runtime_sha256, "hardware evidence runtime SHA-256"),
        (&binding.device_sha256, "hardware evidence device SHA-256"),
        (
            &binding.environment_sha256,
            "hardware evidence environment SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    binding.runtime_device.validate()
}

fn validate_storage_report(report: &StorageBenchmarkReport) -> Result<()> {
    if report.schema != StorageBenchmarkReport::SCHEMA {
        return Err(PowerError::InvalidFormat(
            "storage report schema is unsupported".to_string(),
        ));
    }
    validate_label("Power version", &report.power_version)?;
    validate_revision(&report.power_commit, "storage report Power commit")?;
    validate_sha256(
        &report.model_collection_sha256,
        "storage report model collection SHA-256",
    )?;
    validate_sha256(&report.sequence_sha256, "storage report sequence SHA-256")?;
    validate_sha256(&report.output_sha256, "storage report output SHA-256")?;
    validate_system(&report.system)?;
    if report.sources.is_empty() || report.sources.len() > MAX_STORAGE_SOURCES {
        return Err(PowerError::InvalidFormat(format!(
            "storage report must contain between 1 and {MAX_STORAGE_SOURCES} sources"
        )));
    }
    if report.samples.is_empty() || report.samples.len() > MAX_BENCHMARK_SAMPLES {
        return Err(PowerError::InvalidFormat(format!(
            "storage report must contain between 1 and {MAX_BENCHMARK_SAMPLES} samples"
        )));
    }
    if report.concurrency == 0
        || report.concurrency > MAX_BENCHMARK_CONCURRENCY
        || report.tensor_count == 0
        || report.requested_bytes_per_sample == 0
        || report.integrity_open_nanos == 0
        || report.output_validation_nanos == 0
    {
        return Err(PowerError::InvalidFormat(
            "storage report violates its bounded measurement shape".to_string(),
        ));
    }
    for (index, source) in report.sources.iter().enumerate() {
        let expected_role = if index == 0 {
            WeightSourceRole::Primary
        } else {
            WeightSourceRole::Replica
        };
        if source.index != index
            || source.role != expected_role
            || source.read_strategy != report.strategy
            || source.configured_read_weight == 0
            || source.effective_read_weight == 0
            || source.verified_files == 0
            || source.verified_tensors == 0
            || source.verified_bytes == 0
        {
            return Err(PowerError::InvalidFormat(
                "storage report contains an invalid source summary".to_string(),
            ));
        }
        source.representation.validate()?;
        if index == 0
            && (source.coverage != WeightSourceCoverage::Complete
                || source.representation != WeightSourceRepresentation::CanonicalSafeTensors)
        {
            return Err(PowerError::InvalidFormat(
                "storage report primary source must be complete canonical SafeTensors".to_string(),
            ));
        }
    }
    let sample_count = u64::try_from(report.samples.len()).map_err(|_| {
        PowerError::InvalidFormat("storage report sample count overflowed".to_string())
    })?;
    let expected_total = report
        .requested_bytes_per_sample
        .checked_mul(sample_count)
        .ok_or_else(|| {
            PowerError::InvalidFormat("storage report requested byte count overflowed".to_string())
        })?;
    let mut actual_total = 0_u64;
    for sample in &report.samples {
        if sample.latency_nanos == 0
            || sample.bytes_read != report.requested_bytes_per_sample
            || sample.source_fallbacks > u64::try_from(report.tensor_count).unwrap_or(u64::MAX)
        {
            return Err(PowerError::InvalidFormat(
                "storage report contains an invalid measurement sample".to_string(),
            ));
        }
        actual_total = actual_total.checked_add(sample.bytes_read).ok_or_else(|| {
            PowerError::InvalidFormat("storage report sample bytes overflowed".to_string())
        })?;
    }
    if report.total_requested_bytes != expected_total
        || report.total_read_bytes != expected_total
        || actual_total != expected_total
    {
        return Err(PowerError::InvalidFormat(
            "storage report aggregate bytes do not match its samples".to_string(),
        ));
    }
    Ok(())
}

fn validate_system(system: &StorageBenchmarkSystem) -> Result<()> {
    for (label, value) in [
        ("operating system", system.os.as_str()),
        ("architecture", system.architecture.as_str()),
        ("CPU model", system.cpu_model.as_str()),
        ("filesystem class", system.filesystem_class.as_str()),
        ("device class", system.device_class.as_str()),
    ] {
        validate_label(label, value)?;
    }
    if system.logical_cpus == 0 || system.ram_bytes == 0 {
        return Err(PowerError::InvalidFormat(
            "hardware evidence system requires positive CPU and RAM capacity".to_string(),
        ));
    }
    Ok(())
}

fn validate_tuning_binding_strings(binding: &TuningProfileBinding) -> Result<()> {
    for (value, label) in [
        (
            &binding.weights_sha256,
            "hardware evidence tuning weights SHA-256",
        ),
        (
            &binding.graph_source_sha256,
            "hardware evidence tuning graph/source SHA-256",
        ),
        (
            &binding.runtime_sha256,
            "hardware evidence tuning runtime SHA-256",
        ),
        (
            &binding.device_sha256,
            "hardware evidence tuning device SHA-256",
        ),
        (
            &binding.environment_sha256,
            "hardware evidence tuning environment SHA-256",
        ),
    ] {
        validate_sha256(value, label)?;
    }
    validate_execution_digest(
        &binding.calibration_workload,
        "hardware evidence calibration workload",
    )
}

fn validate_execution_digest(digest: &ExecutionDigest, label: &str) -> Result<()> {
    validate_sha256(&digest.sha256, label)?;
    if digest.byte_length == 0 || digest.item_count == 0 {
        return Err(PowerError::InvalidFormat(format!(
            "{label} must describe at least one byte and item"
        )));
    }
    Ok(())
}

fn validate_revision(value: &str, label: &str) -> Result<()> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidFormat(format!(
            "{label} must contain 40 or 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_label(label: &str, value: &str) -> Result<()> {
    if value.trim() != value
        || value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(PowerError::InvalidFormat(format!(
            "hardware evidence {label} must be a bounded non-control string without surrounding whitespace"
        )));
    }
    Ok(())
}
