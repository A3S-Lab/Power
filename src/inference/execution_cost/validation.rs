use crate::error::{PowerError, Result};

use super::super::{HardwareEvidenceBinding, RuntimeDeviceKind};
use super::{
    TensorBatchBenchmarkConfig, TensorBatchBenchmarkMode, TensorBatchBenchmarkReport,
    TensorBatchBenchmarkSample, TensorBatchBenchmarkSummary,
};

pub(super) const MAX_BENCHMARK_ITEMS: usize = 1_024;
pub(super) const MAX_MEASURED_ROUNDS: usize = 1_000;
pub(super) const MAX_WARMUP_ROUNDS: usize = 100;

pub(super) fn validate_config(config: &TensorBatchBenchmarkConfig) -> Result<()> {
    if config.warmup_rounds > MAX_WARMUP_ROUNDS
        || config.measured_rounds == 0
        || config.measured_rounds > MAX_MEASURED_ROUNDS
    {
        return Err(PowerError::InvalidRequest(format!(
            "tensor batch benchmark warmups must be at most {MAX_WARMUP_ROUNDS} and measured rounds must be between 1 and {MAX_MEASURED_ROUNDS}"
        )));
    }
    Ok(())
}

pub(super) fn verify_report(report: &TensorBatchBenchmarkReport) -> Result<()> {
    if report.schema != TensorBatchBenchmarkReport::SCHEMA
        || report.item_count < 2
        || report.item_count > MAX_BENCHMARK_ITEMS
        || report.measured_rounds == 0
        || report.measured_rounds > MAX_MEASURED_ROUNDS
        || report.warmup_rounds > MAX_WARMUP_ROUNDS
        || !report.exact_output_parity
    {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark report shape is invalid".to_string(),
        ));
    }
    validate_sha256(
        &report.input_sequence_sha256,
        "tensor batch benchmark input sequence SHA-256",
    )?;
    validate_sha256(
        &report.output_sha256,
        "tensor batch benchmark output SHA-256",
    )?;
    validate_sha256(&report.sha256, "tensor batch benchmark report SHA-256")?;

    let expected_binding = HardwareEvidenceBinding::new(
        report.binding.power_version.clone(),
        report.binding.power_commit.clone(),
        report.binding.weights_sha256.clone(),
        report.binding.graph_source_sha256.clone(),
        report.binding.runtime_device,
        &report.system,
    )?;
    if expected_binding != report.binding {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark binding does not match its named hardware system".to_string(),
        ));
    }

    let expected_samples = report.measured_rounds.checked_mul(2).ok_or_else(|| {
        PowerError::InvalidFormat("benchmark sample count overflowed".to_string())
    })?;
    if report.samples.len() != expected_samples {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark does not contain two samples per measured round".to_string(),
        ));
    }
    for (round, pair) in report.samples.chunks_exact(2).enumerate() {
        validate_pair(report, round, pair)?;
    }
    let expected_summaries = summaries(&report.samples)?;
    if report.summaries != expected_summaries {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark summaries do not match their raw samples".to_string(),
        ));
    }
    let actual = super::digest::report_sha256(report)?;
    if actual != report.sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "tensor batch benchmark report".to_string(),
            expected: report.sha256.clone(),
            actual,
        });
    }
    Ok(())
}

fn validate_pair(
    report: &TensorBatchBenchmarkReport,
    round: usize,
    pair: &[TensorBatchBenchmarkSample],
) -> Result<()> {
    let order = super::TensorBatchBenchmarkOrder::for_round(round);
    let modes = order.modes();
    for (sample, expected_mode) in pair.iter().zip(modes) {
        if sample.round != round
            || sample.order != order
            || sample.mode != expected_mode
            || sample.item_count != report.item_count
            || sample.output_sha256 != report.output_sha256
        {
            return Err(PowerError::InvalidFormat(
                "tensor batch benchmark samples are not canonically ordered or bound".to_string(),
            ));
        }
        validate_sample_shape(report, sample)?;
    }
    let individual = pair
        .iter()
        .find(|sample| sample.mode == TensorBatchBenchmarkMode::Individual)
        .ok_or_else(|| PowerError::InvalidFormat("individual sample is missing".to_string()))?;
    let batch = pair
        .iter()
        .find(|sample| sample.mode == TensorBatchBenchmarkMode::LeadingBatch)
        .ok_or_else(|| PowerError::InvalidFormat("batch sample is missing".to_string()))?;
    if individual.boundary.input_host_bytes != batch.boundary.input_host_bytes
        || individual.boundary.output_host_bytes != batch.boundary.output_host_bytes
    {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark modes do not cover the same host boundary bytes".to_string(),
        ));
    }
    Ok(())
}

fn validate_sample_shape(
    report: &TensorBatchBenchmarkReport,
    sample: &TensorBatchBenchmarkSample,
) -> Result<()> {
    validate_sha256(
        &sample.output_sha256,
        "tensor batch benchmark sample output SHA-256",
    )?;
    let execution_count = match sample.mode {
        TensorBatchBenchmarkMode::Individual => report.item_count,
        TensorBatchBenchmarkMode::LeadingBatch => 1,
    };
    if sample.execution_count != execution_count
        || sample.boundary.input_materializations != execution_count as u64
        || sample.boundary.output_materializations != execution_count as u64
        || sample.boundary.input_host_bytes == 0
        || sample.boundary.output_host_bytes == 0
    {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark sample execution boundary is inconsistent".to_string(),
        ));
    }
    let expected_device_copies = match report.binding.runtime_device.kind {
        RuntimeDeviceKind::Cpu => 0,
        RuntimeDeviceKind::Cuda | RuntimeDeviceKind::Metal => execution_count as u64,
    };
    if sample.boundary.host_to_device_copy_operations != expected_device_copies
        || sample.boundary.device_to_host_copy_operations != expected_device_copies
    {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark sample device copy count is inconsistent".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn summaries(
    samples: &[TensorBatchBenchmarkSample],
) -> Result<Vec<TensorBatchBenchmarkSummary>> {
    [
        TensorBatchBenchmarkMode::Individual,
        TensorBatchBenchmarkMode::LeadingBatch,
    ]
    .into_iter()
    .map(|mode| summary(samples, mode))
    .collect()
}

fn summary(
    samples: &[TensorBatchBenchmarkSample],
    mode: TensorBatchBenchmarkMode,
) -> Result<TensorBatchBenchmarkSummary> {
    let matching = samples
        .iter()
        .filter(|sample| sample.mode == mode)
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(PowerError::InvalidFormat(
            "tensor batch benchmark summary has no source samples".to_string(),
        ));
    }
    Ok(TensorBatchBenchmarkSummary {
        mode,
        sample_count: matching.len(),
        median_elapsed_nanos: median(matching.iter().map(|sample| sample.elapsed_nanos)),
        median_host_allocation_count: median(
            matching
                .iter()
                .map(|sample| sample.host_allocations.allocation_count),
        ),
        median_host_allocated_bytes: median(
            matching
                .iter()
                .map(|sample| sample.host_allocations.allocated_bytes),
        ),
        median_host_reallocation_count: median(
            matching
                .iter()
                .map(|sample| sample.host_allocations.reallocation_count),
        ),
        median_host_reallocated_bytes: median(
            matching
                .iter()
                .map(|sample| sample.host_allocations.reallocated_bytes),
        ),
        median_input_materialization_nanos: median(
            matching
                .iter()
                .map(|sample| sample.boundary.input_materialization_nanos),
        ),
        median_output_materialization_nanos: median(
            matching
                .iter()
                .map(|sample| sample.boundary.output_materialization_nanos),
        ),
    })
}

fn median(values: impl Iterator<Item = u64>) -> u64 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[(values.len() - 1) / 2]
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
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
