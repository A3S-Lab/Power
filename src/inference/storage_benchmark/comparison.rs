use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::{
    StorageBenchmarkReport, StorageBenchmarkSource, StorageBenchmarkSystem,
    StorageCachePreparation, StorageCacheState, WeightReadStrategy, WeightSourceCoverage,
    WeightSourceRole, WeightSourceWeighting, REPORT_SCHEMA,
};

const COMPARISON_SCHEMA: &str = "a3s.power.storage-benchmark-comparison.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageDistributionSummary {
    pub count: usize,
    pub minimum: u64,
    pub p50: u64,
    pub p95: u64,
    pub maximum: u64,
    pub mean: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkSourceSummary {
    pub index: usize,
    pub role: WeightSourceRole,
    pub coverage: WeightSourceCoverage,
    pub read_strategy: WeightReadStrategy,
    pub configured_read_weight: u32,
    pub effective_read_weight: u32,
    pub source_weighting: WeightSourceWeighting,
    pub validation_bytes_per_second: StorageDistributionSummary,
    pub io_block_size: u64,
    pub verified_files: usize,
    pub verified_tensors: usize,
    pub verified_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkGroup {
    pub strategy: WeightReadStrategy,
    pub cache_state: StorageCacheState,
    pub cache_preparation: StorageCachePreparation,
    pub cache_state_procedures: Vec<String>,
    pub cache_state_verified: bool,
    pub concurrency: usize,
    pub sources: Vec<StorageBenchmarkSourceSummary>,
    pub source_profile_sha256: String,
    pub report_count: usize,
    pub sample_count: usize,
    pub total_requested_bytes: u64,
    pub total_read_bytes: u64,
    pub integrity_open_nanos: StorageDistributionSummary,
    pub latency_nanos: StorageDistributionSummary,
    pub bytes_per_second: StorageDistributionSummary,
    pub output_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkComparison {
    pub schema: String,
    pub power_version: String,
    pub power_commit: String,
    pub model_collection_sha256: String,
    pub system: StorageBenchmarkSystem,
    pub sequence_sha256: String,
    pub tensor_count: usize,
    pub requested_bytes_per_sample: u64,
    pub output_byte_parity: bool,
    pub output_sha256s: Vec<String>,
    pub groups: Vec<StorageBenchmarkGroup>,
}

/// Combines separately recorded cold/warm, strategy, and source-count runs.
/// Reports must share one exact model, deterministic sequence, Power revision,
/// and named hardware environment. A mismatch is retained as explicit output
/// parity evidence instead of being silently discarded.
pub fn compare_storage_benchmarks(
    reports: &[StorageBenchmarkReport],
) -> Result<StorageBenchmarkComparison> {
    if reports.len() < 2 {
        return Err(PowerError::InvalidRequest(
            "storage benchmark comparison requires at least two reports".to_string(),
        ));
    }
    for report in reports {
        validate_report(report)?;
    }
    let identity = &reports[0];
    for report in &reports[1..] {
        if report.power_version != identity.power_version
            || report.power_commit != identity.power_commit
            || report.model_collection_sha256 != identity.model_collection_sha256
            || report.system != identity.system
            || report.sequence_sha256 != identity.sequence_sha256
            || report.tensor_count != identity.tensor_count
            || report.requested_bytes_per_sample != identity.requested_bytes_per_sample
        {
            return Err(PowerError::InvalidRequest(
                "storage benchmark reports do not share one revision, model, sequence, and named hardware environment"
                    .to_string(),
            ));
        }
    }

    let mut groups = BTreeMap::<GroupKey, GroupAccumulator>::new();
    let mut all_output_sha256s = BTreeSet::new();
    for report in reports {
        let source_profile_sha256 = source_profile_digest(&report.sources)?;
        let key = GroupKey {
            strategy: report.strategy,
            cache_state: report.cache_state,
            cache_preparation: report.cache_preparation,
            concurrency: report.concurrency,
            source_profile_sha256: source_profile_sha256.clone(),
        };
        let group = groups.entry(key).or_insert_with(|| GroupAccumulator {
            sources: report.sources.clone(),
            source_validation_rates: vec![Vec::new(); report.sources.len()],
            procedures: BTreeSet::new(),
            cache_state_verified: true,
            report_count: 0,
            sample_count: 0,
            total_requested_bytes: 0,
            total_read_bytes: 0,
            integrity_open_nanos: Vec::new(),
            latency_nanos: Vec::new(),
            bytes_per_second: Vec::new(),
            output_sha256s: BTreeSet::new(),
        });
        if !same_source_profiles(&group.sources, &report.sources) {
            return Err(PowerError::InvalidRequest(
                "storage benchmark source profile digest collision".to_string(),
            ));
        }
        for (rates, source) in group
            .source_validation_rates
            .iter_mut()
            .zip(&report.sources)
        {
            rates.push(source.validation_bytes_per_second);
        }
        group
            .procedures
            .insert(report.cache_state_procedure.clone());
        group.cache_state_verified &= report.cache_state_verified;
        group.report_count = group.report_count.saturating_add(1);
        group.sample_count = group.sample_count.saturating_add(report.samples.len());
        group.total_requested_bytes = group
            .total_requested_bytes
            .checked_add(report.total_requested_bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat("comparison requested byte count overflowed".to_string())
            })?;
        group.total_read_bytes = group
            .total_read_bytes
            .checked_add(report.total_read_bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat("comparison read byte count overflowed".to_string())
            })?;
        group.integrity_open_nanos.push(report.integrity_open_nanos);
        group
            .latency_nanos
            .extend(report.samples.iter().map(|sample| sample.latency_nanos));
        group
            .bytes_per_second
            .extend(report.samples.iter().map(|sample| sample.bytes_per_second));
        group.output_sha256s.insert(report.output_sha256.clone());
        all_output_sha256s.insert(report.output_sha256.clone());
    }

    let groups = groups
        .into_iter()
        .map(|(key, group)| {
            Ok(StorageBenchmarkGroup {
                strategy: key.strategy,
                cache_state: key.cache_state,
                cache_preparation: key.cache_preparation,
                cache_state_procedures: group.procedures.into_iter().collect(),
                cache_state_verified: group.cache_state_verified,
                concurrency: key.concurrency,
                sources: summarize_sources(&group.sources, &group.source_validation_rates)?,
                source_profile_sha256: key.source_profile_sha256,
                report_count: group.report_count,
                sample_count: group.sample_count,
                total_requested_bytes: group.total_requested_bytes,
                total_read_bytes: group.total_read_bytes,
                integrity_open_nanos: distribution(&group.integrity_open_nanos)?,
                latency_nanos: distribution(&group.latency_nanos)?,
                bytes_per_second: distribution(&group.bytes_per_second)?,
                output_sha256s: group.output_sha256s.into_iter().collect(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let output_sha256s = all_output_sha256s.into_iter().collect::<Vec<_>>();
    Ok(StorageBenchmarkComparison {
        schema: COMPARISON_SCHEMA.to_string(),
        power_version: identity.power_version.clone(),
        power_commit: identity.power_commit.clone(),
        model_collection_sha256: identity.model_collection_sha256.clone(),
        system: identity.system.clone(),
        sequence_sha256: identity.sequence_sha256.clone(),
        tensor_count: identity.tensor_count,
        requested_bytes_per_sample: identity.requested_bytes_per_sample,
        output_byte_parity: output_sha256s.len() == 1,
        output_sha256s,
        groups,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct GroupKey {
    strategy: WeightReadStrategy,
    cache_state: StorageCacheState,
    cache_preparation: StorageCachePreparation,
    concurrency: usize,
    source_profile_sha256: String,
}

struct GroupAccumulator {
    sources: Vec<StorageBenchmarkSource>,
    source_validation_rates: Vec<Vec<u64>>,
    procedures: BTreeSet<String>,
    cache_state_verified: bool,
    report_count: usize,
    sample_count: usize,
    total_requested_bytes: u64,
    total_read_bytes: u64,
    integrity_open_nanos: Vec<u64>,
    latency_nanos: Vec<u64>,
    bytes_per_second: Vec<u64>,
    output_sha256s: BTreeSet<String>,
}

fn validate_report(report: &StorageBenchmarkReport) -> Result<()> {
    if report.schema != REPORT_SCHEMA
        || report.sources.is_empty()
        || report.samples.is_empty()
        || report.total_requested_bytes != report.total_read_bytes
        || report.output_sha256.len() != 64
        || !report
            .output_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
        || report
            .sources
            .iter()
            .any(|source| source.read_strategy != report.strategy)
    {
        return Err(PowerError::InvalidFormat(
            "storage benchmark report violates its canonical evidence contract".to_string(),
        ));
    }
    let cache_contract_is_valid = match (report.cache_state, report.cache_preparation) {
        (StorageCacheState::Warm, StorageCachePreparation::WarmSequence) => true,
        (StorageCacheState::Cold, StorageCachePreparation::LinuxFadviseDontNeed) => {
            report.samples.len() == 1
        }
        _ => false,
    };
    if !cache_contract_is_valid
        || !report.cache_state_verified
        || report.cache_state_procedure != report.cache_preparation.procedure()
    {
        return Err(PowerError::InvalidFormat(
            "storage benchmark report lacks runner-verified cache-state evidence".to_string(),
        ));
    }
    let sample_bytes = report.samples.iter().try_fold(0_u64, |total, sample| {
        total.checked_add(sample.bytes_read).ok_or_else(|| {
            PowerError::InvalidFormat("benchmark sample byte count overflowed".to_string())
        })
    })?;
    if sample_bytes != report.total_read_bytes {
        return Err(PowerError::InvalidFormat(
            "storage benchmark sample bytes do not match the report total".to_string(),
        ));
    }
    Ok(())
}

fn source_profile_digest(sources: &[StorageBenchmarkSource]) -> Result<String> {
    let canonical = serde_json::to_vec(&stable_source_profiles(sources))?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

fn stable_source_profiles(sources: &[StorageBenchmarkSource]) -> Vec<StorageBenchmarkSource> {
    sources
        .iter()
        .cloned()
        .map(|mut source| {
            source.validation_bytes_per_second = 0;
            source
        })
        .collect()
}

fn same_source_profiles(
    first: &[StorageBenchmarkSource],
    second: &[StorageBenchmarkSource],
) -> bool {
    stable_source_profiles(first) == stable_source_profiles(second)
}

fn summarize_sources(
    sources: &[StorageBenchmarkSource],
    validation_rates: &[Vec<u64>],
) -> Result<Vec<StorageBenchmarkSourceSummary>> {
    if sources.len() != validation_rates.len() {
        return Err(PowerError::InvalidFormat(
            "storage benchmark source validation evidence is inconsistent".to_string(),
        ));
    }
    sources
        .iter()
        .zip(validation_rates)
        .map(|(source, rates)| {
            Ok(StorageBenchmarkSourceSummary {
                index: source.index,
                role: source.role,
                coverage: source.coverage,
                read_strategy: source.read_strategy,
                configured_read_weight: source.configured_read_weight,
                effective_read_weight: source.effective_read_weight,
                source_weighting: source.source_weighting,
                validation_bytes_per_second: distribution(rates)?,
                io_block_size: source.io_block_size,
                verified_files: source.verified_files,
                verified_tensors: source.verified_tensors,
                verified_bytes: source.verified_bytes,
            })
        })
        .collect()
}

fn distribution(values: &[u64]) -> Result<StorageDistributionSummary> {
    if values.is_empty() {
        return Err(PowerError::InvalidFormat(
            "storage benchmark distribution is empty".to_string(),
        ));
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let sum = sorted.iter().fold(0_u128, |total, value| {
        total.saturating_add(u128::from(*value))
    });
    let mean = sum / u128::try_from(sorted.len()).unwrap_or(1);
    Ok(StorageDistributionSummary {
        count: sorted.len(),
        minimum: sorted[0],
        p50: percentile(&sorted, 50),
        p95: percentile(&sorted, 95),
        maximum: sorted[sorted.len() - 1],
        mean: u64::try_from(mean).unwrap_or(u64::MAX),
    })
}

fn percentile(sorted: &[u64], percent: usize) -> u64 {
    let rank = sorted.len().saturating_mul(percent).saturating_add(99) / 100;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{
        StorageBenchmarkSample, WeightSourceCoverage, WeightSourceRole, WeightSourceWeighting,
    };

    fn report(strategy: WeightReadStrategy, state: StorageCacheState) -> StorageBenchmarkReport {
        StorageBenchmarkReport {
            schema: REPORT_SCHEMA.to_string(),
            power_version: "0.7.0".to_string(),
            power_commit: "0123456789abcdef0123456789abcdef01234567".to_string(),
            model_collection_sha256: "1".repeat(64),
            sources: vec![StorageBenchmarkSource {
                index: 0,
                role: WeightSourceRole::Primary,
                coverage: WeightSourceCoverage::Complete,
                read_strategy: strategy,
                configured_read_weight: 1,
                effective_read_weight: 1,
                source_weighting: WeightSourceWeighting::Configured,
                validation_bytes_per_second: 1,
                io_block_size: 4096,
                verified_files: 1,
                verified_tensors: 1,
                verified_bytes: 4,
            }],
            system: StorageBenchmarkSystem {
                os: "test".to_string(),
                architecture: "test".to_string(),
                cpu_model: "test".to_string(),
                logical_cpus: 1,
                ram_bytes: 1,
                filesystem_class: "test".to_string(),
                device_class: "test".to_string(),
            },
            cache_state: state,
            cache_preparation: match state {
                StorageCacheState::Cold => StorageCachePreparation::LinuxFadviseDontNeed,
                StorageCacheState::Warm => StorageCachePreparation::WarmSequence,
            },
            cache_state_procedure: match state {
                StorageCacheState::Cold => StorageCachePreparation::LinuxFadviseDontNeed,
                StorageCacheState::Warm => StorageCachePreparation::WarmSequence,
            }
            .procedure()
            .to_string(),
            cache_state_verified: true,
            strategy,
            concurrency: 1,
            sequence_sha256: "2".repeat(64),
            tensor_count: 1,
            requested_bytes_per_sample: 4,
            total_requested_bytes: 4,
            total_read_bytes: 4,
            integrity_open_nanos: 10,
            samples: vec![StorageBenchmarkSample {
                latency_nanos: 20,
                bytes_read: 4,
                bytes_per_second: 200,
                source_fallbacks: 0,
            }],
            output_sha256: "3".repeat(64),
        }
    }

    #[test]
    fn comparison_groups_cold_and_warm_strategies_with_parity() {
        let reports = [
            report(WeightReadStrategy::Mmap, StorageCacheState::Cold),
            report(
                WeightReadStrategy::PositionalBuffered,
                StorageCacheState::Warm,
            ),
        ];
        let comparison = compare_storage_benchmarks(&reports).unwrap();
        assert!(comparison.output_byte_parity);
        assert_eq!(comparison.groups.len(), 2);
        assert_eq!(comparison.groups[0].latency_nanos.count, 1);
    }

    #[test]
    fn comparison_records_output_mismatch_instead_of_hiding_it() {
        let first = report(WeightReadStrategy::Mmap, StorageCacheState::Warm);
        let mut second = report(
            WeightReadStrategy::PositionalBuffered,
            StorageCacheState::Warm,
        );
        second.output_sha256 = "4".repeat(64);
        let comparison = compare_storage_benchmarks(&[first, second]).unwrap();
        assert!(!comparison.output_byte_parity);
        assert_eq!(comparison.output_sha256s.len(), 2);
    }

    #[test]
    fn comparison_aggregates_cold_processes_with_different_validation_rates() {
        let first = report(WeightReadStrategy::Mmap, StorageCacheState::Cold);
        let mut second = first.clone();
        second.sources[0].validation_bytes_per_second = 9;
        let comparison = compare_storage_benchmarks(&[first, second]).unwrap();

        assert_eq!(comparison.groups.len(), 1);
        assert_eq!(comparison.groups[0].report_count, 2);
        assert_eq!(
            comparison.groups[0].sources[0]
                .validation_bytes_per_second
                .count,
            2
        );
    }
}
