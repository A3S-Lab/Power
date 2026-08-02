use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::{
    InferenceLimits, WeightReadStrategy, WeightSourceCoverage, WeightSourceRepresentation,
    WeightSourceRole, WeightSourceWeighting, WeightStore, WeightStoreConfig,
};

mod cache;
mod comparison;

use cache::prepare_cache_state;

pub use comparison::{
    compare_storage_benchmarks, StorageBenchmarkComparison, StorageBenchmarkGroup,
    StorageBenchmarkSourceSummary, StorageDistributionSummary,
};

const REPORT_SCHEMA: &str = "a3s.power.storage-benchmark.v1";
const MAX_BENCHMARK_SAMPLES: usize = 1_000;
const MAX_BENCHMARK_CONCURRENCY: usize = 256;
const MAX_LABEL_BYTES: usize = 512;
const WARM_SEQUENCE_PROCEDURE: &str =
    "one complete unmeasured tensor sequence immediately before measurement";
const LINUX_COLD_PROCEDURE: &str =
    "fsync and POSIX_FADV_DONTNEED after integrity-open, followed by mincore verification of every requested file-backed page";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageCacheState {
    Cold,
    Warm,
}

/// Runner-controlled page-cache preparation performed after the mandatory
/// integrity-open pass and before the measured storage reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StorageCachePreparation {
    /// Read the complete deterministic sequence once outside measurement.
    WarmSequence,
    /// Linux-only: discard clean file pages, then prove that every page backing
    /// the requested tensor ranges is non-resident with `mincore`.
    LinuxFadviseDontNeed,
}

impl StorageCachePreparation {
    fn procedure(self) -> &'static str {
        match self {
            Self::WarmSequence => WARM_SEQUENCE_PROCEDURE,
            Self::LinuxFadviseDontNeed => LINUX_COLD_PROCEDURE,
        }
    }
}

/// Explicit configuration for one inference-independent storage benchmark.
///
/// The model roots are intentionally absent from [`StorageBenchmarkReport`].
/// This config has a redacted debug implementation so an embedding host cannot
/// accidentally log private model paths through normal diagnostics.
#[derive(Clone)]
pub struct StorageBenchmarkConfig {
    pub weights: WeightStoreConfig,
    pub power_commit: String,
    pub filesystem_class: String,
    pub device_class: String,
    pub cpu_model: String,
    pub ram_bytes: u64,
    pub cache_state: StorageCacheState,
    pub cache_preparation: StorageCachePreparation,
    pub concurrency: usize,
    pub samples: usize,
    pub max_tensors: usize,
}

impl std::fmt::Debug for StorageBenchmarkConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageBenchmarkConfig")
            .field(
                "source_count",
                &self.weights.replicas.len().saturating_add(1),
            )
            .field("power_commit", &self.power_commit)
            .field("cache_state", &self.cache_state)
            .field("cache_preparation", &self.cache_preparation)
            .field("concurrency", &self.concurrency)
            .field("samples", &self.samples)
            .field("max_tensors", &self.max_tensors)
            .finish_non_exhaustive()
    }
}

impl StorageBenchmarkConfig {
    pub fn validate(&self) -> Result<WeightReadStrategy> {
        for (label, value) in [
            ("power commit", self.power_commit.as_str()),
            ("filesystem class", self.filesystem_class.as_str()),
            ("device class", self.device_class.as_str()),
            ("CPU model", self.cpu_model.as_str()),
        ] {
            validate_label(label, value)?;
        }
        if !matches!(self.power_commit.len(), 40 | 64)
            || !self
                .power_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(PowerError::InvalidRequest(
                "storage benchmark power commit must be a lowercase 40- or 64-character hexadecimal revision"
                    .to_string(),
            ));
        }
        if self.ram_bytes == 0
            || self.concurrency == 0
            || self.concurrency > MAX_BENCHMARK_CONCURRENCY
            || self.samples == 0
            || self.samples > MAX_BENCHMARK_SAMPLES
            || self.max_tensors == 0
        {
            return Err(PowerError::InvalidRequest(format!(
                "storage benchmark RAM, concurrency (1..={MAX_BENCHMARK_CONCURRENCY}), samples (1..={MAX_BENCHMARK_SAMPLES}), and tensor bounds must be valid"
            )));
        }
        match (self.cache_state, self.cache_preparation) {
            (StorageCacheState::Warm, StorageCachePreparation::WarmSequence) => {}
            (StorageCacheState::Cold, StorageCachePreparation::LinuxFadviseDontNeed) => {
                if !cfg!(target_os = "linux") {
                    return Err(PowerError::BackendNotAvailable(
                        "verified cold page-cache preparation is currently supported only on Linux"
                            .to_string(),
                    ));
                }
                if self.samples != 1 {
                    return Err(PowerError::InvalidRequest(
                        "one process may record only one cold sample; start a new process for each additional sample"
                            .to_string(),
                    ));
                }
            }
            _ => {
                return Err(PowerError::InvalidRequest(
                    "storage benchmark cache state and runner-controlled preparation do not match"
                        .to_string(),
                ));
            }
        }
        let strategy = self.weights.primary.read_strategy;
        if self
            .weights
            .replicas
            .iter()
            .any(|source| source.read_strategy != strategy)
        {
            return Err(PowerError::InvalidRequest(
                "one storage benchmark run must use the same read strategy for every source"
                    .to_string(),
            ));
        }
        Ok(strategy)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkSource {
    pub index: usize,
    pub role: WeightSourceRole,
    pub coverage: WeightSourceCoverage,
    pub read_strategy: WeightReadStrategy,
    #[serde(default)]
    pub representation: WeightSourceRepresentation,
    pub configured_read_weight: u32,
    pub effective_read_weight: u32,
    pub source_weighting: WeightSourceWeighting,
    pub validation_bytes_per_second: u64,
    pub io_block_size: u64,
    pub verified_files: usize,
    pub verified_tensors: usize,
    pub verified_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkSystem {
    pub os: String,
    pub architecture: String,
    pub cpu_model: String,
    pub logical_cpus: usize,
    pub ram_bytes: u64,
    pub filesystem_class: String,
    pub device_class: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkSample {
    pub latency_nanos: u64,
    pub bytes_read: u64,
    pub bytes_per_second: u64,
    pub source_fallbacks: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StorageBenchmarkReport {
    pub schema: String,
    pub power_version: String,
    pub power_commit: String,
    pub model_collection_sha256: String,
    pub sources: Vec<StorageBenchmarkSource>,
    pub system: StorageBenchmarkSystem,
    pub cache_state: StorageCacheState,
    pub cache_preparation: StorageCachePreparation,
    pub cache_state_procedure: String,
    pub cache_state_verified: bool,
    pub strategy: WeightReadStrategy,
    pub concurrency: usize,
    pub sequence_sha256: String,
    pub tensor_count: usize,
    pub requested_bytes_per_sample: u64,
    pub total_requested_bytes: u64,
    pub total_read_bytes: u64,
    pub integrity_open_nanos: u64,
    pub output_validation_nanos: u64,
    pub samples: Vec<StorageBenchmarkSample>,
    pub output_sha256: String,
}

/// Runs a storage-only benchmark. No graph, model architecture, tokenizer,
/// network listener, subprocess, or inference backend is created.
pub fn run_storage_benchmark(
    config: &StorageBenchmarkConfig,
    limits: &InferenceLimits,
) -> Result<StorageBenchmarkReport> {
    let strategy = config.validate()?;
    limits.validate()?;
    let opened = Instant::now();
    let store = Arc::new(WeightStore::open_config(&config.weights, limits)?);
    let integrity_open_nanos = duration_nanos(opened.elapsed());
    let names = store
        .inventory()
        .take(config.max_tensors)
        .map(|descriptor| descriptor.name.clone())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(PowerError::InvalidFormat(
            "storage benchmark selected no verified tensors".to_string(),
        ));
    }
    let requested_bytes_per_sample = names.iter().try_fold(0_u64, |total, name| {
        let bytes = store
            .descriptor(name)
            .map(|descriptor| descriptor.bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "storage benchmark sequence lost a verified tensor".to_string(),
                )
            })?;
        total.checked_add(bytes).ok_or_else(|| {
            PowerError::InvalidFormat(
                "storage benchmark requested byte count overflowed".to_string(),
            )
        })
    })?;
    let sequence_sha256 = sequence_digest(&store, &names)?;
    let validation_started = Instant::now();
    let output_sha256 = run_output_parity(Arc::clone(&store), &names, config.concurrency)?;
    let mut output_validation_nanos = duration_nanos(validation_started.elapsed());
    let cache_state_procedure = prepare_cache_state(config, &store, &names)?;

    let mut samples = Vec::with_capacity(config.samples);
    let mut total_read_bytes = 0_u64;
    for _ in 0..config.samples {
        let sample = run_sample(Arc::clone(&store), &names, config.concurrency)?;
        total_read_bytes = total_read_bytes
            .checked_add(sample.bytes_read)
            .ok_or_else(|| {
                PowerError::InvalidFormat(
                    "storage benchmark total read byte count overflowed".to_string(),
                )
            })?;
        samples.push(StorageBenchmarkSample {
            latency_nanos: sample.latency_nanos,
            bytes_read: sample.bytes_read,
            bytes_per_second: sample.bytes_per_second,
            source_fallbacks: sample.source_fallbacks,
        });
    }
    let total_requested_bytes = requested_bytes_per_sample
        .checked_mul(u64::try_from(config.samples).map_err(|_| {
            PowerError::InvalidFormat("storage benchmark sample count overflowed".to_string())
        })?)
        .ok_or_else(|| {
            PowerError::InvalidFormat(
                "storage benchmark total requested byte count overflowed".to_string(),
            )
        })?;
    if total_read_bytes != total_requested_bytes {
        return Err(PowerError::InvalidFormat(format!(
            "storage benchmark read {total_read_bytes} bytes but requested {total_requested_bytes}"
        )));
    }
    let validation_started = Instant::now();
    let final_output_sha256 = run_output_parity(Arc::clone(&store), &names, config.concurrency)?;
    output_validation_nanos = output_validation_nanos
        .checked_add(duration_nanos(validation_started.elapsed()))
        .ok_or_else(|| {
            PowerError::InvalidFormat(
                "storage benchmark output-validation duration overflowed".to_string(),
            )
        })?;
    if final_output_sha256 != output_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "storage benchmark output".to_string(),
            expected: output_sha256,
            actual: final_output_sha256,
        });
    }

    Ok(StorageBenchmarkReport {
        schema: REPORT_SCHEMA.to_string(),
        power_version: env!("CARGO_PKG_VERSION").to_string(),
        power_commit: config.power_commit.clone(),
        model_collection_sha256: store.sha256().to_string(),
        sources: store
            .sources()
            .into_iter()
            .map(|source| StorageBenchmarkSource {
                index: source.index,
                role: source.role,
                coverage: source.coverage,
                read_strategy: source.read_strategy,
                representation: source.representation,
                configured_read_weight: source.configured_read_weight,
                effective_read_weight: source.read_weight,
                source_weighting: source.source_weighting,
                validation_bytes_per_second: source.validation_bytes_per_second,
                io_block_size: source.io_block_size,
                verified_files: source.verified_files,
                verified_tensors: source.verified_tensors,
                verified_bytes: source.verified_bytes,
            })
            .collect(),
        system: StorageBenchmarkSystem {
            os: std::env::consts::OS.to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            cpu_model: config.cpu_model.clone(),
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            ram_bytes: config.ram_bytes,
            filesystem_class: config.filesystem_class.clone(),
            device_class: config.device_class.clone(),
        },
        cache_state: config.cache_state,
        cache_preparation: config.cache_preparation,
        cache_state_procedure: cache_state_procedure.to_string(),
        cache_state_verified: true,
        strategy,
        concurrency: config.concurrency,
        sequence_sha256,
        tensor_count: names.len(),
        requested_bytes_per_sample,
        total_requested_bytes,
        total_read_bytes,
        integrity_open_nanos,
        output_validation_nanos,
        samples,
        output_sha256: final_output_sha256,
    })
}

struct CompletedSample {
    latency_nanos: u64,
    bytes_read: u64,
    bytes_per_second: u64,
    source_fallbacks: u64,
}

struct CompletedRead {
    index: usize,
    bytes: u64,
    fell_back: bool,
}

struct ParityRead {
    index: usize,
    bytes: u64,
    digest: [u8; 32],
}

fn run_sample(
    store: Arc<WeightStore>,
    names: &[String],
    concurrency: usize,
) -> Result<CompletedSample> {
    let started = Instant::now();
    let worker_count = concurrency.min(names.len());
    let completed = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let store = Arc::clone(&store);
            workers.push(scope.spawn(move || -> Result<Vec<CompletedRead>> {
                let mut reads = Vec::new();
                for index in (worker..names.len()).step_by(worker_count) {
                    let read = store.read_tensor_bytes(&names[index])?;
                    reads.push(CompletedRead {
                        index,
                        bytes: u64::try_from(read.bytes().len()).map_err(|_| {
                            PowerError::InvalidFormat(
                                "benchmark tensor byte length overflowed".to_string(),
                            )
                        })?,
                        fell_back: read.fell_back(),
                    });
                }
                Ok(reads)
            }));
        }
        let mut reads = Vec::with_capacity(names.len());
        for worker in workers {
            let mut worker_reads = worker.join().map_err(|_| {
                PowerError::InferenceFailed("storage benchmark worker panicked".to_string())
            })??;
            reads.append(&mut worker_reads);
        }
        Ok::<_, PowerError>(reads)
    })?;
    let elapsed = started.elapsed();
    let mut completed = completed;
    completed.sort_by_key(|read| read.index);
    if completed.len() != names.len()
        || completed
            .iter()
            .enumerate()
            .any(|(index, read)| index != read.index)
    {
        return Err(PowerError::InferenceFailed(
            "storage benchmark did not complete its deterministic tensor sequence".to_string(),
        ));
    }
    let mut bytes_read = 0_u64;
    let mut source_fallbacks = 0_u64;
    for read in completed {
        bytes_read = bytes_read.checked_add(read.bytes).ok_or_else(|| {
            PowerError::InvalidFormat("storage benchmark read byte count overflowed".to_string())
        })?;
        source_fallbacks = source_fallbacks.saturating_add(u64::from(read.fell_back));
    }
    Ok(CompletedSample {
        latency_nanos: duration_nanos(elapsed),
        bytes_read,
        bytes_per_second: throughput(bytes_read, elapsed),
        source_fallbacks,
    })
}

fn run_output_parity(
    store: Arc<WeightStore>,
    names: &[String],
    concurrency: usize,
) -> Result<String> {
    let worker_count = concurrency.min(names.len());
    let completed = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(worker_count);
        for worker in 0..worker_count {
            let store = Arc::clone(&store);
            workers.push(scope.spawn(move || -> Result<Vec<ParityRead>> {
                let mut reads = Vec::new();
                for index in (worker..names.len()).step_by(worker_count) {
                    let read = store.read_tensor_bytes(&names[index])?;
                    reads.push(ParityRead {
                        index,
                        bytes: u64::try_from(read.bytes().len()).map_err(|_| {
                            PowerError::InvalidFormat(
                                "benchmark tensor byte length overflowed".to_string(),
                            )
                        })?,
                        digest: Sha256::digest(read.bytes()).into(),
                    });
                }
                Ok(reads)
            }));
        }
        let mut reads = Vec::with_capacity(names.len());
        for worker in workers {
            let mut worker_reads = worker.join().map_err(|_| {
                PowerError::InferenceFailed("storage benchmark parity worker panicked".to_string())
            })??;
            reads.append(&mut worker_reads);
        }
        Ok::<_, PowerError>(reads)
    })?;
    let mut completed = completed;
    completed.sort_by_key(|read| read.index);
    if completed.len() != names.len()
        || completed
            .iter()
            .enumerate()
            .any(|(index, read)| index != read.index)
    {
        return Err(PowerError::InferenceFailed(
            "storage benchmark parity did not cover its deterministic tensor sequence".to_string(),
        ));
    }
    let mut output = Sha256::new();
    for read in completed {
        output.update(u64::try_from(read.index).unwrap_or(u64::MAX).to_le_bytes());
        output.update(read.bytes.to_le_bytes());
        output.update(read.digest);
    }
    Ok(format!("{:x}", output.finalize()))
}

fn sequence_digest(store: &WeightStore, names: &[String]) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(store.sha256().as_bytes());
    for (index, name) in names.iter().enumerate() {
        let descriptor = store.descriptor(name).ok_or_else(|| {
            PowerError::InvalidFormat(
                "storage benchmark sequence references an unknown tensor".to_string(),
            )
        })?;
        digest.update(u64::try_from(index).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(u64::try_from(name.len()).unwrap_or(u64::MAX).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(descriptor.bytes.to_le_bytes());
        digest.update(descriptor.dtype.as_bytes());
        for dimension in &descriptor.shape {
            digest.update(u64::try_from(*dimension).unwrap_or(u64::MAX).to_le_bytes());
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_label(label: &str, value: &str) -> Result<()> {
    if value.trim() != value
        || value.is_empty()
        || value.len() > MAX_LABEL_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(PowerError::InvalidRequest(format!(
            "storage benchmark {label} must be a bounded non-control string without surrounding whitespace"
        )));
    }
    Ok(())
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos())
        .unwrap_or(u64::MAX)
        .max(1)
}

fn throughput(bytes: u64, duration: Duration) -> u64 {
    let nanos = duration.as_nanos().max(1);
    let rate = u128::from(bytes)
        .saturating_mul(1_000_000_000)
        .checked_div(nanos)
        .unwrap_or_default();
    u64::try_from(rate).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
