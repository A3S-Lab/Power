use std::path::PathBuf;

use a3s_power::error::{PowerError, Result};
use a3s_power::inference::{
    compare_storage_benchmarks, run_storage_benchmark, InferenceLimits, StorageBenchmarkConfig,
    StorageBenchmarkReport, StorageCachePreparation, StorageCacheState, WeightReadStrategy,
    WeightSourceConfig, WeightSourceRepresentation, WeightSourceWeighting, WeightStoreConfig,
};

const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;

const USAGE: &str = r#"A3S Power storage benchmark

Usage:
  a3s-power-storage-bench \
    --primary <directory> \
    [--primary-shard <directory>]... \
    [--replica <directory>]... \
    [--partial-replica <directory>]... \
    [--lossless-replica <directory>::<artifact-sha256>]... \
    [--partial-lossless-replica <directory>::<artifact-sha256>]... \
    --strategy <mmap|positional-buffered|positional-cache-bypass|positional-direct> \
    --power-commit <lowercase-git-revision> \
    --filesystem-class <label> \
    --device-class <label> \
    --cpu-model <label> \
    --ram-bytes <bytes> \
    --cache-state <cold|warm> \
    --cache-preparation <warm-sequence|linux-fadvise-dontneed> \
    [--source-weighting <configured|validation-throughput>] \
    [--concurrency <count>] \
    [--samples <count>] \
    [--max-tensors <count>]

Cold runs require Linux `POSIX_FADV_DONTNEED` plus `mincore` verification and
exactly one sample per process. The report is written to stdout only and never
contains model paths or tensor names.

Compare separately captured reports:
  a3s-power-storage-bench compare <report.json> <report.json> [...]
"#;

fn main() {
    if let Err(error) = run() {
        eprintln!("storage benchmark failed: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<()> {
    let mut values = std::env::args().skip(1).collect::<Vec<_>>();
    if values.first().map(String::as_str) == Some("compare") {
        values.remove(0);
        return compare_reports(&values);
    }
    let mut parser = Arguments::new(values);
    if parser.take_flag("--help") || parser.take_flag("-h") {
        print!("{USAGE}");
        return Ok(());
    }
    let primary = parser.required_path("--primary")?;
    let primary_shards = parser.paths("--primary-shard")?;
    let strategy = parse_strategy(&parser.required("--strategy")?)?;
    let source_weighting = parser
        .optional("--source-weighting")?
        .map(|value| parse_weighting(&value))
        .transpose()?
        .unwrap_or(WeightSourceWeighting::Configured);
    let mut weights = WeightStoreConfig::new(primary)
        .with_primary_read_strategy(strategy)
        .with_source_weighting(source_weighting);
    for shard in primary_shards {
        weights = weights.with_primary_shard_root(shard);
    }
    for replica in parser.paths("--replica")? {
        weights =
            weights.with_replica(WeightSourceConfig::new(replica).with_read_strategy(strategy));
    }
    for replica in parser.paths("--partial-replica")? {
        weights = weights
            .with_partial_replica(WeightSourceConfig::new(replica).with_read_strategy(strategy));
    }
    for (replica, artifact_sha256) in parser.lossless_sources("--lossless-replica")? {
        weights = weights.with_replica(
            WeightSourceConfig::new(replica)
                .with_read_strategy(strategy)
                .with_representation(WeightSourceRepresentation::LosslessRansNibble256V1 {
                    artifact_sha256,
                }),
        );
    }
    for (replica, artifact_sha256) in parser.lossless_sources("--partial-lossless-replica")? {
        weights = weights.with_partial_replica(
            WeightSourceConfig::new(replica)
                .with_read_strategy(strategy)
                .with_representation(WeightSourceRepresentation::LosslessRansNibble256V1 {
                    artifact_sha256,
                }),
        );
    }
    let config = StorageBenchmarkConfig {
        weights,
        power_commit: parser.required("--power-commit")?,
        filesystem_class: parser.required("--filesystem-class")?,
        device_class: parser.required("--device-class")?,
        cpu_model: parser.required("--cpu-model")?,
        ram_bytes: parser.required_number("--ram-bytes")?,
        cache_state: parse_cache_state(&parser.required("--cache-state")?)?,
        cache_preparation: parse_cache_preparation(&parser.required("--cache-preparation")?)?,
        concurrency: parser.optional_number("--concurrency")?.unwrap_or(1),
        samples: parser.optional_number("--samples")?.unwrap_or(5),
        max_tensors: parser
            .optional_number("--max-tensors")?
            .unwrap_or(usize::MAX),
    };
    parser.finish()?;
    let report = run_storage_benchmark(&config, &InferenceLimits::default())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn compare_reports(paths: &[String]) -> Result<()> {
    if paths.len() < 2 {
        return Err(PowerError::InvalidRequest(
            "storage benchmark compare requires at least two report paths".to_string(),
        ));
    }
    let mut reports = Vec::with_capacity(paths.len());
    for path in paths {
        let path = PathBuf::from(path);
        let metadata = std::fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_file()
            || metadata.len() == 0
            || metadata.len() > MAX_REPORT_BYTES
        {
            return Err(PowerError::InvalidRequest(
                "storage benchmark report must be a bounded regular non-symlink file".to_string(),
            ));
        }
        let bytes = std::fs::read(path)?;
        reports.push(serde_json::from_slice::<StorageBenchmarkReport>(&bytes)?);
    }
    let comparison = compare_storage_benchmarks(&reports)?;
    println!("{}", serde_json::to_string_pretty(&comparison)?);
    if comparison.output_byte_parity {
        Ok(())
    } else {
        Err(PowerError::IntegrityCheckFailed {
            model: "storage benchmark output parity".to_string(),
            expected: comparison
                .output_sha256s
                .first()
                .cloned()
                .unwrap_or_default(),
            actual: comparison
                .output_sha256s
                .get(1)
                .cloned()
                .unwrap_or_default(),
        })
    }
}

fn parse_strategy(value: &str) -> Result<WeightReadStrategy> {
    match value {
        "mmap" => Ok(WeightReadStrategy::Mmap),
        "positional-buffered" => Ok(WeightReadStrategy::PositionalBuffered),
        "positional-cache-bypass" => Ok(WeightReadStrategy::PositionalCacheBypass),
        "positional-direct" => Ok(WeightReadStrategy::PositionalDirect),
        _ => Err(PowerError::InvalidRequest(format!(
            "unsupported storage benchmark strategy '{value}'"
        ))),
    }
}

fn parse_weighting(value: &str) -> Result<WeightSourceWeighting> {
    match value {
        "configured" => Ok(WeightSourceWeighting::Configured),
        "validation-throughput" => Ok(WeightSourceWeighting::ValidationThroughput),
        _ => Err(PowerError::InvalidRequest(format!(
            "unsupported source weighting '{value}'"
        ))),
    }
}

fn parse_cache_state(value: &str) -> Result<StorageCacheState> {
    match value {
        "cold" => Ok(StorageCacheState::Cold),
        "warm" => Ok(StorageCacheState::Warm),
        _ => Err(PowerError::InvalidRequest(format!(
            "unsupported cache state '{value}'"
        ))),
    }
}

fn parse_cache_preparation(value: &str) -> Result<StorageCachePreparation> {
    match value {
        "warm-sequence" => Ok(StorageCachePreparation::WarmSequence),
        "linux-fadvise-dontneed" => Ok(StorageCachePreparation::LinuxFadviseDontNeed),
        _ => Err(PowerError::InvalidRequest(format!(
            "unsupported cache preparation '{value}'"
        ))),
    }
}

struct Arguments {
    values: Vec<String>,
}

impl Arguments {
    fn new(values: Vec<String>) -> Self {
        Self { values }
    }

    fn required(&mut self, name: &str) -> Result<String> {
        self.optional(name)?
            .ok_or_else(|| PowerError::InvalidRequest(format!("missing required argument {name}")))
    }

    fn required_path(&mut self, name: &str) -> Result<PathBuf> {
        self.required(name).map(PathBuf::from)
    }

    fn required_number<T>(&mut self, name: &str) -> Result<T>
    where
        T: std::str::FromStr,
    {
        parse_number(name, &self.required(name)?)
    }

    fn optional_number<T>(&mut self, name: &str) -> Result<Option<T>>
    where
        T: std::str::FromStr,
    {
        self.optional(name)?
            .map(|value| parse_number(name, &value))
            .transpose()
    }

    fn optional(&mut self, name: &str) -> Result<Option<String>> {
        let Some(index) = self.values.iter().position(|value| value == name) else {
            return Ok(None);
        };
        if index.saturating_add(1) >= self.values.len() || self.values[index + 1].starts_with("--")
        {
            return Err(PowerError::InvalidRequest(format!(
                "argument {name} requires a value"
            )));
        }
        self.values.remove(index);
        Ok(Some(self.values.remove(index)))
    }

    fn paths(&mut self, name: &str) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        while let Some(path) = self.optional(name)? {
            paths.push(PathBuf::from(path));
        }
        Ok(paths)
    }

    fn lossless_sources(&mut self, name: &str) -> Result<Vec<(PathBuf, String)>> {
        let mut sources = Vec::new();
        while let Some(value) = self.optional(name)? {
            let (path, artifact_sha256) = value.rsplit_once("::").ok_or_else(|| {
                PowerError::InvalidRequest(format!(
                    "argument {name} must use <directory>::<artifact-sha256>"
                ))
            })?;
            if path.is_empty()
                || artifact_sha256.len() != 64
                || !artifact_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(PowerError::InvalidRequest(format!(
                    "argument {name} requires a directory and a lowercase SHA-256 digest"
                )));
            }
            sources.push((PathBuf::from(path), artifact_sha256.to_string()));
        }
        Ok(sources)
    }

    fn take_flag(&mut self, name: &str) -> bool {
        if let Some(index) = self.values.iter().position(|value| value == name) {
            self.values.remove(index);
            true
        } else {
            false
        }
    }

    fn finish(self) -> Result<()> {
        if self.values.is_empty() {
            Ok(())
        } else {
            Err(PowerError::InvalidRequest(format!(
                "unknown storage benchmark argument '{}'",
                self.values[0]
            )))
        }
    }
}

fn parse_number<T>(name: &str, value: &str) -> Result<T>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| PowerError::InvalidRequest(format!("argument {name} must be a valid number")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_paths_and_flags_are_consumed_without_backend_names() {
        let mut arguments = Arguments::new(vec![
            "--replica".to_string(),
            "/one".to_string(),
            "--help".to_string(),
            "--replica".to_string(),
            "/two".to_string(),
        ]);
        assert_eq!(
            arguments.paths("--replica").unwrap(),
            [PathBuf::from("/one"), PathBuf::from("/two")]
        );
        assert!(arguments.take_flag("--help"));
        arguments.finish().unwrap();
    }

    #[test]
    fn unknown_and_incomplete_arguments_fail_closed() {
        assert!(Arguments::new(vec!["--unknown".to_string()])
            .finish()
            .is_err());
        assert!(Arguments::new(vec!["--primary".to_string()])
            .optional("--primary")
            .is_err());
    }

    #[test]
    fn lossless_sources_require_explicit_lowercase_artifact_pins() {
        let digest = "a".repeat(64);
        let mut arguments = Arguments::new(vec![
            "--lossless-replica".to_string(),
            format!("/compressed::{digest}"),
        ]);
        assert_eq!(
            arguments.lossless_sources("--lossless-replica").unwrap(),
            [(PathBuf::from("/compressed"), digest)]
        );
        arguments.finish().unwrap();

        assert!(Arguments::new(vec![
            "--lossless-replica".to_string(),
            "/compressed::BAD".to_string(),
        ])
        .lossless_sources("--lossless-replica")
        .is_err());
    }
}
