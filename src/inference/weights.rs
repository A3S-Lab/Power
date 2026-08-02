use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};

use super::{InferenceLimits, RuntimeDevice};

const HASH_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorDescriptor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightFileDescriptor {
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// How much of the primary weight collection a read-only source must cover.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightSourceCoverage {
    /// The source must be a byte-identical copy of the complete collection.
    #[default]
    Complete,
    /// The source may contain a non-empty, byte-identical subset of primary
    /// SafeTensors files. Tensors outside that subset stay on the primary.
    Partial,
}

/// Selects how effective source weights are derived.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightSourceWeighting {
    /// Use each source's explicitly configured relative weight.
    #[default]
    Configured,
    /// Derive relative weights from the throughput observed during Power's
    /// mandatory integrity-validation read. This performs no additional scan.
    ValidationThroughput,
}

/// One verified copy of all or part of a weight collection and its relative
/// read-bandwidth weight. The weight affects source selection only; every
/// available file is checked byte-for-byte against the primary and therefore
/// cannot change model semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightSourceConfig {
    pub root: PathBuf,
    pub read_weight: u32,
    #[serde(default)]
    pub coverage: WeightSourceCoverage,
}

impl WeightSourceConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_weight: 1,
            coverage: WeightSourceCoverage::Complete,
        }
    }

    pub fn with_read_weight(mut self, read_weight: u32) -> Self {
        self.read_weight = read_weight;
        self
    }

    /// Marks this source as an explicitly partial read-only replica.
    pub fn with_coverage(mut self, coverage: WeightSourceCoverage) -> Self {
        self.coverage = coverage;
        self
    }
}

/// Primary weight root plus optional verified read-only replicas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightStoreConfig {
    pub primary: WeightSourceConfig,
    pub replicas: Vec<WeightSourceConfig>,
    #[serde(default)]
    pub source_weighting: WeightSourceWeighting,
}

impl WeightStoreConfig {
    pub fn new(primary_root: impl Into<PathBuf>) -> Self {
        Self {
            primary: WeightSourceConfig::new(primary_root),
            replicas: Vec::new(),
            source_weighting: WeightSourceWeighting::Configured,
        }
    }

    pub fn with_primary_read_weight(mut self, read_weight: u32) -> Self {
        self.primary.read_weight = read_weight;
        self
    }

    pub fn with_replica(mut self, replica: WeightSourceConfig) -> Self {
        self.replicas.push(replica);
        self
    }

    /// Adds a source that intentionally contains only an exact subset of the
    /// primary collection's SafeTensors files.
    pub fn with_partial_replica(mut self, mut replica: WeightSourceConfig) -> Self {
        replica.coverage = WeightSourceCoverage::Partial;
        self.replicas.push(replica);
        self
    }

    pub fn with_source_weighting(mut self, source_weighting: WeightSourceWeighting) -> Self {
        self.source_weighting = source_weighting;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightSourceRole {
    Primary,
    Replica,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightSourceDescriptor {
    pub index: usize,
    pub role: WeightSourceRole,
    pub root: PathBuf,
    /// Effective relative weight used by deterministic source selection.
    pub read_weight: u32,
    pub configured_read_weight: u32,
    pub source_weighting: WeightSourceWeighting,
    /// Read throughput observed while hashing this source. It is returned only
    /// through this explicit descriptor API and is never logged automatically.
    pub validation_bytes_per_second: u64,
    pub coverage: WeightSourceCoverage,
    pub verified_files: usize,
    pub verified_tensors: usize,
    pub verified_bytes: u64,
}

/// Validated, mmap-backed SafeTensors collection.
///
/// Duplicate tensor names are refused instead of relying on last-file-wins
/// behavior. The aggregate digest includes each relative file name, length,
/// and content in stable lexical order.
pub struct WeightStore {
    root: PathBuf,
    paths: Vec<PathBuf>,
    tensors: MmapedSafetensors,
    inventory: BTreeMap<String, TensorDescriptor>,
    files: Vec<WeightFileDescriptor>,
    sha256: String,
    bytes: u64,
    read_weight: u32,
    configured_read_weight: u32,
    source_weighting: WeightSourceWeighting,
    validation_bytes_per_second: u64,
    coverage: WeightSourceCoverage,
    replicas: Vec<WeightStore>,
}

pub(crate) struct LoadedWeight {
    pub(crate) tensor: Tensor,
    pub(crate) source_index: usize,
    pub(crate) fell_back: bool,
}

impl WeightStore {
    pub fn open(root: impl AsRef<Path>, limits: &InferenceLimits) -> Result<Self> {
        Self::open_config(&WeightStoreConfig::new(root.as_ref()), limits)
    }

    /// Opens a primary SafeTensors collection and verified read-only replicas.
    /// Complete replicas must match the aggregate collection digest. Partial
    /// replicas may contain a non-empty subset of byte-identical primary files.
    /// Tensor names are mapped only across sources that actually contain them,
    /// using a deterministic bandwidth-weighted hash so demand and prefetch use
    /// the same source. A recoverable replica load error falls back to primary.
    pub fn open_config(config: &WeightStoreConfig, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        if config.primary.coverage != WeightSourceCoverage::Complete {
            return Err(PowerError::Config(
                "the primary weight source must have complete coverage".to_string(),
            ));
        }
        let source_count = config.replicas.len().saturating_add(1);
        if source_count > limits.max_weight_sources {
            return Err(PowerError::Config(format!(
                "weight store declares {source_count} sources, exceeding the {} source limit",
                limits.max_weight_sources
            )));
        }
        let mut primary = Self::open_single(&config.primary, limits)?;
        let mut roots = vec![primary.root.clone()];
        for replica_config in &config.replicas {
            let replica = Self::open_single(replica_config, limits)?;
            if roots.contains(&replica.root) {
                return Err(PowerError::Config(format!(
                    "weight source '{}' is configured more than once",
                    replica.root.display()
                )));
            }
            match replica.coverage {
                WeightSourceCoverage::Complete if replica.sha256 != primary.sha256 => {
                    return Err(PowerError::IntegrityCheckFailed {
                        model: format!("weight replica {}", replica.root.display()),
                        expected: primary.sha256.clone(),
                        actual: replica.sha256,
                    });
                }
                WeightSourceCoverage::Partial => {
                    validate_partial_replica(&primary, &replica)?;
                }
                WeightSourceCoverage::Complete => {}
            }
            roots.push(replica.root.clone());
            primary.replicas.push(replica);
        }
        primary.apply_source_weighting(config.source_weighting);
        Ok(primary)
    }

    fn open_single(config: &WeightSourceConfig, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        if config.read_weight == 0 {
            return Err(PowerError::Config(
                "weight source read weight must be greater than zero".to_string(),
            ));
        }
        let root = &config.root;
        let root = std::fs::canonicalize(root).map_err(|error| {
            PowerError::InvalidFormat(format!(
                "failed to resolve model directory '{}': {error}",
                root.display()
            ))
        })?;
        if !std::fs::metadata(&root)?.is_dir() {
            return Err(PowerError::InvalidFormat(format!(
                "model root '{}' is not a directory",
                root.display()
            )));
        }
        let mut paths = discover_safetensors(&root, limits.max_model_files)?;
        paths.sort();
        if paths.is_empty() {
            return Err(PowerError::InvalidFormat(format!(
                "model root '{}' contains no SafeTensors files",
                root.display()
            )));
        }
        let validation_started = Instant::now();
        let (sha256, bytes, files) = hash_files(&root, &paths, limits.max_model_bytes)?;
        let validation_bytes_per_second = bytes_per_second(bytes, validation_started.elapsed());

        // SAFETY: every path was canonicalized, checked as a regular file
        // beneath `root`, opened and read completely while hashing, and remains
        // owned by this store for at least as long as the memory maps.
        let tensors = unsafe { MmapedSafetensors::multi(&paths) }.map_err(|error| {
            PowerError::InvalidFormat(format!("failed to map SafeTensors weights: {error}"))
        })?;
        let mut inventory = BTreeMap::new();
        for (name, view) in tensors.tensors() {
            let descriptor = TensorDescriptor {
                name: name.clone(),
                dtype: format!("{:?}", view.dtype()).to_ascii_lowercase(),
                shape: view.shape().to_vec(),
                bytes: u64::try_from(view.data().len()).map_err(|_| {
                    PowerError::InvalidFormat(format!(
                        "tensor '{name}' byte length exceeds the supported range"
                    ))
                })?,
            };
            if inventory.insert(name.clone(), descriptor).is_some() {
                return Err(PowerError::InvalidFormat(format!(
                    "duplicate tensor name '{name}' appears in the model container"
                )));
            }
        }
        Ok(Self {
            root,
            paths,
            tensors,
            inventory,
            files,
            sha256,
            bytes,
            read_weight: config.read_weight,
            configured_read_weight: config.read_weight,
            source_weighting: WeightSourceWeighting::Configured,
            validation_bytes_per_second,
            coverage: config.coverage,
            replicas: Vec::new(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    pub fn sources(&self) -> Vec<WeightSourceDescriptor> {
        let mut sources = Vec::with_capacity(self.replicas.len().saturating_add(1));
        sources.push(WeightSourceDescriptor {
            index: 0,
            role: WeightSourceRole::Primary,
            root: self.root.clone(),
            read_weight: self.read_weight,
            configured_read_weight: self.configured_read_weight,
            source_weighting: self.source_weighting,
            validation_bytes_per_second: self.validation_bytes_per_second,
            coverage: WeightSourceCoverage::Complete,
            verified_files: self.files.len(),
            verified_tensors: self.inventory.len(),
            verified_bytes: self.bytes,
        });
        sources.extend(self.replicas.iter().enumerate().map(|(index, replica)| {
            WeightSourceDescriptor {
                index: index.saturating_add(1),
                role: WeightSourceRole::Replica,
                root: replica.root.clone(),
                read_weight: replica.read_weight,
                configured_read_weight: replica.configured_read_weight,
                source_weighting: replica.source_weighting,
                validation_bytes_per_second: replica.validation_bytes_per_second,
                coverage: replica.coverage,
                verified_files: replica.files.len(),
                verified_tensors: replica.inventory.len(),
                verified_bytes: replica.bytes,
            }
        }));
        sources
    }

    pub fn inventory(&self) -> impl ExactSizeIterator<Item = &TensorDescriptor> {
        self.inventory.values()
    }

    pub fn descriptor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.inventory.get(name)
    }

    pub fn files(&self) -> &[WeightFileDescriptor] {
        &self.files
    }

    pub fn contains(&self, name: &str) -> bool {
        self.inventory.contains_key(name)
    }

    /// Verifies the canonical SafeTensors collection digest already computed
    /// while opening the store.
    pub fn verify_integrity(&self, model_name: &str, expected_sha256: &str) -> Result<()> {
        if self.sha256 == expected_sha256 {
            return Ok(());
        }
        Err(PowerError::IntegrityCheckFailed {
            model: model_name.to_string(),
            expected: expected_sha256.to_string(),
            actual: self.sha256.clone(),
        })
    }

    /// Verifies an existing Power Ed25519 model signature over this store's
    /// canonical collection digest.
    pub fn verify_signature(
        &self,
        model_name: &str,
        signature_anchor_path: &Path,
        public_key_hex: &str,
    ) -> Result<()> {
        crate::tee::model_signature::verify_model_signature_hash(
            model_name,
            &self.sha256,
            signature_anchor_path,
            public_key_hex,
        )
    }

    /// Materializes one validated tensor on a resolved Power runtime device.
    pub fn load_tensor(&self, name: &str, device: &RuntimeDevice) -> Result<Tensor> {
        self.load(name, device.tensor_device())
    }

    pub(crate) fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        Ok(self.load_tracked(name, device)?.tensor)
    }

    pub(crate) fn load_tracked(&self, name: &str, device: &Device) -> Result<LoadedWeight> {
        let source_index = self.select_source(name);
        if source_index == 0 {
            return Ok(LoadedWeight {
                tensor: self.load_primary(name, device)?,
                source_index,
                fell_back: false,
            });
        }

        let replica = &self.replicas[source_index - 1];
        match replica.tensors.load(name, device) {
            Ok(tensor) => Ok(LoadedWeight {
                tensor,
                source_index,
                fell_back: false,
            }),
            Err(replica_error) => self
                .load_primary(name, device)
                .map(|tensor| LoadedWeight {
                    tensor,
                    source_index: 0,
                    fell_back: true,
                })
                .map_err(|primary_error| {
                    PowerError::InvalidFormat(format!(
                        "failed to load model tensor '{name}' from replica {source_index} ({replica_error}) and primary ({primary_error})"
                    ))
                }),
        }
    }

    fn load_primary(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.tensors.load(name, device).map_err(|error| {
            PowerError::InvalidFormat(format!("failed to load model tensor '{name}': {error}"))
        })
    }

    fn select_source(&self, name: &str) -> usize {
        if self.replicas.is_empty() {
            return 0;
        }
        let total_weight = self
            .replicas
            .iter()
            .filter(|replica| replica.contains(name))
            .fold(u64::from(self.read_weight), |total, replica| {
                total.saturating_add(u64::from(replica.read_weight))
            });
        if total_weight == u64::from(self.read_weight) {
            return 0;
        }
        let digest = Sha256::digest(name.as_bytes());
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&digest[..8]);
        let mut slot = u64::from_le_bytes(prefix) % total_weight;
        if slot < u64::from(self.read_weight) {
            return 0;
        }
        slot -= u64::from(self.read_weight);
        for (index, replica) in self.replicas.iter().enumerate() {
            if !replica.contains(name) {
                continue;
            }
            let weight = u64::from(replica.read_weight);
            if slot < weight {
                return index.saturating_add(1);
            }
            slot -= weight;
        }
        0
    }

    fn apply_source_weighting(&mut self, source_weighting: WeightSourceWeighting) {
        self.source_weighting = source_weighting;
        for replica in &mut self.replicas {
            replica.source_weighting = source_weighting;
        }
        if source_weighting == WeightSourceWeighting::Configured {
            return;
        }
        let rates = std::iter::once(self.validation_bytes_per_second)
            .chain(
                self.replicas
                    .iter()
                    .map(|replica| replica.validation_bytes_per_second),
            )
            .collect::<Vec<_>>();
        let weights = normalized_throughput_weights(&rates);
        self.read_weight = weights[0];
        for (replica, weight) in self.replicas.iter_mut().zip(weights.into_iter().skip(1)) {
            replica.read_weight = weight;
        }
    }
}

fn bytes_per_second(bytes: u64, elapsed: Duration) -> u64 {
    if bytes == 0 {
        return 0;
    }
    let nanos = elapsed.as_nanos().max(1);
    let rate = u128::from(bytes)
        .saturating_mul(1_000_000_000)
        .checked_div(nanos)
        .unwrap_or_default();
    u64::try_from(rate).unwrap_or(u64::MAX).max(1)
}

fn normalized_throughput_weights(rates: &[u64]) -> Vec<u32> {
    const MAX_RELATIVE_WEIGHT: u128 = 1_024;
    let fastest = rates.iter().copied().max().unwrap_or_default();
    if fastest == 0 {
        return vec![1; rates.len()];
    }
    rates
        .iter()
        .map(|rate| {
            let scaled = u128::from(*rate)
                .saturating_mul(MAX_RELATIVE_WEIGHT)
                .checked_div(u128::from(fastest))
                .unwrap_or_default()
                .max(1);
            u32::try_from(scaled).unwrap_or(1_024)
        })
        .collect()
}

fn validate_partial_replica(primary: &WeightStore, replica: &WeightStore) -> Result<()> {
    let primary_files = primary
        .files
        .iter()
        .map(|file| (file.relative_path.as_str(), file))
        .collect::<BTreeMap<_, _>>();
    for file in &replica.files {
        let Some(primary_file) = primary_files.get(file.relative_path.as_str()) else {
            return Err(PowerError::InvalidFormat(format!(
                "partial weight replica '{}' contains file '{}' that is absent from the primary collection",
                replica.root.display(),
                file.relative_path
            )));
        };
        if file.bytes != primary_file.bytes || file.sha256 != primary_file.sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: format!(
                    "partial weight replica file {}/{}",
                    replica.root.display(),
                    file.relative_path
                ),
                expected: primary_file.sha256.clone(),
                actual: file.sha256.clone(),
            });
        }
    }
    for (name, descriptor) in &replica.inventory {
        if primary.inventory.get(name) != Some(descriptor) {
            return Err(PowerError::InvalidFormat(format!(
                "partial weight replica '{}' changed tensor descriptor '{name}'",
                replica.root.display()
            )));
        }
    }
    Ok(())
}

fn discover_safetensors(root: &Path, max_files: usize) -> Result<Vec<PathBuf>> {
    let mut pending = vec![root.to_path_buf()];
    let mut paths = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(PowerError::InvalidFormat(format!(
                    "model path '{}' is a symbolic link",
                    entry.path().display()
                )));
            }
            if file_type.is_dir() {
                pending.push(entry.path());
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            if entry.path().extension().and_then(|value| value.to_str()) == Some("safetensors") {
                let canonical = std::fs::canonicalize(entry.path())?;
                if !canonical.starts_with(root) {
                    return Err(PowerError::InvalidFormat(format!(
                        "model file '{}' escapes its model root",
                        canonical.display()
                    )));
                }
                paths.push(canonical);
                if paths.len() > max_files {
                    return Err(PowerError::InvalidFormat(format!(
                        "model contains more than {max_files} SafeTensors files"
                    )));
                }
            }
        }
    }
    Ok(paths)
}

fn hash_files(
    root: &Path,
    paths: &[PathBuf],
    max_bytes: u64,
) -> Result<(String, u64, Vec<WeightFileDescriptor>)> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut descriptors = Vec::with_capacity(paths.len());
    let mut buffer = vec![0_u8; HASH_BUFFER_BYTES];
    for path in paths {
        let relative = path.strip_prefix(root).map_err(|_| {
            PowerError::InvalidFormat(format!(
                "model file '{}' is outside its root",
                path.display()
            ))
        })?;
        let relative = relative.to_str().ok_or_else(|| {
            PowerError::InvalidFormat("model file names must be valid UTF-8".to_string())
        })?;
        let metadata = std::fs::metadata(path)?;
        if !metadata.is_file() || metadata.len() == 0 {
            return Err(PowerError::InvalidFormat(format!(
                "model file '{}' must be a non-empty regular file",
                path.display()
            )));
        }
        total = total
            .checked_add(metadata.len())
            .ok_or_else(|| PowerError::InvalidFormat("model byte length overflowed".to_string()))?;
        if total > max_bytes {
            return Err(PowerError::InvalidFormat(format!(
                "model contains {total} bytes, exceeding the {max_bytes} byte limit"
            )));
        }
        hasher.update((relative.len() as u64).to_le_bytes());
        hasher.update(relative.as_bytes());
        hasher.update(metadata.len().to_le_bytes());
        let mut file_hasher = Sha256::new();
        let mut file = File::open(path)?;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            file_hasher.update(&buffer[..read]);
        }
        descriptors.push(WeightFileDescriptor {
            relative_path: relative.to_string(),
            bytes: metadata.len(),
            sha256: format!("{:x}", file_hasher.finalize()),
        });
    }
    Ok((format!("{:x}", hasher.finalize()), total, descriptors))
}

impl std::fmt::Debug for WeightStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WeightStore")
            .field("root", &self.root)
            .field("paths", &self.paths)
            .field("tensor_count", &self.inventory.len())
            .field("files", &self.files)
            .field("sha256", &self.sha256)
            .field("bytes", &self.bytes)
            .field("sources", &self.sources())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
