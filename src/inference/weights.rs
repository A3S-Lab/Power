use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::safetensors::Load;
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};
use safetensors::tensor::TensorView;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

use super::{InferenceLimits, RuntimeDevice};

mod encrypted_collection;
#[cfg(test)]
mod encrypted_collection_tests;
mod encrypted_file;
#[cfg(test)]
mod encrypted_file_tests;
mod encrypted_format;
mod files;
mod index;
mod lossless;
mod range_io;
mod tensor_range;

pub use tensor_range::TensorRangeRead;

pub use encrypted_collection::{
    encrypt_seekable_weight_collection, SeekableEncryptedWeightCollectionReport,
    SeekableEncryptedWeightFile, SeekableEncryptedWeightManifest, SeekableEncryptedWeightSource,
    ENCRYPTED_WEIGHT_MANIFEST_FILE, ENCRYPTED_WEIGHT_MANIFEST_SCHEMA,
};
pub use encrypted_file::{
    encrypt_seekable_weight_file, SeekableEncryptedFile, SeekableEncryptedFileVerification,
    SeekableWeightKey,
};
pub use encrypted_format::{SeekableEncryptedFileDescriptor, DEFAULT_ENCRYPTED_CHUNK_BYTES};
use files::{discover_safetensors, hash_files, resolve_weight_roots};
use index::TensorLocation;
pub use lossless::{
    weight_collection_sha256, LosslessEncodedRecord, LosslessRansNibbleHistogram,
    LosslessRansNibbleTable, WeightSourceRepresentation, LOSSLESS_RANS_FORMAT_METADATA_KEY,
    LOSSLESS_RANS_TABLE_METADATA_KEY,
};
use range_io::WeightFileReader;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorDescriptor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

/// Exact, verified location of one tensor inside one source file.
///
/// `file_index` addresses [`WeightStore::files`] without exposing an absolute
/// host path. Callers must request this descriptor explicitly; Power never
/// logs or persists tensor names or byte ranges automatically.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TensorStorageDescriptor {
    pub file_index: usize,
    pub absolute_offset: u64,
    pub bytes: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightFileDescriptor {
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
}

/// How much of the primary weight collection a read-only source must cover.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
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

/// How an already verified tensor range is materialized from storage.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum WeightReadStrategy {
    /// Candle's validated SafeTensors mmap path. This remains the default.
    #[default]
    Mmap,
    /// Bounded positional reads through the operating-system page cache.
    PositionalBuffered,
    /// Bounded positional reads through a platform cache-bypass handle.
    ///
    /// macOS uses `F_NOCACHE`. This is intentionally distinct from direct I/O:
    /// it does not prove that pages populated by an earlier handle were absent,
    /// and unsupported platforms fail explicitly.
    PositionalCacheBypass,
    /// Explicit OS direct/unbuffered reads with aligned scratch buffers.
    /// Unsupported filesystems and platforms fail explicitly without falling
    /// back to buffered reads under the same source.
    PositionalDirect,
}

/// One verified copy of all or part of a weight collection and its relative
/// read-bandwidth weight. The weight affects source selection only; every
/// available file is checked byte-for-byte against the primary and therefore
/// cannot change model semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightSourceConfig {
    pub root: PathBuf,
    /// Additional physical roots whose disjoint relative SafeTensors paths
    /// form the same logical source. Physical placement never enters the
    /// canonical collection digest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_roots: Vec<PathBuf>,
    pub read_weight: u32,
    #[serde(default)]
    pub coverage: WeightSourceCoverage,
    #[serde(default)]
    pub read_strategy: WeightReadStrategy,
    #[serde(default)]
    pub representation: WeightSourceRepresentation,
}

impl WeightSourceConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            shard_roots: Vec::new(),
            read_weight: 1,
            coverage: WeightSourceCoverage::Complete,
            read_strategy: WeightReadStrategy::Mmap,
            representation: WeightSourceRepresentation::CanonicalSafeTensors,
        }
    }

    pub fn with_read_weight(mut self, read_weight: u32) -> Self {
        self.read_weight = read_weight;
        self
    }

    /// Adds one disjoint physical root to this logical weight source.
    pub fn with_shard_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.shard_roots.push(root.into());
        self
    }

    /// Marks this source as an explicitly partial read-only replica.
    pub fn with_coverage(mut self, coverage: WeightSourceCoverage) -> Self {
        self.coverage = coverage;
        self
    }

    pub fn with_read_strategy(mut self, read_strategy: WeightReadStrategy) -> Self {
        self.read_strategy = read_strategy;
        self
    }

    pub fn with_representation(mut self, representation: WeightSourceRepresentation) -> Self {
        self.representation = representation;
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

    /// Adds one disjoint physical root to the canonical primary collection.
    pub fn with_primary_shard_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.primary.shard_roots.push(root.into());
        self
    }

    pub fn with_primary_read_strategy(mut self, read_strategy: WeightReadStrategy) -> Self {
        self.primary.read_strategy = read_strategy;
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
    /// Additional physical roots in this logical source. This explicit
    /// descriptor API may expose paths; telemetry and receipts never do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shard_roots: Vec<PathBuf>,
    /// Effective relative weight used by deterministic source selection.
    pub read_weight: u32,
    pub configured_read_weight: u32,
    pub source_weighting: WeightSourceWeighting,
    /// Read throughput observed while hashing this source. It is returned only
    /// through this explicit descriptor API and is never logged automatically.
    pub validation_bytes_per_second: u64,
    pub coverage: WeightSourceCoverage,
    pub read_strategy: WeightReadStrategy,
    #[serde(default)]
    pub representation: WeightSourceRepresentation,
    /// Platform-derived storage transfer alignment used for direct I/O.
    pub io_block_size: u64,
    pub verified_files: usize,
    pub verified_tensors: usize,
    pub verified_bytes: u64,
}

/// Validated SafeTensors collection with mmap-default and opt-in positional
/// materialization.
///
/// Duplicate tensor names are refused instead of relying on last-file-wins
/// behavior. The aggregate digest includes each relative file name, length,
/// and content in stable lexical order.
pub struct WeightStore {
    root: PathBuf,
    roots: Vec<PathBuf>,
    paths: Vec<PathBuf>,
    tensors: Option<MmapedSafetensors>,
    inventory: BTreeMap<String, TensorDescriptor>,
    locations: BTreeMap<String, TensorLocation>,
    readers: Vec<WeightFileReader>,
    io_block_size: u64,
    files: Vec<WeightFileDescriptor>,
    sha256: String,
    bytes: u64,
    read_weight: u32,
    configured_read_weight: u32,
    source_weighting: WeightSourceWeighting,
    validation_bytes_per_second: u64,
    coverage: WeightSourceCoverage,
    read_strategy: WeightReadStrategy,
    representation: WeightSourceRepresentation,
    lossless: Option<lossless::LosslessSourceState>,
    replicas: Vec<WeightStore>,
}

pub(crate) struct LoadedWeight {
    pub(crate) tensor: Tensor,
    pub(crate) source_index: usize,
    pub(crate) fell_back: bool,
}

#[cfg(target_os = "linux")]
pub(crate) struct VerifiedWeightCacheRange {
    pub(crate) path: PathBuf,
    pub(crate) absolute_offset: u64,
    pub(crate) bytes: u64,
}

/// Zeroizing raw bytes returned by an explicit storage read.
///
/// This type deliberately has no serialization implementation and its debug
/// representation never exposes tensor bytes, names, paths, or ranges.
pub struct TensorRead {
    bytes: Zeroizing<Vec<u8>>,
    storage: TensorStorageDescriptor,
    strategy: WeightReadStrategy,
    representation: WeightSourceRepresentation,
    source_index: usize,
    fell_back: bool,
}

impl TensorRead {
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub fn storage(&self) -> &TensorStorageDescriptor {
        &self.storage
    }

    pub fn strategy(&self) -> WeightReadStrategy {
        self.strategy
    }

    pub fn representation(&self) -> &WeightSourceRepresentation {
        &self.representation
    }

    pub fn source_index(&self) -> usize {
        self.source_index
    }

    pub fn fell_back(&self) -> bool {
        self.fell_back
    }
}

impl std::fmt::Debug for TensorRead {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorRead")
            .field("bytes", &self.bytes.len())
            .field("strategy", &self.strategy)
            .field("representation", &self.representation)
            .field("source_index", &self.source_index)
            .field("fell_back", &self.fell_back)
            .finish_non_exhaustive()
    }
}

impl WeightStore {
    pub fn open(root: impl AsRef<Path>, limits: &InferenceLimits) -> Result<Self> {
        Self::open_config(&WeightStoreConfig::new(root.as_ref()), limits)
    }

    /// Opens an integrity-pinned collection whose SafeTensors files use
    /// independently authenticated, seekable AES-256-GCM chunks.
    pub fn open_seekable_encrypted(
        source: SeekableEncryptedWeightSource,
        limits: &InferenceLimits,
    ) -> Result<Self> {
        Self::open_seekable_encrypted_with_cancellation(source, limits, &CancellationToken::new())
    }

    /// Opens and authenticates an encrypted collection with cooperative
    /// cancellation between bounded decryption chunks.
    pub fn open_seekable_encrypted_with_cancellation(
        source: SeekableEncryptedWeightSource,
        limits: &InferenceLimits,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        encrypted_collection::open_seekable_encrypted(source, limits, cancellation)
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
        config.primary.representation.validate()?;
        if !config.primary.representation.is_canonical() {
            return Err(PowerError::Config(
                "the primary weight source must use canonical SafeTensors".to_string(),
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
        let mut occupied_roots = primary.roots.clone();
        for replica_config in &config.replicas {
            replica_config.representation.validate()?;
            let replica = if replica_config.representation.is_canonical() {
                Self::open_single(replica_config, limits)?
            } else {
                lossless::open_lossless_source(replica_config, &primary, limits)?
            };
            if replica.roots.iter().any(|root| {
                occupied_roots
                    .iter()
                    .any(|existing| paths_overlap(root, existing))
            }) {
                return Err(PowerError::Config(format!(
                    "weight source '{}' duplicates or overlaps another configured source",
                    replica.root.display()
                )));
            }
            if replica.representation.is_canonical() {
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
            }
            occupied_roots.extend(replica.roots.iter().cloned());
            primary.replicas.push(replica);
        }
        primary.apply_source_weighting(config.source_weighting);
        Ok(primary)
    }

    fn open_single(config: &WeightSourceConfig, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        config.representation.validate()?;
        if !config.representation.is_canonical() {
            return Err(PowerError::Config(
                "canonical source opening received a compressed representation".to_string(),
            ));
        }
        if config.read_weight == 0 {
            return Err(PowerError::Config(
                "weight source read weight must be greater than zero".to_string(),
            ));
        }
        let roots = resolve_weight_roots(&config.root, &config.shard_roots)?;
        let discovered = discover_safetensors(&roots, limits.max_model_files)?;
        let paths = discovered
            .iter()
            .map(|file| file.path.clone())
            .collect::<Vec<_>>();
        let validation_started = Instant::now();
        let (sha256, bytes, files) =
            hash_files(&discovered, limits.max_model_bytes, config.read_strategy)?;
        let validation_bytes_per_second = bytes_per_second(bytes, validation_started.elapsed());

        let mut inventory = BTreeMap::new();
        let mut locations = BTreeMap::new();
        let mut readers = Vec::with_capacity(if config.read_strategy == WeightReadStrategy::Mmap {
            0
        } else {
            paths.len()
        });
        let mut io_block_size = 0_u64;
        for (file_index, (path, file)) in paths.iter().zip(files.iter()).enumerate() {
            let reader = WeightFileReader::open(path, file.bytes, config.read_strategy)?;
            io_block_size = io_block_size.max(reader.io_block_size());
            let indexed =
                index::index_file(&reader, file_index, file.bytes, limits.max_tensor_elements)?;
            for (name, location) in indexed.locations {
                let descriptor = TensorDescriptor {
                    name: name.clone(),
                    dtype: format!("{:?}", location.dtype).to_ascii_lowercase(),
                    shape: location.shape.clone(),
                    bytes: location.bytes,
                };
                if inventory.insert(name.clone(), descriptor).is_some()
                    || locations.insert(name.clone(), location).is_some()
                {
                    return Err(PowerError::InvalidFormat(format!(
                        "duplicate tensor name '{name}' appears in the model container"
                    )));
                }
            }
            if config.read_strategy != WeightReadStrategy::Mmap {
                readers.push(reader);
            }
        }
        let tensors = if config.read_strategy == WeightReadStrategy::Mmap {
            // SAFETY: every path was canonicalized, checked as a regular file
            // beneath one configured root, read completely while hashing, and
            // indexed under the configured tensor bounds. The store retains
            // the path collection for at least as long as the maps.
            Some(
                unsafe { MmapedSafetensors::multi(&paths) }.map_err(|error| {
                    PowerError::InvalidFormat(format!("failed to map SafeTensors weights: {error}"))
                })?,
            )
        } else {
            None
        };
        Ok(Self {
            root: roots[0].clone(),
            roots,
            paths,
            tensors,
            inventory,
            locations,
            readers,
            io_block_size,
            files,
            sha256,
            bytes,
            read_weight: config.read_weight,
            configured_read_weight: config.read_weight,
            source_weighting: WeightSourceWeighting::Configured,
            validation_bytes_per_second,
            coverage: config.coverage,
            read_strategy: config.read_strategy,
            representation: config.representation.clone(),
            lossless: None,
            replicas: Vec::new(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns every physical root in this one logical canonical source.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
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
            shard_roots: self.roots.iter().skip(1).cloned().collect(),
            read_weight: self.read_weight,
            configured_read_weight: self.configured_read_weight,
            source_weighting: self.source_weighting,
            validation_bytes_per_second: self.validation_bytes_per_second,
            coverage: WeightSourceCoverage::Complete,
            read_strategy: self.read_strategy,
            representation: self.representation.clone(),
            io_block_size: self.io_block_size(),
            verified_files: self.files.len(),
            verified_tensors: self.inventory.len(),
            verified_bytes: self.bytes,
        });
        sources.extend(self.replicas.iter().enumerate().map(|(index, replica)| {
            WeightSourceDescriptor {
                index: index.saturating_add(1),
                role: WeightSourceRole::Replica,
                root: replica.root.clone(),
                shard_roots: replica.roots.iter().skip(1).cloned().collect(),
                read_weight: replica.read_weight,
                configured_read_weight: replica.configured_read_weight,
                source_weighting: replica.source_weighting,
                validation_bytes_per_second: replica.validation_bytes_per_second,
                coverage: replica.coverage,
                read_strategy: replica.read_strategy,
                representation: replica.representation.clone(),
                io_block_size: replica.io_block_size(),
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

    pub fn storage_descriptor(&self, name: &str) -> Option<TensorStorageDescriptor> {
        self.locations.get(name).map(storage_descriptor)
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn verified_cache_ranges(
        &self,
        names: &[String],
    ) -> Result<Vec<VerifiedWeightCacheRange>> {
        let sources = std::iter::once(self).chain(self.replicas.iter());
        if sources
            .clone()
            .any(|source| source.representation.is_seekable_encrypted())
        {
            return Err(PowerError::BackendNotAvailable(
                "filesystem cache-range preparation is unavailable for encrypted weight sources"
                    .to_string(),
            ));
        }
        let mut ranges = Vec::new();
        for source in sources {
            for name in names {
                if !source.contains(name) {
                    continue;
                }
                let location = if source.lossless.is_some() {
                    lossless::physical_location(source, name).ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "lossless cache range is missing its verified physical record"
                                .to_string(),
                        )
                    })?
                } else {
                    source.locations.get(name).ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "canonical cache range is missing its verified tensor location"
                                .to_string(),
                        )
                    })?
                };
                let path = source.paths.get(location.file_index).ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "verified tensor range references an unknown source file".to_string(),
                    )
                })?;
                ranges.push(VerifiedWeightCacheRange {
                    path: path.clone(),
                    absolute_offset: location.absolute_offset,
                    bytes: location.bytes,
                });
            }
        }
        if ranges.is_empty() {
            return Err(PowerError::InvalidFormat(
                "cache preparation selected no verified tensor ranges".to_string(),
            ));
        }
        Ok(ranges)
    }

    pub fn files(&self) -> &[WeightFileDescriptor] {
        &self.files
    }

    pub(crate) fn verified_file_path(&self, relative_path: &str) -> Result<&Path> {
        let index = self
            .files
            .binary_search_by(|file| file.relative_path.as_str().cmp(relative_path))
            .map_err(|_| {
                PowerError::InvalidRequest(format!(
                    "verified weight collection does not contain file '{relative_path}'"
                ))
            })?;
        self.paths.get(index).map(PathBuf::as_path).ok_or_else(|| {
            PowerError::InvalidFormat(
                "verified weight file index lost its physical path".to_string(),
            )
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.inventory.contains_key(name)
    }

    fn io_block_size(&self) -> u64 {
        self.io_block_size
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

    /// Materializes one validated tensor while honoring cooperative
    /// cancellation between bounded positional-read chunks.
    pub fn load_tensor_with_cancellation(
        &self,
        name: &str,
        device: &RuntimeDevice,
        cancellation: &CancellationToken,
    ) -> Result<Tensor> {
        Ok(self
            .load_tracked_with_cancellation(name, device.tensor_device(), cancellation)?
            .tensor)
    }

    /// Explicitly reads the exact verified bytes for one tensor through the
    /// configured deterministic source route. The returned buffer zeroizes on
    /// drop and is never emitted through telemetry or execution receipts.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<TensorRead> {
        self.read_tensor_bytes_with_cancellation(name, &CancellationToken::new())
    }

    pub fn read_tensor_bytes_with_cancellation(
        &self,
        name: &str,
        cancellation: &CancellationToken,
    ) -> Result<TensorRead> {
        let source_index = self.select_source(name);
        if source_index == 0 {
            return self
                .read_local_bytes(name, cancellation)
                .map(|(bytes, storage)| TensorRead {
                    bytes,
                    storage,
                    strategy: self.read_strategy,
                    representation: self.representation.clone(),
                    source_index,
                    fell_back: false,
                });
        }

        let replica = &self.replicas[source_index - 1];
        match replica.read_local_bytes(name, cancellation) {
            Ok((bytes, storage)) => Ok(TensorRead {
                bytes,
                storage,
                strategy: replica.read_strategy,
                representation: replica.representation.clone(),
                source_index,
                fell_back: false,
            }),
            Err(replica_error) if cancellation.is_cancelled() => Err(replica_error),
            Err(replica_error) => self
                .read_local_bytes(name, cancellation)
                .map(|(bytes, storage)| TensorRead {
                    bytes,
                    storage,
                    strategy: self.read_strategy,
                    representation: self.representation.clone(),
                    source_index: 0,
                    fell_back: true,
                })
                .map_err(|primary_error| {
                    PowerError::InvalidFormat(format!(
                        "failed to read a verified tensor range from replica {source_index} ({replica_error}) and primary ({primary_error})"
                    ))
                }),
        }
    }

    pub(crate) fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        Ok(self
            .load_tracked_with_cancellation(name, device, &CancellationToken::new())?
            .tensor)
    }

    #[cfg(test)]
    pub(crate) fn load_tracked(&self, name: &str, device: &Device) -> Result<LoadedWeight> {
        self.load_tracked_with_cancellation(name, device, &CancellationToken::new())
    }

    pub(crate) fn load_tracked_with_cancellation(
        &self,
        name: &str,
        device: &Device,
        cancellation: &CancellationToken,
    ) -> Result<LoadedWeight> {
        let source_index = self.select_source(name);
        if source_index == 0 {
            return Ok(LoadedWeight {
                tensor: self.load_local(name, device, cancellation)?,
                source_index,
                fell_back: false,
            });
        }

        let replica = &self.replicas[source_index - 1];
        match replica.load_local(name, device, cancellation) {
            Ok(tensor) => Ok(LoadedWeight {
                tensor,
                source_index,
                fell_back: false,
            }),
            Err(replica_error) if cancellation.is_cancelled() => Err(replica_error),
            Err(replica_error) => self
                .load_local(name, device, cancellation)
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

    fn load_local(
        &self,
        name: &str,
        device: &Device,
        cancellation: &CancellationToken,
    ) -> Result<Tensor> {
        check_read_cancelled(cancellation)?;
        let tensor = if self.lossless.is_none() && self.read_strategy == WeightReadStrategy::Mmap {
            self.tensors
                .as_ref()
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "mmap weight source is missing its validated mapping".to_string(),
                    )
                })?
                .load(name, device)
                .map_err(|error| {
                    PowerError::InvalidFormat(format!(
                        "failed to load model tensor '{name}' through mmap: {error}"
                    ))
                })?
        } else {
            let (bytes, _) = self.read_local_bytes(name, cancellation)?;
            let location = self.locations.get(name).ok_or_else(|| {
                PowerError::InvalidFormat(format!("weight store does not contain tensor '{name}'"))
            })?;
            let view = TensorView::new(location.dtype, location.shape.clone(), bytes.as_slice())
                .map_err(|error| {
                    PowerError::InvalidFormat(format!(
                        "failed to reconstruct verified tensor '{name}': {error}"
                    ))
                })?;
            view.load(device).map_err(|error| {
                PowerError::InvalidFormat(format!(
                    "failed to materialize verified tensor '{name}': {error}"
                ))
            })?
        };
        check_read_cancelled(cancellation)?;
        Ok(tensor)
    }

    fn read_local_bytes(
        &self,
        name: &str,
        cancellation: &CancellationToken,
    ) -> Result<(Zeroizing<Vec<u8>>, TensorStorageDescriptor)> {
        check_read_cancelled(cancellation)?;
        if self.lossless.is_some() {
            return lossless::read_lossless_bytes(self, name, cancellation);
        }
        let location = self.locations.get(name).ok_or_else(|| {
            PowerError::InvalidFormat(format!("weight store does not contain tensor '{name}'"))
        })?;
        let bytes = self.read_physical_bytes(name, location, cancellation)?;
        check_read_cancelled(cancellation)?;
        Ok((bytes, storage_descriptor(location)))
    }

    fn read_physical_bytes(
        &self,
        name: &str,
        location: &TensorLocation,
        cancellation: &CancellationToken,
    ) -> Result<Zeroizing<Vec<u8>>> {
        check_read_cancelled(cancellation)?;
        let bytes = if self.read_strategy == WeightReadStrategy::Mmap {
            let view = self
                .tensors
                .as_ref()
                .ok_or_else(|| {
                    PowerError::InvalidFormat(
                        "mmap weight source is missing its validated mapping".to_string(),
                    )
                })?
                .get(name)
                .map_err(|error| {
                    PowerError::InvalidFormat(format!(
                        "failed to read model tensor '{name}' through mmap: {error}"
                    ))
                })?;
            if view.dtype() != location.dtype
                || view.shape() != location.shape
                || u64::try_from(view.data().len()).ok() != Some(location.bytes)
            {
                return Err(PowerError::InvalidFormat(format!(
                    "mmap tensor '{name}' does not match its verified range index"
                )));
            }
            Zeroizing::new(view.data().to_vec())
        } else {
            let reader = self.readers.get(location.file_index).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "tensor range references an unknown verified source file".to_string(),
                )
            })?;
            reader.read_range(
                self.read_strategy,
                location.absolute_offset,
                location.bytes,
                cancellation,
            )?
        };
        check_read_cancelled(cancellation)?;
        Ok(bytes)
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

fn storage_descriptor(location: &TensorLocation) -> TensorStorageDescriptor {
    TensorStorageDescriptor {
        file_index: location.file_index,
        absolute_offset: location.absolute_offset,
        bytes: location.bytes,
        dtype: format!("{:?}", location.dtype).to_ascii_lowercase(),
        shape: location.shape.clone(),
    }
}

fn check_read_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "weight read was cancelled".to_string(),
        ))
    } else {
        Ok(())
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

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left == right || left.starts_with(right) || right.starts_with(left)
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

impl std::fmt::Debug for WeightStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WeightStore")
            .field("tensor_count", &self.inventory.len())
            .field("file_count", &self.files.len())
            .field("sha256", &self.sha256)
            .field("bytes", &self.bytes)
            .field("read_strategy", &self.read_strategy)
            .field("representation", &self.representation)
            .field("source_count", &self.replicas.len().saturating_add(1))
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod lossless_tests;
#[cfg(test)]
mod tests;
