use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

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

/// One exact copy of a weight collection and its relative read-bandwidth
/// weight. The weight affects placement only; every source must contain the
/// same bytes and therefore cannot change model semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightSourceConfig {
    pub root: PathBuf,
    pub read_weight: u32,
}

impl WeightSourceConfig {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_weight: 1,
        }
    }

    pub fn with_read_weight(mut self, read_weight: u32) -> Self {
        self.read_weight = read_weight;
        self
    }
}

/// Primary weight root plus optional byte-identical read-only replicas.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightStoreConfig {
    pub primary: WeightSourceConfig,
    pub replicas: Vec<WeightSourceConfig>,
}

impl WeightStoreConfig {
    pub fn new(primary_root: impl Into<PathBuf>) -> Self {
        Self {
            primary: WeightSourceConfig::new(primary_root),
            replicas: Vec::new(),
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
    pub read_weight: u32,
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

    /// Opens a primary SafeTensors collection and byte-identical read-only
    /// replicas. Tensor names are mapped to sources with a deterministic
    /// bandwidth-weighted hash, so demand and prefetch always use the same
    /// source. A recoverable replica load error falls back to the primary.
    pub fn open_config(config: &WeightStoreConfig, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        let source_count = config.replicas.len().saturating_add(1);
        if source_count > limits.max_weight_sources {
            return Err(PowerError::Config(format!(
                "weight store declares {source_count} sources, exceeding the {} source limit",
                limits.max_weight_sources
            )));
        }
        let mut primary =
            Self::open_single(&config.primary.root, config.primary.read_weight, limits)?;
        let mut roots = vec![primary.root.clone()];
        for replica_config in &config.replicas {
            let replica =
                Self::open_single(&replica_config.root, replica_config.read_weight, limits)?;
            if roots.contains(&replica.root) {
                return Err(PowerError::Config(format!(
                    "weight source '{}' is configured more than once",
                    replica.root.display()
                )));
            }
            if replica.sha256 != primary.sha256 {
                return Err(PowerError::IntegrityCheckFailed {
                    model: format!("weight replica {}", replica.root.display()),
                    expected: primary.sha256.clone(),
                    actual: replica.sha256,
                });
            }
            roots.push(replica.root.clone());
            primary.replicas.push(replica);
        }
        Ok(primary)
    }

    fn open_single(root: &Path, read_weight: u32, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        if read_weight == 0 {
            return Err(PowerError::Config(
                "weight source read weight must be greater than zero".to_string(),
            ));
        }
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
        let (sha256, bytes, files) = hash_files(&root, &paths, limits.max_model_bytes)?;

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
            read_weight,
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
        });
        sources.extend(self.replicas.iter().enumerate().map(|(index, replica)| {
            WeightSourceDescriptor {
                index: index.saturating_add(1),
                role: WeightSourceRole::Replica,
                root: replica.root.clone(),
                read_weight: replica.read_weight,
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
            .fold(u64::from(self.read_weight), |total, replica| {
                total.saturating_add(u64::from(replica.read_weight))
            });
        let digest = Sha256::digest(name.as_bytes());
        let mut prefix = [0_u8; 8];
        prefix.copy_from_slice(&digest[..8]);
        let mut slot = u64::from_le_bytes(prefix) % total_weight;
        if slot < u64::from(self.read_weight) {
            return 0;
        }
        slot -= u64::from(self.read_weight);
        for (index, replica) in self.replicas.iter().enumerate() {
            let weight = u64::from(replica.read_weight);
            if slot < weight {
                return index.saturating_add(1);
            }
            slot -= weight;
        }
        0
    }
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
mod tests {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    use super::*;

    fn write_weights(root: &Path, byte: u8) {
        let values = [byte; 4];
        let tensors = (0..128)
            .map(|index| {
                (
                    format!("layer.0.weight.{index}"),
                    TensorView::new(Dtype::F32, vec![1], values.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(tensors, None, &root.join("model.safetensors")).unwrap();
    }

    #[test]
    fn weighted_replicas_are_exact_and_deterministic() {
        let primary = tempfile::tempdir().unwrap();
        let replica = tempfile::tempdir().unwrap();
        write_weights(primary.path(), 1);
        std::fs::copy(
            primary.path().join("model.safetensors"),
            replica.path().join("model.safetensors"),
        )
        .unwrap();
        let config = WeightStoreConfig::new(primary.path())
            .with_primary_read_weight(1)
            .with_replica(WeightSourceConfig::new(replica.path()).with_read_weight(3));

        let store = WeightStore::open_config(&config, &InferenceLimits::default()).unwrap();
        assert_eq!(store.sources().len(), 2);
        assert_eq!(store.sources()[1].role, WeightSourceRole::Replica);
        let selected = store
            .inventory()
            .map(|descriptor| store.select_source(&descriptor.name))
            .collect::<Vec<_>>();
        assert!(selected.contains(&0));
        assert!(selected.contains(&1));
        for descriptor in store.inventory() {
            let first = store.select_source(&descriptor.name);
            let second = store.select_source(&descriptor.name);
            assert_eq!(first, second);
            let loaded = store.load_tracked(&descriptor.name, &Device::Cpu).unwrap();
            assert_eq!(loaded.source_index, first);
            assert!(!loaded.fell_back);
        }
    }

    #[test]
    fn replica_digest_mismatch_fails_closed() {
        let primary = tempfile::tempdir().unwrap();
        let replica = tempfile::tempdir().unwrap();
        write_weights(primary.path(), 1);
        write_weights(replica.path(), 2);
        let config = WeightStoreConfig::new(primary.path())
            .with_replica(WeightSourceConfig::new(replica.path()));

        assert!(matches!(
            WeightStore::open_config(&config, &InferenceLimits::default()),
            Err(PowerError::IntegrityCheckFailed { .. })
        ));
    }

    #[test]
    fn duplicate_or_zero_weight_sources_are_rejected() {
        let primary = tempfile::tempdir().unwrap();
        write_weights(primary.path(), 1);
        let duplicate = WeightStoreConfig::new(primary.path())
            .with_replica(WeightSourceConfig::new(primary.path()));
        assert!(WeightStore::open_config(&duplicate, &InferenceLimits::default()).is_err());

        let zero = WeightStoreConfig::new(primary.path()).with_primary_read_weight(0);
        assert!(WeightStore::open_config(&zero, &InferenceLimits::default()).is_err());

        let limited = InferenceLimits {
            max_weight_sources: 1,
            ..InferenceLimits::default()
        };
        assert!(WeightStore::open_config(&duplicate, &limited).is_err());
    }
}
