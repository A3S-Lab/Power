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
}

impl WeightStore {
    pub fn open(root: impl AsRef<Path>, limits: &InferenceLimits) -> Result<Self> {
        limits.validate()?;
        let root = std::fs::canonicalize(root.as_ref()).map_err(|error| {
            PowerError::InvalidFormat(format!(
                "failed to resolve model directory '{}': {error}",
                root.as_ref().display()
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
        self.tensors.load(name, device).map_err(|error| {
            PowerError::InvalidFormat(format!("failed to load model tensor '{name}': {error}"))
        })
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
            .finish_non_exhaustive()
    }
}
