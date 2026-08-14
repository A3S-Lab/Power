use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::range_io::WeightFileReader;
use super::{
    bytes_per_second, encrypt_seekable_weight_file, index, SeekableEncryptedFile,
    SeekableWeightKey, TensorDescriptor, WeightFileDescriptor, WeightReadStrategy,
    WeightSourceCoverage, WeightSourceRepresentation, WeightSourceWeighting, WeightStore,
};

pub const ENCRYPTED_WEIGHT_MANIFEST_SCHEMA: &str = "a3s.power.seekable-encrypted-weights.v1";
pub const ENCRYPTED_WEIGHT_MANIFEST_FILE: &str = "encrypted-weights.json";

const MAX_MANIFEST_BYTES: u64 = 4 * 1024 * 1024;

/// One logical SafeTensors file and its authenticated ciphertext container.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeekableEncryptedWeightFile {
    pub logical_path: String,
    pub encrypted_path: String,
    pub plaintext_bytes: u64,
    pub plaintext_sha256: String,
    pub ciphertext_bytes: u64,
    pub ciphertext_sha256: String,
    pub chunks: u64,
}

/// Integrity manifest for a complete logical SafeTensors collection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeekableEncryptedWeightManifest {
    pub schema: String,
    pub plaintext_sha256: String,
    pub chunk_bytes: u32,
    pub files: Vec<SeekableEncryptedWeightFile>,
}

/// Reproducible properties of a completed encryption operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SeekableEncryptedWeightCollectionReport {
    pub destination: PathBuf,
    pub manifest_sha256: String,
    pub plaintext_sha256: String,
    pub plaintext_bytes: u64,
    pub ciphertext_bytes: u64,
    pub files: usize,
    pub chunk_bytes: u32,
    pub peak_plaintext_chunk_bytes: u64,
}

/// Typed trust anchor and key used to open an encrypted weight collection.
#[derive(Clone)]
pub struct SeekableEncryptedWeightSource {
    root: PathBuf,
    manifest_sha256: String,
    key: SeekableWeightKey,
}

impl SeekableEncryptedWeightSource {
    pub fn new(
        root: impl Into<PathBuf>,
        manifest_sha256: impl Into<String>,
        key: SeekableWeightKey,
    ) -> Result<Self> {
        let manifest_sha256 = manifest_sha256.into();
        validate_sha256(&manifest_sha256, "encrypted weight manifest")?;
        Ok(Self {
            root: root.into(),
            manifest_sha256,
            key,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

impl std::fmt::Debug for SeekableEncryptedWeightSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SeekableEncryptedWeightSource")
            .field("root", &self.root)
            .field("manifest_sha256", &self.manifest_sha256)
            .field("key", &self.key)
            .finish()
    }
}

pub fn encrypt_seekable_weight_collection(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    key: &SeekableWeightKey,
    chunk_bytes: u32,
    limits: &InferenceLimits,
) -> Result<SeekableEncryptedWeightCollectionReport> {
    let store = WeightStore::open(source, limits)?;
    let destination = absolute_destination(destination.as_ref())?;
    if destination.exists() {
        return Err(PowerError::Config(
            "encrypted weight collection destination already exists".to_string(),
        ));
    }
    let parent = destination.parent().ok_or_else(|| {
        PowerError::Config("encrypted weight collection destination has no parent".to_string())
    })?;
    std::fs::create_dir_all(parent)?;
    let parent = std::fs::canonicalize(parent)?;
    let destination = parent.join(destination.file_name().ok_or_else(|| {
        PowerError::Config("encrypted weight collection destination has no name".to_string())
    })?);
    if destination.starts_with(store.root()) || store.root().starts_with(&destination) {
        return Err(PowerError::Config(
            "encrypted weight destination must not overlap the plaintext source".to_string(),
        ));
    }

    let staging = tempfile::Builder::new()
        .prefix(".a3s-power-encrypted-weights-")
        .tempdir_in(&parent)?;
    let mut files = Vec::with_capacity(store.files().len());
    let mut plaintext_bytes = 0_u64;
    let mut ciphertext_bytes = 0_u64;
    let mut peak_plaintext_chunk_bytes = 0_u64;
    for file in store.files() {
        let encrypted_path = format!("{}.a3se", file.relative_path);
        let output = resolve_relative(staging.path(), &encrypted_path, false)?;
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let verification = encrypt_seekable_weight_file(
            store.verified_file_path(&file.relative_path)?,
            &output,
            key,
            chunk_bytes,
        )?;
        if verification.descriptor.plaintext_bytes != file.bytes
            || verification.descriptor.plaintext_sha256 != file.sha256
        {
            return Err(PowerError::IntegrityCheckFailed {
                model: "encrypted weight source".to_string(),
                expected: file.sha256.clone(),
                actual: verification.descriptor.plaintext_sha256,
            });
        }
        plaintext_bytes = plaintext_bytes
            .checked_add(verification.descriptor.plaintext_bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat("plaintext byte count overflowed".to_string())
            })?;
        ciphertext_bytes = ciphertext_bytes
            .checked_add(verification.descriptor.ciphertext_bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat("ciphertext byte count overflowed".to_string())
            })?;
        peak_plaintext_chunk_bytes =
            peak_plaintext_chunk_bytes.max(verification.peak_plaintext_chunk_bytes);
        files.push(SeekableEncryptedWeightFile {
            logical_path: file.relative_path.clone(),
            encrypted_path,
            plaintext_bytes: verification.descriptor.plaintext_bytes,
            plaintext_sha256: verification.descriptor.plaintext_sha256,
            ciphertext_bytes: verification.descriptor.ciphertext_bytes,
            ciphertext_sha256: verification.ciphertext_sha256,
            chunks: verification.descriptor.chunks,
        });
    }
    let manifest = SeekableEncryptedWeightManifest {
        schema: ENCRYPTED_WEIGHT_MANIFEST_SCHEMA.to_string(),
        plaintext_sha256: store.sha256().to_string(),
        chunk_bytes,
        files,
    };
    let mut manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    manifest_bytes.push(b'\n');
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
    std::fs::write(
        staging.path().join(ENCRYPTED_WEIGHT_MANIFEST_FILE),
        &manifest_bytes,
    )?;
    let report = SeekableEncryptedWeightCollectionReport {
        destination: destination.clone(),
        manifest_sha256,
        plaintext_sha256: store.sha256().to_string(),
        plaintext_bytes,
        ciphertext_bytes,
        files: store.files().len(),
        chunk_bytes,
        peak_plaintext_chunk_bytes,
    };
    drop(store);
    let staged_path = staging.keep();
    if let Err(error) = std::fs::rename(&staged_path, &destination) {
        let _ = std::fs::remove_dir_all(&staged_path);
        return Err(PowerError::Io(error));
    }
    Ok(report)
}

pub(super) fn open_seekable_encrypted(
    source: SeekableEncryptedWeightSource,
    limits: &InferenceLimits,
    cancellation: &CancellationToken,
) -> Result<WeightStore> {
    limits.validate()?;
    check_cancelled(cancellation)?;
    let root = canonical_directory(&source.root)?;
    let manifest_path = root.join(ENCRYPTED_WEIGHT_MANIFEST_FILE);
    let metadata = std::fs::symlink_metadata(&manifest_path)?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > MAX_MANIFEST_BYTES
    {
        return Err(PowerError::InvalidFormat(
            "encrypted weight manifest must be a bounded regular non-symlink file".to_string(),
        ));
    }
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_bytes));
    if manifest_sha256 != source.manifest_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "encrypted weight manifest".to_string(),
            expected: source.manifest_sha256,
            actual: manifest_sha256,
        });
    }
    let manifest: SeekableEncryptedWeightManifest = serde_json::from_slice(&manifest_bytes)?;
    validate_manifest(&manifest, limits)?;

    let started = Instant::now();
    let mut collection_hasher = Sha256::new();
    let mut paths = Vec::with_capacity(manifest.files.len());
    let mut readers = Vec::with_capacity(manifest.files.len());
    let mut files = Vec::with_capacity(manifest.files.len());
    let mut total_bytes = 0_u64;
    let mut expected_artifacts = BTreeSet::new();
    expected_artifacts.insert(std::fs::canonicalize(&manifest_path)?);
    for entry in &manifest.files {
        let path = resolve_relative(&root, &entry.encrypted_path, true)?;
        expected_artifacts.insert(path.clone());
        let encrypted = SeekableEncryptedFile::open(&path, &source.key)?;
        let descriptor = encrypted.descriptor();
        if descriptor.plaintext_bytes != entry.plaintext_bytes
            || descriptor.plaintext_sha256 != entry.plaintext_sha256
            || descriptor.ciphertext_bytes != entry.ciphertext_bytes
            || descriptor.chunk_bytes != manifest.chunk_bytes
            || descriptor.chunks != entry.chunks
        {
            return Err(PowerError::InvalidFormat(
                "encrypted weight file does not match its pinned manifest entry".to_string(),
            ));
        }
        collection_hasher.update((entry.logical_path.len() as u64).to_le_bytes());
        collection_hasher.update(entry.logical_path.as_bytes());
        collection_hasher.update(entry.plaintext_bytes.to_le_bytes());
        let verification = encrypted.verify_into(&mut collection_hasher, cancellation)?;
        if verification.ciphertext_sha256 != entry.ciphertext_sha256 {
            return Err(PowerError::IntegrityCheckFailed {
                model: "encrypted weight ciphertext".to_string(),
                expected: entry.ciphertext_sha256.clone(),
                actual: verification.ciphertext_sha256,
            });
        }
        total_bytes = total_bytes
            .checked_add(entry.plaintext_bytes)
            .ok_or_else(|| {
                PowerError::InvalidFormat("encrypted weight byte count overflowed".to_string())
            })?;
        files.push(WeightFileDescriptor {
            relative_path: entry.logical_path.clone(),
            bytes: entry.plaintext_bytes,
            sha256: entry.plaintext_sha256.clone(),
        });
        paths.push(path);
        readers.push(WeightFileReader::from_encrypted(encrypted));
    }
    verify_artifact_inventory(&root, &expected_artifacts, limits.max_model_files)?;
    let plaintext_sha256 = format!("{:x}", collection_hasher.finalize());
    if plaintext_sha256 != manifest.plaintext_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "encrypted weight plaintext collection".to_string(),
            expected: manifest.plaintext_sha256,
            actual: plaintext_sha256,
        });
    }

    let mut inventory = BTreeMap::new();
    let mut locations = BTreeMap::new();
    for (file_index, (reader, file)) in readers.iter().zip(files.iter()).enumerate() {
        let indexed =
            index::index_file(reader, file_index, file.bytes, limits.max_tensor_elements)?;
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
                    "duplicate tensor name '{name}' appears in encrypted weights"
                )));
            }
        }
    }
    Ok(WeightStore {
        root: root.clone(),
        roots: vec![root],
        paths,
        tensors: None,
        inventory,
        locations,
        readers,
        io_block_size: 0,
        files,
        sha256: plaintext_sha256,
        bytes: total_bytes,
        read_weight: 1,
        configured_read_weight: 1,
        source_weighting: WeightSourceWeighting::Configured,
        validation_bytes_per_second: bytes_per_second(total_bytes, started.elapsed()),
        coverage: WeightSourceCoverage::Complete,
        read_strategy: WeightReadStrategy::PositionalBuffered,
        representation: WeightSourceRepresentation::SeekableAes256GcmV1 {
            manifest_sha256: source.manifest_sha256().to_string(),
        },
        lossless: None,
        replicas: Vec::new(),
    })
}

fn validate_manifest(
    manifest: &SeekableEncryptedWeightManifest,
    limits: &InferenceLimits,
) -> Result<()> {
    if manifest.schema != ENCRYPTED_WEIGHT_MANIFEST_SCHEMA {
        return Err(PowerError::InvalidFormat(
            "encrypted weight manifest schema is unsupported".to_string(),
        ));
    }
    validate_sha256(&manifest.plaintext_sha256, "encrypted plaintext collection")?;
    if manifest.files.is_empty() || manifest.files.len() > limits.max_model_files {
        return Err(PowerError::InvalidFormat(format!(
            "encrypted weight manifest must contain 1..={} files",
            limits.max_model_files
        )));
    }
    let mut previous = None;
    let mut total = 0_u64;
    for file in &manifest.files {
        validate_relative(&file.logical_path)?;
        validate_relative(&file.encrypted_path)?;
        if !file.logical_path.ends_with(".safetensors")
            || file.encrypted_path != format!("{}.a3se", file.logical_path)
        {
            return Err(PowerError::InvalidFormat(
                "encrypted weight paths must map '<logical>.safetensors' to '<logical>.safetensors.a3se'"
                    .to_string(),
            ));
        }
        if previous.is_some_and(|value: &str| value >= file.logical_path.as_str()) {
            return Err(PowerError::InvalidFormat(
                "encrypted weight manifest files must be unique and sorted".to_string(),
            ));
        }
        previous = Some(file.logical_path.as_str());
        validate_sha256(&file.plaintext_sha256, "encrypted weight plaintext")?;
        validate_sha256(&file.ciphertext_sha256, "encrypted weight ciphertext")?;
        if file.plaintext_bytes == 0 || file.ciphertext_bytes == 0 || file.chunks == 0 {
            return Err(PowerError::InvalidFormat(
                "encrypted weight file lengths and chunk count must be non-zero".to_string(),
            ));
        }
        total = total.checked_add(file.plaintext_bytes).ok_or_else(|| {
            PowerError::InvalidFormat("encrypted weight byte count overflowed".to_string())
        })?;
        if total > limits.max_model_bytes {
            return Err(PowerError::InvalidFormat(format!(
                "encrypted weights exceed the {} byte model limit",
                limits.max_model_bytes
            )));
        }
    }
    Ok(())
}

fn canonical_directory(root: &Path) -> Result<PathBuf> {
    let metadata = std::fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PowerError::InvalidFormat(
            "encrypted weight root must be a regular non-symlink directory".to_string(),
        ));
    }
    Ok(std::fs::canonicalize(root)?)
}

fn resolve_relative(root: &Path, relative: &str, must_exist: bool) -> Result<PathBuf> {
    validate_relative(relative)?;
    let mut path = root.to_path_buf();
    for component in relative.split(['/', '\\']) {
        path.push(component);
    }
    if !must_exist {
        return Ok(path);
    }
    let metadata = std::fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PowerError::InvalidFormat(
            "encrypted weight artifact must be a regular non-symlink file".to_string(),
        ));
    }
    let canonical = std::fs::canonicalize(path)?;
    if !canonical.starts_with(root) {
        return Err(PowerError::InvalidFormat(
            "encrypted weight artifact escapes its collection root".to_string(),
        ));
    }
    Ok(canonical)
}

fn validate_relative(relative: &str) -> Result<()> {
    if relative.is_empty()
        || relative.split(['/', '\\']).any(|component| {
            component.is_empty() || component == "." || component == ".." || component.contains(':')
        })
    {
        return Err(PowerError::InvalidFormat(
            "encrypted weight manifest contains an unsafe relative path".to_string(),
        ));
    }
    Ok(())
}

fn verify_artifact_inventory(
    root: &Path,
    expected: &BTreeSet<PathBuf>,
    max_model_files: usize,
) -> Result<()> {
    let mut actual = BTreeSet::new();
    let mut pending = vec![root.to_path_buf()];
    let mut visited = 0_usize;
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            visited = visited.saturating_add(1);
            if visited > max_model_files.saturating_mul(4).saturating_add(1) {
                return Err(PowerError::InvalidFormat(
                    "encrypted weight collection contains too many filesystem entries".to_string(),
                ));
            }
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(PowerError::InvalidFormat(
                    "encrypted weight collection must not contain symbolic links".to_string(),
                ));
            }
            if file_type.is_dir() {
                pending.push(entry.path());
            } else if file_type.is_file() {
                actual.insert(std::fs::canonicalize(entry.path())?);
                if actual.len() > max_model_files.saturating_add(1) {
                    return Err(PowerError::InvalidFormat(
                        "encrypted weight collection contains too many artifacts".to_string(),
                    ));
                }
            } else {
                return Err(PowerError::InvalidFormat(
                    "encrypted weight collection contains an unsupported artifact".to_string(),
                ));
            }
        }
    }
    if actual != *expected {
        return Err(PowerError::InvalidFormat(
            "encrypted weight artifact inventory does not match its manifest".to_string(),
        ));
    }
    Ok(())
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "encrypted weight collection verification was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PowerError::InvalidFormat(format!(
            "{label} SHA-256 must contain 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn absolute_destination(destination: &Path) -> Result<PathBuf> {
    if destination.as_os_str().is_empty() {
        return Err(PowerError::Config(
            "encrypted weight destination must not be empty".to_string(),
        ));
    }
    if destination.is_absolute() {
        Ok(destination.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(destination))
    }
}
