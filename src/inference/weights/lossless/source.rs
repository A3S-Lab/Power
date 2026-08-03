use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::safetensors::MmapedSafetensors;
use safetensors::tensor::Dtype;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};
use crate::inference::InferenceLimits;

use super::{LosslessRansNibbleTable, LosslessSourceState};
use crate::inference::weights::files::{discover_safetensors, hash_files};
use crate::inference::weights::range_io::WeightFileReader;
use crate::inference::weights::{
    bytes_per_second, index, storage_descriptor, TensorDescriptor, TensorStorageDescriptor,
    WeightFileDescriptor, WeightReadStrategy, WeightSourceConfig, WeightSourceWeighting,
    WeightStore,
};

struct VerifiedCollection {
    root: PathBuf,
    paths: Vec<PathBuf>,
    sha256: String,
    bytes: u64,
    files: Vec<WeightFileDescriptor>,
}

/// Computes the canonical collection digest for a prospective weight artifact.
///
/// The digest covers stable relative names, lengths, and complete file bytes in
/// lexical order. It does not parse or trust container metadata.
pub fn weight_collection_sha256(
    root: impl AsRef<Path>,
    limits: &InferenceLimits,
) -> Result<String> {
    Ok(verify_collection(root.as_ref(), limits, WeightReadStrategy::Mmap)?.sha256)
}

pub(in crate::inference::weights) fn open_lossless_source(
    config: &WeightSourceConfig,
    primary: &WeightStore,
    limits: &InferenceLimits,
) -> Result<WeightStore> {
    limits.validate()?;
    config.representation.validate()?;
    if config.read_weight == 0 {
        return Err(PowerError::Config(
            "weight source read weight must be greater than zero".to_string(),
        ));
    }
    let expected_artifact_sha256 = config.representation.artifact_sha256().ok_or_else(|| {
        PowerError::Config(
            "lossless source opening requires a compressed representation".to_string(),
        )
    })?;

    // The complete physical collection is pinned before SafeTensors metadata,
    // record lengths, or decoded allocation sizes are inspected.
    let validation_started = Instant::now();
    let collection = verify_collection(&config.root, limits, config.read_strategy)?;
    if collection.sha256 != expected_artifact_sha256 {
        return Err(PowerError::IntegrityCheckFailed {
            model: "lossless weight artifact".to_string(),
            expected: expected_artifact_sha256.to_string(),
            actual: collection.sha256,
        });
    }

    let tensors = if config.read_strategy == WeightReadStrategy::Mmap {
        // SAFETY: every path was canonicalized, checked as a regular file
        // beneath `root`, and read completely while verifying the pinned
        // collection digest. The returned store retains every path and map.
        Some(
            unsafe { MmapedSafetensors::multi(&collection.paths) }.map_err(|error| {
                PowerError::InvalidFormat(format!(
                    "failed to map lossless SafeTensors weights: {error}"
                ))
            })?,
        )
    } else {
        None
    };

    let mut inventory = BTreeMap::new();
    let mut canonical_locations = BTreeMap::new();
    let mut record_locations = BTreeMap::new();
    let mut tables = Vec::with_capacity(collection.paths.len());
    let mut readers = Vec::with_capacity(if config.read_strategy == WeightReadStrategy::Mmap {
        0
    } else {
        collection.paths.len()
    });
    let mut io_block_size = 0_u64;
    let scratch_limit = limits.max_state_bytes;

    for (file_index, (path, file)) in collection
        .paths
        .iter()
        .zip(collection.files.iter())
        .enumerate()
    {
        let reader = WeightFileReader::open(path, file.bytes, config.read_strategy)?;
        io_block_size = io_block_size.max(reader.io_block_size());
        let indexed = index::index_file(&reader, file_index, file.bytes)?;
        let table = LosslessRansNibbleTable::from_safetensors_metadata(&indexed.metadata)?;
        if indexed.locations.is_empty() {
            return Err(PowerError::InvalidFormat(
                "lossless SafeTensors shards must contain at least one record".to_string(),
            ));
        }

        for (name, record_location) in indexed.locations {
            if record_location.dtype != Dtype::U8
                || record_location.shape.as_slice()
                    != [usize::try_from(record_location.bytes).map_err(|_| {
                        PowerError::InvalidFormat(
                            "lossless record length exceeds the host address range".to_string(),
                        )
                    })?]
            {
                return Err(PowerError::InvalidFormat(format!(
                    "lossless record '{name}' must be a one-dimensional U8 tensor"
                )));
            }
            let canonical_descriptor = primary.inventory.get(&name).ok_or_else(|| {
                PowerError::InvalidFormat(format!(
                    "lossless source contains tensor '{name}' that is absent from the canonical primary"
                ))
            })?;
            let canonical_location = primary.locations.get(&name).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "canonical tensor inventory lost its verified location".to_string(),
                )
            })?;
            if record_location.bytes >= canonical_location.bytes {
                return Err(PowerError::InvalidFormat(format!(
                    "lossless record '{name}' is not smaller than its canonical tensor"
                )));
            }
            require_decode_scratch(
                record_location.bytes,
                canonical_location.bytes,
                scratch_limit,
            )?;
            require_admission_scratch(canonical_location.bytes, scratch_limit)?;
            if inventory
                .insert(name.clone(), canonical_descriptor.clone())
                .is_some()
                || canonical_locations
                    .insert(name.clone(), canonical_location.clone())
                    .is_some()
                || record_locations
                    .insert(name.clone(), record_location)
                    .is_some()
            {
                return Err(PowerError::InvalidFormat(format!(
                    "duplicate tensor name '{name}' appears in the lossless source"
                )));
            }
        }
        tables.push(table);
        if config.read_strategy != WeightReadStrategy::Mmap {
            readers.push(reader);
        }
    }

    validate_coverage(config, primary, &inventory)?;
    let mut store = WeightStore {
        root: collection.root,
        paths: collection.paths,
        tensors,
        inventory,
        locations: canonical_locations,
        readers,
        io_block_size,
        files: collection.files,
        sha256: collection.sha256,
        bytes: collection.bytes,
        read_weight: config.read_weight,
        configured_read_weight: config.read_weight,
        source_weighting: WeightSourceWeighting::Configured,
        validation_bytes_per_second: 0,
        coverage: config.coverage,
        read_strategy: config.read_strategy,
        representation: config.representation.clone(),
        lossless: Some(LosslessSourceState {
            record_locations,
            tables,
            scratch_limit,
        }),
        replicas: Vec::new(),
    };

    // Admission is all-or-nothing: every record must decode byte-exactly to
    // its canonical primary tensor before this source can enter routing.
    let cancellation = CancellationToken::new();
    let names = store.inventory.keys().cloned().collect::<Vec<_>>();
    for name in names {
        let (decoded, _) = read_lossless_bytes(&store, &name, &cancellation)?;
        let (canonical, _) = primary.read_local_bytes(&name, &cancellation)?;
        if !bytes_are_equal(decoded.as_slice(), canonical.as_slice()) {
            return Err(PowerError::IntegrityCheckFailed {
                model: format!("lossless tensor {name}"),
                expected: format!("{:x}", Sha256::digest(canonical.as_slice())),
                actual: format!("{:x}", Sha256::digest(decoded.as_slice())),
            });
        }
    }
    store.validation_bytes_per_second = bytes_per_second(store.bytes, validation_started.elapsed());
    Ok(store)
}

pub(in crate::inference::weights) fn read_lossless_bytes(
    store: &WeightStore,
    name: &str,
    cancellation: &CancellationToken,
) -> Result<(Zeroizing<Vec<u8>>, TensorStorageDescriptor)> {
    let state = store.lossless.as_ref().ok_or_else(|| {
        PowerError::InvalidFormat("lossless weight source is missing codec state".to_string())
    })?;
    let canonical_location = store.locations.get(name).ok_or_else(|| {
        PowerError::InvalidFormat(format!("weight store does not contain tensor '{name}'"))
    })?;
    let record_location = state.record_locations.get(name).ok_or_else(|| {
        PowerError::InvalidFormat(
            "lossless tensor is missing its verified physical record".to_string(),
        )
    })?;
    require_decode_scratch(
        record_location.bytes,
        canonical_location.bytes,
        state.scratch_limit,
    )?;
    let record = store.read_physical_bytes(name, record_location, cancellation)?;
    let table = state
        .tables
        .get(record_location.file_index)
        .ok_or_else(|| {
            PowerError::InvalidFormat(
                "lossless tensor references an unknown shard-local table".to_string(),
            )
        })?;
    let decoded = table.decode_record_with_cancellation(
        record.as_slice(),
        canonical_location.bytes,
        state.scratch_limit,
        cancellation,
    )?;
    Ok((decoded, storage_descriptor(record_location)))
}

#[cfg(target_os = "linux")]
pub(in crate::inference::weights) fn physical_location<'a>(
    store: &'a WeightStore,
    name: &str,
) -> Option<&'a index::TensorLocation> {
    store
        .lossless
        .as_ref()
        .and_then(|state| state.record_locations.get(name))
}

fn verify_collection(
    root: &Path,
    limits: &InferenceLimits,
    read_strategy: WeightReadStrategy,
) -> Result<VerifiedCollection> {
    limits.validate()?;
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
    let (sha256, bytes, files) = hash_files(&root, &paths, limits.max_model_bytes, read_strategy)?;
    Ok(VerifiedCollection {
        root,
        paths,
        sha256,
        bytes,
        files,
    })
}

fn validate_coverage(
    config: &WeightSourceConfig,
    primary: &WeightStore,
    inventory: &BTreeMap<String, TensorDescriptor>,
) -> Result<()> {
    if inventory.is_empty() {
        return Err(PowerError::InvalidFormat(
            "lossless weight source contains no records".to_string(),
        ));
    }
    match config.coverage {
        crate::inference::weights::WeightSourceCoverage::Complete => {
            if inventory.len() != primary.inventory.len()
                || primary
                    .inventory
                    .keys()
                    .any(|name| !inventory.contains_key(name))
            {
                return Err(PowerError::InvalidFormat(
                    "complete lossless source does not cover every canonical tensor".to_string(),
                ));
            }
        }
        crate::inference::weights::WeightSourceCoverage::Partial => {
            if inventory.len() >= primary.inventory.len() {
                return Err(PowerError::InvalidFormat(
                    "partial lossless source must be a proper non-empty canonical tensor subset"
                        .to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn require_decode_scratch(encoded: u64, decoded: u64, limit: u64) -> Result<()> {
    let required = encoded.checked_add(decoded).ok_or_else(|| {
        PowerError::InvalidFormat("lossless decode scratch arithmetic overflowed".to_string())
    })?;
    if required > limit {
        return Err(PowerError::InvalidFormat(format!(
            "lossless decode requires {required} encoded-plus-decoded bytes, exceeding the {limit} byte state limit"
        )));
    }
    Ok(())
}

fn require_admission_scratch(decoded: u64, limit: u64) -> Result<()> {
    let required = decoded.checked_mul(2).ok_or_else(|| {
        PowerError::InvalidFormat("lossless admission scratch arithmetic overflowed".to_string())
    })?;
    if required > limit {
        return Err(PowerError::InvalidFormat(format!(
            "lossless admission comparison requires {required} bytes, exceeding the {limit} byte state limit"
        )));
    }
    Ok(())
}

fn bytes_are_equal(first: &[u8], second: &[u8]) -> bool {
    if first.len() != second.len() {
        return false;
    }
    first
        .iter()
        .zip(second)
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}
