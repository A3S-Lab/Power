use std::collections::{BTreeMap, HashMap};

use safetensors::tensor::{Dtype, Metadata};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::range_io::WeightFileReader;

const SAFETENSORS_PREFIX_BYTES: u64 = 8;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TensorLocation {
    pub(super) file_index: usize,
    pub(super) absolute_offset: u64,
    pub(super) bytes: u64,
    pub(super) dtype: Dtype,
    pub(super) shape: Vec<usize>,
}

pub(super) struct IndexedFile {
    pub(super) locations: BTreeMap<String, TensorLocation>,
    pub(super) metadata: HashMap<String, String>,
}

pub(super) fn index_file(
    reader: &WeightFileReader,
    file_index: usize,
    verified_bytes: u64,
) -> Result<IndexedFile> {
    let cancellation = CancellationToken::new();
    let prefix = reader.read_range(
        super::WeightReadStrategy::PositionalBuffered,
        0,
        SAFETENSORS_PREFIX_BYTES,
        &cancellation,
    )?;
    let header_bytes = prefix
        .as_slice()
        .try_into()
        .map(u64::from_le_bytes)
        .map_err(|_| {
            PowerError::InvalidFormat("SafeTensors header prefix is invalid".to_string())
        })?;
    if header_bytes == 0 || header_bytes > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(PowerError::InvalidFormat(format!(
            "SafeTensors header declares {header_bytes} bytes, outside the supported bound"
        )));
    }
    let data_start = SAFETENSORS_PREFIX_BYTES
        .checked_add(header_bytes)
        .ok_or_else(|| {
            PowerError::InvalidFormat("SafeTensors header length overflowed".to_string())
        })?;
    if data_start > verified_bytes {
        return Err(PowerError::InvalidFormat(
            "SafeTensors header exceeds its verified file".to_string(),
        ));
    }
    let header = reader.read_range(
        super::WeightReadStrategy::PositionalBuffered,
        SAFETENSORS_PREFIX_BYTES,
        header_bytes,
        &cancellation,
    )?;
    let metadata: Metadata = serde_json::from_slice(header.as_slice()).map_err(|error| {
        PowerError::InvalidFormat(format!("failed to parse SafeTensors metadata: {error}"))
    })?;
    let data_bytes = u64::try_from(metadata.data_len()).map_err(|_| {
        PowerError::InvalidFormat("SafeTensors data length exceeds the supported range".to_string())
    })?;
    let expected_end = data_start.checked_add(data_bytes).ok_or_else(|| {
        PowerError::InvalidFormat("SafeTensors data range overflowed".to_string())
    })?;
    if expected_end != verified_bytes {
        return Err(PowerError::InvalidFormat(format!(
            "SafeTensors metadata covers {expected_end} bytes but the verified file contains {verified_bytes}"
        )));
    }

    let custom_metadata = metadata.metadata().clone().unwrap_or_default();
    let mut locations = BTreeMap::new();
    for (name, info) in metadata.tensors() {
        let relative_start = u64::try_from(info.data_offsets.0).map_err(|_| {
            PowerError::InvalidFormat(
                "SafeTensors tensor offset exceeds the supported range".to_string(),
            )
        })?;
        let relative_end = u64::try_from(info.data_offsets.1).map_err(|_| {
            PowerError::InvalidFormat(
                "SafeTensors tensor offset exceeds the supported range".to_string(),
            )
        })?;
        let absolute_offset = data_start.checked_add(relative_start).ok_or_else(|| {
            PowerError::InvalidFormat("SafeTensors tensor offset overflowed".to_string())
        })?;
        let bytes = relative_end.checked_sub(relative_start).ok_or_else(|| {
            PowerError::InvalidFormat("SafeTensors tensor range is invalid".to_string())
        })?;
        if locations
            .insert(
                name,
                TensorLocation {
                    file_index,
                    absolute_offset,
                    bytes,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                },
            )
            .is_some()
        {
            return Err(PowerError::InvalidFormat(
                "SafeTensors metadata contains a duplicate tensor name".to_string(),
            ));
        }
    }
    Ok(IndexedFile {
        locations,
        metadata: custom_metadata,
    })
}
