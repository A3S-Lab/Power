use tokio_util::sync::CancellationToken;
use zeroize::Zeroizing;

use crate::error::{PowerError, Result};

use super::{
    check_read_cancelled, storage_descriptor, TensorStorageDescriptor, WeightReadStrategy,
    WeightSourceRepresentation, WeightStore,
};

/// Zeroizing bytes returned from a bounded range inside one verified tensor.
///
/// The range is relative to the tensor payload rather than the containing
/// SafeTensors file. The complete tensor storage descriptor remains available
/// so callers can bind the bytes to the verified tensor identity without
/// exposing source paths.
pub struct TensorRangeRead {
    bytes: Zeroizing<Vec<u8>>,
    storage: TensorStorageDescriptor,
    tensor_offset: u64,
    strategy: WeightReadStrategy,
    representation: WeightSourceRepresentation,
    source_index: usize,
    fell_back: bool,
}

impl TensorRangeRead {
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    pub fn storage(&self) -> &TensorStorageDescriptor {
        &self.storage
    }

    pub fn tensor_offset(&self) -> u64 {
        self.tensor_offset
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

impl std::fmt::Debug for TensorRangeRead {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorRangeRead")
            .field("bytes", &self.bytes.len())
            .field("tensor_offset", &self.tensor_offset)
            .field("strategy", &self.strategy)
            .field("representation", &self.representation)
            .field("source_index", &self.source_index)
            .field("fell_back", &self.fell_back)
            .finish_non_exhaustive()
    }
}

impl WeightStore {
    /// Reads a non-empty byte range relative to one verified tensor payload.
    ///
    /// This preserves the configured source selection, positional I/O,
    /// encrypted-source authentication, cancellation, and primary fallback
    /// semantics used by complete tensor reads. Lossless compressed sources
    /// reject subrange reads explicitly because they cannot satisfy the bound
    /// without decoding the complete tensor.
    pub fn read_tensor_range(
        &self,
        name: &str,
        tensor_offset: u64,
        bytes: u64,
    ) -> Result<TensorRangeRead> {
        self.read_tensor_range_with_cancellation(
            name,
            tensor_offset,
            bytes,
            &CancellationToken::new(),
        )
    }

    pub fn read_tensor_range_with_cancellation(
        &self,
        name: &str,
        tensor_offset: u64,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<TensorRangeRead> {
        self.validate_tensor_range(name, tensor_offset, bytes)?;
        let source_index = self.select_source(name);
        if source_index == 0 {
            return self
                .read_local_range(name, tensor_offset, bytes, cancellation)
                .map(|(bytes, storage)| TensorRangeRead {
                    bytes,
                    storage,
                    tensor_offset,
                    strategy: self.read_strategy,
                    representation: self.representation.clone(),
                    source_index,
                    fell_back: false,
                });
        }

        let replica = &self.replicas[source_index - 1];
        match replica.read_local_range(name, tensor_offset, bytes, cancellation) {
            Ok((bytes, storage)) => Ok(TensorRangeRead {
                bytes,
                storage,
                tensor_offset,
                strategy: replica.read_strategy,
                representation: replica.representation.clone(),
                source_index,
                fell_back: false,
            }),
            Err(replica_error) if cancellation.is_cancelled() => Err(replica_error),
            Err(replica_error) => self
                .read_local_range(name, tensor_offset, bytes, cancellation)
                .map(|(bytes, storage)| TensorRangeRead {
                    bytes,
                    storage,
                    tensor_offset,
                    strategy: self.read_strategy,
                    representation: self.representation.clone(),
                    source_index: 0,
                    fell_back: true,
                })
                .map_err(|primary_error| {
                    PowerError::InvalidFormat(format!(
                        "failed to read a verified tensor subrange from replica {source_index} ({replica_error}) and primary ({primary_error})"
                    ))
                }),
        }
    }

    fn read_local_range(
        &self,
        name: &str,
        tensor_offset: u64,
        bytes: u64,
        cancellation: &CancellationToken,
    ) -> Result<(Zeroizing<Vec<u8>>, TensorStorageDescriptor)> {
        check_read_cancelled(cancellation)?;
        if self.lossless.is_some() {
            return Err(PowerError::BackendNotAvailable(
                "verified tensor subrange reads are unavailable for lossless compressed sources"
                    .to_string(),
            ));
        }
        let location = self.locations.get(name).ok_or_else(|| {
            PowerError::InvalidFormat(format!("weight store does not contain tensor '{name}'"))
        })?;
        let end = tensor_offset.checked_add(bytes).ok_or_else(|| {
            PowerError::InvalidRequest("tensor subrange byte count overflowed".to_string())
        })?;
        if bytes == 0 || end > location.bytes {
            return Err(PowerError::InvalidRequest(format!(
                "tensor subrange [{tensor_offset}, {end}) exceeds the {0}-byte tensor payload",
                location.bytes
            )));
        }

        let range = if self.read_strategy == WeightReadStrategy::Mmap {
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
            let start = usize::try_from(tensor_offset).map_err(|_| {
                PowerError::InvalidRequest(
                    "tensor subrange offset exceeds the host address range".to_string(),
                )
            })?;
            let end = usize::try_from(end).map_err(|_| {
                PowerError::InvalidRequest(
                    "tensor subrange end exceeds the host address range".to_string(),
                )
            })?;
            Zeroizing::new(view.data()[start..end].to_vec())
        } else {
            let reader = self.readers.get(location.file_index).ok_or_else(|| {
                PowerError::InvalidFormat(
                    "tensor range references an unknown verified source file".to_string(),
                )
            })?;
            let absolute_offset = location
                .absolute_offset
                .checked_add(tensor_offset)
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "tensor subrange absolute offset overflowed".to_string(),
                    )
                })?;
            reader.read_range(self.read_strategy, absolute_offset, bytes, cancellation)?
        };
        check_read_cancelled(cancellation)?;
        Ok((range, storage_descriptor(location)))
    }

    fn validate_tensor_range(&self, name: &str, tensor_offset: u64, bytes: u64) -> Result<()> {
        let descriptor = self.inventory.get(name).ok_or_else(|| {
            PowerError::InvalidFormat(format!("weight store does not contain tensor '{name}'"))
        })?;
        let end = tensor_offset.checked_add(bytes).ok_or_else(|| {
            PowerError::InvalidRequest("tensor subrange byte count overflowed".to_string())
        })?;
        if bytes == 0 || end > descriptor.bytes {
            return Err(PowerError::InvalidRequest(format!(
                "tensor subrange [{tensor_offset}, {end}) exceeds the {0}-byte tensor payload",
                descriptor.bytes
            )));
        }
        Ok(())
    }
}
