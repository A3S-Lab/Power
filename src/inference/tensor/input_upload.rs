use candle_core::{Device, Tensor};

#[cfg(feature = "embedded-cuda")]
use std::collections::{BTreeMap, VecDeque};
#[cfg(feature = "embedded-cuda")]
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

#[cfg(feature = "embedded-cuda")]
use candle_core::{
    cuda_backend::cudarc::driver::{CudaStream, PinnedHostSlice},
    CudaDevice, CudaStorage, Storage,
};

use crate::error::{PowerError, Result};

/// Request-scoped, byte-bounded storage for completed CUDA input uploads.
///
/// A model execution permit owns one pool, so page-locked allocations can be
/// reused by later graph calls in the same admitted request but cannot remain
/// resident after that request ends. Exact-size reuse avoids copying padding;
/// least-recently-used idle allocations are evicted before the declared input
/// byte bound can be exceeded.
#[derive(Debug, Clone)]
pub(crate) struct InputUploadPool {
    #[cfg(feature = "embedded-cuda")]
    max_retained_bytes: usize,
    #[cfg(feature = "embedded-cuda")]
    state: Arc<Mutex<CudaInputUploadPoolState>>,
}

impl InputUploadPool {
    pub(crate) fn new(max_retained_bytes: usize) -> Self {
        #[cfg(not(feature = "embedded-cuda"))]
        let _ = max_retained_bytes;
        Self {
            #[cfg(feature = "embedded-cuda")]
            max_retained_bytes,
            #[cfg(feature = "embedded-cuda")]
            state: Arc::new(Mutex::new(CudaInputUploadPoolState::default())),
        }
    }

    #[cfg(feature = "embedded-cuda")]
    fn acquire(&self, cuda: &CudaDevice, elements: usize) -> Option<CudaInputUpload> {
        if let Some(pinned) = self.take_exact(elements) {
            return Some(CudaInputUpload {
                pool: self.clone(),
                pinned: Some(pinned),
                reusable: true,
            });
        }

        let stream = cuda.cuda_stream();
        // SAFETY: `try_materialize_pinned` initializes every requested
        // element before submitting a copy, and the returned upload lease
        // retains the allocation until cudarc's built-in event completes.
        let pinned = unsafe { stream.context().alloc_pinned::<f32>(elements) }.ok()?;
        #[cfg(test)]
        {
            self.state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .allocations += 1;
        }
        Some(CudaInputUpload {
            pool: self.clone(),
            pinned: Some(pinned),
            reusable: true,
        })
    }

    #[cfg(feature = "embedded-cuda")]
    fn take_exact(&self, elements: usize) -> Option<PinnedHostSlice<f32>> {
        let mut state = self.try_lock()?;
        let pinned = state.idle.remove(&elements)?;
        state.retained_bytes = state.retained_bytes.saturating_sub(pinned.num_bytes());
        if let Some(position) = state.lru.iter().position(|retained| *retained == elements) {
            state.lru.remove(position);
        }
        Some(pinned)
    }

    #[cfg(feature = "embedded-cuda")]
    fn retain(&self, pinned: PinnedHostSlice<f32>) {
        let bytes = pinned.num_bytes();
        if bytes > self.max_retained_bytes {
            return;
        }
        let Some(mut state) = self.try_lock() else {
            return;
        };
        let elements = pinned.len();
        if state.idle.contains_key(&elements) {
            return;
        }

        let mut evicted = Vec::new();
        while state
            .retained_bytes
            .checked_add(bytes)
            .is_none_or(|total| total > self.max_retained_bytes)
        {
            let Some(oldest) = state.lru.pop_front() else {
                break;
            };
            if let Some(allocation) = state.idle.remove(&oldest) {
                state.retained_bytes = state.retained_bytes.saturating_sub(allocation.num_bytes());
                evicted.push(allocation);
            }
        }
        if state
            .retained_bytes
            .checked_add(bytes)
            .is_some_and(|total| total <= self.max_retained_bytes)
        {
            state.retained_bytes += bytes;
            state.lru.push_back(elements);
            state.idle.insert(elements, pinned);
        }
        drop(state);
        drop(evicted);
    }

    #[cfg(feature = "embedded-cuda")]
    fn try_lock(&self) -> Option<MutexGuard<'_, CudaInputUploadPoolState>> {
        match self.state.try_lock() {
            Ok(state) => Some(state),
            Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    #[cfg(all(test, feature = "embedded-cuda"))]
    pub(super) fn allocation_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocations
    }

    #[cfg(all(test, feature = "embedded-cuda"))]
    pub(super) fn retained_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retained_bytes
    }
}

#[cfg(feature = "embedded-cuda")]
#[derive(Debug, Default)]
struct CudaInputUploadPoolState {
    idle: BTreeMap<usize, PinnedHostSlice<f32>>,
    lru: VecDeque<usize>,
    retained_bytes: usize,
    #[cfg(test)]
    allocations: usize,
}

/// Owns host storage until every asynchronous input copy that references it
/// has completed.
#[derive(Debug, Default)]
pub(crate) struct InputUploadGuard {
    #[cfg(feature = "embedded-cuda")]
    uploads: Vec<CudaInputUpload>,
}

impl InputUploadGuard {
    /// Ends the input-upload lifetime after the caller has materialized every
    /// dependent output. CUDA builds synchronize and recycle pinned storage
    /// through the owned upload values' `Drop` implementations; CPU builds
    /// consume the same typed lifecycle boundary without a fake resource.
    pub(crate) fn complete(self) {}

    #[cfg(feature = "embedded-cuda")]
    pub(super) fn append(&mut self, mut other: Self) {
        self.uploads.append(&mut other.uploads);
    }

    #[cfg(not(feature = "embedded-cuda"))]
    pub(super) fn append(&mut self, _other: Self) {}

    #[cfg(test)]
    pub(super) fn pinned_upload_count(&self) -> usize {
        #[cfg(feature = "embedded-cuda")]
        {
            self.uploads.len()
        }
        #[cfg(not(feature = "embedded-cuda"))]
        {
            0
        }
    }
}

pub(super) fn materialize(
    values: Vec<f32>,
    shape: Vec<usize>,
    device: &Device,
    pool: &InputUploadPool,
    max_pinned_bytes: usize,
    label: &str,
) -> Result<(Tensor, InputUploadGuard)> {
    #[cfg(not(feature = "embedded-cuda"))]
    let _ = (pool, max_pinned_bytes);

    #[cfg(feature = "embedded-cuda")]
    if let Device::Cuda(cuda) = device {
        let bounded = values
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .is_some_and(|bytes| bytes <= max_pinned_bytes);
        if bounded && !values.is_empty() {
            if let Some(materialized) = try_materialize_pinned(&values, &shape, cuda, pool, label)?
            {
                return Ok(materialized);
            }
        }
    }

    let tensor = Tensor::from_vec(values, shape.as_slice(), device).map_err(|error| {
        PowerError::InferenceFailed(format!("failed to materialize {label}: {error}"))
    })?;
    Ok((tensor, InputUploadGuard::default()))
}

#[cfg(feature = "embedded-cuda")]
#[derive(Debug)]
struct CudaInputUpload {
    pool: InputUploadPool,
    pinned: Option<PinnedHostSlice<f32>>,
    reusable: bool,
}

#[cfg(feature = "embedded-cuda")]
impl Drop for CudaInputUpload {
    fn drop(&mut self) {
        let Some(mut pinned) = self.pinned.take() else {
            return;
        };
        let context = pinned.context().clone();
        // A deferred error would otherwise make CudaEvent::synchronize return
        // before issuing its driver synchronization. Clear it, perform the
        // lifetime-critical wait, and then restore both errors for the next
        // fallible CUDA boundary.
        let deferred = context.check_err();
        let completed = pinned.as_mut_slice().map(|_| ());
        let completed_successfully = completed.is_ok();
        context.record_err(deferred);
        context.record_err(completed);
        if self.reusable && completed_successfully {
            self.pool.retain(pinned);
        }
    }
}

#[cfg(feature = "embedded-cuda")]
impl CudaInputUpload {
    fn discard(&mut self) {
        self.reusable = false;
    }
}

#[cfg(feature = "embedded-cuda")]
fn try_materialize_pinned(
    values: &[f32],
    shape: &[usize],
    cuda: &CudaDevice,
    pool: &InputUploadPool,
    label: &str,
) -> Result<Option<(Tensor, InputUploadGuard)>> {
    let stream = cuda.cuda_stream();
    let Some(mut upload) = pool.acquire(cuda, values.len()) else {
        return Ok(None);
    };
    let initialized = match upload.pinned.as_mut() {
        Some(pinned) => pinned.as_mut_slice(),
        None => {
            upload.discard();
            return Err(PowerError::InferenceFailed(format!(
                "pinned storage for {label} was unavailable after allocation"
            )));
        }
    };
    match initialized {
        Ok(storage) => storage.copy_from_slice(values),
        Err(error) => {
            upload.discard();
            return Err(PowerError::InferenceFailed(format!(
                "failed to initialize pinned storage for {label}: {error}"
            )));
        }
    }

    // PinnedHostSlice's HostSlice implementation records its built-in event
    // after the asynchronous copy submission. The upload lease owns that
    // allocation until the event completes, independently of Candle's
    // optional cross-stream tensor event tracking.
    let Some(pinned) = upload.pinned.as_ref() else {
        upload.discard();
        return Err(PowerError::InferenceFailed(format!(
            "pinned storage for {label} was unavailable before submission"
        )));
    };
    let device_values = match stream.clone_htod(pinned) {
        Ok(device_values) => device_values,
        Err(error) => {
            upload.discard();
            return Err(submission_error(
                &stream,
                label,
                "copy pinned input storage",
                error,
            ));
        }
    };

    let storage = CudaStorage::wrap_cuda_slice(device_values, cuda.clone());
    let tensor = Tensor::from((Storage::Cuda(storage), shape.to_vec()));
    Ok(Some((
        tensor,
        InputUploadGuard {
            uploads: vec![upload],
        },
    )))
}

#[cfg(feature = "embedded-cuda")]
fn submission_error(
    stream: &CudaStream,
    label: &str,
    operation: &str,
    error: impl std::fmt::Display,
) -> PowerError {
    let context = stream.context();
    let deferred = context.check_err().err();
    let synchronized = stream.synchronize().err();
    let cleanup = match (deferred, synchronized) {
        (None, None) => String::new(),
        (Some(deferred), None) => format!("; prior deferred CUDA error: {deferred}"),
        (None, Some(synchronized)) => {
            format!("; CUDA cleanup synchronization failed: {synchronized}")
        }
        (Some(deferred), Some(synchronized)) => format!(
            "; prior deferred CUDA error: {deferred}; CUDA cleanup synchronization failed: {synchronized}"
        ),
    };
    PowerError::InferenceFailed(format!(
        "failed to {operation} for {label}: {error}{cleanup}"
    ))
}
