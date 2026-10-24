//! Bounded, explicitly completed CUDA peer copies for model-owned pipelines.
//! No CPU/Metal staging or precision conversion. Candle 0.10.2 transfers the
//! underlying storage; a private contiguous snapshot therefore precedes DMA.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use candle_core::{DType, Tensor};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use super::{InferenceLimits, RuntimeDevice, RuntimeDeviceKind};
use crate::error::{PowerError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CudaPeerScalar {
    U8,
    U32,
    F16,
    Bf16,
    F32,
}
impl CudaPeerScalar {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "f16" => Ok(Self::F16),
            "bf16" => Ok(Self::Bf16),
            "f32" => Ok(Self::F32),
            _ => Err(invalid(
                "CUDA peer scalar requires u8, u32, f16, bf16 or f32",
            )),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
    fn from_dtype(dtype: DType) -> Result<Self> {
        match dtype {
            DType::U8 => Ok(Self::U8),
            DType::U32 => Ok(Self::U32),
            DType::F16 => Ok(Self::F16),
            DType::BF16 => Ok(Self::Bf16),
            DType::F32 => Ok(Self::F32),
            _ => Err(invalid(
                "CUDA peer scalar requires u8, u32, f16, bf16 or f32",
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CudaPeerCopyReceipt {
    pub source_ordinal: usize,
    pub target_ordinal: usize,
    pub scalar: CudaPeerScalar,
    pub shape: Vec<usize>,
    pub bytes: u64,
    /// Only completed DMA has a receipt; same-owner borrows are not transfers.
    pub peer_dma_completed: bool,
}

/// Resolved devices and directed permissions. Construction enables only the
/// declared destination->source access directions within these owned contexts.
/// Driver resources stay enabled until process exit; dropping a group must not
/// disable access that another group in the same process may still be using.
pub struct CudaPeerGroup {
    devices: BTreeMap<usize, RuntimeDevice>,
    edges: BTreeSet<(usize, usize)>,
    limits: InferenceLimits,
    copy_lock: Mutex<()>,
    poisoned: AtomicBool,
    quarantine: Mutex<Vec<Tensor>>,
}

impl CudaPeerGroup {
    pub fn new(
        devices: &[RuntimeDevice],
        edges: &[(usize, usize)],
        limits: InferenceLimits,
    ) -> Result<Self> {
        limits.validate()?;
        if !(2..=8).contains(&devices.len()) || edges.is_empty() || edges.len() > 56 {
            return Err(invalid(
                "CUDA peer group requires 2..8 devices and 1..56 directed edges",
            ));
        }
        let mut registered = BTreeMap::new();
        for device in devices {
            if device.kind() != RuntimeDeviceKind::Cuda || !device.tensor_device().is_cuda() {
                return Err(unavailable(
                    "CUDA peer group requires actual CUDA devices; CPU/Metal fallback is refused",
                ));
            }
            let ordinal = device
                .ordinal()
                .ok_or_else(|| invalid("CUDA peer device has no ordinal"))?;
            if registered.insert(ordinal, device.clone()).is_some() {
                return Err(invalid("CUDA peer group ordinals must be distinct"));
            }
        }
        let mut directed = BTreeSet::new();
        for &(source, target) in edges {
            if source == target
                || !registered.contains_key(&source)
                || !registered.contains_key(&target)
                || !directed.insert((source, target))
            {
                return Err(invalid(
                    "CUDA peer edge is duplicated, self-directed or references an unknown device",
                ));
            }
        }
        #[cfg(feature = "embedded-cuda")]
        for &(source, target) in &directed {
            enable_peer(&registered[&source], &registered[&target])?;
        }
        #[cfg(not(feature = "embedded-cuda"))]
        return Err(unavailable("CUDA peer copies require embedded-cuda"));
        #[cfg(feature = "embedded-cuda")]
        Ok(Self {
            devices: registered,
            edges: directed,
            limits,
            copy_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
            quarantine: Mutex::new(Vec::with_capacity(2)),
        })
    }

    pub fn begin(&self, max_bytes: u64, max_copies: usize) -> Result<CudaPeerSession<'_>> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(unavailable(
                "CUDA peer group is quarantined after an undrainable copy",
            ));
        }
        if max_bytes == 0
            || max_bytes > self.limits.max_state_bytes
            || max_copies == 0
            || max_copies > 4096
        {
            return Err(invalid(
                "CUDA peer step requires a nonzero bounded byte budget and 1..4096 copies",
            ));
        }
        let metadata = max_copies
            .checked_mul(256)
            .ok_or_else(|| invalid("CUDA peer receipt metadata overflowed"))?;
        if metadata > self.limits.max_input_bytes {
            return Err(invalid(
                "CUDA peer receipt plan exceeds its host metadata budget",
            ));
        }
        Ok(CudaPeerSession {
            group: self,
            max_bytes,
            max_copies,
            reserved_bytes: 0,
            records: Vec::with_capacity(max_copies),
            failed: false,
        })
    }

    pub fn ordinal_of(&self, tensor: &Tensor) -> Result<usize> {
        self.ordinal_for_device(tensor.device())
    }

    pub fn ordinal_for_device(&self, tensor_device: &candle_core::Device) -> Result<usize> {
        self.devices
            .iter()
            .find_map(|(&ordinal, device)| {
                tensor_device
                    .same_device(device.tensor_device())
                    .then_some(ordinal)
            })
            .ok_or_else(|| invalid("CUDA peer tensor is not owned by a registered device"))
    }

    pub fn tensor_device(&self, ordinal: usize) -> Result<&candle_core::Device> {
        self.devices
            .get(&ordinal)
            .map(RuntimeDevice::tensor_device)
            .ok_or_else(|| invalid("CUDA peer target ordinal is not registered"))
    }

    pub fn limits(&self) -> &InferenceLimits {
        &self.limits
    }
}

pub struct CudaPeerSession<'a> {
    group: &'a CudaPeerGroup,
    max_bytes: u64,
    max_copies: usize,
    reserved_bytes: u64,
    records: Vec<CudaPeerCopyReceipt>,
    failed: bool,
}

impl CudaPeerSession<'_> {
    pub fn receipts(&self) -> &[CudaPeerCopyReceipt] {
        &self.records
    }
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }

    pub fn copy(
        &mut self,
        tensor: &Tensor,
        target: usize,
        cancellation: &CancellationToken,
    ) -> Result<Tensor> {
        if self.failed {
            return Err(invalid("CUDA peer step was poisoned by an earlier refusal"));
        }
        self.failed = true;
        let output = self.copy_inner(tensor, target, cancellation)?;
        self.failed = false;
        Ok(output)
    }

    pub fn validate_complete(&self, cancellation: &CancellationToken) -> Result<()> {
        check_cancel(cancellation)?;
        if self.failed || self.group.poisoned.load(Ordering::Acquire) {
            return Err(unavailable(
                "CUDA peer step is poisoned or its group is quarantined",
            ));
        }
        Ok(())
    }

    fn copy_inner(
        &mut self,
        tensor: &Tensor,
        target: usize,
        cancellation: &CancellationToken,
    ) -> Result<Tensor> {
        check_cancel(cancellation)?;
        self.validate_complete_for_copy(cancellation)?;
        let source = self.group.ordinal_of(tensor)?;
        let target_device = self.group.tensor_device(target)?;
        let (scalar, bytes) = admit_tensor(&self.group.limits, tensor, self.max_bytes)?;
        if source == target {
            return Ok(tensor.clone().detach());
        }
        if !self.group.edges.contains(&(source, target)) {
            return Err(invalid(
                "CUDA peer copy does not match a declared directed edge",
            ));
        }
        // A not-yet-ready compressed cache can have zero rows. Construct its
        // exact empty shape on the consumer; this is not a DMA receipt.
        if bytes == 0 {
            let empty = Tensor::zeros(tensor.shape(), tensor.dtype(), target_device)
                .map_err(tensor_error)?;
            check_cancel(cancellation)?;
            return Ok(empty);
        }
        let next = self
            .reserved_bytes
            .checked_add(bytes)
            .ok_or_else(|| invalid("CUDA peer aggregate bytes overflowed"))?;
        if next > self.max_bytes || self.records.len() >= self.max_copies {
            return Err(invalid("CUDA peer step exceeds its byte/count budget"));
        }
        let live = self
            .reserved_bytes
            .checked_add(
                bytes
                    .checked_mul(3)
                    .ok_or_else(|| invalid("CUDA peer copy live bytes overflowed"))?,
            )
            .ok_or_else(|| invalid("CUDA peer retained copies and snapshot bytes overflowed"))?;
        self.group.limits.checked_state_bytes(
            live,
            "CUDA peer retained copies, source view, snapshot and destination",
        )?;
        let _serial = self
            .group
            .copy_lock
            .lock()
            .map_err(|_| unavailable("CUDA peer copy lock is poisoned"))?;
        self.validate_complete_for_copy(cancellation)?;
        // Snapshot only this view. Tensor::to_device otherwise copies its
        // entire backing allocation, even for a narrow/non-contiguous view.
        self.reserved_bytes = next;
        let snapshot = tensor.force_contiguous().map_err(tensor_error)?.detach();
        check_cancel(cancellation)?;
        let copied = snapshot.to_device(target_device).map(|t| t.detach());
        // Cudarc waits for a source event on the destination stream. Keep the
        // snapshot alive through destination completion as well: source-side
        // event tracking alone is insufficient to express its cross-context use.
        let destination_drained = target_device.synchronize();
        let source_drained = snapshot.device().synchronize();
        if destination_drained.is_err() || source_drained.is_err() {
            self.group.poisoned.store(true, Ordering::Release);
            let mut held = match self.group.quarantine.lock() {
                Ok(held) => held,
                Err(poisoned) => poisoned.into_inner(),
            };
            held.push(snapshot);
            if let Ok(output) = copied {
                held.push(output);
            }
            return Err(unavailable(
                "CUDA peer completion failed; group quarantined until process exit",
            ));
        }
        let output = copied.map_err(tensor_error)?;
        check_cancel(cancellation)?;
        if !output.device().same_device(target_device)
            || output.dtype() != tensor.dtype()
            || output.shape() != tensor.shape()
        {
            return Err(unavailable(
                "CUDA peer output differs from its exact shape/dtype/device binding",
            ));
        }
        self.records.push(CudaPeerCopyReceipt {
            source_ordinal: source,
            target_ordinal: target,
            scalar,
            shape: tensor.dims().to_vec(),
            bytes,
            peer_dma_completed: true,
        });
        Ok(output)
    }

    fn validate_complete_for_copy(&self, cancellation: &CancellationToken) -> Result<()> {
        check_cancel(cancellation)?;
        if self.group.poisoned.load(Ordering::Acquire) {
            return Err(unavailable("CUDA peer group is quarantined"));
        }
        Ok(())
    }
}

impl Drop for CudaPeerGroup {
    fn drop(&mut self) {
        // A failed completion cannot authorize freeing memory still potentially
        // referenced by another context. This bounded fatal-path retention ends
        // with the owning worker process; no further copies are accepted.
        let held = match self.quarantine.get_mut() {
            Ok(v) => v,
            Err(poisoned) => poisoned.into_inner(),
        };
        if !held.is_empty() {
            let drained = self
                .devices
                .values()
                .all(|d| d.tensor_device().synchronize().is_ok());
            if !drained {
                for tensor in held.drain(..) {
                    std::mem::forget(tensor);
                }
            }
        }
    }
}

fn admit_tensor(
    limits: &InferenceLimits,
    tensor: &Tensor,
    max_bytes: u64,
) -> Result<(CudaPeerScalar, u64)> {
    if tensor.rank() > 8 {
        return Err(invalid("CUDA peer view rank exceeds eight dimensions"));
    }
    let scalar = CudaPeerScalar::from_dtype(tensor.dtype())?;
    let elements = if tensor.elem_count() == 0 {
        // Empty cache rows are meaningful, but their remaining axes must
        // still fit the element budget; zero cannot hide an absurd extent.
        let extents: Vec<_> = tensor.dims().iter().map(|&n| n.max(1)).collect();
        limits.checked_elements(&extents, "CUDA empty peer view extents")?;
        0
    } else {
        limits.checked_elements(tensor.dims(), "CUDA peer view")?
    };
    let bytes = (elements as u64)
        .checked_mul(tensor.dtype().size_in_bytes() as u64)
        .ok_or_else(|| invalid("CUDA peer view bytes overflowed"))?;
    if max_bytes == 0 || bytes > max_bytes {
        return Err(invalid("CUDA peer view exceeds its nonzero byte budget"));
    }
    let live = bytes
        .checked_mul(3)
        .ok_or_else(|| invalid("CUDA peer snapshot/copy live bytes overflowed"))?;
    limits.checked_state_bytes(
        live,
        "CUDA peer source view, private snapshot and completed destination",
    )?;
    Ok((scalar, bytes))
}

#[cfg(feature = "embedded-cuda")]
fn enable_peer(source: &RuntimeDevice, target: &RuntimeDevice) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let src = source
        .tensor_device()
        .as_cuda_device()
        .map_err(tensor_error)?;
    let dst = target
        .tensor_device()
        .as_cuda_device()
        .map_err(tensor_error)?;
    if !src.is_event_tracking() || !dst.is_event_tracking() {
        return Err(unavailable(
            "CUDA peer copies require event-tracked device allocations",
        ));
    }
    let source_stream = src.cuda_stream();
    let target_stream = dst.cuda_stream();
    let source_context = source_stream.context();
    let target_context = target_stream.context();
    let mut supported = 0;
    // SAFETY: both live context handles belong to the resolved group; the
    // output pointer is valid for this synchronous driver capability query.
    let status = unsafe {
        sys::cuDeviceCanAccessPeer(
            &mut supported,
            target_context.cu_device(),
            source_context.cu_device(),
        )
    };
    if status != sys::CUresult::CUDA_SUCCESS || supported != 1 {
        return Err(unavailable(
            "declared CUDA direction has no direct P2P access",
        ));
    }
    target_context
        .bind_to_thread()
        .map_err(|_| unavailable("CUDA peer target context cannot be made current"))?;
    // SAFETY: the destination context is current, source remains owned/live,
    // and CUDA's reserved flags are exactly zero. Existing enablement is kept.
    let status = unsafe { sys::cuCtxEnablePeerAccess(source_context.cu_ctx(), 0) };
    if status != sys::CUresult::CUDA_SUCCESS
        && status != sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
    {
        return Err(unavailable(
            "declared CUDA peer direction could not be enabled",
        ));
    }
    Ok(())
}

fn check_cancel(token: &CancellationToken) -> Result<()> {
    if token.is_cancelled() {
        Err(PowerError::InferenceCancelled)
    } else {
        Ok(())
    }
}
fn invalid(reason: &str) -> PowerError {
    PowerError::InvalidRequest(reason.into())
}
fn unavailable(reason: &str) -> PowerError {
    PowerError::BackendNotAvailable(reason.into())
}
fn tensor_error(error: candle_core::Error) -> PowerError {
    PowerError::InferenceFailed(format!("CUDA peer operation failed: {error}"))
}

#[cfg(test)]
mod tests;
