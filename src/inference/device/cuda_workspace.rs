use std::{ffi::c_void, sync::Arc};

use candle_core::cuda_backend::{
    cudarc::{
        cublas::{sys, CudaBlas},
        driver::{CudaSlice, CudaStream, DevicePtrMut},
    },
    CudaDevice,
};

use crate::error::{PowerError, Result};

const MEBIBYTE: usize = 1024 * 1024;
const OTHER_ARCHITECTURE_WORKSPACE_BYTES: usize = 4 * MEBIBYTE;
const HOPPER_WORKSPACE_BYTES: usize = 32 * MEBIBYTE;
const REQUIRED_ALIGNMENT_BYTES: u64 = 256;

/// One user-owned cuBLAS workspace bound to one Candle stream and handle.
///
/// NVIDIA documents separate per-stream workspaces as a bit-reproducibility
/// requirement when independent CUDA streams execute concurrently. The fixed
/// size is architecture-derived, content-free, and allocated once with the
/// owning runtime device.
pub(super) struct CudaBlasWorkspace {
    storage: Option<CudaSlice<u8>>,
    handle: Arc<CudaBlas>,
    stream: Arc<CudaStream>,
    bytes: usize,
}

impl CudaBlasWorkspace {
    pub(super) fn configure(device: &CudaDevice) -> Result<Self> {
        let stream = device.cuda_stream();
        let compute_capability = stream.context().compute_capability().map_err(|error| {
            backend_error(format!(
                "failed to inspect CUDA compute capability for the cuBLAS workspace: {error}"
            ))
        })?;
        let bytes = recommended_workspace_bytes(compute_capability);
        let handle = device.cublas_handle();
        let mut storage = unsafe { stream.alloc::<u8>(bytes) }.map_err(|error| {
            backend_error(format!(
                "failed to allocate the {bytes}-byte cuBLAS workspace: {error}"
            ))
        })?;
        let (pointer, pointer_guard) = storage.device_ptr_mut(&stream);
        if pointer % REQUIRED_ALIGNMENT_BYTES != 0 {
            drop(pointer_guard);
            return Err(backend_error(format!(
                "CUDA returned a cuBLAS workspace address that is not {REQUIRED_ALIGNMENT_BYTES}-byte aligned"
            )));
        }
        let status = unsafe {
            sys::cublasSetWorkspace_v2(*handle.handle(), pointer as *mut c_void, bytes).result()
        };
        drop(pointer_guard);
        status.map_err(|error| {
            backend_error(format!(
                "failed to bind the {bytes}-byte user-owned cuBLAS workspace: {error}"
            ))
        })?;
        Ok(Self {
            storage: Some(storage),
            handle,
            stream,
            bytes,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for CudaBlasWorkspace {
    fn drop(&mut self) {
        let reset = self.stream.synchronize().is_ok()
            && unsafe {
                // NVIDIA documents that setting the current stream resets the
                // handle to its default workspace pool. Reusing the same stream
                // changes no execution lane.
                sys::cublasSetStream_v2(*self.handle.handle(), self.stream.cu_stream() as _)
                    .result()
                    .is_ok()
            };
        if !reset {
            // Never leave a still-live cuBLAS handle pointing at freed device
            // memory. CUDA teardown errors cannot be returned from Drop, so the
            // bounded buffer is intentionally retained until process teardown.
            if let Some(storage) = self.storage.take() {
                let _ = storage.leak();
            }
        }
    }
}

fn recommended_workspace_bytes(compute_capability: (i32, i32)) -> usize {
    if compute_capability.0 == 9 {
        HOPPER_WORKSPACE_BYTES
    } else {
        OTHER_ARCHITECTURE_WORKSPACE_BYTES
    }
}

fn backend_error(message: String) -> PowerError {
    PowerError::BackendNotAvailable(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_size_follows_the_vendor_architecture_contract() {
        assert_eq!(recommended_workspace_bytes((9, 0)), 32 * MEBIBYTE);
        assert_eq!(recommended_workspace_bytes((9, 1)), 32 * MEBIBYTE);
        assert_eq!(recommended_workspace_bytes((8, 9)), 4 * MEBIBYTE);
        assert_eq!(recommended_workspace_bytes((10, 0)), 4 * MEBIBYTE);
    }

    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn cuda_workspace_is_bound_at_the_vendor_recommended_size() {
        let device = candle_core::Device::new_cuda_with_stream(0).unwrap();
        let candle_core::Device::Cuda(cuda) = device else {
            panic!("explicit CUDA construction returned another device kind");
        };
        let compute_capability = cuda.cuda_stream().context().compute_capability().unwrap();
        let workspace = CudaBlasWorkspace::configure(&cuda).unwrap();

        assert_eq!(
            workspace.bytes(),
            recommended_workspace_bytes(compute_capability)
        );
    }
}
