use candle_core::Device;
use serde::{Deserialize, Serialize};
#[cfg(feature = "embedded-cuda")]
use std::sync::Arc;

use crate::error::{PowerError, Result};

#[cfg(feature = "embedded-cuda")]
mod cuda_workspace;
#[cfg(feature = "embedded-cuda")]
use cuda_workspace::CudaBlasWorkspace;

/// Typed device selection for an embedded model session.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", tag = "kind")]
pub enum DevicePreference {
    /// Prefer an available accelerator supported by this build, then CPU.
    #[default]
    Auto,
    Cpu,
    /// Select a CUDA device by ordinal.
    Cuda {
        ordinal: usize,
    },
    /// Select a Metal device by ordinal.
    Metal {
        ordinal: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeDeviceKind {
    Cpu,
    Cuda,
    Metal,
}

/// Stable, serializable identity for a resolved embedded execution device.
///
/// The private Candle handle remains on [`RuntimeDevice`]. Public plans,
/// declarations, and receipts use this value so a device choice cannot be
/// represented by an unvalidated backend string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeDeviceIdentity {
    pub kind: RuntimeDeviceKind,
    pub ordinal: Option<usize>,
}

impl RuntimeDeviceIdentity {
    pub fn name(self) -> String {
        match (self.kind, self.ordinal) {
            (RuntimeDeviceKind::Cpu, None) => "cpu".to_string(),
            (RuntimeDeviceKind::Cuda, Some(ordinal)) => format!("cuda:{ordinal}"),
            (RuntimeDeviceKind::Metal, Some(ordinal)) => format!("metal:{ordinal}"),
            (kind, ordinal) => format!("invalid:{kind:?}:{ordinal:?}"),
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        match (self.kind, self.ordinal) {
            (RuntimeDeviceKind::Cpu, None)
            | (RuntimeDeviceKind::Cuda, Some(_))
            | (RuntimeDeviceKind::Metal, Some(_)) => Ok(()),
            _ => Err(PowerError::InvalidFormat(
                "runtime device identity has an invalid kind/ordinal combination".to_string(),
            )),
        }
    }
}

/// Resolved device identity paired with the private tensor device handle.
#[derive(Clone)]
pub struct RuntimeDevice {
    kind: RuntimeDeviceKind,
    ordinal: Option<usize>,
    name: String,
    isolated_session_stream: bool,
    pub(crate) candle: Device,
    // Keep the device buffer alive until after Candle drops its cuBLAS handle.
    // Rust drops fields in declaration order, so this field must follow
    // `candle`.
    #[cfg(feature = "embedded-cuda")]
    cublas_workspace: Option<Arc<CudaBlasWorkspace>>,
}

impl RuntimeDevice {
    pub fn resolve(preference: DevicePreference) -> Result<Self> {
        match preference {
            DevicePreference::Cpu => Ok(Self::cpu()),
            DevicePreference::Auto => Self::auto(),
            DevicePreference::Cuda { ordinal } => Self::cuda(ordinal),
            DevicePreference::Metal { ordinal } => Self::metal(ordinal),
        }
    }

    pub fn kind(&self) -> RuntimeDeviceKind {
        self.kind
    }

    pub fn ordinal(&self) -> Option<usize> {
        self.ordinal
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn identity(&self) -> RuntimeDeviceIdentity {
        RuntimeDeviceIdentity {
            kind: self.kind,
            ordinal: self.ordinal,
        }
    }

    /// Low-level tensor device for model crates built on Power's native
    /// inference engine.
    pub fn tensor_device(&self) -> &Device {
        &self.candle
    }

    /// Resolves a device for a model-session lane whose tensors never cross
    /// into another CUDA stream.
    ///
    /// Model session inputs and outputs cross the Power boundary as bounded
    /// host tensors. Each CUDA replica owns a unique Candle device identity
    /// and one stream, so per-activation cross-stream events duplicate the
    /// stream's ordering. General runtimes retain event tracking for explicit
    /// accelerator-mesh transfers.
    pub(crate) fn resolve_model_session(preference: DevicePreference) -> Result<Self> {
        Self::resolve(preference)?.into_isolated_session_stream()
    }

    /// Creates one additional execution stream on the same resolved CUDA
    /// device. Replica zero is the pool's original stream; higher replica
    /// indices are admitted and bounded by the owning session pool.
    pub(crate) fn execution_replica(&self, replica_index: usize) -> Result<Self> {
        if replica_index == 0 {
            return Ok(self.clone());
        }
        match (self.kind, self.ordinal) {
            (RuntimeDeviceKind::Cuda, Some(ordinal)) => {
                let replica = Self::cuda(ordinal)?;
                if self.isolated_session_stream {
                    replica.into_isolated_session_stream()
                } else {
                    Ok(replica)
                }
            }
            _ => Err(PowerError::InvalidRequest(
                "additional execution replicas require a CUDA device".to_string(),
            )),
        }
    }

    fn cpu() -> Self {
        Self {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
            name: "cpu".to_string(),
            isolated_session_stream: false,
            candle: Device::Cpu,
            #[cfg(feature = "embedded-cuda")]
            cublas_workspace: None,
        }
    }

    #[allow(unused_mut)] // Mutated only by the embedded CUDA build.
    fn into_isolated_session_stream(mut self) -> Result<Self> {
        #[cfg(feature = "embedded-cuda")]
        if let Device::Cuda(device) = &self.candle {
            // SAFETY: this mode is constructed only by ModelSessionPool. One
            // pool entry owns one unique Candle device and one stream; clones
            // retain that stream, and another execution replica receives a
            // distinct DeviceId. TensorInput/TensorOutput are host boundaries,
            // so no tensor allocated here is consumed by another stream.
            unsafe { device.disable_event_tracking() };
            self.cublas_workspace = Some(Arc::new(CudaBlasWorkspace::configure(device)?));
            self.isolated_session_stream = true;
        }
        Ok(self)
    }

    fn auto() -> Result<Self> {
        #[cfg(feature = "embedded-cuda")]
        if let Ok(device) = Self::cuda(0) {
            return Ok(device);
        }
        #[cfg(all(feature = "embedded-metal", target_os = "macos"))]
        if let Ok(device) = Self::metal(0) {
            return Ok(device);
        }
        Ok(Self::cpu())
    }

    #[cfg(feature = "embedded-cuda")]
    fn cuda(ordinal: usize) -> Result<Self> {
        let candle = Device::new_cuda_with_stream(ordinal).map_err(|error| {
            PowerError::BackendNotAvailable(format!(
                "failed to initialize CUDA device {ordinal}: {error}"
            ))
        })?;
        Ok(Self {
            kind: RuntimeDeviceKind::Cuda,
            ordinal: Some(ordinal),
            name: format!("cuda:{ordinal}"),
            isolated_session_stream: false,
            candle,
            cublas_workspace: None,
        })
    }

    #[cfg(not(feature = "embedded-cuda"))]
    fn cuda(ordinal: usize) -> Result<Self> {
        Err(PowerError::BackendNotAvailable(format!(
            "CUDA device {ordinal} requires a build with the embedded-cuda feature"
        )))
    }

    #[cfg(all(feature = "embedded-metal", target_os = "macos"))]
    fn metal(ordinal: usize) -> Result<Self> {
        let candle = Device::new_metal(ordinal).map_err(|error| {
            PowerError::BackendNotAvailable(format!(
                "failed to initialize Metal device {ordinal}: {error}"
            ))
        })?;
        Ok(Self {
            kind: RuntimeDeviceKind::Metal,
            ordinal: Some(ordinal),
            name: format!("metal:{ordinal}"),
            isolated_session_stream: false,
            candle,
            #[cfg(feature = "embedded-cuda")]
            cublas_workspace: None,
        })
    }

    #[cfg(not(all(feature = "embedded-metal", target_os = "macos")))]
    fn metal(ordinal: usize) -> Result<Self> {
        Err(PowerError::BackendNotAvailable(format!(
            "Metal device {ordinal} requires a macOS build with the embedded-metal feature"
        )))
    }

    #[cfg(test)]
    pub(crate) fn test_accelerator(kind: RuntimeDeviceKind, ordinal: usize) -> Result<Self> {
        if kind == RuntimeDeviceKind::Cpu {
            return Err(PowerError::Config(
                "test accelerator kind must not be CPU".to_string(),
            ));
        }
        Ok(Self {
            kind,
            ordinal: Some(ordinal),
            name: RuntimeDeviceIdentity {
                kind,
                ordinal: Some(ordinal),
            }
            .name(),
            isolated_session_stream: false,
            // Contract tests use CPU storage while exercising the logical
            // accelerator control path. This constructor is absent from
            // production builds.
            candle: Device::Cpu,
            #[cfg(feature = "embedded-cuda")]
            cublas_workspace: None,
        })
    }
}

impl std::fmt::Debug for RuntimeDevice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeDevice")
            .field("kind", &self.kind)
            .field("ordinal", &self.ordinal)
            .field("name", &self.name)
            .field("cublas_workspace_bytes", &{
                #[cfg(feature = "embedded-cuda")]
                {
                    self.cublas_workspace
                        .as_ref()
                        .map_or(0, |workspace| workspace.bytes())
                }
                #[cfg(not(feature = "embedded-cuda"))]
                {
                    0_usize
                }
            })
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_identity_is_explicit() {
        let device = RuntimeDevice::resolve(DevicePreference::Cpu).unwrap();
        assert_eq!(device.kind(), RuntimeDeviceKind::Cpu);
        assert_eq!(device.ordinal(), None);
        assert_eq!(device.name(), "cpu");
    }

    #[test]
    fn unavailable_metal_fails_instead_of_falling_back() {
        #[cfg(not(all(feature = "embedded-metal", target_os = "macos")))]
        assert!(RuntimeDevice::resolve(DevicePreference::Metal { ordinal: 0 }).is_err());
    }

    #[test]
    fn unavailable_cuda_fails_instead_of_falling_back() {
        #[cfg(not(feature = "embedded-cuda"))]
        assert!(RuntimeDevice::resolve(DevicePreference::Cuda { ordinal: 0 }).is_err());
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn model_session_streams_omit_only_redundant_cross_stream_events() {
        let general = RuntimeDevice::resolve(DevicePreference::Cuda { ordinal: 0 }).unwrap();
        let session =
            RuntimeDevice::resolve_model_session(DevicePreference::Cuda { ordinal: 0 }).unwrap();
        let replica = session.execution_replica(1).unwrap();

        let Device::Cuda(general_cuda) = &general.candle else {
            panic!("explicit CUDA resolution returned another device kind");
        };
        let Device::Cuda(session_cuda) = &session.candle else {
            panic!("model-session CUDA resolution returned another device kind");
        };
        let Device::Cuda(replica_cuda) = &replica.candle else {
            panic!("model-session replica returned another device kind");
        };

        assert!(general_cuda.is_event_tracking());
        assert!(!session_cuda.is_event_tracking());
        assert!(!replica_cuda.is_event_tracking());
        assert!(!session.candle.same_device(&replica.candle));
        assert!(general.cublas_workspace.is_none());
        for runtime in [&session, &replica] {
            assert!(runtime
                .cublas_workspace
                .as_ref()
                .is_some_and(|workspace| workspace.bytes() > 0));
        }
        assert!(!Arc::ptr_eq(
            session.cublas_workspace.as_ref().unwrap(),
            replica.cublas_workspace.as_ref().unwrap(),
        ));
    }

    #[cfg(feature = "embedded-cuda")]
    #[test]
    #[ignore = "requires an explicit CUDA device"]
    fn model_session_workspace_resets_before_an_escaped_device_is_reused() {
        let escaped = {
            let session =
                RuntimeDevice::resolve_model_session(DevicePreference::Cuda { ordinal: 0 })
                    .unwrap();
            assert!(session.cublas_workspace.is_some());
            session.tensor_device().clone()
        };
        let left =
            candle_core::Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (2, 2), &escaped).unwrap();
        let right =
            candle_core::Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], (2, 2), &escaped).unwrap();
        let output = left.matmul(&right).unwrap().to_vec2::<f32>().unwrap();

        assert_eq!(output, [[19.0, 22.0], [43.0, 50.0]]);
    }
}
