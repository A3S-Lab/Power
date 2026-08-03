use candle_core::Device;
use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
    pub(crate) candle: Device,
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

    fn cpu() -> Self {
        Self {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
            name: "cpu".to_string(),
            candle: Device::Cpu,
        }
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
        let candle = Device::new_cuda(ordinal).map_err(|error| {
            PowerError::BackendNotAvailable(format!(
                "failed to initialize CUDA device {ordinal}: {error}"
            ))
        })?;
        Ok(Self {
            kind: RuntimeDeviceKind::Cuda,
            ordinal: Some(ordinal),
            name: format!("cuda:{ordinal}"),
            candle,
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
            candle,
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
            // Contract tests use CPU storage while exercising the logical
            // accelerator control path. This constructor is absent from
            // production builds.
            candle: Device::Cpu,
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
}
