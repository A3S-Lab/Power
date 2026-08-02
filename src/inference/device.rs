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
    /// Select a Metal device by ordinal.
    Metal {
        ordinal: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeDeviceKind {
    Cpu,
    Metal,
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

    fn cpu() -> Self {
        Self {
            kind: RuntimeDeviceKind::Cpu,
            ordinal: None,
            name: "cpu".to_string(),
            candle: Device::Cpu,
        }
    }

    fn auto() -> Result<Self> {
        #[cfg(all(feature = "embedded-metal", target_os = "macos"))]
        if let Ok(device) = Self::metal(0) {
            return Ok(device);
        }
        Ok(Self::cpu())
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
}
