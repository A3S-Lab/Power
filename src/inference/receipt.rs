use serde::{Deserialize, Serialize};

use super::{RuntimeDevice, RUNTIME_NAME};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelIdentity {
    pub family: String,
    pub revision: String,
    pub weights_sha256: String,
}

impl ModelIdentity {
    pub fn new(
        family: impl Into<String>,
        revision: impl Into<String>,
        weights_sha256: impl Into<String>,
    ) -> Self {
        Self {
            family: family.into(),
            revision: revision.into(),
            weights_sha256: weights_sha256.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub name: String,
    pub version: String,
    pub device: String,
}

impl RuntimeIdentity {
    pub(crate) fn current(device: &RuntimeDevice) -> Self {
        Self {
            name: RUNTIME_NAME.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            device: device.name().to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionReceipt {
    pub model: ModelIdentity,
    pub runtime: RuntimeIdentity,
    pub input_sha256: String,
    pub output_sha256: String,
    pub input_elements: usize,
    pub output_elements: usize,
}
