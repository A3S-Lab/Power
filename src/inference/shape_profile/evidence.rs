use serde::{Deserialize, Serialize};

use crate::error::{PowerError, Result};

use super::super::RuntimeDeviceIdentity;
use super::digest::validate_canonical_sha256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShapeProfileFallbackReason {
    ShapeClassUnavailable,
    BatchBoundExceeded,
    TensorElementBoundExceeded,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    tag = "kind"
)]
pub enum ShapeProfileExecutionPath {
    Profile {
        profile_sha256: String,
        implementation_sha256: String,
    },
    DynamicFallback {
        reason: ShapeProfileFallbackReason,
        implementation_sha256: String,
    },
}

impl ShapeProfileExecutionPath {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Profile {
                profile_sha256,
                implementation_sha256,
            } => {
                validate_canonical_sha256(profile_sha256, "selected shape profile")?;
                validate_canonical_sha256(
                    implementation_sha256,
                    "selected shape-profile implementation",
                )?;
            }
            Self::DynamicFallback {
                implementation_sha256,
                ..
            } => validate_canonical_sha256(
                implementation_sha256,
                "dynamic shape-profile implementation",
            )?,
        }
        Ok(())
    }

    pub fn implementation_sha256(&self) -> &str {
        match self {
            Self::Profile {
                implementation_sha256,
                ..
            }
            | Self::DynamicFallback {
                implementation_sha256,
                ..
            } => implementation_sha256,
        }
    }
}

impl std::fmt::Debug for ShapeProfileExecutionPath {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Profile { .. } => formatter
                .debug_struct("Profile")
                .field("profile", &"sha256")
                .field("implementation", &"sha256")
                .finish(),
            Self::DynamicFallback { reason, .. } => formatter
                .debug_struct("DynamicFallback")
                .field("reason", reason)
                .field("implementation", &"sha256")
                .finish(),
        }
    }
}

/// Digest-only profile selection evidence safe to attach to a receipt.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShapeProfileExecutionEvidence {
    pub schema: String,
    pub declaration_sha256: String,
    pub binding_sha256: String,
    pub request_sha256: String,
    pub weights_sha256: String,
    pub runtime_device: RuntimeDeviceIdentity,
    pub input_sha256: String,
    pub path: ShapeProfileExecutionPath,
}

impl ShapeProfileExecutionEvidence {
    pub const SCHEMA: &'static str = "a3s.power.shape-profile-execution.v1";

    pub fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA {
            return Err(PowerError::InvalidFormat(
                "shape-profile execution evidence has an unsupported schema".to_string(),
            ));
        }
        for (value, label) in [
            (&self.declaration_sha256, "shape-profile declaration"),
            (&self.binding_sha256, "shape-profile binding"),
            (&self.request_sha256, "shape-profile request"),
            (&self.weights_sha256, "shape-profile weights"),
            (&self.input_sha256, "shape-profile input"),
        ] {
            validate_canonical_sha256(value, label)?;
        }
        self.runtime_device.validate()?;
        self.path.validate()?;
        Ok(())
    }
}

impl std::fmt::Debug for ShapeProfileExecutionEvidence {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path = match self.path {
            ShapeProfileExecutionPath::Profile { .. } => "profile",
            ShapeProfileExecutionPath::DynamicFallback { .. } => "dynamic-fallback",
        };
        formatter
            .debug_struct("ShapeProfileExecutionEvidence")
            .field("schema", &self.schema)
            .field("declaration", &"sha256")
            .field("binding", &"sha256")
            .field("request", &"sha256")
            .field("weights", &"sha256")
            .field("runtime_device", &self.runtime_device)
            .field("input", &"sha256")
            .field("path", &path)
            .finish()
    }
}

/// Opaque validated selection. It can only be created from a declaration.
#[derive(Clone, PartialEq, Eq)]
pub struct ShapeProfileSelection {
    evidence: ShapeProfileExecutionEvidence,
}

impl ShapeProfileSelection {
    pub(super) fn new(evidence: ShapeProfileExecutionEvidence) -> Self {
        Self { evidence }
    }

    pub fn evidence(&self) -> &ShapeProfileExecutionEvidence {
        &self.evidence
    }

    pub fn path(&self) -> &ShapeProfileExecutionPath {
        &self.evidence.path
    }

    pub fn implementation_sha256(&self) -> &str {
        self.evidence.path.implementation_sha256()
    }

    pub(in crate::inference) fn validate_for_receipt(
        &self,
        weights_sha256: &str,
        runtime_device: RuntimeDeviceIdentity,
        input_sha256: &str,
    ) -> Result<()> {
        self.evidence.validate()?;
        if self.evidence.weights_sha256 != weights_sha256
            || self.evidence.runtime_device != runtime_device
            || self.evidence.input_sha256 != input_sha256
        {
            return Err(PowerError::InvalidRequest(
                "shape-profile selection does not match the receipt model, runtime, or input"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl std::fmt::Debug for ShapeProfileSelection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShapeProfileSelection")
            .field("evidence", &self.evidence)
            .finish()
    }
}
