use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::super::{RuntimeDeviceIdentity, RuntimeMemoryReservations};
use super::digest::{canonical_sha256, domain_sha256, validate_canonical_sha256};

/// Explicit policy for requests outside the finite model-owned profile set.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    rename_all = "kebab-case",
    rename_all_fields = "camelCase",
    tag = "mode"
)]
pub enum DynamicShapeFallback {
    Deny,
    Allow { implementation_sha256: String },
}

impl DynamicShapeFallback {
    pub fn allow(implementation_sha256: impl Into<String>) -> Result<Self> {
        Ok(Self::Allow {
            implementation_sha256: canonical_sha256(
                implementation_sha256,
                "dynamic shape fallback implementation",
            )?,
        })
    }

    pub(super) fn validate(&self) -> Result<()> {
        if let Self::Allow {
            implementation_sha256,
        } = self
        {
            validate_canonical_sha256(
                implementation_sha256,
                "dynamic shape fallback implementation",
            )?;
        }
        Ok(())
    }

    pub(super) fn implementation_sha256(&self) -> Option<&str> {
        match self {
            Self::Deny => None,
            Self::Allow {
                implementation_sha256,
            } => Some(implementation_sha256),
        }
    }
}

impl std::fmt::Debug for DynamicShapeFallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Deny => formatter.write_str("Deny"),
            Self::Allow { .. } => formatter
                .debug_struct("Allow")
                .field("implementation", &"sha256")
                .finish(),
        }
    }
}

/// Exact runtime state against which a finite shape-profile set was reviewed.
///
/// Shape classes remain opaque digests owned by the model crate. This binding
/// commits only to model/runtime identities and aggregate memory reservations;
/// it does not assign meaning to dimensions, buckets, padding, or geometry.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ShapeProfileBinding {
    pub weights_sha256: String,
    pub graph_sha256: String,
    pub runtime_device: RuntimeDeviceIdentity,
    pub device_topology_sha256: String,
    pub runtime_reservations: RuntimeMemoryReservations,
    pub tee_policy_sha256: String,
}

impl ShapeProfileBinding {
    pub fn new(
        weights_sha256: impl Into<String>,
        graph_sha256: impl Into<String>,
        runtime_device: RuntimeDeviceIdentity,
        device_topology_sha256: impl Into<String>,
        runtime_reservations: RuntimeMemoryReservations,
        tee_policy_sha256: impl Into<String>,
    ) -> Result<Self> {
        let binding = Self {
            weights_sha256: canonical_sha256(weights_sha256, "shape-profile weights")?,
            graph_sha256: canonical_sha256(graph_sha256, "shape-profile graph")?,
            runtime_device,
            device_topology_sha256: canonical_sha256(
                device_topology_sha256,
                "shape-profile device topology",
            )?,
            runtime_reservations,
            tee_policy_sha256: canonical_sha256(tee_policy_sha256, "shape-profile TEE policy")?,
        };
        binding.validate()?;
        Ok(binding)
    }

    /// Builds the canonical topology digest for the executor's one resolved
    /// device instead of asking a model integration to duplicate that logic.
    pub fn for_single_device(
        weights_sha256: impl Into<String>,
        graph_sha256: impl Into<String>,
        runtime_device: RuntimeDeviceIdentity,
        runtime_reservations: RuntimeMemoryReservations,
        tee_policy_sha256: impl Into<String>,
    ) -> Result<Self> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Topology {
            schema: &'static str,
            runtime_device: RuntimeDeviceIdentity,
        }
        runtime_device.validate()?;
        let topology_sha256 = domain_sha256(
            b"a3s-power-shape-profile-single-device-topology-v1\0",
            &serde_json::to_vec(&Topology {
                schema: "a3s.power.shape-profile-single-device-topology.v1",
                runtime_device,
            })?,
        )?;
        Self::new(
            weights_sha256,
            graph_sha256,
            runtime_device,
            topology_sha256,
            runtime_reservations,
            tee_policy_sha256,
        )
    }

    pub fn binding_sha256(&self) -> Result<String> {
        self.validate()?;
        domain_sha256(
            b"a3s-power-shape-profile-binding-v1\0",
            &serde_json::to_vec(self)?,
        )
    }

    pub(super) fn validate(&self) -> Result<()> {
        validate_canonical_sha256(&self.weights_sha256, "shape-profile weights")?;
        validate_canonical_sha256(&self.graph_sha256, "shape-profile graph")?;
        validate_canonical_sha256(
            &self.device_topology_sha256,
            "shape-profile device topology",
        )?;
        validate_canonical_sha256(&self.tee_policy_sha256, "shape-profile TEE policy")?;
        self.runtime_device.validate()?;
        self.runtime_reservations.validate()?;
        Ok(())
    }
}

impl std::fmt::Debug for ShapeProfileBinding {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ShapeProfileBinding")
            .field("weights", &"sha256")
            .field("graph", &"sha256")
            .field("runtime_device", &self.runtime_device)
            .field("device_topology", &"sha256")
            .field("runtime_reservations", &"bounded")
            .field("tee_policy", &"sha256")
            .finish()
    }
}
