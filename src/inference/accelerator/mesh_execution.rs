use std::collections::BTreeSet;

use candle_core::{Device, Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::{InferenceLimits, RuntimeDeviceIdentity};

use super::mesh::{AcceleratorDeviceMesh, AcceleratorDeviceMeshDeclaration};

/// Result of one Power-managed tensor transfer. Backend allocation or copy
/// failure is intentionally typed separately from malformed bounds.
pub enum AcceleratorPeerTransferOutcome {
    Transferred(Tensor),
    Unavailable,
}

impl std::fmt::Debug for AcceleratorPeerTransferOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Transferred(tensor) => formatter
                .debug_struct("Transferred")
                .field("shape", &tensor.dims())
                .finish(),
            Self::Unavailable => formatter.write_str("Unavailable"),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TransferRecord {
    edge_index: usize,
    sequence: u32,
    bytes: u64,
}

pub(super) struct AcceleratorMeshExecutionSummary {
    pub(super) mesh_sha256: String,
    pub(super) execution_devices: Vec<RuntimeDeviceIdentity>,
    pub(super) peer_transfers_sha256: String,
}

impl AcceleratorMeshExecutionSummary {
    pub(super) fn empty(
        declaration: &AcceleratorDeviceMeshDeclaration,
        execution_device: RuntimeDeviceIdentity,
    ) -> Result<Self> {
        Ok(Self {
            mesh_sha256: declaration.mesh_sha256.clone(),
            execution_devices: vec![execution_device],
            peer_transfers_sha256: transfer_digest(&declaration.mesh_sha256, &[])?,
        })
    }
}

/// Per-launch transfer guard passed to a model-owned mesh kernel.
///
/// It exposes only resolved devices from the declaration and accounts every
/// Power-managed copy against per-edge and aggregate byte/count limits.
pub struct AcceleratorMeshExecution<'a> {
    mesh: &'a AcceleratorDeviceMesh,
    declaration: &'a AcceleratorDeviceMeshDeclaration,
    limits: &'a InferenceLimits,
    cancellation: &'a CancellationToken,
    edge_counts: Vec<u32>,
    transferred_bytes: u64,
    records: Vec<TransferRecord>,
    used_devices: BTreeSet<RuntimeDeviceIdentity>,
}

impl<'a> AcceleratorMeshExecution<'a> {
    pub(super) fn new(
        mesh: &'a AcceleratorDeviceMesh,
        declaration: &'a AcceleratorDeviceMeshDeclaration,
        limits: &'a InferenceLimits,
        cancellation: &'a CancellationToken,
    ) -> Self {
        let mut used_devices = BTreeSet::new();
        used_devices.insert(declaration.primary_runtime_device());
        Self {
            mesh,
            declaration,
            limits,
            cancellation,
            edge_counts: vec![0; declaration.peer_transfers.len()],
            transferred_bytes: 0,
            records: Vec::new(),
            used_devices,
        }
    }

    pub fn tensor_device(&mut self, node_id: &str) -> Result<&Device> {
        let index = self.node_index(node_id)?;
        let device = self.mesh.device(index).ok_or_else(|| {
            PowerError::InvalidFormat("resolved accelerator mesh lost a device".to_string())
        })?;
        self.used_devices.insert(device.runtime_device().identity());
        Ok(device.runtime_device().tensor_device())
    }

    pub fn transfer(
        &mut self,
        source_node_id: &str,
        target_node_id: &str,
        tensor: &Tensor,
    ) -> Result<AcceleratorPeerTransferOutcome> {
        check_cancelled(self.cancellation)?;
        let source = self.node_index(source_node_id)?;
        let target = self.node_index(target_node_id)?;
        let edge = self
            .declaration
            .peer_transfers
            .iter()
            .find(|edge| edge.source_node_index == source && edge.target_node_index == target)
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "accelerator mesh transfer does not match a declared directed edge".to_string(),
                )
            })?;
        let source_device = self
            .mesh
            .device(source)
            .ok_or_else(|| {
                PowerError::InvalidFormat("resolved accelerator mesh lost its source".to_string())
            })?
            .runtime_device();
        let target_device = self
            .mesh
            .device(target)
            .ok_or_else(|| {
                PowerError::InvalidFormat("resolved accelerator mesh lost its target".to_string())
            })?
            .runtime_device();
        if !tensor.device().same_device(source_device.tensor_device()) {
            return Err(PowerError::InvalidRequest(
                "accelerator peer transfer tensor is not on the declared source device".to_string(),
            ));
        }
        self.limits
            .checked_elements(tensor.dims(), "accelerator peer transfer tensor")?;
        let bytes = u64::try_from(
            tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "accelerator peer transfer byte length overflowed".to_string(),
                    )
                })?,
        )
        .map_err(|_| {
            PowerError::InvalidRequest(
                "accelerator peer transfer byte length cannot be represented".to_string(),
            )
        })?;
        let count = self.edge_counts[edge.canonical_index];
        let next_total = self.transferred_bytes.checked_add(bytes).ok_or_else(|| {
            PowerError::InvalidRequest(
                "accelerator peer transfer aggregate byte length overflowed".to_string(),
            )
        })?;
        if bytes == 0
            || bytes > edge.max_transfer_bytes
            || count >= edge.max_transfers
            || next_total > self.declaration.max_total_transfer_bytes
        {
            return Err(PowerError::InvalidRequest(
                "accelerator peer transfer exceeds its declared byte or count bounds".to_string(),
            ));
        }
        let transferred = match tensor.to_device(target_device.tensor_device()) {
            Ok(tensor) => tensor,
            Err(_) => return Ok(AcceleratorPeerTransferOutcome::Unavailable),
        };
        check_cancelled(self.cancellation)?;
        if !transferred
            .device()
            .same_device(target_device.tensor_device())
        {
            return Ok(AcceleratorPeerTransferOutcome::Unavailable);
        }
        self.edge_counts[edge.canonical_index] = count.saturating_add(1);
        self.transferred_bytes = next_total;
        self.records.push(TransferRecord {
            edge_index: edge.canonical_index,
            sequence: count,
            bytes,
        });
        self.used_devices.insert(source_device.identity());
        self.used_devices.insert(target_device.identity());
        Ok(AcceleratorPeerTransferOutcome::Transferred(transferred))
    }

    pub fn transfer_count(&self) -> usize {
        self.records.len()
    }

    pub fn transferred_bytes(&self) -> u64 {
        self.transferred_bytes
    }

    pub(super) fn finish(
        mut self,
        execution_device: RuntimeDeviceIdentity,
    ) -> Result<AcceleratorMeshExecutionSummary> {
        self.used_devices.insert(execution_device);
        Ok(AcceleratorMeshExecutionSummary {
            mesh_sha256: self.declaration.mesh_sha256.clone(),
            execution_devices: self.used_devices.into_iter().collect(),
            peer_transfers_sha256: transfer_digest(&self.declaration.mesh_sha256, &self.records)?,
        })
    }

    fn node_index(&self, node_id: &str) -> Result<usize> {
        self.declaration
            .nodes
            .iter()
            .position(|node| node.node_id == node_id)
            .ok_or_else(|| {
                PowerError::InvalidRequest(
                    "accelerator mesh operation references an unknown node".to_string(),
                )
            })
    }
}

impl std::fmt::Debug for AcceleratorMeshExecution<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorMeshExecution")
            .field("mesh_sha256", &self.declaration.mesh_sha256)
            .field("transfer_count", &self.records.len())
            .field("transferred_bytes", &self.transferred_bytes)
            .finish_non_exhaustive()
    }
}

fn transfer_digest(mesh_sha256: &str, records: &[TransferRecord]) -> Result<String> {
    let encoded = serde_json::to_vec(records)?;
    let mut hasher = Sha256::new();
    hasher.update(b"a3s-power-accelerator-peer-transfers-v1\0");
    hasher.update(mesh_sha256.as_bytes());
    hasher.update(encoded);
    Ok(format!("{:x}", hasher.finalize()))
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "accelerator mesh execution was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}
