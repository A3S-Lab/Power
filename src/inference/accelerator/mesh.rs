use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{PowerError, Result};
use crate::inference::{InferenceLimits, RuntimeDevice, RuntimeDeviceIdentity, RuntimeDeviceKind};

use super::types::{validate_identifier, validate_sha256, AcceleratorSecurityRequirement};

const ABSOLUTE_MAX_MESH_DEVICES: usize = 16;
const ABSOLUTE_MAX_PEER_TRANSFERS: usize = ABSOLUTE_MAX_MESH_DEVICES * 15;

/// One resolved execution device in a model-neutral accelerator mesh.
#[derive(Clone)]
pub struct AcceleratorMeshDevice {
    node_id: String,
    runtime_device: RuntimeDevice,
    attestation_gpu_claim_index: Option<u32>,
}

impl AcceleratorMeshDevice {
    pub fn new(node_id: impl Into<String>, runtime_device: RuntimeDevice) -> Self {
        Self {
            node_id: node_id.into(),
            runtime_device,
            attestation_gpu_claim_index: None,
        }
    }

    /// Binds this CUDA node to an exact entry in NVIDIA's attested claims
    /// array. The claim index is deliberately not inferred from the CUDA
    /// ordinal and is never treated as a UEID.
    pub fn with_attestation_gpu_claim_index(mut self, index: u32) -> Self {
        self.attestation_gpu_claim_index = Some(index);
        self
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn runtime_device(&self) -> &RuntimeDevice {
        &self.runtime_device
    }
}

impl std::fmt::Debug for AcceleratorMeshDevice {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorMeshDevice")
            .field("runtime_device", &self.runtime_device.identity())
            .field(
                "attestation_gpu_claim_index",
                &self.attestation_gpu_claim_index,
            )
            .finish_non_exhaustive()
    }
}

/// One directed, bounded transfer edge. Power does not promise that the
/// backend implements it as direct DMA; a failed backend copy is exposed as a
/// typed unavailability outcome so the exact model-owned fallback can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceleratorPeerTransferSpec {
    source_node_id: String,
    target_node_id: String,
    max_transfer_bytes: u64,
    max_transfers: u32,
}

impl AcceleratorPeerTransferSpec {
    pub fn new(
        source_node_id: impl Into<String>,
        target_node_id: impl Into<String>,
        max_transfer_bytes: u64,
        max_transfers: u32,
    ) -> Self {
        Self {
            source_node_id: source_node_id.into(),
            target_node_id: target_node_id.into(),
            max_transfer_bytes,
            max_transfers,
        }
    }
}

/// Resolved runtime handles plus a private, bounded transfer topology.
///
/// Node and edge identifiers may reveal graph placement. Power therefore does
/// not serialize, log, persist, or place this value in receipts automatically.
#[derive(Clone)]
pub struct AcceleratorDeviceMesh {
    primary_node_id: String,
    devices: Vec<AcceleratorMeshDevice>,
    peer_transfers: Vec<AcceleratorPeerTransferSpec>,
    max_total_transfer_bytes: u64,
    attestation_fabric_claim_indices: Vec<u32>,
}

impl AcceleratorDeviceMesh {
    pub fn new(
        primary_node_id: impl Into<String>,
        mut devices: Vec<AcceleratorMeshDevice>,
        mut peer_transfers: Vec<AcceleratorPeerTransferSpec>,
        max_total_transfer_bytes: u64,
    ) -> Result<Self> {
        let primary_node_id = primary_node_id.into();
        if devices.is_empty() || devices.len() > ABSOLUTE_MAX_MESH_DEVICES {
            return Err(PowerError::InvalidRequest(format!(
                "accelerator mesh must contain between 1 and {ABSOLUTE_MAX_MESH_DEVICES} devices"
            )));
        }
        devices.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        let mut ids = BTreeSet::new();
        let mut identities = BTreeSet::new();
        for device in &devices {
            validate_identifier(&device.node_id, 1_024, "accelerator mesh node")?;
            device.runtime_device.identity().validate()?;
            if !ids.insert(device.node_id.clone())
                || !identities.insert(device.runtime_device.identity())
            {
                return Err(PowerError::InvalidRequest(
                    "accelerator mesh contains a duplicate node or runtime device".to_string(),
                ));
            }
        }
        if !ids.contains(&primary_node_id) {
            return Err(PowerError::InvalidRequest(
                "accelerator mesh primary node is not present".to_string(),
            ));
        }
        if peer_transfers.len() > ABSOLUTE_MAX_PEER_TRANSFERS {
            return Err(PowerError::InvalidRequest(format!(
                "accelerator mesh exceeds {ABSOLUTE_MAX_PEER_TRANSFERS} peer-transfer edges"
            )));
        }
        peer_transfers.sort_by(|left, right| {
            left.source_node_id
                .cmp(&right.source_node_id)
                .then_with(|| left.target_node_id.cmp(&right.target_node_id))
        });
        validate_topology(
            &primary_node_id,
            &ids,
            &peer_transfers,
            max_total_transfer_bytes,
        )?;
        Ok(Self {
            primary_node_id,
            devices,
            peer_transfers,
            max_total_transfer_bytes,
            attestation_fabric_claim_indices: Vec::new(),
        })
    }

    /// Binds all attested NVIDIA fabric entries (currently NVSwitch claims) to
    /// the mesh declaration. Presence does not claim that NRAS proves a
    /// particular edge; actual transfer availability is checked at execution.
    pub fn with_attestation_fabric_claim_indices(mut self, mut indices: Vec<u32>) -> Result<Self> {
        indices.sort_unstable();
        if indices.windows(2).any(|window| window[0] == window[1]) {
            return Err(PowerError::InvalidRequest(
                "accelerator mesh fabric claim indices must be unique".to_string(),
            ));
        }
        self.attestation_fabric_claim_indices = indices;
        Ok(self)
    }

    pub fn primary_node_id(&self) -> &str {
        &self.primary_node_id
    }

    pub fn devices(&self) -> &[AcceleratorMeshDevice] {
        &self.devices
    }

    pub(super) fn declaration(
        &self,
        limits: &InferenceLimits,
        security: AcceleratorSecurityRequirement,
    ) -> Result<AcceleratorDeviceMeshDeclaration> {
        if self.devices.len() > limits.max_graph_nodes.min(ABSOLUTE_MAX_MESH_DEVICES)
            || self.peer_transfers.len() > limits.max_graph_nodes.min(ABSOLUTE_MAX_PEER_TRANSFERS)
            || self.max_total_transfer_bytes > limits.max_state_bytes
        {
            return Err(PowerError::InvalidRequest(
                "accelerator mesh exceeds embedded runtime topology or transfer bounds".to_string(),
            ));
        }
        let mut claim_indices = BTreeSet::new();
        let mut nodes = Vec::with_capacity(self.devices.len());
        for (canonical_index, device) in self.devices.iter().enumerate() {
            validate_identifier(
                &device.node_id,
                limits.max_graph_name_bytes,
                "accelerator mesh node",
            )?;
            match (security, device.runtime_device.kind()) {
                (AcceleratorSecurityRequirement::Local, _) => {
                    if device.attestation_gpu_claim_index.is_some() {
                        return Err(PowerError::PolicyViolation(
                            "local accelerator meshes cannot carry confidential GPU claim bindings"
                                .to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Cuda) => {
                    let index = device.attestation_gpu_claim_index.ok_or_else(|| {
                        PowerError::PolicyViolation(
                            "every confidential CUDA mesh node requires an explicit attestation claim index"
                                .to_string(),
                        )
                    })?;
                    if !claim_indices.insert(index) {
                        return Err(PowerError::PolicyViolation(
                            "confidential GPU claim indices must be unique across mesh nodes"
                                .to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Cpu) => {
                    if device.attestation_gpu_claim_index.is_some() {
                        return Err(PowerError::PolicyViolation(
                            "CPU mesh nodes cannot bind NVIDIA GPU claim indices".to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Metal) => {
                    return Err(PowerError::PolicyViolation(
                        "confidential GPU meshes currently support attested CUDA nodes, not Metal"
                            .to_string(),
                    ));
                }
            }
            nodes.push(AcceleratorMeshDeviceDeclaration {
                canonical_index,
                node_id: device.node_id.clone(),
                runtime_device: device.runtime_device.identity(),
                attestation_gpu_claim_index: device.attestation_gpu_claim_index,
            });
        }
        if security == AcceleratorSecurityRequirement::Local
            && !self.attestation_fabric_claim_indices.is_empty()
        {
            return Err(PowerError::PolicyViolation(
                "local accelerator meshes cannot carry attested fabric claims".to_string(),
            ));
        }
        for index in &self.attestation_fabric_claim_indices {
            if !claim_indices.insert(*index) {
                return Err(PowerError::PolicyViolation(
                    "GPU and fabric attestation claim indices must not overlap".to_string(),
                ));
            }
        }
        let node_indices = nodes
            .iter()
            .map(|node| (node.node_id.as_str(), node.canonical_index))
            .collect::<BTreeMap<_, _>>();
        let max_peer_transfer_bytes = u64::try_from(limits.max_input_bytes).map_err(|_| {
            PowerError::InvalidRequest(
                "embedded input byte limit cannot be represented as a peer-transfer bound"
                    .to_string(),
            )
        })?;
        let mut transfers = Vec::with_capacity(self.peer_transfers.len());
        for (canonical_index, edge) in self.peer_transfers.iter().enumerate() {
            if edge.max_transfer_bytes > max_peer_transfer_bytes {
                return Err(PowerError::InvalidRequest(
                    "accelerator peer edge exceeds the per-transfer runtime byte limit".to_string(),
                ));
            }
            transfers.push(AcceleratorPeerTransferDeclaration {
                canonical_index,
                source_node_index: node_indices[edge.source_node_id.as_str()],
                target_node_index: node_indices[edge.target_node_id.as_str()],
                max_transfer_bytes: edge.max_transfer_bytes,
                max_transfers: edge.max_transfers,
            });
        }
        let primary_node_index = node_indices[self.primary_node_id.as_str()];
        AcceleratorDeviceMeshDeclaration::build(
            primary_node_index,
            nodes,
            transfers,
            self.max_total_transfer_bytes,
            self.attestation_fabric_claim_indices.clone(),
        )
    }

    pub(super) fn matches_declaration(
        &self,
        declaration: &AcceleratorDeviceMeshDeclaration,
        limits: &InferenceLimits,
        security: AcceleratorSecurityRequirement,
    ) -> Result<bool> {
        Ok(self.declaration(limits, security)? == *declaration)
    }

    pub(super) fn device(&self, index: usize) -> Option<&AcceleratorMeshDevice> {
        self.devices.get(index)
    }
}

impl std::fmt::Debug for AcceleratorDeviceMesh {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let primary = self
            .devices
            .iter()
            .find(|device| device.node_id == self.primary_node_id)
            .map(|device| device.runtime_device.identity());
        formatter
            .debug_struct("AcceleratorDeviceMesh")
            .field("primary_runtime_device", &primary)
            .field("devices", &self.devices.len())
            .field("peer_transfers", &self.peer_transfers.len())
            .field("max_total_transfer_bytes", &self.max_total_transfer_bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorMeshDeviceDeclaration {
    pub canonical_index: usize,
    pub node_id: String,
    pub runtime_device: RuntimeDeviceIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attestation_gpu_claim_index: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorPeerTransferDeclaration {
    pub canonical_index: usize,
    pub source_node_index: usize,
    pub target_node_index: usize,
    pub max_transfer_bytes: u64,
    pub max_transfers: u32,
}

/// Canonical private topology bound into an accelerator declaration. Receipts
/// expose only `mesh_sha256` and aggregate actual device identities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AcceleratorDeviceMeshDeclaration {
    pub schema: String,
    pub primary_node_index: usize,
    pub nodes: Vec<AcceleratorMeshDeviceDeclaration>,
    pub peer_transfers: Vec<AcceleratorPeerTransferDeclaration>,
    pub max_total_transfer_bytes: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attestation_fabric_claim_indices: Vec<u32>,
    pub mesh_sha256: String,
}

impl AcceleratorDeviceMeshDeclaration {
    pub const SCHEMA: &'static str = "a3s.power.accelerator-device-mesh.v1";

    fn build(
        primary_node_index: usize,
        nodes: Vec<AcceleratorMeshDeviceDeclaration>,
        peer_transfers: Vec<AcceleratorPeerTransferDeclaration>,
        max_total_transfer_bytes: u64,
        attestation_fabric_claim_indices: Vec<u32>,
    ) -> Result<Self> {
        let mut declaration = Self {
            schema: Self::SCHEMA.to_string(),
            primary_node_index,
            nodes,
            peer_transfers,
            max_total_transfer_bytes,
            attestation_fabric_claim_indices,
            mesh_sha256: String::new(),
        };
        declaration.mesh_sha256 = declaration.recompute_sha256()?;
        declaration.validate()?;
        Ok(declaration)
    }

    pub(super) fn validate(&self) -> Result<()> {
        if self.schema != Self::SCHEMA
            || self.nodes.is_empty()
            || self.nodes.len() > ABSOLUTE_MAX_MESH_DEVICES
            || self.peer_transfers.len() > ABSOLUTE_MAX_PEER_TRANSFERS
            || self.primary_node_index >= self.nodes.len()
        {
            return Err(PowerError::InvalidFormat(
                "accelerator device mesh has an invalid schema or topology shape".to_string(),
            ));
        }
        let mut ids = BTreeSet::new();
        let mut devices = BTreeSet::new();
        let mut prior_id: Option<&str> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            node.runtime_device.validate()?;
            if node.canonical_index != index
                || node.node_id.is_empty()
                || node.node_id.len() > 1_024
                || node.node_id.chars().any(char::is_control)
                || prior_id.is_some_and(|prior| prior >= node.node_id.as_str())
                || !ids.insert(node.node_id.as_str())
                || !devices.insert(node.runtime_device)
            {
                return Err(PowerError::InvalidFormat(
                    "accelerator device mesh contains a non-canonical or duplicate node"
                        .to_string(),
                ));
            }
            prior_id = Some(&node.node_id);
        }
        let mut edge_pairs = BTreeSet::new();
        let mut prior_pair = None;
        for (index, edge) in self.peer_transfers.iter().enumerate() {
            let pair = (edge.source_node_index, edge.target_node_index);
            if edge.canonical_index != index
                || edge.source_node_index >= self.nodes.len()
                || edge.target_node_index >= self.nodes.len()
                || edge.source_node_index == edge.target_node_index
                || edge.max_transfer_bytes == 0
                || edge.max_transfers == 0
                || prior_pair.is_some_and(|prior| prior >= pair)
                || !edge_pairs.insert(pair)
            {
                return Err(PowerError::InvalidFormat(
                    "accelerator device mesh contains an invalid peer-transfer edge".to_string(),
                ));
            }
            prior_pair = Some(pair);
        }
        if (self.peer_transfers.is_empty() && self.max_total_transfer_bytes != 0)
            || (!self.peer_transfers.is_empty() && self.max_total_transfer_bytes == 0)
            || self
                .attestation_fabric_claim_indices
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err(PowerError::InvalidFormat(
                "accelerator device mesh has invalid aggregate transfer or fabric bounds"
                    .to_string(),
            ));
        }
        let ids = self
            .nodes
            .iter()
            .map(|node| node.node_id.clone())
            .collect::<BTreeSet<_>>();
        let specs = self
            .peer_transfers
            .iter()
            .map(|edge| AcceleratorPeerTransferSpec {
                source_node_id: self.nodes[edge.source_node_index].node_id.clone(),
                target_node_id: self.nodes[edge.target_node_index].node_id.clone(),
                max_transfer_bytes: edge.max_transfer_bytes,
                max_transfers: edge.max_transfers,
            })
            .collect::<Vec<_>>();
        validate_topology(
            &self.nodes[self.primary_node_index].node_id,
            &ids,
            &specs,
            self.max_total_transfer_bytes,
        )
        .map_err(|_| {
            PowerError::InvalidFormat(
                "accelerator device mesh is not connected to and from its primary node".to_string(),
            )
        })?;
        validate_sha256(&self.mesh_sha256, "accelerator device mesh")?;
        if self.mesh_sha256 != self.recompute_sha256()? {
            return Err(PowerError::InvalidFormat(
                "accelerator device mesh digest does not match its canonical payload".to_string(),
            ));
        }
        Ok(())
    }

    pub(super) fn validate_security(&self, security: AcceleratorSecurityRequirement) -> Result<()> {
        let mut indices = BTreeSet::new();
        for node in &self.nodes {
            match (security, node.runtime_device.kind) {
                (AcceleratorSecurityRequirement::Local, _) => {
                    if node.attestation_gpu_claim_index.is_some() {
                        return Err(PowerError::InvalidFormat(
                            "local accelerator mesh contains a confidential GPU claim binding"
                                .to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Cuda) => {
                    let index = node.attestation_gpu_claim_index.ok_or_else(|| {
                        PowerError::InvalidFormat(
                            "confidential CUDA mesh node is missing its attestation claim index"
                                .to_string(),
                        )
                    })?;
                    if !indices.insert(index) {
                        return Err(PowerError::InvalidFormat(
                            "confidential mesh reuses an attestation claim index".to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Cpu) => {
                    if node.attestation_gpu_claim_index.is_some() {
                        return Err(PowerError::InvalidFormat(
                            "CPU mesh node contains a GPU attestation claim index".to_string(),
                        ));
                    }
                }
                (AcceleratorSecurityRequirement::ConfidentialGpu, RuntimeDeviceKind::Metal) => {
                    return Err(PowerError::InvalidFormat(
                        "confidential accelerator mesh contains an unattested Metal node"
                            .to_string(),
                    ));
                }
            }
        }
        if security == AcceleratorSecurityRequirement::Local
            && !self.attestation_fabric_claim_indices.is_empty()
        {
            return Err(PowerError::InvalidFormat(
                "local accelerator mesh contains attested fabric claims".to_string(),
            ));
        }
        for index in &self.attestation_fabric_claim_indices {
            if !indices.insert(*index) {
                return Err(PowerError::InvalidFormat(
                    "accelerator mesh reuses a GPU/fabric attestation claim index".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn primary_runtime_device(&self) -> RuntimeDeviceIdentity {
        self.nodes[self.primary_node_index].runtime_device
    }

    fn recompute_sha256(&self) -> Result<String> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Payload<'a> {
            schema: &'a str,
            primary_node_index: usize,
            nodes: &'a [AcceleratorMeshDeviceDeclaration],
            peer_transfers: &'a [AcceleratorPeerTransferDeclaration],
            max_total_transfer_bytes: u64,
            attestation_fabric_claim_indices: &'a [u32],
        }
        let encoded = serde_json::to_vec(&Payload {
            schema: &self.schema,
            primary_node_index: self.primary_node_index,
            nodes: &self.nodes,
            peer_transfers: &self.peer_transfers,
            max_total_transfer_bytes: self.max_total_transfer_bytes,
            attestation_fabric_claim_indices: &self.attestation_fabric_claim_indices,
        })?;
        let mut hasher = Sha256::new();
        hasher.update(b"a3s-power-accelerator-device-mesh-v1\0");
        hasher.update(encoded);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn validate_topology(
    primary_node_id: &str,
    node_ids: &BTreeSet<String>,
    edges: &[AcceleratorPeerTransferSpec],
    max_total_transfer_bytes: u64,
) -> Result<()> {
    let mut pairs = BTreeSet::new();
    for edge in edges {
        if edge.source_node_id == edge.target_node_id
            || !node_ids.contains(&edge.source_node_id)
            || !node_ids.contains(&edge.target_node_id)
            || edge.max_transfer_bytes == 0
            || edge.max_transfers == 0
            || !pairs.insert((edge.source_node_id.clone(), edge.target_node_id.clone()))
        {
            return Err(PowerError::InvalidRequest(
                "accelerator mesh contains an invalid or duplicate directed transfer edge"
                    .to_string(),
            ));
        }
    }
    if (edges.is_empty() && max_total_transfer_bytes != 0)
        || (!edges.is_empty() && max_total_transfer_bytes == 0)
    {
        return Err(PowerError::InvalidRequest(
            "accelerator mesh aggregate transfer bytes do not match its edge set".to_string(),
        ));
    }
    if node_ids.len() == 1 {
        return Ok(());
    }
    if edges.is_empty()
        || !all_reachable(primary_node_id, node_ids, edges, false)
        || !all_reachable(primary_node_id, node_ids, edges, true)
    {
        return Err(PowerError::InvalidRequest(
            "every accelerator mesh node must be reachable from and able to return to the primary"
                .to_string(),
        ));
    }
    Ok(())
}

fn all_reachable(
    primary: &str,
    nodes: &BTreeSet<String>,
    edges: &[AcceleratorPeerTransferSpec],
    reverse: bool,
) -> bool {
    let mut reached = BTreeSet::from([primary.to_string()]);
    loop {
        let before = reached.len();
        for edge in edges {
            let (source, target) = if reverse {
                (&edge.target_node_id, &edge.source_node_id)
            } else {
                (&edge.source_node_id, &edge.target_node_id)
            };
            if reached.contains(source) {
                reached.insert(target.clone());
            }
        }
        if reached.len() == before {
            break;
        }
    }
    reached.len() == nodes.len()
}
