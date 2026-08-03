mod binding;
mod execution;
mod mesh;
mod mesh_execution;
mod types;

use std::collections::BTreeMap;

use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::residency::DeviceResidencyAcquire;
use super::{ExecutionPermit, RuntimeDeviceKind, WeightHierarchy, WeightKey, WeightTier};

pub use binding::ConfidentialGpuBinding;
pub use execution::{
    AcceleratorBatchResolution, AcceleratorFallback, AcceleratorFusedBatch,
    AcceleratorFusedBatchOutput, AcceleratorFusedExecution, AcceleratorFusedGroup,
    AcceleratorKernelOutcome,
};
pub use mesh::{
    AcceleratorDeviceMesh, AcceleratorDeviceMeshDeclaration, AcceleratorMeshDevice,
    AcceleratorMeshDeviceDeclaration, AcceleratorPeerTransferDeclaration,
    AcceleratorPeerTransferSpec,
};
pub use mesh_execution::{AcceleratorMeshExecution, AcceleratorPeerTransferOutcome};
pub use types::{
    AcceleratorExecutionCompletion, AcceleratorExecutionEvidence, AcceleratorExecutionPath,
    AcceleratorFallbackMode, AcceleratorFallbackReason, AcceleratorFallbackTarget,
    AcceleratorFusedBatchSpec, AcceleratorResidencyDeclaration, AcceleratorResidencyGroup,
    AcceleratorSecurityRequirement,
};

impl WeightHierarchy {
    /// Binds a model-owned fused implementation to exact device-tier groups in
    /// the currently active residency plan.
    ///
    /// This method never materializes or pins a second copy. The active plan is
    /// already the source of truth for admission and pin provenance.
    pub fn declare_accelerator_residency(
        &self,
        spec: &AcceleratorFusedBatchSpec,
    ) -> Result<AcceleratorResidencyDeclaration> {
        self.declare_accelerator_residency_internal(spec, None)
    }

    /// Binds the same active device-residency plan to a resolved, bounded
    /// execution mesh. Resident weights continue to come from the one active
    /// cache; model-owned kernels use the mesh guard for explicit peer copies
    /// and graph partitioning.
    pub fn declare_accelerator_mesh_residency(
        &self,
        spec: &AcceleratorFusedBatchSpec,
        mesh: &AcceleratorDeviceMesh,
    ) -> Result<AcceleratorResidencyDeclaration> {
        self.declare_accelerator_residency_internal(spec, Some(mesh))
    }

    fn declare_accelerator_residency_internal(
        &self,
        spec: &AcceleratorFusedBatchSpec,
        mesh: Option<&AcceleratorDeviceMesh>,
    ) -> Result<AcceleratorResidencyDeclaration> {
        let runtime = self.runtime();
        if runtime.device().kind() == RuntimeDeviceKind::Cpu {
            return Err(PowerError::BackendNotAvailable(
                "accelerator residency requires an explicitly resolved CUDA or Metal device"
                    .to_string(),
            ));
        }
        if spec.security == AcceleratorSecurityRequirement::ConfidentialGpu
            && runtime.device().kind() != RuntimeDeviceKind::Cuda
        {
            return Err(PowerError::PolicyViolation(
                "confidential GPU residency currently requires a CUDA device bound to NVIDIA evidence"
                    .to_string(),
            ));
        }
        spec.validate(
            runtime.limits().max_graph_nodes,
            runtime.limits().max_graph_name_bytes,
        )?;
        let device_mesh = mesh
            .map(|mesh| {
                let declaration = mesh.declaration(runtime.limits(), spec.security)?;
                if declaration.primary_runtime_device() != runtime.device().identity() {
                    return Err(PowerError::InvalidRequest(
                        "accelerator mesh primary must be the hierarchy runtime device".to_string(),
                    ));
                }
                Ok(declaration)
            })
            .transpose()?;
        let plan = self.active_residency_plan().ok_or_else(|| {
            PowerError::InvalidRequest(
                "accelerator residency requires an active, applied residency plan".to_string(),
            )
        })?;
        if plan.weights_sha256 != self.store().sha256()
            || plan.runtime_device != runtime.device().name()
        {
            return Err(PowerError::InvalidFormat(
                "active residency plan is not bound to this hierarchy and runtime".to_string(),
            ));
        }

        let by_id = plan
            .groups
            .iter()
            .map(|group| (group.id.as_str(), group))
            .collect::<BTreeMap<_, _>>();
        let mut groups = Vec::with_capacity(spec.residency_group_ids.len());
        let mut total_weights = 0_usize;
        let mut total_bytes = 0_u64;
        for (canonical_index, id) in spec.residency_group_ids.iter().enumerate() {
            let group = by_id.get(id.as_str()).ok_or_else(|| {
                PowerError::InvalidRequest(format!(
                    "active residency plan does not contain declared group '{id}'"
                ))
            })?;
            if group.tier != WeightTier::Device {
                return Err(PowerError::InvalidRequest(format!(
                    "declared accelerator group '{id}' is not wholly device-resident"
                )));
            }
            total_weights = total_weights
                .checked_add(group.weights.len())
                .ok_or_else(|| {
                    PowerError::InvalidRequest(
                        "accelerator declaration weight count overflowed".to_string(),
                    )
                })?;
            total_bytes = total_bytes.checked_add(group.bytes).ok_or_else(|| {
                PowerError::InvalidRequest(
                    "accelerator declaration resident byte count overflowed".to_string(),
                )
            })?;
            groups.push(AcceleratorResidencyGroup {
                canonical_index,
                residency_group_id: id.clone(),
                bytes: group.bytes,
                weights: group.weights.clone(),
            });
        }
        if total_weights > runtime.limits().max_graph_initializers
            || total_bytes > runtime.limits().max_resident_weight_bytes
        {
            return Err(PowerError::InvalidRequest(
                "accelerator declaration exceeds embedded inference weight bounds".to_string(),
            ));
        }

        AcceleratorResidencyDeclaration::build(
            self.store().sha256().to_string(),
            plan.sha256()?,
            runtime.device().identity(),
            spec,
            runtime.limits().max_input_bytes,
            runtime.limits().max_tensor_elements,
            groups,
            total_weights,
            total_bytes,
            device_mesh,
        )
    }

    /// Resolves a declaration without silently loading, promoting, or falling
    /// back. A ready batch contains only device-cache hits from the active
    /// plan. An allowed exact fallback is returned as a distinct typed value.
    pub fn resolve_accelerator_batch(
        &self,
        declaration: &AcceleratorResidencyDeclaration,
        confidential_binding: Option<&ConfidentialGpuBinding>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<AcceleratorBatchResolution> {
        if declaration.device_mesh.is_some() {
            return Err(PowerError::InvalidRequest(
                "mesh declarations must use resolve_accelerator_mesh_batch so peer execution is tracked"
                    .to_string(),
            ));
        }
        self.resolve_accelerator_batch_internal(
            declaration,
            None,
            confidential_binding,
            permit,
            cancellation,
        )
    }

    /// Resolves a mesh declaration against the exact runtime handles used to
    /// create it. A different node, edge, bound, or claim mapping fails before
    /// any kernel launch.
    pub fn resolve_accelerator_mesh_batch(
        &self,
        declaration: &AcceleratorResidencyDeclaration,
        mesh: &AcceleratorDeviceMesh,
        confidential_binding: Option<&ConfidentialGpuBinding>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<AcceleratorBatchResolution> {
        if declaration.device_mesh.is_none() {
            return Err(PowerError::InvalidRequest(
                "single-device declarations must use resolve_accelerator_batch".to_string(),
            ));
        }
        self.resolve_accelerator_batch_internal(
            declaration,
            Some(mesh),
            confidential_binding,
            permit,
            cancellation,
        )
    }

    fn resolve_accelerator_batch_internal(
        &self,
        declaration: &AcceleratorResidencyDeclaration,
        mesh: Option<&AcceleratorDeviceMesh>,
        confidential_binding: Option<&ConfidentialGpuBinding>,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<AcceleratorBatchResolution> {
        declaration.validate()?;
        let runtime = self.runtime();
        if declaration.weights_sha256 != self.store().sha256()
            || declaration.runtime_device != runtime.device().identity()
            || declaration.max_input_bytes != runtime.limits().max_input_bytes
            || declaration.max_tensor_elements != runtime.limits().max_tensor_elements
        {
            return Err(PowerError::InvalidFormat(
                "accelerator declaration belongs to a different hierarchy, device, or runtime limit set"
                    .to_string(),
            ));
        }
        match (&declaration.device_mesh, mesh) {
            (None, None) => {}
            (Some(expected), Some(mesh))
                if mesh.matches_declaration(
                    expected,
                    runtime.limits(),
                    declaration.security,
                )? => {}
            (Some(_), Some(_)) => {
                return Err(PowerError::InvalidFormat(
                    "resolved accelerator mesh does not match the declaration".to_string(),
                ))
            }
            _ => {
                return Err(PowerError::InvalidRequest(
                    "accelerator declaration and resolved mesh shape do not match".to_string(),
                ))
            }
        }
        let claims_sha256 = match declaration.security {
            AcceleratorSecurityRequirement::Local => None,
            AcceleratorSecurityRequirement::ConfidentialGpu => {
                let binding = confidential_binding.ok_or_else(|| {
                    PowerError::PolicyViolation(
                        "confidential GPU execution requires a matching attestation binding"
                            .to_string(),
                    )
                })?;
                binding.validate_for(declaration)?;
                Some(binding.claims_sha256().to_string())
            }
        };

        let Some(plan) = self.active_residency_plan() else {
            return self.accelerator_fallback(
                declaration,
                claims_sha256,
                permit,
                AcceleratorFallbackReason::PlanChanged,
            );
        };
        if plan.sha256()? != declaration.active_plan_sha256
            || !declaration_groups_match_plan(declaration, &plan)
        {
            return self.accelerator_fallback(
                declaration,
                claims_sha256,
                permit,
                AcceleratorFallbackReason::PlanChanged,
            );
        }

        let keys = declaration
            .groups
            .iter()
            .flat_map(|group| group.weights.iter().cloned())
            .collect::<Vec<WeightKey>>();
        let weights =
            match self.acquire_declared_device_weights(&plan, &keys, permit, cancellation)? {
                DeviceResidencyAcquire::Ready(weights) => weights,
                DeviceResidencyAcquire::PlanChanged => {
                    return self.accelerator_fallback(
                        declaration,
                        claims_sha256,
                        permit,
                        AcceleratorFallbackReason::PlanChanged,
                    )
                }
                DeviceResidencyAcquire::WeightUnavailable => {
                    return self.accelerator_fallback(
                        declaration,
                        claims_sha256,
                        permit,
                        AcceleratorFallbackReason::ResidencyUnavailable,
                    )
                }
            };

        let mut weights = weights.into_iter();
        let groups = declaration
            .groups
            .iter()
            .map(|group_decl| {
                let group_weights = weights
                    .by_ref()
                    .take(group_decl.weights.len())
                    .collect::<Vec<_>>();
                if group_weights.len() != group_decl.weights.len() {
                    return Err(PowerError::InferenceFailed(
                        "accelerator device-weight acquisition returned an incomplete group"
                            .to_string(),
                    ));
                }
                Ok(execution::group(group_decl.canonical_index, group_weights))
            })
            .collect::<Result<Vec<_>>>()?;
        if weights.next().is_some() {
            return Err(PowerError::InferenceFailed(
                "accelerator device-weight acquisition returned excess weights".to_string(),
            ));
        }
        Ok(AcceleratorBatchResolution::Ready(
            AcceleratorFusedBatch::new(
                declaration,
                runtime.device().clone(),
                runtime.limits().clone(),
                claims_sha256,
                permit.clone(),
                groups,
                mesh.cloned(),
            ),
        ))
    }

    fn accelerator_fallback(
        &self,
        declaration: &AcceleratorResidencyDeclaration,
        confidential_claims_sha256: Option<String>,
        permit: &ExecutionPermit,
        reason: AcceleratorFallbackReason,
    ) -> Result<AcceleratorBatchResolution> {
        if declaration.fallback_mode == AcceleratorFallbackMode::Deny {
            return Err(PowerError::InferenceFailed(format!(
                "accelerator residency became unavailable ({reason:?}) and exact fallback is denied"
            )));
        }
        Ok(AcceleratorBatchResolution::Fallback(
            AcceleratorFallback::new(
                declaration,
                self.runtime().device().clone(),
                self.runtime().limits().clone(),
                confidential_claims_sha256,
                permit.clone(),
                reason,
            )?,
        ))
    }
}

fn declaration_groups_match_plan(
    declaration: &AcceleratorResidencyDeclaration,
    plan: &super::ResidencyPlan,
) -> bool {
    let by_id = plan
        .groups
        .iter()
        .map(|group| (group.id.as_str(), group))
        .collect::<BTreeMap<_, _>>();
    declaration.groups.iter().all(|declared| {
        by_id
            .get(declared.residency_group_id.as_str())
            .is_some_and(|planned| {
                planned.tier == WeightTier::Device
                    && planned.bytes == declared.bytes
                    && planned.weights == declared.weights
            })
    })
}
