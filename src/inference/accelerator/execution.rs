use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::{
    AcceleratorExecutionEvidence, ExecutionDigest, ExecutionPermit, InferenceLimits,
    ResidentWeight, RuntimeDevice, RuntimeDeviceIdentity,
};

use super::mesh_execution::AcceleratorMeshExecutionSummary;
use super::types::{
    AcceleratorExecutionCompletion, AcceleratorExecutionPath, AcceleratorFallbackMode,
    AcceleratorFallbackReason, AcceleratorFallbackTarget, AcceleratorResidencyDeclaration,
};
use super::{AcceleratorDeviceMesh, AcceleratorDeviceMeshDeclaration, AcceleratorMeshExecution};

#[derive(Clone)]
pub struct AcceleratorFusedGroup {
    canonical_index: usize,
    weights: Vec<ResidentWeight>,
}

impl AcceleratorFusedGroup {
    pub fn canonical_index(&self) -> usize {
        self.canonical_index
    }

    pub fn weights(&self) -> &[ResidentWeight] {
        &self.weights
    }
}

impl std::fmt::Debug for AcceleratorFusedGroup {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorFusedGroup")
            .field("canonical_index", &self.canonical_index)
            .field("weights", &self.weights.len())
            .finish()
    }
}

/// Typed result from a model-owned accelerator kernel.
///
/// Only `Unavailable` enters the declared exact fallback path. Arithmetic,
/// shape, integrity, or policy errors remain ordinary failures.
pub enum AcceleratorKernelOutcome {
    Output(Tensor),
    Unavailable,
}

impl std::fmt::Debug for AcceleratorKernelOutcome {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Output(tensor) => formatter
                .debug_struct("Output")
                .field("shape", &tensor.dims())
                .finish(),
            Self::Unavailable => formatter.write_str("Unavailable"),
        }
    }
}

pub enum AcceleratorFusedExecution {
    Output(AcceleratorFusedBatchOutput),
    Fallback(AcceleratorFallback),
}

impl std::fmt::Debug for AcceleratorFusedExecution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Output(output) => formatter.debug_tuple("Output").field(output).finish(),
            Self::Fallback(fallback) => formatter.debug_tuple("Fallback").field(fallback).finish(),
        }
    }
}

pub struct AcceleratorFusedBatch {
    declaration_sha256: String,
    weights_sha256: String,
    fused_kernel_sha256: String,
    exact_fallback_sha256: String,
    fallback_mode: AcceleratorFallbackMode,
    fallback_target: AcceleratorFallbackTarget,
    runtime_device: RuntimeDevice,
    limits: InferenceLimits,
    confidential_claims_sha256: Option<String>,
    permit: ExecutionPermit,
    groups: Vec<AcceleratorFusedGroup>,
    device_mesh_declaration: Option<Box<AcceleratorDeviceMeshDeclaration>>,
    device_mesh: Option<Box<AcceleratorDeviceMesh>>,
}

impl AcceleratorFusedBatch {
    pub(super) fn new(
        declaration: &AcceleratorResidencyDeclaration,
        runtime_device: RuntimeDevice,
        limits: InferenceLimits,
        confidential_claims_sha256: Option<String>,
        permit: ExecutionPermit,
        groups: Vec<AcceleratorFusedGroup>,
        device_mesh: Option<AcceleratorDeviceMesh>,
    ) -> Self {
        Self {
            declaration_sha256: declaration.declaration_sha256.clone(),
            weights_sha256: declaration.weights_sha256.clone(),
            fused_kernel_sha256: declaration.fused_kernel_sha256.clone(),
            exact_fallback_sha256: declaration.exact_fallback_sha256.clone(),
            fallback_mode: declaration.fallback_mode,
            fallback_target: declaration.fallback_target,
            runtime_device,
            limits,
            confidential_claims_sha256,
            permit,
            groups,
            device_mesh_declaration: declaration.device_mesh.clone().map(Box::new),
            device_mesh: device_mesh.map(Box::new),
        }
    }

    pub fn groups(&self) -> &[AcceleratorFusedGroup] {
        &self.groups
    }

    /// Executes one model-owned fused Candle operation under Power's existing
    /// permit lifetime, cancellation boundary, typed device, and tensor bound.
    /// Power neither interprets the operation nor changes its arithmetic.
    pub fn execute<F>(
        self,
        input: &Tensor,
        cancellation: &CancellationToken,
        operation: F,
    ) -> Result<AcceleratorFusedBatchOutput>
    where
        F: FnOnce(&Tensor, &[AcceleratorFusedGroup], &CancellationToken) -> Result<Tensor>,
    {
        match self.execute_or_fallback(input, cancellation, |input, groups, cancellation| {
            operation(input, groups, cancellation).map(AcceleratorKernelOutcome::Output)
        })? {
            AcceleratorFusedExecution::Output(output) => Ok(output),
            AcceleratorFusedExecution::Fallback(_) => Err(PowerError::InferenceFailed(
                "infallible fused-kernel adapter unexpectedly requested fallback".to_string(),
            )),
        }
    }

    /// Executes a fused kernel that can explicitly report backend
    /// unavailability. That one typed outcome becomes the declared exact
    /// fallback; every other error remains fail-closed.
    pub fn execute_or_fallback<F>(
        self,
        input: &Tensor,
        cancellation: &CancellationToken,
        operation: F,
    ) -> Result<AcceleratorFusedExecution>
    where
        F: FnOnce(
            &Tensor,
            &[AcceleratorFusedGroup],
            &CancellationToken,
        ) -> Result<AcceleratorKernelOutcome>,
    {
        if self.device_mesh.is_some() {
            return Err(PowerError::InvalidRequest(
                "mesh accelerator batches must use execute_mesh_or_fallback so peer transfers are tracked"
                    .to_string(),
            ));
        }
        self.validate_tensor(input, "accelerator fused input", false)?;
        check_cancelled(cancellation)?;
        match operation(input, &self.groups, cancellation)? {
            AcceleratorKernelOutcome::Output(output) => {
                check_cancelled(cancellation)?;
                self.validate_tensor(&output, "accelerator fused output", true)?;
                Ok(AcceleratorFusedExecution::Output(
                    self.finish_accelerated(output, None),
                ))
            }
            AcceleratorKernelOutcome::Unavailable => self
                .into_fallback(AcceleratorFallbackReason::KernelUnavailable)
                .map(AcceleratorFusedExecution::Fallback),
        }
    }

    /// Converts a pre-launch backend capability failure into the declared
    /// exact fallback without falsely recording a fused accelerator launch.
    pub fn fallback(self) -> Result<AcceleratorFallback> {
        self.into_fallback(AcceleratorFallbackReason::KernelUnavailable)
    }

    /// Executes one model-owned mesh kernel with exact single-device semantics
    /// and Power-managed transfer accounting.
    pub fn execute_mesh<F>(
        self,
        input: &Tensor,
        cancellation: &CancellationToken,
        operation: F,
    ) -> Result<AcceleratorFusedBatchOutput>
    where
        F: FnOnce(
            &Tensor,
            &[AcceleratorFusedGroup],
            &mut AcceleratorMeshExecution<'_>,
            &CancellationToken,
        ) -> Result<Tensor>,
    {
        match self.execute_mesh_or_fallback(
            input,
            cancellation,
            |input, groups, mesh, cancellation| {
                operation(input, groups, mesh, cancellation).map(AcceleratorKernelOutcome::Output)
            },
        )? {
            AcceleratorFusedExecution::Output(output) => Ok(output),
            AcceleratorFusedExecution::Fallback(_) => Err(PowerError::InferenceFailed(
                "infallible mesh-kernel adapter unexpectedly requested fallback".to_string(),
            )),
        }
    }

    /// Executes a mesh kernel whose backend can explicitly become unavailable.
    /// Bounds, cancellation, and arithmetic errors remain failures; only the
    /// typed outcome enters the declaration's exact fallback.
    pub fn execute_mesh_or_fallback<F>(
        self,
        input: &Tensor,
        cancellation: &CancellationToken,
        operation: F,
    ) -> Result<AcceleratorFusedExecution>
    where
        F: FnOnce(
            &Tensor,
            &[AcceleratorFusedGroup],
            &mut AcceleratorMeshExecution<'_>,
            &CancellationToken,
        ) -> Result<AcceleratorKernelOutcome>,
    {
        self.validate_tensor(input, "accelerator mesh input", false)?;
        check_cancelled(cancellation)?;
        let mesh = self.device_mesh.as_deref().ok_or_else(|| {
            PowerError::InvalidRequest(
                "single-device accelerator batches cannot use mesh execution".to_string(),
            )
        })?;
        let declaration = self.device_mesh_declaration.as_deref().ok_or_else(|| {
            PowerError::InvalidFormat(
                "resolved accelerator mesh batch lost its declaration".to_string(),
            )
        })?;
        let mut execution =
            AcceleratorMeshExecution::new(mesh, declaration, &self.limits, cancellation);
        let outcome = operation(input, &self.groups, &mut execution, cancellation)?;
        check_cancelled(cancellation)?;
        match outcome {
            AcceleratorKernelOutcome::Output(output) => {
                self.validate_tensor(&output, "accelerator mesh output", true)?;
                let summary = execution.finish(self.runtime_device.identity())?;
                Ok(AcceleratorFusedExecution::Output(
                    self.finish_accelerated(output, Some(summary)),
                ))
            }
            AcceleratorKernelOutcome::Unavailable => {
                let fallback_device = self
                    .fallback_target
                    .identity(self.runtime_device.identity());
                let summary = execution.finish(fallback_device)?;
                self.into_fallback_with_mesh(
                    AcceleratorFallbackReason::KernelUnavailable,
                    Some(summary),
                )
                .map(AcceleratorFusedExecution::Fallback)
            }
        }
    }

    fn validate_tensor(&self, tensor: &Tensor, label: &str, output: bool) -> Result<()> {
        if !tensor
            .device()
            .same_device(self.runtime_device.tensor_device())
        {
            let message = if output {
                "accelerator fused kernel returned a tensor on a different device"
            } else {
                "accelerator fused input belongs to a different tensor device"
            };
            return Err(if output {
                PowerError::InferenceFailed(message.to_string())
            } else {
                PowerError::InvalidRequest(message.to_string())
            });
        }
        self.limits.checked_elements(tensor.dims(), label)?;
        Ok(())
    }

    fn finish_accelerated(
        self,
        tensor: Tensor,
        mesh: Option<AcceleratorMeshExecutionSummary>,
    ) -> AcceleratorFusedBatchOutput {
        let runtime_device = self.runtime_device.identity();
        AcceleratorFusedBatchOutput {
            tensor,
            completion: AcceleratorExecutionCompletion {
                declaration_sha256: self.declaration_sha256,
                weights_sha256: self.weights_sha256,
                runtime_device,
                execution_device: runtime_device,
                path: AcceleratorExecutionPath::Accelerator,
                fallback_target: None,
                implementation_sha256: self.fused_kernel_sha256,
                confidential_claims_sha256: self.confidential_claims_sha256,
                mesh,
                _permit: self.permit,
            },
        }
    }

    fn into_fallback(self, reason: AcceleratorFallbackReason) -> Result<AcceleratorFallback> {
        let runtime_identity = self.runtime_device.identity();
        let execution_identity = self.fallback_target.identity(runtime_identity);
        let mesh = self
            .device_mesh_declaration
            .as_deref()
            .map(|declaration| {
                AcceleratorMeshExecutionSummary::empty(declaration, execution_identity)
            })
            .transpose()?;
        self.into_fallback_with_mesh(reason, mesh)
    }

    fn into_fallback_with_mesh(
        self,
        reason: AcceleratorFallbackReason,
        mesh: Option<AcceleratorMeshExecutionSummary>,
    ) -> Result<AcceleratorFallback> {
        if self.fallback_mode == AcceleratorFallbackMode::Deny {
            return Err(PowerError::InferenceFailed(format!(
                "accelerator kernel became unavailable ({reason:?}) and exact fallback is denied"
            )));
        }
        let runtime_identity = self.runtime_device.identity();
        Ok(AcceleratorFallback::from_parts(
            self.declaration_sha256,
            self.weights_sha256,
            self.exact_fallback_sha256,
            self.fallback_target,
            self.runtime_device,
            self.limits,
            self.confidential_claims_sha256,
            self.permit,
            runtime_identity,
            reason,
            mesh,
        ))
    }
}

impl std::fmt::Debug for AcceleratorFusedBatch {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorFusedBatch")
            .field("declaration_sha256", &self.declaration_sha256)
            .field("runtime_device", &self.runtime_device.identity())
            .field("groups", &self.groups.len())
            .field(
                "mesh",
                &self.device_mesh.as_deref().map(|mesh| mesh.devices().len()),
            )
            .finish_non_exhaustive()
    }
}

pub struct AcceleratorFusedBatchOutput {
    tensor: Tensor,
    completion: AcceleratorExecutionCompletion,
}

impl AcceleratorFusedBatchOutput {
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_parts(self) -> (Tensor, AcceleratorExecutionCompletion) {
        (self.tensor, self.completion)
    }
}

impl std::fmt::Debug for AcceleratorFusedBatchOutput {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorFusedBatchOutput")
            .field("shape", &self.tensor.dims())
            .field("completion", &self.completion)
            .finish()
    }
}

pub struct AcceleratorFallback {
    reason: AcceleratorFallbackReason,
    target: AcceleratorFallbackTarget,
    execution_device: Device,
    limits: InferenceLimits,
    completion: AcceleratorExecutionCompletion,
}

impl AcceleratorFallback {
    pub(super) fn new(
        declaration: &AcceleratorResidencyDeclaration,
        runtime_device: RuntimeDevice,
        limits: InferenceLimits,
        confidential_claims_sha256: Option<String>,
        permit: ExecutionPermit,
        reason: AcceleratorFallbackReason,
    ) -> Result<Self> {
        let runtime_identity = runtime_device.identity();
        let mesh = declaration
            .device_mesh
            .as_ref()
            .map(|mesh| {
                AcceleratorMeshExecutionSummary::empty(
                    mesh,
                    declaration.fallback_target.identity(runtime_identity),
                )
            })
            .transpose()?;
        Ok(Self::from_parts(
            declaration.declaration_sha256.clone(),
            declaration.weights_sha256.clone(),
            declaration.exact_fallback_sha256.clone(),
            declaration.fallback_target,
            runtime_device,
            limits,
            confidential_claims_sha256,
            permit,
            runtime_identity,
            reason,
            mesh,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        declaration_sha256: String,
        weights_sha256: String,
        exact_fallback_sha256: String,
        target: AcceleratorFallbackTarget,
        runtime_device: RuntimeDevice,
        limits: InferenceLimits,
        confidential_claims_sha256: Option<String>,
        permit: ExecutionPermit,
        runtime_identity: RuntimeDeviceIdentity,
        reason: AcceleratorFallbackReason,
        mesh: Option<AcceleratorMeshExecutionSummary>,
    ) -> Self {
        let execution_device = match target {
            AcceleratorFallbackTarget::Cpu => Device::Cpu,
            AcceleratorFallbackTarget::RuntimeDevice => runtime_device.tensor_device().clone(),
        };
        Self {
            reason,
            target,
            execution_device,
            limits,
            completion: AcceleratorExecutionCompletion {
                declaration_sha256,
                weights_sha256,
                runtime_device: runtime_identity,
                execution_device: target.identity(runtime_identity),
                path: AcceleratorExecutionPath::Fallback { reason },
                fallback_target: Some(target),
                implementation_sha256: exact_fallback_sha256,
                confidential_claims_sha256,
                mesh,
                _permit: permit,
            },
        }
    }

    pub fn reason(&self) -> AcceleratorFallbackReason {
        self.reason
    }

    pub fn target(&self) -> AcceleratorFallbackTarget {
        self.target
    }

    pub fn implementation_sha256(&self) -> &str {
        &self.completion.implementation_sha256
    }

    pub fn tensor_device(&self) -> &Device {
        &self.execution_device
    }

    /// Runs an exact model-owned fallback on its declared target while the
    /// original execution permit remains held.
    pub fn execute<F>(
        self,
        input: &Tensor,
        cancellation: &CancellationToken,
        operation: F,
    ) -> Result<AcceleratorFusedBatchOutput>
    where
        F: FnOnce(&Tensor, &CancellationToken) -> Result<Tensor>,
    {
        check_cancelled(cancellation)?;
        if !input.device().same_device(&self.execution_device) {
            return Err(PowerError::InvalidRequest(
                "exact accelerator fallback input belongs to a different device".to_string(),
            ));
        }
        self.limits
            .checked_elements(input.dims(), "exact accelerator fallback input")?;
        let output = operation(input, cancellation)?;
        check_cancelled(cancellation)?;
        if !output.device().same_device(&self.execution_device) {
            return Err(PowerError::InferenceFailed(
                "exact accelerator fallback returned a tensor on a different device".to_string(),
            ));
        }
        self.limits
            .checked_elements(output.dims(), "exact accelerator fallback output")?;
        Ok(AcceleratorFusedBatchOutput {
            tensor: output,
            completion: self.completion,
        })
    }

    pub fn complete(
        self,
        input: &ExecutionDigest,
        output: &ExecutionDigest,
    ) -> Result<AcceleratorExecutionEvidence> {
        self.completion.complete(input, output)
    }
}

impl std::fmt::Debug for AcceleratorFallback {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AcceleratorFallback")
            .field("reason", &self.reason)
            .field("target", &self.target)
            .field("completion", &self.completion)
            .finish()
    }
}

pub enum AcceleratorBatchResolution {
    Ready(AcceleratorFusedBatch),
    Fallback(AcceleratorFallback),
}

impl std::fmt::Debug for AcceleratorBatchResolution {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(batch) => formatter.debug_tuple("Ready").field(batch).finish(),
            Self::Fallback(fallback) => formatter.debug_tuple("Fallback").field(fallback).finish(),
        }
    }
}

pub(super) fn group(canonical_index: usize, weights: Vec<ResidentWeight>) -> AcceleratorFusedGroup {
    AcceleratorFusedGroup {
        canonical_index,
        weights,
    }
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "accelerator fused execution was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}
