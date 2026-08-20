use std::time::Instant;

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};
use crate::inference::runtime::ResidentTensorReservation;
use crate::inference::{
    EmbeddedRuntime, ExecutionDigest, ExecutionPermit, GraphExecutionBoundaryMeasurement,
    RuntimeDeviceIdentity, RuntimeDeviceKind, TensorInput, TensorOutput,
};

use super::boundary::{duration_nanos, tensor_bytes};
use super::GraphExecutor;

/// Provider-neutral dtype carried by an opaque resident graph tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GraphTensorDType {
    F32,
}

/// Value-free descriptor for one opaque tensor retained on a runtime device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphTensorDescriptor {
    pub dtype: GraphTensorDType,
    pub shape: Vec<usize>,
    pub device: RuntimeDeviceIdentity,
}

/// One opaque, non-cloneable tensor retained between reviewed graph calls.
///
/// The handle keeps the originating execution permit and aggregate byte
/// reservation alive. It can be consumed only by another graph using the same
/// request permit and runtime, or materialized into an explicit owned fallback.
pub struct ResidentGraphTensor {
    tensor: Tensor,
    runtime: EmbeddedRuntime,
    permit: ExecutionPermit,
    reservation: ResidentTensorReservation,
    input_digest: ExecutionDigest,
    boundary: GraphExecutionBoundaryMeasurement,
}

impl ResidentGraphTensor {
    pub fn descriptor(&self) -> GraphTensorDescriptor {
        GraphTensorDescriptor {
            dtype: GraphTensorDType::F32,
            shape: self.tensor.dims().to_vec(),
            device: self.runtime.device().identity(),
        }
    }

    /// Copies the retained tensor to one owned F32 output and computes the
    /// canonical v1 tensor digest from that same copy.
    ///
    /// The returned input digest is the canonical digest captured before the
    /// first graph in this resident chain. Intermediate graph calls neither
    /// replace it nor require a host round trip.
    pub fn materialize(
        self,
        cancellation: &CancellationToken,
    ) -> Result<ResidentGraphMaterialization> {
        check_cancelled(cancellation)?;
        let output_host_bytes =
            validate_tensor(&self.tensor, &self.runtime, "resident graph output", true)?;
        let output_started = Instant::now();
        let output = TensorOutput::from_candle(&self.tensor, self.runtime.limits())?;
        let output_materialization_nanos = duration_nanos(output_started.elapsed());
        check_cancelled(cancellation)?;
        let output_digest = ExecutionDigest::f32_tensor(&output.shape, &output.values);
        check_cancelled(cancellation)?;
        let device_copy_operations =
            u64::from(self.runtime.device().kind() != RuntimeDeviceKind::Cpu);
        let boundary = self
            .boundary
            .checked_add(GraphExecutionBoundaryMeasurement {
                output_materializations: 1,
                output_host_bytes,
                device_to_host_copy_operations: device_copy_operations,
                output_materialization_nanos,
                ..GraphExecutionBoundaryMeasurement::default()
            })?;
        Ok(ResidentGraphMaterialization {
            output,
            input_digest: self.input_digest,
            output_digest,
            boundary,
        })
    }

    fn validate_for(&self, runtime: &EmbeddedRuntime, permit: &ExecutionPermit) -> Result<()> {
        if !self.permit.same_admission(permit) || !self.permit.belongs_to(runtime) {
            return Err(PowerError::InvalidRequest(
                "resident graph tensor belongs to a different request permit or runtime"
                    .to_string(),
            ));
        }
        validate_tensor(&self.tensor, runtime, "resident graph input", false).map(|_| ())
    }
}

impl std::fmt::Debug for ResidentGraphTensor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentGraphTensor")
            .field("dtype", &GraphTensorDType::F32)
            .field("device", &self.runtime.device().identity())
            .finish_non_exhaustive()
    }
}

/// Owned end of a resident graph chain with canonical input/output digests and
/// aggregate host/device boundary accounting.
#[derive(Clone, PartialEq)]
pub struct ResidentGraphMaterialization {
    pub output: TensorOutput,
    pub input_digest: ExecutionDigest,
    pub output_digest: ExecutionDigest,
    pub boundary: GraphExecutionBoundaryMeasurement,
}

impl std::fmt::Debug for ResidentGraphMaterialization {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentGraphMaterialization")
            .field("output_shape", &self.output.shape)
            .field("input_digest", &self.input_digest)
            .field("output_digest", &self.output_digest)
            .field("boundary", &self.boundary)
            .finish_non_exhaustive()
    }
}

impl GraphExecutor {
    /// Executes from an owned F32 input and retains the reviewed graph output
    /// on the runtime device under the caller's existing request permit.
    pub fn run_to_resident(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidentGraphTensor> {
        validate_call(&self.runtime, permit, cancellation)?;
        input.validate(self.runtime.limits())?;
        let mut shape_bindings = self
            .plan
            .validate_input_shape(&input.shape, self.runtime.limits())?;
        let input_digest = ExecutionDigest::f32_tensor(&input.shape, &input.values);
        check_cancelled(cancellation)?;
        let input_host_bytes = tensor_bytes(input.values.len(), "resident input")?;
        let input_started = Instant::now();
        let input = input.into_candle(self.runtime.device().tensor_device())?;
        let input_materialization_nanos = duration_nanos(input_started.elapsed());
        check_cancelled(cancellation)?;
        let output = self.run_tensor(input, cancellation)?;
        self.plan.validate_output_shape(
            output.dims(),
            &mut shape_bindings,
            self.runtime.limits(),
        )?;
        check_cancelled(cancellation)?;
        let output_bytes = validate_tensor(&output, &self.runtime, "resident graph output", true)?;
        let reservation = self.runtime.reserve_resident_tensor(output_bytes)?;
        check_cancelled(cancellation)?;
        let device_copy_operations =
            u64::from(self.runtime.device().kind() != RuntimeDeviceKind::Cpu);
        Ok(ResidentGraphTensor {
            tensor: output,
            runtime: self.runtime.clone(),
            permit: permit.clone(),
            reservation,
            input_digest,
            boundary: GraphExecutionBoundaryMeasurement {
                input_materializations: 1,
                input_host_bytes,
                host_to_device_copy_operations: device_copy_operations,
                input_materialization_nanos,
                ..GraphExecutionBoundaryMeasurement::default()
            },
        })
    }

    /// Consumes one same-request resident tensor as the input to this reviewed
    /// graph and returns its output without an owned host materialization.
    pub fn run_resident(
        &self,
        input: ResidentGraphTensor,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<ResidentGraphTensor> {
        validate_call(&self.runtime, permit, cancellation)?;
        input.validate_for(&self.runtime, permit)?;
        let mut shape_bindings = self
            .plan
            .validate_input_shape(input.tensor.dims(), self.runtime.limits())?;
        let ResidentGraphTensor {
            tensor,
            runtime: _,
            permit,
            mut reservation,
            input_digest,
            boundary,
        } = input;
        let output = self.run_tensor(tensor, cancellation)?;
        self.plan.validate_output_shape(
            output.dims(),
            &mut shape_bindings,
            self.runtime.limits(),
        )?;
        check_cancelled(cancellation)?;
        let output_bytes = validate_tensor(&output, &self.runtime, "resident graph output", true)?;
        reservation.resize(output_bytes)?;
        check_cancelled(cancellation)?;
        Ok(ResidentGraphTensor {
            tensor: output,
            runtime: self.runtime.clone(),
            permit,
            reservation,
            input_digest,
            boundary,
        })
    }
}

fn validate_call(
    runtime: &EmbeddedRuntime,
    permit: &ExecutionPermit,
    cancellation: &CancellationToken,
) -> Result<()> {
    if !permit.belongs_to(runtime) {
        return Err(PowerError::InvalidRequest(
            "graph execution permit belongs to a different embedded runtime".to_string(),
        ));
    }
    check_cancelled(cancellation)
}

fn validate_tensor(
    tensor: &Tensor,
    runtime: &EmbeddedRuntime,
    label: &str,
    output: bool,
) -> Result<u64> {
    if tensor.dtype() != DType::F32 {
        let message = format!("{label} must be F32, found {:?}", tensor.dtype());
        return Err(if output {
            PowerError::InferenceFailed(message)
        } else {
            PowerError::InvalidRequest(message)
        });
    }
    if !tensor
        .device()
        .same_device(runtime.device().tensor_device())
    {
        let message = format!("{label} belongs to a different tensor device");
        return Err(if output {
            PowerError::InferenceFailed(message)
        } else {
            PowerError::InvalidRequest(message)
        });
    }
    let elements = runtime.limits().checked_elements(tensor.dims(), label)?;
    tensor_bytes(elements, label)
}

fn check_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        Err(PowerError::InferenceFailed(
            "resident graph execution was cancelled".to_string(),
        ))
    } else {
        Ok(())
    }
}
