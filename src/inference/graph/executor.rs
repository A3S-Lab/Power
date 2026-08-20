use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::{
    EmbeddedRuntime, ExecutionPermit, GraphExecutionBoundaryMeasurement, HardwareEvidenceBinding,
    RuntimeDeviceKind, RuntimeMemoryReservations, ShapeProfileBinding, StorageBenchmarkSystem,
    TensorInput, TensorOutput, WeightStore,
};
use super::plan::{GraphNode, GraphOp, GraphPlan};
use super::value::GraphValue;

mod biased_activation;
mod depthwise;
mod gated_hard_sigmoid;
mod gelu_erf;
mod layer_norm_affine;
mod liveness;
mod support;

use support::{
    axis_index, convolution_pads, execution_error, nonnegative_usize, normalized_axes, pad_spatial,
    pair, pool_pads, positive_usize, quad, resolve_reshape, slice_bounds, slice_tensor,
    subsample_spatial,
};

/// Validated single-input/single-output static graph executor.
pub struct GraphExecutor {
    plan: GraphPlan,
    constants: HashMap<String, GraphValue>,
    scalar_constants: HashMap<String, f32>,
    value_use_counts: HashMap<String, usize>,
    weights_sha256: String,
    runtime: EmbeddedRuntime,
}

impl GraphExecutor {
    pub fn new(
        plan: GraphPlan,
        weights: Arc<WeightStore>,
        runtime: EmbeddedRuntime,
    ) -> Result<Self> {
        let weights_sha256 = weights.sha256().to_string();
        let mut constants = HashMap::with_capacity(plan.initializers.len());
        let mut scalar_constants = HashMap::new();
        for initializer in &plan.initializers {
            let (value, scalar) =
                GraphValue::load(initializer, &weights, runtime.device().tensor_device())?;
            constants.insert(initializer.name.clone(), value);
            if let Some(scalar) = scalar {
                scalar_constants.insert(initializer.name.clone(), scalar);
            }
        }
        let value_use_counts = liveness::value_use_counts(&plan);
        Ok(Self {
            plan,
            constants,
            scalar_constants,
            value_use_counts,
            weights_sha256,
            runtime,
        })
    }

    /// Executes a graph under a permit from the same shared runtime.
    pub fn run(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<TensorOutput> {
        self.run_with_output_projection(input, permit, cancellation, |output| Ok(output.clone()))
    }

    /// Executes a graph while measuring only its owned-host-tensor boundary.
    /// Kernel time remains part of the caller's end-to-end sample rather than
    /// being misreported as copy time.
    pub fn run_measured(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
    ) -> Result<(TensorOutput, GraphExecutionBoundaryMeasurement)> {
        self.run_with_output_projection_measured(input, permit, cancellation, |output| {
            Ok(output.clone())
        })
    }

    /// Executes a graph and applies one model-owned projection before the
    /// bounded output is copied back to the host.
    ///
    /// The projection remains on the graph device and is useful when a model
    /// consumes a compact deterministic view of a much larger graph output.
    /// The model crate owns the projection arithmetic and must bind it into
    /// its execution identity. Power still enforces permit, cancellation,
    /// device-residency, tensor-element, dtype, and finite-output bounds.
    pub fn run_with_output_projection<F>(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<TensorOutput>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        self.run_with_output_projection_measured(input, permit, cancellation, projection)
            .map(|(output, _)| output)
    }

    /// Measured form of [`Self::run_with_output_projection`]. The returned
    /// counters describe the generic host/device boundary only; they do not
    /// claim visibility into backend allocator or DMA internals.
    pub fn run_with_output_projection_measured<F>(
        &self,
        input: TensorInput,
        permit: &ExecutionPermit,
        cancellation: &CancellationToken,
        projection: F,
    ) -> Result<(TensorOutput, GraphExecutionBoundaryMeasurement)>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        if !permit.belongs_to(&self.runtime) {
            return Err(PowerError::InvalidRequest(
                "graph execution permit belongs to a different embedded runtime".to_string(),
            ));
        }
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        input.validate(self.runtime.limits())?;
        let input_host_bytes = tensor_bytes(input.values.len(), "input")?;
        let input_started = Instant::now();
        let input = input.into_candle(self.runtime.device().tensor_device())?;
        let input_materialization_nanos = duration_nanos(input_started.elapsed());
        let output = self.run_tensor(input, cancellation)?;
        let projected = projection(&output)?;
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        if !projected.device().same_device(output.device()) {
            return Err(PowerError::InferenceFailed(
                "static graph output projection changed tensor devices".to_string(),
            ));
        }
        let output_elements = self
            .runtime
            .limits()
            .checked_elements(projected.dims(), "projected output tensor")?;
        let output_host_bytes = tensor_bytes(output_elements, "output")?;
        let output_started = Instant::now();
        let output = TensorOutput::from_candle(&projected, self.runtime.limits())?;
        let output_materialization_nanos = duration_nanos(output_started.elapsed());
        let device_copy_operations =
            u64::from(self.runtime.device().kind() != RuntimeDeviceKind::Cpu);
        Ok((
            output,
            GraphExecutionBoundaryMeasurement {
                input_materializations: 1,
                input_host_bytes,
                host_to_device_copy_operations: device_copy_operations,
                input_materialization_nanos,
                output_materializations: 1,
                output_host_bytes,
                device_to_host_copy_operations: device_copy_operations,
                output_materialization_nanos,
            },
        ))
    }

    pub(crate) fn runtime(&self) -> &EmbeddedRuntime {
        &self.runtime
    }

    /// Derives the current finite shape-profile binding from the validated
    /// graph, exact weight collection, and resolved single-device executor.
    /// Model crates still own the opaque shape classes and TEE policy digest.
    pub fn shape_profile_binding(
        &self,
        runtime_reservations: RuntimeMemoryReservations,
        tee_policy_sha256: impl Into<String>,
    ) -> Result<ShapeProfileBinding> {
        let graph_sha256 = self.plan.identity().binding_sha256()?;
        ShapeProfileBinding::for_single_device(
            &self.weights_sha256,
            graph_sha256,
            self.runtime.device().identity(),
            runtime_reservations,
            tee_policy_sha256,
        )
    }

    pub(crate) fn benchmark_binding(
        &self,
        power_commit: &str,
        system: &StorageBenchmarkSystem,
    ) -> Result<HardwareEvidenceBinding> {
        HardwareEvidenceBinding::new(
            env!("CARGO_PKG_VERSION"),
            power_commit,
            &self.weights_sha256,
            self.plan.identity().source_sha256,
            self.runtime.device().identity(),
            system,
        )
    }

    fn run_tensor(&self, input: Tensor, cancellation: &CancellationToken) -> Result<Tensor> {
        let input_name = self.plan.inputs[0].name.clone();
        let output_name = self.plan.outputs[0].name.clone();
        let mut values = self.constants.clone();
        let mut remaining_uses = self.value_use_counts.clone();
        values.insert(input_name, GraphValue::Tensor(input));
        let mut node_index = 0;
        while let Some(node) = self.plan.nodes.get(node_index) {
            if cancellation.is_cancelled() {
                return Err(PowerError::InferenceFailed(
                    "static graph execution was cancelled".to_string(),
                ));
            }
            if let Some(fused) = biased_activation::try_execute(
                &self.plan.nodes[node_index..],
                biased_activation::ExecutionContext {
                    values: &values,
                    scalar_constants: &self.scalar_constants,
                    use_counts: &self.value_use_counts,
                    retained_output: &output_name,
                    device: self.runtime.device().tensor_device(),
                    element_limit: self.runtime.limits().max_tensor_elements,
                    cancellation,
                },
            )? {
                let window = &self.plan.nodes[node_index..node_index + fused.consumed_nodes];
                for fused_node in &window[..window.len() - 1] {
                    liveness::release_consumed_values(
                        &fused_node.inputs,
                        &output_name,
                        &mut remaining_uses,
                        &mut values,
                    );
                }
                commit_node_output(
                    &window[window.len() - 1],
                    fused.value,
                    &output_name,
                    self.runtime.limits().max_tensor_elements,
                    &mut remaining_uses,
                    &mut values,
                )?;
                node_index += fused.consumed_nodes;
                continue;
            }
            if let Some(window) = self.plan.nodes.get(node_index..node_index + 5) {
                if let Some(output) = layer_norm_affine::try_execute(
                    window,
                    &values,
                    &self.scalar_constants,
                    &self.value_use_counts,
                    &output_name,
                    self.runtime.limits().max_tensor_elements,
                    cancellation,
                )? {
                    for fused_node in &window[..4] {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            &output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        &window[4],
                        output,
                        &output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    node_index += 5;
                    continue;
                }
            }
            if let Some(window) = self.plan.nodes.get(node_index..node_index + 5) {
                if let Some(output) = gelu_erf::try_execute(
                    window,
                    &values,
                    &self.scalar_constants,
                    &self.value_use_counts,
                    &output_name,
                    cancellation,
                )? {
                    for fused_node in &window[..4] {
                        liveness::release_consumed_values(
                            &fused_node.inputs,
                            &output_name,
                            &mut remaining_uses,
                            &mut values,
                        );
                    }
                    commit_node_output(
                        &window[4],
                        output,
                        &output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    node_index += 5;
                    continue;
                }
            }
            if let Some(next) = self.plan.nodes.get(node_index + 1) {
                if let Some(output) = gated_hard_sigmoid::try_mul(
                    node,
                    next,
                    &values,
                    &self.value_use_counts,
                    &output_name,
                    cancellation,
                )? {
                    liveness::release_consumed_values(
                        &node.inputs,
                        &output_name,
                        &mut remaining_uses,
                        &mut values,
                    );
                    commit_node_output(
                        next,
                        output,
                        &output_name,
                        self.runtime.limits().max_tensor_elements,
                        &mut remaining_uses,
                        &mut values,
                    )?;
                    node_index += 2;
                    continue;
                }
            }
            let output = execute(node, &values, self.runtime.device().tensor_device())?;
            commit_node_output(
                node,
                output,
                &output_name,
                self.runtime.limits().max_tensor_elements,
                &mut remaining_uses,
                &mut values,
            )?;
            node_index += 1;
        }
        values
            .remove(&output_name)
            .ok_or_else(|| {
                PowerError::InferenceFailed("static graph returned no output".to_string())
            })?
            .tensor("graph output")
            .cloned()
    }
}

fn tensor_bytes(elements: usize, label: &str) -> Result<u64> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph {label} tensor byte count overflowed"
            ))
        })?;
    u64::try_from(bytes).map_err(|_| {
        PowerError::InferenceFailed(format!(
            "static graph {label} tensor byte count exceeds u64"
        ))
    })
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn commit_node_output(
    node: &GraphNode,
    output: GraphValue,
    retained_output: &str,
    element_limit: usize,
    remaining_uses: &mut HashMap<String, usize>,
    values: &mut HashMap<String, GraphValue>,
) -> Result<()> {
    #[cfg(test)]
    trace_non_finite(node, &output)?;
    let elements = output
        .shape()
        .iter()
        .try_fold(1_usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' tensor element count overflowed",
                node.name
            ))
        })?;
    if elements > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {elements} tensor elements, exceeding the {element_limit}-element limit",
            node.name,
        )));
    }
    liveness::release_consumed_values(&node.inputs, retained_output, remaining_uses, values);
    values.insert(node.outputs[0].clone(), output);
    Ok(())
}

#[cfg(test)]
fn trace_non_finite(node: &GraphNode, value: &GraphValue) -> Result<()> {
    if std::env::var_os("A3S_POWER_TRACE_NONFINITE").is_none() {
        return Ok(());
    }
    let GraphValue::Tensor(tensor) = value else {
        return Ok(());
    };
    let values = tensor
        .to_dtype(candle_core::DType::F32)
        .and_then(|value| value.to_device(&Device::Cpu))
        .and_then(|value| value.flatten_all())
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| execution_error(node, error))?;
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(execution_error(
            node,
            format!("produced non-finite value {value} at flat index {index}"),
        ));
    }
    Ok(())
}

fn execute(
    node: &GraphNode,
    values: &HashMap<String, GraphValue>,
    device: &Device,
) -> Result<GraphValue> {
    let inputs = node
        .inputs
        .iter()
        .map(|name| {
            values.get(name).ok_or_else(|| {
                PowerError::InferenceFailed(format!(
                    "static graph node '{}' could not resolve input '{name}'",
                    node.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let value = match node.op {
        GraphOp::Add => binary(node, &inputs, Tensor::broadcast_add)?,
        GraphOp::Sub => binary(node, &inputs, Tensor::broadcast_sub)?,
        GraphOp::Mul => binary(node, &inputs, Tensor::broadcast_mul)?,
        GraphOp::Div => binary(node, &inputs, Tensor::broadcast_div)?,
        GraphOp::Pow => pow(node, &inputs)?,
        GraphOp::Erf => unary_tensor(node, &inputs, Tensor::erf)?,
        GraphOp::Relu => unary_tensor(node, &inputs, Tensor::relu)?,
        GraphOp::Sqrt => unary_tensor(node, &inputs, Tensor::sqrt)?,
        GraphOp::Sigmoid => GraphValue::Tensor(
            candle_nn::ops::sigmoid(required_tensor(node, &inputs, 0)?)
                .map_err(|error| execution_error(node, error))?,
        ),
        GraphOp::HardSigmoid => hard_sigmoid(node, &inputs)?,
        GraphOp::Identity => required(node, &inputs, 0)?.clone(),
        GraphOp::Concat => concat(node, &inputs)?,
        GraphOp::ReduceMean => reduce_mean(node, &inputs)?,
        GraphOp::GlobalAveragePool => global_average_pool(node, &inputs)?,
        GraphOp::Conv => conv(node, &inputs, device)?,
        GraphOp::ConvTranspose => conv_transpose(node, &inputs)?,
        GraphOp::MaxPool => pool(node, &inputs, true)?,
        GraphOp::AveragePool => pool(node, &inputs, false)?,
        GraphOp::Resize => resize(node, &inputs)?,
        GraphOp::BatchNormalization => batch_norm(node, &inputs)?,
        GraphOp::MatMul => matmul(node, &inputs)?,
        GraphOp::Reshape => reshape(node, &inputs)?,
        GraphOp::Shape => GraphValue::Ints {
            values: required(node, &inputs, 0)?
                .shape()
                .iter()
                .map(|value| i64::try_from(*value))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| execution_error(node, "shape dimension exceeds i64"))?,
            shape: vec![required(node, &inputs, 0)?.shape().len()],
        },
        GraphOp::Slice => slice(node, &inputs, device)?,
        GraphOp::Squeeze => squeeze(node, &inputs)?,
        GraphOp::Unsqueeze => unsqueeze(node, &inputs)?,
        GraphOp::Transpose => transpose(node, &inputs)?,
        GraphOp::Softmax => softmax(node, &inputs)?,
    };
    Ok(value)
}

fn required<'a>(
    node: &GraphNode,
    inputs: &'a [&GraphValue],
    index: usize,
) -> Result<&'a GraphValue> {
    inputs.get(index).copied().ok_or_else(|| {
        PowerError::InvalidFormat(format!(
            "static graph node '{}' is missing input {index}",
            node.name
        ))
    })
}

fn required_tensor<'a>(
    node: &GraphNode,
    inputs: &'a [&GraphValue],
    index: usize,
) -> Result<&'a Tensor> {
    required(node, inputs, index)?.tensor(&node.name)
}

fn binary(
    node: &GraphNode,
    inputs: &[&GraphValue],
    operation: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
) -> Result<GraphValue> {
    let left = required_tensor(node, inputs, 0)?;
    let right = required_tensor(node, inputs, 1)?;
    operation(left, right)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn matmul(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    // ONNX Transpose and Slice legitimately produce strided views. Candle's
    // matmul kernels require contiguous operands, so materialize only this
    // operator boundary instead of rejecting a valid reviewed graph.
    let left = required_tensor(node, inputs, 0)?
        .contiguous()
        .map_err(|error| execution_error(node, error))?;
    let right = required_tensor(node, inputs, 1)?
        .contiguous()
        .map_err(|error| execution_error(node, error))?;
    left.broadcast_matmul(&right)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn unary_tensor(
    node: &GraphNode,
    inputs: &[&GraphValue],
    operation: fn(&Tensor) -> candle_core::Result<Tensor>,
) -> Result<GraphValue> {
    operation(required_tensor(node, inputs, 0)?)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn pow(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let base = required_tensor(node, inputs, 0)?;
    let exponent = required_tensor(node, inputs, 1)?;
    let exponent = exponent
        .to_dtype(candle_core::DType::F32)
        .and_then(|value| value.to_device(&Device::Cpu))
        .and_then(|value| value.flatten_all())
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| execution_error(node, error))?;
    if exponent.as_slice() != [2.0] {
        return Err(execution_error(
            node,
            "the static graph executor only permits a scalar square exponent",
        ));
    }
    base.sqr()
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn hard_sigmoid(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let alpha = node.float("alpha", 0.2)?;
    let beta = node.float("beta", 0.5)?;
    let value = (required_tensor(node, inputs, 0)? * alpha)
        .and_then(|value| value.affine(1.0, beta))
        .and_then(|value| value.clamp(0.0, 1.0))
        .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(value))
}

fn concat(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axis = node.int("axis", 0)?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(_) => {
            let tensors = inputs
                .iter()
                .map(|value| value.tensor(&node.name))
                .collect::<Result<Vec<_>>>()?;
            let rank = tensors[0].rank();
            let axis = axis_index(axis, rank, node)?;
            Tensor::cat(&tensors, axis)
                .map(GraphValue::Tensor)
                .map_err(|error| execution_error(node, error))
        }
        GraphValue::Ints { .. } => {
            if axis != 0 {
                return Err(execution_error(
                    node,
                    "control concatenation axis must be zero",
                ));
            }
            let mut values = Vec::new();
            for value in inputs {
                values.extend_from_slice(value.ints(&node.name)?);
            }
            let length = values.len();
            Ok(GraphValue::Ints {
                values,
                shape: vec![length],
            })
        }
    }
}

fn reduce_mean(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let axes = node.ints("axes", &[])?;
    let axes = normalized_axes(&axes, input.rank(), node)?;
    let keep = node.int("keepdims", 1)? != 0;
    let output = if keep {
        input.mean_keepdim(axes.as_slice())
    } else {
        input.mean(axes.as_slice())
    }
    .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

fn global_average_pool(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    if input.rank() < 3 {
        return Err(execution_error(
            node,
            "global average pool requires rank >= 3",
        ));
    }
    let axes = (2..input.rank()).collect::<Vec<_>>();
    input
        .mean_keepdim(axes.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn conv(node: &GraphNode, inputs: &[&GraphValue], device: &Device) -> Result<GraphValue> {
    let mut input = required_tensor(node, inputs, 0)?.clone();
    let kernel = required_tensor(node, inputs, 1)?;
    let kernel_shape = pair(&node.ints("kernel_shape", &[])?, "kernel_shape", node)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dilations = pair(&node.ints("dilations", &[1, 1])?, "dilations", node)?;
    if dilations.0 != dilations.1 {
        return Err(execution_error(
            node,
            "mixed convolution dilation is unsupported",
        ));
    }
    let groups = positive_usize(node.int("group", 1)?, "group", node)?;
    let dimensions = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let kernel_dimensions = kernel
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let pads = convolution_pads(node, dimensions, kernel_shape, strides, dilations)?;
    input = pad_spatial(&input, pads, node)?;
    let common_stride = if strides.0 == strides.1 { strides.0 } else { 1 };
    let bias = inputs
        .get(2)
        .map(|value| value.tensor(&node.name))
        .transpose()?;
    let cuda_depthwise = device.is_cuda()
        && groups == dimensions.1
        && kernel_dimensions.0 == dimensions.1
        && kernel_dimensions.1 == 1
        && input.dtype() == DType::F32
        && kernel.dtype() == DType::F32
        && bias.is_none_or(|value| value.dtype() == DType::F32);
    let output = if cuda_depthwise {
        depthwise::conv2d(&input, kernel, bias, strides, dilations.0)
    } else {
        input.conv2d(kernel, 0, common_stride, dilations.0, groups)
    }
    .map_err(|error| execution_error(node, error))?;
    let mut output = if !cuda_depthwise && strides.0 != strides.1 {
        subsample_spatial(&output, strides, device, node)?
    } else {
        output
    };
    if !cuda_depthwise {
        if let Some(bias) = bias {
            let channels = bias.dims1().map_err(|error| execution_error(node, error))?;
            output = output
                .broadcast_add(
                    &bias
                        .reshape((1, channels, 1, 1))
                        .map_err(|error| execution_error(node, error))?,
                )
                .map_err(|error| execution_error(node, error))?;
        }
    }
    Ok(GraphValue::Tensor(output))
}

fn conv_transpose(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let kernel = required_tensor(node, inputs, 1)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dilations = pair(&node.ints("dilations", &[1, 1])?, "dilations", node)?;
    let pads = quad(&node.ints("pads", &[0, 0, 0, 0])?, "pads", node)?;
    if strides.0 != strides.1
        || dilations.0 != dilations.1
        || pads.0 != pads.1
        || pads.0 != pads.2
        || pads.0 != pads.3
        || node.int("group", 1)? != 1
    {
        return Err(execution_error(
            node,
            "asymmetric or grouped transposed convolution is unsupported",
        ));
    }
    let mut output = input
        .conv_transpose2d(kernel, pads.0, 0, strides.0, dilations.0)
        .map_err(|error| execution_error(node, error))?;
    if let Some(bias) = inputs.get(2) {
        let bias = bias.tensor(&node.name)?;
        let channels = bias.dims1().map_err(|error| execution_error(node, error))?;
        output = output
            .broadcast_add(
                &bias
                    .reshape((1, channels, 1, 1))
                    .map_err(|error| execution_error(node, error))?,
            )
            .map_err(|error| execution_error(node, error))?;
    }
    Ok(GraphValue::Tensor(output))
}

fn pool(node: &GraphNode, inputs: &[&GraphValue], maximum: bool) -> Result<GraphValue> {
    let mut input = required_tensor(node, inputs, 0)?.clone();
    let kernel = pair(&node.ints("kernel_shape", &[])?, "kernel_shape", node)?;
    let strides = pair(&node.ints("strides", &[1, 1])?, "strides", node)?;
    let dimensions = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let pads = pool_pads(node, dimensions, kernel, strides)?;
    input = pad_spatial(&input, pads, node)?;
    let output = if maximum {
        input.max_pool2d_with_stride(kernel, strides)
    } else {
        if node.int("count_include_pad", 0)? != 0 {
            return Err(execution_error(node, "count_include_pad is unsupported"));
        }
        input.avg_pool2d_with_stride(kernel, strides)
    }
    .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

fn resize(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let (_, _, height, width) = input
        .dims4()
        .map_err(|error| execution_error(node, error))?;
    let mode = node.string("mode", "nearest")?;
    if mode != "nearest"
        || node.string("coordinate_transformation_mode", "half_pixel")? != "asymmetric"
        || node.string("nearest_mode", "round_prefer_floor")? != "floor"
    {
        return Err(execution_error(node, "unsupported Resize policy"));
    }
    let scales = inputs
        .get(2)
        .ok_or_else(|| execution_error(node, "Resize requires scale factors"))?
        .tensor(&node.name)?
        .flatten_all()
        .and_then(|value| value.to_vec1::<f32>())
        .map_err(|error| execution_error(node, error))?;
    if scales.len() != 4 || scales[0] != 1.0 || scales[1] != 1.0 {
        return Err(execution_error(
            node,
            "Resize scales must be NCHW spatial scales",
        ));
    }
    let target_height = ((height as f64) * f64::from(scales[2])).floor() as usize;
    let target_width = ((width as f64) * f64::from(scales[3])).floor() as usize;
    input
        .upsample_nearest2d(target_height, target_width)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn batch_norm(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let channels = input.dim(1).map_err(|error| execution_error(node, error))?;
    let broadcast = |index| -> Result<Tensor> {
        required_tensor(node, inputs, index)?
            .reshape((1, channels, 1, 1))
            .map_err(|error| execution_error(node, error))
    };
    let scale = broadcast(1)?;
    let bias = broadcast(2)?;
    let mean = broadcast(3)?;
    let variance = broadcast(4)?;
    let epsilon = node.float("epsilon", 1e-5)?;
    let output = input
        .broadcast_sub(&mean)
        .and_then(|value| {
            variance
                .affine(1.0, epsilon)
                .and_then(|variance| variance.sqrt())
                .and_then(|stddev| value.broadcast_div(&stddev))
        })
        .and_then(|value| value.broadcast_mul(&scale))
        .and_then(|value| value.broadcast_add(&bias))
        .map_err(|error| execution_error(node, error))?;
    Ok(GraphValue::Tensor(output))
}

fn reshape(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let requested = required(node, inputs, 1)?.ints(&node.name)?;
    let shape = resolve_reshape(input.dims(), requested, node)?;
    input
        .reshape(shape.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn slice(node: &GraphNode, inputs: &[&GraphValue], device: &Device) -> Result<GraphValue> {
    let starts = required(node, inputs, 1)?.ints(&node.name)?;
    let ends = required(node, inputs, 2)?.ints(&node.name)?;
    let default_axes = (0..starts.len())
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    let axes = inputs
        .get(3)
        .map(|value| value.ints(&node.name))
        .transpose()?
        .unwrap_or(default_axes.as_slice());
    let default_steps = vec![1_i64; starts.len()];
    let steps = inputs
        .get(4)
        .map(|value| value.ints(&node.name))
        .transpose()?
        .unwrap_or(default_steps.as_slice());
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != steps.len() {
        return Err(execution_error(
            node,
            "Slice controls have different lengths",
        ));
    }
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let mut output = input.clone();
            for (((start, end), axis), step) in starts.iter().zip(ends).zip(axes).zip(steps) {
                let axis = axis_index(*axis, output.rank(), node)?;
                output = slice_tensor(&output, axis, *start, *end, *step, device, node)?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            if shape.len() != 1 || axes != [0] || steps != [1] {
                return Err(execution_error(node, "unsupported control Slice layout"));
            }
            let (start, end) = slice_bounds(values.len(), starts[0], ends[0], node)?;
            let values = values[start..end].to_vec();
            let length = values.len();
            Ok(GraphValue::Ints {
                values,
                shape: vec![length],
            })
        }
    }
}

fn squeeze(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axes = node.ints("axes", &[])?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let mut axes = normalized_axes(&axes, input.rank(), node)?;
            axes.sort_unstable_by(|left, right| right.cmp(left));
            let mut output = input.clone();
            for axis in axes {
                output = output
                    .squeeze(axis)
                    .map_err(|error| execution_error(node, error))?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            let mut shape = shape.clone();
            let mut axes = normalized_axes(&axes, shape.len(), node)?;
            axes.sort_unstable_by(|left, right| right.cmp(left));
            for axis in axes {
                if shape[axis] != 1 {
                    return Err(execution_error(node, "cannot squeeze a non-unit dimension"));
                }
                shape.remove(axis);
            }
            Ok(GraphValue::Ints {
                values: values.clone(),
                shape,
            })
        }
    }
}

fn unsqueeze(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let axes = node.ints("axes", &[])?;
    match required(node, inputs, 0)? {
        GraphValue::Tensor(input) => {
            let final_rank = input.rank() + axes.len();
            let mut axes = normalized_axes(&axes, final_rank, node)?;
            axes.sort_unstable();
            let mut output = input.clone();
            for axis in axes {
                output = output
                    .unsqueeze(axis)
                    .map_err(|error| execution_error(node, error))?;
            }
            Ok(GraphValue::Tensor(output))
        }
        GraphValue::Ints { values, shape } => {
            let final_rank = shape.len() + axes.len();
            let mut axes = normalized_axes(&axes, final_rank, node)?;
            axes.sort_unstable();
            let mut shape = shape.clone();
            for axis in axes {
                shape.insert(axis, 1);
            }
            Ok(GraphValue::Ints {
                values: values.clone(),
                shape,
            })
        }
    }
}

fn transpose(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let default = (0..input.rank())
        .rev()
        .map(|value| value as i64)
        .collect::<Vec<_>>();
    let permutation = node.ints("perm", &default)?;
    let permutation = permutation
        .into_iter()
        .map(|value| nonnegative_usize(value, "perm", node))
        .collect::<Result<Vec<_>>>()?;
    let mut reviewed = permutation.clone();
    reviewed.sort_unstable();
    if reviewed != (0..input.rank()).collect::<Vec<_>>() {
        return Err(execution_error(node, "perm must be a complete permutation"));
    }
    input
        .permute(permutation.as_slice())
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}

fn softmax(node: &GraphNode, inputs: &[&GraphValue]) -> Result<GraphValue> {
    let input = required_tensor(node, inputs, 0)?;
    let axis = axis_index(node.int("axis", -1)?, input.rank(), node)?;
    candle_nn::ops::softmax(input, axis)
        .map(GraphValue::Tensor)
        .map_err(|error| execution_error(node, error))
}
