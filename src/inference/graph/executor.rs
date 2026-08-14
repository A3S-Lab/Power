use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::{EmbeddedRuntime, ExecutionPermit, TensorInput, TensorOutput, WeightStore};
use super::plan::{GraphNode, GraphOp, GraphPlan};
use super::value::GraphValue;

mod depthwise;
mod liveness;

/// Validated single-input/single-output static graph executor.
pub struct GraphExecutor {
    plan: GraphPlan,
    constants: HashMap<String, GraphValue>,
    value_use_counts: HashMap<String, usize>,
    runtime: EmbeddedRuntime,
}

impl GraphExecutor {
    pub fn new(
        plan: GraphPlan,
        weights: Arc<WeightStore>,
        runtime: EmbeddedRuntime,
    ) -> Result<Self> {
        let mut constants = HashMap::with_capacity(plan.initializers.len());
        for initializer in &plan.initializers {
            constants.insert(
                initializer.name.clone(),
                GraphValue::load(initializer, &weights, runtime.device().tensor_device())?,
            );
        }
        let value_use_counts = liveness::value_use_counts(&plan);
        Ok(Self {
            plan,
            constants,
            value_use_counts,
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
        let input = input.into_candle(self.runtime.device().tensor_device())?;
        let output = self.run_tensor(input, cancellation)?;
        TensorOutput::from_candle(&output, self.runtime.limits())
    }

    fn run_tensor(&self, input: Tensor, cancellation: &CancellationToken) -> Result<Tensor> {
        let input_name = self.plan.inputs[0].name.clone();
        let output_name = self.plan.outputs[0].name.clone();
        let mut values = self.constants.clone();
        let mut remaining_uses = self.value_use_counts.clone();
        values.insert(input_name, GraphValue::Tensor(input));
        for node in &self.plan.nodes {
            if cancellation.is_cancelled() {
                return Err(PowerError::InferenceFailed(
                    "static graph execution was cancelled".to_string(),
                ));
            }
            let output = execute(node, &values, self.runtime.device().tensor_device())?;
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
            let limit = self.runtime.limits().max_tensor_elements;
            if elements > limit {
                return Err(PowerError::InferenceFailed(format!(
                    "static graph node '{}' produced {elements} tensor elements, exceeding the {limit}-element limit",
                    node.name,
                )));
            }
            liveness::release_consumed_values(
                &node.inputs,
                &output_name,
                &mut remaining_uses,
                &mut values,
            );
            values.insert(node.outputs[0].clone(), output);
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
    let cuda_depthwise = device.is_cuda()
        && groups == dimensions.1
        && kernel_dimensions.0 == dimensions.1
        && kernel_dimensions.1 == 1;
    let output = if cuda_depthwise {
        depthwise::conv2d(&input, kernel, strides, dilations.0)
    } else {
        input.conv2d(kernel, 0, common_stride, dilations.0, groups)
    }
    .map_err(|error| execution_error(node, error))?;
    let mut output = if !cuda_depthwise && strides.0 != strides.1 {
        subsample_spatial(&output, strides, device, node)?
    } else {
        output
    };
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

fn convolution_pads(
    node: &GraphNode,
    dimensions: (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    dilation: (usize, usize),
) -> Result<(usize, usize, usize, usize)> {
    match node.string("auto_pad", "NOTSET")? {
        "NOTSET" => quad(&node.ints("pads", &[0, 0, 0, 0])?, "pads", node),
        "SAME_UPPER" => {
            let (_, _, height, width) = dimensions;
            let (top, bottom) = same_upper_padding(height, kernel.0, stride.0, dilation.0);
            let (left, right) = same_upper_padding(width, kernel.1, stride.1, dilation.1);
            Ok((top, left, bottom, right))
        }
        other => Err(execution_error(
            node,
            format!("unsupported auto_pad '{other}'"),
        )),
    }
}

fn pool_pads(
    node: &GraphNode,
    dimensions: (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
) -> Result<(usize, usize, usize, usize)> {
    convolution_pads(node, dimensions, kernel, stride, (1, 1))
}

fn same_upper_padding(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    let output = input.div_ceil(stride);
    let effective = dilation * (kernel.saturating_sub(1)) + 1;
    let total = ((output.saturating_sub(1)) * stride + effective).saturating_sub(input);
    (total / 2, total - total / 2)
}

fn pad_spatial(
    input: &Tensor,
    pads: (usize, usize, usize, usize),
    node: &GraphNode,
) -> Result<Tensor> {
    input
        .pad_with_zeros(2, pads.0, pads.2)
        .and_then(|value| value.pad_with_zeros(3, pads.1, pads.3))
        .map_err(|error| execution_error(node, error))
}

fn subsample_spatial(
    input: &Tensor,
    stride: (usize, usize),
    device: &Device,
    node: &GraphNode,
) -> Result<Tensor> {
    let mut output = input.clone();
    for (axis, step) in [(2, stride.0), (3, stride.1)] {
        if step == 1 {
            continue;
        }
        let length = output
            .dim(axis)
            .map_err(|error| execution_error(node, error))?;
        let indices = (0..length)
            .step_by(step)
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| execution_error(node, "subsample index exceeds u32"))?;
        let indices = Tensor::from_vec(indices.clone(), indices.len(), device)
            .map_err(|error| execution_error(node, error))?;
        output = output
            .index_select(&indices, axis)
            .map_err(|error| execution_error(node, error))?;
    }
    Ok(output)
}

fn slice_tensor(
    input: &Tensor,
    axis: usize,
    start: i64,
    end: i64,
    step: i64,
    device: &Device,
    node: &GraphNode,
) -> Result<Tensor> {
    if step <= 0 {
        return Err(execution_error(node, "Slice step must be positive"));
    }
    let length = input
        .dim(axis)
        .map_err(|error| execution_error(node, error))?;
    let (start, end) = slice_bounds(length, start, end, node)?;
    if step == 1 {
        return input
            .narrow(axis, start, end - start)
            .map_err(|error| execution_error(node, error));
    }
    let step = usize::try_from(step).map_err(|_| execution_error(node, "invalid Slice step"))?;
    let indices = (start..end)
        .step_by(step)
        .map(u32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| execution_error(node, "Slice index exceeds u32"))?;
    let indices = Tensor::from_vec(indices.clone(), indices.len(), device)
        .map_err(|error| execution_error(node, error))?;
    input
        .index_select(&indices, axis)
        .map_err(|error| execution_error(node, error))
}

fn slice_bounds(length: usize, start: i64, end: i64, node: &GraphNode) -> Result<(usize, usize)> {
    let length_i64 = i64::try_from(length).map_err(|_| execution_error(node, "axis too large"))?;
    let normalize = |value: i64| {
        if value < 0 {
            (length_i64 + value).max(0)
        } else {
            value.min(length_i64)
        }
    };
    let start = normalize(start);
    let end = normalize(end);
    if end < start {
        return Err(execution_error(node, "Slice end precedes start"));
    }
    Ok((start as usize, end as usize))
}

fn resolve_reshape(input: &[usize], requested: &[i64], node: &GraphNode) -> Result<Vec<usize>> {
    if requested.is_empty() {
        return Err(execution_error(node, "Reshape target must not be empty"));
    }
    let input_elements = input.iter().product::<usize>();
    let mut output = Vec::with_capacity(requested.len());
    let mut inferred = None;
    let mut known = 1_usize;
    for (index, dimension) in requested.iter().copied().enumerate() {
        match dimension {
            -1 if inferred.is_none() => {
                inferred = Some(index);
                output.push(1);
            }
            0 if index < input.len() => {
                known = known
                    .checked_mul(input[index])
                    .ok_or_else(|| execution_error(node, "Reshape dimensions overflowed"))?;
                output.push(input[index]);
            }
            value if value > 0 => {
                let value = usize::try_from(value)
                    .map_err(|_| execution_error(node, "invalid Reshape dimension"))?;
                known = known
                    .checked_mul(value)
                    .ok_or_else(|| execution_error(node, "Reshape dimensions overflowed"))?;
                output.push(value);
            }
            _ => return Err(execution_error(node, "invalid Reshape target")),
        }
    }
    if let Some(index) = inferred {
        if known == 0 || !input_elements.is_multiple_of(known) {
            return Err(execution_error(node, "Reshape target cannot be inferred"));
        }
        output[index] = input_elements / known;
    } else if known != input_elements {
        return Err(execution_error(node, "Reshape changes the element count"));
    }
    Ok(output)
}

fn normalized_axes(axes: &[i64], rank: usize, node: &GraphNode) -> Result<Vec<usize>> {
    axes.iter()
        .map(|axis| axis_index(*axis, rank, node))
        .collect()
}

fn axis_index(axis: i64, rank: usize, node: &GraphNode) -> Result<usize> {
    let rank_i64 = i64::try_from(rank).map_err(|_| execution_error(node, "rank exceeds i64"))?;
    let axis = if axis < 0 { rank_i64 + axis } else { axis };
    if axis < 0 || axis >= rank_i64 {
        return Err(execution_error(
            node,
            format!("axis {axis} is out of range for rank {rank}"),
        ));
    }
    Ok(axis as usize)
}

fn pair(values: &[i64], name: &str, node: &GraphNode) -> Result<(usize, usize)> {
    if values.len() != 2 {
        return Err(execution_error(
            node,
            format!("{name} must contain two values"),
        ));
    }
    Ok((
        positive_usize(values[0], name, node)?,
        positive_usize(values[1], name, node)?,
    ))
}

fn quad(values: &[i64], name: &str, node: &GraphNode) -> Result<(usize, usize, usize, usize)> {
    if values.len() != 4 {
        return Err(execution_error(
            node,
            format!("{name} must contain four values"),
        ));
    }
    Ok((
        nonnegative_usize(values[0], name, node)?,
        nonnegative_usize(values[1], name, node)?,
        nonnegative_usize(values[2], name, node)?,
        nonnegative_usize(values[3], name, node)?,
    ))
}

fn positive_usize(value: i64, name: &str, node: &GraphNode) -> Result<usize> {
    if value <= 0 {
        return Err(execution_error(node, format!("{name} must be positive")));
    }
    usize::try_from(value).map_err(|_| execution_error(node, format!("{name} exceeds usize")))
}

fn nonnegative_usize(value: i64, name: &str, node: &GraphNode) -> Result<usize> {
    if value < 0 {
        return Err(execution_error(
            node,
            format!("{name} must be non-negative"),
        ));
    }
    usize::try_from(value).map_err(|_| execution_error(node, format!("{name} exceeds usize")))
}

fn execution_error(node: &GraphNode, error: impl std::fmt::Display) -> PowerError {
    PowerError::InferenceFailed(format!(
        "static graph node '{}' ({:?}) failed: {error}",
        node.name, node.op
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node() -> GraphNode {
        GraphNode {
            name: "test".to_string(),
            op: GraphOp::Reshape,
            inputs: Vec::new(),
            outputs: vec!["out".to_string()],
            attributes: Default::default(),
        }
    }

    #[test]
    fn reshape_resolves_zero_and_inferred_dimensions() {
        assert_eq!(
            resolve_reshape(&[2, 3, 4], &[0, -1], &node()).unwrap(),
            [2, 12]
        );
        assert!(resolve_reshape(&[2, 3], &[-1, -1], &node()).is_err());
    }

    #[test]
    fn same_upper_padding_puts_odd_pixel_at_end() {
        assert_eq!(same_upper_padding(48, 2, 1, 1), (0, 1));
        assert_eq!(same_upper_padding(48, 3, 2, 1), (0, 1));
    }

    #[test]
    fn matmul_materializes_a_transposed_rhs_view() {
        let mut node = node();
        node.op = GraphOp::MatMul;
        let left = GraphValue::Tensor(
            Tensor::zeros((3, 8, 41, 15), candle_core::DType::F32, &Device::Cpu).unwrap(),
        );
        let right = Tensor::zeros((3, 8, 41, 15), candle_core::DType::F32, &Device::Cpu)
            .unwrap()
            .transpose(2, 3)
            .unwrap();
        assert!(!right.is_contiguous());
        let right = GraphValue::Tensor(right);

        let output = matmul(&node, &[&left, &right]).unwrap();

        assert_eq!(output.shape(), [3, 8, 41, 41]);
    }
}
