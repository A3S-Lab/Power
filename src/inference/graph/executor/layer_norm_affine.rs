use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

mod cpu;
#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes an exact private decomposition of last-axis LayerNorm as one CUDA
/// operation. The matcher is based only on graph topology, scalar ownership,
/// and tensor shape; unsupported graphs retain the existing operator path.
pub(super) fn try_execute_full(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<GraphValue>> {
    let Some(matched) = matched_full_inputs(nodes, scalar_constants, use_counts, retained_output)
    else {
        return Ok(None);
    };
    let input = tensor(values, matched.input, &nodes[0])?;
    let scale = tensor(values, matched.scale, &nodes[7])?;
    let bias = tensor(values, matched.bias, &nodes[8])?;
    let epsilon = scalar_constants[matched.epsilon];
    if !epsilon.is_finite()
        || epsilon <= 0.0
        || !is_last_axis(matched.mean_axis, input.rank())
        || !is_last_axis(matched.variance_axis, input.rank())
        || input.dtype() != DType::F32
        || scale.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !input.device().is_cuda()
        || !scale.device().same_device(input.device())
        || !bias.device().same_device(input.device())
        || !scale.is_contiguous()
        || !bias.is_contiguous()
        || !full_layer_norm_shapes(input.dims(), scale.dims(), bias.dims())
        || input.elem_count() == 0
        || u32::try_from(input.elem_count()).is_err()
    {
        return Ok(None);
    }
    if input.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            nodes[8].name,
            input.elem_count(),
        )));
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }

    #[cfg(feature = "embedded-cuda")]
    {
        let output = cuda::execute_full(input, scale, bias, epsilon).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused full LayerNorm failed: {error}",
                nodes[0].name, nodes[8].name
            ))
        })?;
        Ok(Some(GraphValue::Tensor(output)))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

/// Executes the pointwise tail of a decomposed last-axis LayerNorm in one
/// device operation while retaining the graph's existing mean and variance
/// reductions. Implementations preserve every original f32 rounding boundary.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<GraphValue>> {
    let Some(matched) = matched_inputs(nodes, scalar_constants, use_counts, retained_output) else {
        return Ok(None);
    };
    let centered = tensor(values, matched.centered, &nodes[0])?;
    let variance = tensor(values, matched.variance, &nodes[0])?;
    let scale = tensor(values, matched.scale, &nodes[3])?;
    let bias = tensor(values, matched.bias, &nodes[4])?;
    let epsilon = scalar_constants[matched.epsilon];
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Ok(None);
    }
    if [centered, variance, scale, bias].iter().any(|tensor| {
        tensor.dtype() != DType::F32
            || !tensor.device().same_device(centered.device())
            || !tensor.is_contiguous()
    }) {
        return Ok(None);
    }
    if !centered.device().is_cpu() && !centered.device().is_cuda() {
        return Ok(None);
    }
    if centered.elem_count() == 0
        || u32::try_from(centered.elem_count()).is_err()
        || !last_axis_layer_norm_shapes(centered.dims(), variance.dims(), scale.dims(), bias.dims())
    {
        return Ok(None);
    }
    if centered.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            nodes[4].name,
            centered.elem_count(),
        )));
    }

    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }
    if centered.device().is_cpu() {
        let output = cpu::execute(centered, variance, scale, bias, epsilon).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused CPU LayerNorm affine tail failed: {error}",
                nodes[0].name, nodes[4].name
            ))
        })?;
        return Ok(Some(GraphValue::Tensor(output)));
    }

    #[cfg(feature = "embedded-cuda")]
    if centered.device().is_cuda() {
        let output = cuda::execute(centered, variance, scale, bias, epsilon).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused LayerNorm affine tail failed: {error}",
                nodes[0].name, nodes[4].name
            ))
        })?;
        return Ok(Some(GraphValue::Tensor(output)));
    }

    Ok(None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatchedInputs<'a> {
    centered: &'a str,
    variance: &'a str,
    epsilon: &'a str,
    scale: &'a str,
    bias: &'a str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatchedFullInputs<'a> {
    input: &'a str,
    mean_axis: i64,
    variance_axis: i64,
    exponent: &'a str,
    epsilon: &'a str,
    scale: &'a str,
    bias: &'a str,
}

fn matched_full_inputs<'a>(
    nodes: &'a [GraphNode],
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedFullInputs<'a>> {
    let [mean, subtract, power, variance, add_epsilon, sqrt, divide, multiply, add_bias] = nodes
    else {
        return None;
    };
    let mean_axis = last_axis_mean(mean)?;
    let variance_axis = last_axis_mean(variance)?;
    if [
        subtract.op,
        power.op,
        add_epsilon.op,
        sqrt.op,
        divide.op,
        multiply.op,
        add_bias.op,
    ] != [
        GraphOp::Sub,
        GraphOp::Pow,
        GraphOp::Add,
        GraphOp::Sqrt,
        GraphOp::Div,
        GraphOp::Mul,
        GraphOp::Add,
    ] || [
        subtract,
        power,
        add_epsilon,
        sqrt,
        divide,
        multiply,
        add_bias,
    ]
    .iter()
    .any(|node| !node.attributes.is_empty())
    {
        return None;
    }

    let [input] = mean.inputs.as_slice() else {
        return None;
    };
    let [mean_output] = mean.outputs.as_slice() else {
        return None;
    };
    let [subtract_input, subtract_mean] = subtract.inputs.as_slice() else {
        return None;
    };
    let [centered] = subtract.outputs.as_slice() else {
        return None;
    };
    if subtract_input != input || subtract_mean != mean_output {
        return None;
    }
    let [power_input, exponent] = power.inputs.as_slice() else {
        return None;
    };
    let [power_output] = power.outputs.as_slice() else {
        return None;
    };
    if power_input != centered || scalar_constants.get(exponent).copied() != Some(2.0) {
        return None;
    }
    let [variance_input] = variance.inputs.as_slice() else {
        return None;
    };
    let [variance_output] = variance.outputs.as_slice() else {
        return None;
    };
    if variance_input != power_output {
        return None;
    }
    let [epsilon_left, epsilon_right] = add_epsilon.inputs.as_slice() else {
        return None;
    };
    let epsilon = match (
        epsilon_left == variance_output,
        epsilon_right == variance_output,
    ) {
        (true, false) if scalar_constants.contains_key(epsilon_right) => epsilon_right.as_str(),
        (false, true) if scalar_constants.contains_key(epsilon_left) => epsilon_left.as_str(),
        _ => return None,
    };
    let [shifted_variance] = add_epsilon.outputs.as_slice() else {
        return None;
    };
    let [sqrt_input] = sqrt.inputs.as_slice() else {
        return None;
    };
    let [denominator] = sqrt.outputs.as_slice() else {
        return None;
    };
    if sqrt_input != shifted_variance {
        return None;
    }
    let [divide_centered, divide_denominator] = divide.inputs.as_slice() else {
        return None;
    };
    let [normalized] = divide.outputs.as_slice() else {
        return None;
    };
    if divide_centered != centered || divide_denominator != denominator {
        return None;
    }
    let [multiply_left, multiply_right] = multiply.inputs.as_slice() else {
        return None;
    };
    let scale = match (multiply_left == normalized, multiply_right == normalized) {
        (true, false) => multiply_right.as_str(),
        (false, true) => multiply_left.as_str(),
        _ => return None,
    };
    let [scaled] = multiply.outputs.as_slice() else {
        return None;
    };
    let [bias_left, bias_right] = add_bias.inputs.as_slice() else {
        return None;
    };
    let bias = match (bias_left == scaled, bias_right == scaled) {
        (true, false) => bias_right.as_str(),
        (false, true) => bias_left.as_str(),
        _ => return None,
    };

    for (output, expected_uses) in [
        (mean_output, 1),
        (centered, 2),
        (power_output, 1),
        (variance_output, 1),
        (shifted_variance, 1),
        (denominator, 1),
        (normalized, 1),
        (scaled, 1),
    ] {
        if output == retained_output || use_counts.get(output).copied() != Some(expected_uses) {
            return None;
        }
    }
    Some(MatchedFullInputs {
        input,
        mean_axis,
        variance_axis,
        exponent,
        epsilon,
        scale,
        bias,
    })
}

fn last_axis_mean(node: &GraphNode) -> Option<i64> {
    if node.op != GraphOp::ReduceMean
        || node.inputs.len() != 1
        || node.outputs.len() != 1
        || node
            .attributes
            .keys()
            .any(|name| name != "axes" && name != "keepdims")
        || node.int("keepdims", 1).ok()? != 1
    {
        return None;
    }
    let axes = node.ints("axes", &[]).ok()?;
    (axes.len() == 1).then_some(axes[0])
}

fn is_last_axis(axis: i64, rank: usize) -> bool {
    axis == -1
        || usize::try_from(axis)
            .ok()
            .is_some_and(|axis| axis + 1 == rank)
}

fn full_layer_norm_shapes(input: &[usize], scale: &[usize], bias: &[usize]) -> bool {
    input
        .last()
        .copied()
        .is_some_and(|features| features > 0 && scale == [features] && bias == [features])
}

fn matched_inputs<'a>(
    nodes: &'a [GraphNode],
    scalar_constants: &HashMap<String, f32>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedInputs<'a>> {
    let [add_epsilon, sqrt, divide, multiply, add_bias] = nodes else {
        return None;
    };
    if [add_epsilon.op, sqrt.op, divide.op, multiply.op, add_bias.op]
        != [
            GraphOp::Add,
            GraphOp::Sqrt,
            GraphOp::Div,
            GraphOp::Mul,
            GraphOp::Add,
        ]
        || nodes.iter().any(|node| !node.attributes.is_empty())
    {
        return None;
    }

    let [variance_or_epsilon, epsilon_or_variance] = add_epsilon.inputs.as_slice() else {
        return None;
    };
    let (variance, epsilon) = match (
        scalar_constants.contains_key(variance_or_epsilon),
        scalar_constants.contains_key(epsilon_or_variance),
    ) {
        (false, true) => (variance_or_epsilon.as_str(), epsilon_or_variance.as_str()),
        (true, false) => (epsilon_or_variance.as_str(), variance_or_epsilon.as_str()),
        _ => return None,
    };
    let [epsilon_output] = add_epsilon.outputs.as_slice() else {
        return None;
    };
    let [sqrt_input] = sqrt.inputs.as_slice() else {
        return None;
    };
    let [sqrt_output] = sqrt.outputs.as_slice() else {
        return None;
    };
    if sqrt_input != epsilon_output {
        return None;
    }
    let [centered, divisor] = divide.inputs.as_slice() else {
        return None;
    };
    if divisor != sqrt_output || centered == variance {
        return None;
    }
    let [divide_output] = divide.outputs.as_slice() else {
        return None;
    };
    let [multiply_left, multiply_right] = multiply.inputs.as_slice() else {
        return None;
    };
    let scale = match (
        multiply_left == divide_output,
        multiply_right == divide_output,
    ) {
        (true, false) => multiply_right.as_str(),
        (false, true) => multiply_left.as_str(),
        _ => return None,
    };
    let [multiply_output] = multiply.outputs.as_slice() else {
        return None;
    };
    let [bias_left, bias_right] = add_bias.inputs.as_slice() else {
        return None;
    };
    let bias = match (bias_left == multiply_output, bias_right == multiply_output) {
        (true, false) => bias_right.as_str(),
        (false, true) => bias_left.as_str(),
        _ => return None,
    };

    for output in [epsilon_output, sqrt_output, divide_output, multiply_output] {
        if output == retained_output || use_counts.get(output).copied() != Some(1) {
            return None;
        }
    }
    Some(MatchedInputs {
        centered,
        variance,
        epsilon,
        scale,
        bias,
    })
}

fn last_axis_layer_norm_shapes(
    centered: &[usize],
    variance: &[usize],
    scale: &[usize],
    bias: &[usize],
) -> bool {
    let Some((&features, prefix)) = centered.split_last() else {
        return false;
    };
    let Some((&variance_features, variance_prefix)) = variance.split_last() else {
        return false;
    };
    features > 0
        && prefix == variance_prefix
        && variance_features == 1
        && scale == [features]
        && bias == [features]
}

fn tensor<'a>(
    values: &'a HashMap<String, GraphValue>,
    input: &str,
    node: &GraphNode,
) -> Result<&'a candle_core::Tensor> {
    values
        .get(input)
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' could not resolve input '{input}'",
                node.name
            ))
        })?
        .tensor(&node.name)
}

#[cfg(test)]
mod tests;
