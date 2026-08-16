use std::collections::HashMap;

use candle_core::DType;
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

#[cfg(feature = "embedded-cuda")]
mod cuda;

/// Executes the pointwise tail of a decomposed last-axis LayerNorm in one
/// CUDA kernel while retaining the graph's existing mean and variance
/// reductions. The CUDA implementation preserves every original f32
/// rounding boundary.
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
            || !tensor.device().is_cuda()
            || !tensor.device().same_device(centered.device())
            || !tensor.is_contiguous()
    }) {
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

    #[cfg(feature = "embedded-cuda")]
    {
        if cancellation.is_cancelled() {
            return Err(PowerError::InferenceFailed(
                "static graph execution was cancelled".to_string(),
            ));
        }
        let output = cuda::execute(centered, variance, scale, bias, epsilon).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused LayerNorm affine tail failed: {error}",
                nodes[0].name, nodes[4].name
            ))
        })?;
        Ok(Some(GraphValue::Tensor(output)))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    {
        let _ = cancellation;
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MatchedInputs<'a> {
    centered: &'a str,
    variance: &'a str,
    epsilon: &'a str,
    scale: &'a str,
    bias: &'a str,
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
