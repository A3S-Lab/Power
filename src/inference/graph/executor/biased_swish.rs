use std::collections::HashMap;

use candle_core::{DType, Tensor};
use tokio_util::sync::CancellationToken;

use crate::error::{PowerError, Result};

use super::super::plan::{GraphNode, GraphOp};
use super::super::value::GraphValue;

#[cfg(feature = "embedded-cuda")]
mod cuda;

pub(super) struct FusedOutput {
    pub(super) value: GraphValue,
    pub(super) consumed_nodes: usize,
}

/// Executes an exact private `Add(last-axis bias) -> Sigmoid -> Mul` window in
/// one CUDA pass. Eligibility is defined only by graph ownership, dtype,
/// device, layout, tensor geometry, and declared resource limits.
pub(super) fn try_execute(
    nodes: &[GraphNode],
    values: &HashMap<String, GraphValue>,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
    element_limit: usize,
    cancellation: &CancellationToken,
) -> Result<Option<FusedOutput>> {
    let Some(matched) = matched_window(nodes, use_counts, retained_output) else {
        return Ok(None);
    };
    let left = tensor(values, matched.left, matched.add)?;
    let right = tensor(values, matched.right, matched.add)?;
    let Some((input, bias)) = last_axis_bias_operands(left, right) else {
        return Ok(None);
    };
    if input.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !input.device().is_cuda()
        || !bias.device().same_device(input.device())
        || !input.is_contiguous()
        || !bias.is_contiguous()
        || input.elem_count() == 0
        || u32::try_from(input.elem_count()).is_err()
    {
        return Ok(None);
    }
    if input.elem_count() > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {} tensor elements, exceeding the {element_limit}-element limit",
            matched.multiply.name,
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
        let output = cuda::execute(input, bias).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused biased Swish failed: {error}",
                matched.add.name, matched.multiply.name
            ))
        })?;
        Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output),
            consumed_nodes: 3,
        }))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

#[derive(Clone, Copy, Debug)]
struct MatchedWindow<'a> {
    add: &'a GraphNode,
    multiply: &'a GraphNode,
    left: &'a str,
    right: &'a str,
}

fn matched_window<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedWindow<'a>> {
    let [add, sigmoid, multiply, ..] = nodes else {
        return None;
    };
    if [add.op, sigmoid.op, multiply.op] != [GraphOp::Add, GraphOp::Sigmoid, GraphOp::Mul]
        || [add, sigmoid, multiply]
            .iter()
            .any(|node| !node.attributes.is_empty() || node.outputs.len() != 1)
    {
        return None;
    }
    let [left, right] = add.inputs.as_slice() else {
        return None;
    };
    let [add_output] = add.outputs.as_slice() else {
        return None;
    };
    let [sigmoid_input] = sigmoid.inputs.as_slice() else {
        return None;
    };
    let [sigmoid_output] = sigmoid.outputs.as_slice() else {
        return None;
    };
    let [multiply_left, multiply_right] = multiply.inputs.as_slice() else {
        return None;
    };
    if sigmoid_input != add_output
        || add_output == retained_output
        || sigmoid_output == retained_output
        || use_counts.get(add_output).copied() != Some(2)
        || use_counts.get(sigmoid_output).copied() != Some(1)
        || !((multiply_left == add_output && multiply_right == sigmoid_output)
            || (multiply_right == add_output && multiply_left == sigmoid_output))
    {
        return None;
    }
    Some(MatchedWindow {
        add,
        multiply,
        left,
        right,
    })
}

fn last_axis_bias_operands<'a>(
    left: &'a Tensor,
    right: &'a Tensor,
) -> Option<(&'a Tensor, &'a Tensor)> {
    match (left.rank(), right.rank()) {
        (1, 1) if left.dims() == right.dims() => Some((left, right)),
        (1, right_rank) if right_rank > 1 && right.dims().last() == left.dims().first() => {
            Some((right, left))
        }
        (left_rank, 1) if left_rank > 1 && left.dims().last() == right.dims().first() => {
            Some((left, right))
        }
        _ => None,
    }
}

fn tensor<'a>(
    values: &'a HashMap<String, GraphValue>,
    name: &str,
    node: &GraphNode,
) -> Result<&'a Tensor> {
    values
        .get(name)
        .ok_or_else(|| {
            PowerError::InferenceFailed(format!(
                "static graph node '{}' could not resolve input '{name}'",
                node.name
            ))
        })?
        .tensor(&node.name)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn node(name: &str, op: GraphOp, inputs: &[&str], output: &str) -> GraphNode {
        GraphNode {
            name: name.to_string(),
            op,
            inputs: inputs.iter().map(|input| (*input).to_string()).collect(),
            outputs: vec![output.to_string()],
            attributes: Default::default(),
        }
    }

    fn window() -> Vec<GraphNode> {
        vec![
            node("bias", GraphOp::Add, &["input", "bias"], "biased"),
            node("gate", GraphOp::Sigmoid, &["biased"], "sigmoid"),
            node("product", GraphOp::Mul, &["sigmoid", "biased"], "output"),
        ]
    }

    #[test]
    fn matcher_requires_the_exact_private_swish_formula() {
        let nodes = window();
        let uses = HashMap::from([("biased".to_string(), 2), ("sigmoid".to_string(), 1)]);

        let matched = matched_window(&nodes, &uses, "output").unwrap();

        assert_eq!(matched.left, "input");
        assert_eq!(matched.right, "bias");
    }

    #[test]
    fn matcher_rejects_shared_or_retained_intermediates() {
        let nodes = window();
        for (uses, retained) in [
            (
                HashMap::from([("biased".to_string(), 3), ("sigmoid".to_string(), 1)]),
                "output",
            ),
            (
                HashMap::from([("biased".to_string(), 2), ("sigmoid".to_string(), 2)]),
                "output",
            ),
            (
                HashMap::from([("biased".to_string(), 2), ("sigmoid".to_string(), 1)]),
                "biased",
            ),
        ] {
            assert!(matched_window(&nodes, &uses, retained).is_none());
        }
    }

    #[test]
    fn operand_selection_uses_only_rank_and_exact_last_axis_geometry() {
        let device = candle_core::Device::Cpu;
        let input = Tensor::zeros((2, 3, 5), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(5, DType::F32, &device).unwrap();
        let wrong = Tensor::zeros(4, DType::F32, &device).unwrap();

        let (selected_input, selected_bias) = last_axis_bias_operands(&bias, &input).unwrap();
        assert_eq!(selected_input.dims(), [2, 3, 5]);
        assert_eq!(selected_bias.dims(), [5]);
        assert!(last_axis_bias_operands(&input, &wrong).is_none());
    }
}
