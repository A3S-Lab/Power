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

/// Executes a private `MatMul -> Add(last-axis bias)` window while retaining
/// the authoritative GEMM and post-GEMM F32 rounding boundaries.
///
/// Admission depends only on topology, liveness, dtype, layout, tensor
/// geometry, co-location, cancellation, and the declared tensor limit.
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
    let left = tensor(values, matched.left, matched.matmul)?;
    let right = tensor(values, matched.right, matched.matmul)?;
    let bias = tensor(values, matched.bias, matched.add)?;
    let Some(output_elements) = output_elements(left, right, bias) else {
        return Ok(None);
    };
    if left.dtype() != DType::F32
        || right.dtype() != DType::F32
        || bias.dtype() != DType::F32
        || !left.device().is_cuda()
        || !right.device().same_device(left.device())
        || !bias.device().same_device(left.device())
        || !left.is_contiguous()
        || !right.is_contiguous()
        || !bias.is_contiguous()
        || u32::try_from(output_elements).is_err()
    {
        return Ok(None);
    }
    if output_elements > element_limit {
        return Err(PowerError::InferenceFailed(format!(
            "static graph node '{}' produced {output_elements} tensor elements, exceeding the {element_limit}-element limit",
            matched.add.name,
        )));
    }
    if cancellation.is_cancelled() {
        return Err(PowerError::InferenceFailed(
            "static graph execution was cancelled".to_string(),
        ));
    }

    #[cfg(feature = "embedded-cuda")]
    {
        let output = cuda::execute(left, right, bias, matched.post_operation).map_err(|error| {
            PowerError::InferenceFailed(format!(
                "static graph nodes '{}' through '{}' fused MatMul bias failed: {error}",
                matched.matmul.name, matched.add.name,
            ))
        })?;
        Ok(Some(FusedOutput {
            value: GraphValue::Tensor(output),
            consumed_nodes: matched.consumed_nodes,
        }))
    }

    #[cfg(not(feature = "embedded-cuda"))]
    Ok(None)
}

#[derive(Clone, Copy, Debug)]
struct MatchedWindow<'a> {
    matmul: &'a GraphNode,
    add: &'a GraphNode,
    left: &'a str,
    right: &'a str,
    bias: &'a str,
    #[cfg(any(test, feature = "embedded-cuda"))]
    post_operation: PostOperation,
    #[cfg(any(test, feature = "embedded-cuda"))]
    consumed_nodes: usize,
}

#[cfg(any(test, feature = "embedded-cuda"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PostOperation {
    Bias,
    Swish,
}

fn matched_window<'a>(
    nodes: &'a [GraphNode],
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> Option<MatchedWindow<'a>> {
    let [matmul, add, ..] = nodes else {
        return None;
    };
    if [matmul.op, add.op] != [GraphOp::MatMul, GraphOp::Add]
        || [matmul, add]
            .iter()
            .any(|node| !node.attributes.is_empty() || node.outputs.len() != 1)
    {
        return None;
    }
    let [left, right] = matmul.inputs.as_slice() else {
        return None;
    };
    let [matmul_output] = matmul.outputs.as_slice() else {
        return None;
    };
    let [add_left, add_right] = add.inputs.as_slice() else {
        return None;
    };
    if matmul_output == retained_output || use_counts.get(matmul_output).copied() != Some(1) {
        return None;
    }
    let bias = if add_left == matmul_output && add_right != matmul_output {
        add_right.as_str()
    } else if add_right == matmul_output && add_left != matmul_output {
        add_left.as_str()
    } else {
        return None;
    };
    #[cfg(any(test, feature = "embedded-cuda"))]
    let (post_operation, consumed_nodes) =
        if matches_private_swish_tail(nodes, &add.outputs[0], use_counts, retained_output) {
            (PostOperation::Swish, 4)
        } else {
            (PostOperation::Bias, 2)
        };
    Some(MatchedWindow {
        matmul,
        add,
        left,
        right,
        bias,
        #[cfg(any(test, feature = "embedded-cuda"))]
        post_operation,
        #[cfg(any(test, feature = "embedded-cuda"))]
        consumed_nodes,
    })
}

#[cfg(any(test, feature = "embedded-cuda"))]
fn matches_private_swish_tail(
    nodes: &[GraphNode],
    add_output: &str,
    use_counts: &HashMap<String, usize>,
    retained_output: &str,
) -> bool {
    let Some([sigmoid, multiply]) = nodes.get(2..4) else {
        return false;
    };
    if [sigmoid.op, multiply.op] != [GraphOp::Sigmoid, GraphOp::Mul]
        || [sigmoid, multiply]
            .iter()
            .any(|node| !node.attributes.is_empty() || node.outputs.len() != 1)
    {
        return false;
    }
    let [sigmoid_input] = sigmoid.inputs.as_slice() else {
        return false;
    };
    let [sigmoid_output] = sigmoid.outputs.as_slice() else {
        return false;
    };
    let [multiply_left, multiply_right] = multiply.inputs.as_slice() else {
        return false;
    };
    sigmoid_input == add_output
        && add_output != retained_output
        && sigmoid_output != retained_output
        && use_counts.get(add_output).copied() == Some(2)
        && use_counts.get(sigmoid_output).copied() == Some(1)
        && ((multiply_left == add_output && multiply_right == sigmoid_output)
            || (multiply_right == add_output && multiply_left == sigmoid_output))
}

fn output_elements(left: &Tensor, right: &Tensor, bias: &Tensor) -> Option<usize> {
    if left.rank() < 2 || right.rank() != 2 || bias.rank() != 1 {
        return None;
    }
    let left_dimensions = left.dims();
    let right_dimensions = right.dims();
    let rows = left_dimensions[left_dimensions.len() - 2];
    let inner = left_dimensions[left_dimensions.len() - 1];
    let columns = right_dimensions[1];
    if rows == 0
        || inner == 0
        || columns == 0
        || right_dimensions[0] != inner
        || bias.dims() != [columns]
    {
        return None;
    }
    left_dimensions[..left_dimensions.len() - 2]
        .iter()
        .try_fold(rows.checked_mul(columns)?, |elements, dimension| {
            elements.checked_mul(*dimension)
        })
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
                node.name,
            ))
        })?
        .tensor(&node.name)
}

#[cfg(test)]
mod tests {
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
            node("product", GraphOp::MatMul, &["left", "right"], "product"),
            node("bias", GraphOp::Add, &["product", "bias"], "output"),
        ]
    }

    #[test]
    fn matcher_requires_a_private_matmul_output_and_commutative_bias_add() {
        let nodes = window();
        let uses = HashMap::from([("product".to_string(), 1)]);
        let matched = matched_window(&nodes, &uses, "output").unwrap();
        assert_eq!(matched.left, "left");
        assert_eq!(matched.right, "right");
        assert_eq!(matched.bias, "bias");
        assert_eq!(matched.post_operation, PostOperation::Bias);
        assert_eq!(matched.consumed_nodes, 2);

        let reversed = vec![
            nodes[0].clone(),
            node("bias", GraphOp::Add, &["bias", "product"], "output"),
        ];
        assert_eq!(
            matched_window(&reversed, &uses, "output").unwrap().bias,
            "bias"
        );
    }

    #[test]
    fn matcher_composes_the_exact_private_swish_tail() {
        let mut nodes = window();
        nodes.extend([
            node("gate", GraphOp::Sigmoid, &["output"], "gate"),
            node("activation", GraphOp::Mul, &["gate", "output"], "activated"),
        ]);
        let uses = HashMap::from([
            ("product".to_string(), 1),
            ("output".to_string(), 2),
            ("gate".to_string(), 1),
        ]);

        let matched = matched_window(&nodes, &uses, "activated").unwrap();

        assert_eq!(matched.post_operation, PostOperation::Swish);
        assert_eq!(matched.consumed_nodes, 4);
    }

    #[test]
    fn matcher_rejects_shared_or_retained_matmul_outputs() {
        let nodes = window();
        for (uses, retained) in [
            (HashMap::from([("product".to_string(), 2)]), "output"),
            (HashMap::from([("product".to_string(), 1)]), "product"),
        ] {
            assert!(matched_window(&nodes, &uses, retained).is_none());
        }
    }

    #[test]
    fn geometry_accepts_any_nonempty_left_prefix_and_exact_last_axis_bias() {
        let device = candle_core::Device::Cpu;
        let right = Tensor::zeros((5, 7), DType::F32, &device).unwrap();
        let bias = Tensor::zeros(7, DType::F32, &device).unwrap();
        for shape in [vec![3, 5], vec![2, 3, 5], vec![2, 4, 3, 5]] {
            let left = Tensor::zeros(shape.as_slice(), DType::F32, &device).unwrap();
            let rows = shape[shape.len() - 2];
            let batch = shape[..shape.len() - 2].iter().product::<usize>();
            assert_eq!(
                output_elements(&left, &right, &bias),
                Some(batch * rows * 7)
            );
        }

        let wrong_bias = Tensor::zeros(6, DType::F32, &device).unwrap();
        assert!(output_elements(
            &Tensor::zeros((2, 3, 5), DType::F32, &device).unwrap(),
            &right,
            &wrong_bias,
        )
        .is_none());
    }
}
